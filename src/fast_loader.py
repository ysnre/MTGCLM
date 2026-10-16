"""
fast_loader.py — hızlı veri yükleyici (K16, 16 Eki 2026)
=========================================================
Eski yol (MTGH5Dataset.__getitem__ + DataLoader, num_workers=0) her örneği H5'ten TEK TEK okur:
~0,88 ms/örnek -> train epoch'u başına ~3,2 dk yalnız okuma; Colab T4 GPU'su bu yüzden ~%1 doluydu.
Bu modül aynı batch'leri, AYNI SIRADA, ardışık blok okumalarıyla üretir (~20x+ hızlı).

Modlar (train.py / train_unet.py --loader):
  memory : bölmedeki satırlar tek geçişte RAM'e (float16) alınır; epoch'lar diskten bağımsız.
           Gerekli RAM ≈ satır sayısı x kanal x 33 x 33 x 2 bayt (22 kanal: ~47,9 KB/satır;
           split_v2_verify'ın tamamı ~16 GB, yalnız train+val ~10,6 GB).
  stream : her epoch dosya ardışık bloklarla okunur (RAM ~birkaç yüz MB). Bloklar arka plan
           iş parçacığında önceden okunur.
  auto   : (varsayılan) boş RAM'e göre seçer: hepsi RAM'e -> yalnız train+val RAM'e -> stream.
  legacy : eski DataLoader yolu (karşılaştırma/doğrulama için).

Önbellek (--cache_dir, isteğe bağlı; Colab için önerilir): bölme satırları bölme sırasıyla ve 22 kanalın
tamamıyla ayrı .npy dosyalarına (float16, bellek eşlemli) BİR KEZ yazılır. Sonraki koşular yalnız gereken
satırları ardışık okur (epoch başına ~10,7 GB yerine tüm H5'in iki kez taranması ~38 GB). Önbellek bölme
dosyası adına bağlıdır (<cache_dir>/<bölme>__<küme>.npy) ve satırlar + H5 boyutu eşleşmezse yeniden yazılır.

Eşdeğerlik: bölme indeksleri artan sıralı; eski yol da shuffle=False ile bu sırayla okur (kompakt H5
önceden karıştırılmış). Batch'ler bu yüzden birebir aynı örneklerden oluşur. Tek fark: yatay/dikey
çevirme (augmentation) artık batch bazında, koşuya ve epoch'a bağlı bir torch.Generator ile seçilir
(eski: örnek başına Python random). Yükleyici float16 + ham aux döndürür; float32'ye çevirme ve aux
z-skoru train.py'deki prep_features() ile cihazda yapılır (eski f16 yolu ile aynı matematik).

Uyumluluk: dönen nesne DataLoader gibi davranır (iter, len, .batch_size) ve .dataset zinciri eski
kodun beklediği gibidir: train -> AugView -> RowView -> MTGH5Dataset; diğerleri -> RowView -> MTGH5Dataset.
RowView.indices = bölme satırları (sınıf ağırlıkları, test_partial okta'sı, tahmin kaydı için).
"""
import math
import os
import queue
import sys
import threading
import time

import h5py
import numpy as np
import torch

PATCH = 33


# ----------------------------------------------------------------------------- uyumluluk görünümleri
class RowView:
    """torch.utils.data.Subset yerine: .dataset (MTGH5Dataset) ve .indices (np.int64, artan)."""
    def __init__(self, base, indices):
        self.dataset = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)


class AugView:
    """Eski AugmentedDataset zincirini taklit eder (train -> .dataset -> RowView)."""
    def __init__(self, view):
        self.dataset = view

    def __len__(self):
        return len(self.dataset)


# ----------------------------------------------------------------------------- RAM yardımcıları
def available_ram_bytes():
    """Boş (kullanılabilir) RAM; psutil yoksa Windows/Linux yedek yolları. Bilinmiyorsa None."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMSTAT(); st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullAvailPhys)
        except Exception:
            return None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def row_bytes(base):
    return (int(base.base_channels) + len(base.aux_names)) * PATCH * PATCH * 2


# ----------------------------------------------------------------------------- satır deposu
class H5RowStore:
    """
    Kompakt H5'ten satır okur. mem_rows: {ad: satırlar} — bunlar kurulumda tek geçişte RAM'e alınır.
    Diğer kümeler batches() çağrısında dosyadan ardışık bloklarla akıtılır.
    """
    def __init__(self, h5_path, base, mem_rows=None, block_rows=4096, verbose=True,
                 cache_dir=None, cache_tag=None, cache_rows=None):
        self.h5_path = h5_path
        self.base = base
        self.block_rows = int(block_rows)
        self.n_base = int(base.base_channels)
        self.aux_idx = None if base.aux_idx is None else np.asarray(base.aux_idx, dtype=np.int64)
        self.C = self.n_base + (0 if self.aux_idx is None else len(self.aux_idx))
        self.N = int(base.length)
        self.labels = np.asarray(base.labels, dtype=np.int64)
        self.mem = {}
        self.cache = {}
        self.verbose = verbose
        if cache_dir and cache_rows:
            self._prepare_cache(cache_dir, cache_tag or "split", cache_rows)
        if mem_rows:
            self._load(mem_rows)

    # --- önbellek: tüm kanallar (17 + 5 aux), bölme sırasıyla
    def _ch_sel(self, total):
        sel = list(range(self.n_base)) + ([] if self.aux_idx is None else [self.n_base + int(i) for i in self.aux_idx])
        return None if len(sel) == total else np.asarray(sel)

    def _prepare_cache(self, cache_dir, tag, cache_rows):
        os.makedirs(cache_dir, exist_ok=True)
        h5_size = os.path.getsize(self.h5_path)
        todo = {}
        for k, r in cache_rows.items():
            r = np.asarray(r, dtype=np.int64)
            p = os.path.join(cache_dir, f"{tag}__{k}.npy")
            ok = False
            if os.path.exists(p) and os.path.exists(p + ".rows.npy") and os.path.exists(p + ".h5size"):
                try:
                    ok = (np.array_equal(np.load(p + ".rows.npy"), r)
                          and open(p + ".h5size").read().strip() == str(h5_size))
                except Exception:
                    ok = False
            if ok:
                self.cache[k] = (r, np.load(p, mmap_mode="r"))
            else:
                todo[k] = (r, p)
        if todo:
            self._build_cache(todo, h5_size)
            for k, (r, p) in todo.items():
                self.cache[k] = (r, np.load(p, mmap_mode="r"))
        if self.verbose:
            print(f"  [loader] önbellek: {cache_dir} ({', '.join(self.cache)})")

    def _build_cache(self, todo, h5_size):
        from numpy.lib.format import open_memmap
        with self._open() as f:
            n_aux = f["aux"].shape[1] if "aux" in f else 0
            CT = self.n_base + n_aux
            outs = {}
            for k, (r, p) in todo.items():
                outs[k] = (r, p, open_memmap(p + ".tmp.npy", mode="w+", dtype=np.float16, shape=(len(r), CT, PATCH, PATCH)))
            need = sum(len(r) for r, _, _ in outs.values()) * CT * PATCH * PATCH * 2
            if self.verbose:
                print(f"  [loader] önbellek yazılıyor ({need / 1e9:.1f} GB): {', '.join(outs)}")
            union = np.unique(np.concatenate([r for r, _, _ in outs.values()]))
            t0 = time.time(); last = t0
            blocks = self._blocks_for(union)
            for bi, (lo, hi) in enumerate(blocks):
                blk = np.empty((hi - lo, CT, PATCH, PATCH), dtype=np.float16)
                blk[:, :self.n_base] = f["patches"][lo:hi]
                if n_aux:
                    blk[:, self.n_base:] = f["aux"][lo:hi]
                for k, (r, p, X) in outs.items():
                    j0, j1 = np.searchsorted(r, lo), np.searchsorted(r, hi)
                    if j1 > j0:
                        X[j0:j1] = blk[r[j0:j1] - lo]
                if self.verbose and (time.time() - last > 30 or bi == len(blocks) - 1):
                    last = time.time()
                    print(f"    {bi + 1}/{len(blocks)} blok  {time.time() - t0:5.0f} s", flush=True)
        for k, (r, p, X) in outs.items():
            X.flush(); del X
            os.replace(p + ".tmp.npy", p)
            np.save(p + ".rows.npy", r)
            with open(p + ".h5size", "w") as fh:
                fh.write(str(h5_size))

    # --- düşük seviye: [lo, hi) aralığını (n, C, 33, 33) float16 olarak oku
    def _read_range(self, f, lo, hi):
        out = np.empty((hi - lo, self.C, PATCH, PATCH), dtype=np.float16)
        out[:, :self.n_base] = f["patches"][lo:hi]
        if self.aux_idx is not None:
            a = f["aux"][lo:hi]
            out[:, self.n_base:] = a[:, self.aux_idx]
        return out

    def _blocks_for(self, rows):
        """rows (artan) içeren dosya blokları: [(lo, hi), ...]"""
        if len(rows) == 0:
            return []
        b = np.unique(rows // self.block_rows)
        return [(int(k) * self.block_rows, min((int(k) + 1) * self.block_rows, self.N)) for k in b]

    def _open(self):
        return h5py.File(self.h5_path, "r", rdcc_nbytes=64 * 1024 * 1024)

    # --- RAM'e yükleme (tek geçiş, tüm kümeler birlikte)
    def _load(self, mem_rows):
        sets = {k: np.asarray(v, dtype=np.int64) for k, v in mem_rows.items() if len(v)}
        for k, r in sets.items():
            if np.any(np.diff(r) <= 0):
                raise ValueError(f"{k}: satırlar artan ve tekil olmalı")
        need = sum(len(r) for r in sets.values()) * self.C * PATCH * PATCH * 2
        if self.verbose:
            print(f"  [loader] RAM'e yükleniyor: {', '.join(f'{k}={len(r)}' for k, r in sets.items())} "
                  f"-> {need / 1e9:.1f} GB (float16, {self.C} kanal)")
        for k, r in sets.items():
            self.mem[k] = (r, np.empty((len(r), self.C, PATCH, PATCH), dtype=np.float16), self.labels[r])
        t0 = time.time()
        for k in [k for k in sets if k in self.cache and np.array_equal(self.cache[k][0], sets[k])]:
            arr = self.cache[k][1]; X = self.mem[k][1]; sel = self._ch_sel(arr.shape[1])
            for s0 in range(0, len(X), 8192):
                X[s0:s0 + 8192] = arr[s0:s0 + 8192] if sel is None else arr[s0:s0 + 8192][:, sel]
            sets.pop(k)
        if not sets:
            if self.verbose:
                print(f"  [loader] önbellekten yüklendi: {time.time() - t0:.0f} s")
            return
        union = np.unique(np.concatenate(list(sets.values())))
        blocks = self._blocks_for(union)
        t0 = time.time(); last = t0
        with self._open() as f:
            for bi, (lo, hi) in enumerate(blocks):
                blk = self._read_range(f, lo, hi)
                for k, (r, X, _) in self.mem.items():
                    if k not in sets:          # önbellekten zaten yüklendi
                        continue
                    j0, j1 = np.searchsorted(r, lo), np.searchsorted(r, hi)
                    if j1 > j0:
                        X[j0:j1] = blk[r[j0:j1] - lo]
                if self.verbose and (time.time() - last > 20 or bi == len(blocks) - 1):
                    last = time.time()
                    done = hi * self.C * PATCH * PATCH * 2
                    print(f"    {bi + 1}/{len(blocks)} blok  {time.time() - t0:5.0f} s  "
                          f"(~{done / max(time.time() - t0, 1e-6) / 1e6:.0f} MB/s)", flush=True)
        if self.verbose:
            print(f"  [loader] yükleme bitti: {time.time() - t0:.0f} s")

    # --- batch üretimi: rows sırasıyla, ardışık batch_size'lık parçalar (eski DataLoader ile aynı)
    def batches(self, name, rows, batch_size):
        rows = np.asarray(rows, dtype=np.int64)
        if name in self.mem and np.array_equal(self.mem[name][0], rows):
            _, X, Y = self.mem[name]
            for s in range(0, len(rows), batch_size):
                yield X[s:s + batch_size], Y[s:s + batch_size]
            return
        if name in self.cache and np.array_equal(self.cache[name][0], rows):
            yield from _prefetch(self._stream_cache(name, batch_size), depth=3)
            return
        yield from _prefetch(self._stream(rows, batch_size), depth=3)

    def _stream_cache(self, name, batch_size, chunk=8192):
        r, arr = self.cache[name]
        Y = self.labels[r]
        sel = self._ch_sel(arr.shape[1])
        chunk = max(batch_size, (chunk // batch_size) * batch_size)
        for s0 in range(0, len(r), chunk):
            blk = np.ascontiguousarray(arr[s0:s0 + chunk] if sel is None else arr[s0:s0 + chunk][:, sel])
            for s in range(0, len(blk), batch_size):
                yield blk[s:s + batch_size], Y[s0 + s:s0 + s + batch_size]

    def _stream(self, rows, batch_size):
        pend_x, pend_y, n_pend = [], [], 0
        with self._open() as f:
            for lo, hi in self._blocks_for(rows):
                j0, j1 = np.searchsorted(rows, lo), np.searchsorted(rows, hi)
                if j1 <= j0:
                    continue
                r = rows[j0:j1]
                blk = self._read_range(f, lo, hi)[r - lo]
                pend_x.append(blk); pend_y.append(self.labels[r]); n_pend += len(r)
                while n_pend >= batch_size:
                    x = np.concatenate(pend_x) if len(pend_x) > 1 else pend_x[0]
                    y = np.concatenate(pend_y) if len(pend_y) > 1 else pend_y[0]
                    k = (n_pend // batch_size) * batch_size
                    for s in range(0, k, batch_size):
                        yield x[s:s + batch_size], y[s:s + batch_size]
                    pend_x, pend_y = ([x[k:]], [y[k:]]) if k < n_pend else ([], [])
                    n_pend -= k
        if n_pend:
            yield np.concatenate(pend_x), np.concatenate(pend_y)


def _prefetch(gen, depth=3):
    """Üreteci arka plan iş parçacığında çalıştırır (disk okuması hesapla örtüşsün)."""
    q = queue.Queue(maxsize=depth)
    END = object()
    err = []

    def work():
        try:
            for item in gen:
                q.put(item)
        except BaseException as e:      # hata ana iş parçacığına taşınır
            err.append(e)
        finally:
            q.put(END)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is END:
            break
        yield item
    t.join()
    if err:
        raise err[0]


# ----------------------------------------------------------------------------- DataLoader yerine
class FastLoader:
    """DataLoader arayüzü: iter -> (features f16 [B,C,33,33], scalars [B,3], labels [B])."""
    def __init__(self, store, name, rows, batch_size, base, augment=False, seed=0):
        self.store, self.name = store, name
        self.rows = np.asarray(rows, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.augment = augment
        self.seed = int(seed)
        self.epoch = 0
        view = RowView(base, self.rows)
        self.dataset = AugView(view) if augment else view

    def __len__(self):
        return math.ceil(len(self.rows) / self.batch_size)

    def set_epoch(self, epoch: int):
        """Çevirme rastgeleliği (seed, epoch) ile belirlenir -> checkpoint'ten devamda da aynı."""
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self.epoch)
        for xb, yb in self.store.batches(self.name, self.rows, self.batch_size):
            x = torch.from_numpy(xb)            # bellek modunda kopyasız görünüm — YERİNDE DEĞİŞTİRME
            y = torch.from_numpy(np.ascontiguousarray(yb))
            if self.augment:
                x = random_flips(x, g)
            yield x, torch.zeros(len(y), 3), y


def random_flips(x, g):
    """Örnek başına %50 yatay + %50 dikey çevirme; merkez piksel (16,16) yerinde kalır. Yeni tensör döner."""
    b = x.shape[0]
    h = (torch.rand(b, generator=g) < 0.5).view(b, 1, 1, 1)
    v = (torch.rand(b, generator=g) < 0.5).view(b, 1, 1, 1)
    x = torch.where(h, x.flip(-1), x)
    return torch.where(v, x.flip(-2), x)


def choose_mem_splits(mode, base, split_rows, mem_splits=None, safety=0.75, verbose=True):
    """Hangi kümeler RAM'e alınacak? mode: memory | stream | auto. Dönen: {ad: satırlar}."""
    if mode == "stream":
        return {}
    rb = row_bytes(base)
    names = list(split_rows)
    if mem_splits:
        names = [n for n in names if n in mem_splits]
    if mode == "memory":
        return {n: split_rows[n] for n in names}
    avail = available_ram_bytes()
    if avail is None:
        if verbose:
            print("  [loader] auto: boş RAM ölçülemedi -> stream")
        return {}
    budget = avail * safety
    total = sum(len(split_rows[n]) for n in names) * rb
    tv = [n for n in names if n in ("train", "val")]
    tv_bytes = sum(len(split_rows[n]) for n in tv) * rb
    if total <= budget:
        pick = names
    elif tv and tv_bytes <= budget:
        pick = tv
    else:
        pick = []
    if verbose:
        print(f"  [loader] auto: boş RAM {avail / 1e9:.1f} GB, tümü {total / 1e9:.1f} GB, train+val {tv_bytes / 1e9:.1f} GB "
              f"-> RAM'de: {', '.join(pick) if pick else 'hiçbiri (stream)'}")
    return {n: split_rows[n] for n in pick}
