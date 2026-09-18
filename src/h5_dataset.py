import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MTGH5Dataset(Dataset):
    """
    High-performance PyTorch Dataset that reads pre-extracted patches from a compressed HDF5 file.
    Opens the file lazily in __getitem__ to be compatible with PyTorch DataLoader multiprocessing (num_workers > 0).

    use_aux=True: 'aux' dataset'i (update_h5_aux_channels.py) varsa, normalize edilip 17 uydu/radar
    kanalının arkasına eklenir -> patch [17 + n_aux, 33, 33]. num_channels buna göre güncellenir.
    aux_channels: kullanılacak aux kanal adlarının alt kümesi (ör. ["t2m","rh2m","cos_sza"]); None = hepsi.
    Dönen tuple eski kodla uyumlu kalır: (patch, scalars[3], label). scalars, 'scalars' dataset'i yoksa sıfırdır.
    """
    def __init__(self, h5_path: str, use_aux: bool = False, aux_channels=None, f16: bool = False):
        self.h5_path = h5_path
        self.use_aux = use_aux
        # f16=True: yamalar float16 olarak DÖNER ve aux normalize EDİLMEZ. float32'ye çevirme ve
        # aux z-skoru GPU'da yapılır (train.py: prep_features). CPU'da yapılan float32 dönüşümü +
        # concat + stack, veri yolunun ~%70'ini yiyordu; bu mod onu ortadan kaldırır.
        # Sonuç matematiksel olarak birebir aynıdır (float16 -> float32 genişletme kayıpsızdır,
        # z-skor yine float32 aritmetiğiyle yapılır); yalnızca nerede yapıldığı değişir.
        self.f16 = f16

        # Open file once in main thread to load metadata, length, and stats
        # Set chunk cache size to 512 MB to accommodate large 144.6 MB chunks
        with h5py.File(self.h5_path, 'r', rdcc_nbytes=512*1024*1024) as f:
            self.length = len(f['labels'])
            self.means = f.attrs['means'][:]
            self.stds = f.attrs['stds'][:]
            # Preload small metadata arrays into memory for fast splitting/indexing and class weights calculation
            self.timestamps = [ts.decode('utf-8') for ts in f['timestamps'][:]]
            self.wmoids = f['wmoids'][:]
            self.labels = f['labels'][:]
            self.okta = f['okta'][:] if 'okta' in f else None
            self.pre_shuffled = bool(f.attrs.get('pre_shuffled', 0))
            self.base_channels = int(f['patches'].shape[1])
            self.has_scalars = 'scalars' in f

            self.aux_idx = None
            if use_aux:
                if 'aux' not in f:
                    raise RuntimeError("use_aux=True ama H5'te 'aux' yok. Önce src/update_h5_aux_channels.py çalıştırın.")
                names = json.loads(f.attrs['aux_channels'])
                sel = names if aux_channels is None else [c for c in aux_channels if c in names]
                self.aux_names = sel
                self.aux_idx = np.array([names.index(c) for c in sel], dtype=np.int64)
                self.aux_means = np.asarray(f.attrs['aux_means'], dtype=np.float32)[self.aux_idx][:, None, None]
                self.aux_stds = np.asarray(f.attrs['aux_stds'], dtype=np.float32)[self.aux_idx][:, None, None]
            else:
                self.aux_names = []

        self.num_channels = self.base_channels + len(self.aux_names)
        self.h5_file = None  # Opened lazily per worker process
        self.patches_ds = None
        self.labels_ds = None
        self.scalars_ds = None
        self.aux_ds = None

    def __len__(self) -> int:
        return self.length

    def _open(self):
        # Open the file in read-only mode with 512 MB chunk cache
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True, rdcc_nbytes=512*1024*1024)
        # Store dataset references to prevent open/close overhead
        self.patches_ds = self.h5_file['patches']
        self.labels_ds = self.h5_file['labels']
        if 'scalars' in self.h5_file:
            self.scalars_ds = self.h5_file['scalars']
        if self.use_aux:
            self.aux_ds = self.h5_file['aux']

    def __getitem__(self, idx: int):
        if self.h5_file is None:
            self._open()

        # Load patch and label directly from cached datasets
        label = self.labels_ds[idx]
        if self.f16:
            patch = self.patches_ds[idx]
            if self.aux_ds is not None:
                patch = np.concatenate([patch, self.aux_ds[idx][self.aux_idx]], axis=0)
        else:
            patch = self.patches_ds[idx].astype(np.float32, copy=False)
            if self.aux_ds is not None:
                aux = self.aux_ds[idx].astype(np.float32)[self.aux_idx]
                aux = (aux - self.aux_means) / self.aux_stds
                patch = np.concatenate([patch, aux], axis=0)

        if self.scalars_ds is not None:
            scalar = self.scalars_ds[idx]
        else:
            scalar = np.zeros(3, dtype=np.float32)  # Fallback if scalars are not baked in

        return torch.from_numpy(np.ascontiguousarray(patch)), torch.tensor(scalar, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------
    # Toplu okuma (PyTorch >= 2.0 __getitems__ protokolü)
    # ------------------------------------------------------------------
    # DataLoader bir batch'i normalde __getitem__'i 64 kez çağırarak toplar; bu da
    # 64 (patches) + 64 (aux) ayrı HDF5 hyperslab okuması demektir. Asıl maliyet
    # diskte değil, çağrı başına düşen Python/HDF5 yükündedir. Burada batch'in
    # tamamını TEK okumada alıyoruz. Dönen listenin sırası __getitem__ ile birebir
    # aynıdır; yani örnekler, sıraları ve değerleri değişmez, sadece daha az çağrı yapılır.
    def __getitems__(self, indices):
        if self.h5_file is None:
            self._open()

        idx = np.asarray(indices, dtype=np.int64)
        order = np.argsort(idx, kind="stable")     # h5py artan sıralı seçim ister
        srt = idx[order]
        lo, hi = int(srt[0]), int(srt[-1]) + 1
        span, need = hi - lo, len(srt)

        # Aralık, gerçekten okunacak satır sayısının 3 katından darsa dilim okuması
        # (tek, bitişik hyperslab) noktasal seçimden hızlıdır; değilse fancy indeksleme.
        if span <= 3 * need:
            rel = srt - lo
            patches = self.patches_ds[lo:hi][rel]
            aux_raw = self.aux_ds[lo:hi][rel] if self.aux_ds is not None else None
            scalars = self.scalars_ds[lo:hi][rel] if self.scalars_ds is not None else None
        else:
            uniq, inv = np.unique(srt, return_inverse=True)
            patches = self.patches_ds[uniq][inv]
            aux_raw = self.aux_ds[uniq][inv] if self.aux_ds is not None else None
            scalars = self.scalars_ds[uniq][inv] if self.scalars_ds is not None else None

        labels = self.labels_ds[lo:hi][srt - lo] if span <= 3 * need else self.labels_ds[srt]
        if self.f16:
            if aux_raw is not None:
                patches = np.concatenate([patches, aux_raw[:, self.aux_idx]], axis=1)
        else:
            patches = patches.astype(np.float32, copy=False)
            if aux_raw is not None:
                aux = aux_raw.astype(np.float32)[:, self.aux_idx]
                aux = (aux - self.aux_means[None]) / self.aux_stds[None]
                patches = np.concatenate([patches, aux], axis=1)

        # argsort'u geri alıp çağrıldığı sıraya döndür
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        out = [None] * len(idx)
        for pos, j in enumerate(inv_order):
            scalar = scalars[j] if scalars is not None else np.zeros(3, dtype=np.float32)
            out[pos] = (torch.from_numpy(np.ascontiguousarray(patches[j])),
                        torch.tensor(scalar, dtype=torch.float32),
                        torch.tensor(labels[j], dtype=torch.long))
        return out
