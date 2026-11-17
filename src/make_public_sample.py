"""
make_public_sample.py
=====================
Kompakt H5'ten küçük, paylaşılabilir bir örneklem üretir (Zenodo / hakem incelemesi), İKİ KATMANDA (K20, P17):

  (a) <out>_core.h5  — AÇIK: 16 FCI kanalı + statik/geometrik alanlar (elev, cos_sza) + SYNOP/METAR etiketleri
                       ve meta veriler. WMO Unified Data Policy kapsamında paylaşılır.
  (b) <out>_mgm.h5   — MGM'ye ait: radar kanalı (patches[:,16]) + radar_cov, t2m, rh2m (AWOS IDW) (+ varsa aux_flags).
                       MGM'nin açık paylaşım teyidi gelene kadar yalnız talep üzerine verilir.

İki dosya AYNI satır sırasını taşır (`row` = kompakt H5 satır indeksi her ikisinde de var); `--merge` ile birleştirilince
kompakt H5 ile aynı biçimde 17 + 5 kanallı tam bir örneklem elde edilir.

Örnekleme: her bölme kümesinden (train/val/test_*), küme boyutuyla orantılı; küme içinde sınıf (açık/bulutlu) ×
aydınlanma (gündüz/gece, merkez pikselde cos_sza > 0) tabakalarına eşit pay (yetmeyen tabakanın payı diğerlerine dağıtılır).
test_partial (etiketsiz 1–4 okta) yalnız aydınlanmaya göre tabakalanır. Bölme indeksleri `<out>_split.npz` olarak
örneklem satırlarına göre yeniden yazılır → küçük dosyayla hat uçtan uca koşar.

Kullanım:
    python src/make_public_sample.py --src C:/mtg/mtg_train_v2.h5 --split splits/split_v2_verify.npz ^
        --out data/public/mtg_sample --n 2000
    # yalnız açık katmanla deneme (16 FCI + elev, cos_sza):
    python src/train.py --h5 data/public/mtg_sample_core.h5 --split data/public/mtg_sample_split.npz ^
        --model medium_flat --aux --epochs 3 --outdir artifacts_sample
    # iki katman elindeyse tam örneklem:
    python src/make_public_sample.py --merge data/public/mtg_sample --out data/public/mtg_sample_full.h5
    python src/make_public_sample.py --check data/public/mtg_sample      # tutarlılık denetimi
"""
import argparse
import json
import os

import h5py
import numpy as np

SUBSETS = ("train", "val", "test_time", "test_station", "test_both", "test_partial")
N_FCI = 16                       # patches[:, :16] = FCI, patches[:, 16] = radar
CORE_AUX = ("elev", "cos_sza")   # açık katman
MGM_AUX = ("t2m", "rh2m", "radar_cov")
MGM_META = ("aux_flags",)        # AWOS/radar bulunurluk bayrakları -> (b)
SKIP_META = ("patches", "aux")
CENTER = 16


def _attr_names(src):
    v = src.attrs["aux_channels"]
    return json.loads(v) if isinstance(v, (str, bytes)) else [str(x) for x in v]


def _alloc(total, sizes, minimum):
    """total'ı sizes ile orantılı dağıt (her biri en az minimum, en çok size)."""
    s = np.asarray(sizes, float)
    k = np.minimum(np.maximum(minimum, np.round(total * s / s.sum())), s).astype(int)
    return k


def _equal_strata(rng, groups, k):
    """groups: {ad: indeks dizisi}; toplam k örneği tabakalara mümkün olduğunca eşit dağıt."""
    names = [g for g in groups if len(groups[g])]
    take = {g: 0 for g in groups}
    left = k
    while left > 0 and names:
        share = max(1, left // len(names))
        nxt = []
        for g in names:
            room = len(groups[g]) - take[g]
            t = min(room, share, left)
            take[g] += t
            left -= t
            if len(groups[g]) - take[g] > 0:
                nxt.append(g)
            if left == 0:
                break
        names = nxt
    out = [rng.choice(groups[g], take[g], replace=False) for g in groups if take[g]]
    return np.sort(np.concatenate(out)) if out else np.array([], np.int64), take


def _copy_rows(src, dst, name, sel, data=None):
    data = src[name][sel] if data is None else data
    dt = data.dtype
    if dt.kind in ("O", "S") or h5py.check_string_dtype(dt) is not None:
        vals = [t.decode() if isinstance(t, bytes) else str(t) for t in data]
        dst.create_dataset(name, data=np.array(vals, dtype=object), dtype=h5py.string_dtype())
    elif data.ndim > 1:
        dst.create_dataset(name, data=data, chunks=(min(64, len(data)),) + data.shape[1:],
                           compression="gzip", compression_opts=4)
    else:
        dst.create_dataset(name, data=data)


def _jsonable(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def sample(args):
    split = np.load(args.split)
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.src, "r") as src:
        N = len(src["labels"])
        labels = src["labels"][:]
        names = _attr_names(src)
        i_cos = names.index("cos_sza")
        # merkez pikselde cos_sza, tüm satırlar için bir kez (ardışık bloklarla; nokta seçiminden çok daha hızlı)
        B = 20000
        cz_all = np.concatenate([src["aux"][i:i + B, i_cos, CENTER, CENTER].astype(np.float32) for i in range(0, N, B)])
        subs = [s for s in SUBSETS if s in split.files and len(split[s])]
        sizes = [len(split[s]) for s in subs]
        ks = _alloc(args.n, sizes, args.min_per_subset)
        picks, strata = {}, {}
        for s, k in zip(subs, ks):
            idx = np.sort(split[s].astype(np.int64))
            day = cz_all[idx] > 0
            if s == "test_partial":
                groups = {"day": idx[day], "night": idx[~day]}
            else:
                lab = labels[idx]
                groups = {f"{c}_{t}": idx[(lab == ci) & (day if t == "day" else ~day)]
                          for ci, c in ((0, "clear"), (1, "cloudy")) for t in ("day", "night")}
            picks[s], take = _equal_strata(rng, groups, int(k))
            strata[s] = {g: int(v) for g, v in take.items()}
            print(f"  {s:13s} {len(picks[s]):5d} / {len(idx):7d}  {strata[s]}")

        sel = np.unique(np.concatenate(list(picks.values())))
        pos = {int(v): i for i, v in enumerate(sel)}
        M = len(sel)
        print(f"Toplam {M} satır (kümeler arası örtüşme yok sayılırsa {sum(len(v) for v in picks.values())}).")

        base = args.out
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        patches = src["patches"][sel]
        aux = src["aux"][sel]
        means = np.asarray(src.attrs["means"]); stds = np.asarray(src.attrs["stds"])
        amu = np.asarray(src.attrs["aux_means"]); asd = np.asarray(src.attrs["aux_stds"])
        ic = [names.index(c) for c in CORE_AUX]
        im = [names.index(c) for c in MGM_AUX]
        meta = [n for n in src if n not in SKIP_META and src[n].shape[:1] == (N,)]
        common = {k: _jsonable(v) for k, v in src.attrs.items()
                  if k not in ("means", "stds", "aux_means", "aux_stds", "aux_channels")}

        with h5py.File(base + "_core.h5", "w") as dst:
            _copy_rows(src, dst, "patches", sel, patches[:, :N_FCI])
            _copy_rows(src, dst, "aux", sel, aux[:, ic])
            for n in meta:
                if n not in MGM_META:
                    _copy_rows(src, dst, n, sel)
            dst.create_dataset("row", data=sel.astype(np.int64))
            for k, v in common.items():
                dst.attrs[k] = v if not isinstance(v, (list, dict)) else json.dumps(v)
            dst.attrs["means"] = means[:N_FCI]; dst.attrs["stds"] = stds[:N_FCI]
            dst.attrs["aux_channels"] = json.dumps(list(CORE_AUX))
            dst.attrs["aux_means"] = amu[ic]; dst.attrs["aux_stds"] = asd[ic]
            dst.attrs.update({"pre_shuffled": 1, "public_sample": 1, "layer": "core (a): FCI + static + labels",
                              "sample_of": os.path.basename(args.src), "split_of": os.path.basename(args.split),
                              "sample_seed": args.seed})

        with h5py.File(base + "_mgm.h5", "w") as dst:
            _copy_rows(src, dst, "radar", sel, patches[:, N_FCI:N_FCI + 1])
            _copy_rows(src, dst, "aux", sel, aux[:, im])
            for n in meta:
                if n in MGM_META:
                    _copy_rows(src, dst, n, sel)
            dst.create_dataset("row", data=sel.astype(np.int64))
            dst.attrs["radar_mean"] = means[N_FCI:N_FCI + 1]; dst.attrs["radar_std"] = stds[N_FCI:N_FCI + 1]
            dst.attrs["aux_channels"] = json.dumps(list(MGM_AUX))
            dst.attrs["aux_means"] = amu[im]; dst.attrs["aux_stds"] = asd[im]
            dst.attrs.update({"public_sample": 1, "layer": "mgm (b): radar + AWOS fields, TSMS-owned",
                              "sample_of": os.path.basename(args.src), "sample_seed": args.seed,
                              "full_aux_order": json.dumps(names)})

    np.savez_compressed(base + "_split.npz",
                        **{s: np.array([pos[int(v)] for v in picks[s]], dtype=np.int32) for s in picks})
    info = {"source": os.path.basename(args.src), "source_split": os.path.basename(args.split), "n_rows": int(M),
            "seed": args.seed, "subsets": {s: int(len(v)) for s, v in picks.items()}, "strata": strata,
            "files": {}}
    for suf in ("_core.h5", "_mgm.h5", "_split.npz"):
        info["files"][os.path.basename(base + suf)] = os.path.getsize(base + suf)
    with open(base + "_split.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    for k, v in info["files"].items():
        print(f"Yazıldı: {k} ({v / 1e6:.1f} MB)")


def merge(base, out):
    """(a) + (b) -> kompakt H5 biçiminde tam örneklem (17 temel kanal + 5 aux, kaynak sırası)."""
    with h5py.File(base + "_core.h5", "r") as a, h5py.File(base + "_mgm.h5", "r") as b:
        if not np.array_equal(a["row"][:], b["row"][:]):
            raise SystemExit("Katmanların satır sırası farklı; aynı koşunun dosyalarını verin.")
        order = json.loads(b.attrs["full_aux_order"])
        an, bn = json.loads(a.attrs["aux_channels"]), json.loads(b.attrs["aux_channels"])
        aux = np.empty((len(a["row"]), len(order)) + a["aux"].shape[2:], a["aux"].dtype)
        mu = np.empty(len(order)); sd = np.empty(len(order))
        for src, nm in ((a, an), (b, bn)):
            for j, c in enumerate(nm):
                i = order.index(c)
                aux[:, i] = src["aux"][:, j]; mu[i] = src.attrs["aux_means"][j]; sd[i] = src.attrs["aux_stds"][j]
        with h5py.File(out, "w") as d:
            _copy_rows(a, d, "patches", slice(None), np.concatenate([a["patches"][:], b["radar"][:]], axis=1))
            _copy_rows(a, d, "aux", slice(None), aux)
            for n in list(a) + list(b):
                if n not in ("patches", "aux", "radar") and n not in d:
                    _copy_rows(a if n in a else b, d, n, slice(None))
            for k, v in a.attrs.items():
                d.attrs[k] = v
            d.attrs["means"] = np.concatenate([a.attrs["means"], b.attrs["radar_mean"]])
            d.attrs["stds"] = np.concatenate([a.attrs["stds"], b.attrs["radar_std"]])
            d.attrs["aux_channels"] = json.dumps(order)
            d.attrs["aux_means"] = mu; d.attrs["aux_stds"] = sd
            d.attrs["layer"] = "merged (a)+(b)"
    print(f"Yazıldı: {out}")


def check(base, src_path=None):
    ok = True
    sp = np.load(base + "_split.npz")
    with h5py.File(base + "_core.h5", "r") as a:
        n = len(a["labels"]); lab = a["labels"][:]
        print(f"core: patches {a['patches'].shape}, aux {a['aux'].shape} {a.attrs['aux_channels']}")
        for s in sp.files:
            v = sp[s]
            bad = (v.min() < 0) or (v.max() >= n)
            lv = set(np.unique(lab[v]).tolist())
            exp = {-1} if s == "test_partial" else {0, 1}
            ok &= (not bad) and lv <= exp
            print(f"  {s:13s} n={len(v):5d} etiketler={sorted(lv)} {'OK' if not bad and lv <= exp else 'HATA'}")
        if os.path.exists(base + "_mgm.h5"):
            with h5py.File(base + "_mgm.h5", "r") as b:
                same = np.array_equal(a["row"][:], b["row"][:])
                ok &= same
                print(f"mgm : radar {b['radar'].shape}, aux {b['aux'].shape} {b.attrs['aux_channels']}  satır eşleşmesi {'OK' if same else 'HATA'}")
                if src_path:
                    rows = a["row"][:]
                    with h5py.File(src_path, "r") as s:
                        k = np.sort(np.random.default_rng(0).choice(len(rows), min(50, len(rows)), replace=False))
                        p = s["patches"][rows[k]]
                        e1 = np.array_equal(p[:, :N_FCI], a["patches"][k]) and np.array_equal(p[:, N_FCI:], b["radar"][k])
                        e2 = np.array_equal(s["labels"][rows[k]], a["labels"][k])
                        ok &= e1 and e2
                        print(f"kaynakla birebir (50 satır): patches {'OK' if e1 else 'HATA'}, labels {'OK' if e2 else 'HATA'}")
    print("SONUÇ:", "OK" if ok else "HATA")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="Kompakt H5 (C:/mtg/mtg_train_v2.h5); --h5 ile aynı")
    ap.add_argument("--h5", default=None, help="--src ile eşdeğer (proje kuralı: her zaman --h5 verin)")
    ap.add_argument("--split", default="splits/split_v2_verify.npz")
    ap.add_argument("--out", default="data/public/mtg_sample", help="Çıktı öneki (…_core.h5, …_mgm.h5, …_split.npz)")
    ap.add_argument("--n", type=int, default=2000, help="Toplam örnek sayısı (yaklaşık)")
    ap.add_argument("--min_per_subset", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", default=None, metavar="ÖNEK", help="(a)+(b) katmanlarını --out dosyasına birleştir")
    ap.add_argument("--check", default=None, metavar="ÖNEK", help="Örneklemi denetle (--h5 verilirse kaynakla kıyasla)")
    args = ap.parse_args()
    args.src = args.src or args.h5
    if args.merge:
        return merge(args.merge, args.out)
    if args.check:
        raise SystemExit(0 if check(args.check, args.src) else 1)
    if not args.src:
        ap.error("--h5 (veya --src) gerekli")
    sample(args)
    check(args.out, args.src)


if __name__ == "__main__":
    main()
