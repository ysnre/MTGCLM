"""
make_public_sample.py
=====================
Kompakt H5'ten küçük, paylaşılabilir bir örneklem üretir (GitHub release / Zenodo / hakem incelemesi).
Alt küme yapısını korur: her bölmeden (train/val/test_*) orantılı ve sınıf-dengeli örnek alır,
yeni bir bölme dosyası (`<out>_split.npz`) yazar; böylece küçük dosyayla tüm hat uçtan uca koşar.

Kullanım:
    python src/make_public_sample.py --src F:/radar/mtg_train_v2.h5 --split splits/split_v1_compact.npz ^
        --out data/mtg_sample.h5 --n 2000
Sonra:
    python src/train.py --h5 data/mtg_sample.h5 --split data/mtg_sample_split.npz --model mtg_flat --aux --epochs 3
"""
import argparse
import json
import os

import h5py
import numpy as np

SUBSETS = ("train", "val", "test_time", "test_station", "test_both", "test_partial")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="F:/radar/mtg_train_v2.h5")
    ap.add_argument("--split", default="splits/split_v1_compact.npz")
    ap.add_argument("--out", default="data/mtg_sample.h5")
    ap.add_argument("--n", type=int, default=2000, help="Toplam örnek sayısı (yaklaşık)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split = np.load(args.split)
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.src, "r") as src:
        labels = src["labels"][:]
        total = sum(len(split[s]) for s in SUBSETS if s in split.files)
        picks = {}
        for s in SUBSETS:
            if s not in split.files or len(split[s]) == 0:
                continue
            idx = split[s].astype(np.int64)
            k = max(20, int(round(args.n * len(idx) / total)))
            k = min(k, len(idx))
            if s == "test_partial":
                picks[s] = np.sort(rng.choice(idx, k, replace=False))
            else:  # sınıf dengeli
                per = {c: idx[labels[idx] == c] for c in (0, 1)}
                take = {c: min(len(v), k // 2) for c, v in per.items()}
                picks[s] = np.sort(np.concatenate([rng.choice(per[c], take[c], replace=False) for c in (0, 1)]))
            print(f"  {s}: {len(picks[s])} / {len(idx)}")

        sel = np.unique(np.concatenate(list(picks.values())))
        pos = {int(v): i for i, v in enumerate(sel)}
        M = len(sel)
        print(f"Toplam {M} örnek seçildi.")

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with h5py.File(args.out, "w") as dst:
            for name in src:
                d = src[name]
                if d.shape[0] != len(labels):
                    continue
                data = d[sel]
                if name == "timestamps":
                    dst.create_dataset(name, data=np.array([t.decode() if isinstance(t, bytes) else t for t in data],
                                                           dtype=object), dtype=h5py.special_dtype(vlen=str))
                else:
                    chunks = (min(64, M),) + d.shape[1:] if d.ndim > 1 else None
                    dst.create_dataset(name, data=data, chunks=chunks,
                                       compression="gzip" if d.ndim > 1 else None, compression_opts=4 if d.ndim > 1 else None)
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["pre_shuffled"] = 1
            dst.attrs["public_sample"] = 1
            dst.attrs["sample_of"] = os.path.basename(args.src)

    out_split = os.path.splitext(args.out)[0] + "_split.npz"
    np.savez_compressed(out_split, **{s: np.array([pos[int(v)] for v in picks[s]], dtype=np.int32) for s in picks})
    with open(os.path.splitext(args.out)[0] + "_split.json", "w", encoding="utf-8") as f:
        json.dump({"source": os.path.abspath(args.src), "source_split": os.path.abspath(args.split),
                   "n": int(M), "seed": args.seed,
                   "subsets": {s: int(len(v)) for s, v in picks.items()}}, f, indent=2, ensure_ascii=False)
    print(f"Yazıldı: {args.out} ({os.path.getsize(args.out)/1e6:.0f} MB) ve {out_split}")


if __name__ == "__main__":
    main()
