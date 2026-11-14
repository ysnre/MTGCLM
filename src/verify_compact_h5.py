"""Yeniden üretilen kompakt H5'i silinen veri setinin indeksiyle karşılaştırır.

Kullanım:
    python src/verify_compact_h5.py --h5 C:/mtg/mtg_train_v2.h5

`data/processed/h5meta_train_v2.npz` dosyasındaki labels, okta, wmoids, obs_time ve timestamps
dizileri, yeni dosyadaki karşılıklarıyla eleman eleman karşılaştırılır. Hepsi birebir tutuyorsa
mevcut splits/*.npz ve artifacts*/ sonuçları geçerliliğini korur.
"""
import argparse
import os
import sys

import h5py
import numpy as np

KEYS = ("labels", "okta", "wmoids", "obs_time", "timestamps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="C:/mtg/mtg_train_v2.h5")
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    args = ap.parse_args()

    meta = np.load(args.meta, allow_pickle=True)
    ok_all = True
    with h5py.File(args.h5, "r") as f:
        n_new = len(f["labels"])
        n_ref = len(meta["labels"])
        print(f"N: yeni={n_new}  referans={n_ref}")
        if n_new != n_ref:
            print("  BASARISIZ - uzunluklar farkli; bolmeler yeniden uretilmeli")
            sys.exit(1)

        for k in KEYS:
            if k not in f:
                print(f"{k:11s} ATLANDI - yeni dosyada yok")
                ok_all = False
                continue
            a = f[k][:]
            b = meta[k]
            if a.dtype.kind == "S":
                a = a.astype(str)
            if b.dtype.kind == "S":
                b = b.astype(str)
            if a.dtype.kind in "SU" or b.dtype.kind in "SU":
                a = a.astype(str)
                b = b.astype(str)
            same = a.shape == b.shape and bool((a == b).all())
            if same:
                print(f"{k:11s} TAMAM")
            else:
                ok_all = False
                diff = np.flatnonzero(a != b) if a.shape == b.shape else np.array([0])
                print(f"{k:11s} BASARISIZ - farkli eleman: {diff.size}, ilk indeks: {diff[0]}")
                print(f"            yeni={a[diff[0]]!r}  referans={b[diff[0]]!r}")

    print()
    if ok_all:
        print("SONUC: veri seti birebir yeniden uretildi. splits/ ve artifacts*/ gecerli.")
        sys.exit(0)
    print("SONUC: fark var. splits/*.npz yeniden uretilmeli, eski sonuclar kiyas disi.")
    sys.exit(2)


if __name__ == "__main__":
    main()
