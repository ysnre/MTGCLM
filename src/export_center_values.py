"""
export_center_values.py — Kompakt H5'ten her satırın istasyon pikseli (16,16) değerlerini ham birimlerde dışa aktarır (P8 hata
analizi ve benzeri tablo işleri için; H5'e her seferinde gitmemek için). Salt okuma.

Çıktı data/processed/center_values.npz:
  base   [N,17] float32  16 FCI kanalı (VIS/NIR %, IR K) + radar (mavi kanal), ham birim (z-skor geri çevrilmiş)
  aux    [N,5]  float32  t2m (°C), rh2m (%), elev (m), cos_sza, radar_cov — H5'te zaten ham
  radar_max [N] float32  33x33 yamada radar en büyüğü (ham)
  base_names, aux_names

    python src/export_center_values.py --h5 C:/mtg/mtg_train_v2.h5
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import BASE_CHANNELS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/processed/center_values.npz"))
    ap.add_argument("--block", type=int, default=8192)
    a = ap.parse_args()
    t0 = time.time()
    with h5py.File(a.h5, "r") as f:
        N = len(f["labels"])
        means, stds = np.asarray(f.attrs["means"]), np.asarray(f.attrs["stds"])
        aux_names = json.loads(f.attrs["aux_channels"])
        P, A = f["patches"], f["aux"]
        ri = BASE_CHANNELS.index("radar")
        base = np.empty((N, P.shape[1]), np.float32)
        aux = np.empty((N, A.shape[1]), np.float32)
        rmax = np.empty(N, np.float32)
        for s in range(0, N, a.block):
            e = min(s + a.block, N)
            p = P[s:e].astype(np.float32)
            base[s:e] = p[:, :, 16, 16] * stds + means
            rmax[s:e] = p[:, ri].max(axis=(1, 2)) * stds[ri] + means[ri]
            aux[s:e] = A[s:e][:, :, 16, 16].astype(np.float32)
            if (s // a.block) % 10 == 0:
                print(f"  {e}/{N}  ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(a.out, base=base, aux=aux, radar_max=rmax, base_names=np.array(BASE_CHANNELS),
                        aux_names=np.array(aux_names))
    print(f"Yazıldı: {a.out}  ({N} satır, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
