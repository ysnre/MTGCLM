"""
make_sample_index.py
====================
Kompakt veri kümesinin (ya da açık örneklemin) satır dizini: her satır için FCI sahne zamanı, istasyon ve ızgara
konumu ile bölme kümesi. Paylaşım için (Zenodo v1.3, K63): FCI kanal değerleri, radar/AWOS alanları ve bulut
etiketleri YAZILMAZ.

  * EUMETSAT Data Policy (25 Kas 2025) Art. 4, 6.2: saatlik olmayan FCI Level-1 verisi "Recommended" — özgün sayısal
    verinin (yeniden örneklenmiş kanal değerleri dahil) yeniden dağıtımı yasak. Dizin yalnız zaman/konum verir;
    yamalar Data Store'dan indirilen FCI L1c ile `coknc.py` + `create_patch_dataset.py` zinciriyle yeniden üretilir.
  * Radar, AWOS ve SYNOP/METAR gözlemleri MGM verisidir (yalnız kullanım izni) → dizine girmez.

Sütunlar: row, set (split dosyasına göre; hiçbir kümede değilse boş), fci_scene (Türkiye kesiti dosya zaman damgası,
UTC, YYYYmmddHHMMSS), obs_time_utc (gözlem zamanı), wmoid, station_y, station_x (800 × 1000 ızgara; satır 0 = 42.995 K),
lat, lon (piksel merkezi).

Kullanım:
    python src/make_sample_index.py --h5 C:/mtg/mtg_train_v2.h5 --split splits/split_v2_verify.npz --out data/public/mtg_index.csv
    python src/make_sample_index.py --h5 data/public/mtg_sample_core.h5 --split data/public/mtg_sample_split.npz \
        --out data/public/mtg_sample_index.csv
"""
import argparse
import csv
import datetime as dt

import h5py
import numpy as np

# Izgara (CLAUDE.md §1): lon 25.01 + 0.02·x, lat 42.995 − 0.01·y
LON0, DLON, LAT0, DLAT = 25.01, 0.02, 42.995, -0.01


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="kompakt H5 ya da açık örneklem (_core.h5)")
    ap.add_argument("--split", required=True, help="bölme dosyası (.npz; küme adı → satır indisleri)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=200_000)
    args = ap.parse_args()

    sp = np.load(args.split)
    with h5py.File(args.h5, "r") as f:
        n = f["wmoids"].shape[0]
        sets = np.full(n, "", dtype=object)
        for k in sp.files:
            idx = np.asarray(sp[k]).ravel()
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise SystemExit(f"Bölme '{k}' H5 ile uyumsuz (n={n}, en büyük indis {idx.max()})")
            clash = sets[idx] != ""
            if clash.any():
                raise SystemExit(f"Bölme '{k}' başka bir kümeyle çakışıyor ({int(clash.sum())} satır)")
            sets[idx] = k
        row = f["row"][:] if "row" in f else np.arange(n)   # açık örneklemde kompakt H5 satırı
        cols = ["row", "set", "fci_scene", "obs_time_utc", "wmoid", "station_y", "station_x", "lat", "lon"]
        with open(args.out, "w", newline="", encoding="utf-8") as fo:
            w = csv.writer(fo)
            w.writerow(cols)
            for a in range(0, n, args.block):
                b = min(n, a + args.block)
                ts = f["timestamps"][a:b]
                ot = f["obs_time"][a:b]
                wm = f["wmoids"][a:b]
                sy = f["station_y"][a:b].astype(int)
                sx = f["station_x"][a:b].astype(int)
                for i in range(b - a):
                    t = ts[i].decode() if isinstance(ts[i], bytes) else str(ts[i])
                    o = dt.datetime.fromtimestamp(int(ot[i]), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    w.writerow([int(row[a + i]), sets[a + i], t, o, int(wm[i]), sy[i], sx[i],
                                f"{LAT0 + DLAT * sy[i]:.3f}", f"{LON0 + DLON * sx[i]:.2f}"])
                print(f"  {b}/{n} satır ({100 * b / n:.0f} %)", flush=True)
    filled = int((sets != "").sum())
    print(f"Yazıldı: {args.out} ({n} satır; kümeye atanmış {filled}, dışarıda {n - filled})")


if __name__ == "__main__":
    main()
