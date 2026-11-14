"""Kompakt H5'teki 16 FCI kanalinin gercek sirasini ham NetCDF kesitleriyle karsilastirarak belirler (SALT OKUMA).

Neden: create_patch_dataset.get_channel_names() sirayi ILK NetCDF dosyasinin h5py anahtar sirasindan alir; bu sira
dosyadan dosyaya degisiyor (data/raw_mtg'de 3 farkli sira goruldu, 14 Kas 2026). Kanallar her dosyada ADLA okundugu
icin veri tutarlidir, ama sira hicbir yere kaydedilmedi. baselines.py CHANNEL_NAMES ise baska bir dosyanin sirasini
varsayiyor; gece istatistikleri bu varsayimla celisiyor (ör. "vis_06" diye okunan kanal 13 gece ~249 K).

Yontem: rastgele N ornek icin yamanin merkez pikselini (16,16) z-skordan ham degere cevirir, ayni zamanin
turkiye_kesit_<YYYYMMDDHHMMSS>_*.nc dosyasinda (station_y, station_x) pikselindeki 16 degiskenle karsilastirir,
her H5 kanali icin en kucuk mutlak farki veren degiskeni secer. Cikti: kanal -> ad eslemesi, oy orani, hata.

    python src\\verify_channel_order.py --h5 C:/mtg/mtg_train_v2.h5 --nc_dir F:/turkiye_kesitler --n 200 ^
        --out artifacts_v3/channel_order.json
H5'e ve NetCDF'lere yazmaz.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import h5py
import numpy as np

SKIP = ("x", "y", "time", "spatial_ref", "crs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--nc_dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts_v3/channel_order.json")
    a = ap.parse_args()

    index = {}
    for p in glob.glob(os.path.join(a.nc_dir, "**", "turkiye_kesit_*.nc"), recursive=True):
        index.setdefault(os.path.basename(p)[14:28], p)
    print(f"NetCDF: {len(index)} dosya")

    votes, errs = defaultdict(Counter), defaultdict(list)
    used = 0
    with h5py.File(a.h5, "r") as f:
        N = f["patches"].shape[0]
        mu, sd = np.asarray(f.attrs["means"], float), np.asarray(f.attrs["stds"], float)
        rng = np.random.default_rng(a.seed)
        for i in np.sort(rng.choice(N, size=min(a.n * 3, N), replace=False)):
            if used >= a.n:
                break
            t = f["timestamps"][i]
            t = (t.decode() if isinstance(t, bytes) else str(t))[:14]
            if t not in index:
                continue
            y, x = int(f["station_y"][i]), int(f["station_x"][i])
            raw = f["patches"][i, :16, 16, 16].astype(float) * sd[:16] + mu[:16]
            with h5py.File(index[t], "r") as g:
                vals = {k: float(g[k][y, x]) for k in g.keys() if k not in SKIP and g[k].ndim == 2}
            if not all(np.isfinite(list(vals.values()))):
                continue
            used += 1
            for c in range(16):
                name, v = min(vals.items(), key=lambda kv: abs(kv[1] - raw[c]))
                votes[c][name] += 1
                errs[c].append(abs(v - raw[c]))
    if not used:
        raise SystemExit("Eslesen ornek yok: --nc_dir ve dosya adlarini kontrol edin.")
    order, report = [], []
    for c in range(16):
        name, k = votes[c].most_common(1)[0]
        order.append(name)
        report.append({"channel": c, "name": name, "vote_share": k / used, "median_abs_err": float(np.median(errs[c]))})
        print(f"{c:2d} {name:7s} oy {k / used:5.1%}  medyan |fark| {np.median(errs[c]):.4f}")
    ok = len(set(order)) == 16 and min(r["vote_share"] for r in report) > 0.9
    out = {"n_samples": used, "order": order, "unique": len(set(order)) == 16, "confident": ok, "channels": report}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(("TAMAM" if ok else "BELIRSIZ") + f": {used} ornek; sira -> {a.out}")
    try:
        from baselines import CHANNEL_NAMES
        diff = [(c, CHANNEL_NAMES[c], order[c]) for c in range(16) if CHANNEL_NAMES[c] != order[c]]
        print("baselines.CHANNEL_NAMES ile fark:", diff if diff else "yok")
    except Exception:
        pass


if __name__ == "__main__":
    main()
