"""
batch_scenes.py — P6: çalışma dönemi boyunca 3 saatte bir (00, 03, ..., 21 UTC) tam sahne çıkarımı + aynı zamanın FCI CLM'si.

Dönem: veri kümesindeki etiketli günler (data/processed/h5meta_train_v2.npz; Mar–Nis 2025, Ara 2025–Haz 2026). Her nominal
saat için en yakın FCI sahnesi (<= 10 dk; Türkiye taraması ~HH:58 ya da HH:08) alınır. Girdiler ve model infer_scene.py ile
birebir aynıdır (U-Net, faz-hizalı, --seeds ortalaması). AWOS anları tek geçişte taranıp önbelleğe yazılır.

Sahne başına <out>/scenes/cf_<YYYYmmddHHMMSS>.npz:
  prob_u8   uint8  round(250 * olasılık); 255 = geçersiz (uydu NaN ya da radar dosyası yok)
  flag      uint8  infer_scene.py kalite bayrakları
  clm       int8   FCI CLM ikili (1 bulutlu: 2,3; 0 açık: 1,8; -1 geçersiz/yok) — en yakın CLM pikseli, floor 10 dk
Kaldığı yerden devam eder (manifest'te 'ok' olan ya da dosyası var olan zaman atlanır; no_scene/error yeniden denenir). <out>/manifest.csv: sahne başına durum (ok / no_scene / error),
sahne zamanı, radar/CLM dosyası, süre.

    python src/batch_scenes.py --h5 C:/mtg/mtg_train_v2.h5 --seeds 1,2,3 --threads 12
(run_all.py --stage climatology bu betiği ve ardından cloud_frequency.py'yi çağırır.)
"""
import argparse
import csv
import datetime as dt
import os
import sys
import time
import traceback

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_scene as IS                                                 # noqa: E402
from build_scene_pack import clm_index, find_clm                         # noqa: E402

ROOT = IS.ROOT


def nominal_times(meta_path, hours, start=None, end=None):
    z = np.load(meta_path)
    days = sorted(set(str(t)[:8] for t in z["timestamps"]))
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    return [dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=h) for d in days for h in hours]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--hours", default="0,3,6,9,12,15,18,21")
    ap.add_argument("--start", default=None, help="YYYYmmdd (dahil)")
    ap.add_argument("--end", default=None, help="YYYYmmdd (dahil)")
    ap.add_argument("--limit", type=int, default=None, help="Test: ilk N zaman")
    ap.add_argument("--nc_dir", default="F:/turkiye_kesitler")
    ap.add_argument("--radar_npy_dir", default="F:/radar/radar_aligned")
    ap.add_argument("--radar_dir", default=None)
    ap.add_argument("--radar_names", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--radar_csv", default=os.path.join(ROOT, "data/raw_radar/radar.csv"))
    ap.add_argument("--awos_dir", default=os.path.join(ROOT, "data/raw_awos_observation"))
    ap.add_argument("--awos_csv", default=None)
    ap.add_argument("--awos_cache", default=os.path.join(ROOT, "data/processed/awos_scene_cache"))
    ap.add_argument("--dem", default=os.path.join(ROOT, "data/dem_800x1000.npy"))
    ap.add_argument("--clm_dir", default="F:/mtg_fci_clm")
    ap.add_argument("--results_dir", default=os.path.join(ROOT, "artifacts_v3"))
    ap.add_argument("--exp", default="D_unet")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--ckpt", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/processed/cloud_freq"))
    ap.add_argument("--unet_mode", choices=["aligned", "single"], default="aligned",
                    help="aligned: 16 geçiş (D_unet için zorunlu); single: tek geçiş — yalnız P20 jitter'lı modelde (D_unet_jit; K47)")
    a = ap.parse_args()
    a.pad = "train"
    if a.unet_mode == "single" and "jit" not in a.exp:
        raise SystemExit(f"--unet_mode single yalnız jitter'lı modelle kullanılmalı (D_unet tek geçişte maskenin %38–54'ünü değiştirir; K27). "
                         f"--exp {a.exp} verildi.")
    if a.threads:
        torch.set_num_threads(a.threads)

    hours = [int(h) for h in a.hours.split(",")]
    times = nominal_times(a.meta, hours, a.start, a.end)
    if a.limit:
        times = times[:a.limit]
    sdir = os.path.join(a.out, "scenes")
    os.makedirs(sdir, exist_ok=True)
    man_path = os.path.join(a.out, "manifest.csv")
    done = set()
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["status"] == "ok":
                    done.add(r["nominal"])

    # sahne dosyalarını önceden bul (AWOS anlarını tek geçişte taramak için)
    todo = []
    for t in times:
        key = t.strftime("%Y%m%d%H%M")
        if key in done:
            continue
        try:
            nc = IS.find_scene(a.nc_dir, key)
        except SystemExit:
            todo.append((key, None))
            continue
        if os.path.exists(os.path.join(sdir, f"cf_{IS.scene_ts(nc)}.npz")):
            continue
        todo.append((key, nc))
    n_scene = sum(1 for _, p in todo if p)
    print(f"Model: {a.exp} (seeds {a.seeds}, U-Net modu {a.unet_mode}) -> {a.out}")
    print(f"Nominal zaman: {len(times)}  tamamlanmış: {len(done)}  kalan: {len(todo)} (sahnesi olan {n_scene})")
    if not todo:
        return

    norm = IS.load_norm(a.h5)
    models = IS.resolve_models(a)
    if not models[0][3]:
        raise SystemExit("batch_scenes yalnız U-Net için (kayan pencere CNN sahne başına ~3 dk/seed).")
    first = next(p for _, p in todo if p) if n_scene else None
    if first is None:
        print("Sahnesi olan zaman yok.")
    statics = IS.load_statics(a, norm, IS.LatLonGrid.from_nc(first)) if first else None
    keys = set()
    for _, p in todo:
        if p:
            ts = dt.datetime.strptime(IS.scene_ts(p), "%Y%m%d%H%M%S")
            t = IS.floor_to_10min(ts)
            keys.add(f"{t.year};{t.month};{t.day};{t.hour};{t.minute}")
    awos = IS.awos_fields(sorted(keys), statics["awos_ids"], a.awos_csv or os.path.join(a.awos_dir, "observation_awos.csv"),
                          a.awos_cache) if keys else {}

    new = not os.path.exists(man_path)
    mf = open(man_path, "a", newline="", encoding="utf-8")
    w = csv.writer(mf)
    if new:
        w.writerow(["nominal", "status", "scene", "radar_file", "clm_file", "clm_disposition", "valid_frac", "seconds", "note"])
    clm_rc = None
    t_all = time.perf_counter()
    for k, (key, nc) in enumerate(todo):
        t0 = time.perf_counter()
        if nc is None:
            w.writerow([key, "no_scene", "", "", "", "", "", "", "FCI sahnesi yok (±10 dk)"])
            mf.flush()
            continue
        ts = IS.scene_ts(nc)
        out = os.path.join(sdir, f"cf_{ts}.npz")
        try:
            X, flag, grid, t_sat, info = IS.build_inputs(nc, a, norm, statics, awos)
            H, W = grid.shape
            prob = np.mean([IS.predict_unet_full(net, X, H, W, a.unet_mode) for _, _, net, _ in models], axis=0)
            valid = (flag & (IS.FLAG_SAT_NAN | IS.FLAG_NO_RADAR)) == 0
            pu8 = np.where(valid, np.clip(np.rint(prob * 250), 0, 250), 255).astype(np.uint8)
            clm = np.full((H, W), -1, np.int8)
            cf, disp = find_clm(a.clm_dir, t_sat)
            if cf:
                if clm_rc is None:
                    lon2d, lat2d = np.meshgrid(grid.lons, grid.lats)
                    clm_rc = clm_index(lat2d, lon2d)
                r, c = clm_rc
                r0, c0 = r.min(), c.min()
                with h5py.File(cf, "r") as f:
                    win = f["cloud_state"][r0:r.max() + 1, c0:c.max() + 1].astype(np.int16)
                v = win[r - r0, c - c0]
                clm = np.where(np.isin(v, [2, 3]), 1, np.where(np.isin(v, [1, 8]), 0, -1)).astype(np.int8)
            tmp = out + ".tmp.npz"
            np.savez_compressed(tmp, prob_u8=pu8, flag=flag, clm=clm, scene=ts, nominal=key,
                                radar_file=str(info.get("radar_file")), clm_file=os.path.basename(cf) if cf else "",
                                clm_disposition=disp or "")
            os.replace(tmp, out)
            sec = time.perf_counter() - t0
            w.writerow([key, "ok", ts, info.get("radar_file") or "", os.path.basename(cf) if cf else "", disp or "",
                        round(float(valid.mean()), 4), round(sec, 1), ""])
        except Exception as e:                                   # tek sahne hatası koşuyu durdurmaz
            w.writerow([key, "error", ts, "", "", "", "", round(time.perf_counter() - t0, 1), f"{type(e).__name__}: {e}"])
            traceback.print_exc()
        mf.flush()
        el = time.perf_counter() - t_all
        if k % 10 == 0 or k == len(todo) - 1:
            print(f"  {k + 1}/{len(todo)} (%{100 * (k + 1) / len(todo):.0f})  {ts}  ({el / (k + 1):.1f} s/sahne, kalan ~{el / (k + 1) * (len(todo) - k - 1) / 3600:.1f} sa)",
                  flush=True)
    mf.close()
    print(f"Bitti: {(time.perf_counter() - t_all) / 3600:.2f} sa")


if __name__ == "__main__":
    main()
