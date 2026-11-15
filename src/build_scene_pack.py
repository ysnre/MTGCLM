"""
build_scene_pack.py — P5: seçilen sahneler için Fig. 8 girdilerini 800x1000 model ızgarasında tek dosyada toplar.

Sahne başına data/processed/scene_pack/pack_<YYYYmmddHHMMSS>.npz:
  img      [800,1000,3] uint8  gündüz: EUMETSAT doğal renk (R NIR1.6, G VIS0.8, B VIS0.6; güneş zenitine göre
                               normalize, 0–100 %); gece: IR10.5 parlaklık sıcaklığı, ters gri (210–300 K)
  is_day   bool                alanın ortasında cos(SZA) > 0,2
  prob     [800,1000] float32  model olasılığı (artifacts_v3/scenes/scene_<ts>_<exp>*.nc, infer_scene.py)
  clm      [800,1000] int8     FCI CLM ikili: 1 bulutlu (cloud_state 2 bulaşmış, 3 dolu), 0 açık (1, 8 kar/buz), -1 geçersiz
  clm_state[800,1000] int8     ham cloud_state (en yakın CLM pikseli)
  nwc      [800,1000] int8     NWC SAF CT ikili: 1 bulutlu (sınıf 5–15), 0 açık (0–4), -1 tanınmayan renk/yok
  nwc_class[800,1000] uint8    ham sınıf
  st_lat, st_lon, st_label     sahnenin etiketli test istasyonları
  meta (json)                  dosya adları, CLM tekrar döngüsü ve dönemi (C/O), NWC slotu
Ayrıca artifacts_v3/scenes/scene_pack_summary.csv: sahne başına istasyonlarda model/CLM/NWC doğruluğu (aynı istasyonlar,
ürünün geçerli olduğu yerde), bulutlu piksel oranları ve model–ürün piksel uyumu (Bulgular metni bu dosyadan).

Izgara eşlemeleri:
  CLM: FCI 2 km tam disk, geos (h 35 786,4 km, sweep y, WGS84). col = floor(X/2000,02 + 2784), row = floor(Y/2000,02 + 2784);
       satır 0 güney. data/clm_station_index.npz'deki satpy indeksleriyle 279/279 istasyonda birebir doğrulandı (15 Kas 2026).
  NWC: Mercator PNG, köşeler MGM (extract_nwcsaf_ct.py), piksel = floor.

    python src/build_scene_pack.py            (varsayılan: artifacts_v3/scenes/scenes_selected.csv)
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import load_meta                                   # noqa: E402
from extract_nwcsaf_ct import H as NWC_H, W as NWC_W, pixel_of, sample_time_to_slot, to_classes  # noqa: E402
from grid_utils import LatLonGrid, cos_solar_zenith                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLM_RE = re.compile(r"_OPE_(\d{14})_(\d{14})_N__([A-Z])_(\d{4})_")
CLM_PX, CLM_OFF = 2000.02, 2784.0


def geos_xy(lat, lon, h=35786400.0, a=6378137.0, b=6356752.31414):
    """Yerdurağan projeksiyon (sweep y, alt boylam 0), metre — pyproj geos ile < 1e-6 m aynı."""
    la, lo = np.radians(lat), np.radians(lon)
    c_lat = np.arctan((b * b) / (a * a) * np.tan(la))
    rl = b / np.sqrt(1 - (a * a - b * b) / (a * a) * np.cos(c_lat) ** 2)
    r1 = h + a - rl * np.cos(c_lat) * np.cos(lo)
    r2 = -rl * np.cos(c_lat) * np.sin(lo)
    r3 = rl * np.sin(c_lat)
    rn = np.sqrt(r1 * r1 + r2 * r2 + r3 * r3)
    return -np.arctan(r2 / r1) * h, np.arcsin(r3 / rn) * h


def clm_index(lat, lon):
    x, y = geos_xy(lat, lon)
    return np.floor(y / CLM_PX + CLM_OFF).astype(np.int64), np.floor(x / CLM_PX + CLM_OFF).astype(np.int64)


def check_clm_index(path):
    z = np.load(path)
    r, c = clm_index(z["lat"], z["lon"])
    ok = (r == z["row"]) & (c == z["col"])
    return int(ok.sum()), len(ok)


def find_clm(clm_dir, t_sat):
    t = t_sat.replace(minute=(t_sat.minute // 10) * 10, second=0)
    for f in glob.glob(os.path.join(clm_dir, f"*FCI-2-CLM*_OPE_{t:%Y%m%d%H%M}*.nc")):
        m = CLM_RE.search(os.path.basename(f))
        if m and m.group(1)[:12] == f"{t:%Y%m%d%H%M}":
            return f, m.group(3)
    return None, None


def find_nwc(nwc_dir, ts):
    slot = str(sample_time_to_slot([ts])[0])          # extract_nwcsaf_ct ile aynı eşleme (−9 dk, 15 dk'ya yuvarla)
    f = sorted(glob.glob(os.path.join(nwc_dir, f"{slot}_CT*.png")))
    return (f[0] if f else None), slot


def composite(nc, grid, t_sat):
    with h5py.File(nc, "r") as f:
        v06, v08, n16, i105 = (f[n][:].astype(np.float32) for n in ("vis_06", "vis_08", "nir_16", "ir_105"))
    lon2d, lat2d = np.meshgrid(grid.lons, grid.lats)
    mu = cos_solar_zenith(lat2d, lon2d, t_sat)
    is_day = bool(mu[400, 500] > 0.2)
    if is_day:
        norm = 1.0 / np.clip(mu, 0.1, None)
        rgb = np.stack([n16 * norm, v08 * norm, v06 * norm], -1)
        img = np.clip(np.nan_to_num(rgb) / 100.0, 0, 1) ** (1 / 1.2)
    else:
        g = 1 - np.clip((np.nan_to_num(i105, nan=300) - 210.0) / 90.0, 0, 1)
        img = np.repeat(g[..., None], 3, -1)
    return (img * 255).round().astype(np.uint8), is_day


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default=os.path.join(ROOT, "artifacts_v3/scenes/scenes_selected.csv"))
    ap.add_argument("--scene_dir", default=os.path.join(ROOT, "artifacts_v3/scenes"))
    ap.add_argument("--exp", default="D_unet")
    ap.add_argument("--nc_dir", default="F:/turkiye_kesitler")
    ap.add_argument("--clm_dir", default="F:/mtg_fci_clm")
    ap.add_argument("--nwc_dir", default="F:/nwcsaf")
    ap.add_argument("--clm_index", default=os.path.join(ROOT, "data/clm_station_index.npz"))
    ap.add_argument("--h5", default=None, help="Yalnız meta npz yoksa")
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--split", default=os.path.join(ROOT, "splits/split_v2_verify.npz"))
    ap.add_argument("--stations", default=os.path.join(ROOT, "data/raw_observation/station_list.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/processed/scene_pack"))
    a = ap.parse_args()
    import pandas as pd

    if os.path.exists(a.clm_index):
        ok, n = check_clm_index(a.clm_index)
        print(f"CLM ızgara eşlemesi: satpy indeksiyle {ok}/{n} istasyon aynı")
        if ok != n:
            raise SystemExit("CLM eşlemesi doğrulanamadı — durduruldu.")
    sel = pd.read_csv(a.scenes, dtype={"scene": str})
    meta = load_meta(a.meta, a.h5, a.stations)
    sp = np.load(a.split)
    test_rows = np.unique(np.concatenate([sp["test_time"], sp["test_both"]]))
    ts_all = meta["time"].dt.strftime("%Y%m%d%H%M%S").values
    os.makedirs(a.out, exist_ok=True)
    summary = []

    for _, s in sel.iterrows():
        ts = s["scene"]
        t_sat = dt.datetime.strptime(ts, "%Y%m%d%H%M%S")
        nc = sorted(glob.glob(os.path.join(a.nc_dir, f"turkiye_kesit_{ts}_*.nc")))
        if not nc:
            print(f"! {ts}: FCI sahnesi yok ({a.nc_dir})")
            continue
        grid = LatLonGrid.from_nc(nc[0])
        img, is_day = composite(nc[0], grid, t_sat)

        pf = sorted(glob.glob(os.path.join(a.scene_dir, f"scene_{ts}_{a.exp}*.nc")))
        pf = [p for p in pf if "_single" not in p and "_patch" not in p and "_reflect" not in p]
        if not pf:
            print(f"! {ts}: model çıktısı yok — önce infer_scene.py --time {ts[:12]}")
            continue
        with h5py.File(pf[0], "r") as f:
            prob = f["cloud_prob"][:].astype(np.float32)

        lon2d, lat2d = np.meshgrid(grid.lons, grid.lats)
        meta_d = {"nc": os.path.basename(nc[0]), "prob": os.path.basename(pf[0]), "category": s["category"]}
        clm = np.full(grid.shape, -1, np.int8)
        cst = np.full(grid.shape, -1, np.int8)
        cf, disp = find_clm(a.clm_dir, t_sat)
        if cf:
            r, c = clm_index(lat2d, lon2d)
            r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
            with h5py.File(cf, "r") as f:
                w = f["cloud_state"][r0:r1, c0:c1].astype(np.int16)
            v = w[r - r0, c - c0]
            v = np.where((v < 0) | (v > 9), 0, v)
            cst = v.astype(np.int8)
            clm = np.where(np.isin(v, [2, 3]), 1, np.where(np.isin(v, [1, 8]), 0, -1)).astype(np.int8)
            meta_d.update(clm=os.path.basename(cf), clm_disposition=disp)
        else:
            print(f"  {ts}: CLM dosyası yok ({a.clm_dir})")
        nwc = np.full(grid.shape, -1, np.int8)
        ncl = np.full(grid.shape, 255, np.uint8)
        nf, slot = find_nwc(a.nwc_dir, ts)
        meta_d["nwc_slot"] = slot
        if nf:
            from PIL import Image
            rgb = np.asarray(Image.open(nf).convert("RGB")).reshape(-1, 3)
            pr, pc = pixel_of(lat2d, lon2d)
            ok = (pr >= 0) & (pr < NWC_H) & (pc >= 0) & (pc < NWC_W)
            cls = np.full(grid.shape, 255, np.uint8)
            cls[ok] = to_classes(rgb[(pr[ok] * NWC_W + pc[ok])])
            ncl = cls
            nwc = np.where(cls == 255, -1, np.where(cls >= 5, 1, 0)).astype(np.int8)
            meta_d["nwc"] = os.path.basename(nf)
        else:
            print(f"  {ts}: NWC PNG yok ({slot})")

        rows = test_rows[ts_all[test_rows] == ts]
        st = meta.iloc[rows]
        st = st[st["label"] >= 0]
        out = os.path.join(a.out, f"pack_{ts}.npz")
        np.savez_compressed(out, img=img, is_day=is_day, prob=prob, clm=clm, clm_state=cst, nwc=nwc, nwc_class=ncl,
                            st_lat=st["lat"].values, st_lon=st["lon"].values, st_label=st["label"].values.astype(np.int8),
                            lats=grid.lats, lons=grid.lons, meta=json.dumps(meta_d))
        agree = lambda m: float(((prob >= 0.5).astype(np.int8) == m)[m >= 0].mean()) if (m >= 0).any() else None
        sy, sx = grid.latlon_to_index(st["lat"].values, st["lon"].values)
        inside = (sy >= 0) & (sy < grid.ny) & (sx >= 0) & (sx < grid.nx)
        sy, sx, lab = sy[inside], sx[inside], st["label"].values[inside]

        def st_acc(binmap):
            v = binmap[sy, sx]
            ok = v >= 0
            return (float((v[ok] == lab[ok]).mean()) if ok.any() else None), int(ok.sum())
        acc_m, _ = st_acc((prob >= 0.5).astype(np.int8))
        acc_c, n_c = st_acc(clm)
        acc_n, n_n = st_acc(nwc)
        both = (clm[sy, sx] >= 0) & (nwc[sy, sx] >= 0)
        acc_m_both = float((((prob[sy, sx] >= 0.5).astype(int)) == lab)[both].mean()) if both.any() else None
        summary.append({"category": s["category"], "scene": ts, "is_day": is_day, "n_stations": int(len(lab)),
                        "n_clear": int((lab == 0).sum()), "acc_model": acc_m, "acc_clm": acc_c, "n_clm": n_c,
                        "acc_nwc": acc_n, "n_nwc": n_n, "acc_model_where_both": acc_m_both,
                        "cloudfrac_model": float(np.mean(prob >= 0.5)),
                        "cloudfrac_clm": float(np.mean(clm[clm >= 0] == 1)) if (clm >= 0).any() else None,
                        "cloudfrac_nwc": float(np.mean(nwc[nwc >= 0] == 1)) if (nwc >= 0).any() else None,
                        "agree_model_clm": agree(clm), "agree_model_nwc": agree(nwc),
                        "clm_disposition": meta_d.get("clm_disposition"), "clm_snow_frac": float(np.mean(cst == 8)),
                        "nwc_semitransparent_frac": float(np.mean((ncl >= 11) & (ncl <= 15)))})
        print(f"{ts} [{s['category']}] {'gündüz' if is_day else 'gece'}  model bulutlu {np.mean(prob >= 0.5):.2f} | "
              f"CLM bulutlu {np.mean(clm[clm >= 0] == 1) if (clm >= 0).any() else float('nan'):.2f} (uyum {agree(clm)}) | "
              f"NWC bulutlu {np.mean(nwc[nwc >= 0] == 1) if (nwc >= 0).any() else float('nan'):.2f} (uyum {agree(nwc)}) | "
              f"{len(st)} istasyon -> {out}")
    if summary:
        import pandas as pd
        sp_out = os.path.join(a.scene_dir, "scene_pack_summary.csv")
        pd.DataFrame(summary).to_csv(sp_out, index=False, encoding="utf-8")
        print(f"Özet: {sp_out}")


if __name__ == "__main__":
    main()
