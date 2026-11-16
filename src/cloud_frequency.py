"""
cloud_frequency.py — P6: batch_scenes.py çıktılarından piksel başına bulut oluşum sıklığı (çalışma dönemi; klimatoloji DEĞİL).

Sıklık = bulutlu sahne / geçerli sahne. Model: olasılık >= 0,5 (eğitimdeki karar eşiği); geçerli = uydu NaN değil ve radar
dosyası var (prob_u8 != 255). CLM karşılaştırması aynı geçerlilik maskesinde: yalnız hem modelin hem CLM'nin geçerli olduğu
sahne–pikseller (joint). Ayrıca ortalama olasılık (eşiksiz) yazılır.

Çıktılar (<out>, varsayılan artifacts_v3/climatology):
  cloud_freq_all.nc     freq_model, freq_model_joint, freq_clm_joint, mean_prob, n_valid, n_joint   [lat, lon]
  cloud_freq_month.nc   aynı değişkenler [month, lat, lon]  (month = YYYYMM)
  cloud_freq_hour.nc    aynı değişkenler [hour, lat, lon]   (hour = nominal UTC saati)
  cloud_freq_summary.csv  grup başına sahne sayıları (nominal / ok / sahne yok / hata / radar yok / CLM var),
                          alan ortalamaları (kara: DEM > 0), istasyon doğrulaması özeti
  station_frequency.csv istasyon başına SYNOP gözlenen sıklık (aynı sahne zamanlarında, etiket 0 / >= 5 okta) ile model ve
                          CLM sıklığı (istasyon pikseli); 'subset' sütunu: all (eğitim örnekleri dahil, iyimser), test_weeks
                          (test_time ∪ test_both satırları), unseen_stations (test_station ∪ test_both satırları)
  station_freq_summary.csv alt küme başına bias / MAE / korelasyon (model ve CLM)
  cloud_freq_regions.csv  bölge × grup (all / month_YYYYMM / hour_HH) alan ortalamaları: Türkiye (kara ve bir AWOS istasyonuna
                          <= 25 km), kıyı (denize <= 30 km), doğu yaylaları (>= 38,5° D, >= 1500 m, kıyı değil), iç kesim (kalan)
  regions.npz             bölge maskeleri (şekil için)
--station_only: harita toplamasını atlar, yalnız istasyon doğrulamasını (yeniden) yazar.
--regions_only: yalnız bölge ortalamalarını NetCDF'lerden (yeniden) yazar.
NetCDF'ler h5py ile (CF öznitelikleri, boyut ölçekleri) yazılır.

    python src/cloud_frequency.py
"""
import argparse
import csv
import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import load_meta     # noqa: E402
from grid_utils import LatLonGrid       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THR_U8 = 125                             # 0,5 * 250
VARS = ("freq_model", "freq_model_joint", "freq_clm_joint", "mean_prob", "n_valid", "n_joint")


class Acc:
    """Bir grup (tüm dönem / ay / saat) için piksel sayaçları."""

    def __init__(self, shape):
        self.n_valid = np.zeros(shape, np.int32)
        self.n_cloud = np.zeros(shape, np.int32)
        self.sum_prob = np.zeros(shape, np.float64)
        self.n_joint = np.zeros(shape, np.int32)
        self.n_cloud_mj = np.zeros(shape, np.int32)
        self.n_cloud_cj = np.zeros(shape, np.int32)
        self.n_scenes = 0
        self.n_scenes_clm = 0

    def add(self, pu8, clm):
        v = pu8 != 255
        cm = v & (pu8 >= THR_U8)
        self.n_valid += v
        self.n_cloud += cm
        self.sum_prob += np.where(v, pu8, 0) / 250.0
        self.n_scenes += 1
        cv = clm >= 0
        if cv.any():
            self.n_scenes_clm += 1
            j = v & cv
            self.n_joint += j
            self.n_cloud_mj += j & cm
            self.n_cloud_cj += j & (clm == 1)

    def fields(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            nv = np.where(self.n_valid > 0, self.n_valid, np.nan)
            nj = np.where(self.n_joint > 0, self.n_joint, np.nan)
            return {"freq_model": (self.n_cloud / nv).astype(np.float32),
                    "freq_model_joint": (self.n_cloud_mj / nj).astype(np.float32),
                    "freq_clm_joint": (self.n_cloud_cj / nj).astype(np.float32),
                    "mean_prob": (self.sum_prob / nv).astype(np.float32),
                    "n_valid": self.n_valid, "n_joint": self.n_joint}


ATTRS = {
    "freq_model": ("cloud occurrence frequency, model (probability >= 0.5)", "1"),
    "freq_model_joint": ("cloud occurrence frequency, model, scenes where FCI CLM is also valid", "1"),
    "freq_clm_joint": ("cloud occurrence frequency, FCI CLM (contaminated or filled), same scenes as freq_model_joint", "1"),
    "mean_prob": ("mean model cloud probability", "1"),
    "n_valid": ("number of valid model scenes", "1"),
    "n_joint": ("number of scenes valid for both model and FCI CLM", "1"),
}


def write_nc(path, grid, groups, dim=None, coord=None, attrs=None):
    """groups: [Acc] (dim None ise tek eleman)."""
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        lat = f.create_dataset("lat", data=grid.lats)
        lon = f.create_dataset("lon", data=grid.lons)
        lat.attrs.update(units="degrees_north", standard_name="latitude")
        lon.attrs.update(units="degrees_east", standard_name="longitude")
        lat.make_scale("lat")
        lon.make_scale("lon")
        scales = [lat, lon]
        if dim:
            d = f.create_dataset(dim, data=np.asarray(coord, np.int32))
            d.make_scale(dim)
            d.attrs["long_name"] = {"month": "month (YYYYMM)", "hour": "nominal hour (UTC)"}[dim]
            scales = [d] + scales
        F = [g.fields() for g in groups]
        for v in VARS:
            arr = np.stack([x[v] for x in F]) if dim else F[0][v]
            ch = ((1,) if dim else ()) + (200, 250)
            ds = f.create_dataset(v, data=arr, compression="gzip", compression_opts=4, chunks=ch,
                                  **({"fillvalue": np.float32(np.nan)} if arr.dtype == np.float32 else {}))
            ds.attrs.update(long_name=ATTRS[v][0], units=ATTRS[v][1])
            for i, s in enumerate(scales):
                ds.dims[i].attach_scale(s)
        if dim:
            f.create_dataset("n_scenes", data=np.array([g.n_scenes for g in groups], np.int32)).dims[0].attach_scale(f[dim])
            f.create_dataset("n_scenes_clm", data=np.array([g.n_scenes_clm for g in groups], np.int32)).dims[0].attach_scale(f[dim])
        else:
            f.attrs["n_scenes"] = groups[0].n_scenes
            f.attrs["n_scenes_clm"] = groups[0].n_scenes_clm
        f.attrs.update({"Conventions": "CF-1.8", "title": "MTGCLM cloud occurrence frequency during the study period",
                        "comment": "Study period only (Mar-Apr 2025, Dec 2025-Jun 2026); not a climatology. "
                                   "Model: U-Net, probability >= 0.5; FCI CLM: cloud contaminated or filled = cloudy.",
                        **(attrs or {})})
    os.replace(tmp, path)


def region_masks(grid, dem, awos_csv, d_tr_km=25.0, d_coast_km=30.0, east_lon=38.5, east_elev=1500.0):
    from scipy.spatial import cKDTree
    from update_h5_aux_channels import parse_awos_stationlist
    st = parse_awos_stationlist(awos_csv)
    sy, sx = grid.to_km(np.array([v[0] for v in st.values()]), np.array([v[1] for v in st.values()]))
    lat2d, lon2d = grid.full_latlon()
    gy, gx = grid.to_km(lat2d, lon2d)
    q = np.c_[gy.ravel(), gx.ravel()]
    d_st = cKDTree(np.c_[sy, sx]).query(q)[0].reshape(grid.shape)
    land = dem > 0
    d_sea = cKDTree(q[~land.ravel()]).query(q)[0].reshape(grid.shape)
    tr = land & (d_st <= d_tr_km)
    coast = tr & (d_sea <= d_coast_km)
    east = tr & ~coast & (lon2d >= east_lon) & (dem >= east_elev)
    interior = tr & ~coast & ~east
    return {"turkiye": tr, "coastal": coast, "interior": interior, "eastern_highlands": east}


def region_means(a, grid, dem):
    M = region_masks(grid, dem, a.awos_stations)
    np.savez_compressed(os.path.join(a.out, "regions.npz"), **M)
    rows = []
    for fname, dim in (("cloud_freq_all.nc", None), ("cloud_freq_month.nc", "month"), ("cloud_freq_hour.nc", "hour")):
        with h5py.File(os.path.join(a.out, fname), "r") as f:
            coords = [None] if dim is None else list(f[dim][:])
            V = {v: f[v][:] for v in ("freq_model", "freq_model_joint", "freq_clm_joint", "mean_prob")}
            ns = [int(f.attrs["n_scenes"])] if dim is None else list(f["n_scenes"][:])
        for i, c in enumerate(coords):
            grp = "all" if dim is None else (f"month_{c}" if dim == "month" else f"hour_{int(c):02d}")
            for rn, m in M.items():
                r = {"group": grp, "region": rn, "n_pixels": int(m.sum()), "n_scenes": int(ns[i])}
                for v, arr in V.items():
                    x = arr if dim is None else arr[i]
                    r[v] = float(np.nanmean(x[m]))
                r["clm_minus_model_joint"] = r["freq_clm_joint"] - r["freq_model_joint"]
                rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "cloud_freq_regions.csv"), index=False, encoding="utf-8")
    print(R[R["group"] == "all"].round(3).to_string(index=False))
    return R


def station_validation(a, S, grid, shape):
    """SYNOP etiketleri, sahne zamanında (aynı nc zaman damgası); alt kümeler: all / test_weeks / unseen_stations."""
    meta = load_meta(a.meta, a.h5, a.stations)
    meta["ts"] = meta["time"].dt.strftime("%Y%m%d%H%M%S")
    sp = np.load(a.split)
    subsets = {"all": None,
               "test_weeks": np.unique(np.concatenate([sp["test_time"], sp["test_both"]])),
               "unseen_stations": np.unique(np.concatenate([sp["test_station"], sp["test_both"]]))}
    ok_scenes = set(S.loc[S["valid"], "scene"].astype(str))
    obs = meta[(meta["src"] == "SYNOP") & (meta["label"] >= 0) & meta["ts"].isin(ok_scenes)].copy()
    y, x = grid.latlon_to_index(obs["lat"].values, obs["lon"].values)
    inside = (y >= 0) & (y < shape[0]) & (x >= 0) & (x < shape[1])
    obs, y, x = obs[inside], y[inside], x[inside]
    obs["y"], obs["x"] = y, x
    pm, pc = np.full(len(obs), np.nan), np.full(len(obs), np.nan)
    pos = {r: i for i, r in enumerate(obs.index)}
    for ts, g in obs.groupby("ts"):
        z = np.load(os.path.join(a.src, "scenes", f"cf_{ts}.npz"))
        pu8, clm = z["prob_u8"][g["y"].values, g["x"].values], z["clm"][g["y"].values, g["x"].values]
        idx = np.array([pos[r] for r in g.index])
        pm[idx] = np.where(pu8 == 255, np.nan, (pu8 >= THR_U8).astype(float))
        pc[idx] = np.where(clm < 0, np.nan, clm.astype(float))
    obs["model"], obs["clm"] = pm, pc
    obs = obs[np.isfinite(obs["model"])]
    tabs, summ = [], []
    for name, rows in subsets.items():
        o = obs if rows is None else obs[obs.index.isin(rows)]
        st = o.groupby("wmoid").agg(lat=("lat", "first"), lon=("lon", "first"), elev=("elev", "first"), n=("label", "size"),
                                    obs_freq=("label", "mean"), model_freq=("model", "mean"),
                                    n_clm=("clm", "count"), clm_freq=("clm", "mean"))
        st["obs_freq_clmrows"] = o[np.isfinite(o["clm"])].groupby("wmoid")["label"].mean()
        st["model_freq_clmrows"] = o[np.isfinite(o["clm"])].groupby("wmoid")["model"].mean()
        st.insert(0, "subset", name)
        tabs.append(st.reset_index())
        s1 = st[st["n"] >= a.min_obs]
        s2 = st[st["n_clm"] >= a.min_obs]
        d1 = s1["model_freq"] - s1["obs_freq"]
        d2 = s2["clm_freq"] - s2["obs_freq_clmrows"]
        d3 = s2["model_freq_clmrows"] - s2["obs_freq_clmrows"]
        r = lambda u, v: float(np.corrcoef(u, v)[0, 1]) if len(u) > 2 else None
        summ.append({"subset": name, "stations": int(len(s1)), "obs_n": int(s1["n"].sum()),
                     "obs_freq_mean": float(s1["obs_freq"].mean()) if len(s1) else None,
                     "bias_model": float(d1.mean()) if len(s1) else None, "mae_model": float(d1.abs().mean()) if len(s1) else None,
                     "r_model": r(s1["model_freq"], s1["obs_freq"]),
                     "stations_clm": int(len(s2)), "obs_n_clm": int(s2["n_clm"].sum()),
                     "bias_clm": float(d2.mean()) if len(s2) else None, "mae_clm": float(d2.abs().mean()) if len(s2) else None,
                     "r_clm": r(s2["clm_freq"], s2["obs_freq_clmrows"]),
                     "bias_model_clmrows": float(d3.mean()) if len(s2) else None,
                     "mae_model_clmrows": float(d3.abs().mean()) if len(s2) else None,
                     "r_model_clmrows": r(s2["model_freq_clmrows"], s2["obs_freq_clmrows"])})
    pd.concat(tabs).to_csv(os.path.join(a.out, "station_frequency.csv"), index=False, encoding="utf-8")
    Q = pd.DataFrame(summ)
    Q.to_csv(os.path.join(a.out, "station_freq_summary.csv"), index=False, encoding="utf-8")
    print(Q.round(4).to_string(index=False))
    q = summ[0]
    return {"stations": q["stations"], "station_obs_n": q["obs_n"], "station_bias_model": q["bias_model"],
            "station_mae_model": q["mae_model"], "station_r_model": q["r_model"], "stations_clm": q["stations_clm"],
            "station_bias_clm": q["bias_clm"], "station_mae_clm": q["mae_clm"], "station_r_clm": q["r_clm"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(ROOT, "data/processed/cloud_freq"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/climatology"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--h5", default=None)
    ap.add_argument("--stations", default=os.path.join(ROOT, "data/raw_observation/station_list.csv"))
    ap.add_argument("--dem", default=os.path.join(ROOT, "data/dem_800x1000.npy"))
    ap.add_argument("--min_obs", type=int, default=30, help="İstasyon doğrulamasında en az gözlem")
    ap.add_argument("--split", default=os.path.join(ROOT, "splits/split_v2_verify.npz"))
    ap.add_argument("--station_only", action="store_true")
    ap.add_argument("--regions_only", action="store_true")
    ap.add_argument("--awos_stations", default=os.path.join(ROOT, "data/raw_awos_observation/stationlist_awos.csv"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.src, "scenes", "cf_*.npz")))
    if not files:
        raise SystemExit(f"{a.src}/scenes içinde sahne yok — önce batch_scenes.py")
    man = pd.read_csv(os.path.join(a.src, "manifest.csv"), dtype=str) if os.path.exists(os.path.join(a.src, "manifest.csv")) \
        else pd.DataFrame(columns=["nominal", "status"])
    man = man.drop_duplicates("nominal", keep="last")
    grid = LatLonGrid.default_turkiye()
    dem = np.load(a.dem)
    land = dem > 0
    shape = grid.shape

    if a.regions_only:
        region_means(a, grid, dem)
        return
    if a.station_only:
        S = pd.read_csv(os.path.join(a.out, "cloud_freq_scenes.csv"), dtype={"scene": str, "nominal": str})
        station_validation(a, S, grid, shape)
        return
    acc_all = Acc(shape)
    acc_m, acc_h = {}, {}
    scene_rows = []
    for k, p in enumerate(files):
        z = np.load(p)
        pu8, clm = z["prob_u8"], z["clm"]
        nominal = str(z["nominal"])
        month, hour = int(nominal[:6]), int(nominal[8:10])
        if (pu8 == 255).all():                            # radar dosyası yok vb.: sahne geçersiz
            scene_rows.append({"nominal": nominal, "scene": str(z["scene"]), "valid": False, "clm": bool((clm >= 0).any())})
            continue
        acc_all.add(pu8, clm)
        acc_m.setdefault(month, Acc(shape)).add(pu8, clm)
        acc_h.setdefault(hour, Acc(shape)).add(pu8, clm)
        scene_rows.append({"nominal": nominal, "scene": str(z["scene"]), "valid": True, "clm": bool((clm >= 0).any()),
                           "cloud_frac_land": float((pu8[land & (pu8 != 255)] >= THR_U8).mean()),
                           "_pu8": None})
        if k % 200 == 0:
            print(f"  {k + 1}/{len(files)}")
    S = pd.DataFrame(scene_rows).drop(columns=["_pu8"], errors="ignore")
    S.to_csv(os.path.join(a.out, "cloud_freq_scenes.csv"), index=False, encoding="utf-8")

    months = sorted(acc_m)
    hours = sorted(acc_h)
    write_nc(os.path.join(a.out, "cloud_freq_all.nc"), grid, [acc_all])
    write_nc(os.path.join(a.out, "cloud_freq_month.nc"), grid, [acc_m[m] for m in months], "month", months)
    write_nc(os.path.join(a.out, "cloud_freq_hour.nc"), grid, [acc_h[h] for h in hours], "hour", hours)

    # --- özet tablo
    def summ(name, acc, nominal_mask):
        mm = man[nominal_mask(man["nominal"])] if len(man) else man
        f = acc.fields()
        joint_land = land & (acc.n_joint > 0)
        return {"group": name, "n_nominal": int(len(mm)), "n_no_scene": int((mm["status"] == "no_scene").sum()) if len(mm) else 0,
                "n_error": int((mm["status"] == "error").sum()) if len(mm) else 0,
                "n_valid_scenes": acc.n_scenes, "n_clm_scenes": acc.n_scenes_clm,
                "freq_model_land": float(np.nanmean(f["freq_model"][land])),
                "freq_model_joint_land": float(np.nanmean(f["freq_model_joint"][joint_land])) if joint_land.any() else None,
                "freq_clm_joint_land": float(np.nanmean(f["freq_clm_joint"][joint_land])) if joint_land.any() else None,
                "mean_prob_land": float(np.nanmean(f["mean_prob"][land]))}
    rows = [summ("all", acc_all, lambda s: s.notna())]
    rows += [summ(f"month_{m}", acc_m[m], lambda s, m=m: s.str[:6] == str(m)) for m in months]
    rows += [summ(f"hour_{h:02d}", acc_h[h], lambda s, h=h: s.str[8:10] == f"{h:02d}") for h in hours]
    excl = S[~S["valid"]]
    rows[0]["n_invalid_scenes"] = int(len(excl))

    rows[0].update(station_validation(a, S, grid, shape))
    region_means(a, grid, dem)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "cloud_freq_summary.csv"), index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(T.drop(columns=[c for c in T.columns if c.startswith("station_")], errors="ignore").to_string(index=False))
    print({k: v for k, v in rows[0].items() if k.startswith("station")})
    print(f"Yazıldı: {a.out}")


if __name__ == "__main__":
    main()
