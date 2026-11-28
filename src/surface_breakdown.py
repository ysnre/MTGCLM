"""B3 (27 Kas, ön-kayıtlı, betimsel): istasyon yüksekliği ve kıyıya uzaklık kırılımı.

Kırılımlar (+ --slope ile eğim tertilleri, B3b; + --landcover ile ESA WorldCover 2021 baskın sınıfı, B3c): yükseklik bandı (< 500, 500–1000, 1000–1500, ≥ 1500 m; station_list.csv) ve kıyı (istasyondan en yakın deniz
pikseline < 30 km / ≥ 30 km; deniz = DEM ≤ 0, cloud_frequency.region_masks ile aynı tanım). Modeller: B_full ve D_unet_jit (tohum
başına metrik, ortalama); ürünler: FCI CLM r30, NWC SAF CT r30 (src/ref_rows_cache.py; yalnız kapsanan satırlar). Etiket: tümü ve
yalnız SYNOP (ürün kıyası SYNOP'ta). Metrikler: n, BalAcc, POD, POFD, FAR; ayrıca B_full − ürün (aynı kapsanan satırlar, gün blokları
bootstrap GA; test_station'da istasyon blokları). n < 300 → belirsiz. Yorum: yalnız betimsel (docs/revizyon_onkayit.md §3, 27 Kas 14:30).
Çıktı: <out>/surface_breakdown.csv
    python src/surface_breakdown.py --dir artifacts_v3 --out artifacts_v3/analysis
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import (BLOCK, SUBSETS, aligned, balacc_from, block_counts, boot_weights, by_exp, common_rows,  # noqa: E402
                           counts, load_meta, load_runs)
from grid_utils import LatLonGrid  # noqa: E402
from ref_rows_cache import get_refs  # noqa: E402

ELEV = ([-1e9, 500, 1000, 1500, 1e9], ["<500 m", "500–1000 m", "1000–1500 m", "≥1500 m"])


def scores(pred, y):
    c = counts(pred, y)
    tp, fn, tn, fp = c
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"BalAcc": float(balacc_from(c)), "POD": tp / (tp + fn), "POFD": fp / (fp + tn), "FAR": fp / (fp + tp)}


def coast_km(lat, lon, dem_path):
    g = LatLonGrid.default_turkiye()
    dem = np.load(dem_path)
    la, lo = g.full_latlon()
    gy, gx = g.to_km(la, lo)
    sea = dem <= 0
    tree = cKDTree(np.c_[gy[sea], gx[sea]])
    sy, sx = g.to_km(np.asarray(lat), np.asarray(lon))
    return tree.query(np.c_[sy, sx])[0]


def slope_deg(lat, lon, dem_path):
    """İstasyon pikselinde DEM eğimi (derece): 3 × 3 merkezî fark, piksel 0,02° D–B × 0,01° K–G (km, grid_utils.to_km ile)."""
    g = LatLonGrid.default_turkiye()
    dem = np.load(dem_path).astype(float)
    la, lo = g.full_latlon()
    gy, gx = g.to_km(la, lo)
    dy_km = np.abs(np.gradient(gy, axis=0)); dx_km = np.abs(np.gradient(gx, axis=1))
    dzdy = np.gradient(dem, axis=0) / (dy_km * 1000.0); dzdx = np.gradient(dem, axis=1) / (dx_km * 1000.0)
    sl = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    iy, ix = g.latlon_to_index(np.asarray(lat), np.asarray(lon))
    return sl[np.asarray(iy), np.asarray(ix)]


WC = {10: "ağaç", 20: "çalı", 30: "çayır", 40: "tarım", 50: "yerleşim", 60: "çıplak/seyrek", 70: "kar/buz", 80: "su",
      90: "sulak alan", 95: "mangrov", 100: "yosun/liken"}


def landcover_class(lat, lon, tif):
    """ESA WorldCover 2021 (v200), GEE'de bu ızgaraya baskın sınıf (mode) olarak indirgenmiş GeoTIFF; istasyon pikseli."""
    from PIL import Image
    a = np.array(Image.open(tif))
    assert a.shape == (800, 1000), a.shape
    iy, ix = LatLonGrid.default_turkiye().latlon_to_index(np.asarray(lat), np.asarray(lon))
    return [WC.get(int(v), f"sınıf {int(v)}") for v in a[np.asarray(iy), np.asarray(ix)]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--out", default="artifacts_v3/analysis")
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--dem", default="data/dem_800x1000.npy")
    ap.add_argument("--models", default="B_full,D_unet_jit")
    ap.add_argument("--slope", action="store_true", help="B3b: eğim tertilleri de (ön-kayıt 28 Kas 10:50)")
    ap.add_argument("--landcover", default=None, help="B3c: data/landcover/mtgclm_worldcover2021_mode.tif (ön-kayıt 28 Kas 10:50)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    meta = load_meta(a.meta, None, a.stations)
    st = meta.drop_duplicates("wmoid")[["wmoid", "lat", "lon"]]
    ck = dict(zip(st.wmoid, coast_km(st.lat.values, st.lon.values, a.dem)))
    meta["kıyı"] = np.where(meta.wmoid.map(ck) < 30, "<30 km", "≥30 km")
    meta["yükseklik"] = pd.cut(meta.elev, ELEV[0], labels=ELEV[1]).astype(str)
    factors = ["yükseklik", "kıyı"]
    if a.slope:
        sd = dict(zip(st.wmoid, slope_deg(st.lat.values, st.lon.values, a.dem)))
        q = np.quantile(list(sd.values()), [1 / 3, 2 / 3])
        lab = [f"<{q[0]:.1f}°", f"{q[0]:.1f}–{q[1]:.1f}°", f"≥{q[1]:.1f}°"]
        meta["eğim"] = pd.cut(meta.wmoid.map(sd), [-1, q[0], q[1], 91], labels=lab).astype(str)
        pd.DataFrame({"wmoid": list(sd), "eğim_derece": list(sd.values())}).to_csv(
            os.path.join(a.out, "station_slope.csv"), index=False, encoding="utf-8", float_format="%.3f")
        factors.append("eğim")
    if a.landcover:
        lc = dict(zip(st.wmoid, landcover_class(st.lat.values, st.lon.values, a.landcover)))
        meta["arazi"] = meta.wmoid.map(lc).astype(str)
        pd.DataFrame({"wmoid": list(lc), "arazi": list(lc.values())}).to_csv(
            os.path.join(a.out, "station_landcover.csv"), index=False, encoding="utf-8")
        factors.append("arazi")
    refs = get_refs(meta)
    E = by_exp(load_runs(a.dir))
    models = [m for m in a.models.split(",") if m in E]
    out = []
    for s in SUBSETS:
        runs = {m: [r for r in E[m] if s in r["sets"]] for m in models}
        rows = common_rows([r for v in runs.values() for r in v], s)
        mt = meta.iloc[rows].reset_index(drop=True)
        P = {m: [(lambda yp: ((yp[1] >= 0.5).astype(int), yp[0]))(aligned(r, s, rows)) for r in v] for m, v in runs.items()}
        y = next(iter(P.values()))[0][1]
        for lab in ("tümü", "SYNOP"):
            lm = np.ones(len(rows), bool) if lab == "tümü" else (mt.src.values == "SYNOP")
            for fac in factors:
                for lev in sorted(mt[fac].unique()):
                    sel = lm & (mt[fac].values == lev)
                    if sel.sum() == 0:
                        continue
                    base = {"küme": s, "etiket": lab, "etken": fac, "düzey": lev}
                    nst = {"istasyon": int(mt.wmoid[sel].nunique())}
                    for m in models:
                        sc = pd.DataFrame([scores(pr[sel], y[sel]) for pr, _ in P[m]]).mean()
                        out.append({**base, "ürün": m, **nst, "n": int(sel.sum()), **sc.to_dict(), "belirsiz": bool(sel.sum() < 300)})
                    if lab != "SYNOP":
                        continue
                    for prod in ("CLM_r30", "NWC_r30"):
                        rp = refs[prod][rows]
                        cs = sel & np.isfinite(rp)
                        if cs.sum() == 0:
                            continue
                        pr = rp[cs].astype(int)
                        codes, uni = pd.factorize(mt[BLOCK[s]].values[cs])
                        W = boot_weights(len(uni), a.boot, rng)
                        bm = np.mean([balacc_from(W @ block_counts(p[cs], y[cs], codes, len(uni))) for p, _ in P["B_full"]], 0)
                        br = balacc_from(W @ block_counts(pr, y[cs], codes, len(uni)))
                        d = bm - br
                        ptm = np.mean([balacc_from(counts(p[cs], y[cs])) for p, _ in P["B_full"]])
                        sc = scores(pr, y[cs])
                        lo, hi = np.nanpercentile(d, [2.5, 97.5])
                        out.append({**base, "ürün": prod, **nst, "n": int(cs.sum()), **sc, "belirsiz": bool(cs.sum() < 300),
                                    "B_full_aynı_satır": ptm, "B_full_eksi_ürün": ptm - sc["BalAcc"], "GA_alt": lo, "GA_üst": hi})
    D = pd.DataFrame(out)
    os.makedirs(a.out, exist_ok=True)
    D.to_csv(os.path.join(a.out, "surface_breakdown.csv"), index=False, encoding="utf-8", float_format="%.6f")
    pd.set_option("display.width", 230)
    q = D[(D.küme == "test_time") & (D.etiket == "SYNOP")]
    print(q.drop(columns=["küme", "etiket"]).round(3).to_string())


if __name__ == "__main__":
    main()
