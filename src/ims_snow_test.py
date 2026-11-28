"""C1 (28 Kas, ön-kayıtlı 27 Kas 18:50): bağımsız kar analizi (NOAA/USNIC IMS 1 km) ile kış gündüzü CLM − model farkı.

Soru: FCI CLM'nin doğu yaylalarında kış gündüzü modelden fazla bulut göstermesi (Fig. 9b/f) IMS'in karlı saydığı piksellerde mi
toplanıyor? Yeni çıkarım yok: sahne dosyaları data/processed/cloud_freq_jit/scenes/cf_*.npz (prob_u8 = round(250 p), 255 geçersiz;
clm 0/1, < 0 geçersiz; src/batch_scenes.py). IMS: G02156 1 km ASCII (ims{YYYY}{GGG}_00UTC_1km_v1.3.asc.gz), dosya adındaki günün
00 UTC haritası; ilk veri satırı haritanın ALTI; sınıflar 1 deniz, 2 karsız kara, 3 buz, 4 karlı kara. 800 × 1000 ızgaraya en yakın
komşu (pyproj; indeks data/processed/ims_index_800x1000.npz). Gün başına kesit önbelleği data/processed/ims_1km/ims_<YYYYMMDD>.npz.

Birincil (docs/revizyon_onkayit.md §3, 27 Kas 18:50): doğu yaylaları, kış gündüzü (Ara–Mar, 06/09/12 UTC), CLM ve model ikisi de
geçerli sahne-pikselleri; Δ = (CLM − model)_karlı(4) − (CLM − model)_karsız(2); model eşiği 0,5; gün blokları bootstrap (B 2000, tohum 0).
Kural: GA sıfırı dışlar, Δ ≥ 0,05 VE frekans-eşleyen eşikte (val, 0,176) de aynı → "fazla, bağımsız analizde karlı piksellerde".
İkincil: ertesi günün haritası, tüm Türkiye maskesi, aylar.
Çıktı: artifacts_v3/analysis/ims_snow_test.csv, ims_snow_test_months.csv
    python src/ims_snow_test.py --ims_dir F:/ims
"""
import argparse
import datetime as dt
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_utils import LatLonGrid  # noqa: E402

N = 24576
PROJ = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-80 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6356257 +units=m +no_defs"
SNOW, NOSNOW = 4, 2
WINTER, HOURS = (12, 1, 2, 3), (6, 9, 12)


def ims_index(path):
    if os.path.exists(path):
        z = np.load(path)
        return z["col"], z["row_bottom"]
    from pyproj import Transformer
    la, lo = LatLonGrid.default_turkiye().full_latlon()
    x, y = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True).transform(lo, la)
    col = np.floor((x + 12288000) / 1000).astype(np.int32)
    rb = np.floor((y + 12288000) / 1000).astype(np.int32)
    np.savez_compressed(path, col=col, row_bottom=rb)
    return col, rb


def ims_day(day, ims_dir, cache_dir, col, rb):
    """day (date) → 800×1000 uint8 IMS sınıfı ya da None (dosya yok)."""
    cp = os.path.join(cache_dir, f"ims_{day:%Y%m%d}.npz")
    if os.path.exists(cp):
        return np.load(cp)["ims"]
    f = os.path.join(ims_dir, f"ims{day.year}{day.timetuple().tm_yday:03d}_00UTC_1km_v1.3.asc.gz")
    if not os.path.exists(f):
        return None
    need = set(np.unique(rb).tolist()); last = max(need)
    rows, i = {}, -1
    with gzip.open(f, "rt") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if len(s) != N:
                continue
            i += 1
            if i in need:
                rows[i] = np.frombuffer(s.encode(), np.uint8) - 48
            if i >= last:
                break
    A = np.zeros(rb.shape, np.uint8)
    for r in np.unique(rb):
        m = rb == r
        A[m] = rows[r][col[m]]
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cp, ims=A)
    return A


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ims_dir", default="F:/ims")
    ap.add_argument("--scenes", default="data/processed/cloud_freq_jit/scenes")
    ap.add_argument("--regions", default="artifacts_v3/climatology/regions.npz")
    ap.add_argument("--sens_info", default="artifacts_v3/climatology/cloud_freq_sensitivity_info.json")
    ap.add_argument("--index", default="data/processed/ims_index_800x1000.npz")
    ap.add_argument("--cache", default="data/processed/ims_1km")
    ap.add_argument("--out", default="artifacts_v3/analysis")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prepare_only", action="store_true", help="yalnız IMS kesit önbelleğini üret")
    a = ap.parse_args()
    col, rb = ims_index(a.index)
    R = np.load(a.regions)
    regions = {"doğu yaylaları": R["eastern_highlands"], "Türkiye": R["turkiye"]}
    t_bias = json.load(open(a.sens_info, encoding="utf-8"))["t_bias"]
    thr = {"0.5": 125, f"frekans-eşleyen ({t_bias})": int(round(250 * t_bias))}
    files = sorted(glob.glob(os.path.join(a.scenes, "cf_*.npz")))
    sc = []
    for p in files:
        z = np.load(p)
        nom = str(z["nominal"]); t = dt.datetime.strptime(nom, "%Y%m%d%H%M")
        if t.month in WINTER and t.hour in HOURS:
            sc.append((t, p))
    days = sorted({t.date() for t, _ in sc})
    for d in days:
        for dd in (d, d + dt.timedelta(days=1)):
            ims_day(dd, a.ims_dir, a.cache, col, rb)
    print(f"kış gündüzü sahne {len(sc)}, gün {len(days)}; IMS önbelleği hazır", flush=True)
    if a.prepare_only:
        return
    # gün × (bölge, harita, eşik, sınıf) → toplam(CLM − model), piksel sayısı, toplam CLM, toplam model
    acc = {}
    nused = {"aynı gün": 0, "ertesi gün": 0}
    for t, p in sc:
        z = np.load(p)
        pu8, clm = z["prob_u8"], z["clm"]
        j = (pu8 != 255) & (clm >= 0)
        if not j.any():
            continue
        for mname, dd in (("aynı gün", t.date()), ("ertesi gün", t.date() + dt.timedelta(days=1))):
            I = ims_day(dd, a.ims_dir, a.cache, col, rb)
            if I is None:
                continue
            nused[mname] += 1
            for rname, rm in regions.items():
                for tname, tu in thr.items():
                    mod = (pu8 >= tu).astype(np.int16)
                    for cname, c in (("karlı", SNOW), ("karsız", NOSNOW)):
                        sel = j & rm & (I == c)
                        k = (t.date(), rname, mname, tname, cname)
                        v = acc.setdefault(k, np.zeros(3))
                        v += [sel.sum(), clm[sel].sum(), mod[sel].sum()]
    D = pd.DataFrame([(*k, *v) for k, v in acc.items()], columns=["gün", "bölge", "harita", "eşik", "sınıf", "n", "clm", "model"])
    rng = np.random.default_rng(a.seed)
    out = []
    for (rname, mname, tname), g in D.groupby(["bölge", "harita", "eşik"], sort=False):
        P = g.pivot_table(index="gün", columns="sınıf", values=["n", "clm", "model"], aggfunc="sum").fillna(0)
        dd = P.index.values
        idx = rng.integers(0, len(dd), (a.boot, len(dd)))
        W = np.stack([np.bincount(i, minlength=len(dd)) for i in idx]).astype(float)

        def diff(c, w=None):
            n, cl, mo = P[("n", c)].values, P[("clm", c)].values, P[("model", c)].values
            if w is None:
                return (cl.sum() - mo.sum()) / n.sum(), cl.sum() / n.sum(), mo.sum() / n.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                return (w @ cl - w @ mo) / (w @ n)
        ds, cs, ms = diff("karlı"); dn, cn, mn = diff("karsız")
        b = diff("karlı", W) - diff("karsız", W)
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        out.append({"bölge": rname, "harita": mname, "eşik": tname, "gün": len(dd),
                    "n_karlı": int(P[("n", "karlı")].sum()), "n_karsız": int(P[("n", "karsız")].sum()),
                    "CLM_karlı": cs, "model_karlı": ms, "CLM_eksi_model_karlı": ds,
                    "CLM_karsız": cn, "model_karsız": mn, "CLM_eksi_model_karsız": dn,
                    "Delta": ds - dn, "GA_alt": lo, "GA_üst": hi,
                    "kural_GA_ve_0.05": bool(lo > 0 and ds - dn >= 0.05)})
    O = pd.DataFrame(out)
    prim = O[(O["bölge"] == "doğu yaylaları") & (O["harita"] == "aynı gün")]
    verdict = bool(prim["kural_GA_ve_0.05"].all())
    O["birincil"] = (O["bölge"] == "doğu yaylaları") & (O["harita"] == "aynı gün") & (O["eşik"] == "0.5")
    os.makedirs(a.out, exist_ok=True)
    O.to_csv(os.path.join(a.out, "ims_snow_test.csv"), index=False, encoding="utf-8", float_format="%.6f")
    # aylar (birincil yapılandırma, betimsel)
    G = D[(D["bölge"] == "doğu yaylaları") & (D["harita"] == "aynı gün") & (D["eşik"] == "0.5")].copy()
    G["ay"] = pd.to_datetime(G["gün"]).dt.strftime("%Y-%m")
    M = G.groupby(["ay", "sınıf"])[["n", "clm", "model"]].sum().reset_index()
    M["CLM_eksi_model"] = (M["clm"] - M["model"]) / M["n"]
    M["CLM"] = M["clm"] / M["n"]; M["model"] = M["model"] / M["n"]
    M.drop(columns=["clm"]).to_csv(os.path.join(a.out, "ims_snow_test_months.csv"), index=False, encoding="utf-8", float_format="%.6f")
    json.dump({"sahne": len(sc), "gün": len(days), "kullanılan_sahne": nused, "eşikler_u8": thr, "kural_sağlandı": verdict},
              open(os.path.join(a.out, "ims_snow_test_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.set_option("display.width", 250)
    print(O.round(4).to_string(), "\nKURAL:", "SAĞLANDI" if verdict else "sağlanmadı")


if __name__ == "__main__":
    main()
