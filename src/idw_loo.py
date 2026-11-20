#!/usr/bin/env python
"""R18 — AWOS t2m / rh2m IDW interpolasyonunun leave-one-out doğrulaması (ön kayıt: docs/revizyon_onkayit.md, R18).

Aux kanalları üretilirken kullanılan IDW ile birebir aynı kurallar (update_h5_aux_channels.IDWInterpolator):
k = 8 komşu, p = 2, 0,5 km taban mesafe (< 0,5 km ise en yakının değeri), 250 km azami mesafe, sıcaklıkta 6,5 K/km yükseklik
düzeltmesi; mesafeler grid_utils.to_km (aynı referans enlemi) ile km cinsinden. Her istasyon, kendisi DIŞARIDA bırakılarak
diğerlerinden tahmin edilir. Hedef yükseklik istasyon yüksekliğidir (H5'te piksel DEM yüksekliği kullanılır; fark küçük).

Zaman örneklemi: her ay için her nominal saatte (00, 03, …, 21 UTC) rastgele --days_per_cell gün (tohum 0), dakika 00.
Yalnız etiketli verinin olduğu aylar (Mar–Nis 2025, Ara 2025–Haz 2026).

Çıktı: <out>/idw_loo.csv (değişken × alt küme × grup: n, bias, MAE, RMSE), <out>/idw_loo_info.json
Önbellek: data/processed/idw_loo_sample.npz (T, RH, zaman anahtarları, istasyon kimlikleri) — CSV taraması bir kez yapılır.
Tarama ~10–20 dk sürer (4,4 GB); ilerleme yüzde olarak yazılır.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MONTHS = ["2025-03", "2025-04", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
HOURS = [0, 3, 6, 9, 12, 15, 18, 21]


def sample_keys(days_per_cell, seed):
    rng = np.random.default_rng(seed)
    keys = []
    for m in MONTHS:
        y, mo = map(int, m.split("-"))
        nd = pd.Period(m).days_in_month
        for h in HOURS:
            for d in sorted(rng.choice(np.arange(1, nd + 1), size=days_per_cell, replace=False)):
                keys.append(f"{y};{mo};{int(d)};{h};0")
    return keys


def scan(path, keys, ids, total_bytes):
    """update_h5_aux_channels.scan_awos_csv ile aynı ayrıştırma; ilerleme yüzdesi eklenmiş."""
    t_index = {k: i for i, k in enumerate(keys)}
    s_index = {str(s): i for i, s in enumerate(ids)}
    T = np.full((len(keys), len(ids)), np.nan, np.float32)
    RH = np.full_like(T, np.nan)
    t0, done, step, nxt = time.time(), 0, 0.05, 0.05
    with open(path, "r", encoding="latin1", errors="replace") as f:
        done += len(next(f))
        for line in f:
            done += len(line)
            if done / total_bytes >= nxt:
                el = time.time() - t0
                print(f"  tarama %{100 * nxt:3.0f}  ({el / 60:.1f} dk, kalan ~{el / nxt * (1 - nxt) / 60:.1f} dk)", flush=True)
                nxt += step
            parts = line.rstrip("\n").split(";", 11)
            if len(parts) < 11:
                continue
            ti = t_index.get(parts[2] + ";" + parts[3] + ";" + parts[4] + ";" + parts[5] + ";" + parts[6])
            if ti is None:
                continue
            si = s_index.get(parts[0])
            if si is None:
                continue
            try:
                ts = parts[7].replace(",", ".").strip()
                if ts and ts.lower() not in ("nan", "-9999"):
                    T[ti, si] = float(ts)
                rs = parts[10].replace(",", ".").strip()
                if rs and rs.lower() not in ("nan", "-9999"):
                    RH[ti, si] = float(rs)
            except ValueError:
                continue
    T[(T < -60) | (T > 60)] = np.nan
    RH[(RH < 0) | (RH > 100)] = np.nan
    return T, RH


def loo(y_km, x_km, vals, elev, k, power, max_dist, lapse_per_m):
    """Her istasyon için kendisi hariç IDW tahmini (IDWInterpolator ile aynı formül)."""
    from scipy.spatial import cKDTree
    ok = ~np.isnan(vals)
    idx_ok = np.nonzero(ok)[0]
    est = np.full(len(vals), np.nan)
    nn = np.full(len(vals), np.nan)
    if len(idx_ok) < k + 2:
        return est, nn
    tree = cKDTree(np.c_[y_km[idx_ok], x_km[idx_ok]])
    d, j = tree.query(np.c_[y_km[idx_ok], x_km[idx_ok]], k=k + 1)
    self_mask = idx_ok[j] == idx_ok[:, None]
    # kendini çıkar: her satırda kendisi olan sütunu at, ilk k'yı tut
    order = np.argsort(self_mask, axis=1, kind="stable")
    d = np.take_along_axis(d, order, 1)[:, :k]
    j = np.take_along_axis(j, order, 1)[:, :k]
    src = idx_ok[j]
    v = vals[src].astype(np.float64)
    if lapse_per_m and elev is not None:
        zs = elev[src]
        zq = elev[idx_ok][:, None]
        good = ~np.isnan(zs) & ~np.isnan(zq)
        v = v + np.where(good, -lapse_per_m * (zq - zs), 0.0)
    w = 1.0 / np.maximum(d, 0.5) ** power
    w[d > max_dist] = 0.0
    exact = d[:, 0] < 0.5
    out = np.where(exact, v[:, 0], (w * v).sum(1) / np.maximum(w.sum(1), 1e-12))
    out[(w.sum(1) == 0) & ~exact] = np.nan
    est[idx_ok] = out
    nn[idx_ok] = d[:, 0]
    return est, nn


def stats(err):
    e = err[np.isfinite(err)]
    if len(e) == 0:
        return {"n": 0, "bias": np.nan, "MAE": np.nan, "RMSE": np.nan}
    return {"n": int(len(e)), "bias": float(e.mean()), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "p95_abs": float(np.percentile(np.abs(e), 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--awos_dir", default="data/raw_awos_observation")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--cache", default="data/processed/idw_loo_sample.npz")
    ap.add_argument("--out", default="artifacts_v3/analysis")
    ap.add_argument("--days_per_cell", type=int, default=4, help="ay × saat hücresi başına gün (9 × 8 × 4 = 288 an)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--max_dist_km", type=float, default=250.0)
    ap.add_argument("--lapse_rate", type=float, default=6.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from grid_utils import LatLonGrid
    from update_h5_aux_channels import parse_awos_stationlist

    A = parse_awos_stationlist(os.path.join(a.awos_dir, "stationlist_awos.csv"))
    ids = np.array(sorted(A))
    lat = np.array([A[s][0] for s in ids]); lon = np.array([A[s][1] for s in ids])
    elev = np.array([A[s][2] for s in ids], dtype=np.float64)
    grid = LatLonGrid.default_turkiye()
    y_km, x_km = grid.to_km(lat, lon)

    keys = sample_keys(a.days_per_cell, a.seed)
    if os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=False)
        if list(z["keys"]) == keys and np.array_equal(z["ids"], ids):
            T, RH = z["T"], z["RH"]
            print(f"önbellek: {a.cache}")
        else:
            raise SystemExit(f"Önbellek farklı örneklem içeriyor: {a.cache} (silin ya da --cache değiştirin)")
    else:
        path = os.path.join(a.awos_dir, "observation_awos.csv")
        print(f"AWOS taranıyor: {len(keys)} an × {len(ids)} istasyon", flush=True)
        T, RH = scan(path, keys, ids, os.path.getsize(path))
        np.savez_compressed(a.cache, T=T, RH=RH, keys=np.array(keys), ids=ids)

    lab = set(np.unique(np.load(a.meta)["wmoids"]).astype(int).tolist())
    is_lab = np.isin(ids, list(lab))
    lapse = a.lapse_rate / 1000.0
    recs = []
    for ti, key in enumerate(keys):
        h = int(key.split(";")[3])
        for var, vals, lp in (("t2m", T[ti], lapse), ("t2m_düzeltmesiz", T[ti], 0.0), ("rh2m", RH[ti], 0.0)):
            est, nn = loo(y_km, x_km, vals.astype(np.float64), elev if var == "t2m" else None, a.k, a.power, a.max_dist_km, lp)
            m = np.isfinite(est) & np.isfinite(vals)
            recs.append(pd.DataFrame({"var": var, "saat": h, "ay": "-".join(key.split(";")[:2]), "wmoid": ids[m],
                                      "etiket_ist": is_lab[m], "nn_km": nn[m], "err": est[m] - vals[m]}))
    D = pd.concat(recs, ignore_index=True)
    # en yakın komşuya yükseklik farkı (istasyon sabit)
    from scipy.spatial import cKDTree
    tr = cKDTree(np.c_[y_km, x_km]); _, j = tr.query(np.c_[y_km, x_km], k=2)
    dz = pd.Series(np.abs(elev - elev[j[:, 1]]), index=ids)
    D["dz_nn"] = D.wmoid.map(dz).values
    D["nn_grup"] = pd.cut(D.nn_km, [0, 5, 10, 20, 1e9], labels=["≤ 5 km", "5–10 km", "10–20 km", "> 20 km"]).astype(str)
    D["dz_grup"] = pd.cut(D.dz_nn, [-1, 100, 300, 600, 1e9], labels=["< 100 m", "100–300 m", "300–600 m", "≥ 600 m"]).astype(str)
    D["zaman"] = np.where(D.saat.isin([9, 12]), "gündüz (09–12 UTC)", np.where(D.saat.isin([18, 21, 0]), "gece (18–00 UTC)", "geçiş"))
    rows = []
    for var, g in D.groupby("var"):
        for sub_name, sub in (("tüm AWOS", g), ("etiket istasyonları", g[g.etiket_ist])):
            rows.append({"değişken": var, "hedef": sub_name, "grup_türü": "tümü", "grup": "tümü", **stats(sub.err.values),
                         "istasyon": int(sub.wmoid.nunique())})
            for gt in ("nn_grup", "dz_grup", "zaman", "ay"):
                for gv, gg in sub.groupby(gt):
                    rows.append({"değişken": var, "hedef": sub_name, "grup_türü": gt, "grup": gv, **stats(gg.err.values),
                                 "istasyon": int(gg.wmoid.nunique())})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "idw_loo.csv"), index=False, float_format="%.4f")
    info = {"n_an": len(keys), "n_awos": int(len(ids)), "k": a.k, "power": a.power, "max_dist_km": a.max_dist_km,
            "lapse_rate": a.lapse_rate, "days_per_cell": a.days_per_cell, "seed": a.seed,
            "nn_km_medyan_tüm": float(np.nanmedian(D[D["var"] == "t2m"].nn_km))}
    json.dump(info, open(os.path.join(a.out, "idw_loo_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.set_option("display.width", 200)
    print(R[R.grup_türü.isin(["tümü", "nn_grup", "dz_grup"])].to_string(index=False))


if __name__ == "__main__":
    main()
