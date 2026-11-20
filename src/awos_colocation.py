#!/usr/bin/env python
"""R9 — holdout istasyonları ↔ AWOS ko-lokasyonu ve okta-9 / RH bağlaşımı (ön kayıt: docs/revizyon_onkayit.md, R9).

1) Her etiket istasyonu için: kendi AWOS kaydı var mı (aynı wmoid, sıcaklık kaydı sayısı), en yakın AWOS mesafesi
   (kendisi dahil) ve kendisi hariç en yakın AWOS mesafesi. IDW (k = 8, p = 2) sıfır mesafede istasyonun kendi ölçümünü
   verdiği için ko-lokasyonlu istasyonda merkez pikselin t2m / rh2m değeri pratikte istasyonun kendi termometresidir.
2) Ön-kayıtlı kıyas: B_full − C_noawos (ve C_t2m − C_noawos) test_station BalAcc, ko-lokasyonlu (≤ 1 km) ve olmayan holdout
   grupları ayrı ayrı (istasyon blokları). Bir grup boşsa satır "grup boş" olarak yazılır. Not: istasyon listesi ile AWOS
   listesindeki koordinatlar aynı istasyon için 1–3 km farklı olabilir; bu yüzden "kendi AWOS'u var (aynı wmoid)" grubu da verilir. Betimsel ek: kendisi hariç en yakın
   AWOS mesafesine göre gruplar (≤ 5 km / > 5 km).
3) Tam harita bağlamı: Türkiye kara piksellerinde en yakın AWOS mesafesi dağılımı (yüzdelikler).
4) okta-9 / RH: okta 9 etiketi istasyon RH'sine bağlı (RH ≥ %90 → bulutlu). Test kümelerinde okta-9 satır sayısı ve
   bu satırlarda B_full / C_noawos doğruluğu.

Çıktı: <out>/awos_colocation_stations.csv, awos_colocation.csv, awos_pixel_distance.csv, okta9_rh.csv
AWOS sayımları: data/processed/awos_station_counts.csv (yoksa observation_awos.csv parça parça taranıp yazılır).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import (aligned, balacc_from, block_counts, boot_weights, by_exp, common_rows,  # noqa: E402
                           counts, load_meta, load_runs)

R_EARTH = 6371.0
PAIRS = [("B_full", "C_noawos"), ("C_t2m", "C_noawos"), ("B_full", "A_sat")]


def hav_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def awos_counts(path_counts, path_obs):
    if os.path.exists(path_counts):
        c = pd.read_csv(path_counts, sep=";", header=None, names=["wmoid", "n_t"])
        return dict(zip(c.wmoid.astype(int), c.n_t.astype(int)))
    print(f"AWOS sayımları üretiliyor ({path_obs}, birkaç dk) ...", flush=True)
    tot = {}
    for i, ch in enumerate(pd.read_csv(path_obs, sep=";", usecols=[0, 7], header=0, names=["w", "t"],
                                       chunksize=2_000_000, encoding="utf-8", encoding_errors="replace")):
        v = ch.dropna(subset=["t"]).w.value_counts()
        for k, n in v.items():
            tot[int(k)] = tot.get(int(k), 0) + int(n)
        print(f"  parça {i + 1}", flush=True)
    pd.Series(tot).to_csv(path_counts, sep=";", header=False)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--awos_list", default="data/raw_awos_observation/stationlist_awos.csv")
    ap.add_argument("--awos_obs", default="data/raw_awos_observation/observation_awos.csv")
    ap.add_argument("--awos_counts", default="data/processed/awos_station_counts.csv")
    ap.add_argument("--min_frac", type=float, default=0.5, help="Kendi AWOS'u 'var' sayılması için kayıt sayısı / medyan")
    ap.add_argument("--coloc_km", type=float, default=1.0)
    ap.add_argument("--split", default="splits/split_v2_verify.npz")
    ap.add_argument("--regions", default="artifacts_v3/climatology/regions.npz")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    from update_h5_aux_channels import parse_awos_stationlist

    meta = load_meta(a.meta, None, a.stations)
    split = np.load(a.split)
    A = parse_awos_stationlist(a.awos_list)
    cnt = awos_counts(a.awos_counts, a.awos_obs)
    med = np.median(list(cnt.values()))
    aw = pd.DataFrame([(w, la, lo, cnt.get(w, 0)) for w, (la, lo, _) in A.items()], columns=["wmoid", "lat", "lon", "n_t"])
    aw = aw[aw.n_t >= a.min_frac * med]
    print(f"AWOS: listede {len(A)}, sıcaklık kaydı yeterli {len(aw)} (eşik {a.min_frac:.1f} × medyan {med:.0f})")

    # ---- 1) istasyon tablosu
    ho = set(np.unique(meta.wmoid.values[split["test_station"]]).tolist())
    st = meta.drop_duplicates("wmoid")[["wmoid", "lat", "lon", "elev"]].reset_index(drop=True)
    rows = []
    for r in st.itertuples():
        d = hav_km(r.lat, r.lon, aw.lat.values, aw.lon.values)
        own = aw.wmoid.values == r.wmoid
        rows.append({"wmoid": r.wmoid, "holdout": r.wmoid in ho, "kendi_AWOS": bool(own.any()),
                     "kendi_AWOS_kayıt": int(cnt.get(r.wmoid, 0)), "en_yakın_km": float(d.min()),
                     "kendi_hariç_en_yakın_km": float(d[~own].min()),
                     "5km_içi_AWOS": int((d[~own] <= 5).sum()), "10km_içi_AWOS": int((d[~own] <= 10).sum())})
    S = pd.DataFrame(rows)
    S["ko_lokasyon"] = S.en_yakın_km <= a.coloc_km
    S.to_csv(os.path.join(out, "awos_colocation_stations.csv"), index=False, float_format="%.3f")
    for g, sub in S.groupby("holdout"):
        print(f"  {'holdout' if g else 'eğitim'} istasyonları: {len(sub)}; ko-lokasyonlu {int(sub.ko_lokasyon.sum())}; "
              f"kendi hariç en yakın AWOS medyan {sub.kendi_hariç_en_yakın_km.median():.1f} km "
              f"[min {sub.kendi_hariç_en_yakın_km.min():.1f}, maks {sub.kendi_hariç_en_yakın_km.max():.1f}]")

    # ---- 2) eşleştirilmiş farklar, test_station, gruplar
    E = by_exp(load_runs(a.dir))
    s = "test_station"
    smap = S.set_index("wmoid")
    out_rows = []
    groupings = {
        "ko-lokasyon (≤ 1 km)": lambda w: smap.ko_lokasyon.reindex(w).fillna(False).values,
        "ko-lokasyon yok (> 1 km)": lambda w: ~smap.ko_lokasyon.reindex(w).fillna(False).values,
        "kendi hariç en yakın ≤ 5 km": lambda w: (smap.kendi_hariç_en_yakın_km.reindex(w) <= 5).values,
        "kendi hariç en yakın > 5 km": lambda w: (smap.kendi_hariç_en_yakın_km.reindex(w) > 5).values,
        "kendi AWOS'u var (aynı wmoid)": lambda w: smap.kendi_AWOS.reindex(w).fillna(False).values.astype(bool),
        "kendi AWOS'u yok": lambda w: ~smap.kendi_AWOS.reindex(w).fillna(False).values.astype(bool),
        "tümü": lambda w: np.ones(len(w), bool),
    }
    for lab in ("tümü", "yalnız SYNOP"):
        for gname, gf in groupings.items():
            for e1, e2 in PAIRS:
                if e1 not in E or e2 not in E:
                    continue
                rr = common_rows(E[e1] + E[e2], s)
                w = meta.wmoid.values[rr]
                m = gf(w)
                if lab == "yalnız SYNOP":
                    m &= meta.src.values[rr] == "SYNOP"
                row = {"etiket": lab, "grup": gname, "çift": f"{e1} − {e2}", "n": int(m.sum()),
                       "istasyon": int(len(np.unique(w[m])))}
                if row["istasyon"] < 3:
                    row["not"] = "grup boş" if row["istasyon"] == 0 else "çok az istasyon"
                    out_rows.append(row)
                    continue
                y = meta.label.values[rr][m]
                codes, uni = pd.factorize(w[m])
                W = boot_weights(len(uni), a.boot, rng)
                res = {}
                for e in (e1, e2):
                    pts, bts = [], []
                    for run in E[e]:
                        if s not in run["sets"]:
                            continue
                        _, p = aligned(run, s, rr)
                        pred = (p[m] >= 0.5).astype(int)
                        pts.append(float(balacc_from(counts(pred, y))))
                        bts.append(balacc_from(W @ block_counts(pred, y, codes, len(uni))))
                    res[e] = (np.mean(pts), np.mean(bts, axis=0))
                d = res[e1][1] - res[e2][1]
                lo, hi = np.percentile(d, [2.5, 97.5])
                row.update({e1 + "_BalAcc": res[e1][0], e2 + "_BalAcc": res[e2][0], "fark": res[e1][0] - res[e2][0],
                            "GA": f"[{lo:+.4f}, {hi:+.4f}]", "GA_sifiri_disliyor": bool(lo > 0 or hi < 0)})
                out_rows.append(row)
    P = pd.DataFrame(out_rows)
    P.to_csv(os.path.join(out, "awos_colocation.csv"), index=False, float_format="%.6f")
    print(P[P.çift == "B_full − C_noawos"].drop(columns=[c for c in P.columns if c.endswith("_BalAcc")]).to_string(index=False))

    # ---- 3) kara piksellerinde en yakın AWOS mesafesi
    try:
        from scipy.spatial import cKDTree
        lat = 42.995 - 0.01 * np.arange(800)
        lon = 25.01 + 0.02 * np.arange(1000)
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        reg = np.load(a.regions)
        key = [k for k in reg.files if "turk" in k.lower() or k.lower() in ("tr", "mask", "turkey")]
        mask = reg[key[0]].astype(bool) if key else None
        def xyz(la, lo):
            la, lo = np.radians(la), np.radians(lo)
            return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
        tree = cKDTree(xyz(aw.lat.values, aw.lon.values))
        sel = mask if mask is not None else np.ones(LA.shape, bool)
        dch, _ = tree.query(xyz(LA[sel], LO[sel]))
        dkm = 2 * R_EARTH * np.arcsin(np.clip(dch / 2, 0, 1))
        q = [5, 25, 50, 75, 95, 99]
        D = pd.DataFrame({"yüzdelik": q, "km": np.percentile(dkm, q)})
        D["maske"] = key[0] if key else "ızgara"
        D["pay_≤1km"] = float((dkm <= 1).mean()); D["pay_≤5km"] = float((dkm <= 5).mean())
        D.to_csv(os.path.join(out, "awos_pixel_distance.csv"), index=False, float_format="%.4f")
        print(f"  piksel → en yakın AWOS ({D.maske[0]}): medyan {np.median(dkm):.1f} km, %95 {np.percentile(dkm, 95):.1f} km, "
              f"≤ 1 km payı {(dkm <= 1).mean():.3f}")
    except Exception as ex:  # scipy ya da maske yoksa bu bölüm atlanır
        print(f"  piksel mesafesi atlandı: {ex}")

    # ---- 4) okta 9 / RH
    orow = []
    for s2 in ("test_time", "test_station", "test_both"):
        if "B_full" not in E or "C_noawos" not in E:
            break
        rr = common_rows(E["B_full"] + E["C_noawos"], s2)
        ok9 = meta.okta.values[rr] == 9
        lab = meta.label.values[rr]
        r = {"küme": s2, "n_küme": int(len(rr)), "n_okta9": int(ok9.sum()), "pay_okta9": float(ok9.mean())}
        for e in ("B_full", "C_noawos"):
            P_ = np.mean([aligned(run, s2, rr)[1] for run in E[e] if s2 in run["sets"]], axis=0)
            r[f"{e}_doğru_okta9"] = float(((P_[ok9] >= 0.5) == (lab[ok9] == 1)).mean()) if ok9.any() else np.nan
        orow.append(r)
    O = pd.DataFrame(orow)
    O.to_csv(os.path.join(out, "okta9_rh.csv"), index=False, float_format="%.5f")
    print(O.to_string(index=False))


if __name__ == "__main__":
    main()
