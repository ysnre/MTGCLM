#!/usr/bin/env python
"""R7 — kar ve uydu görüş açısı (VZA) kırılımı (ön kayıt: docs/revizyon_onkayit.md, R7).

Ürünler: B_full, D_unet, D_unet_jit (tohum başına tahmin, metrik tohum ortalaması) ve FCI CLM / NWC SAF CT (merkez, r30; eşikler
train'de, analyze_preds.reference_preds ile aynı). Satırlar: verify test kümeleri (test_time, test_station, test_both) ve
birleşimleri ("tüm test"). Referanslarla kıyas her hücrede iki ürünün de kapsadığı ORTAK satırlarda yapılır.

Kar:
  kar_imzası_açık      : gündüz, açık (0 okta), merkez pikselde VIS0.6 ≥ %30 ve NIR1.6 ≤ 0,4 × VIS0.6 (select_scenes.py /
                         error_analysis.py ile aynı ölçüt) → yalnız yanlış alarm oranı (POFD; açıkların bulutlu tahmin edilen payı)
  kış_gündüz_yüksek    : Ara–Mar, gündüz, istasyon > 1500 m (iki sınıf) → BalAcc, FAR, POFD
  kıyas hücreleri       : kış gündüz ≤ 1500 m; Nis–Haz gündüz > 1500 m
VZA: istasyon VZA'sının (MTG-I1 0°, küresel Dünya; parallax_shift.geometry) üçte birlik dilimleri; BalAcc, FAR, POFD. Betimsel
(VZA boylamla karışık).
Kural: n < 300 olan hücrede "fark yok" denmez, "belirsiz" yazılır (sütun `belirsiz`).
Belirsizlik: gün blokları (test_station tek başına: istasyon blokları), B = 2000; model − referans farkı için %95 GA.

Çıktı: <out>/snow_vza.csv (uzun biçim), <out>/snow_vza_diff.csv (model − referans, GA),
       <out>/snow_vza_stations.csv (kış gündüz > 1500 m açık raporlar, istasyon başına yanlış alarm oranı)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import block_counts, boot_weights, by_exp, counts, load_meta, load_runs, reference_preds  # noqa: E402
from revision_metrics import preds_of, scores, seed_probs  # noqa: E402

SETS = ("test_time", "test_station", "test_both")
MODELS = ("B_full", "D_unet", "D_unet_jit")
REFS = ("CLM_merkez", "CLM_r30", "NWC_merkez", "NWC_r30")
MIN_N = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--split", default="splits/split_v2_verify.npz")
    ap.add_argument("--centre", default="data/processed/center_values.npz")
    ap.add_argument("--clm_dir", default="data/processed/clm_station")
    ap.add_argument("--nwc_dir", default="data/processed/nwcsaf_station")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    from parallax_shift import geometry

    meta = load_meta(a.meta, None, a.stations)
    sp = np.load(a.split)
    E = by_exp(load_runs(a.dir))
    P = seed_probs(E, MODELS, a.dir)
    print("[1/3] referanslar ...", flush=True)
    refs, _ = reference_preds(meta, sp, a.clm_dir, a.nwc_dir)

    z = np.load(a.centre)
    bn = [str(n) for n in z["base_names"]]
    vis = z["base"][:, bn.index("vis_06")].astype(float)
    nir = z["base"][:, bn.index("nir_16")].astype(float)
    sig = (vis >= 30) & (nir <= 0.4 * vis)
    vza, _ = geometry(meta.lat.values, meta.lon.values)
    st_vza = pd.Series(vza, index=meta.wmoid.values).groupby(level=0).first()
    q = np.quantile(st_vza.values, [1 / 3, 2 / 3])
    vza_bin = np.where(vza < q[0], f"VZA < {q[0]:.1f}°", np.where(vza < q[1], f"VZA {q[0]:.1f}–{q[1]:.1f}°", f"VZA ≥ {q[1]:.1f}°"))
    mon = pd.DatetimeIndex(meta.time).month.values
    day = meta.illum.values == "gündüz"
    winter = np.isin(mon, [12, 1, 2, 3]); spring = np.isin(mon, [4, 5, 6])
    hi = meta.elev.values > 1500
    clear = meta.label.values == 0
    cells = {
        "kar_imzası_açık": day & clear & sig,
        "kış_gündüz_açık_imzasız": day & clear & ~sig & winter,
        "kış_gündüz_yüksek (>1500 m)": day & winter & hi,
        "kış_gündüz_alçak (≤1500 m)": day & winter & ~hi,
        "ilkbahar-yaz_gündüz_yüksek (>1500 m)": day & spring & hi,
    }
    for b in sorted(set(vza_bin)):
        cells[b] = vza_bin == b

    groups = [(s, sp[s]) for s in SETS] + [("tüm test", np.unique(np.concatenate([sp[s] for s in SETS])))]
    rows, drows = [], []
    print("[2/3] hücreler ...", flush=True)
    for gname, grows in groups:
        grows = np.sort(grows)
        grows = grows[meta.label.values[grows] >= 0]
        blk = "wmoid" if gname == "test_station" else "day"
        for cname, cm in cells.items():
            r = grows[cm[grows]]
            if len(r) == 0:
                continue
            y = meta.label.values[r]
            codes, uni = pd.factorize(meta[blk].values[r])
            W = boot_weights(len(uni), a.boot, rng)
            only_clear = (y == 1).sum() == 0
            boot = {}
            for name in MODELS + REFS:
                if name in MODELS:
                    pl = preds_of(P, name, r) if name in P else None
                    if pl is None:
                        continue
                    m = np.ones(len(r), bool)
                else:
                    v = refs[name][r]
                    m = np.isfinite(v)
                    pl = [np.nan_to_num(v).astype(int)]
                # ürünün kendi kapsamı (modeller her yerde)
                S = [scores(counts(p[m], y[m])) for p in pl]
                row = {"küme": gname, "hücre": cname, "ürün": name, "n": int(m.sum()), "n_bulutlu": int((y[m] == 1).sum()),
                       "n_açık": int((y[m] == 0).sum()), "belirsiz": bool(m.sum() < MIN_N)}
                for k in ("POFD",) if only_clear else ("BalAcc", "POD", "FAR", "POFD"):
                    row[k] = float(np.nanmean([s[k] for s in S]))
                rows.append(row)
                boot[name] = (pl, m)
            # model − referans, ortak kapsamda
            key = "POFD" if only_clear else "BalAcc"
            for mo in MODELS:
                if mo not in boot:
                    continue
                for rf in REFS:
                    if rf not in boot:
                        continue
                    m = boot[rf][1]
                    if m.sum() < 30:
                        continue
                    cm_ = codes[m]
                    u2, c2 = np.unique(cm_, return_inverse=True)
                    W2 = W[:, u2]
                    def bstat(pl):
                        return np.mean([scores(W2 @ block_counts(p[m], y[m], c2, len(u2)))[key] for p in pl], axis=0)
                    def pstat(pl):
                        return float(np.mean([scores(counts(p[m], y[m]))[key] for p in pl]))
                    d = bstat(boot[mo][0]) - bstat(boot[rf][0])
                    lo, hi_ = np.nanpercentile(d, [2.5, 97.5])
                    drows.append({"küme": gname, "hücre": cname, "ölçüt": key, "model": mo, "referans": rf, "n_ortak": int(m.sum()),
                                  "model_değer": pstat(boot[mo][0]), "ref_değer": pstat(boot[rf][0]),
                                  "fark": pstat(boot[mo][0]) - pstat(boot[rf][0]), "GA_alt": lo, "GA_üst": hi_,
                                  "GA_sifiri_disliyor": bool(lo > 0 or hi_ < 0), "belirsiz": bool(m.sum() < MIN_N)})
        print(f"      {gname} tamam", flush=True)
    # istasyon başına: kış gündüz > 1500 m açık raporlar (tüm test) — hücrenin birkaç istasyona dayanıp dayanmadığını göstermek için
    allr = groups[-1][1]
    sel = allr[cells["kış_gündüz_yüksek (>1500 m)"][allr] & clear[allr]]
    srows = []
    for w in np.unique(meta.wmoid.values[sel]):
        s_ = np.sort(sel[meta.wmoid.values[sel] == w])
        row = {"wmoid": int(w), "n_açık": int(len(s_)), "elev": float(meta.elev.values[s_][0]), "lon": float(meta.lon.values[s_][0]),
               "lat": float(meta.lat.values[s_][0])}
        for mo in MODELS:
            pl = preds_of(P, mo, s_) if mo in P else None
            row[f"{mo}_POFD"] = float(np.mean([p.mean() for p in pl])) if pl else np.nan
        for rf in REFS:
            v = refs[rf][s_]; v = v[np.isfinite(v)]
            row[f"{rf}_POFD"] = float(v.mean()) if len(v) else np.nan
            row[f"{rf}_n"] = int(len(v))
        srows.append(row)
    pd.DataFrame(srows).to_csv(os.path.join(out, "snow_vza_stations.csv"), index=False, float_format="%.4f")

    R, D = pd.DataFrame(rows), pd.DataFrame(drows)
    R.to_csv(os.path.join(out, "snow_vza.csv"), index=False, float_format="%.5f")
    D.to_csv(os.path.join(out, "snow_vza_diff.csv"), index=False, float_format="%.5f")
    print("[3/3] özet (tüm test):", flush=True)
    pd.set_option("display.width", 220)
    t = R[(R.küme == "tüm test") & R.ürün.isin(["B_full", "CLM_r30", "NWC_r30"])]
    print(t.to_string(index=False))
    print(D[(D.küme == "tüm test") & (D.model == "B_full") & D.referans.isin(["CLM_r30", "NWC_r30"])].to_string(index=False))
    print(f"VZA dilim sınırları (istasyon): {q.round(1)}; istasyon VZA {st_vza.min():.1f}–{st_vza.max():.1f}°")


if __name__ == "__main__":
    main()
