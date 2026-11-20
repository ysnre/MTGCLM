#!/usr/bin/env python
"""R11 — gözlemci belirsizlik tavanı (ön kayıt: docs/revizyon_onkayit.md, R11).

1) SYNOP–METAR çifti: aynı havalimanında SYNOP (HH:00) ile 10 dk önceki METAR (HH-1:50). İki raporun ikili sınıfları
   (açık = 0 okta / NSC-CAVOK, bulutlu = ≥ 5 okta) karşılaştırılır. METAR'ı SYNOP etiketinin "tahmini" sayarak
   BalAcc / POD / FAR → gözlemciler arası uyumdan ima edilen tavan. Ayrıca 3 sınıflı (0 / 1–4 / ≥5) karışıklık.
   Tavan İYİMSER kabul edilir: iki rapor çoğu zaman aynı gözlemciden / aynı nöbetten gelir. Öte yandan NSC ≠ 0 okta
   (önemli bulut yok) olduğundan METAR açıkların bir kısmı tanım farkıdır; bu tavanı aşağı çeker. İkisi birlikte yazılır.
2) Aynı çift satırlarında (test kümeleri) modelin SYNOP'a karşı BalAcc'ı, METAR–SYNOP uyumuyla yan yana.
3) Okta başına model hatası (yalnız SYNOP, test kümeleri): okta 0 yanlış alarm; okta 5, 6, 7, 8 kaçırma.

Belirsizlik: gün blokları (test_station için istasyon), B = 2000.
Çıktı: <out>/observer_ceiling.csv, <out>/observer_ceiling_okta.csv, <out>/observer_ceiling_confusion.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import (BLOCK, SUBSETS, aligned, balacc_from, block_counts, boot_weights, by_exp,  # noqa: E402
                           common_rows, counts, load_meta, load_runs)

MODELS = ["B_full", "D_unet"]


def cls3(okta):
    o = np.asarray(okta)
    return np.select([o == 0, (o >= 1) & (o <= 4), (o >= 5) & (o <= 9)], [0, 1, 2], -1)


def contingency(pred, y):
    c = counts(pred, y)        # [[tn, fp], [fn, tp]] varsayımı yerine açık hesap
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"BalAcc": float(balacc_from(c)), "POD": tp / max(tp + fn, 1), "FAR": fp / max(tp + fp, 1),
            "POFD": fp / max(fp + tn, 1), "n": int(len(y)), "n_bulutlu": tp + fn, "n_açık": fp + tn}


def gi(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"[{lo:.4f}, {hi:.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--split", default="splits/split_v2_verify.npz")
    ap.add_argument("--lag_min", type=int, default=10, help="METAR'ın SYNOP'tan kaç dk önce olduğu (varsayılan :50)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    meta = load_meta(a.meta, None, a.stations)
    z = np.load(a.meta)
    meta["obs"] = pd.to_datetime(z["obs_time"], unit="s")
    meta["row"] = np.arange(len(meta))
    split = np.load(a.split)
    setof = np.full(len(meta), "", dtype=object)
    for s in ["train", "val"] + SUBSETS:
        if s in split.files:
            setof[split[s]] = s
    meta["küme"] = setof

    # ---- 1) çiftler: aynı (istasyon, gözlem zamanı) tekrarları tek satıra indirgenir
    obs = meta.drop_duplicates(["wmoid", "obs"])
    syn = obs[obs.src == "SYNOP"]
    met = obs[obs.src == "METAR"][["wmoid", "obs", "okta", "row"]]
    met = met.assign(obs=met.obs + pd.Timedelta(minutes=a.lag_min)).rename(columns={"okta": "okta_m", "row": "row_m"})
    P = syn.merge(met, on=["wmoid", "obs"], how="inner")
    P["c_s"], P["c_m"] = cls3(P.okta.values), cls3(P.okta_m.values)
    P = P[(P.c_s >= 0) & (P.c_m >= 0)]
    print(f"SYNOP–METAR çifti: {len(P):,} ({P.wmoid.nunique()} havalimanı); SYNOP satırının kümesi: "
          f"{P.küme.replace('', 'bölme dışı').value_counts().to_dict()}")
    conf = pd.crosstab(P.c_s.map({0: "S:0", 1: "S:1-4", 2: "S:>=5"}), P.c_m.map({0: "M:0/NSC", 1: "M:1-4", 2: "M:>=5"}))
    conf.to_csv(os.path.join(out, "observer_ceiling_confusion.csv"))
    print(conf)

    rows = []
    B2 = P[(P.c_s != 1) & (P.c_m != 1)]
    for lab, sub in [("tüm çiftler", B2), ("yalnız test kümeleri", B2[B2.küme.isin(SUBSETS)])]:
        y = (sub.c_s.values == 2).astype(int); pm = (sub.c_m.values == 2).astype(int)
        codes, uni = pd.factorize(sub.day.values)
        W = boot_weights(len(uni), a.boot, rng)
        r = {"analiz": "METAR→SYNOP ikili uyum", "alt_küme": lab, **contingency(pm, y),
             "BalAcc_GA": gi(balacc_from(W @ block_counts(pm, y, codes, len(uni)))),
             "havalimanı": int(sub.wmoid.nunique()),
             "SYNOP_açık_METAR_bulutlu": int(((y == 0) & (pm == 1)).sum()),
             "SYNOP_bulutlu_METAR_açık": int(((y == 1) & (pm == 0)).sum())}
        # 1–4 okta'yı ara sınıf olarak dahil etmeyen, ama bir rapor ikili, diğeri 1–4 olan çiftlerin payı
        r["bir_rapor_1_4_payı"] = float(((P.c_s == 1) ^ (P.c_m == 1)).sum() / max(((P.c_s != 1) | (P.c_m != 1)).sum(), 1))
        rows.append(r)
        print(f"  {lab}: BalAcc {r['BalAcc']:.4f} {r['BalAcc_GA']}, POD {r['POD']:.3f}, FAR {r['FAR']:.3f}, n {r['n']:,}")

    # ---- 2) aynı çift satırlarında model (SYNOP etiketi, test kümeleri)
    E = by_exp(load_runs(a.dir))
    for s in SUBSETS:
        sub = B2[B2.küme == s]
        if len(sub) < 200 or any(e not in E for e in MODELS):
            continue
        rr = common_rows([r for e in MODELS for r in E[e]], s)
        sub = sub[np.isin(sub.row.values, rr)].sort_values("row")
        y = (sub.c_s.values == 2).astype(int); pm = (sub.c_m.values == 2).astype(int)
        codes, uni = pd.factorize(meta[BLOCK[s]].values[sub.row.values])
        W = boot_weights(len(uni), a.boot, rng)
        bm = balacc_from(W @ block_counts(pm, y, codes, len(uni)))
        r = {"analiz": "aynı çiftlerde model vs METAR", "alt_küme": s, "METAR_BalAcc": float(balacc_from(counts(pm, y))),
             "METAR_BalAcc_GA": gi(bm), "n": int(len(sub)), "havalimanı": int(sub.wmoid.nunique())}
        for e in MODELS:
            pts, bts = [], []
            for run in E[e]:
                if s not in run["sets"]:
                    continue
                _, p = aligned(run, s, sub.row.values)
                pred = (p >= 0.5).astype(int)
                pts.append(float(balacc_from(counts(pred, y))))
                bts.append(balacc_from(W @ block_counts(pred, y, codes, len(uni))))
            d = np.mean(bts, axis=0) - bm
            r[f"{e}_BalAcc"] = float(np.mean(pts))
            r[f"{e}_eksi_METAR"] = float(np.mean(pts)) - r["METAR_BalAcc"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            r[f"{e}_eksi_METAR_GA"] = f"[{lo:+.4f}, {hi:+.4f}]"
        rows.append(r)
        print(f"  {s} (n {len(sub):,}): METAR {r['METAR_BalAcc']:.4f}, B_full {r['B_full_BalAcc']:.4f} "
              f"({r['B_full_eksi_METAR']:+.4f} {r['B_full_eksi_METAR_GA']})")
    pd.DataFrame(rows).to_csv(os.path.join(out, "observer_ceiling.csv"), index=False, float_format="%.6f")

    # ---- 3) okta başına model hatası (yalnız SYNOP)
    orow = []
    for s in SUBSETS:
        if "B_full" not in E:
            break
        rr = common_rows(E["B_full"], s)
        m = meta.src.values[rr] == "SYNOP"
        rr = rr[m]
        ok = meta.okta.values[rr]
        P_ = np.mean([aligned(run, s, rr)[1] for run in E["B_full"] if s in run["sets"]], axis=0)
        for o in [0, 5, 6, 7, 8, 9]:
            k = ok == o
            if k.sum() == 0:
                continue
            err = (P_[k] >= 0.5) if o == 0 else (P_[k] < 0.5)
            orow.append({"küme": s, "okta": o, "n": int(k.sum()), "hata_türü": "yanlış alarm" if o == 0 else "kaçırma",
                         "hata_oranı": float(err.mean()), "ort_olasılık": float(P_[k].mean())})
    O = pd.DataFrame(orow)
    O.to_csv(os.path.join(out, "observer_ceiling_okta.csv"), index=False, float_format="%.6f")
    print(O[O.küme == "test_time"].to_string(index=False))


if __name__ == "__main__":
    main()
