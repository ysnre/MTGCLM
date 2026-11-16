"""
references_by_period.py — P7: FCI CLM karşılaştırması, ürün dönemine göre (C = commissioning, Mar–Nis 2025; O = operasyonel,
Ara 2025–Haz 2026; dönem bayrağı CLM dosya adından) ve "cloud contaminated" (cloud_state 2) eşleme duyarlılığı.

Yalnız SYNOP etiketleri, CLM kapsamındaki örnekler (analyze_preds.py --refs ile aynı kurallar). Üç eşleme:
  cont_cloudy   : 2, 3 -> bulutlu; 1, 8 -> açık  (makaledeki varsayılan)
  cont_clear    : 3 -> bulutlu; 1, 2, 8 -> açık
  cont_excluded : merkezi 2 olan örnekler çıkarılır; 30 km oranı = dolu / (açık + dolu + kar)
30 km eşiği her eşleme için yalnız train örneklerinde (tüm dönemler) seçilir. Model: seed ortalaması (BalAcc noktası) ve
gün blokları (test_station: istasyon blokları) üzerinde eşleştirilmiş bootstrap farkı (%95 GA).
Kümeler: test_time, test_station, test_both ve bunların birleşimi (all_test; gün blokları).

    python src/references_by_period.py      -> artifacts_v3/analysis/references_by_period.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import (BLOCK, SUBSETS, aligned, balacc_from, best_threshold, block_counts, boot_weights,  # noqa: E402
                           by_exp, common_rows, counts, load_meta, load_runs)

R30 = 5          # extract_fci_clm GROUPS içindeki r30 sırası
MODELS = ("B_full", "D_unet")
VARIANTS = ("cont_cloudy", "cont_clear", "cont_excluded")


def load_clm_counts(keys, clm_dir):
    keys = keys.assign(day=keys.rc_start.str[:8])
    parts = []
    for day, g in keys.groupby("day"):
        p = os.path.join(clm_dir, f"clm_{day}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ri = {r: i for i, r in enumerate(z["rc_start"])}
        si = {int(w): i for i, w in enumerate(z["wmoid"])}
        fi, wi = g.rc_start.map(ri), g.wmoid.map(si)
        ok = fi.notna() & wi.notna()
        if not ok.any():
            continue
        g = g[ok]
        fi, wi = fi[ok].astype(int).values, wi[ok].astype(int).values
        okf = z["ok"][fi]
        c = z["counts"][fi, wi, R30].astype(np.int32)
        parts.append(pd.DataFrame({"row": g.index.values[okf], "disp": z["disposition"][fi][okf],
                                   "cstate": z["center_state"][fi, wi][okf],
                                   "n1": c[okf, 1], "n2": c[okf, 2], "n3": c[okf, 3], "n8": c[okf, 8]}))
    return pd.concat(parts).set_index("row")


def variant(df, v):
    """-> (merkez tahmini, r30 oranı, geçerli maske)"""
    cs = df.cstate.values
    valid = np.isin(cs, [1, 2, 3, 8])
    if v == "cont_cloudy":
        ctr = np.isin(cs, [2, 3])
        frac = (df.n2 + df.n3) / np.maximum(df.n1 + df.n2 + df.n3 + df.n8, 1)
    elif v == "cont_clear":
        ctr = cs == 3
        frac = df.n3 / np.maximum(df.n1 + df.n2 + df.n3 + df.n8, 1)
    else:
        valid &= cs != 2
        ctr = cs == 3
        frac = df.n3 / np.maximum(df.n1 + df.n3 + df.n8, 1)
    return ctr.astype(int), frac.values, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--split", default="splits/split_v2_verify.npz")
    ap.add_argument("--clm_dir", default="data/processed/clm_station")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts_v3/analysis/references_by_period.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    meta = load_meta(a.meta, None, a.stations)
    meta["rc_start"] = pd.DatetimeIndex(meta.time).floor("10min").strftime("%Y%m%d%H%M")
    split = np.load(a.split)
    E = {e: v for e, v in by_exp(load_runs(a.dir)).items() if e in MODELS}
    tr = meta.iloc[split["train"]]
    tr = tr[tr.label >= 0]
    ctr = load_clm_counts(tr[["rc_start", "wmoid"]], a.clm_dir).join(tr[["label"]], how="inner")
    thr = {}
    for v in VARIANTS:
        _, frac, valid = variant(ctr, v)
        ba, th = best_threshold(frac[valid], ctr.label.values[valid])
        thr[v] = th
        print(f"  {v}: r30 eşiği {th} (train BalAcc {ba:.3f}, n {valid.sum()})")

    sets = {s: common_rows([r for e in MODELS for r in E[e]], s) for s in SUBSETS}
    sets["all_test"] = np.unique(np.concatenate([x for x in sets.values() if x is not None]))
    clm_te = load_clm_counts(meta.iloc[sets["all_test"]][["rc_start", "wmoid"]], a.clm_dir)

    def model_preds(e, s, rows):
        """seed başına tahmin [n] (rows sırasında); all_test için kümelerden birleştirilir."""
        out = []
        for r in E[e]:
            p = np.full(len(rows), np.nan)
            for ss in SUBSETS:
                if ss not in r["sets"] or (s != "all_test" and ss != s):
                    continue
                rr = r["sets"][ss][0]
                m = np.isin(rows, rr)
                if m.any():
                    _, pp = aligned(r, ss, rows[m])
                    p[m] = pp
            out.append(p)
        return out

    res = []
    for s, rows in sets.items():
        if rows is None:
            continue
        M = meta.iloc[rows]
        syn = (M.src.values == "SYNOP") & (M.label.values >= 0)
        C = clm_te.reindex(rows)
        has = C.cstate.notna().values
        block = BLOCK.get(s, "day")
        preds = {e: model_preds(e, s, rows) for e in MODELS}
        for v in VARIANTS:
            sub = C[has]
            ctr_v, frac_v, val_v = variant(sub.astype({"cstate": int}), v)
            cpred = np.full(len(rows), -1)
            rpred = np.full(len(rows), -1)
            vmask = np.zeros(len(rows), bool)
            cpred[has], rpred[has] = ctr_v, (frac_v >= thr[v]).astype(int)
            vmask[has] = val_v
            disp = np.where(has, C.disp.astype(str).values, "")
            for per in ("C", "O", "C+O"):
                m = syn & vmask & ((disp == per) if per != "C+O" else np.isin(disp, ["C", "O"]))
                if m.sum() < 200:
                    continue
                y = M.label.values[m]
                codes, uni = pd.factorize(M[block].values[m])
                W = boot_weights(len(uni), a.boot, rng)
                row = {"küme": s, "dönem": per, "eşleme": v, "n": int(m.sum()), "n_blok": len(uni), "r30_eşik": thr[v]}
                for name, rp in (("CLM_merkez", cpred[m]), ("CLM_r30", rpred[m])):
                    rb = balacc_from(W @ block_counts(rp, y, codes, len(uni)))
                    row[name] = float(balacc_from(counts(rp, y)))
                    for e in MODELS:
                        pts, bts = [], []
                        for p in preds[e]:
                            pr = (p[m] >= 0.5).astype(int)
                            pts.append(float(balacc_from(counts(pr, y))))
                            bts.append(balacc_from(W @ block_counts(pr, y, codes, len(uni))))
                        d = np.mean(bts, axis=0) - rb
                        row[e] = float(np.mean(pts))
                        row[f"{e}−{name}"] = float(np.mean(pts) - row[name])
                        row[f"{e}−{name}_GA"] = f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
                res.append(row)
    R = pd.DataFrame(res)
    R.to_csv(a.out, index=False, encoding="utf-8")
    pd.set_option("display.width", 250)
    cols = ["küme", "dönem", "eşleme", "n", "CLM_merkez", "CLM_r30", "B_full", "B_full−CLM_r30", "B_full−CLM_r30_GA"]
    print(R[cols].round(4).to_string(index=False))
    print(f"Yazıldı: {a.out}")


if __name__ == "__main__":
    main()
