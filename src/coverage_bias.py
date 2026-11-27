"""B1 (27 Kas, ön-kayıtlı, betimsel): FCI CLM ve NWC SAF CT kapsamı rastgele mi?

Test kümelerinde (tüm etiketler ve yalnız SYNOP) her ürün için kapsanan / kapsanmayan satırlar: kapsama oranı (ay, UTC saati,
aydınlanma, yükseklik bandı tabakalarında), bulutlu oranı ve CNN (B_full, 3 tohum) BalAcc'i; BalAcc(kapsanan) − BalAcc(kapsanmayan)
farkı gün blokları (test_station'da istasyon blokları) üzerinden bootstrap GA ile. Ürün kapsamı: data/processed/ref_rows_all.npz
(src/ref_rows_cache.py). Yorum kuralı docs/revizyon_onkayit.md §3 (27 Kas 14:30).
Çıktı: <out>/coverage_bias.csv (özet), coverage_bias_strata.csv (tabakalar)
    python src/coverage_bias.py --dir artifacts_v3 --out artifacts_v3/analysis
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import (BLOCK, SUBSETS, aligned, balacc_from, block_counts, boot_weights, by_exp, common_rows,  # noqa: E402
                           counts, load_meta, load_runs)
from ref_rows_cache import get_refs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--out", default="artifacts_v3/analysis")
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--model", default="B_full")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    meta = load_meta(a.meta, None, a.stations)
    meta["hour"] = pd.DatetimeIndex(meta.time).hour.astype(str).str.zfill(2)
    refs = get_refs(meta)
    runs = by_exp(load_runs(a.dir))[a.model]
    out, strata = [], []
    for s in SUBSETS:
        rows = common_rows(runs, s)
        m = meta.iloc[rows].reset_index(drop=True)
        YP = [aligned(r, s, rows) for r in runs if s in r["sets"]]
        y = YP[0][0]
        preds = [(p >= 0.5).astype(int) for _, p in YP]
        for lab, mask in (("tümü", np.ones(len(rows), bool)), ("SYNOP", m.src.values == "SYNOP")):
            codes, uni = pd.factorize(m[BLOCK[s]].values[mask])
            W = boot_weights(len(uni), a.boot, rng)
            for prod in ("CLM", "NWC"):
                cov = np.isfinite(refs[f"{prod}_r30"][rows])[mask]
                yy = y[mask]
                ba = {}
                for name, sel in (("kapsanan", cov), ("kapsanmayan", ~cov)):
                    if sel.sum() == 0:
                        ba[name] = (np.nan, np.full(a.boot, np.nan))
                        continue
                    pt = np.mean([balacc_from(counts(pr[mask][sel], yy[sel])) for pr in preds])
                    bs = np.mean([balacc_from(W @ block_counts(pr[mask][sel], yy[sel], codes[sel], len(uni))) for pr in preds], 0)
                    ba[name] = (pt, bs)
                d = ba["kapsanan"][1] - ba["kapsanmayan"][1]
                ok = np.isfinite(d)
                lo, hi = np.percentile(d[ok], [2.5, 97.5]) if ok.sum() > 100 else (np.nan, np.nan)
                out.append({"küme": s, "etiket": lab, "ürün": prod, "n": int(mask.sum()), "n_kapsanan": int(cov.sum()),
                            "kapsama": cov.mean(), "bulutlu_kapsanan": yy[cov].mean() if cov.any() else np.nan,
                            "bulutlu_kapsanmayan": yy[~cov].mean() if (~cov).any() else np.nan,
                            "model": a.model, "BalAcc_kapsanan": ba["kapsanan"][0], "BalAcc_kapsanmayan": ba["kapsanmayan"][0],
                            "fark": ba["kapsanan"][0] - ba["kapsanmayan"][0], "GA_alt": lo, "GA_üst": hi,
                            "blok": BLOCK[s], "blok_sayısı": len(uni)})
                mm = m[mask].assign(cov=cov, y=yy)
                for fac in ("month", "hour", "illum", "elev_bin"):
                    g = mm.groupby(fac).agg(n=("y", "size"), kapsama=("cov", "mean"), bulutlu=("y", "mean")).reset_index()
                    for _, r in g.iterrows():
                        strata.append({"küme": s, "etiket": lab, "ürün": prod, "etken": fac, "düzey": r[fac], "n": int(r["n"]),
                                       "kapsama": r["kapsama"], "bulutlu": r["bulutlu"]})
    D = pd.DataFrame(out)
    os.makedirs(a.out, exist_ok=True)
    D.to_csv(os.path.join(a.out, "coverage_bias.csv"), index=False, encoding="utf-8", float_format="%.6f")
    pd.DataFrame(strata).to_csv(os.path.join(a.out, "coverage_bias_strata.csv"), index=False, encoding="utf-8", float_format="%.6f")
    pd.set_option("display.width", 220)
    print(D.round(4).to_string())


if __name__ == "__main__":
    main()
