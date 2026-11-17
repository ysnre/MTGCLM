"""
cv_weeks.py — Bloklu CV'de hangi haftalar/aylar skoru düşürüyor (K12 açık işi). CNN, tüm girdiler (CV_B_f*_s1) varsayılan.
Her örnek, kendi katının modeliyle tahmin edilmiştir (havuzlanmış test kümesi). Çıktı artifacts_v3/analysis/cv_weeks.csv:
  düzey ∈ {kat, ay, hafta}; anahtar; n; bulutlu oranı; BalAcc; (kat için) Mar–Nis 2025 payı.

    python src/cv_weeks.py [--variant B]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def balacc(y, p):
    yh = p >= 0.5
    tpr = (yh & (y == 1)).sum() / max(1, (y == 1).sum())
    tnr = (~yh & (y == 0)).sum() / max(1, (y == 0).sum())
    return 0.5 * (tpr + tnr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="B")
    ap.add_argument("--preds", default=os.path.join(ROOT, "artifacts_v3/preds"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/analysis/cv_weeks.csv"))
    ap.add_argument("--min_n", type=int, default=500, help="hafta düzeyinde en az örnek")
    a = ap.parse_args()
    m = np.load(a.meta, allow_pickle=True)
    ts = pd.to_datetime(pd.Series(m["timestamps"]).str[:12], format="%Y%m%d%H%M")
    parts = []
    for f in sorted(glob.glob(os.path.join(a.preds, f"CV_{a.variant}_f*_s1__*.npz"))):
        k = int(re.search(r"_f(\d)_", f).group(1))
        z = np.load(f)
        parts.append(pd.DataFrame({"kat": k, "row": z["test_time_rows"], "y": z["test_time_label"].astype(int),
                                   "p": z["test_time_prob"].astype(float)}))
    d = pd.concat(parts, ignore_index=True)
    d["t"] = ts.values[d["row"].values]
    d["ay"] = d["t"].dt.strftime("%Y-%m")
    iso = d["t"].dt.isocalendar()
    d["hafta"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    d["bahar25"] = d["t"] < "2025-05-01"
    rows = []
    for lvl in ("kat", "ay", "hafta"):
        for key, g in d.groupby(lvl):
            if lvl == "hafta" and len(g) < a.min_n:
                continue
            rows.append({"düzey": lvl, "anahtar": str(key), "n": len(g), "bulutlu_oran": round(g.y.mean(), 4),
                         "BalAcc": round(balacc(g.y.values, g.p.values), 4),
                         "bahar25_pay": round(g.bahar25.mean(), 4) if lvl == "kat" else np.nan})
    rows.append({"düzey": "dönem", "anahtar": "Mar-Nis 2025", "n": int(d.bahar25.sum()),
                 "bulutlu_oran": round(d[d.bahar25].y.mean(), 4), "BalAcc": round(balacc(d[d.bahar25].y.values, d[d.bahar25].p.values), 4)})
    rows.append({"düzey": "dönem", "anahtar": "Ara 2025-Haz 2026", "n": int((~d.bahar25).sum()),
                 "bulutlu_oran": round(d[~d.bahar25].y.mean(), 4), "BalAcc": round(balacc(d[~d.bahar25].y.values, d[~d.bahar25].p.values), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(out[out["düzey"] != "hafta"].to_string(index=False))
    w = out[out["düzey"] == "hafta"].sort_values("BalAcc").head(6)
    print("\nEn zayıf haftalar:\n", w.to_string(index=False))


if __name__ == "__main__":
    main()
