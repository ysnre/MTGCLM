"""B2 (27 Kas, ön-kayıtlı, betimsel): verify politikasının çıkardığı doğrulanmamış METAR açık raporlarının profili.

Silinen = split_v1_compact'ın etiketli kümelerinde olup split_v2_verify'da olmayan satırlar (hepsi METAR, etiket 0). Tutulan
açıklar METAR ve SYNOP ayrı. Her grup için: n, istasyon sayısı, gece / gündüz payı, UTC saat ve ay dağılımı, aynı istasyonun aynı
yuvarlanmış saatinde (±30 dk) SYNOP raporu bulunma oranı ve FCI CLM r30 / NWC SAF CT r30'a göre bulutlu oranı (eşikler train'de,
src/ref_rows_cache.py). Birincil: silinen − tutulan METAR açıkları NWC r30 bulutlu oranı farkı, istasyon blokları bootstrap.
Yorum kuralı docs/revizyon_onkayit.md §3 (27 Kas 14:30). "(ek)" grubu ön-kayıtta yoktu (sonuç sonrası, betimsel).
Çıktı: <out>/removed_metar_profile.csv (özet), removed_metar_profile_strata.csv (saat, ay, küme)
    python src/removed_metar_profile.py --out artifacts_v3/analysis
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import load_meta  # noqa: E402
from ref_rows_cache import get_refs  # noqa: E402

SETS = ["train", "val", "test_time", "test_station", "test_both"]


def frac_diff_boot(g1, g2, col, B, rng):
    """g1 − g2 oran farkı; istasyon blokları (iki grubun istasyon birleşimi yeniden örneklenir)."""
    a = g1.dropna(subset=[col]).groupby("wmoid")[col].agg(["sum", "count"])
    b = g2.dropna(subset=[col]).groupby("wmoid")[col].agg(["sum", "count"])
    st = a.index.union(b.index)
    a, b = a.reindex(st, fill_value=0).values, b.reindex(st, fill_value=0).values
    idx = rng.integers(0, len(st), (B, len(st)))
    W = np.stack([np.bincount(i, minlength=len(st)) for i in idx]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (W @ a[:, 0]) / (W @ a[:, 1]) - (W @ b[:, 0]) / (W @ b[:, 1])
    pt = a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return pt, lo, hi, len(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts_v3/analysis")
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--v1", default="splits/split_v1_compact.npz")
    ap.add_argument("--verify", default="splits/split_v2_verify.npz")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    meta = load_meta(a.meta, None, a.stations)
    refs = get_refs(meta)
    s1, sv = np.load(a.v1), np.load(a.verify)
    setof = np.full(len(meta), "", dtype=object)
    for k in SETS:
        setof[s1[k]] = k
    kept = np.zeros(len(meta), bool)
    kept[np.concatenate([sv[k] for k in SETS])] = True
    lab = np.zeros(len(meta), bool)
    lab[np.concatenate([s1[k] for k in SETS])] = True
    M = meta.assign(set=setof, CLM=refs["CLM_r30"], NWC=refs["NWC_r30"])
    M["hour"] = pd.DatetimeIndex(M.time).hour
    M["hkey"] = pd.DatetimeIndex(M.time).round("60min").strftime("%Y%m%d%H")
    syn = set(zip(M.wmoid[M.src == "SYNOP"], M.hkey[M.src == "SYNOP"]))
    M["synop_var"] = [(w, h) in syn for w, h in zip(M.wmoid, M.hkey)]
    removed = M[lab & ~kept]
    assert (removed.src == "METAR").all() and (removed.label == 0).all(), "silinen satırlar METAR açık olmalı"
    groups = {"silinen METAR açık": removed,
              "tutulan METAR açık": M[kept & (M.src == "METAR") & (M.label == 0)],
              "tutulan SYNOP açık": M[kept & (M.src == "SYNOP") & (M.label == 0)],
              # ek (ön-kayıtta yok, birincil sonuç görüldükten sonra eklendi): aynı saatte SYNOP'u olmayan silinenler
              "silinen METAR açık, aynı saatte SYNOP yok (ek)": removed[~removed.synop_var]}
    rows = []
    for g, d in groups.items():
        rows.append({"grup": g, "n": len(d), "istasyon": d.wmoid.nunique(), "gece_payı": (d.illum == "gece").mean(),
                     "gündüz_payı": (d.illum == "gündüz").mean(), "SYNOP_aynı_saatte": d.synop_var.mean(),
                     "CLM_kapsam": d.CLM.notna().mean(), "CLM_r30_bulutlu": d.CLM.mean(),
                     "NWC_kapsam": d.NWC.notna().mean(), "NWC_r30_bulutlu": d.NWC.mean(),
                     "fark_tutulan_METAR": np.nan, "GA_alt": np.nan, "GA_üst": np.nan})
    for col in ("NWC", "CLM"):
        pt, lo, hi, ns = frac_diff_boot(groups["silinen METAR açık"], groups["tutulan METAR açık"], col, a.boot, rng)
        rows.append({"grup": f"silinen − tutulan METAR, {col} r30 bulutlu", "n": len(removed), "istasyon": ns,
                     "fark_tutulan_METAR": pt, "GA_alt": lo, "GA_üst": hi})
    D = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    D.to_csv(os.path.join(a.out, "removed_metar_profile.csv"), index=False, encoding="utf-8", float_format="%.6f")
    st = []
    for g, d in groups.items():
        for fac in ("hour", "month", "set"):
            c = d.groupby(fac).agg(n=("wmoid", "size"), NWC_bulutlu=("NWC", "mean"), CLM_bulutlu=("CLM", "mean"),
                                   SYNOP_aynı_saatte=("synop_var", "mean")).reset_index()
            c["pay"] = c.n / len(d)
            for _, r in c.iterrows():
                st.append({"grup": g, "etken": fac, "düzey": r[fac], **{k: r[k] for k in ("n", "pay", "NWC_bulutlu", "CLM_bulutlu",
                                                                                          "SYNOP_aynı_saatte")}})
    pd.DataFrame(st).to_csv(os.path.join(a.out, "removed_metar_profile_strata.csv"), index=False, encoding="utf-8",
                            float_format="%.6f")
    pd.set_option("display.width", 220)
    print(D.round(4).to_string())


if __name__ == "__main__":
    main()
