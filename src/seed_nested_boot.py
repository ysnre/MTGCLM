"""Tohum (eğitim) varyansını belirsizliğe katan iç içe bootstrap (26 Kas iç hakem turu, İkincil 1; K-B kararı, K66).

analyze_preds.py'deki eşleştirilmiş farklar her yeniden örneklemde 3 tohumun ORTALAMASINI alır → eğitim varyansı sabit kabul
edilir. Burada her yinelemede (a) gün blokları (test_station'da istasyon blokları) ve (b) her deneyin tohumları yerine koyarak
yeniden örneklenir; fark = mean(seçilen tohumların BalAcc'i)_B − mean(...)_A. Aynı blok ağırlıklarıyla eski (sabit tohum
ortalaması) GA da hesaplanır → GA genişlemesi doğrudan karşılaştırılır. Ayrıca tohum çiftleri (3 × 3) arasındaki nokta
farklarının aralığı verilir. Yalnız tek bölme; CV katlarında tek tohum var (CV_*_s1) → kapsam dışı.

Not: 3 tohumla tohum dağılımı kaba temsil edilir (yeniden örneklemede yalnız 10 farklı çoklu küme); GA'lar eğitim varyansının
alt sınırı gibi okunmalı. Yeni eğitim yok; kayıtlı tahminler (preds/*.npz) kullanılır.

Çıktı: <out>/seed_nested_pairs.csv (+ _info.json)
    python src/seed_nested_boot.py --dir artifacts_v3 --out artifacts_v3/analysis
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import (BLOCK, SUBSETS, aligned, balacc_from, block_counts, boot_weights, by_exp, common_rows,  # noqa: E402
                           counts, load_meta, load_runs)

PAIRS = ("B_full-A_sat,C_noawos-A_sat,B_full-C_noawos,C_t2m-C_noawos,A_sat-A_norad,B_full-B_norad,"
         "B_full-B_full_shuf,D_unet-A_sat,RESNET-A_sat,RESNET_CIFAR-A_sat,D_unet_jit-D_unet")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--h5", default=None)
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    meta = load_meta(a.meta, a.h5, a.stations)
    E = {e: v for e, v in by_exp(load_runs(a.dir)).items() if not e.startswith("CV_")}
    pairs = [p.strip().split("-", 1) for p in a.pairs.split(",") if p.strip()]
    need = sorted({x for p in pairs for x in p})
    miss = [e for e in need if e not in E]
    if miss:
        print("eksik deneyler (çiftler atlanır):", miss)
    rng = np.random.default_rng(a.seed)
    rows_out = []
    for s in SUBSETS:
        exps = [e for e in need if e in E and any(s in r["sets"] for r in E[e])]
        rows = common_rows([r for e in exps for r in E[e]], s)
        blk = pd.factorize(meta[BLOCK[s]].values[rows])[0]
        for lab, mask in (("tümü", np.ones(len(rows), bool)), ("SYNOP", meta.src.values[rows] == "SYNOP")):
            codes, uni = pd.factorize(blk[mask])
            W = boot_weights(len(uni), a.boot, rng)
            SB, PT = {}, {}                       # deney -> [B, n_tohum] blok-bootstrap BalAcc; tohum başına nokta
            for e in exps:
                bs, pt = [], []
                for r in E[e]:
                    if s not in r["sets"]:
                        continue
                    y, p = aligned(r, s, rows)
                    y, p = y[mask], p[mask]
                    pred = (p >= 0.5).astype(int)
                    pt.append(float(balacc_from(counts(pred, y))))
                    bs.append(balacc_from(W @ block_counts(pred, y, codes, len(uni))))
                SB[e], PT[e] = np.stack(bs, 1), np.array(pt)
            for e2, e1 in pairs:
                if e1 not in SB or e2 not in SB:
                    continue
                k2, k1 = SB[e2].shape[1], SB[e1].shape[1]
                d_fix = SB[e2].mean(1) - SB[e1].mean(1)
                i2 = rng.integers(0, k2, (a.boot, k2))
                i1 = rng.integers(0, k1, (a.boot, k1))
                b2 = np.take_along_axis(SB[e2], i2, 1).mean(1)
                b1 = np.take_along_axis(SB[e1], i1, 1).mean(1)
                d_nest = b2 - b1
                lo_f, hi_f = np.percentile(d_fix, [2.5, 97.5])
                lo_n, hi_n = np.percentile(d_nest, [2.5, 97.5])
                sd = (PT[e2][:, None] - PT[e1][None, :]).ravel()
                rows_out.append({
                    "küme": s, "etiket": lab, "çift": f"{e2} − {e1}", "tohum_B": k2, "tohum_A": k1,
                    "fark": PT[e2].mean() - PT[e1].mean(),
                    "GA_alt_sabit": lo_f, "GA_üst_sabit": hi_f,
                    "GA_alt_içiçe": lo_n, "GA_üst_içiçe": hi_n,
                    "genişleme_oranı": (hi_n - lo_n) / (hi_f - lo_f),
                    "sıfırı_dışlıyor_sabit": bool(lo_f > 0 or hi_f < 0),
                    "sıfırı_dışlıyor_içiçe": bool(lo_n > 0 or hi_n < 0),
                    "p_içiçe": min(1.0, 2 * min((d_nest <= 0).mean(), (d_nest >= 0).mean())),
                    "tohum_çifti_fark_min": sd.min(), "tohum_çifti_fark_maks": sd.max(),
                    "tohum_çifti_pozitif": f"{int((sd > 0).sum())}/{len(sd)}",
                    "blok": BLOCK[s], "blok_sayısı": len(uni)})
        print(f"  {s}: {time.time() - t0:.0f} s")
    D = pd.DataFrame(rows_out)
    D.to_csv(os.path.join(out, "seed_nested_pairs.csv"), index=False, encoding="utf-8", float_format="%.6f")
    info = {"boot": a.boot, "seed": a.seed, "pairs": a.pairs, "not": "tohumlar her deney için bağımsız, yerine koyarak; "
            "CV kapsam dışı (kat başına tek tohum)",
            "medyan_genişleme_tümü": float(D[D["etiket"] == "tümü"]["genişleme_oranı"].median()),
            "sıfırı_dışlayan_sabit_tümü": int(D[D["etiket"] == "tümü"]["sıfırı_dışlıyor_sabit"].sum()),
            "sıfırı_dışlayan_içiçe_tümü": int(D[D["etiket"] == "tümü"]["sıfırı_dışlıyor_içiçe"].sum()),
            "n_çift_tümü": int((D["etiket"] == "tümü").sum())}
    json.dump(info, open(os.path.join(out, "seed_nested_pairs_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False, indent=1), f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
