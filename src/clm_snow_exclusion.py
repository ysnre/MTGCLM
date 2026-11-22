#!/usr/bin/env python
"""Keşifsel (ön kayıtta yok): kar mevsimi yüksek istasyon satırları çıkarılınca model − referans farkı.

R7'de FCI CLM'nin kış gündüzü > 1500 m istasyonlarda açık raporların %91'ini bulutlu saydığı görüldü (ürünün kar testi kapalı,
EUM2045564). Bu betik, modelin CLM'ye üstünlüğünün bu satırlardan ne kadar geldiğini gösterir. Hesap analyze_preds.reference_analysis
ile aynı (aynı satırlar, aynı r30 eşikleri, aynı blok bootstrap); yalnız satır süzgeci eklenir.

Varyantlar:
  tümü                         : süzgeç yok (references.csv ile aynı sayılar)
  kış gündüzü >1500 m hariç    : Ara–Mar, gündüz, istasyon > 1500 m satırları çıkarılır
  tüm kış >1500 m hariç        : Ara–Mar, istasyon > 1500 m satırları (gece dahil) çıkarılır
Etiket: tümü ve yalnız SYNOP. Modeller: B_full, D_unet. Referanslar: CLM ve NWC (merkez, r30).
Yorum kuralı yoktur (sonuç görüldükten sonra tanımlandı); ek tabloda "keşifsel" olarak raporlanır.

Çıktı: <out>/clm_snow_exclusion.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import (BLOCK, SUBSETS, aligned, balacc_from, block_counts, boot_weights, by_exp,  # noqa: E402
                           common_rows, counts, load_meta, load_runs, reference_preds)

MODELS = ("B_full", "D_unet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--split", default="splits/split_v2_verify.npz")
    ap.add_argument("--clm_dir", default="data/processed/clm_station")
    ap.add_argument("--nwc_dir", default="data/processed/nwcsaf_station")
    ap.add_argument("--elev", type=float, default=1500.0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    meta = load_meta(a.meta, None, a.stations)
    sp = np.load(a.split)
    E = by_exp(load_runs(a.dir))
    refs, _ = reference_preds(meta, sp, a.clm_dir, a.nwc_dir)
    mon = pd.DatetimeIndex(meta.time).month.values
    winter_hi = np.isin(mon, [12, 1, 2, 3]) & (meta.elev.values > a.elev)
    day = meta.illum.values == "gündüz"
    variants = {"tümü": np.zeros(len(meta), bool),
                f"kış gündüzü >{a.elev:.0f} m hariç": winter_hi & day,
                f"tüm kış >{a.elev:.0f} m hariç": winter_hi}
    rows = []
    for s in SUBSETS:
        rr = common_rows([r for e in MODELS for r in E[e]], s)
        if rr is None:
            continue
        for lab in ("tümü", "SYNOP"):
            base = np.ones(len(rr), bool) if lab == "tümü" else meta.src.values[rr] == "SYNOP"
            for vname, drop in variants.items():
                keep = base & ~drop[rr]
                for ref, arr in refs.items():
                    rp = arr[rr]
                    m = keep & np.isfinite(rp)
                    if m.sum() < 200:
                        continue
                    y = meta.label.values[rr][m]
                    codes, uni = pd.factorize(meta[BLOCK[s]].values[rr][m])
                    W = boot_weights(len(uni), a.boot, rng)
                    rpred = rp[m].astype(int)
                    rb = balacc_from(W @ block_counts(rpred, y, codes, len(uni)))
                    row = {"küme": s, "etiket": lab, "varyant": vname, "referans": ref, "n": int(m.sum()),
                           "çıkarılan": int((base & drop[rr] & np.isfinite(rp)).sum()),
                           "ref_BalAcc": float(balacc_from(counts(rpred, y)))}
                    for e in MODELS:
                        pts, bts = [], []
                        for r in E[e]:
                            if s not in r["sets"]:
                                continue
                            _, p = aligned(r, s, rr)
                            pred = (p[m] >= 0.5).astype(int)
                            pts.append(float(balacc_from(counts(pred, y))))
                            bts.append(balacc_from(W @ block_counts(pred, y, codes, len(uni))))
                        d = np.mean(bts, axis=0) - rb
                        lo, hi = np.percentile(d, [2.5, 97.5])
                        row[f"{e}_BalAcc"] = float(np.mean(pts))
                        row[f"{e}_eksi_ref"] = float(np.mean(pts)) - row["ref_BalAcc"]
                        row[f"{e}_GA_alt"], row[f"{e}_GA_üst"] = float(lo), float(hi)
                    rows.append(row)
        print(f"  {s} tamam", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "clm_snow_exclusion.csv"), index=False, float_format="%.6f")
    pd.set_option("display.width", 220)
    t = R[R.referans.isin(["CLM_r30", "NWC_r30"])][["küme", "etiket", "varyant", "referans", "n", "çıkarılan", "ref_BalAcc",
                                                    "B_full_BalAcc", "B_full_eksi_ref", "B_full_GA_alt", "B_full_GA_üst"]]
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
