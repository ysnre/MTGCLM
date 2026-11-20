#!/usr/bin/env python
"""R13 — normalizasyon kayması tanısı (ön kayıt: docs/revizyon_onkayit.md, R13; plan §1 ölçütü).

Kanal başına (istasyon pikseli, ham birim; data/processed/center_values.npz) ay × aydınlanma sınıfı (gündüz / alacakaranlık /
gece; analyze_preds ile aynı güneş yüksekliği tanımı) ortalaması, EĞİTİM kümesinin aynı aydınlanma sınıfındaki ortalama ve
standart sapmasına göre σ biriminde:  kayma = (ort_{ay, sınıf} − ort_{train, sınıf}) / σ_{train, sınıf}.
  (a) ön-kayıtlı ölçüt: ayın tüm etiketli satırları (train + val + test) — ayın eğitim iklimine göre konumu.
  (b) tanısal: aynı ay × sınıf içinde test kümeleri − train (bölmenin kendisinin yarattığı kayma).
Kural: yalnız IR kanallarında (ve radar dışında) |kayma| > 0,5σ olursa yeniden eğitim için kullanıcı kararı istenir.
Güneş kanalları (VIS/NIR) ve aux yalnız tanısal raporlanır.

Çıktı: <out>/normalization_shift.csv (uzun biçim) ve normalization_shift_summary.csv (kanal × sınıf: en büyük |kayma|, ay)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import SUBSETS, load_meta  # noqa: E402

THRESH = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centre", default="data/processed/center_values.npz")
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--split", default="splits/split_v2_verify.npz")
    ap.add_argument("--out", default="artifacts_v3/analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meta = load_meta(a.meta, None, a.stations)
    z = np.load(a.centre)
    base, aux = z["base"].astype(np.float64), z["aux"].astype(np.float64)
    names = [str(n) for n in z["base_names"]] + [str(n) for n in z["aux_names"]]
    X = np.concatenate([base, aux], 1)
    assert len(X) == len(meta), (len(X), len(meta))
    sp = np.load(a.split)
    setof = np.full(len(meta), "", dtype=object)
    for s in ["train", "val"] + SUBSETS:
        if s in sp.files:
            setof[sp[s]] = "train" if s == "train" else ("val" if s == "val" else "test")
    labelled = setof != ""

    def kind(n):
        if n.startswith("ir_") or n.startswith("wv_"):
            return "IR"
        if n.startswith("vis_") or n.startswith("nir_"):
            return "güneş"
        return "radar" if n == "radar" else "aux"

    rows = []
    for c in ("gündüz", "alacakaranlık", "gece"):
        cm = meta.illum.values == c
        tr = cm & (setof == "train")
        mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0)
        sd = np.where(sd > 0, sd, np.nan)
        for mon in sorted(meta.month.unique()):
            mm = cm & (meta.month.values == mon)
            a_ = mm & labelled
            te = mm & (setof == "test"); trm = mm & (setof == "train")
            if a_.sum() < 50:
                continue
            ma = np.nanmean(X[a_], 0)
            mte = np.nanmean(X[te], 0) if te.sum() >= 50 else np.full(len(names), np.nan)
            mtr = np.nanmean(X[trm], 0) if trm.sum() >= 50 else np.full(len(names), np.nan)
            for j, n in enumerate(names):
                rows.append({"aydınlanma": c, "ay": mon, "kanal": n, "tür": kind(n), "n_ay": int(a_.sum()),
                             "n_test": int(te.sum()), "kayma_ay_sigma": (ma[j] - mu[j]) / sd[j],
                             "kayma_test_eksi_train_sigma": (mte[j] - mtr[j]) / sd[j],
                             "train_ort": mu[j], "train_sd": sd[j]})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "normalization_shift.csv"), index=False, float_format="%.4f")

    S = []
    for (c, n), g in D.groupby(["aydınlanma", "kanal"], sort=False):
        i = g.kayma_ay_sigma.abs().idxmax(); k = g.kayma_test_eksi_train_sigma.abs().idxmax()
        S.append({"aydınlanma": c, "kanal": n, "tür": g.tür.iloc[0],
                  "maks_abs_kayma_ay": abs(g.kayma_ay_sigma[i]), "ay": g.ay[i], "işaret": np.sign(g.kayma_ay_sigma[i]),
                  "maks_abs_test_eksi_train": abs(g.kayma_test_eksi_train_sigma[k]) if pd.notna(k) else np.nan,
                  "ay_test": g.ay[k] if pd.notna(k) else "",
                  "eşik_aşan_ay_sayısı": int((g.kayma_ay_sigma.abs() > THRESH).sum())})
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(a.out, "normalization_shift_summary.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(S.to_string(index=False))
    ir = S[S.tür == "IR"]
    hit = ir[ir.maks_abs_kayma_ay > THRESH]
    print(f"\nKURAL (a): IR kanallarında |kayma| > {THRESH}σ olan kanal × sınıf: {len(hit)} / {len(ir)}")
    print(f"tanısal (b): IR test − train en büyük |kayma| {ir.maks_abs_test_eksi_train.max():.3f}σ")


if __name__ == "__main__":
    main()
