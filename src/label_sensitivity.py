"""
label_sensitivity.py — P9: yer gerçeği tanımının skora ve model–referans farkına etkisi (yeniden eğitim yok).

Etiket tanımları (modeller verify etiketleriyle eğitildi; yalnız değerlendirme satırları değişir):
  verify   : makaledeki tanım (0 okta açık, >= 5 bulutlu; METAR açık yalnız aynı saatte SYNOP N=0 ise)
  keep     : verify + doğrulanmamış METAR açık satırları (split_v1_compact − split_v2_verify; predict_rows.py tahminleri)
  okta_ge1 : verify test_time + 1–4 okta satırları (test_partial, tamamı test haftalarında) bulutlu sayılarak
Modeller: üç seed'in her biri ayrı değerlendirilip BalAcc ortalanır; bootstrap'ta seed ortalaması (analyze_preds ile aynı).
Referanslar (FCI CLM, NWC SAF CT; merkez ve 30 km, eşikler yalnız train'de) aynı satırlarda; model − referans farkı
referansın kapsadığı satırlarda. Bloklar: gün (test_station: istasyon). Tüm etiketler (keep METAR ekler, SYNOP-only anlamsız).
Çıktı: artifacts_v3/analysis/label_sensitivity.csv

    python src/label_sensitivity.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import (BLOCK, balacc_from, block_counts, boot_weights, by_exp, counts, load_meta,  # noqa: E402
                           load_runs, reference_preds)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = ("test_time", "test_station", "test_both")


class _Split(dict):
    @property
    def files(self):
        return list(self.keys())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(ROOT, "artifacts_v3"))
    ap.add_argument("--exps", default="B_full,A_sat,C_noawos,C_t2m,D_unet")
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--stations", default=os.path.join(ROOT, "data/raw_observation/station_list.csv"))
    ap.add_argument("--split", default=os.path.join(ROOT, "splits/split_v2_verify.npz"))
    ap.add_argument("--split_keep", default=os.path.join(ROOT, "splits/split_v1_compact.npz"))
    ap.add_argument("--clm_dir", default=os.path.join(ROOT, "data/processed/clm_station"))
    ap.add_argument("--nwc_dir", default=os.path.join(ROOT, "data/processed/nwcsaf_station"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/analysis/label_sensitivity.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    meta = load_meta(a.meta, None, a.stations)
    sp = np.load(a.split)
    kp = np.load(a.split_keep)
    exps = a.exps.split(",")
    E = by_exp(load_runs(a.dir))
    # satır -> olasılık (seed listesi) sözlükleri: verify kümeleri + test_partial + keep ekleri
    P = {e: [] for e in exps}
    have_keep = True
    for e in exps:
        for r in sorted(E[e], key=lambda r: r["seed"]):
            prob = {}
            for s in SETS:
                rr, _, pp = r["sets"][s]
                prob.update(zip(rr.tolist(), pp.tolist()))
            pf = None
            for cand in os.listdir(os.path.join(a.dir, "preds")):
                if cand.startswith(f"{e}_s{r['seed']}__") and cand.endswith(".npz"):
                    pf = os.path.join(a.dir, "preds", cand)
            zz = np.load(pf)
            prob.update(zip(zz["test_partial_rows"].tolist(), zz["test_partial_prob"].tolist()))
            xf = os.path.join(a.dir, "preds_extra", os.path.basename(pf).replace(".npz", "__keep_extra.npz"))
            if os.path.exists(xf):
                zx = np.load(xf)
                for s in SETS:
                    prob.update(zip(zx[f"{s}_rows"].tolist(), zx[f"{s}_prob"].tolist()))
            else:
                have_keep = False
            P[e].append(prob)
    if not have_keep:
        print("UYARI: preds_extra eksik — 'keep' tanımı atlanıyor (önce predict_rows.py).")

    defs = {}
    for s in SETS:
        defs[("verify", s)] = (np.sort(sp[s]), None)
        if have_keep:
            defs[("keep", s)] = (np.sort(kp[s]), None)
    lab_ge1 = meta["label"].values.copy()
    tp = sp["test_partial"]
    lab_ge1[tp] = 1
    defs[("okta_ge1", "test_time")] = (np.sort(np.concatenate([sp["test_time"], tp])), lab_ge1)

    all_rows = np.unique(np.concatenate([v[0] for v in defs.values()]))
    split_for_refs = _Split(train=sp["train"], test_time=all_rows)
    refs, info = reference_preds(meta, split_for_refs, a.clm_dir, a.nwc_dir)
    print("Referans eşikleri (train):", {k: v["eşik"] for k, v in info.items()})

    out = []
    for (name, s), (rows, lab_override) in defs.items():
        y_all = (meta["label"].values if lab_override is None else lab_override)[rows]
        keep_m = y_all >= 0
        rows, y_all = rows[keep_m], y_all[keep_m]
        src = meta["src"].values[rows]
        blk_all = meta[BLOCK[s]].values[rows]
        for ref in [None] + list(refs):
            m = np.ones(len(rows), bool) if ref is None else np.isfinite(refs[ref][rows])
            if m.sum() < 200:
                continue
            y = y_all[m]
            codes, uni = pd.factorize(blk_all[m])
            W = boot_weights(len(uni), a.boot, rng)
            row = {"tanım": name, "küme": s, "referans": ref or "-", "n": int(m.sum()), "n_açık": int((y == 0).sum()),
                   "n_bulutlu": int((y == 1).sum()), "n_METAR": int((src[m] == "METAR").sum()), "n_blok": len(uni)}
            if ref is not None:
                rp = refs[ref][rows][m].astype(int)
                row["ref_BalAcc"] = float(balacc_from(counts(rp, y)))
                rb = balacc_from(W @ block_counts(rp, y, codes, len(uni)))
            for e in exps:
                pts, bts = [], []
                for prob in P[e]:
                    p = np.array([prob.get(int(r), np.nan) for r in rows[m]])
                    if np.isnan(p).any():
                        raise SystemExit(f"{e}: {name}/{s} satırlarında tahmin eksik ({np.isnan(p).sum()})")
                    pred = (p >= 0.5).astype(int)
                    pts.append(float(balacc_from(counts(pred, y))))
                    bts.append(balacc_from(W @ block_counts(pred, y, codes, len(uni))))
                row[e] = float(np.mean(pts))
                bm = np.mean(bts, axis=0)
                if ref is None:
                    row[f"{e}_GA"] = f"[{np.percentile(bm, 2.5):.4f}, {np.percentile(bm, 97.5):.4f}]"
                else:
                    d = bm - rb
                    row[f"{e}−ref"] = row[e] - row["ref_BalAcc"]
                    row[f"{e}−ref_GA"] = f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
            out.append(row)
    R = pd.DataFrame(out)
    R.to_csv(a.out, index=False, encoding="utf-8")
    pd.set_option("display.width", 250)
    c = ["tanım", "küme", "referans", "n", "n_açık", "n_METAR", "ref_BalAcc", "B_full", "B_full−ref", "B_full−ref_GA"]
    print(R[[x for x in c if x in R]].round(4).to_string(index=False))
    print(f"Yazıldı: {a.out}")


if __name__ == "__main__":
    main()
