#!/usr/bin/env python
"""R12 — referans-kalibre kıyas modeli (ön kayıt: docs/revizyon_onkayit.md, R12).

FCI CLM ve NWC SAF CT ürünlerinin istasyon çevresi sınıf oranları (merkez, 3×3, r5, r10, r20, r30) öznitelik olarak
alınır ve aynı etiketlerle (yalnız train, yalnız SYNOP) bir sınıflandırıcı eğitilir. Böylece ürünler gözlemcinin
"0 okta / ≥5 okta" ölçeğine kalibre edilir; CNN − REF_cal farkı, ürünlerin kalibrasyonunun ötesindeki ek bilgiyi ölçer.

  Birincil : HGB, CLM + NWC öznitelikleri, test_time yalnız SYNOP, iki ürünün ortak kapsamı.
  İkincil  : LR (aynı öznitelikler); yalnız CLM; yalnız NWC; test_station / test_both.
  Ayarlar  : baselines.py ile aynı (HGB max_iter=300, lr=0.1, class_weight=balanced; LR StandardScaler + max_iter=2000,
             class_weight=balanced). Hiperparametre araması yok. Karar eşiği 0,5.

Belirsizlik: analyze_preds ile aynı blok bootstrap (test_time/test_both gün, test_station istasyon blokları), B = 2000;
CNN ve U-Net için 3 tohumun bootstrap BalAcc ortalaması. Ayrıca REF_cal − ham r30 (eşik train'de) farkı.

Çıktı: <out>/reference_calibrated.csv (+ _info.json: eğitim sayıları, sklearn sürümü)
Kullanım:
  python src/reference_calibrated_baseline.py --dir artifacts_v3
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
                           counts, load_meta, load_runs, reference_preds)

GROUPS = ["c1", "3x3", "r5", "r10", "r20", "r30"]
CLM_CLASSES = [1, 2, 3, 8]          # geçerli CLM sınıfları (açık, bulutlu, kirlenmiş, kar/buz); diğerleri geçersiz
MODELS = ["B_full", "D_unet"]


def clm_features(keys, clm_dir):
    """keys: [rc_start, wmoid] -> CLM sınıf oranları (6 grup × 4 sınıf) + geçerli oran + merkez durumu (tek-sıcak)."""
    keys = keys.assign(day=keys.rc_start.str[:8])
    parts = []
    for day, g in keys.groupby("day"):
        p = os.path.join(clm_dir, f"clm_{day}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ri = {r: i for i, r in enumerate(z["rc_start"])}
        si = {int(w): i for i, w in enumerate(z["wmoid"])}
        fi = g.rc_start.map(ri); wi = g.wmoid.map(si); ok = fi.notna() & wi.notna()
        if not ok.any():
            continue
        g = g[ok]; fi = fi[ok].astype(int).values; wi = wi[ok].astype(int).values
        cs = z["center_state"][fi, wi]
        keep = np.isin(cs, CLM_CLASSES)            # analyze_preds ile aynı kapsam tanımı
        if not keep.any():
            continue
        g, fi, wi, cs = g[keep], fi[keep], wi[keep], cs[keep]
        c = z["counts"][fi, wi].astype(np.float64)                    # (n, 6, 10)
        tot = np.maximum(c.sum(-1), 1)
        valid = c[..., CLM_CLASSES].sum(-1)
        d = {"row": g.index.values}
        for k, gn in enumerate(GROUPS):
            v = np.maximum(valid[:, k], 1)
            for cl in CLM_CLASSES:
                d[f"clm_{gn}_c{cl}"] = c[:, k, cl] / v
            d[f"clm_{gn}_valid"] = valid[:, k] / tot[:, k]
        for cl in CLM_CLASSES:
            d[f"clm_center_is{cl}"] = (cs == cl).astype(float)
        parts.append(pd.DataFrame(d))
    return pd.concat(parts).set_index("row")


def nwc_features(keys, nwc_dir):
    """keys: [slot, wmoid] -> NWC CT 16 sınıf oranı (6 grup) + merkez sınıf (tek-sıcak). Bozuk kareler (ok=False) dışarıda."""
    keys = keys.assign(day=keys.slot.str[:8])
    parts = []
    for day, g in keys.groupby("day"):
        p = os.path.join(nwc_dir, f"nwc_{day}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ri = {r: i for i, r in enumerate(z["slot"])}
        si = {int(w): i for i, w in enumerate(z["wmoid"])}
        fi = g.slot.map(ri); wi = g.wmoid.map(si); ok = fi.notna() & wi.notna()
        if not ok.any():
            continue
        g = g[ok]; fi = fi[ok].astype(int).values; wi = wi[ok].astype(int).values
        cc = z["center_class"][fi, wi]
        keep = z["ok"][fi] & (cc != 255)
        if not keep.any():
            continue
        g, fi, wi, cc = g[keep], fi[keep], wi[keep], cc[keep]
        c = z["counts"][fi, wi].astype(np.float64)                    # (n, 6, 16)
        tot = np.maximum(c.sum(-1), 1)
        d = {"row": g.index.values}
        for k, gn in enumerate(GROUPS):
            for cl in range(c.shape[-1]):
                d[f"nwc_{gn}_c{cl}"] = c[:, k, cl] / tot[:, k]
        for cl in range(c.shape[-1]):
            d[f"nwc_center_is{cl}"] = (cc == cl).astype(float)
        parts.append(pd.DataFrame(d))
    return pd.concat(parts).set_index("row")


def make_model(kind, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def gi(lo, hi):
    return f"[{lo:+.4f}, {hi:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--out", default=None, help="Varsayılan: <dir>/analysis")
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--split", default="splits/split_v2_verify.npz")
    ap.add_argument("--clm_dir", default="data/processed/clm_station")
    ap.add_argument("--nwc_dir", default="data/processed/nwcsaf_station")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis")
    os.makedirs(out, exist_ok=True)
    import sklearn
    t00 = time.time()

    meta = load_meta(a.meta, None, a.stations)
    meta["rc_start"] = pd.DatetimeIndex(meta.time).floor("10min").strftime("%Y%m%d%H%M")
    meta["slot"] = (pd.DatetimeIndex(meta.time) - pd.Timedelta("9min")).round("15min").strftime("%Y%m%d%H%M")
    split = np.load(a.split)
    E = by_exp(load_runs(a.dir))
    miss = [m for m in MODELS if m not in E]
    if miss:
        raise SystemExit(f"Tahmin dosyası bulunamayan deneyler: {miss}")

    # ---- öznitelikler: train (SYNOP, etiketli) + test kümeleri
    tr_rows = split["train"]
    tr_rows = tr_rows[(meta.label.values[tr_rows] >= 0) & (meta.src.values[tr_rows] == "SYNOP")]
    te_rows = np.unique(np.concatenate([split[s] for s in SUBSETS if s in split.files]))
    need = np.union1d(tr_rows, te_rows)
    print(f"[1/4] öznitelikler okunuyor ({len(need):,} satır) ...", flush=True)
    t0 = time.time()
    FC = clm_features(meta.iloc[need][["rc_start", "wmoid"]], a.clm_dir)
    print(f"      CLM: {len(FC):,} satır, {FC.shape[1]} öznitelik ({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    FN = nwc_features(meta.iloc[need][["slot", "wmoid"]], a.nwc_dir)
    print(f"      NWC: {len(FN):,} satır, {FN.shape[1]} öznitelik ({time.time() - t0:.0f} s)", flush=True)
    FB = FC.join(FN, how="inner")
    feats = {"CLM+NWC": FB, "CLM": FC, "NWC": FN}

    # ---- ham r30 referansları (eşik train'de; analyze_preds ile aynı)
    print("[2/4] ham r30 referansları ...", flush=True)
    raw, rinfo = reference_preds(meta, split, a.clm_dir, a.nwc_dir)

    # ---- eğitim
    print("[3/4] REF_cal modelleri eğitiliyor ...", flush=True)
    N = len(meta)
    cal, info = {}, {"sklearn": sklearn.__version__, "split": a.split, "raw_r30": rinfo, "models": {}}
    jobs = [(fs, kind) for fs in feats for kind in ("hgb", "lr")]
    for j, (fs, kind) in enumerate(jobs, 1):
        F = feats[fs]
        r = np.intersect1d(tr_rows, F.index.values)
        X, y = F.loc[r].values, meta.label.values[r]
        t0 = time.time()
        m = make_model(kind, a.seed).fit(X, y)
        p = np.full(N, np.nan)
        rt = np.intersect1d(te_rows, F.index.values)
        p[rt] = m.predict_proba(F.loc[rt].values)[:, 1]
        name = f"REFcal_{kind}_{fs}"
        cal[name] = p
        tr_ba = float(balacc_from(counts((m.predict_proba(X)[:, 1] >= 0.5).astype(int), y)))
        info["models"][name] = {"n_train": int(len(r)), "n_feat": int(X.shape[1]), "train_BalAcc": tr_ba,
                                "clear_frac_train": float((y == 0).mean())}
        print(f"      [{j}/{len(jobs)} %{100 * j // len(jobs):3d}] {name}: n_train {len(r):,}, {X.shape[1]} öz., "
              f"train BalAcc {tr_ba:.3f} ({time.time() - t0:.0f} s)", flush=True)

    # ---- değerlendirme: yalnız SYNOP, CLM ve NWC'nin ortak kapsamı (birincil); ayrıca her ürünün kendi kapsamı
    print("[4/4] blok bootstrap ...", flush=True)
    rng = np.random.default_rng(a.seed)
    rows_out = []
    for s in SUBSETS:
        rows = common_rows([r for e in MODELS for r in E[e]], s)
        if rows is None:
            continue
        syn = meta.src.values[rows] == "SYNOP"
        y_all = meta.label.values[rows]
        for cov in ("CLM+NWC", "CLM", "NWC"):
            if cov == "CLM+NWC":
                cm = np.isfinite(raw["CLM_r30"][rows]) & np.isfinite(raw["NWC_r30"][rows])
            else:
                cm = np.isfinite(raw[f"{cov}_r30"][rows])
            m = syn & cm & (y_all >= 0)
            # ortak kapsamda kalibre modellerin de tahmini olmalı (aynı tanım; güvenlik)
            refs_here = [k for k in cal if k.endswith("_" + cov) or cov == "CLM+NWC"]
            for k in refs_here:
                m &= np.isfinite(cal[k][rows])
            if m.sum() < 200:
                continue
            y = y_all[m]
            codes, uni = pd.factorize(meta[BLOCK[s]].values[rows][m])
            W = boot_weights(len(uni), a.boot, rng)

            def bb(pred):
                return balacc_from(W @ block_counts(pred, y, codes, len(uni)))

            mod_pt, mod_bt = {}, {}
            for e in MODELS:
                pts, bts = [], []
                for r in E[e]:
                    if s not in r["sets"]:
                        continue
                    _, p = aligned(r, s, rows)
                    pred = (p[m] >= 0.5).astype(int)
                    pts.append(float(balacc_from(counts(pred, y))))
                    bts.append(bb(pred))
                mod_pt[e], mod_bt[e] = float(np.mean(pts)), np.mean(bts, axis=0)
            raw_here = ["CLM_r30", "NWC_r30"] if cov == "CLM+NWC" else [f"{cov}_r30"]
            raw_pt = {k: float(balacc_from(counts(raw[k][rows][m].astype(int), y))) for k in raw_here}
            raw_bt = {k: bb(raw[k][rows][m].astype(int)) for k in raw_here}
            for k in refs_here:
                if cov != "CLM+NWC" and not k.endswith("_" + cov):
                    continue
                pred = (cal[k][rows][m] >= 0.5).astype(int)
                c = counts(pred, y)
                rp, rb = float(balacc_from(c)), bb(pred)
                row = {"küme": s, "kapsam": cov, "referans": k, "n": int(m.sum()), "n_blok": int(len(uni)),
                       "açık_oranı": float((y == 0).mean()), "REFcal_BalAcc": rp,
                       "REFcal_GA": gi(*np.percentile(rb, [2.5, 97.5]))}
                for e in MODELS:
                    d = mod_bt[e] - rb
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    row[f"{e}_BalAcc"] = mod_pt[e]
                    row[f"{e}_eksi_REFcal"] = mod_pt[e] - rp
                    row[f"{e}_eksi_REFcal_GA"] = gi(lo, hi)
                    row[f"{e}_GA_sifiri_disliyor"] = bool(lo > 0 or hi < 0)
                for rk in raw_here:
                    d = rb - raw_bt[rk]
                    row[f"{rk}_BalAcc"] = raw_pt[rk]
                    row[f"REFcal_eksi_{rk}"] = rp - raw_pt[rk]
                    row[f"REFcal_eksi_{rk}_GA"] = gi(*np.percentile(d, [2.5, 97.5]))
                row["birincil"] = bool(s == "test_time" and cov == "CLM+NWC" and k == "REFcal_hgb_CLM+NWC")
                rows_out.append(row)
        print(f"      {s} tamam", flush=True)

    R = pd.DataFrame(rows_out)
    R.to_csv(os.path.join(out, "reference_calibrated.csv"), index=False, float_format="%.6f")
    info["wall_s"] = time.time() - t00
    json.dump(info, open(os.path.join(out, "reference_calibrated_info.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    pd.set_option("display.width", 250)
    cols = ["küme", "kapsam", "referans", "n", "REFcal_BalAcc", "B_full_BalAcc", "B_full_eksi_REFcal",
            "B_full_eksi_REFcal_GA", "D_unet_eksi_REFcal_GA"]
    print(R[cols].to_string(index=False))
    p = R[R.birincil]
    if len(p):
        p = p.iloc[0]
        print(f"\nBİRİNCİL (test_time, SYNOP, CLM+NWC ortak kapsam, HGB): REF_cal {p.REFcal_BalAcc:.4f}; "
              f"CNN − REF_cal {p.B_full_eksi_REFcal:+.4f} {p.B_full_eksi_REFcal_GA}; "
              f"U-Net − REF_cal {p.D_unet_eksi_REFcal:+.4f} {p.D_unet_eksi_REFcal_GA}")
    print(f"süre {info['wall_s'] / 60:.1f} dk; sklearn {sklearn.__version__}")


if __name__ == "__main__":
    main()
