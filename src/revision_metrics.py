"""
revision_metrics.py — revizyon analizleri R6, R17, R20 (ön-kayıt: docs/revizyon_onkayit.md; yeniden eğitim yok)

R6  Kontenjans metrikleri (meteorolojik doğrulama dili). Her test kümesi × {tümü, SYNOP} için
    a = isabet (tahmin bulutlu, gözlem bulutlu), b = yanlış alarm, c = kaçırma, d = doğru açık;
    POD = a/(a+c), POD_açık = d/(b+d), POFD = b/(b+d), FAR = b/(a+b), frekans bias = (a+b)/(a+c), CSI = a/(a+b+c),
    KSS (Hanssen–Kuipers) = POD − POFD (= 2·BalAcc − 1), HSS (Heidke).
    Modeller tüm satırlarda; FCI CLM ve NWC SAF CT (merkez, 30 km) yalnız kapsadıkları satırlarda ve aynı satırlarda
    B_full / D_unet / D_unet_jit ile. Sayımlar seed ortalaması; KSS ve HSS için %95 blok-bootstrap GA (seed ortalaması).
R17 Doğrulanmamış METAR açık raporları ("verify" politikasının çıkardığı satırlar; test kümelerinde): yalnız açık etiket
    olduğu için POD tanımsız → doğru açık oranı (özgüllük) = d/(b+d), modeller ve referanslar, blok-bootstrap GA.
R20 Bulutlu sınıf sınırı ≥ 6 okta (NWC SAF doğrulamasıyla uyumlu): yalnız SYNOP satırları, 5 okta raporları çıkarılır,
    açık = 0 okta; eşik 0,5, yeniden eğitim yok. BalAcc model ve referans, fark ve GA (label_sensitivity.csv ile aynı biçim).

Bloklar: test_time / test_both gün, test_station istasyon (analyze_preds ile aynı). Eşikler (CLM/NWC r30) yalnız train'de.
Çıktılar: artifacts_v3/analysis/{contingency.csv, unverified_metar_clear.csv, label_ge6.csv}

    python src/revision_metrics.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import (BLOCK, ORDER, balacc_from, block_counts, boot_weights, by_exp, counts,  # noqa: E402
                           load_meta, load_runs, reference_preds)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = ("test_time", "test_station", "test_both")
REF_MODELS = ("B_full", "D_unet", "D_unet_jit")
KEEP_EXPS = ("B_full", "A_sat", "C_noawos", "C_t2m", "D_unet")      # preds_extra (P9) bu deneylerde var


class _Split(dict):
    @property
    def files(self):
        return list(self.keys())


def scores(c):
    """c [..., 4] = tp(a), fn(c), tn(d), fp(b) — analyze_preds.counts sırası."""
    a, cc, d, b = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        pod = a / (a + cc)
        pofd = b / (b + d)
        n = a + b + cc + d
        exp = ((a + cc) * (a + b) + (cc + d) * (b + d)) / n
        return {"POD": pod, "POD_açık": d / (b + d), "POFD": pofd, "FAR": b / (a + b), "bias": (a + b) / (a + cc),
                "CSI": a / (a + b + cc), "KSS": pod - pofd, "HSS": (a + d - exp) / (n - exp), "BalAcc": balacc_from(c)}


def seed_probs(E, exps, d):
    """{deney: [ {satır: olasılık} (seed başına) ]} — verify test kümeleri + (varsa) P9 keep ekleri."""
    P = {}
    for e in exps:
        if e not in E:
            continue
        P[e] = []
        for r in sorted(E[e], key=lambda r: r["seed"]):
            prob = {}
            for s in SETS:
                if s in r["sets"]:
                    rr, _, pp = r["sets"][s]
                    prob.update(zip(rr.tolist(), pp.tolist()))
            pf = next((os.path.join(d, "preds", f) for f in os.listdir(os.path.join(d, "preds"))
                       if f.startswith(f"{e}_s{r['seed']}__") and f.endswith(".npz")), None)
            xf = pf and os.path.join(d, "preds_extra", os.path.basename(pf).replace(".npz", "__keep_extra.npz"))
            if xf and os.path.exists(xf):
                zx = np.load(xf)
                for s in SETS:
                    prob.update(zip(zx[f"{s}_rows"].tolist(), zx[f"{s}_prob"].tolist()))
            P[e].append(prob)
    return P


def preds_of(P, e, rows):
    out = []
    for prob in P[e]:
        p = np.array([prob.get(int(r), np.nan) for r in rows])
        if np.isnan(p).any():
            return None
        out.append((p >= 0.5).astype(int))
    return out


def contingency_rows(name, preds_list, y, codes, nb, W, base):
    """preds_list: seed başına ikili tahmin listesi (referans için tek eleman)."""
    C = np.mean([counts(p, y) for p in preds_list], axis=0)
    S = {k: float(np.nanmean([scores(counts(p, y))[k] for p in preds_list])) for k in scores(C)}
    bs = [scores(W @ block_counts(p, y, codes, nb)) for p in preds_list]
    row = dict(base, ürün=name, seed=len(preds_list), a_isabet=C[0], b_yanlış_alarm=C[3], c_kaçırma=C[1], d_doğru_açık=C[2], **S)
    for k in ("KSS", "HSS"):
        bm = np.mean([b[k] for b in bs], axis=0)
        row[f"{k}_GA_alt"], row[f"{k}_GA_üst"] = np.percentile(bm, [2.5, 97.5])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(ROOT, "artifacts_v3"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--stations", default=os.path.join(ROOT, "data/raw_observation/station_list.csv"))
    ap.add_argument("--split", default=os.path.join(ROOT, "splits/split_v2_verify.npz"))
    ap.add_argument("--split_keep", default=os.path.join(ROOT, "splits/split_v1_compact.npz"))
    ap.add_argument("--clm_dir", default=os.path.join(ROOT, "data/processed/clm_station"))
    ap.add_argument("--nwc_dir", default=os.path.join(ROOT, "data/processed/nwcsaf_station"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/analysis"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    meta = load_meta(a.meta, None, a.stations)
    sp, kp = np.load(a.split), np.load(a.split_keep)
    E = by_exp(load_runs(a.dir))
    models = [e for e in ORDER if e in E and not e.startswith(("logreg", "rf", "hgb_c", "hgb_aux"))]
    P = seed_probs(E, models, a.dir)
    unver = {s: np.setdiff1d(kp[s], sp[s]) for s in SETS}
    all_rows = np.unique(np.concatenate([sp[s] for s in SETS] + list(unver.values())))
    refs, info = reference_preds(meta, _Split(train=sp["train"], test_time=all_rows), a.clm_dir, a.nwc_dir)
    print("Referans eşikleri (train):", {k: v["eşik"] for k, v in info.items()})

    # ---------------------------------------------------------------- R6
    R6 = []
    for s in SETS:
        rows0 = np.sort(sp[s]); y0 = meta.label.values[rows0]
        rows0, y0 = rows0[y0 >= 0], y0[y0 >= 0]
        for lab in ("tümü", "SYNOP"):
            msk = np.ones(len(rows0), bool) if lab == "tümü" else meta.src.values[rows0] == "SYNOP"
            for samp in ["tüm satırlar"] + list(refs):
                m = msk if samp == "tüm satırlar" else msk & np.isfinite(refs[samp][rows0])
                rows, y = rows0[m], y0[m]
                codes, uni = pd.factorize(meta[BLOCK[s]].values[rows])
                W = boot_weights(len(uni), a.boot, rng)
                base = {"küme": s, "etiket": lab, "örneklem": samp, "n": int(m.sum()), "blok": len(uni)}
                names = models if samp == "tüm satırlar" else [e for e in REF_MODELS if e in P]
                for e in names:
                    pl = preds_of(P, e, rows)
                    if pl is not None:
                        R6.append(contingency_rows(e, pl, y, codes, len(uni), W, base))
                if samp != "tüm satırlar":
                    R6.append(contingency_rows(samp, [refs[samp][rows].astype(int)], y, codes, len(uni), W, base))
    R6 = pd.DataFrame(R6)
    R6.to_csv(os.path.join(a.out, "contingency.csv"), index=False, encoding="utf-8")

    # ---------------------------------------------------------------- R17
    R17 = []
    for s in SETS:
        rows = unver[s]
        assert (meta.label.values[rows] == 0).all() and (meta.src.values[rows] == "METAR").all(), "beklenmeyen satır"
        for samp in ["tüm satırlar"] + list(refs):
            m = np.ones(len(rows), bool) if samp == "tüm satırlar" else np.isfinite(refs[samp][rows])
            rr = rows[m]
            codes, uni = pd.factorize(meta[BLOCK[s]].values[rr])
            W = boot_weights(len(uni), a.boot, rng)
            names = [e for e in KEEP_EXPS if e in P] if samp == "tüm satırlar" else ["B_full", "D_unet"]
            items = [(e, preds_of(P, e, rr)) for e in names] + ([] if samp == "tüm satırlar"
                                                                 else [(samp, [refs[samp][rr].astype(int)])])
            for name, pl in items:
                if pl is None:
                    continue
                spec = [float((p == 0).mean()) for p in pl]
                bs = np.mean([np.bincount(codes, weights=(p == 0).astype(float), minlength=len(uni)) for p in pl], axis=0)
                nb = np.bincount(codes, minlength=len(uni)).astype(float)
                boot = (W @ bs) / (W @ nb)
                R17.append({"küme": s, "örneklem": samp, "ürün": name, "n": int(m.sum()), "blok": len(uni), "seed": len(pl),
                            "doğru_açık_oranı": float(np.mean(spec)), "GA_alt": np.percentile(boot, 2.5),
                            "GA_üst": np.percentile(boot, 97.5)})
    R17 = pd.DataFrame(R17)
    R17.to_csv(os.path.join(a.out, "unverified_metar_clear.csv"), index=False, encoding="utf-8")

    # ---------------------------------------------------------------- R20
    R20 = []
    exps20 = [e for e in ("B_full", "A_sat", "C_noawos", "C_t2m", "D_unet", "D_unet_jit") if e in P]
    for s in SETS:
        rows0 = np.sort(sp[s])
        y_all = meta.label.values[rows0]
        m0 = (y_all >= 0) & (meta.src.values[rows0] == "SYNOP") & (meta.okta.values[rows0] != 5)
        rows0, y0 = rows0[m0], y_all[m0]
        for ref in [None] + list(refs):
            m = np.ones(len(rows0), bool) if ref is None else np.isfinite(refs[ref][rows0])
            rows, y = rows0[m], y0[m]
            codes, uni = pd.factorize(meta[BLOCK[s]].values[rows])
            W = boot_weights(len(uni), a.boot, rng)
            row = {"tanım": "SYNOP, ≥6 okta bulutlu (5 okta çıkarıldı)", "küme": s, "referans": ref or "-", "n": int(m.sum()),
                   "n_açık": int((y == 0).sum()), "n_bulutlu": int((y == 1).sum()), "n_blok": len(uni)}
            if ref is not None:
                rp = refs[ref][rows].astype(int)
                row["ref_BalAcc"] = float(balacc_from(counts(rp, y)))
                rb = balacc_from(W @ block_counts(rp, y, codes, len(uni)))
            for e in exps20:
                pl = preds_of(P, e, rows)
                pts = [float(balacc_from(counts(p, y))) for p in pl]
                bm = np.mean([balacc_from(W @ block_counts(p, y, codes, len(uni))) for p in pl], axis=0)
                row[e] = float(np.mean(pts))
                if ref is None:
                    row[f"{e}_GA"] = f"[{np.percentile(bm, 2.5):.4f}, {np.percentile(bm, 97.5):.4f}]"
                else:
                    d = bm - rb
                    row[f"{e}−ref"] = row[e] - row["ref_BalAcc"]
                    row[f"{e}−ref_GA"] = f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
            R20.append(row)
    R20 = pd.DataFrame(R20)
    R20.to_csv(os.path.join(a.out, "label_ge6.csv"), index=False, encoding="utf-8")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    q = R6[(R6.küme == "test_time") & (R6.etiket == "SYNOP")]
    print(q[["örneklem", "ürün", "n", "POD", "FAR", "bias", "CSI", "KSS", "KSS_GA_alt", "KSS_GA_üst", "HSS"]].round(4).to_string(index=False))
    print(R17.round(4).to_string(index=False))
    print(R20[[c for c in ("küme", "referans", "n", "n_açık", "n_bulutlu", "ref_BalAcc", "B_full", "B_full−ref", "B_full−ref_GA",
                           "D_unet−ref", "D_unet−ref_GA") if c in R20]].round(4).to_string(index=False))
    print("Yazıldı:", ", ".join(os.path.join(a.out, f) for f in ("contingency.csv", "unverified_metar_clear.csv", "label_ge6.csv")))


if __name__ == "__main__":
    main()
