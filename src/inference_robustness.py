"""
inference_robustness.py — revizyon R8 (ön-kayıt: docs/revizyon_onkayit.md; yeniden eğitim yok)

R8a Çoklu karşılaştırma. Bootstrap p değerleri çözünürlükle sınırlı olduğundan p = 2·(k+1)/(B+1) (k = sıfırın öbür
    yanındaki tekrar sayısı, en çok 1). Aileler (ön-kayıtta tanımlı):
      Aile 1  model − referans farkları (B_full, D_unet × FCI CLM / NWC SAF CT merkez ve 30 km × 3 test kümesi × {tümü, SYNOP})
      Aile 2  tek bölme modalite / ablasyon / mimari farkları (artifacts_v3/analysis/pairs.csv)
      Aile 3  CV birleşik farkları (artifacts_v3/analysis/cv_pairs.csv)
    Aile içinde Benjamini–Hochberg (q = 0,05; birincil) ve Holm (duyarlılık).
R8b Blok uzunluğu duyarlılığı: örtüşen (moving) blok bootstrap, 1 / 2 / 3 / 5 gün, test_time ve test_both (tümü);
    Aile 1 ve Aile 2 farklarının GA'ları. Günler takvim sırasıyla dizilir; L günlük blokların başlangıcı rastgele seçilir,
    günler L'li dizilerle örneklenip gün sayısına kesilir (L = 1 olağan gün bootstrap'ı). test_station istasyon blokludur
    (bu analiz dışında).

Çıktılar: artifacts_v3/analysis/{multiplicity.csv, block_length_sensitivity.csv}

    python src/inference_robustness.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import (BLOCK, DEFAULT_PAIRS, aligned, balacc_from, block_counts, boot_weights, by_exp,  # noqa: E402
                           common_rows, counts, load_meta, load_runs, reference_preds)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = ("test_time", "test_station", "test_both")
REF_MODELS = ("B_full", "D_unet")
LENGTHS = (1, 2, 3, 5)


class _Split(dict):
    @property
    def files(self):
        return list(self.keys())


def p_boot(d, B):
    k = min((d <= 0).sum(), (d >= 0).sum())
    return min(1.0, 2 * (k + 1) / (B + 1))


def p_from_reported(p, B):
    """pairs.csv / cv_pairs.csv'deki p = 2·k/B -> 2·(k+1)/(B+1)."""
    k = round(p * B / 2)
    return min(1.0, 2 * (k + 1) / (B + 1))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); adj[o[i]] = prev
    return adj


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, p[j] * (n - i))); adj[j] = run
    return adj


def moving_block_weights(ndays, L, B, rng):
    """B × ndays ağırlık matrisi (her günün örneklemde kaç kez yer aldığı)."""
    W = np.zeros((B, ndays))
    nblk = int(np.ceil(ndays / L))
    starts_max = ndays - L
    for b in range(B):
        st = rng.integers(0, starts_max + 1, nblk)
        idx = (st[:, None] + np.arange(L)[None, :]).ravel()[:ndays]
        W[b] = np.bincount(idx, minlength=ndays)
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(ROOT, "artifacts_v3"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--stations", default=os.path.join(ROOT, "data/raw_observation/station_list.csv"))
    ap.add_argument("--split", default=os.path.join(ROOT, "splits/split_v2_verify.npz"))
    ap.add_argument("--clm_dir", default=os.path.join(ROOT, "data/processed/clm_station"))
    ap.add_argument("--nwc_dir", default=os.path.join(ROOT, "data/processed/nwcsaf_station"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/analysis"))
    a = ap.parse_args()
    B = a.boot
    rng = np.random.default_rng(a.seed)

    meta = load_meta(a.meta, None, a.stations)
    sp = np.load(a.split)
    E = {e: v for e, v in by_exp(load_runs(a.dir)).items() if not e.startswith("CV_")}
    refs, _ = reference_preds(meta, _Split(train=sp["train"], test_time=np.unique(np.concatenate([sp[s] for s in SETS]))),
                              a.clm_dir, a.nwc_dir)
    pairs = [p.strip() for p in DEFAULT_PAIRS.split(",") if p.strip()]

    def model_preds(e, s, rows):
        return [(aligned(r, s, rows)[1] >= 0.5).astype(int) for r in E[e] if s in r["sets"]]

    fam1, blk = [], []
    for s in SETS:
        rows_all = common_rows([r for e in E for r in E[e]], s)
        for lab in ("tümü", "SYNOP"):
            base = np.ones(len(rows_all), bool) if lab == "tümü" else meta.src.values[rows_all] == "SYNOP"
            # --- Aile 1: model − referans (olağan blok bootstrap, analyze_preds ile aynı bloklar)
            for ref, arr in refs.items():
                m = base & np.isfinite(arr[rows_all])
                rows = rows_all[m]; y = meta.label.values[rows]
                codes, uni = pd.factorize(meta[BLOCK[s]].values[rows])
                W = boot_weights(len(uni), B, rng)
                rp = arr[rows].astype(int)
                rb = balacc_from(W @ block_counts(rp, y, codes, len(uni)))
                for e in REF_MODELS:
                    pl = [p[m] for p in model_preds(e, s, rows_all)]
                    bm = np.mean([balacc_from(W @ block_counts(p, y, codes, len(uni))) for p in pl], axis=0)
                    d = bm - rb
                    pt = np.mean([balacc_from(counts(p, y)) for p in pl]) - balacc_from(counts(rp, y))
                    fam1.append({"aile": 1, "küme": s, "etiket": lab, "karşılaştırma": f"{e} − {ref}", "fark": pt,
                                 "GA_alt": np.percentile(d, 2.5), "GA_üst": np.percentile(d, 97.5), "p_ham": p_boot(d, B)})
            # --- R8b: blok uzunluğu (yalnız gün bloklu kümeler, tüm etiketler)
            if s == "test_station" or lab != "tümü":
                continue
            days_all = meta.day.values[rows_all]
            day_order = np.sort(np.unique(days_all))
            day_idx = np.searchsorted(day_order, days_all)
            nd = len(day_order)
            items = []
            for ref, arr in refs.items():
                m = np.isfinite(arr[rows_all])
                items += [(f"{e} − {ref}", m, model_preds(e, s, rows_all), [arr[rows_all].astype(int)]) for e in REF_MODELS]
            for pr in pairs:
                e2, e1 = pr.split("-", 1)
                if e1 in E and e2 in E:
                    items.append((f"{e2} − {e1}", np.ones(len(rows_all), bool), model_preds(e2, s, rows_all),
                                  model_preds(e1, s, rows_all)))
            for L in LENGTHS:
                W = moving_block_weights(nd, L, B, rng)
                for name, m, p2, p1 in items:
                    y = meta.label.values[rows_all][m]
                    di = day_idx[m]
                    b2 = np.mean([balacc_from(W @ block_counts(p[m], y, di, nd)) for p in p2], axis=0)
                    b1 = np.mean([balacc_from(W @ block_counts(p[m], y, di, nd)) for p in p1], axis=0)
                    d = b2 - b1
                    pt = np.mean([balacc_from(counts(p[m], y)) for p in p2]) - np.mean([balacc_from(counts(p[m], y)) for p in p1])
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    blk.append({"küme": s, "karşılaştırma": name, "aile": 1 if " − CLM" in name or " − NWC" in name else 2,
                                "blok_gün": L, "gün": nd, "fark": pt, "GA_alt": lo, "GA_üst": hi, "GA_genişlik": hi - lo,
                                "sıfırı_dışlıyor": bool(lo > 0 or hi < 0)})
        print(f"  {s} bitti")

    F1 = pd.DataFrame(fam1)
    P = pd.read_csv(os.path.join(a.dir, "analysis", "pairs.csv"))
    F2 = pd.DataFrame({"aile": 2, "küme": P["küme"], "etiket": P["etiket"], "karşılaştırma": P["çift"], "fark": P["fark"],
                       "GA_alt": P["GA_alt"], "GA_üst": P["GA_üst"], "p_ham": [p_from_reported(x, B) for x in P["p_bootstrap"]]})
    C = pd.read_csv(os.path.join(a.dir, "analysis", "cv_pairs.csv"))
    F3 = pd.DataFrame({"aile": 3, "küme": "CV (5 kat)", "etiket": C["etiket"], "karşılaştırma": C["çift"], "fark": C["birleşik_fark"],
                       "GA_alt": C["GA_alt"], "GA_üst": C["GA_üst"], "p_ham": [p_from_reported(x, B) for x in C["p_bootstrap"]]})
    M = []
    for F in (F1, F2, F3):
        F = F.copy()
        F["GA_sıfırı_dışlıyor"] = (F.GA_alt > 0) | (F.GA_üst < 0)
        F["p_BH"] = bh(F.p_ham.values); F["anlamlı_BH"] = F.p_BH < 0.05
        F["p_Holm"] = holm(F.p_ham.values); F["anlamlı_Holm"] = F.p_Holm < 0.05
        F["aile_boyu"] = len(F)
        M.append(F)
    M = pd.concat(M, ignore_index=True)
    M.to_csv(os.path.join(a.out, "multiplicity.csv"), index=False, encoding="utf-8")
    K = pd.DataFrame(blk)
    K.to_csv(os.path.join(a.out, "block_length_sensitivity.csv"), index=False, encoding="utf-8")

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    for f in (1, 2, 3):
        q = M[M.aile == f]
        print(f"Aile {f}: {len(q)} karşılaştırma | GA sıfırı dışlıyor {int(q.GA_sıfırı_dışlıyor.sum())} | "
              f"BH sonrası anlamlı {int(q.anlamlı_BH.sum())} | Holm sonrası {int(q.anlamlı_Holm.sum())}")
    lost = M[M.GA_sıfırı_dışlıyor & ~M.anlamlı_BH]
    print("GA sıfırı dışlayıp BH sonrası düşenler:")
    print(lost[["aile", "küme", "etiket", "karşılaştırma", "fark", "GA_alt", "GA_üst", "p_ham", "p_BH", "p_Holm"]].round(5).to_string(index=False))
    flip = K.groupby(["küme", "karşılaştırma"]).sıfırı_dışlıyor.agg(["min", "max"])
    print("Blok uzunluğuna duyarlı (bir L'de sıfırı dışlayıp başka L'de dışlamayan):")
    print(flip[flip["min"] != flip["max"]].to_string())
    print(K.groupby(["küme", "blok_gün"]).GA_genişlik.median().round(5).to_string())


if __name__ == "__main__":
    main()
