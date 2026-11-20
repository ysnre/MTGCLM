"""
analyze_preds.py — v3 örnek bazlı tahmin analizi (K18, 23 Eki 2026)
====================================================================
run_all.py koşularının kaydettiği örnek bazlı olasılıklardan (<dir>/preds/*.npz) makale için gereken
istatistikleri üretir. Eğitim gerekmez; yalnız numpy + pandas (+ isteğe bağlı matplotlib).

Analizler
  1) Ana tablo: deney x test kümesi — BalAcc, MCC (seed ort. ± s.s.), %95 blok-bootstrap güven aralığı,
     Brier, ECE; hem tüm etiketler hem yalnız SYNOP. Son epoch ile en iyi (val) model farkı.
  2) Eşleştirilmiş karşılaştırmalar (--pairs, "B-A" = B eksi A): fark, %95 GA ve iki yönlü p (blok bootstrap,
     aynı örnekler); McNemar p (seed başına, kümelenmeyi yok sayar — yalnız ek bilgi).
     Blok: test_time ve test_both -> gün, test_station -> istasyon (gözlemler bağımsız değil).
  3) CV (CV_<v>_f<k>): kat bazında BalAcc, kat bazında eşleştirilmiş farklar ve tüm katların test_time
     örnekleri birleştirilerek gün bloklu bootstrap.
  4) Kırılımlar (test_time, test_station): gündüz / alacakaranlık / gece (güneş yüksekliği istasyon konumu
     ve zamandan), etiket kaynağı (SYNOP/METAR), istasyon yüksekliği, ay; bulutlu etiketlerde okta başına
     yakalama oranı.
  5) --refs: FCI CLM ve NWC SAF CT (merkez piksel + 30 km alan; eşik YALNIZ train'de seçilir) ile aynı,
     kapsanan örneklerde eşleştirilmiş kıyas (özellikle yalnız SYNOP). data/processed/{clm_station,
     nwcsaf_station} gerekir (extract_fci_clm.py, extract_nwcsaf_ct.py çıktıları).

Kullanım (proje kökünden):
    python src/analyze_preds.py --dir artifacts_v3 --out artifacts_v3/analysis
    python src/analyze_preds.py --dir artifacts_v3 --out artifacts_v3/analysis --refs
Meta: --meta data/processed/h5meta_train_v2.npz (varsa) ya da --h5 <kompakt H5>.
Çıktı: <out>/analysis.md, analysis.json, *.csv, forest_*.png, reliability_test_time.png
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):           # Windows cp1254 borusu (K17)
    try:
        _stream.reconfigure(errors="backslashreplace")
    except Exception:
        pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RUN_RE = re.compile(r"^(?P<exp>.+?)(?:_f(?P<fold>\d+))?_s(?P<seed>\d+)$")
SUBSETS = ["test_time", "test_station", "test_both"]
PROB_BASELINES = ("logreg_aux", "rf_aux", "hgb_aux")       # olasılık üreten klasik modeller (v3)


def is_prob_baseline(name):
    """Olasılık üreten klasik modeller: logreg_/rf_/hgb_ önekli adlar (P2: hgb_center1, hgb_texture, ...)."""
    return name in PROB_BASELINES or name.split("_")[0] in ("logreg", "rf", "hgb")
ORDER = ["A_sat", "B_full", "C_noawos", "C_t2m", "D_unet", "RESNET", "RESNET_CIFAR",
         "hgb_aux", "rf_aux", "logreg_aux",
         # P1/P2 (15 Kas 2026) — sonda: mevcut tablo sırası değişmez
         "A_norad", "B_norad", "B_full_shuf", "hgb_center1", "hgb_center5", "hgb_current", "hgb_texture",
         # P20 (19 Kas 2026) — sonda
         "D_unet_jit",
         # R13 ADIM 3 (20 Kas 2026) — sonda
         "B_full_trainnorm"]
LABEL = {"A_sat": "A_sat (uydu+radar)", "B_full": "B_full (+tüm aux)", "C_noawos": "C_noawos (aux, AWOS'suz)",
         "C_t2m": "C_t2m (C + t2m)", "D_unet": "U-Net", "RESNET": "ResNet18 (ImageNet girişi)",
         "RESNET_CIFAR": "ResNet18 (CIFAR girişi)", "hgb_aux": "HGB", "rf_aux": "Rastgele orman",
         "logreg_aux": "Lojistik regresyon", "A_norad": "A_norad (yalnız uydu)", "B_norad": "B_norad (uydu + aux, radarsız)",
         "B_full_shuf": "B_full (uzamsal karıştırma)", "hgb_center1": "HGB (merkez piksel)", "hgb_center5": "HGB (merkez 5x5)",
         "hgb_current": "HGB (mevcut öznitelikler)", "hgb_texture": "HGB (+ doku)",
         "D_unet_jit": "U-Net (P20 merkez jitter)", "B_full_trainnorm": "B_full (eğitim kümesi z-skor sabitleri)"}
DEFAULT_PAIRS = ("B_full-A_sat,C_noawos-A_sat,B_full-C_noawos,C_t2m-C_noawos,"
                 "D_unet-A_sat,RESNET-A_sat,RESNET_CIFAR-A_sat,D_unet-RESNET_CIFAR,hgb_aux-A_sat,hgb_aux-B_full,"
                 # P1: radarın katkısı (aux yokken / aux varken); P2: uzamsal yapı ve doku
                 "A_sat-A_norad,B_full-B_norad,B_full-B_full_shuf,hgb_texture-hgb_current,hgb_current-hgb_center1,"
                 "hgb_center5-hgb_center1,B_full-hgb_texture,"
                 # P20: jitter'lı U-Net, merkez denetimli U-Net'e karşı (ön-kayıt ölçütü ii)
                 "D_unet_jit-D_unet")
CV_PAIRS = "B-A,C-A,B-C,Ct2m-C,B-Bnorad"
GROUPS = ["c1", "3x3", "r5", "r10", "r20", "r30"]
BLOCK = {"test_time": "day", "test_both": "day", "test_station": "wmoid"}
LEVEL_ORDER = {"illum": ["gündüz", "alacakaranlık", "gece"], "elev_bin": ["<250 m", "250–750 m", "750–1250 m", "≥1250 m"],
               "src": ["SYNOP", "METAR"], "açık etiket": ["SYNOP", "METAR"], "radar": ["kapsam içi", "kapsam dışı"]}


# ----------------------------------------------------------------------------- metrikler
def counts(pred, y):
    return np.array([((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 1)).sum(),
                     ((pred == 0) & (y == 0)).sum(), ((pred == 1) & (y == 0)).sum()], dtype=np.float64)


def balacc_from(c):
    """c [..., 4] = tp, fn, tn, fp"""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (c[..., 0] / (c[..., 0] + c[..., 1]) + c[..., 2] / (c[..., 2] + c[..., 3]))


def mcc_from(c):
    tp, fn, tn, fp = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (tp * tn - fp * fn) / den, np.nan)


def block_counts(pred, y, blk, nb):
    out = np.zeros((nb, 4))
    for j, m in enumerate([(pred == 1) & (y == 1), (pred == 0) & (y == 1), (pred == 0) & (y == 0), (pred == 1) & (y == 0)]):
        out[:, j] = np.bincount(blk, weights=m.astype(np.float64), minlength=nb)
    return out


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece(p, y, bins=15):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(idx, minlength=bins)
    conf = np.bincount(idx, weights=p, minlength=bins)
    acc = np.bincount(idx, weights=y.astype(float), minlength=bins)
    m = n > 0
    return float(np.sum(np.abs(acc[m] - conf[m])) / len(p))


def mcnemar_p(a_ok, b_ok):
    """İki yönlü McNemar: kesin binom (n<=200) ya da süreklilik düzeltmeli normal yaklaşım."""
    b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok)); n = b + c
    if n == 0:
        return 1.0, b, c
    if n <= 200:
        k = min(b, c)
        p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        return min(1.0, p), b, c
    z = (abs(b - c) - 1) / math.sqrt(n)
    return math.erfc(z / math.sqrt(2)), b, c


def boot_weights(nb, B, rng):
    W = np.zeros((B, nb))
    for i in range(B):
        W[i] = np.bincount(rng.integers(0, nb, nb), minlength=nb)
    return W


# ----------------------------------------------------------------------------- meta
def load_meta(meta_path, h5_path, station_csv):
    if meta_path and os.path.exists(meta_path):
        z = np.load(meta_path, allow_pickle=False)
        d = {k: z[k] for k in ("labels", "okta", "wmoids", "obs_time", "timestamps")}
    elif h5_path:
        import h5py
        with h5py.File(h5_path, "r") as f:
            d = {"labels": f["labels"][:], "okta": f["okta"][:], "wmoids": f["wmoids"][:], "obs_time": f["obs_time"][:],
                 "timestamps": np.array([t.decode() if isinstance(t, bytes) else str(t) for t in f["timestamps"][:]])}
    else:
        raise SystemExit("Meta için --meta (data/processed/h5meta_train_v2.npz) ya da --h5 verin.")
    ts = pd.to_datetime(pd.Series(d["timestamps"]).astype(str).str[:14], format="%Y%m%d%H%M%S")
    meta = pd.DataFrame({"label": d["labels"].astype(int), "okta": d["okta"].astype(int), "wmoid": d["wmoids"].astype(int)})
    meta["time"] = ts.values
    meta["day"] = ts.dt.strftime("%Y%m%d").values
    meta["month"] = ts.dt.strftime("%Y-%m").values
    meta["src"] = np.where((d["obs_time"] // 60) % 60 == 0, "SYNOP", "METAR")
    st = pd.read_csv(station_csv).dropna(subset=["wmoid", "latitude", "longitude"]).drop_duplicates("wmoid")
    st = st.assign(wmoid=st.wmoid.astype(int)).set_index("wmoid")
    meta["lat"] = meta.wmoid.map(st.latitude).astype(float)
    meta["lon"] = meta.wmoid.map(st.longitude).astype(float)
    meta["elev"] = meta.wmoid.map(st.elevation).astype(float)
    sel = solar_elevation(meta.time.values, meta.lat.values, meta.lon.values)
    meta["illum"] = np.select([sel >= 5, sel <= -5], ["gündüz", "gece"], "alacakaranlık")
    meta["elev_bin"] = pd.cut(meta.elev, [-1e9, 250, 750, 1250, 1e9],
                              labels=["<250 m", "250–750 m", "750–1250 m", "≥1250 m"]).astype(str)
    return meta


def radar_cov_center(path, h5_path, n):
    """İstasyon pikselindeki (16,16) radar_cov — radar kapsamı içi/dışı kırılımı için (P1). Önbellek .npy."""
    if path and os.path.exists(path):
        v = np.load(path)
        if len(v) == n:
            return v
        print(f"  {path}: satır sayısı uyuşmuyor ({len(v)} != {n}); yok sayıldı")
    if not h5_path:
        return None
    import h5py
    with h5py.File(h5_path, "r") as f:
        names = json.loads(f.attrs["aux_channels"])
        j = names.index("radar_cov")
        N = f["aux"].shape[0]
        if N != n:
            raise SystemExit(f"--aux_h5 satır sayısı {N} != meta {n}")
        v = np.empty(N, np.float32)
        for lo in range(0, N, 8192):
            v[lo:lo + 8192] = f["aux"][lo:lo + 8192, j, 16, 16]
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, v)
        print(f"  radar_cov merkez değerleri kaydedildi -> {path} (kapsam dışı payı {np.mean(v <= 0):.3f})")
    return v


def solar_elevation(t, lat, lon):
    """Güneş yükseklik açısı (derece); t UTC. NOAA yaklaşık formülü (±0,5°)."""
    t = pd.DatetimeIndex(t)
    doy = t.dayofyear.values
    hour = t.hour.values + t.minute.values / 60 + t.second.values / 3600
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    ha = np.radians((hour * 60 + eqt + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    cz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(cz, -1, 1)))


# ----------------------------------------------------------------------------- koşular
def load_runs(d):
    runs = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.startswith("run_manifest"):
            continue
        try:
            data = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print(f"  atlandı {stem}: {e}")
            continue
        m = RUN_RE.match(stem)
        fold = int(m.group("fold")) if m and m.group("fold") else None
        for name, res in data.items():
            if not isinstance(res, dict) or "config" not in res:
                continue
            cfg = res["config"]
            if cfg.get("baseline"):
                if not is_prob_baseline(name):
                    continue
                pf, exp, pref = os.path.join(d, "preds", stem + ".npz"), name, f"{name}__"
            else:
                pf, exp, pref = os.path.join(d, cfg.get("preds_file") or "-"), (m.group("exp") if m else stem), ""
            if not os.path.exists(pf):
                print(f"  tahmin dosyası yok: {pf}")
                continue
            z = np.load(pf)
            sets, last = {}, {}
            for s in ["val"] + SUBSETS:
                k = f"{pref}{s}_prob"
                if k in z.files and f"{s}_rows" in z.files:
                    sets[s] = (z[f"{s}_rows"].astype(np.int64), z[f"{s}_label"].astype(np.int64), z[k].astype(np.float64))
                if f"last_{s}_prob" in z.files and not pref:
                    last[s] = z[f"last_{s}_prob"].astype(np.float64)
            runs.append({"exp": exp, "seed": cfg.get("seed"), "fold": fold, "file": stem, "sets": sets, "last": last,
                         "env": cfg.get("env", {}), "split": cfg.get("split_file") or os.path.basename(str(cfg.get("split")))})
    return runs


def by_exp(runs):
    out = {}
    for r in runs:
        out.setdefault(r["exp"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: (r["fold"] if r["fold"] is not None else -1, r["seed"] or 0))
    return out


def common_rows(runs, s):
    rows = None
    for r in runs:
        if s not in r["sets"]:
            continue
        rr = r["sets"][s][0]
        rows = rr if rows is None else (rows if np.array_equal(rows, rr) else np.intersect1d(rows, rr))
    return rows


def aligned(run, s, rows):
    rr, y, p = run["sets"][s]
    if np.array_equal(rr, rows):
        return y, p
    pos = np.searchsorted(rr, rows)
    return y[pos], p[pos]


# ----------------------------------------------------------------------------- 1-2) ana tablo + çiftler
def main_analysis(E, meta, pairs, B, rng):
    rows_out, pair_out, boot_store = [], [], {}
    for s in SUBSETS:
        exps = [e for e in ORDER if e in E and any(s in r["sets"] for r in E[e])] + \
               [e for e in E if e not in ORDER and any(s in r["sets"] for r in E[e])]
        if not exps:
            continue
        rows = common_rows([r for e in exps for r in E[e]], s)
        y_true = meta.label.values[rows]
        blk_codes = pd.factorize(meta[BLOCK[s]].values[rows])[0]
        for lab, mask in (("tümü", np.ones(len(rows), bool)), ("SYNOP", meta.src.values[rows] == "SYNOP")):
            codes, uni = pd.factorize(blk_codes[mask])
            W = boot_weights(len(uni), B, rng)
            BA, PRED_OK = {}, {}
            for e in exps:
                per_seed, boots, okl, extra = [], [], [], []
                for r in E[e]:
                    if s not in r["sets"]:
                        continue
                    y, p = aligned(r, s, rows)
                    y, p = y[mask], p[mask]
                    pred = (p >= 0.5).astype(int)
                    c = counts(pred, y)
                    bc = block_counts(pred, y, codes, len(uni))
                    boots.append(balacc_from(W @ bc))
                    okl.append(pred == y)
                    lastba = np.nan
                    if s in r["last"]:
                        pl = (r["last"][s][np.searchsorted(r["sets"][s][0], rows)][mask] >= 0.5).astype(int)
                        lastba = float(balacc_from(counts(pl, y)))
                    per_seed.append((float(balacc_from(c)), float(mcc_from(c)), brier(p, y), ece(p, y), lastba))
                a = np.array(per_seed)
                BA[e] = np.mean(boots, axis=0); PRED_OK[e] = okl
                lo, hi = np.percentile(BA[e], [2.5, 97.5])
                rows_out.append({"küme": s, "etiket": lab, "deney": e, "n": int(mask.sum()), "seed": len(a),
                                 "BalAcc": a[:, 0].mean(), "BalAcc_ss": a[:, 0].std(ddof=1) if len(a) > 1 else np.nan,
                                 "BalAcc_GA_alt": lo, "BalAcc_GA_üst": hi, "MCC": a[:, 1].mean(),
                                 "Brier": a[:, 2].mean(), "ECE": a[:, 3].mean(), "BalAcc_son_epoch": np.nanmean(a[:, 4])
                                 if np.isfinite(a[:, 4]).any() else np.nan})
            boot_store[(s, lab)] = BA
            for pr in pairs:
                e2, e1 = pr.split("-", 1)
                if e1 not in BA or e2 not in BA:
                    continue
                d = BA[e2] - BA[e1]
                point = np.mean([x["BalAcc"] for x in rows_out if x["küme"] == s and x["etiket"] == lab and x["deney"] == e2]) - \
                        np.mean([x["BalAcc"] for x in rows_out if x["küme"] == s and x["etiket"] == lab and x["deney"] == e1])
                lo, hi = np.percentile(d, [2.5, 97.5])
                p_boot = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                mc = []
                o1, o2 = PRED_OK[e1], PRED_OK[e2]
                for i in range(max(len(o1), len(o2))):
                    mc.append(mcnemar_p(o1[min(i, len(o1) - 1)], o2[min(i, len(o2) - 1)])[0])
                pair_out.append({"küme": s, "etiket": lab, "çift": f"{e2} − {e1}", "fark": point, "GA_alt": lo, "GA_üst": hi,
                                 "p_bootstrap": p_boot, "McNemar_p_seedler": ";".join(f"{x:.3g}" for x in mc),
                                 "blok": BLOCK[s], "blok_sayısı": len(uni)})
    return pd.DataFrame(rows_out), pd.DataFrame(pair_out)


# ----------------------------------------------------------------------------- 3) CV
def cv_analysis(E, meta, B, rng):
    var = {e[3:]: E[e] for e in E if e.startswith("CV_")}
    if not var:
        return pd.DataFrame(), pd.DataFrame()
    fold_rows, pooled = [], {}
    for v, runs in var.items():
        for r in runs:
            if "test_time" not in r["sets"]:
                continue
            rr, y, p = r["sets"]["test_time"]
            for lab, m in (("tümü", np.ones(len(rr), bool)), ("SYNOP", meta.src.values[rr] == "SYNOP")):
                fold_rows.append({"varyant": v, "kat": r["fold"], "etiket": lab, "n": int(m.sum()),
                                  "BalAcc": float(balacc_from(counts((p[m] >= 0.5).astype(int), y[m])))})
            pooled.setdefault(v, []).append((rr, y, p))
    F = pd.DataFrame(fold_rows)
    out = []
    for lab in ("tümü", "SYNOP"):
        allrows = np.unique(np.concatenate([x[0] for v in pooled for x in pooled[v]]))
        m_all = meta.src.values[allrows] == "SYNOP" if lab == "SYNOP" else np.ones(len(allrows), bool)
        keep = allrows[m_all]
        codes_all = pd.factorize(meta.day.values[keep])[0]
        nb = codes_all.max() + 1
        W = boot_weights(nb, B, rng)
        BA = {}
        for v, parts in pooled.items():
            rr = np.concatenate([x[0] for x in parts]); y = np.concatenate([x[1] for x in parts]); p = np.concatenate([x[2] for x in parts])
            o = np.argsort(rr); rr, y, p = rr[o], y[o], p[o]
            sel = np.isin(rr, keep)
            rr, y, p = rr[sel], y[sel], p[sel]
            codes = codes_all[np.searchsorted(keep, rr)]
            pred = (p >= 0.5).astype(int)
            BA[v] = (float(balacc_from(counts(pred, y))), balacc_from(W @ block_counts(pred, y, codes, nb)))
        for pr in CV_PAIRS.split(","):
            v2, v1 = pr.split("-")
            if v1 not in BA or v2 not in BA:
                continue
            d = BA[v2][1] - BA[v1][1]
            f2 = F[(F.varyant == v2) & (F.etiket == lab)].set_index("kat").BalAcc
            f1 = F[(F.varyant == v1) & (F.etiket == lab)].set_index("kat").BalAcc
            fd = (f2 - f1).dropna()
            out.append({"etiket": lab, "çift": f"{v2} − {v1}", "birleşik_fark": BA[v2][0] - BA[v1][0],
                        "GA_alt": np.percentile(d, 2.5), "GA_üst": np.percentile(d, 97.5),
                        "p_bootstrap": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())),
                        "kat_farkları": " ".join(f"{x:+.4f}" for x in fd.values), "pozitif_kat": f"{int((fd > 0).sum())}/{len(fd)}",
                        "gün_bloğu": int(nb)})
    return F, pd.DataFrame(out)


# ----------------------------------------------------------------------------- 4) kırılımlar
def breakdowns(E, meta):
    out = []
    for s in ("test_time", "test_station"):
        for e in [x for x in ORDER if x in E]:
            for r in E[e]:
                if s not in r["sets"]:
                    continue
                rr, y, p = r["sets"][s]
                pred = (p >= 0.5).astype(int)
                M = meta.iloc[rr]
                for fac in ("illum", "src", "elev_bin", "month") + (("radar", "illum_radar") if "radar" in M.columns else ()):
                    for lev, idx in M.groupby(fac).indices.items():
                        if len(idx) < 200:
                            continue
                        c = counts(pred[idx], y[idx])
                        out.append({"küme": s, "deney": e, "seed": r["seed"], "etken": fac, "düzey": str(lev), "n": len(idx),
                                    "değer": float(balacc_from(c)), "ölçüt": "BalAcc"})
                ok = M.okta.values
                for o in (5, 6, 7, 8, 9):
                    idx = np.where((y == 1) & (ok == o))[0]
                    if len(idx) >= 50:
                        out.append({"küme": s, "deney": e, "seed": r["seed"], "etken": "okta (bulutlu)", "düzey": str(o),
                                    "n": len(idx), "değer": float((pred[idx] == 1).mean()), "ölçüt": "yakalama"})
                for srcv in ("SYNOP", "METAR"):
                    idx = np.where((y == 0) & (M.src.values == srcv))[0]
                    if len(idx) >= 50:
                        out.append({"küme": s, "deney": e, "seed": r["seed"], "etken": "açık etiket", "düzey": srcv,
                                    "n": len(idx), "değer": float((pred[idx] == 0).mean()), "ölçüt": "özgüllük"})
    D = pd.DataFrame(out)
    if D.empty:
        return D
    return D.groupby(["küme", "deney", "etken", "düzey", "ölçüt"], as_index=False).agg(n=("n", "first"), değer=("değer", "mean"))


# ----------------------------------------------------------------------------- 5) referanslar
def load_nwc_for(keys, nwc_dir):
    keys = keys.assign(day=keys.slot.str[:8]); parts = []
    for day, g in keys.groupby("day"):
        p = os.path.join(nwc_dir, f"nwc_{day}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ri = {r: i for i, r in enumerate(z["slot"])}; si = {int(w): i for i, w in enumerate(z["wmoid"])}
        fi = g.slot.map(ri); wi = g.wmoid.map(si); ok = fi.notna() & wi.notna()
        if not ok.any():
            continue
        g = g[ok]; fi = fi[ok].astype(int).values; wi = wi[ok].astype(int).values
        okf = z["ok"][fi]
        c = z["counts"][fi, wi].astype(np.int32)
        tot = np.maximum(c.sum(-1), 1)
        d = {"row": g.index.values[okf], "ncls": z["center_class"][fi, wi][okf]}
        for k, gn in enumerate(GROUPS):
            d["n_" + gn] = (c[:, k, 5:16].sum(-1) / tot[:, k])[okf]
        parts.append(pd.DataFrame(d))
    return pd.concat(parts).set_index("row") if parts else pd.DataFrame()


def best_threshold(frac, y):
    best = (-1, 0.5)
    for th in np.arange(0.05, 1.0, 0.05):
        ba = float(balacc_from(counts((frac >= th).astype(int), y)))
        if ba > best[0]:
            best = (ba, round(float(th), 2))
    return best


def reference_preds(meta, split, clm_dir, nwc_dir):
    """Tüm satırlar için {ad: dizi(NaN = kapsam yok)}; alan eşikleri yalnız train'de seçilir."""
    from eval_clm_on_splits import load_clm_for
    meta = meta.copy()
    meta["rc_start"] = pd.DatetimeIndex(meta.time).floor("10min").strftime("%Y%m%d%H%M")
    meta["slot"] = (pd.DatetimeIndex(meta.time) - pd.Timedelta("9min")).round("15min").strftime("%Y%m%d%H%M")
    N = len(meta)
    refs, info = {k: np.full(N, np.nan) for k in ("CLM_merkez", "CLM_r30", "NWC_merkez", "NWC_r30")}, {}
    tr = meta.iloc[split["train"]]; tr = tr[tr.label >= 0]
    need = np.unique(np.concatenate([split[s] for s in SUBSETS if s in split.files]))
    te = meta.iloc[need]
    t0 = time.time()
    c_tr = load_clm_for(tr[["rc_start", "wmoid"]], clm_dir).join(tr[["label"]], how="inner")
    c_tr = c_tr[c_tr.cstate.isin([1, 2, 3, 8])]
    ba, th = best_threshold(c_tr.f_r30.values, c_tr.label.values)
    info["CLM_r30"] = {"eşik": th, "train_BalAcc": ba, "n_train": len(c_tr)}
    c_te = load_clm_for(te[["rc_start", "wmoid"]], clm_dir)
    c_te = c_te[c_te.cstate.isin([1, 2, 3, 8])]
    refs["CLM_merkez"][c_te.index.values] = c_te.cstate.isin([2, 3]).astype(float).values
    refs["CLM_r30"][c_te.index.values] = (c_te.f_r30.values >= th).astype(float)
    print(f"  CLM: eşik r30 >= {th} (train BalAcc {ba:.3f}); {time.time() - t0:.0f} s")
    t0 = time.time()
    n_tr = load_nwc_for(tr[["slot", "wmoid"]], nwc_dir).join(tr[["label"]], how="inner")
    n_tr = n_tr[n_tr.ncls != 255]
    ba, th = best_threshold(n_tr.n_r30.values, n_tr.label.values)
    info["NWC_r30"] = {"eşik": th, "train_BalAcc": ba, "n_train": len(n_tr)}
    n_te = load_nwc_for(te[["slot", "wmoid"]], nwc_dir)
    n_te = n_te[n_te.ncls != 255]
    refs["NWC_merkez"][n_te.index.values] = ((n_te.ncls >= 5) & (n_te.ncls <= 15)).astype(float).values
    refs["NWC_r30"][n_te.index.values] = (n_te.n_r30.values >= th).astype(float)
    print(f"  NWC: eşik r30 >= {th} (train BalAcc {ba:.3f}); {time.time() - t0:.0f} s")
    return refs, info


def reference_analysis(E, meta, refs, B, rng):
    out = []
    models = [e for e in ORDER if e in E]
    for s in SUBSETS:
        rows = common_rows([r for e in models for r in E[e]], s)
        if rows is None:
            continue
        for lab in ("tümü", "SYNOP"):
            base = np.ones(len(rows), bool) if lab == "tümü" else meta.src.values[rows] == "SYNOP"
            for ref, arr in refs.items():
                rp = arr[rows]
                m = base & np.isfinite(rp)
                if m.sum() < 200:
                    continue
                y = meta.label.values[rows][m]
                codes, uni = pd.factorize(meta[BLOCK[s]].values[rows][m])
                W = boot_weights(len(uni), B, rng)
                rpred = rp[m].astype(int)
                rb = balacc_from(W @ block_counts(rpred, y, codes, len(uni)))
                row = {"küme": s, "etiket": lab, "referans": ref, "n": int(m.sum()), "kapsam": float(m.sum() / base.sum()),
                       "ref_BalAcc": float(balacc_from(counts(rpred, y)))}
                for e in models:
                    pts, bts = [], []
                    for r in E[e]:
                        if s not in r["sets"]:
                            continue
                        _, p = aligned(r, s, rows)
                        pred = (p[m] >= 0.5).astype(int)
                        pts.append(float(balacc_from(counts(pred, y))))
                        bts.append(balacc_from(W @ block_counts(pred, y, codes, len(uni))))
                    d = np.mean(bts, axis=0) - rb
                    row[e] = float(np.mean(pts))
                    row[f"{e}_fark_GA"] = f"{np.mean(pts) - row['ref_BalAcc']:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
                out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- çıktı
def pfmt(p, B):
    return f"<{1 / B:.4f}" if p <= 0 else f"{p:.3f}"


def fmt(x, d=4):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def md_table(df, cols, headers=None):
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def plots(P, T, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  matplotlib yok; grafikler atlandı")
        return []
    files = []
    for s in SUBSETS:
        q = P[P.küme == s]
        if q.empty:
            continue
        pairs = list(dict.fromkeys(q["çift"]))
        fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(pairs) + 1.4))
        for k, (lab, col, off) in enumerate((("tümü", "#2a6fdb", -0.15), ("SYNOP", "#e07b39", 0.15))):
            qq = q[q.etiket == lab].set_index("çift").reindex(pairs)
            yy = np.arange(len(pairs)) + off
            ax.errorbar(qq.fark * 100, yy, xerr=[(qq.fark - qq.GA_alt) * 100, (qq.GA_üst - qq.fark) * 100],
                        fmt="o", color=col, ms=4, capsize=2, label="tüm etiketler" if lab == "tümü" else "yalnız SYNOP")
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_yticks(np.arange(len(pairs))); ax.set_yticklabels(pairs); ax.invert_yaxis()
        ax.set_xlabel("BalAcc farkı (puan, %95 blok-bootstrap GA)"); ax.set_title(s)
        ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
        f = os.path.join(out, f"forest_{s}.png"); fig.savefig(f, dpi=150); plt.close(fig); files.append(f)
    return files


def reliability(E, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8)
    for e in [x for x in ORDER if x in E][:8]:
        ps, ys = [], []
        for r in E[e]:
            if "test_time" in r["sets"]:
                _, y, p = r["sets"]["test_time"]; ps.append(p); ys.append(y)
        if not ps:
            continue
        p = np.concatenate(ps); y = np.concatenate(ys)
        bins = np.linspace(0, 1, 11); idx = np.minimum((p * 10).astype(int), 9)
        n = np.bincount(idx, minlength=10); m = n > 0
        ax.plot((np.bincount(idx, weights=p, minlength=10)[m] / n[m]), (np.bincount(idx, weights=y, minlength=10)[m] / n[m]),
                marker="o", ms=3, lw=1, label=e)
    ax.set_xlabel("tahmin edilen P(bulutlu)"); ax.set_ylabel("gözlenen bulutlu oranı"); ax.set_title("Güvenilirlik — test_time")
    ax.legend(fontsize=7); fig.tight_layout()
    f = os.path.join(out, "reliability_test_time.png"); fig.savefig(f, dpi=150); plt.close(fig)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--out", default=None, help="Varsayılan: <dir>/analysis")
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--h5", default=None, help="--meta yoksa meta buradan okunur")
    ap.add_argument("--radar_cov", default="data/processed/radar_cov_center.npy",
                    help="P1: istasyon pikselindeki radar_cov (satır başına). Yoksa ve --aux_h5 verilmişse üretilip kaydedilir.")
    ap.add_argument("--aux_h5", default=None, help="radar_cov'u okumak için kompakt H5 (yalnız bir kez gerekir)")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--split", default="splits/split_v2_verify.npz", help="--refs eşik seçimi için train satırları")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refs", action="store_true", help="CLM/NWC ile aynı örneklerde kıyas")
    ap.add_argument("--clm_dir", default="data/processed/clm_station")
    ap.add_argument("--nwc_dir", default="data/processed/nwcsaf_station")
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    meta = load_meta(a.meta, a.h5, a.stations)
    rc = radar_cov_center(a.radar_cov, a.aux_h5, len(meta))
    if rc is not None:
        meta["radar"] = np.where(rc > 0, "kapsam içi", "kapsam dışı")
        meta["illum_radar"] = meta["illum"] + " / " + meta["radar"]
    runs = load_runs(a.dir)
    E_all = by_exp(runs)
    E = {e: v for e, v in E_all.items() if not e.startswith("CV_")}
    print(f"{len(runs)} koşu: " + ", ".join(f"{e}×{len(v)}" for e, v in E_all.items()))
    pairs = [p.strip() for p in a.pairs.split(",") if p.strip()]

    T, P = main_analysis(E, meta, pairs, a.boot, rng)
    print(f"  ana tablo + çiftler: {time.time() - t0:.0f} s")
    F, C = cv_analysis(E_all, meta, a.boot, rng)
    D = breakdowns(E, meta)
    R, rinfo = pd.DataFrame(), {}
    if a.refs:
        split = np.load(a.split)
        refs, rinfo = reference_preds(meta, split, a.clm_dir, a.nwc_dir)
        R = reference_analysis(E, meta, refs, a.boot, rng)

    for name, df in (("main_table", T), ("pairs", P), ("cv_folds", F), ("cv_pairs", C), ("breakdowns", D), ("references", R)):
        if not df.empty:
            df.to_csv(os.path.join(out, f"{name}.csv"), index=False, encoding="utf-8")
    figs = plots(P, T, out) if not P.empty else []
    rel = reliability(E, out)

    # ---- Markdown rapor
    envs = sorted({(r["env"].get("device"), r["env"].get("git_commit")) for r in runs if r["env"]})
    L = [f"# MTGCLM v3 — örnek bazlı analiz", "",
         f"Üretildi: {time.strftime('%Y-%m-%d %H:%M')} · kaynak `{a.dir}` · {len(runs)} koşu · bootstrap {a.boot} tekrar",
         f"Cihaz/kod: " + "; ".join(f"{d} @ {g}" for d, g in envs), "",
         "Karar eşiği 0,5. GA: %95 blok-bootstrap (test_time/test_both gün, test_station istasyon blokları; seed'ler aynı "
         "örneklemde ortalanır). p_bootstrap iki yönlü. McNemar kümelenmeyi yok sayar, yalnız ek bilgi.", ""]
    for s in SUBSETS:
        q = T[T.küme == s]
        if q.empty:
            continue
        L += [f"## 1. {s}", ""]
        for lab in ("tümü", "SYNOP"):
            qq = q[q.etiket == lab].copy()
            qq["BalAcc ± s.s."] = [f"{b:.4f} ± {sd:.4f}" if np.isfinite(sd) else f"{b:.4f}" for b, sd in zip(qq.BalAcc, qq.BalAcc_ss)]
            qq["%95 GA"] = [f"[{l:.4f}, {h:.4f}]" for l, h in zip(qq.BalAcc_GA_alt, qq.BalAcc_GA_üst)]
            qq["deney"] = [LABEL.get(e, e) for e in qq.deney]
            L += [f"**{'Tüm etiketler' if lab == 'tümü' else 'Yalnız SYNOP etiketleri'}** (n = {int(qq.n.iloc[0])})", "",
                  md_table(qq, ["deney", "seed", "BalAcc ± s.s.", "%95 GA", "MCC", "Brier", "ECE", "BalAcc_son_epoch"],
                           ["Deney", "Seed", "BalAcc ± s.s.", "%95 GA", "MCC", "Brier", "ECE", "Son epoch BalAcc"]), ""]
    if not P.empty:
        L += ["## 2. Eşleştirilmiş karşılaştırmalar (fark = ilk − ikinci, BalAcc)", ""]
        PP = P.copy()
        PP["fark [%95 GA]"] = [f"{f:+.4f} [{l:+.4f}, {h:+.4f}]" for f, l, h in zip(PP.fark, PP.GA_alt, PP.GA_üst)]
        PP["p"] = [pfmt(x, a.boot) for x in PP.p_bootstrap]
        PP["anlamlı"] = ["evet" if (l > 0 or h < 0) else "" for l, h in zip(PP.GA_alt, PP.GA_üst)]
        L += [md_table(PP, ["küme", "etiket", "çift", "fark [%95 GA]", "p", "anlamlı", "blok_sayısı"],
                       ["Küme", "Etiket", "Çift", "Fark [%95 GA]", "p (bootstrap)", "GA sıfırı dışlıyor", "Blok"]), "",
              "McNemar p değerleri (seed başına) `pairs.csv`'de; örnekler istasyon/gün içinde bağımlı olduğu için "
              "McNemar aşırı iyimserdir, çıkarım blok-bootstrap'a dayanmalıdır.", ""]
        L += [f"![{os.path.basename(f)}]({os.path.basename(f)})" for f in figs] + [""]
    if not C.empty:
        L += ["## 3. Bloklu 5-katlı CV (test_time)", ""]
        piv = F[F.etiket == "tümü"].pivot(index="varyant", columns="kat", values="BalAcc")
        piv["ort"] = piv.mean(axis=1); piv["s.s."] = piv.iloc[:, :-1].std(axis=1, ddof=1)
        L += ["Kat bazında BalAcc (tüm etiketler):", "", md_table(piv.reset_index(), ["varyant"] + list(piv.columns),
                                                                  ["Varyant"] + [f"kat {c}" if isinstance(c, (int, np.integer)) else c for c in piv.columns]), ""]
        CC = C.copy()
        CC["birleşik fark [%95 GA]"] = [f"{f:+.4f} [{l:+.4f}, {h:+.4f}]" for f, l, h in zip(CC.birleşik_fark, CC.GA_alt, CC.GA_üst)]
        CC["p"] = [pfmt(x, a.boot) for x in CC.p_bootstrap]
        L += ["Eşleştirilmiş farklar (birleşik: 5 katın test_time örnekleri birlikte, gün blokları):", "",
              md_table(CC, ["etiket", "çift", "birleşik fark [%95 GA]", "p", "kat_farkları", "pozitif_kat"],
                       ["Etiket", "Çift", "Birleşik fark [%95 GA]", "p", "Kat farkları", "Pozitif kat"]), ""]
    if not D.empty:
        L += ["## 4. Kırılımlar (seed ortalaması)", ""]
        for s in ("test_time", "test_station"):
            for fac, title in (("illum", "Aydınlanma — BalAcc"), ("src", "Etiket kaynağı — BalAcc"),
                               ("elev_bin", "İstasyon yüksekliği — BalAcc"), ("okta (bulutlu)", "Bulutlu etiketlerde okta — yakalama oranı"),
                               ("açık etiket", "Açık etiketlerde kaynak — özgüllük"), ("month", "Ay — BalAcc")):
                q = D[(D.küme == s) & (D.etken == fac)]
                if q.empty:
                    continue
                pv = q.pivot(index="deney", columns="düzey", values="değer").reindex([e for e in ORDER if e in set(q.deney)])
                pref = LEVEL_ORDER.get(fac)
                if pref:
                    pv = pv[[c for c in pref if c in pv.columns] + [c for c in pv.columns if c not in pref]]
                nn = q.groupby("düzey").n.first()
                pv.columns = [f"{c} (n={nn[c]})" for c in pv.columns]
                L += [f"**{s} — {title}**", "", md_table(pv.reset_index(), ["deney"] + list(pv.columns), ["Deney"] + list(pv.columns)), ""]
    if not R.empty:
        L += ["## 5. Operasyonel ürünlerle aynı örneklerde kıyas", "",
              "Eşikler yalnız train'de: " + "; ".join(f"{k} ≥ {v['eşik']} (train BalAcc {v['train_BalAcc']:.3f})" for k, v in rinfo.items()),
              "", "Fark sütunu: modelin BalAcc'ı − referansın BalAcc'ı, aynı (kapsanan) örnekler, [%95 GA].", ""]
        models = [e for e in ORDER if e in R.columns]
        cols = ["küme", "etiket", "referans", "n", "kapsam", "ref_BalAcc"] + models
        L += [md_table(R, cols, ["Küme", "Etiket", "Referans", "n", "Kapsam", "Referans BalAcc"] + models), ""]
        fc = [f"{e}_fark_GA" for e in ("A_sat", "C_noawos", "D_unet", "hgb_aux") if f"{e}_fark_GA" in R.columns]
        L += [md_table(R, ["küme", "etiket", "referans"] + fc, ["Küme", "Etiket", "Referans"] + [c.replace("_fark_GA", " − ref") for c in fc]), ""]
    if rel:
        L += [f"![güvenilirlik]({os.path.basename(rel)})", ""]
    open(os.path.join(out, "analysis.md"), "w", encoding="utf-8").write("\n".join(L))
    json.dump({"main_table": T.to_dict(orient="records"), "pairs": P.to_dict(orient="records"),
               "cv_folds": F.to_dict(orient="records"), "cv_pairs": C.to_dict(orient="records"),
               "references": R.to_dict(orient="records"), "reference_thresholds": rinfo},
              open(os.path.join(out, "analysis.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"Yazıldı: {out}/analysis.md (+ csv, json, png) — {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
