#!/usr/bin/env python
"""R10 — bulut sıklığı duyarlılığı: doğu yaylalarında CLM − model farkı (ön kayıt: docs/revizyon_onkayit.md, R10).

Model = ürün modeli D_unet_jit (3 tohum ortalaması, tek geçiş; batch_scenes.py çıktısı prob_u8 = round(250 p)).
Varyantlar (her biri aynı sahne–piksel geçerlilik maskesinde, yani model ve CLM birlikte geçerli — cloud_frequency.py "joint"):
  eşik 0,4 / 0,5 / 0,6;  val'de bias = 1 veren eşik (bulutlu tahmin oranı = val'deki bulutlu oranı);
  val'de izotonik kalibrasyon: (i) kalibre olasılık ≥ 0,5 (ikili) ve (ii) kalibre olasılığın ortalaması (beklenen sıklık);
  bölge maskesi: Türkiye maskesinin AWOS istasyonu yarıçapı 15 / 25 / 50 km (cloud_frequency.region_masks, diğer
  tanımlar aynı: kıyı ≤ 30 km deniz, doğu ≥ 38,5° D ve ≥ 1500 m).
Kural: "CLM karı bulut sayıyor" bulgu olarak kalır ancak fark (tüm dönem) TÜM varyantlarda > +0,03 ise; aksi halde hipotez.
Tanısal ek alt küme: kış gündüzü (Ara–Mar, 06–12 UTC).
Val tahminleri yalnız eşik/kalibrasyon seçimi için kullanılır (test'e bakılmaz).

Çıktı: artifacts_v3/climatology/cloud_freq_sensitivity.csv (+ _info.json)
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import by_exp, load_runs  # noqa: E402

RADII = [15.0, 25.0, 50.0]
FIXED = [0.4, 0.5, 0.6]
RULE = 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/cloud_freq_jit")
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--exp", default="D_unet_jit")
    ap.add_argument("--dem", default="data/dem_800x1000.npy")
    ap.add_argument("--awos_stations", default="data/raw_awos_observation/stationlist_awos.csv")
    ap.add_argument("--out", default="artifacts_v3/climatology")
    a = ap.parse_args()
    from sklearn.isotonic import IsotonicRegression
    import sklearn
    from cloud_frequency import region_masks
    from grid_utils import LatLonGrid

    # ---- 1) val'den eşik ve izotonik kalibrasyon
    E = by_exp(load_runs(a.dir))
    runs = [r for r in E[a.exp] if "val" in r["sets"]]
    rows0, y = runs[0]["sets"]["val"][0], runs[0]["sets"]["val"][1]
    for r in runs[1:]:
        assert np.array_equal(r["sets"]["val"][0], rows0)
    pv = np.mean([r["sets"]["val"][2] for r in runs], axis=0)
    m = y >= 0
    pv, y = pv[m], y[m]
    cand = np.arange(0, 251) / 250.0
    frac = np.array([(pv >= t).mean() for t in cand])
    t_bias = float(cand[np.argmin(np.abs(frac - y.mean()))])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, y)
    lut = iso.predict(cand)                     # prob_u8 → kalibre olasılık tablosu (0..250)
    t_iso = float(cand[np.argmax(lut >= 0.5)]) if (lut >= 0.5).any() else 1.0
    thr = {f"eşik {t:.1f}": t for t in FIXED}
    thr[f"bias=1 (val, eşik {t_bias:.3f})"] = t_bias
    thr[f"izotonik ≥ 0,5 (eşik {t_iso:.3f})"] = t_iso
    thr_u8 = {k: int(round(250 * t)) for k, t in thr.items()}
    print(f"val: n {len(y):,}, bulutlu oranı {y.mean():.3f}; bias=1 eşiği {t_bias:.3f}; izotonik 0,5 eşiği {t_iso:.3f}", flush=True)

    # ---- 2) bölge maskeleri
    grid = LatLonGrid.default_turkiye()
    dem = np.load(a.dem)
    masks = {r: region_masks(grid, dem, a.awos_stations, d_tr_km=r)["eastern_highlands"] for r in RADII}
    union = np.zeros(grid.shape, bool)
    for mm in masks.values():
        union |= mm
    yy, xx = np.nonzero(union)
    sub = {r: masks[r][yy, xx] for r in RADII}
    print("doğu yaylaları piksel sayısı: " + ", ".join(f"{int(r)} km → {int(v.sum())}" for r, v in masks.items()), flush=True)

    # ---- 3) sahneler
    files = sorted(glob.glob(os.path.join(a.src, "scenes", "cf_*.npz")))
    P = len(yy)
    groups = ("tüm dönem", "kış gündüzü (Ara–Mar, 06–12 UTC)")
    acc = {g: {"nj": np.zeros(P), "clm": np.zeros(P), "iso_mean": np.zeros(P),
               **{k: np.zeros(P) for k in thr}} for g in groups}
    nsc = {g: 0 for g in groups}
    t0 = time.time()
    for i, f in enumerate(files, 1):
        z = np.load(f)
        nom = str(z["nominal"])
        mon, hr = int(nom[4:6]), int(nom[8:10])
        pu8 = z["prob_u8"][yy, xx]; clm = z["clm"][yy, xx]
        j = (pu8 != 255) & (clm >= 0)
        if not j.any():
            continue
        gl = [groups[0]] + ([groups[1]] if mon in (12, 1, 2, 3) and 6 <= hr <= 12 else [])
        cal = lut[np.minimum(pu8, 250)]
        for g in gl:
            A = acc[g]
            A["nj"] += j
            A["clm"] += j & (clm == 1)
            A["iso_mean"] += np.where(j, cal, 0.0)
            for k, tu in thr_u8.items():
                A[k] += j & (pu8 >= tu)
            nsc[g] += 1
        if i % max(1, len(files) // 20) == 0 or i == len(files):
            el = time.time() - t0
            print(f"  sahne %{100 * i // len(files):3d} ({i}/{len(files)}, {el / 60:.1f} dk, kalan ~{el / i * (len(files) - i) / 60:.1f} dk)",
                  flush=True)

    # ---- 4) özet: piksel sıklıklarının bölge ortalaması (cloud_frequency.region_means ile aynı: nanmean of per-pixel frequency)
    out = []
    for g in groups:
        A = acc[g]
        with np.errstate(invalid="ignore", divide="ignore"):
            nj = np.where(A["nj"] > 0, A["nj"], np.nan)
            fc = A["clm"] / nj
            fm = {k: A[k] / nj for k in thr}
            fm["izotonik ortalama olasılık"] = A["iso_mean"] / nj
        for r in RADII:
            s = sub[r]
            for k, v in fm.items():
                clm_f, mod_f = float(np.nanmean(fc[s])), float(np.nanmean(v[s]))
                out.append({"alt_küme": g, "sahne": nsc[g], "yarıçap_km": r, "piksel": int(s.sum()), "varyant": k,
                            "freq_model": mod_f, "freq_clm": clm_f, "clm_eksi_model": clm_f - mod_f})
    R = pd.DataFrame(out)
    R["kural_eşiği_aşıyor"] = R.clm_eksi_model > RULE
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, "cloud_freq_sensitivity.csv"), index=False, float_format="%.4f")
    main_ = R[R.alt_küme == groups[0]]
    verdict = "bulgu" if bool(main_.kural_eşiği_aşıyor.all()) else "hipotez"
    info = {"exp": a.exp, "sklearn": sklearn.__version__, "val_n": int(len(y)), "val_cloudy_frac": float(y.mean()),
            "t_bias": t_bias, "t_iso": t_iso, "n_files": len(files), "scenes": nsc, "rule": RULE, "verdict": verdict,
            "min_diff_all_period": float(main_.clm_eksi_model.min()), "max_diff_all_period": float(main_.clm_eksi_model.max())}
    json.dump(info, open(os.path.join(a.out, "cloud_freq_sensitivity_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.set_option("display.width", 200)
    print(R.pivot_table(index=["alt_küme", "varyant"], columns="yarıçap_km", values="clm_eksi_model").round(3).to_string())
    print(f"\nKURAL (tüm dönem, tüm varyantlar > +{RULE}): {verdict}  "
          f"[en küçük {info['min_diff_all_period']:+.3f}, en büyük {info['max_diff_all_period']:+.3f}]")


if __name__ == "__main__":
    main()
