"""
error_analysis.py — P8: CNN (tüm girdiler, B_full s1–3) hatalarının gece / sis / alçak bulut / kar odaklı analizi. Yeni eğitim yok.

Tahmin: üç seed'in olasılık ortalaması >= 0,5 (topluluk). Kümeler: test_time, test_station (ayrı) ve ikisi birlikte (both).
Oranlar: hata = yanlış / n; kaçırma (FN) = bulutlu raporlarda "açık" tahmini; yanlış alarm (FP) = açık raporlarda "bulutlu".
Kırılımlar: güneş yüksekliği (5°), RH2m (istasyon pikseli, IDW), okta (0, 5–9), istasyon yüksekliği, ay, etiket kaynağı;
gece için NWC SAF CT merkez sınıfı (aynı 15 dk slot; bozuk kareler hariç).
Odak: gece kaçırmalarında alçak bulut / sis payı (NWC 5–6 çok alçak/alçak, okta 9, RH >= %90); kar: Ara–Mar, gündüz,
istasyon > 1500 m, açık raporlarda yanlış alarm oranı (karşılaştırma: aynı aylarda <= 1500 m ve Nis–Haz > 1500 m).
10 temsilî hata (tohum 0) için merkez piksel değerleri ve gözlem.

Girdi: preds/B_full_s*, h5meta, data/processed/center_values.npz (src/export_center_values.py; yoksa RH ve kanal sütunları boş),
data/processed/nwcsaf_station, data/processed/clm_station.
Çıktı: artifacts_v3/analysis/error_analysis.csv, error_cases.csv, error_summary.json

    python src/error_analysis.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import by_exp, load_meta, load_nwc_for, load_runs, solar_elevation  # noqa: E402
from eval_clm_on_splits import load_clm_for                                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = ("test_time", "test_station")


def bins(x, edges, labels=None):
    return pd.cut(x, edges, right=False, labels=labels).astype(str)


def rates(d):
    cl, cr = d["label"] == 1, d["label"] == 0
    return {"n": int(len(d)), "n_cloudy": int(cl.sum()), "n_clear": int(cr.sum()),
            "error": float((d["pred"] != d["label"]).mean()) if len(d) else np.nan,
            "miss_rate": float((d.loc[cl, "pred"] == 0).mean()) if cl.any() else np.nan,
            "false_alarm_rate": float((d.loc[cr, "pred"] == 1).mean()) if cr.any() else np.nan,
            "n_miss": int(((d["pred"] == 0) & cl).sum()), "n_false_alarm": int(((d["pred"] == 1) & cr).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(ROOT, "artifacts_v3"))
    ap.add_argument("--exp", default="B_full")
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--stations", default=os.path.join(ROOT, "data/raw_observation/station_list.csv"))
    ap.add_argument("--centers", default=os.path.join(ROOT, "data/processed/center_values.npz"))
    ap.add_argument("--nwc_dir", default=os.path.join(ROOT, "data/processed/nwcsaf_station"))
    ap.add_argument("--clm_dir", default=os.path.join(ROOT, "data/processed/clm_station"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/analysis"))
    ap.add_argument("--min_n", type=int, default=100)
    a = ap.parse_args()

    meta = load_meta(a.meta, None, a.stations)
    runs = by_exp(load_runs(a.dir))[a.exp]
    parts = []
    for s in SETS:
        rows = runs[0]["sets"][s][0]
        P = []
        for r in runs:
            rr, y, p = r["sets"][s]
            assert np.array_equal(rr, rows)
            P.append(p)
        d = meta.iloc[rows].copy()
        d["row"], d["set"] = rows, s
        d["prob"] = np.mean(P, axis=0)
        d["pred"] = (d["prob"] >= 0.5).astype(int)
        parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    D["sun_elev"] = solar_elevation(D["time"].values, D["lat"].values, D["lon"].values)
    D["mon"] = D["time"].dt.month
    if os.path.exists(a.centers):
        z = np.load(a.centers)
        bn, an = list(z["base_names"]), list(z["aux_names"])
        for k, c in (("t2m", "t2m"), ("rh2m", "rh2m")):
            D[k] = z["aux"][D["row"].values, an.index(c)]
        for c in ("vis_06", "nir_16", "ir_38", "ir_105", "ir_123", "wv_73"):
            D[c] = z["base"][D["row"].values, bn.index(c)]
        D["radar_max"] = z["radar_max"][D["row"].values]
        D["bt38_105"] = D["ir_38"] - D["ir_105"]
        D["bt105_t2m"] = D["ir_105"] - (D["t2m"] + 273.15)
    else:
        print(f"UYARI: {a.centers} yok — RH ve kanal kırılımları atlanıyor (önce export_center_values.py).")
    D["slot"] = (pd.DatetimeIndex(D["time"]) - pd.Timedelta("9min")).round("15min").strftime("%Y%m%d%H%M")
    nwc = load_nwc_for(D[["slot", "wmoid"]], a.nwc_dir)
    D["nwc"] = nwc["ncls"].reindex(D.index).fillna(255).astype(int).values
    D["rc_start"] = pd.DatetimeIndex(D["time"]).floor("10min").strftime("%Y%m%d%H%M")
    clm = load_clm_for(D[["rc_start", "wmoid"]], a.clm_dir)
    D["clm_state"] = clm["cstate"].reindex(D.index).fillna(-1).astype(int).values

    # --- kırılım etkenleri
    F = {
        "sun_elev": bins(D["sun_elev"], [-90, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 90]),
        "okta": D["okta"].astype(str),
        "elev_band": bins(D["elev"], [-1e9, 250, 750, 1250, 1750, 1e9], ["<250 m", "250–750 m", "750–1250 m", "1250–1750 m", "≥1750 m"]),
        "month": D["month"].astype(str), "src": D["src"], "illum": D["illum"],
    }
    if "rh2m" in D:
        F["rh2m"] = bins(D["rh2m"], [-1, 50, 70, 80, 90, 95, 101], ["<50 %", "50–70 %", "70–80 %", "80–90 %", "90–95 %", "≥95 %"])
    NWC_NAME = {1: "clear land", 2: "clear sea", 3: "snow", 5: "very low", 6: "low", 7: "mid-level", 8: "high opaque",
                9: "very high opaque", 10: "fractional", 11: "semi-transp. thin", 12: "semi-transp. moderate",
                13: "semi-transp. thick", 14: "semi-transp. above low", 15: "semi-transp. above snow", 0: "white/no data"}
    F["nwc_night"] = pd.Series(np.where(D["illum"] == "gece", D["nwc"].map(lambda c: NWC_NAME.get(c, "n/a")), "not night"),
                               index=D.index)
    rows = []
    for setname, sub in (("both", D), *[(s, D[D["set"] == s]) for s in SETS]):
        rows.append({"set": setname, "factor": "all", "level": "all", **rates(sub)})
        for f, lev in F.items():
            for l, g in sub.groupby(lev.loc[sub.index]):
                if len(g) >= a.min_n and l != "nan":
                    rows.append({"set": setname, "factor": f, "level": l, **rates(g)})
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(a.out, "error_analysis.csv"), index=False, encoding="utf-8")

    # --- odak sorular
    S = {}
    night = D["illum"] == "gece"
    fn_n = D[night & (D["label"] == 1) & (D["pred"] == 0)]
    hits_n = D[night & (D["label"] == 1) & (D["pred"] == 1)]
    S["night"] = {"n": int(night.sum()), **rates(D[night]), "day": rates(D[D["illum"] == "gündüz"])}
    def shares(x):
        out = {"n": int(len(x)), "okta9": float((x["okta"] == 9).mean()) if len(x) else None,
               "nwc_known": int((x["nwc"] < 255).sum()),
               "nwc_low_or_verylow": float(x.loc[x["nwc"] < 255, "nwc"].isin([5, 6]).mean()) if (x["nwc"] < 255).any() else None,
               "nwc_clear": float(x.loc[x["nwc"] < 255, "nwc"].isin([0, 1, 2, 3]).mean()) if (x["nwc"] < 255).any() else None}
        if "rh2m" in x:
            out["rh_ge90"] = float((x["rh2m"] >= 90).mean()) if len(x) else None
        return out
    S["night_misses"] = shares(fn_n)
    S["night_hits"] = shares(hits_n)
    winter = D["mon"].isin([12, 1, 2, 3])
    day = D["illum"] == "gündüz"
    clear = D["label"] == 0
    def fa(m):
        g = D[m & clear]
        return {"n_clear": int(len(g)), "false_alarm_rate": float((g["pred"] == 1).mean()) if len(g) else None}
    S["snow"] = {"winter_day_high": fa(winter & day & (D["elev"] > 1500)),
                 "winter_day_low": fa(winter & day & (D["elev"] <= 1500)),
                 "spring_day_high": fa(~winter & day & (D["elev"] > 1500))}
    with open(os.path.join(a.out, "error_summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=2, ensure_ascii=False)

    # --- temsilî vakalar
    err = D[D["pred"] != D["label"]]
    cases = err.sample(10, random_state=0).sort_values("time")
    cols = ["set", "time", "wmoid", "lat", "lon", "elev", "src", "illum", "sun_elev", "okta", "label", "prob", "nwc", "clm_state"]
    cols += [c for c in ("t2m", "rh2m", "vis_06", "nir_16", "ir_38", "ir_105", "bt38_105", "bt105_t2m", "radar_max") if c in D]
    cases[cols].to_csv(os.path.join(a.out, "error_cases.csv"), index=False, encoding="utf-8")

    pd.set_option("display.width", 250)
    q = E[(E["set"] == "both")]
    for f in ["all", "illum", "sun_elev", "rh2m", "okta", "elev_band", "nwc_night"]:
        x = q[q["factor"] == f]
        if len(x):
            print(x[["factor", "level", "n", "error", "miss_rate", "false_alarm_rate", "n_miss", "n_false_alarm"]].round(3).to_string(index=False))
    print(json.dumps(S, indent=1, ensure_ascii=False))
    print(cases[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
