"""
select_scenes.py — P5: Fig. 8 için test haftalarından 5 sahneyi ÖNCEDEN TANIMLI ölçütlerle, görüntülere bakmadan seçer.

Aday sahneler: split_v2_verify'ın test_time ve test_both kümelerindeki zaman damgaları (görülmemiş haftalar), en az
--min_rows etiketli istasyon. Her sahne türü için bir puan hesaplanır, en yüksek puanlı sahne seçilir; aynı gün ikinci kez
seçilmez (türler sırayla: konveksiyon, kar, sirrüs, gece alçak bulut/sis, gündüz açık — nadirden yaygına).

Türler ve puanlar (istasyon = etiketli test satırı; değerler istasyon pikselinde, H5'ten):
  convection  : yamada radar mavi kanal en büyüğü >= --radar_raw (varsayılan 128; PNG ölçeği 0–80 dBZ doğrusal ise ~40 dBZ)
                olan istasyon sayısı
  snow        : Ara–Mar, gündüz (istasyonların >= %90'ı güneş yüksekliği >= 5°); etiket açık, istasyon yüksekliği >= 1500 m,
                VIS0.6 >= %30 ve NIR1.6 / VIS0.6 <= 0.4 (kar: görünürde parlak, 1.6 µm'de koyu) olan istasyon sayısı
  cirrus      : gündüz; NWC SAF CT merkez sınıfı yarı saydam (11–15) olan istasyon sayısı
  night_low   : gece (istasyonların tümü <= −5°); etiket bulutlu ve (okta 9 ya da RH2m >= %90) olan istasyon sayısı
  day_clear   : gündüz; açık etiket oranı (eşitlikte istasyon sayısı)
Çıktılar: <out>/scenes_selected.csv (seçilenler, puan ve gerekçe), <out>/scenes_candidates.csv (her türün ilk 10 adayı).

    python src/select_scenes.py --h5 C:/mtg/mtg_train_v2.h5
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import load_meta          # noqa: E402
from channels import BASE_CHANNELS           # noqa: E402
from extract_nwcsaf_ct import sample_time_to_slot  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["convection", "snow", "cirrus", "night_low", "day_clear"]
DESC = {
    "convection": "stations with patch radar maximum >= {radar_raw} (blue channel)",
    "snow": "clear stations >= 1500 m with VIS0.6 >= 30 % and NIR1.6/VIS0.6 <= 0.4 (Dec-Mar, day)",
    "cirrus": "stations with NWC SAF CT semi-transparent class (11-15), day",
    "night_low": "cloudy stations with okta 9 or RH2m >= 90 %, night",
    "day_clear": "fraction of clear stations, day",
}


def station_values(h5_path, rows, block=1024):
    """rows (artan) için merkez piksel değerleri: vis_06, nir_16 (ham %), radar yama maksimumu (ham), rh2m, elev."""
    rows = np.asarray(rows)
    c_vis, c_nir, c_rad = BASE_CHANNELS.index("vis_06"), BASE_CHANNELS.index("nir_16"), BASE_CHANNELS.index("radar")
    out = {k: np.empty(len(rows), np.float32) for k in ("vis06", "nir16", "radar_max", "rh2m", "elev_dem")}
    with h5py.File(h5_path, "r") as f:
        means, stds = f.attrs["means"], f.attrs["stds"]
        aux_names = json.loads(f.attrs["aux_channels"])
        i_rh, i_el = aux_names.index("rh2m"), aux_names.index("elev")
        P, A = f["patches"], f["aux"]
        for s in range(0, len(rows), block):
            r = rows[s:s + block]
            lo, hi = int(r[0]), int(r[-1]) + 1
            if hi - lo <= 12 * len(r):                  # yoğun blok: tek dilim oku, sonra seç
                p = P[lo:hi][r - lo]
                a = A[lo:hi][r - lo]
            else:
                p = P[r]
                a = A[r]
            p = p.astype(np.float32)
            out["vis06"][s:s + len(r)] = p[:, c_vis, 16, 16] * stds[c_vis] + means[c_vis]
            out["nir16"][s:s + len(r)] = p[:, c_nir, 16, 16] * stds[c_nir] + means[c_nir]
            out["radar_max"][s:s + len(r)] = p[:, c_rad].max(axis=(1, 2)) * stds[c_rad] + means[c_rad]
            out["rh2m"][s:s + len(r)] = a[:, i_rh, 16, 16].astype(np.float32)
            out["elev_dem"][s:s + len(r)] = a[:, i_el, 16, 16].astype(np.float32)
    return out


def nwc_center_class(meta_rows, nwc_dir):
    """Her satır için NWC SAF CT merkez sınıfı (yoksa 255)."""
    slots = sample_time_to_slot(meta_rows["ts"].values)
    out = np.full(len(meta_rows), 255, np.uint8)
    for day in sorted(set(s[:8] for s in slots)):
        p = os.path.join(nwc_dir, f"nwc_{day}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        si = {s: i for i, s in enumerate(z["slot"])}
        wi = {int(w): i for i, w in enumerate(z["wmoid"])}
        cc = z["center_class"]
        for k in np.flatnonzero(np.array([s[:8] == day for s in slots])):
            i, j = si.get(slots[k]), wi.get(int(meta_rows["wmoid"].values[k]))
            if i is not None and j is not None:
                out[k] = cc[i, j]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--split", default=os.path.join(ROOT, "splits/split_v2_verify.npz"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--stations", default=os.path.join(ROOT, "data/raw_observation/station_list.csv"))
    ap.add_argument("--nwc_dir", default=os.path.join(ROOT, "data/processed/nwcsaf_station"))
    ap.add_argument("--min_rows", type=int, default=25)
    ap.add_argument("--radar_raw", type=float, default=128.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/scenes"))
    a = ap.parse_args()

    meta = load_meta(a.meta, a.h5, a.stations)
    with h5py.File(a.h5, "r") as f:
        ts = np.array([t.decode() if isinstance(t, bytes) else str(t) for t in f["timestamps"][:]])
    sp = np.load(a.split)
    rows = np.unique(np.concatenate([sp["test_time"], sp["test_both"]]))
    m = meta.iloc[rows].copy()
    m["row"] = rows
    m["ts"] = ts[rows]
    m = m[m["label"] >= 0]
    print(f"Test haftası satırı: {len(m)}  sahne: {m['ts'].nunique()}")

    v = station_values(a.h5, m["row"].values)
    for k, x in v.items():
        m[k] = x
    m["nwc"] = nwc_center_class(m, a.nwc_dir)
    m["month_i"] = m["time"].dt.month

    m["f_conv"] = m["radar_max"] >= a.radar_raw
    m["f_snow"] = (m["label"] == 0) & (m["elev"] >= 1500) & (m["vis06"] >= 30) & (m["nir16"] <= 0.4 * m["vis06"])
    m["f_ci"] = m["nwc"].between(11, 15)
    m["f_low"] = (m["label"] == 1) & ((m["okta"] == 9) | (m["rh2m"] >= 90))
    g = m.groupby("ts")
    S = pd.DataFrame({
        "n_rows": g.size(), "clear_frac": g["label"].apply(lambda s: float((s == 0).mean())),
        "day_frac": g["illum"].apply(lambda s: float((s == "gündüz").mean())),
        "night_frac": g["illum"].apply(lambda s: float((s == "gece").mean())),
        "month": g["month_i"].first(), "n_conv": g["f_conv"].sum(), "n_snow": g["f_snow"].sum(),
        "n_ci": g["f_ci"].sum(), "n_low": g["f_low"].sum(), "nwc_cover": g["nwc"].apply(lambda s: float((s < 255).mean())),
    })
    S = S[S["n_rows"] >= a.min_rows].copy()
    S["day"] = S.index.str[:8]
    day = S["day_frac"] >= 0.9
    S["score_convection"] = S["n_conv"].astype(float)
    S["score_snow"] = np.where(day & S["month"].isin([12, 1, 2, 3]), S["n_snow"], -1.0)
    S["score_cirrus"] = np.where(day, S["n_ci"], -1.0)
    S["score_night_low"] = np.where(S["night_frac"] >= 1.0, S["n_low"], -1.0)
    S["score_day_clear"] = np.where(day, S["clear_frac"], -1.0)

    chosen, used_days, cand = [], set(), []
    for cat in ORDER:
        T = S.sort_values([f"score_{cat}", "n_rows"], ascending=False)
        T = T[T[f"score_{cat}"] > 0]
        top = T.head(10).assign(category=cat)
        cand.append(top.reset_index())
        pick = T[~T["day"].isin(used_days)].head(1)
        if pick.empty:
            print(f"UYARI: {cat} için uygun sahne yok")
            continue
        r = pick.iloc[0]
        used_days.add(r["day"])
        chosen.append({"category": cat, "scene": pick.index[0], "score": float(r[f"score_{cat}"]),
                       "criterion": DESC[cat].format(radar_raw=a.radar_raw), "n_rows": int(r["n_rows"]),
                       "clear_frac": round(float(r["clear_frac"]), 3), "day_frac": round(float(r["day_frac"]), 3),
                       "n_conv": int(r["n_conv"]), "n_snow": int(r["n_snow"]), "n_ci": int(r["n_ci"]),
                       "n_low": int(r["n_low"]), "nwc_cover": round(float(r["nwc_cover"]), 3)})
    C = pd.DataFrame(chosen)
    os.makedirs(a.out, exist_ok=True)
    C.to_csv(os.path.join(a.out, "scenes_selected.csv"), index=False, encoding="utf-8")
    pd.concat(cand).to_csv(os.path.join(a.out, "scenes_candidates.csv"), index=False, encoding="utf-8")
    print(C.to_string(index=False))
    times = ",".join(s[:12] for s in C["scene"])
    print("\nSıradaki komutlar:")
    print(f"  python src\\infer_scene.py --h5 {a.h5} --time {times} --seeds 1,2,3 --threads 12")
    print(f"  python src\\build_scene_pack.py")


if __name__ == "__main__":
    main()
