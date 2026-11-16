"""
parallax_shift.py — P10: paralaks düzeltmesi yapılmadığı için bulutun görünür konum kaymasının büyüklüğü (eğitim yok).

Geometri: MTG-I1 yerdurağan, alt nokta 0° boylam, uydu yarıçapı 42 164 km; küresel Dünya (R = 6 371 km). Gerçek konumu Q olan
ve h km yükseklikteki bulut tepesi C, uydudan bakıldığında S→C doğrusunun yeryüzünü kestiği P' noktasında görünür.
Kayma = Q ile P' arasındaki büyük çember uzaklığı; yön = uydudan uzağa (Türkiye için kuzey-doğu). Piksel karşılığı model
ızgarasında (0,02° boylam × 0,01° enlem). VZA ve uydu azimutu da yazılır. (satpy'nin paralaks modülüyle karşılaştırma:
satpy bu ortamda yok; analitik çözüm ellipsoid etkisini ihmal eder, Türkiye enlemlerinde fark < %1 mertebesinde.)

Çıktı: artifacts_v3/analysis/parallax_shift.csv (istasyon × CTH), parallax_summary.csv (CTH başına min/medyan/maks),
       artifacts_v3/analysis/parallax_grid_6km.npz (tüm ızgara, ek harita için)

    python src/parallax_shift.py
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_E, R_S = 6371.0, 42164.0
CTH = (1.0, 3.0, 6.0, 10.0)


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def to_latlon(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(v[..., 2])), np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def geometry(lat, lon, sat_lon=0.0):
    """VZA (°) ve uydu azimutu (°, kuzeyden saat yönünde) yerdeki noktada."""
    q = unit(lat, lon) * R_E
    s = unit(np.zeros_like(lat), np.full_like(lon, sat_lon)) * R_S
    los = s - q
    n = q / R_E
    cosz = (los * n).sum(-1) / np.linalg.norm(los, axis=-1)
    vza = np.degrees(np.arccos(np.clip(cosz, -1, 1)))
    la, lo = np.radians(lat), np.radians(lon)
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
    az = np.degrees(np.arctan2((los * east).sum(-1), (los * north).sum(-1))) % 360
    return vza, az


def apparent(lat, lon, h, sat_lon=0.0):
    """h km'deki bulut tepesinin görünür yer konumu (lat, lon)."""
    c = unit(lat, lon) * (R_E + h)
    s = unit(np.zeros_like(lat), np.full_like(lon, sat_lon)) * R_S
    d = c - s
    a = (d * d).sum(-1)
    b = 2 * (s * d).sum(-1)
    cc = (s * s).sum(-1) - R_E ** 2
    t = (-b + np.sqrt(b * b - 4 * a * cc)) / (2 * a)       # yakın kesişim C'den sonra (uzak taraf değil): küçük kök C'ye yakın
    t = np.where(t < 1.0, t, (-b - np.sqrt(b * b - 4 * a * cc)) / (2 * a))
    p = s + t[..., None] * d
    return to_latlon(p)


def gc_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * R_E * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--stations", default=os.path.join(ROOT, "data/raw_observation/station_list.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/analysis"))
    a = ap.parse_args()
    used = set(np.load(a.meta)["wmoids"].astype(int))
    st = pd.read_csv(a.stations).dropna(subset=["wmoid", "latitude", "longitude"]).drop_duplicates("wmoid")
    st = st[st.wmoid.astype(int).isin(used)]
    lat, lon = st.latitude.values.astype(float), st.longitude.values.astype(float)
    vza, az = geometry(lat, lon)
    rows = []
    for h in CTH:
        la2, lo2 = apparent(lat, lon, h)
        d = gc_km(lat, lon, la2, lo2)
        for k in range(len(lat)):
            rows.append({"wmoid": int(st.wmoid.values[k]), "lat": lat[k], "lon": lon[k], "vza": vza[k], "sat_azimuth": az[k],
                         "cth_km": h, "shift_km": d[k], "dlat": la2[k] - lat[k], "dlon": lo2[k] - lon[k],
                         "shift_px_ns": (la2[k] - lat[k]) / 0.01, "shift_px_ew": (lo2[k] - lon[k]) / 0.02})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "parallax_shift.csv"), index=False, encoding="utf-8")
    S = T.groupby("cth_km").agg(stations=("wmoid", "count"), shift_min=("shift_km", "min"), shift_median=("shift_km", "median"),
                                shift_max=("shift_km", "max"), px_ns_median=("shift_px_ns", "median"),
                                px_ew_median=("shift_px_ew", "median"), px_ns_max=("shift_px_ns", "max"),
                                px_ew_max=("shift_px_ew", "max")).reset_index()
    S["vza_min"], S["vza_median"], S["vza_max"] = vza.min(), np.median(vza), vza.max()
    S.to_csv(os.path.join(a.out, "parallax_summary.csv"), index=False, encoding="utf-8")
    # tüm ızgara, 6 km
    lons = 25.01 + 0.02 * np.arange(1000)
    lats = 42.995 - 0.01 * np.arange(800)
    LO, LA = np.meshgrid(lons, lats)
    la2, lo2 = apparent(LA, LO, 6.0)
    np.savez_compressed(os.path.join(a.out, "parallax_grid_6km.npz"), shift_km=gc_km(LA, LO, la2, lo2).astype(np.float32),
                        vza=geometry(LA, LO)[0].astype(np.float32), lats=lats, lons=lons)
    pd.set_option("display.width", 200)
    print(S.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
