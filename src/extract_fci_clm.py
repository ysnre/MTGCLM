"""
extract_fci_clm.py
==================
EUMETSAT FCI Level-2 Cloud Mask (CLM, EO:EUM:DAT:0678, netCDF) dosyalarından istasyon çevresi
sınıf sayımlarını çıkarır. Amaç: CLM'yi modellerimizle AYNI istasyon–zaman örneklerinde, hem merkez
piksel hem de gözlemci ayak izine yakın alanlar (3x3, 5/10/20/30 km yarıçap) üzerinden karşılaştırmak.

İki adım:
  1) İndeks (bir kez):  istasyon -> CLM piksel indeksleri (merkez + her yarıçap grubu).
     FCI 2 km tam disk ızgarası sabit olduğu için bir kez hesaplanır. satpy (fci_l2_nc okuyucusu)
     gerekir; hazır indeks dosyası data/clm_station_index.npz olarak verilmiştir.
       python src/extract_fci_clm.py --build_index --sample_nc F:/mtg_fci_clm/<herhangi bir CLM>.nc ^
           --stations data/raw_observation/station_list.csv --index data/clm_station_index.npz
  2) Çıkarma (tüm dosyalar, kaldığı yerden devam eder; yalnız numpy + netCDF4 gerekir):
       python src/extract_fci_clm.py --clm_dir F:/mtg_fci_clm --index data/clm_station_index.npz ^
           --out data/processed/clm_station
     Gün başına bir .npz yazılır (clm_YYYYMMDD.npz). ~28 bin dosya için tek çekirdekte ~1 saat mertebesi.

Çıktı (gün dosyası):
  rc_start [F]      : tekrar döngüsü başlangıcı 'YYYYmmddHHMM' (dosya adındaki _OPE_<başlangıç>)
  rc_count [F]      : günün tekrar döngüsü numarası (1..144)
  disposition [F]   : 'C' (commissioning, ön-operasyonel) | 'O' (operasyonel)  -- makalede ayrı raporlanmalı
  counts [F,S,G,10] : uint16; her istasyon (S) ve alan grubu (G) için cloud_state 0..9 sınıf sayıları
  center_state/quality/illum [F,S] : merkez pikselin cloud_state, quality_overall_processing, quality_illumination
  wmoid [S], groups [G]

cloud_state (ürün kılavuzu / .agent/rules/rule_fci_clm.md):
  0 işlenmedi | 1 bulutsuz | 2 bulut bulaşmış (kısmi/yarı saydam) | 3 bulutla dolu (opak) |
  4 toz bulaşmış | 5 toz dolu | 6 kül bulaşmış | 7 kül dolu | 8 bulutsuz (kar/buz) | 9 tanımsız

ÖNEMLİ — dizi yönü: CLM dizisinde satır 0 GÜNEY, sütun 0 BATI kenarıdır (sütun doğuya doğru artar).
netCDF'teki 'x' koordinatının işareti pyproj'un geos x'iyle ters; indeksler bu yüzden satpy'nin
alan tanımıyla (pyresample) hesaplanır. Gün/gece deseniyle doğrulandı (12 Eki 2026).

Birleştirme: modellerimizin örnek zamanı (Türkiye taraması, tekrar döngüsünün ~8. dakikası) 10 dk'ya
aşağı yuvarlanınca rc_start ile eşleşir. Bkz. load_station_table() ve join notu en altta.
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

GROUPS = ["c1", "3x3", "r5", "r10", "r20", "r30"]   # merkez, 3x3, yarıçap (km)
RADII_KM = {"r5": 5.0, "r10": 10.0, "r20": 20.0, "r30": 30.0}
MARGIN_KM = 40.0
FNAME_RE = re.compile(r"_OPE_(\d{14})_(\d{14})_N__([A-Z])_(\d{4})_")


# --------------------------------------------------------------------------- indeks
def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def build_index(sample_nc, stations_csv, out_path):
    import pandas as pd
    from satpy import Scene
    st = pd.read_csv(stations_csv)
    st = st.dropna(subset=["wmoid", "latitude", "longitude"]).drop_duplicates("wmoid")
    wmoid = st["wmoid"].astype(np.int64).values
    lat = st["latitude"].astype(float).values
    lon = st["longitude"].astype(float).values

    scn = Scene(reader="fci_l2_nc", filenames=[sample_nc])
    scn.load(["cloud_state"])
    area = scn["cloud_state"].attrs["area"]
    col, row = area.get_array_indices_from_lonlat(lon, lat)
    col = np.asarray(np.ma.filled(col, -1)); row = np.asarray(np.ma.filled(row, -1))
    if (col < 0).any() or (row < 0).any():
        raise SystemExit("Bazı istasyonlar CLM diskinin dışında görünüyor: " + str(wmoid[(col < 0) | (row < 0)]))

    # pencere: tüm istasyonlar + MARGIN_KM (CLM pikseli Türkiye'de ~2.8 x 3.5 km)
    pad_r, pad_c = int(np.ceil(MARGIN_KM / 3.0)) + 2, int(np.ceil(MARGIN_KM / 2.5)) + 2
    r0, r1 = int(row.min()) - pad_r, int(row.max()) + pad_r + 1
    c0, c1 = int(col.min()) - pad_c, int(col.max()) + pad_c + 1
    wlon, wlat = area.get_lonlats(data_slice=(slice(r0, r1), slice(c0, c1)))
    wlon = np.asarray(wlon, dtype=np.float64); wlat = np.asarray(wlat, dtype=np.float64)
    H, W = wlat.shape

    out = {"wmoid": wmoid, "lat": lat, "lon": lon, "row": row.astype(np.int32), "col": col.astype(np.int32),
           "window": np.array([r0, r1, c0, c1], dtype=np.int32), "groups": np.array(GROUPS)}
    for g in GROUPS:
        idx_all, ptr = [], [0]
        for k in range(len(wmoid)):
            rr, cc = int(row[k]) - r0, int(col[k]) - c0
            if g == "c1":
                idx = np.array([rr * W + cc])
            elif g == "3x3":
                ys, xs = np.meshgrid(np.arange(rr - 1, rr + 2), np.arange(cc - 1, cc + 2), indexing="ij")
                idx = (ys * W + xs).ravel()
            else:
                R = RADII_KM[g]
                y0, y1 = max(rr - pad_r, 0), min(rr + pad_r + 1, H)
                x0, x1 = max(cc - pad_c, 0), min(cc + pad_c + 1, W)
                d = haversine_km(lat[k], lon[k], wlat[y0:y1, x0:x1], wlon[y0:y1, x0:x1])
                ys, xs = np.nonzero(d <= R)
                idx = (ys + y0) * W + (xs + x0)
                if idx.size == 0:
                    idx = np.array([rr * W + cc])
            idx_all.append(idx.astype(np.int32)); ptr.append(ptr[-1] + idx.size)
        out[f"idx_{g}"] = np.concatenate(idx_all)
        out[f"ptr_{g}"] = np.array(ptr, dtype=np.int32)
    # makale için: istasyonda CLM piksel boyutu (km) ve uydu zenit açısı
    rr, cc = row - r0, col - c0
    ew = haversine_km(wlat[rr, cc], wlon[rr, cc], wlat[rr, np.minimum(cc + 1, W - 1)], wlon[rr, np.minimum(cc + 1, W - 1)])
    ns = haversine_km(wlat[rr, cc], wlon[rr, cc], wlat[np.minimum(rr + 1, H - 1), cc], wlon[np.minimum(rr + 1, H - 1), cc])
    out["pix_km_ew"], out["pix_km_ns"] = ew.astype(np.float32), ns.astype(np.float32)
    cosc = np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    c = np.arccos(cosc); rs = 42164.0 / 6371.0
    out["vza_deg"] = np.degrees(np.arctan2(np.sin(c), np.cos(c) - 1.0 / rs)).astype(np.float32)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **out)
    n = {g: float(np.mean(np.diff(out[f"ptr_{g}"]))) for g in GROUPS}
    print(f"İndeks yazıldı: {out_path}  istasyon={len(wmoid)}  pencere={out['window'].tolist()}")
    print("  grup başına ortalama piksel:", {g: round(v, 1) for g, v in n.items()})
    print(f"  CLM pikseli (istasyonlarda, km): D-B {ew.mean():.2f}, K-G {ns.mean():.2f};  VZA {out['vza_deg'].min():.1f}-{out['vza_deg'].max():.1f}°")


# --------------------------------------------------------------------------- çıkarma
def load_index(path):
    z = np.load(path, allow_pickle=False)
    ix = {k: z[k] for k in z.files}
    S = len(ix["wmoid"])
    ix["sid"] = {g: np.repeat(np.arange(S), np.diff(ix[f"ptr_{g}"])) for g in GROUPS}
    return ix


def process_file(path, ix):
    import netCDF4
    r0, r1, c0, c1 = [int(v) for v in ix["window"]]
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        cs = np.asarray(ds["cloud_state"][r0:r1, c0:c1], dtype=np.int16)
        q = np.asarray(ds["quality_overall_processing"][r0:r1, c0:c1], dtype=np.int8)
        il = np.asarray(ds["quality_illumination"][r0:r1, c0:c1], dtype=np.int8)
    cs = np.where((cs < 0) | (cs > 9), 0, cs)          # fill değerleri -> 0 (işlenmedi)
    flat = cs.ravel()
    S = len(ix["wmoid"])
    counts = np.zeros((S, len(GROUPS), 10), dtype=np.uint16)
    for gi, g in enumerate(GROUPS):
        v = flat[ix[f"idx_{g}"]]
        counts[:, gi, :] = np.bincount(ix["sid"][g] * 10 + v, minlength=S * 10).reshape(S, 10)
    c = ix["idx_c1"]
    return counts, flat[c].astype(np.int8), q.ravel()[c], il.ravel()[c]


def run_extract(clm_dir, index_path, out_dir, force=False, limit_days=None):
    ix = load_index(index_path)
    files = sorted(glob.glob(os.path.join(clm_dir, "*FCI-2-CLM*.nc")))
    by_day = {}
    for f in files:
        m = FNAME_RE.search(os.path.basename(f))
        if not m:
            continue
        by_day.setdefault(m.group(1)[:8], []).append((m.group(1)[:12], int(m.group(4)), m.group(3), f))
    days = sorted(by_day)
    if limit_days:
        days = days[:limit_days]
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(files)} CLM dosyası, {len(days)} gün -> {out_dir}")
    t0 = time.time(); done = 0
    for di, day in enumerate(days):
        out_path = os.path.join(out_dir, f"clm_{day}.npz")
        if os.path.exists(out_path) and not force:
            continue
        items = sorted(by_day[day])
        F, S = len(items), len(ix["wmoid"])
        counts = np.zeros((F, S, len(GROUPS), 10), dtype=np.uint16)
        cst = np.zeros((F, S), np.int8); cq = np.zeros((F, S), np.int8); cil = np.zeros((F, S), np.int8)
        ok = np.ones(F, bool)
        for k, (rc_start, rc_count, disp, f) in enumerate(items):
            try:
                counts[k], cst[k], cq[k], cil[k] = process_file(f, ix)
            except Exception as e:                       # bozuk dosya: işaretle, devam et
                ok[k] = False
                print(f"  ! okunamadı: {os.path.basename(f)} ({e})")
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(tmp, rc_start=np.array([i[0] for i in items]), rc_count=np.array([i[1] for i in items]),
                            disposition=np.array([i[2] for i in items]), ok=ok, counts=counts,
                            center_state=cst, center_quality=cq, center_illum=cil,
                            wmoid=ix["wmoid"], groups=np.array(GROUPS))
        os.replace(tmp, out_path)
        done += F
        el = time.time() - t0
        print(f"  {day}: {F} dosya  (toplam {done}, {done / max(el, 1e-6):.1f} dosya/s)")
    print(f"Bitti ({(time.time() - t0) / 60:.1f} dk).")


# --------------------------------------------------------------------------- analiz yardımcıları
def load_station_table(out_dir, group="r10", days=None):
    """Gün dosyalarını uzun tabloya çevirir: rc_start, wmoid, disposition, n_valid, n_clear, n_cont,
    n_fill, n_snow, n_other ve iki bulut oranı:
      frac_cloud_incl = (bulaşmış + dolu) / geçerli        -> 'contaminated = bulutlu' politikası
      frac_cloud_opq  = dolu / geçerli                      -> yalnız opak bulut
    geçerli = 1,2,3,8 sınıfları (toz/kül/işlenmedi/tanımsız hariç). Merkez piksel durumu da eklenir."""
    import pandas as pd
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "clm_*.npz"))):
        if days and os.path.basename(p)[4:12] not in days:
            continue
        z = np.load(p)
        gi = list(z["groups"]).index(group)
        c = z["counts"][:, :, gi, :].astype(np.int32)              # [F, S, 10]
        F, S = c.shape[:2]
        n_valid = c[..., [1, 2, 3, 8]].sum(-1)
        df = pd.DataFrame({
            "rc_start": np.repeat(z["rc_start"], S), "disposition": np.repeat(z["disposition"], S),
            "ok": np.repeat(z["ok"], S), "wmoid": np.tile(z["wmoid"], F),
            "n_valid": n_valid.ravel(), "n_clear": (c[..., 1] + c[..., 8]).ravel(), "n_cont": c[..., 2].ravel(),
            "n_fill": c[..., 3].ravel(), "n_snow": c[..., 8].ravel(),
            "n_other": (c[..., [0, 4, 5, 6, 7, 9]].sum(-1)).ravel(),
            "center_state": z["center_state"].ravel(), "center_quality": z["center_quality"].ravel(),
            "center_illum": z["center_illum"].ravel(),
        })
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    v = df["n_valid"].replace(0, np.nan)
    df["frac_cloud_incl"] = (df["n_cont"] + df["n_fill"]) / v
    df["frac_cloud_opq"] = df["n_fill"] / v
    return df


def sample_time_to_rc_start(ts):
    """Model örneğinin zaman damgası ('YYYYmmddHHMMSS...') -> CLM rc_start ('YYYYmmddHHMM', 10 dk'ya aşağı).
    NOT: H5'teki 'timestamps' alanının Türkiye taramasının zamanı olduğu varsayılır (dosya adlarındaki
    ~HH:M8 değeri). Eşleşmeyi birkaç örnekte CLM time_coverage_start ile doğrulayın."""
    import pandas as pd
    t = pd.to_datetime(pd.Series(ts).astype(str).str[:14], format="%Y%m%d%H%M%S")
    return t.dt.floor("10min").dt.strftime("%Y%m%d%H%M").values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build_index", action="store_true")
    ap.add_argument("--sample_nc", help="--build_index için herhangi bir CLM dosyası")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--index", default="data/clm_station_index.npz")
    ap.add_argument("--clm_dir", default="F:/mtg_fci_clm")
    ap.add_argument("--out", default="data/processed/clm_station")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit_days", type=int, default=None, help="deneme için ilk N gün")
    ap.add_argument("--summary", action="store_true", help="çıkarılmış gün dosyalarından kısa özet bas")
    ap.add_argument("--group", default="r10", choices=GROUPS)
    a = ap.parse_args()
    if a.build_index:
        if not a.sample_nc:
            raise SystemExit("--build_index için --sample_nc verin")
        build_index(a.sample_nc, a.stations, a.index)
    elif a.summary:
        df = load_station_table(a.out, a.group)
        print(df.groupby("disposition")[["frac_cloud_incl", "frac_cloud_opq"]].describe().T)
        print("merkez piksel sınıf dağılımı:", df["center_state"].value_counts(normalize=True).round(3).to_dict())
    else:
        run_extract(a.clm_dir, a.index, a.out, force=a.force, limit_days=a.limit_days)


if __name__ == "__main__":
    main()

# Birleştirme notu (analiz betiğinde):
#   rc = sample_time_to_rc_start(h5['timestamps'][idx])
#   tab = load_station_table('data/processed/clm_station', group='r10')
#   merged = pd.DataFrame({'rc_start': rc, 'wmoid': h5['wmoids'][idx], 'label': ...}).merge(tab, on=['rc_start','wmoid'])
#   CLM ikili karar (ör.):  merkez: center_state in (2,3) -> bulutlu, (1,8) -> açık
#                           alan:   frac_cloud_incl >= eşik (eşik ve grup yalnız TRAIN örneklerinde seçilir)
