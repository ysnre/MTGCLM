"""
update_h5_aux_channels.py
=========================
mtg_patch_dataset.h5 içine, her 33x33 yama için 5 adet RASTER yardımcı kanal ekler:

    aux[N, 5, 33, 33]  (float16)
      0  t2m        : 1982 AWOS istasyonundan IDW ile interpole 2 m sıcaklık (°C), isteğe bağlı lapse-rate düzeltmeli
      1  rh2m       : IDW ile interpole bağıl nem (%)
      2  elev       : yükseklik (m)  -> --dem verilirse DEM, verilmezse istasyon yüksekliklerinden IDW (kaba!)
      3  cos_sza    : cos(güneş zenit açısı), uydu tarama anı
      4  radar_cov  : radar yakınlığı 1 - d/r (radarda 1, menzil dışında 0; 0 = kapsama yok).
                      --radar_dir verilirse o anda PNG'si olan radarlar (zaman-bağımlı), yoksa statik birleşim

Ek datasetler:  station_y, station_x [N] int16 (yama merkez pikseli),
                t2m_center, rh2m_center [N] float32 (merkez pikseldeki interpole değer; etiket betiği sis kararı için kullanır),
                aux_flags [N] uint8   (bit0: o anda hiç AWOS verisi yok, bit1: istasyonun kendi AWOS verisi yok, bit2: konum yok)
Attrs:          aux_channels, aux_means, aux_stds (normalizasyon için; cos_sza ve radar_cov ham bırakılır),
                aux_params (JSON), aux_done (resume sayacı)

Uydu yamalarını YENİDEN ÇIKARMAZ; yalnızca (wmoid -> y,x) ve timestamp bilgisini kullanır.
update_h5_scalars.py'nin yerine geçer ('scalars' dataset'ine dokunmaz).

Kullanım (yerel):
    python src/update_h5_aux_channels.py --h5 F:/radar/mtg_patch_dataset.h5 ^
        --awos_dir e:/belgeler/MTGCLM/data/raw_awos_observation ^
        --station_csv e:/belgeler/MTGCLM/data/raw_observation/station_list.csv ^
        --radar_csv e:/belgeler/MTGCLM/data/raw_radar/radar.csv ^
        --radar_dir F:/radar/raw_radar   (opsiyonel, zaman-bağımlı kapsama) ^
        --dem e:/belgeler/MTGCLM/data/dem_800x1000.npy  (opsiyonel)
Kesintide tekrar çalıştırılırsa kaldığı yerden devam eder (--restart ile sıfırdan).
"""
import argparse
import bisect
import csv
import datetime as dt
import json
import os
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from grid_utils import (LatLonGrid, cos_solar_zenith, floor_to_10min, nearest_10min,
                        padded_window, parse_sat_timestamp)

AUX_CHANNELS = ["t2m", "rh2m", "elev", "cos_sza", "radar_cov"]
RADAR_MATCH_SEC = 300  # process_radar_pipeline_large.py ile aynı (+/- 5 dk)


# ----------------------------------------------------------------------------- istasyonlar
def parse_awos_stationlist(path: str) -> dict:
    """stationlist_awos.csv -> {wmoid: (lat, lon, elev)}. Satırların tamamı tırnaklı olabilir."""
    out = {}
    with open(path, "r", encoding="latin1", errors="replace") as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1].replace('""', '"')
            try:
                row = next(csv.reader([line]))
                wmoid = int(row[0])
                lat, lon = float(row[2]), float(row[3])
                elev = float(row[4]) if row[4].strip() and row[4].lower() != "nan" else np.nan
                out[wmoid] = (lat, lon, elev)
            except Exception:
                continue
    return out


def parse_label_stationlist(path: str) -> dict:
    """station_list.csv (SYNOP/METAR) -> {wmoid: (lat, lon, elev)}"""
    import pandas as pd
    df = pd.read_csv(path)
    df = df.dropna(subset=["latitude", "longitude"])
    out = {}
    for r in df.itertuples(index=False):
        elev = float(getattr(r, "elevation")) if hasattr(r, "elevation") and not np.isnan(float(getattr(r, "elevation"))) else np.nan
        out[int(r.wmoid)] = (float(r.latitude), float(r.longitude), elev)
    return out


# ----------------------------------------------------------------------------- AWOS tarama
def scan_awos_csv(observation_path: str, time_keys: list, station_ids: list):
    """
    4.4 GB CSV'yi tek geçişte tarar. Yalnızca time_keys içindeki (yıl,ay,gün,saat,dakika)
    anlarını ve station_ids içindeki istasyonları tutar.
    Döndürür: T[len(time_keys), len(station_ids)], RH[...] (float32, NaN = eksik)
    """
    t_index = {k: i for i, k in enumerate(time_keys)}     # "2025;3;1;6;40" -> idx
    s_index = {str(s): i for i, s in enumerate(station_ids)}
    T = np.full((len(time_keys), len(station_ids)), np.nan, dtype=np.float32)
    RH = np.full_like(T, np.nan)

    n_lines = n_hit = 0
    t0 = time.time()
    with open(observation_path, "r", encoding="latin1", errors="replace") as f:
        header = next(f)
        for line in f:
            n_lines += 1
            parts = line.rstrip("\n").split(";", 11)
            if len(parts) < 11:
                continue
            key = parts[2] + ";" + parts[3] + ";" + parts[4] + ";" + parts[5] + ";" + parts[6]
            ti = t_index.get(key)
            if ti is None:
                continue
            si = s_index.get(parts[0])
            if si is None:
                continue
            try:
                ts = parts[7].replace(",", ".").strip()
                if ts and ts.lower() not in ("nan", "-9999"):
                    T[ti, si] = float(ts)
                rs = parts[10].replace(",", ".").strip()
                if rs and rs.lower() not in ("nan", "-9999"):
                    RH[ti, si] = float(rs)
                n_hit += 1
            except ValueError:
                continue
            if n_lines % 20_000_000 == 0:
                print(f"   ... {n_lines/1e6:.0f} M satır, {n_hit/1e6:.2f} M eşleşme, {time.time()-t0:.0f} s")
    print(f"   Tarama bitti: {n_lines/1e6:.1f} M satır, {n_hit/1e6:.2f} M eşleşme, {time.time()-t0:.0f} s")
    # fiziksel aralık dışını at
    T[(T < -60) | (T > 60)] = np.nan
    RH[(RH < 0) | (RH > 100)] = np.nan
    return T, RH


# ----------------------------------------------------------------------------- IDW
class IDWInterpolator:
    """Bir zaman anı için, geçerli istasyonlardan k-NN ters-mesafe-kare interpolasyonu (km)."""

    def __init__(self, sy_km, sx_km, values, elev=None, k=8, power=2.0, max_dist_km=250.0):
        valid = ~np.isnan(values)
        self.n = int(valid.sum())
        self.k = min(k, self.n)
        self.power = power
        self.max_dist = max_dist_km
        if self.n == 0:
            return
        self.tree = cKDTree(np.c_[sy_km[valid], sx_km[valid]])
        self.vals = values[valid].astype(np.float64)
        self.elev = None if elev is None else elev[valid].astype(np.float64)

    def __call__(self, qy_km, qx_km, q_elev=None, lapse_per_m=0.0):
        """qy_km, qx_km: (H,W). q_elev: (H,W) veya None. Döndürür (H,W) float64."""
        if self.n == 0:
            return np.full(qy_km.shape, np.nan)
        q = np.c_[qy_km.ravel(), qx_km.ravel()]
        d, idx = self.tree.query(q, k=self.k)
        if self.k == 1:
            d, idx = d[:, None], idx[:, None]
        v = self.vals[idx]                                  # (Q,k)
        if lapse_per_m and self.elev is not None and q_elev is not None:
            z_st = self.elev[idx]
            ok = ~np.isnan(z_st)
            corr = np.where(ok, -lapse_per_m * (q_elev.ravel()[:, None] - z_st), 0.0)
            v = v + corr
        w = 1.0 / np.maximum(d, 0.5) ** self.power           # 0.5 km taban: istasyonun üstündeki piksel
        w[d > self.max_dist] = 0.0
        exact = d[:, 0] < 0.5
        out = np.where(exact, v[:, 0], (w * v).sum(1) / np.maximum(w.sum(1), 1e-12))
        out[(w.sum(1) == 0) & ~exact] = np.nan
        return out.reshape(qy_km.shape)


# ----------------------------------------------------------------------------- radar
def load_radar_sites(radar_csv: str) -> dict:
    """radar.csv -> {rn: (lat, lon, r_km)}. Aynı radar için en büyük menzil alınır."""
    sites = {}
    with open(radar_csv, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                rn = row["rn"].strip()
                lat, lon, r = float(row["lt"]), float(row["ln"]), float(row["r"])
            except (KeyError, ValueError):
                continue
            if rn not in sites or r > sites[rn][2]:
                sites[rn] = (lat, lon, r)
    return sites


def radar_circle_masks(sites: dict, grid: LatLonGrid) -> dict:
    """
    Her radar için 'yakınlık' alanı: 1 - min(d / r, 1)  (radarda 1, menzil kenarında/dışında 0).
    Sadece 0/1 kapsama yerine sürekli değer: menzil arttıkça ışın yüksekliği/kalite düştüğü için
    modele hem kapsamayı hem de güvenilirliği verir. Kompozit = radarlar üzerinde max.
    """
    lat2d, lon2d = grid.full_latlon()
    gy, gx = grid.to_km(lat2d, lon2d)
    masks = {}
    for rn, (lat, lon, r) in sites.items():
        cy, cx = grid.to_km(lat, lon)
        d = np.sqrt((gy - cy) ** 2 + (gx - cx) ** 2)
        masks[rn] = np.clip(1.0 - d / max(r, 1.0), 0.0, 1.0).astype(np.float32)
    return masks


def index_radar_files(radar_dir: str) -> dict:
    """raw radar PNG adları (AFY260522041206PPI7STZ.png) -> {rn: sorted [unix_ts]}"""
    idx = {}
    for name in os.listdir(radar_dir):
        if not name.lower().endswith(".png") or len(name) < 15:
            continue
        rn = name[:3]
        try:
            t = dt.datetime.strptime(name[3:15], "%y%m%d%H%M%S")
        except ValueError:
            continue
        idx.setdefault(rn, []).append(t.replace(tzinfo=dt.timezone.utc).timestamp())
    for rn in idx:
        idx[rn].sort()
    return idx


def radars_active_at(radar_index: dict, when: dt.datetime) -> list:
    ts = when.replace(tzinfo=dt.timezone.utc).timestamp()
    active = []
    for rn, times in radar_index.items():
        i = bisect.bisect_left(times, ts - RADAR_MATCH_SEC)
        if i < len(times) and times[i] <= ts + RADAR_MATCH_SEC:
            active.append(rn)
    return active


# ----------------------------------------------------------------------------- DEM
def load_dem(path: str, grid: LatLonGrid) -> np.ndarray:
    if path.lower().endswith(".npy"):
        dem = np.load(path).astype(np.float32)
    else:
        try:
            import rasterio
        except ImportError:
            raise SystemExit("GeoTIFF DEM için 'rasterio' gerekli; ya da DEM'i 800x1000 .npy olarak verin.")
        with rasterio.open(path) as src:
            dem = src.read(1).astype(np.float32)
    if dem.shape != grid.shape:
        raise SystemExit(f"DEM şekli {dem.shape} grid {grid.shape} ile uyuşmuyor. Önce grid'e yeniden örnekleyin.")
    dem[np.isnan(dem)] = 0.0
    return dem


def station_elevation_field(grid: LatLonGrid, stations: dict) -> np.ndarray:
    """DEM yoksa: tüm istasyon yüksekliklerinden IDW ile kaba yükseklik alanı (uyarı basılır)."""
    lat = np.array([v[0] for v in stations.values()])
    lon = np.array([v[1] for v in stations.values()])
    z = np.array([v[2] for v in stations.values()], dtype=np.float64)
    sy, sx = grid.to_km(lat, lon)
    idw = IDWInterpolator(sy, sx, z, k=6, power=2.0, max_dist_km=400.0)
    lat2d, lon2d = grid.full_latlon()
    gy, gx = grid.to_km(lat2d, lon2d)
    field = idw(gy, gx)
    field[np.isnan(field)] = np.nanmedian(z)
    return field.astype(np.float32)


# ----------------------------------------------------------------------------- ana akış
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="F:/radar/mtg_patch_dataset.h5")
    ap.add_argument("--awos_dir", default="e:/belgeler/MTGCLM/data/raw_awos_observation")
    ap.add_argument("--station_csv", default="e:/belgeler/MTGCLM/data/raw_observation/station_list.csv",
                    help="Etiketli (SYNOP/METAR) istasyon listesi; yama merkez pikseli buradan hesaplanır")
    ap.add_argument("--sample_nc", default=None, help="Grid için örnek turkiye_kesit_*.nc (yoksa varsayılan Türkiye grid'i)")
    ap.add_argument("--radar_csv", default="e:/belgeler/MTGCLM/data/raw_radar/radar.csv")
    ap.add_argument("--radar_dir", default=None, help="Ham radar PNG klasörü (zaman-bağımlı kapsama). Yoksa statik birleşim.")
    ap.add_argument("--dem", default=None, help="800x1000 DEM (.npy veya GeoTIFF). Yoksa istasyon IDW'si (kaba).")
    ap.add_argument("--patch_size", type=int, default=33)
    ap.add_argument("--time_match", choices=["floor", "nearest"], default="floor",
                    help="floor: uyduyu kapsayan 10 dk penceresi (önerilen); nearest: eski davranış")
    ap.add_argument("--k", type=int, default=8, help="IDW komşu sayısı")
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--max_dist_km", type=float, default=250.0)
    ap.add_argument("--lapse_rate", type=float, default=6.5, help="K/km; 0 = sıcaklıkta yükseklik düzeltmesi yok")
    ap.add_argument("--chunk", type=int, default=2048, help="H5 yazma bloğu (örnek)")
    ap.add_argument("--compress", action="store_true", help="gzip(1) sıkıştırma (Colab'da okuma yavaşlar)")
    ap.add_argument("--restart", action="store_true", help="Var olan 'aux' dataset'ini silip sıfırdan başla")
    ap.add_argument("--limit", type=int, default=None, help="Test: ilk N örnek")
    args = ap.parse_args()

    half = args.patch_size // 2
    grid = LatLonGrid.from_nc(args.sample_nc) if args.sample_nc else LatLonGrid.default_turkiye()
    print(f"Grid: {grid.shape}, lon {grid.lons[0]:.3f}..{grid.lons[-1]:.3f} (d={grid.dlon:+.3f}), "
          f"lat {grid.lats[0]:.3f}..{grid.lats[-1]:.3f} (d={grid.dlat:+.3f})")

    # --- istasyonlar
    awos_st = parse_awos_stationlist(os.path.join(args.awos_dir, "stationlist_awos.csv"))
    label_st = parse_label_stationlist(args.station_csv)
    print(f"AWOS istasyonu: {len(awos_st)}, etiketli istasyon: {len(label_st)}")
    awos_ids = sorted(awos_st)
    a_lat = np.array([awos_st[s][0] for s in awos_ids])
    a_lon = np.array([awos_st[s][1] for s in awos_ids])
    a_elev = np.array([awos_st[s][2] for s in awos_ids], dtype=np.float64)
    a_y_km, a_x_km = grid.to_km(a_lat, a_lon)
    awos_pos = {s: i for i, s in enumerate(awos_ids)}

    # --- statik alanlar
    if args.dem:
        elev_field = load_dem(args.dem, grid)
        print(f"DEM yüklendi: {args.dem}")
    else:
        print("UYARI: DEM verilmedi; yükseklik kanalı istasyon yüksekliklerinden IDW ile (kaba) üretiliyor. "
              "Makale için SRTM/GMTED DEM'i 800x1000 .npy olarak verin.")
        merged = dict(awos_st)
        merged.update({k: v for k, v in label_st.items() if not np.isnan(v[2])})
        elev_field = station_elevation_field(grid, {k: v for k, v in merged.items() if not np.isnan(v[2])})

    radar_sites = load_radar_sites(args.radar_csv) if args.radar_csv and os.path.exists(args.radar_csv) else {}
    radar_masks = radar_circle_masks(radar_sites, grid) if radar_sites else {}
    static_cov = np.zeros(grid.shape, dtype=np.float32)
    for m in radar_masks.values():
        np.maximum(static_cov, m, out=static_cov)
    radar_index = index_radar_files(args.radar_dir) if args.radar_dir else None
    print(f"Radar sahası: {len(radar_sites)} ({'zaman-bağımlı' if radar_index else 'statik'} kapsama); "
          f"statik kapsama (>0) oranı {(static_cov > 0).mean()*100:.1f}%, ortalama yakınlık {static_cov.mean():.2f}")
    if not radar_sites:
        print("UYARI: radar.csv bulunamadı; radar_cov kanalı tamamen 0 yazılacak.")

    # --- H5 metadata
    with h5py.File(args.h5, "r") as f:
        N = len(f["labels"])
        ts_raw = f["timestamps"][:]
        wmoids = f["wmoids"][:].astype(np.int64)
    if args.limit:
        N = min(N, args.limit)
    print(f"H5: {N} örnek")

    sat_times = [parse_sat_timestamp(t) for t in ts_raw[:N]]
    match_fn = floor_to_10min if args.time_match == "floor" else nearest_10min
    awos_times = [match_fn(t) for t in sat_times]
    time_key = [f"{t.year};{t.month};{t.day};{t.hour};{t.minute}" for t in awos_times]
    uniq_keys = sorted(set(time_key))
    key_pos = {k: i for i, k in enumerate(uniq_keys)}
    print(f"Benzersiz AWOS zaman anı: {len(uniq_keys)}")

    # yama merkezleri
    st_y = np.zeros(N, dtype=np.int16)
    st_x = np.zeros(N, dtype=np.int16)
    missing_station = 0
    for i in range(N):
        s = int(wmoids[i])
        if s in label_st:
            y, x = grid.latlon_to_index(label_st[s][0], label_st[s][1])
        elif s in awos_st:
            y, x = grid.latlon_to_index(awos_st[s][0], awos_st[s][1])
        else:
            y, x = -1, -1
            missing_station += 1
        st_y[i], st_x[i] = y, x
    if missing_station:
        print(f"UYARI: {missing_station} örneğin istasyonu hiçbir listede yok (y,x = -1; aux kanalı NaN->0).")

    # --- AWOS CSV taraması (tek geçiş)
    print("AWOS CSV taranıyor (4.4 GB, birkaç dakika)...")
    T, RH = scan_awos_csv(os.path.join(args.awos_dir, "observation_awos.csv"), uniq_keys, awos_ids)
    print(f"   Zaman-istasyon hücrelerinin dolu oranı: T {np.isfinite(T).mean()*100:.1f}%, RH {np.isfinite(RH).mean()*100:.1f}%")

    # --- H5 yazma (resume destekli)
    with h5py.File(args.h5, "a") as f:
        if args.restart and "aux" in f:
            for name in ("aux", "aux_flags", "station_y", "station_x", "t2m_center", "rh2m_center"):
                if name in f:
                    del f[name]
            f.attrs["aux_done"] = 0
        if "aux" not in f:
            kw = dict(compression="gzip", compression_opts=1) if args.compress else {}
            f.create_dataset("aux", shape=(N, len(AUX_CHANNELS), args.patch_size, args.patch_size),
                             dtype=np.float16, chunks=(min(args.chunk, N), len(AUX_CHANNELS), args.patch_size, args.patch_size), **kw)
            f.create_dataset("aux_flags", shape=(N,), dtype=np.uint8, chunks=(min(args.chunk, N),))
            f.create_dataset("station_y", data=st_y, dtype=np.int16)
            f.create_dataset("station_x", data=st_x, dtype=np.int16)
            f.create_dataset("t2m_center", shape=(N,), dtype=np.float32, fillvalue=np.nan)
            f.create_dataset("rh2m_center", shape=(N,), dtype=np.float32, fillvalue=np.nan)
            f.attrs["aux_channels"] = json.dumps(AUX_CHANNELS)
            f.attrs["aux_done"] = 0
        done = int(f.attrs.get("aux_done", 0))
        if done > 0:
            print(f"-> Kaldığı yerden devam: {done}/{N}")

        aux_ds, flag_ds = f["aux"], f["aux_flags"]
        tc_ds, rc_ds = f["t2m_center"], f["rh2m_center"]
        idw_cache = {}
        lapse_per_m = args.lapse_rate / 1000.0
        t0 = time.time()

        for start in range(done, N, args.chunk):
            end = min(start + args.chunk, N)
            buf = np.zeros((end - start, len(AUX_CHANNELS), args.patch_size, args.patch_size), dtype=np.float32)
            flags = np.zeros(end - start, dtype=np.uint8)
            for i in range(start, end):
                j = i - start
                yc, xc = int(st_y[i]), int(st_x[i])
                if yc < 0:
                    flags[j] |= 4
                    continue
                lat2d, lon2d = grid.window_latlon(yc, xc, half)
                qy, qx = grid.to_km(lat2d, lon2d)
                elev_w = padded_window(elev_field, yc, xc, half, fill=0.0).astype(np.float64)

                # IDW (zaman anına göre ağaç önbelleği)
                ti = key_pos[time_key[i]]
                if ti not in idw_cache:
                    if len(idw_cache) > 64:
                        idw_cache.clear()
                    idw_cache[ti] = (IDWInterpolator(a_y_km, a_x_km, T[ti], elev=a_elev, k=args.k, power=args.power, max_dist_km=args.max_dist_km),
                                     IDWInterpolator(a_y_km, a_x_km, RH[ti], k=args.k, power=args.power, max_dist_km=args.max_dist_km))
                idw_t, idw_rh = idw_cache[ti]
                if idw_t.n == 0 and idw_rh.n == 0:
                    flags[j] |= 1
                sp = awos_pos.get(int(wmoids[i]))
                if sp is None or np.isnan(T[ti, sp]):
                    flags[j] |= 2
                t_w = idw_t(qy, qx, q_elev=elev_w, lapse_per_m=lapse_per_m)
                rh_w = idw_rh(qy, qx)

                buf[j, 0] = t_w      # NaN olabilir; blok sonunda doldurulur
                buf[j, 1] = rh_w
                buf[j, 2] = elev_w
                buf[j, 3] = cos_solar_zenith(lat2d, lon2d, sat_times[i])
                if radar_index is not None:
                    cov = np.zeros((args.patch_size, args.patch_size), dtype=np.float32)
                    for rn in radars_active_at(radar_index, sat_times[i]):
                        if rn in radar_masks:
                            np.maximum(cov, padded_window(radar_masks[rn], yc, xc, half, fill=0.0), out=cov)
                    buf[j, 4] = cov
                else:
                    buf[j, 4] = padded_window(static_cov, yc, xc, half, fill=0.0)

            # eksik T/RH: o blok içindeki ortalamayla değil, iklimsel makul sabitle doldur ve bayrakla
            for c, fillv in ((0, 15.0), (1, 60.0)):
                nanmask = np.isnan(buf[:, c])
                if nanmask.any():
                    buf[:, c][nanmask] = fillv
            aux_ds[start:end] = buf.astype(np.float16)
            flag_ds[start:end] = flags
            tc_ds[start:end] = buf[:, 0, half, half]
            rc_ds[start:end] = buf[:, 1, half, half]
            f.attrs["aux_done"] = end
            f.flush()
            rate = (end - done) / max(time.time() - t0, 1e-6)
            print(f"   {end}/{N}  ({rate:.0f} örnek/s, kalan ~{(N-end)/max(rate,1e-6)/60:.1f} dk)")

        # normalizasyon istatistikleri: rastgele örneklem (resume'dan bağımsız, tüm veri üzerinden)
        fl_all = f["aux_flags"][:N]
        ok_idx = np.flatnonzero((fl_all & 4) == 0)
        rng = np.random.default_rng(0)
        samp = np.sort(rng.choice(ok_idx, size=min(20000, len(ok_idx)), replace=False))
        vals = np.concatenate([aux_ds[int(i)][None, :3].astype(np.float32) for i in samp], axis=0)
        means = vals.mean(axis=(0, 2, 3)); stds = vals.std(axis=(0, 2, 3)) + 1e-6
        f.attrs["aux_means"] = np.r_[means, 0.0, 0.0].astype(np.float32)   # cos_sza, radar_cov: ham bırakılır
        f.attrs["aux_stds"] = np.r_[stds, 1.0, 1.0].astype(np.float32)
        print(f"aux_means={np.round(means,2)}  aux_stds={np.round(stds,2)}  (n={len(samp)} örnek)")
        f.attrs["aux_params"] = json.dumps({k: v for k, v in vars(args).items()}, default=str)
        fl = f["aux_flags"][:N]
        print(f"Bayraklar: hiç AWOS yok {(fl & 1 > 0).sum()}, istasyonun kendi verisi yok {(fl & 2 > 0).sum()}, "
              f"istasyon konumu yok {(fl & 4 > 0).sum()}")
    print("Tamamlandı: 'aux' [N,5,33,33] float16 yazıldı.")


if __name__ == "__main__":
    main()
