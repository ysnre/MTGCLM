"""
infer_scene.py — P4: tek zaman adımı için 800x1000 bulut olasılık haritası ve ikili maske.

Girdiler eğitim verisiyle BİREBİR aynı kurallarla üretilir:
  * FCI: turkiye_kesit_<YYYYmmddHHMMSS>_<tarama>.nc (coknc.py çıktısı), kanallar src/channels.py sırasıyla ADLA okunur,
    H5 attrs means/stds ile z-skor, NaN -> 0, float16 yuvarlaması (kompakt H5 float16 saklar).
  * Radar: radar_aligned/<aynı ad>.npy (process_radar_pipeline_large.py; mavi kanal dBZ bileşimi). Yoksa 0 + bayrak.
  * aux (update_h5_aux_channels.py fonksiyonları, parametreler H5 attrs 'aux_params'tan): t2m (IDW, k=8, p=2,
    250 km, 6.5 K/km), rh2m (IDW), elev (DEM), cos_sza (tarama anı), radar_cov (o anda PNG'si olan radarlar;
    --radar_dir yoksa statik birleşim + bayrak). AWOS zamanı: floor (uyduyu kapsayan 10 dk). float16 yuvarlaması,
    sonra aux_means/aux_stds ile normalizasyon (train.py prep_features ile aynı).
  * Grid dışı (yama kenarı): eğitimdeki gibi uydu/radar 0 (normalize uzayda), elev ve radar_cov 0, t2m/rh2m/cos_sza
    ekstrapole enlem/boylamda hesaplanır. Bu "eğitim dolgusu" tam sahnede de kullanılır (--pad train, varsayılan);
    --pad reflect yansıtmalı dolgu ile karşılaştırma içindir.

Modeller (--exp, ağırlıklar artifacts_v3/<exp>_s<seed>.json -> config.best_model_file):
  * D_unet (varsayılan): tam evrişimli U-Net.
      --unet_mode aligned (varsayılan): 16 geçiş; her piksel, eğitimdeki merkez pikselle (16,16) aynı havuzlama
        fazında hesaplanır (U-Net yalnız merkez pikselde denetlendi; 4 piksellik faz deseni önlenir).
      --unet_mode single: tek geçiş (hızlı; faz deseni --artifact_check ile ölçülür).
      --unet_mode patch: her piksel için 33x33 yama (eğitimle birebir; yavaş, doğrulama için).
  * B_full vb. (Medium_Flat): kayan pencere 33x33, adım 1, batch'li (eğitimle birebir).
  * --seeds 1,2,3 -> olasılıkların ortalaması.

Çıktılar (--out, varsayılan artifacts_v3/scenes/): <sahne>_<exp>.nc (lat, lon, cloud_prob, cloud_mask, quality_flag;
h5py ile NetCDF4 uyumlu), <...>_prob.tif (GeoTIFF, EPSG:4326, float32), <...>.png (önizleme), <...>.json (süre,
bellek, bayraklar, tutarlılık testi). Ayrıca <out>/timing.csv'ye bir satır eklenir.

quality_flag bitleri: 1 = uydu kanalı NaN, 2 = sahnede radar dosyası yok, 4 = AWOS IDW boş (sabitle dolduruldu),
8 = radar kapsamı dışı (radar_cov = 0), 16 = radar_cov statik (zaman-bağımlı kapsama bilinmiyor).
Geçerli piksel = (flag & 1) == 0 ve (flag & 2) == 0; diğerleri bilgi amaçlıdır.

Tutarlılık testi (--check): H5'te bu zamana ait satırlar (station_y/x) için (a) yeniden üretilen 33x33 girdi ile H5'teki
yama+aux farkı (kanal başına), (b) yeniden üretilen yamadan model olasılığı ile preds/ dosyasındaki kayıtlı olasılık farkı,
(c) tam sahne çıktısı ile kayıtlı olasılık farkı. Beklenen: (a) ~float16 yuvarlaması, (b) medyan < 1e-3.

Örnek (yerel PC):
  python src\\infer_scene.py --h5 C:/mtg/mtg_train_v2.h5 --time 202605021200 --check --artifact_check
  python src\\infer_scene.py --h5 C:/mtg/mtg_train_v2.h5 --time 202605021200 --exp B_full --check
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import AUX_CHANNELS, FCI_CHANNELS                                   # noqa: E402
from grid_utils import LatLonGrid, cos_solar_zenith, floor_to_10min, nearest_10min  # noqa: E402
from update_h5_aux_channels import (IDWInterpolator, index_radar_files, load_radar_sites,   # noqa: E402
                                    parse_awos_stationlist, radar_circle_masks, radars_active_at, scan_awos_csv)
from model import MTGConvNet, MTGUNet                                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH = 33
HALF = PATCH // 2
MARGIN = 24          # dış dolgu (eğitim kuralı); U-Net faz hizalaması 16 + 3 + 4'e kadar ister
FLAG_SAT_NAN, FLAG_NO_RADAR, FLAG_AWOS_FILL, FLAG_NO_RADAR_COV, FLAG_STATIC_COV = 1, 2, 4, 8, 16
MEDIUM = [32, 64, 128]

try:
    import psutil
    _PROC = psutil.Process()
except Exception:   # psutil yoksa bellek raporu boş kalır
    _PROC = None


def mem_mb():
    if _PROC is None:
        return None
    mi = _PROC.memory_info()
    peak = getattr(mi, "peak_wset", None) or getattr(mi, "peak_rss", None)   # Windows: peak_wset
    return round((peak or mi.rss) / 2 ** 20, 1)


class Timer:
    def __init__(self):
        self.t = {}
        self.mem = 0.0

    def __call__(self, name):
        tm = self

        class _C:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                tm.t[name] = round(tm.t.get(name, 0.0) + time.perf_counter() - self.t0, 3)
                m = mem_mb()
                if m is not None:
                    tm.mem = max(tm.mem, m)
        return _C()


# ----------------------------------------------------------------------------- sahne bulma
def scene_ts(path):
    return os.path.basename(path).split("_")[2][:14]


def find_scene(nc_dir, when):
    """when: 'YYYYmmddHHMM' (UTC). Uydu zamanı aynı 10 dakikalık pencerede olan dosya; yoksa en yakın (<= 10 dk)."""
    t = dt.datetime.strptime(when[:12], "%Y%m%d%H%M")
    cands = []
    for h in (t - dt.timedelta(hours=1), t, t + dt.timedelta(hours=1)):
        cands += glob.glob(os.path.join(nc_dir, f"turkiye_kesit_{h:%Y%m%d%H}*.nc"))
    if not cands:
        raise SystemExit(f"{nc_dir} içinde {when} çevresinde sahne yok.")
    best = min(cands, key=lambda p: abs((dt.datetime.strptime(scene_ts(p), "%Y%m%d%H%M%S") - t).total_seconds()))
    gap = abs((dt.datetime.strptime(scene_ts(best), "%Y%m%d%H%M%S") - t).total_seconds())
    if gap > 600:
        raise SystemExit(f"{when} için en yakın sahne {scene_ts(best)} ({gap/60:.0f} dk) — 10 dk'dan uzak.")
    return best


# ----------------------------------------------------------------------------- normalizasyon sabitleri
def load_norm(h5_path):
    with h5py.File(h5_path, "r") as f:
        a = f.attrs
        means = np.asarray(a["means"], np.float64)
        stds = np.asarray(a["stds"], np.float64)
        aux_names = json.loads(a["aux_channels"])
        aux_means = np.asarray(a["aux_means"], np.float32)
        aux_stds = np.asarray(a["aux_stds"], np.float32)
        params = json.loads(a["aux_params"]) if "aux_params" in a else {}
    if aux_names != AUX_CHANNELS:
        raise SystemExit(f"H5 aux sırası {aux_names} != {AUX_CHANNELS}")
    if len(means) != 17:
        raise SystemExit(f"H5 'means' uzunluğu {len(means)} (17 beklenir)")
    return means, stds, aux_means, aux_stds, params


# ----------------------------------------------------------------------------- AWOS (önbellekli)
def awos_fields(keys, awos_ids, observation_csv, cache_dir):
    """keys: ['Y;M;D;H;M', ...] -> {key: (T[n_st], RH[n_st])}. Eksik anlar tek geçişte taranır ve önbelleğe yazılır."""
    out, missing = {}, []
    ids = np.asarray(awos_ids, np.int64)
    for k in keys:
        p = os.path.join(cache_dir, "awos_" + "_".join(f"{int(v):02d}" for v in k.split(";")) + ".npz") if cache_dir else None
        if p and os.path.exists(p):
            z = np.load(p)
            if np.array_equal(z["ids"], ids):
                out[k] = (z["T"], z["RH"])
                continue
        missing.append(k)
    if missing:
        print(f"AWOS taranıyor: {len(missing)} zaman anı ({observation_csv}) ...")
        T, RH = scan_awos_csv(observation_csv, missing, list(awos_ids))
        for i, k in enumerate(missing):
            out[k] = (T[i], RH[i])
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                p = os.path.join(cache_dir, "awos_" + "_".join(f"{int(v):02d}" for v in k.split(";")) + ".npz")
                np.savez_compressed(p, T=T[i], RH=RH[i], ids=ids)
    return out


# ----------------------------------------------------------------------------- girdi üretimi
def build_inputs(nc_path, a, norm, statics, awos):
    """Döndürür: X_ext [22, H+2M, W+2M] float32 (normalize), flag [H, W] uint8, grid, sat_time, info."""
    means, stds, aux_means, aux_stds, params = norm
    grid = LatLonGrid.from_nc(nc_path)
    H, W = grid.shape
    M = MARGIN
    ts = scene_ts(nc_path)
    t_sat = dt.datetime.strptime(ts, "%Y%m%d%H%M%S")
    info = {"nc": os.path.basename(nc_path), "sat_time": ts}
    flag = np.zeros((H, W), np.uint8)

    # --- uydu + radar (17 kanal), eğitimdeki create_patch_dataset.load_features_fast ile aynı
    base = np.zeros((17, H, W), np.float32)
    nan_any = np.zeros((H, W), bool)
    with h5py.File(nc_path, "r") as f:
        for c, name in enumerate(FCI_CHANNELS):
            if name not in f:
                raise SystemExit(f"{nc_path}: kanal {name} yok")
            v = f[name][:]
            nan_any |= ~np.isfinite(v)
            base[c] = (v - means[c]) / stds[c]
    rpath = os.path.join(a.radar_npy_dir, os.path.basename(nc_path).replace(".nc", ".npy"))
    if os.path.exists(rpath):
        radar = np.load(rpath).astype(np.float32)
        info["radar_file"] = os.path.basename(rpath)
    else:
        radar = np.zeros((H, W), np.float32)
        flag |= FLAG_NO_RADAR
        info["radar_file"] = None
    base[16] = (radar - means[16]) / stds[16]
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    base = base.astype(np.float16).astype(np.float32)          # kompakt H5 float16 saklar
    flag[nan_any] |= FLAG_SAT_NAN

    # --- aux, genişletilmiş grid üzerinde (grid dışı enlem/boylam lineer ekstrapolasyon; window_latlon ile aynı)
    ys = np.arange(-M, H + M)
    xs = np.arange(-M, W + M)
    lat_e = grid.lats[0] + grid.dlat * ys
    lon_e = grid.lons[0] + grid.dlon * xs
    lon2d, lat2d = np.meshgrid(lon_e, lat_e)
    qy, qx = grid.to_km(lat2d, lon2d)
    elev_e = np.pad(statics["dem"], M, mode="constant", constant_values=0.0).astype(np.float64)

    p = params
    t_awos = (floor_to_10min if p.get("time_match", "floor") == "floor" else nearest_10min)(t_sat)
    key = f"{t_awos.year};{t_awos.month};{t_awos.day};{t_awos.hour};{t_awos.minute}"
    T, RH = awos[key]
    sy, sx, selev = statics["awos_ykm"], statics["awos_xkm"], statics["awos_elev"]
    k, pw, md = int(p.get("k", 8)), float(p.get("power", 2.0)), float(p.get("max_dist_km", 250.0))
    lapse = float(p.get("lapse_rate", 6.5)) / 1000.0
    idw_t = IDWInterpolator(sy, sx, T, elev=selev, k=k, power=pw, max_dist_km=md)
    idw_rh = IDWInterpolator(sy, sx, RH, k=k, power=pw, max_dist_km=md)
    t2m = idw_t(qy, qx, q_elev=elev_e, lapse_per_m=lapse)
    rh2m = idw_rh(qy, qx)
    info.update(awos_time=t_awos.strftime("%Y%m%d%H%M"), awos_n_t=int(idw_t.n), awos_n_rh=int(idw_rh.n))
    fill = np.isnan(t2m) | np.isnan(rh2m)
    t2m[np.isnan(t2m)] = 15.0                     # update_h5_aux_channels.py ile aynı sabitler
    rh2m[np.isnan(rh2m)] = 60.0
    flag[fill[M:M + H, M:M + W]] |= FLAG_AWOS_FILL

    if statics["radar_index"] is not None:
        cov = np.zeros((H, W), np.float32)
        active = [rn for rn in radars_active_at(statics["radar_index"], t_sat) if rn in statics["radar_masks"]]
        for rn in active:
            np.maximum(cov, statics["radar_masks"][rn], out=cov)
        info["radars_active"] = sorted(active)
    else:
        cov = statics["static_cov"].copy()
        flag |= FLAG_STATIC_COV
        info["radars_active"] = None
    flag[cov <= 0] |= FLAG_NO_RADAR_COV
    cov_e = np.pad(cov, M, mode="constant", constant_values=0.0)

    aux = np.stack([t2m, rh2m, elev_e, cos_solar_zenith(lat2d, lon2d, t_sat), cov_e]).astype(np.float32)
    aux = aux.astype(np.float16).astype(np.float32)            # H5 'aux' float16
    aux = (aux - aux_means[:, None, None]) / aux_stds[:, None, None]

    X = np.zeros((17 + 5, H + 2 * M, W + 2 * M), np.float32)
    X[:17, M:M + H, M:M + W] = base
    X[17:] = aux
    if a.pad == "reflect":     # karşılaştırma: tüm kanallarda yansıtmalı dış dolgu
        X = np.pad(X[:, M:M + H, M:M + W], ((0, 0), (M, M), (M, M)), mode="reflect")
    return X, flag, grid, t_sat, info


def load_statics(a, norm, grid):
    params = norm[4]
    awos_st = parse_awos_stationlist(os.path.join(a.awos_dir, "stationlist_awos.csv"))
    ids = sorted(awos_st)
    lat = np.array([awos_st[s][0] for s in ids])
    lon = np.array([awos_st[s][1] for s in ids])
    elev = np.array([awos_st[s][2] for s in ids], np.float64)
    ykm, xkm = grid.to_km(lat, lon)
    dem = np.load(a.dem).astype(np.float32)
    dem[np.isnan(dem)] = 0.0
    if dem.shape != grid.shape:
        raise SystemExit(f"DEM {dem.shape} != grid {grid.shape}")
    if a.radar_csv.lower().endswith(".json"):
        sites = {k: tuple(v) for k, v in json.load(open(a.radar_csv)).items()}
    else:
        sites = load_radar_sites(a.radar_csv)
    masks = radar_circle_masks(sites, grid)
    static_cov = np.zeros(grid.shape, np.float32)
    for m in masks.values():
        np.maximum(static_cov, m, out=static_cov)
    radar_dir = a.radar_dir if a.radar_dir else params.get("radar_dir")
    radar_index = None
    if radar_dir and os.path.isdir(radar_dir):
        radar_index = index_radar_files(radar_dir)
    elif a.radar_names and os.path.exists(a.radar_names):          # yalnız test: dosya adları listesi
        tmp = {}
        for name in open(a.radar_names).read().split():
            try:
                t = dt.datetime.strptime(name[3:15], "%y%m%d%H%M%S")
            except ValueError:
                continue
            tmp.setdefault(name[:3], []).append(t.replace(tzinfo=dt.timezone.utc).timestamp())
        radar_index = {k: sorted(v) for k, v in tmp.items()}
    else:
        print(f"UYARI: radar PNG klasörü yok ({radar_dir}); radar_cov statik birleşim (bayrak 16).")
    return dict(awos_ids=ids, awos_ykm=ykm, awos_xkm=xkm, awos_elev=elev, dem=dem,
                radar_masks=masks, static_cov=static_cov, radar_index=radar_index)


# ----------------------------------------------------------------------------- modeller
def resolve_models(a):
    """--exp/--seeds -> [(seed, ad, model)]; --ckpt verilirse doğrudan o ağırlık."""
    out = []
    seeds = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    for i, seed in enumerate(seeds):
        cfg, key = {}, None
        jp = os.path.join(a.results_dir, f"{a.exp}_s{seed}.json")
        if os.path.exists(jp):
            d = json.load(open(jp, encoding="utf-8"))
            key = next(iter(d))
            cfg = d[key]["config"]
        ck = a.ckpt.split(",")[i] if a.ckpt else os.path.join(a.results_dir, "checkpoints", cfg.get("best_model_file", ""))
        if not os.path.isfile(ck):
            raise SystemExit(f"Ağırlık yok: {ck}\n  (U-Net ağırlıkları Colab'da eğitildi: Drive'daki artifacts_v3/checkpoints/"
                             f"best_model_{a.exp}_s{seed}__*.pth dosyasını yerel artifacts_v3/checkpoints/'e kopyalayın.)")
        if cfg:
            if not cfg.get("aux"):
                raise SystemExit(f"{a.exp}: aux'suz model desteklenmiyor (tam harita aux ister).")
            if cfg.get("aux_channels") not in (None, AUX_CHANNELS) or cfg.get("base_channels", "all") != "all" \
                    or cfg.get("spatial_shuffle"):
                raise SystemExit(f"{a.exp}: yalnız 17 temel + 5 aux kanallı, karıştırmasız modeller (ör. B_full, D_unet).")
        is_unet = (key == "MTGUNet") if key else ("unet" in os.path.basename(ck).lower())
        if is_unet:
            net = MTGUNet(in_channels=22, num_classes=2)
        else:
            if key not in (None, "Medium_Flat"):
                raise SystemExit(f"{a.exp}: mimari {key} desteklenmiyor (Medium_Flat ya da MTGUNet).")
            net = MTGConvNet(in_channels=22, num_classes=2, conv_blocks=MEDIUM, use_gap=False)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        net.load_state_dict(sd)
        net.eval()
        out.append((seed, os.path.basename(ck), net, is_unet))
    if len({u for *_, u in out}) != 1:
        raise SystemExit("Seed'ler farklı mimaride.")
    return out


def _center_prob(net, is_unet, xb):
    o = net(xb)
    if is_unet:
        o = o[:, :, HALF, HALF]
    return torch.softmax(o, 1)[:, 1]


@torch.no_grad()
def predict_patches(net, is_unet, patches, batch):
    """patches: [N, 22, 33, 33] numpy/tensor -> [N] olasılık."""
    x = torch.as_tensor(patches)
    return torch.cat([_center_prob(net, is_unet, x[i:i + batch]) for i in range(0, len(x), batch)]).numpy()


@torch.no_grad()
def predict_sliding(net, is_unet, X, H, W, batch, rows_per_step=4, roi=None, log_every=50):
    """Her piksel için 33x33 yama (eğitimle birebir). roi=(y0,y1,x0,x1) yalnız o bölge."""
    M = MARGIN
    y0, y1, x0, x1 = roi if roi else (0, H, 0, W)
    out = np.full((H, W), np.nan, np.float32)
    Xt = torch.from_numpy(X)
    t0 = time.perf_counter()
    for k, r in enumerate(range(y0, y1, rows_per_step)):
        r1 = min(r + rows_per_step, y1)
        strip = Xt[:, r + M - HALF:r1 + M + HALF, x0 + M - HALF:x1 + M + HALF]
        pt = strip.unfold(1, PATCH, 1).unfold(2, PATCH, 1)                  # [C, h, w, 33, 33]
        C, h, w = pt.shape[:3]
        pt = pt.permute(1, 2, 0, 3, 4).reshape(h * w, C, PATCH, PATCH)
        pr = torch.cat([_center_prob(net, is_unet, pt[i:i + batch].contiguous()) for i in range(0, len(pt), batch)])
        out[r:r1, x0:x1] = pr.reshape(h, w).numpy()
        if log_every and k % log_every == 0 and k:
            done = (r1 - y0) / (y1 - y0)
            el = time.perf_counter() - t0
            print(f"   kayan pencere: %{100*done:.0f}  ({el:.0f} s, kalan ~{el*(1-done)/max(done,1e-9):.0f} s)")
    return out


@torch.no_grad()
def unet_pass(net, X, H, W, a_shift, b_shift):
    """Tek U-Net geçişi. Pikselin genişletilmiş dizideki indeksi i+16+a: (i+16+a) % 4 == 0 olan pikseller eğitimdeki merkezle
    aynı havuzlama fazındadır. Boyut 4k+1 olmalı (33 = 4*8+1; up1 output_padding=1)."""
    M = MARGIN
    r0, c0 = M - HALF - a_shift, M - HALF - b_shift
    Lh = H + 2 * HALF + a_shift
    Lh += (1 - Lh) % 4
    Lw = W + 2 * HALF + b_shift
    Lw += (1 - Lw) % 4
    x = torch.from_numpy(np.ascontiguousarray(X[None, :, r0:r0 + Lh, c0:c0 + Lw]))
    p = torch.softmax(net(x), 1)[0, 1].numpy()
    return p[HALF + a_shift:HALF + a_shift + H, HALF + b_shift:HALF + b_shift + W]


def predict_unet_full(net, X, H, W, mode):
    if mode == "single":
        return unet_pass(net, X, H, W, 0, 0)
    out = np.empty((H, W), np.float32)
    for a_ in range(4):
        for b_ in range(4):
            p = unet_pass(net, X, H, W, a_, b_)
            ri, ci = (-a_ - HALF) % 4, (-b_ - HALF) % 4            # (i+16+a) % 4 == 0
            out[ri::4, ci::4] = p[ri::4, ci::4]
    return out


# ----------------------------------------------------------------------------- yazıcılar
def write_netcdf(path, grid, prob, mask, flag, attrs):
    """h5py ile NetCDF4 uyumlu dosya (boyut ölçekleri + CF öznitelikleri)."""
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        lat = f.create_dataset("lat", data=grid.lats.astype(np.float64))
        lon = f.create_dataset("lon", data=grid.lons.astype(np.float64))
        lat.attrs.update(units="degrees_north", standard_name="latitude", long_name="latitude")
        lon.attrs.update(units="degrees_east", standard_name="longitude", long_name="longitude")
        lat.make_scale("lat")
        lon.make_scale("lon")
        kw = dict(compression="gzip", compression_opts=4, chunks=(200, 250))
        v = f.create_dataset("cloud_prob", data=prob.astype(np.float32), fillvalue=np.float32(np.nan), **kw)
        v.attrs.update(long_name="probability of cloud (>= 5 okta) at the pixel", units="1",
                       valid_min=np.float32(0), valid_max=np.float32(1))
        m = f.create_dataset("cloud_mask", data=mask.astype(np.int8), fillvalue=np.int8(-1), **kw)
        m.attrs.update(long_name="binary cloud mask", flag_values=np.array([-1, 0, 1], np.int8),
                       flag_meanings="invalid clear cloudy", units="1")
        q = f.create_dataset("quality_flag", data=flag.astype(np.uint8), **kw)
        q.attrs.update(long_name="input quality flag", units="1",
                       flag_masks=np.array([1, 2, 4, 8, 16], np.uint8),
                       flag_meanings="satellite_nan no_radar_file awos_filled outside_radar_coverage static_radar_coverage")
        for ds in (v, m, q):
            ds.dims[0].attach_scale(lat)
            ds.dims[1].attach_scale(lon)
        f.attrs.update({k: (json.dumps(x) if isinstance(x, (dict, list)) else x) for k, x in attrs.items() if x is not None})
        f.attrs["Conventions"] = "CF-1.8"
    os.replace(tmp, path)


def write_geotiff(path, grid, arr):
    """EPSG:4326 GeoTIFF (float32), PIL ile; rasterio gerekmez. lats[0], lons[0] piksel MERKEZİ (PixelIsArea)."""
    from PIL import Image, TiffImagePlugin
    info = TiffImagePlugin.ImageFileDirectory_v2()
    dx, dy = abs(grid.dlon), abs(grid.dlat)
    info[33550] = (dx, dy, 0.0)                                                   # ModelPixelScale
    info.tagtype[33550] = 12
    info[33922] = (0.0, 0.0, 0.0, float(grid.lons[0]) - dx / 2, float(grid.lats[0]) + dy / 2, 0.0)  # ModelTiepoint
    info.tagtype[33922] = 12
    info[34735] = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)   # GeoKeyDirectory
    info.tagtype[34735] = 3
    info[42113] = "nan"                                                           # GDAL_NODATA
    info.tagtype[42113] = 2
    if grid.dlat > 0:      # kuzey yukarıda olmalı
        arr = arr[::-1]
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32), mode="F").save(path, tiffinfo=info)


def write_png(path, grid, prob, flag, title, stations=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [grid.lons[0] - grid.dlon / 2, grid.lons[-1] + grid.dlon / 2,
           grid.lats[-1] + grid.dlat / 2, grid.lats[0] - grid.dlat / 2]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=110)
    show = np.where((flag & (FLAG_SAT_NAN | FLAG_NO_RADAR)) == 0, prob, np.nan)
    im = ax.imshow(show, extent=ext, origin="upper" if grid.dlat < 0 else "lower", cmap="cividis", vmin=0, vmax=1,
                   interpolation="nearest")
    if stations is not None and len(stations):
        lab = stations["label"]
        for val, mk in ((0, "o"), (1, "o")):
            s = lab == val
            ax.scatter(stations["lon"][s], stations["lat"][s], s=14, marker=mk,
                       facecolors="none" if val == 0 else "white", edgecolors="red", linewidths=0.8)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="Cloud probability", shrink=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- tutarlılık testi
def consistency_check(a, X, H, W, sat_ts, models, prob_map, batch):
    """H5'teki aynı zaman satırları: girdi farkı, yeniden üretilen yamadan olasılık farkı, tam sahne farkı."""
    M = MARGIN
    with h5py.File(a.h5, "r") as f:
        tsd = f["timestamps"][:]
        tsd = np.array([t.decode() if isinstance(t, bytes) else str(t) for t in tsd])
        rows = np.flatnonzero(tsd == sat_ts)
        if len(rows) == 0:
            return {"n_rows": 0, "note": f"H5'te {sat_ts} zamanlı satır yok (bu zaman eğitim/test örneği değil)."}
        sy = f["station_y"][:][rows].astype(int)
        sx = f["station_x"][:][rows].astype(int)
        h5_patch = f["patches"][rows].astype(np.float32)
        h5_aux = f["aux"][rows].astype(np.float32)
        aux_means = np.asarray(f.attrs["aux_means"], np.float32)[:, None, None]
        aux_stds = np.asarray(f.attrs["aux_stds"], np.float32)[:, None, None]
        labels = f["labels"][:][rows]
    h5_in = np.concatenate([h5_patch, (h5_aux - aux_means) / aux_stds], axis=1)
    ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
    rec = np.stack([X[:, y + M - HALF:y + M + HALF + 1, x + M - HALF:x + M + HALF + 1] if o else np.zeros_like(X[:, :PATCH, :PATCH])
                    for y, x, o in zip(sy, sx, ok)])
    names = FCI_CHANNELS + ["radar"] + AUX_CHANNELS
    d_in = np.abs(rec - h5_in)[ok]
    res = {"n_rows": int(len(rows)), "n_in_grid": int(ok.sum()),
           "input_absdiff_max": {n: float(d_in[:, c].max()) for c, n in enumerate(names)},
           "input_absdiff_median": {n: float(np.median(d_in[:, c])) for c, n in enumerate(names)}}

    # kayıtlı tahminler (seed ortalaması) — val/test kümelerinde olan satırlar
    saved = {}
    for seed, ck, net, is_unet in models:
        stem = os.path.splitext(ck)[0].replace("best_model_", "")
        pp = os.path.join(a.results_dir, "preds", stem + ".npz")
        if not os.path.exists(pp):
            continue
        z = np.load(pp)
        for k in z.files:
            if k.endswith("_rows") and not k.startswith("last_"):
                pk = k[:-5] + "_prob"
                if pk in z.files:
                    for r, p in zip(z[k], z[pk]):
                        saved.setdefault(int(r), []).append(float(p))
    has = np.array([int(r) in saved and len(saved[int(r)]) == len(models) for r in rows]) & ok
    p_h5 = np.mean([predict_patches(net, u, h5_in, batch) for _, _, net, u in models], axis=0)
    p_rec = np.mean([predict_patches(net, u, rec, batch) for _, _, net, u in models], axis=0)
    p_map = np.where(ok, prob_map[np.clip(sy, 0, H - 1), np.clip(sx, 0, W - 1)], np.nan)

    def stats(d):
        d = np.abs(d[np.isfinite(d)])
        if not len(d):
            return None
        return {"n": int(len(d)), "median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())}

    res["prob_h5patch_vs_saved"] = stats(np.array([p_h5[i] - np.mean(saved[int(r)]) for i, r in enumerate(rows) if has[i]]))
    res["prob_rebuilt_vs_saved"] = stats(np.array([p_rec[i] - np.mean(saved[int(r)]) for i, r in enumerate(rows) if has[i]]))
    res["prob_rebuilt_vs_h5patch"] = stats((p_rec - p_h5)[ok])
    res["prob_map_vs_h5patch"] = stats((p_map - p_h5)[ok])
    res["prob_map_vs_saved"] = stats(np.array([p_map[i] - np.mean(saved[int(r)]) for i, r in enumerate(rows) if has[i]]))
    lab = labels[ok]
    good = (lab >= 0) & np.isfinite(p_map[ok])
    if good.any():
        res["acc_map_on_labels"] = float(((p_map[ok][good] >= a.threshold).astype(int) == lab[good]).mean())
        res["acc_h5patch_on_labels_same_rows"] = float(((p_h5[ok][good] >= a.threshold).astype(int) == lab[good]).mean())
        res["n_labelled"] = int(good.sum())
    ref = res["prob_rebuilt_vs_h5patch"]
    res["verdict"] = ("OK: yeniden üretilen girdi eğitimle uyumlu" if ref and ref["median"] < 1e-3
                      else "UYUMSUZ: aux/normalizasyon farkı — çıktıları kullanmadan önce inceleyin")
    return res, (sy[ok], sx[ok], labels[ok])


def artifact_check(net, X, H, W, p_aligned):
    """Tek geçiş U-Net ile faz-hizalı çıktının farkı ve 4x4 faz sınıflarına göre ortalama sapma."""
    p1 = unet_pass(net, X, H, W, 0, 0)
    d = p1 - p_aligned
    ph = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            ph[i, j] = float(np.mean(np.abs(d[i::4, j::4])))
    m1 = (p1 >= 0.5) != (p_aligned >= 0.5)
    # 33 piksellik yama ızgarası yok (kayan/tam evrişimli); yine de 33 periyotlu desen var mı: satır/sütun ortalamalarının
    # 33 ve 4 periyottaki spektral gücü (ortalamaya göre)
    def period_power(p, period):
        s = p.mean(axis=0) - p.mean()
        f = np.abs(np.fft.rfft(s)) ** 2
        k = int(round(len(s) / period))
        return float(f[k] / max(f[1:].mean(), 1e-12)) if 0 < k < len(f) else None
    return {"single_vs_aligned_absdiff_mean": float(np.abs(d).mean()), "single_vs_aligned_absdiff_p99": float(np.percentile(np.abs(d), 99)),
            "mask_disagreement_frac": float(m1.mean()), "phase_absdiff_4x4": ph.round(5).tolist(),
            "col_power_ratio_period4_single": period_power(p1, 4), "col_power_ratio_period4_aligned": period_power(p_aligned, 4),
            "col_power_ratio_period33_aligned": period_power(p_aligned, 33)}


# ----------------------------------------------------------------------------- ana akış
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="Eğitim H5'i (normalizasyon attrs; --check için satırlar)")
    ap.add_argument("--time", default=None, help="UTC, virgüllü: 202605021200,202605021500")
    ap.add_argument("--nc", default=None, help="Doğrudan sahne dosyası/dosyaları (virgüllü); --time yerine")
    ap.add_argument("--nc_dir", default="F:/turkiye_kesitler")
    ap.add_argument("--radar_npy_dir", default="F:/radar/radar_aligned")
    ap.add_argument("--radar_dir", default=None, help="Ham radar PNG klasörü (zaman-bağımlı radar_cov); varsayılan H5 aux_params")
    ap.add_argument("--radar_names", default=None, help=argparse.SUPPRESS)       # test: PNG adları listesi
    ap.add_argument("--radar_csv", default=os.path.join(ROOT, "data/raw_radar/radar.csv"))
    ap.add_argument("--awos_dir", default=os.path.join(ROOT, "data/raw_awos_observation"))
    ap.add_argument("--awos_csv", default=None, help="Varsayılan <awos_dir>/observation_awos.csv")
    ap.add_argument("--awos_cache", default=os.path.join(ROOT, "data/processed/awos_scene_cache"))
    ap.add_argument("--dem", default=os.path.join(ROOT, "data/dem_800x1000.npy"))
    ap.add_argument("--results_dir", default=os.path.join(ROOT, "artifacts_v3"))
    ap.add_argument("--exp", default="D_unet")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--ckpt", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--unet_mode", choices=["aligned", "single", "patch"], default="aligned")
    ap.add_argument("--pad", choices=["train", "reflect"], default="train")
    ap.add_argument("--threshold", type=float, default=None, help="Varsayılan 0,5 (eğitimde kullanılan karar eşiği)")
    ap.add_argument("--roi", default=None, help="y0:y1,x0:x1 — kayan pencerede yalnız bu bölge (test)")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="H5 satırları ve preds/ ile tutarlılık testi")
    ap.add_argument("--artifact_check", action="store_true", help="U-Net tek geçiş vs faz-hizalı karşılaştırması")
    ap.add_argument("--no_png", action="store_true")
    ap.add_argument("--no_tif", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/scenes"))
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    a.threshold = 0.5 if a.threshold is None else a.threshold
    os.makedirs(a.out, exist_ok=True)
    if a.nc:
        scenes = a.nc.split(",")
    elif a.time:
        scenes = [find_scene(a.nc_dir, t.strip()) for t in a.time.split(",")]
    else:
        raise SystemExit("--time ya da --nc verin.")

    norm = load_norm(a.h5)
    models = resolve_models(a)
    is_unet = models[0][3]
    mode = (a.unet_mode if is_unet else "sliding")
    grid0 = LatLonGrid.from_nc(scenes[0])
    statics = load_statics(a, norm, grid0)
    params = norm[4]
    keys = []
    for s in scenes:
        t = dt.datetime.strptime(scene_ts(s), "%Y%m%d%H%M%S")
        t = (floor_to_10min if params.get("time_match", "floor") == "floor" else nearest_10min)(t)
        keys.append(f"{t.year};{t.month};{t.day};{t.hour};{t.minute}")
    awos = awos_fields(sorted(set(keys)), statics["awos_ids"],
                       a.awos_csv or os.path.join(a.awos_dir, "observation_awos.csv"), a.awos_cache)

    for nc in scenes:
        tm = Timer()
        print(f"\n=== {os.path.basename(nc)}  ({a.exp}, seeds {a.seeds}, mod {mode}, dolgu {a.pad}) ===")
        with tm("inputs"):
            X, flag, grid, t_sat, info = build_inputs(nc, a, norm, statics, awos)
        H, W = grid.shape
        roi = tuple(int(v) for v in a.roi.replace(":", ",").split(",")) if a.roi else None
        if roi:
            roi = (roi[0], roi[1], roi[2], roi[3])
        probs = []
        with tm("model"):
            for seed, ck, net, u in models:
                if u and mode in ("aligned", "single"):
                    probs.append(predict_unet_full(net, X, H, W, mode))
                else:
                    probs.append(predict_sliding(net, u, X, H, W, a.batch, roi=roi))
        prob = np.mean(probs, axis=0).astype(np.float32)
        valid = ((flag & (FLAG_SAT_NAN | FLAG_NO_RADAR)) == 0) & np.isfinite(prob)
        mask = np.where(valid, (prob >= a.threshold).astype(np.int8), np.int8(-1))

        res = {"scene": info, "exp": a.exp, "seeds": a.seeds, "checkpoints": [ck for _, ck, _, _ in models],
               "mode": mode, "pad": a.pad, "threshold": a.threshold, "roi": roi,
               "n_threads": torch.get_num_threads(),
               "valid_frac": float(valid.mean()), "cloud_frac_valid": float(mask[valid].mean()) if valid.any() else None,
               "flag_frac": {n: float(((flag & b) > 0).mean()) for n, b in
                             (("sat_nan", 1), ("no_radar_file", 2), ("awos_filled", 4), ("outside_radar_cov", 8), ("static_cov", 16))}}
        st = None
        if a.check:
            with tm("check"):
                chk = consistency_check(a, X, H, W, info["sat_time"], models, prob, min(a.batch, 512))
            if isinstance(chk, tuple):
                chk, (cy, cx, cl) = chk
                st = {"lat": grid.lats[cy], "lon": grid.lons[cx], "label": cl}
            res["check"] = chk
            print("Tutarlılık:", json.dumps({k: v for k, v in chk.items() if not k.startswith("input_absdiff")}, ensure_ascii=False))
            if "input_absdiff_max" in chk:
                worst = sorted(chk["input_absdiff_max"].items(), key=lambda kv: -kv[1])[:5]
                print("  girdi farkı (en büyük 5, normalize uzay):", ", ".join(f"{n} {v:.4f}" for n, v in worst))
        if a.artifact_check and is_unet and mode == "aligned" and not roi:
            with tm("artifact_check"):
                res["artifact"] = artifact_check(models[0][2], X, H, W, probs[0])
            print("Faz deseni:", json.dumps(res["artifact"], ensure_ascii=False))

        stem = os.path.join(a.out, f"scene_{info['sat_time']}_{a.exp}" + ("" if a.seeds == "1" else f"_s{a.seeds.replace(',', '')}")
                            + ("" if mode in ("aligned", "sliding") else f"_{mode}") + ("" if a.pad == "train" else f"_{a.pad}"))
        with tm("write"):
            attrs = {"title": "MTGCLM cloud mask", "source": "MTG FCI L1c + TSMS C-band radar + AWOS",
                     "model": a.exp, "seeds": a.seeds, "mode": mode, "threshold": a.threshold,
                     "satellite_time_utc": info["sat_time"], "awos_time_utc": info["awos_time"],
                     "radar_file": info.get("radar_file") or "missing", "parallax_corrected": "no",
                     "pixel_size_km": "approx. 1.73 (W-E) x 1.11 (S-N) at 39N",
                     "history": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ} src/infer_scene.py"}
            write_netcdf(stem + ".nc", grid, np.where(valid, prob, np.nan), mask, flag, attrs)
            if not a.no_tif:
                write_geotiff(stem + "_prob.tif", grid, np.where(valid, prob, np.nan))
            if not a.no_png:
                write_png(stem + ".png", grid, prob, flag,
                          f"{a.exp} (seeds {a.seeds}, {mode})  {t_sat:%Y-%m-%d %H:%M:%S} UTC", st)
        res["seconds"] = tm.t
        res["seconds_total"] = round(sum(tm.t.values()), 2)
        res["peak_mem_mb"] = tm.mem
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        tp = os.path.join(a.out, "timing.csv")
        new = not os.path.exists(tp)
        with open(tp, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["scene", "exp", "seeds", "mode", "threads", "inputs_s", "model_s", "check_s", "write_s",
                            "total_s", "peak_mem_mb", "roi", "host"])
            w.writerow([info["sat_time"], a.exp, a.seeds, mode, torch.get_num_threads(), tm.t.get("inputs"), tm.t.get("model"),
                        tm.t.get("check"), tm.t.get("write"), res["seconds_total"], tm.mem, a.roi or "", os.environ.get("COMPUTERNAME", "")])
        print(f"Yazıldı: {stem}.nc/.json  | süre {res['seconds_total']} s ({tm.t}) | tepe bellek {tm.mem} MB | "
              f"bulutlu {res['cloud_frac_valid']}")


if __name__ == "__main__":
    main()
