"""
extract_nwcsaf_ct.py
====================
MGM'den alınan NWC SAF Cloud Type (CT) PNG'lerinden (MSG tabanlı, 15 dk) istasyon çevresi sınıf sayımlarını çıkarır.
FCI CLM çıkarımıyla (src/extract_fci_clm.py) aynı mantık ve aynı alan grupları: merkez, 3x3, r5/r10/r20/r30 km.

PNG özellikleri (12 Eki 2026'da incelendi):
  - 1550 x 1067, RGBA, kayıpsız; yalnız palet renkleri var (bulutsuz kara/deniz ve büyük olasılıkla veri yok BEYAZ).
  - Projeksiyon Mercator (kare piksel ~0,02223°). Köşeler (MGM, piksel dış kenarları kabul edildi):
      sol-üst  49.203243 K, 12.497211 D      sağ-alt  31.269161 K, 46.958802 D
    CLM ile çapraz korelasyonla bağımsız olarak da doğrulandı (iki tarihte ±0,1°).
Sınıf kodları (bu betiğin çıktısı, palet: .agent/rules/nwcsaf_Palette_CT.png):
   0 beyaz (bulutsuz kara/deniz ya da veri yok)   1 bulutsuz kara   2 bulutsuz deniz   3 kara üzerinde kar   4 deniz buzu
   5 çok alçak   6 alçak   7 orta seviye   8 yüksek opak   9 çok yüksek opak   10 fraksiyonel
  11 yüksek yarı saydam ince   12 ... orta kalın   13 ... kalın   14 alçak/orta bulut üstünde yüksek yarı saydam
  15 kar/buz üstünde yüksek yarı saydam   255 tanınmayan renk
İkili karar için öneri: bulutlu = 5..15 (tüm bulut tipleri; 10 = fraksiyonel ayrıca raporlanmalı), açık = 0..4.

Kullanım (proje kökünden; yalnız numpy + Pillow gerekir, kaldığı yerden devam eder):
    python src/extract_nwcsaf_ct.py --png_dir F:/nwcsaf --stations data/raw_observation/station_list.csv ^
        --out data/processed/nwcsaf_station
Zaman: dosya adındaki YYYYmmddHHMM MSG tarama başlangıcıdır; Türkiye ~9 dk sonra taranır. Model örneğiyle eşleme için
    slot = (örnek zamanı - 9 dk) en yakın 15 dakikaya yuvarlanır (sample_time_to_slot()).
"""
import argparse, glob, os, re, time
import numpy as np

UL_LAT, UL_LON, LR_LAT, LR_LON = 49.203243, 12.497211, 31.269161, 46.958802
W, H = 1550, 1067
GROUPS = ["c1", "3x3", "r5", "r10", "r20", "r30"]
RADII_KM = {"r5": 5.0, "r10": 10.0, "r20": 20.0, "r30": 30.0}
NCLS = 16
PALETTE = {  # RGB (palet görüntüsündeki değerler; PNG'lerde -1/-2 kayma olabilir, en yakın renk kullanılır)
    (255, 255, 255): 0, (0, 120, 0): 1, (0, 0, 0): 2, (250, 190, 250): 3, (220, 160, 220): 4,
    (255, 100, 0): 5, (255, 180, 0): 6, (240, 240, 0): 7, (215, 215, 150): 8, (230, 230, 230): 9,
    (200, 0, 200): 10, (0, 80, 215): 11, (0, 180, 230): 12, (0, 240, 240): 13, (90, 200, 160): 14, (200, 175, 250): 15,
}
CLASS_NAMES = ["beyaz(açık/veri yok)", "bulutsuz kara", "bulutsuz deniz", "kar (kara)", "deniz buzu", "çok alçak", "alçak",
               "orta", "yüksek opak", "çok yüksek opak", "fraksiyonel", "yüksek yarı saydam ince", "yüksek yarı saydam orta",
               "yüksek yarı saydam kalın", "alçak/orta üstü yüksek yarı saydam", "kar/buz üstü yüksek yarı saydam"]
FNAME_RE = re.compile(r"^(\d{12})_CT")


def merc(lat):
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


def imerc(y):
    return np.degrees(2 * np.arctan(np.exp(np.radians(y))) - np.pi / 2)


DX = (LR_LON - UL_LON) / W
DY = (merc(UL_LAT) - merc(LR_LAT)) / H


def pixel_of(lat, lon):
    col = np.floor((np.asarray(lon) - UL_LON) / DX).astype(int)
    row = np.floor((merc(UL_LAT) - merc(np.asarray(lat))) / DY).astype(int)
    return row, col


def pixel_center(row, col):
    return imerc(merc(UL_LAT) - (np.asarray(row) + 0.5) * DY), UL_LON + (np.asarray(col) + 0.5) * DX


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def build_index(stations_csv):
    import pandas as pd
    st = pd.read_csv(stations_csv).dropna(subset=["wmoid", "latitude", "longitude"]).drop_duplicates("wmoid")
    wmoid = st.wmoid.astype(np.int64).values; lat = st.latitude.values.astype(float); lon = st.longitude.values.astype(float)
    row, col = pixel_of(lat, lon)
    ix = {"wmoid": wmoid, "row": row, "col": col}
    pad = 20  # piksel (~38 km)
    for g in GROUPS:
        idx_all, ptr = [], [0]
        for k in range(len(wmoid)):
            r, c = int(row[k]), int(col[k])
            if g == "c1":
                rr, cc = np.array([r]), np.array([c])
            elif g == "3x3":
                rr, cc = [a.ravel() for a in np.meshgrid(np.arange(r - 1, r + 2), np.arange(c - 1, c + 2), indexing="ij")]
            else:
                R0, C0 = np.meshgrid(np.arange(r - pad, r + pad + 1), np.arange(c - pad, c + pad + 1), indexing="ij")
                plat, plon = pixel_center(R0, C0)
                m = haversine_km(lat[k], lon[k], plat, plon) <= RADII_KM[g]
                rr, cc = R0[m], C0[m]
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            idx = (rr[ok] * W + cc[ok]).astype(np.int64)
            idx_all.append(idx); ptr.append(ptr[-1] + idx.size)
        ix[f"idx_{g}"] = np.concatenate(idx_all); ix[f"ptr_{g}"] = np.array(ptr)
        ix[f"sid_{g}"] = np.repeat(np.arange(len(wmoid)), np.diff(ix[f"ptr_{g}"]))
    ix["union"] = np.unique(np.concatenate([ix[f"idx_{g}"] for g in GROUPS]))
    return ix


_PAL_RGB = np.array(list(PALETTE.keys()), dtype=np.int32)
_PAL_CLS = np.array(list(PALETTE.values()), dtype=np.uint8)


def to_classes(rgb_pixels):
    """[n,3] RGB -> sınıf kodu (en yakın palet rengi; uzaklık > 12 ise 255). Benzersiz renkler üzerinden hızlı."""
    code = (rgb_pixels[:, 0].astype(np.int32) << 16) | (rgb_pixels[:, 1].astype(np.int32) << 8) | rgb_pixels[:, 2]
    u, inv = np.unique(code, return_inverse=True)
    urgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1)
    d = ((urgb[:, None, :] - _PAL_RGB[None]) ** 2).sum(-1)
    j = d.argmin(1)
    ucls = _PAL_CLS[j].copy()
    ucls[d[np.arange(len(j)), j] > 144] = 255
    return ucls[inv.ravel()]


def process_png(path, ix):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3)
    cls_all = np.full(W * H, 255, np.uint8)
    u = ix["union"]
    cls_all[u] = to_classes(a[u])
    S = len(ix["wmoid"])
    counts = np.zeros((S, len(GROUPS), NCLS), np.uint16)
    for gi, g in enumerate(GROUPS):
        cls = cls_all[ix[f"idx_{g}"]]
        ok = cls < NCLS
        counts[:, gi, :] = np.bincount(ix[f"sid_{g}"][ok] * NCLS + cls[ok], minlength=S * NCLS).reshape(S, NCLS)
    return counts, cls_all[ix["idx_c1"]]


def run(png_dir, stations_csv, out_dir, force=False):
    ix = build_index(stations_csv)
    files = sorted(glob.glob(os.path.join(png_dir, "*_CT*.png")))
    by_day = {}
    for f in files:
        m = FNAME_RE.match(os.path.basename(f))
        if m:
            by_day.setdefault(m.group(1)[:8], []).append((m.group(1), f))
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(files)} PNG, {len(by_day)} gün -> {out_dir}  (piksel {DX:.5f}° x {DY:.5f}° Mercator)")
    t0 = time.time(); done = 0
    for day in sorted(by_day):
        outp = os.path.join(out_dir, f"nwc_{day}.npz")
        if os.path.exists(outp) and not force:
            continue
        items = sorted(by_day[day]); F, S = len(items), len(ix["wmoid"])
        counts = np.zeros((F, S, len(GROUPS), NCLS), np.uint16); center = np.full((F, S), 255, np.uint8); ok = np.ones(F, bool)
        for k, (slot, f) in enumerate(items):
            try:
                counts[k], center[k] = process_png(f, ix)
            except Exception as e:
                ok[k] = False; print(f"  ! okunamadı: {os.path.basename(f)} ({e})")
        tmp = outp + ".tmp.npz"
        np.savez_compressed(tmp, slot=np.array([i[0] for i in items]), ok=ok, counts=counts, center_class=center,
                            wmoid=ix["wmoid"], groups=np.array(GROUPS), class_names=np.array(CLASS_NAMES))
        os.replace(tmp, outp); done += F
        print(f"  {day}: {F} dosya (toplam {done}, {done / max(time.time() - t0, 1e-6):.1f} dosya/s)")
    print(f"Bitti ({(time.time() - t0) / 60:.1f} dk).")


def sample_time_to_slot(ts):
    """Model örneği zaman damgası ('YYYYmmddHHMMSS') -> NWC/MSG slotu ('YYYYmmddHHMM')."""
    import pandas as pd
    t = pd.to_datetime(pd.Series(ts).astype(str).str[:14], format="%Y%m%d%H%M%S") - pd.Timedelta("9min")
    return t.dt.round("15min").dt.strftime("%Y%m%d%H%M").values


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png_dir", default="F:/nwcsaf")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--out", default="data/processed/nwcsaf_station")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.png_dir, a.stations, a.out, a.force)
