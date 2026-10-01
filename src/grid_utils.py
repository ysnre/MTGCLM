"""
Ortak grid / coğrafya yardımcıları.

- Uydu grid'i düzenli enlem/boylam grid'idir (x = boylam 1D, y = enlem 1D).
  Örnek dosyada: lon 25.01..44.99 (adım +0.02°), lat 42.995..35.005 (adım -0.01°),
  yani piksel ~1.1 km (K-G) x ~1.6 km (D-B).
- Tüm mesafeler equirectangular projeksiyonla km cinsinden hesaplanır (Türkiye
  ölçeğinde IDW/kapsama için yeterli).
- cos(SZA) NOAA güneş konumu yaklaşımıyla (numpy, ek bağımlılık yok) hesaplanır.
"""
import datetime as _dt
import numpy as np

KM_PER_DEG_LAT = 110.574


class LatLonGrid:
    """Düzenli enlem/boylam grid'i. Satır = enlem (y), sütun = boylam (x)."""

    def __init__(self, lons_1d: np.ndarray, lats_1d: np.ndarray):
        self.lons = np.asarray(lons_1d, dtype=np.float64)
        self.lats = np.asarray(lats_1d, dtype=np.float64)
        self.ny = len(self.lats)
        self.nx = len(self.lons)
        self.dlon = float(np.mean(np.diff(self.lons)))
        self.dlat = float(np.mean(np.diff(self.lats)))
        self.lat0_ref = float(np.mean(self.lats))
        self.lon0_ref = float(np.mean(self.lons))
        self.km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.deg2rad(self.lat0_ref))

    # ---------- kurucular ----------
    @classmethod
    def from_nc(cls, nc_path: str) -> "LatLonGrid":
        import h5py
        with h5py.File(nc_path, "r") as f:
            return cls(f["x"][:], f["y"][:])

    @classmethod
    def from_params(cls, lon0: float, dlon: float, nx: int, lat0: float, dlat: float, ny: int) -> "LatLonGrid":
        return cls(lon0 + dlon * np.arange(nx), lat0 + dlat * np.arange(ny))

    @classmethod
    def default_turkiye(cls) -> "LatLonGrid":
        """turkiye_kesit_*.nc dosyalarındaki grid (800 x 1000)."""
        return cls.from_params(25.01, 0.02, 1000, 42.995, -0.01, 800)

    @property
    def shape(self):
        return (self.ny, self.nx)

    # ---------- dönüşümler ----------
    def to_km(self, lat, lon):
        """(lat, lon) -> (y_km, x_km) referans noktasına göre."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        return (lat - self.lat0_ref) * KM_PER_DEG_LAT, (lon - self.lon0_ref) * self.km_per_deg_lon

    def latlon_to_index(self, lat, lon):
        """En yakın (y, x) piksel indeksleri (grid dışına taşabilir; clip edilmez)."""
        y = np.rint((np.asarray(lat, dtype=np.float64) - self.lats[0]) / self.dlat).astype(np.int64)
        x = np.rint((np.asarray(lon, dtype=np.float64) - self.lons[0]) / self.dlon).astype(np.int64)
        return y, x

    def window_latlon(self, yc: int, xc: int, half: int):
        """(yc, xc) merkezli (2*half+1)^2 pencerenin lat/lon 2D dizileri (grid dışı lineer ekstrapolasyon)."""
        ys = np.arange(yc - half, yc + half + 1)
        xs = np.arange(xc - half, xc + half + 1)
        lat = self.lats[0] + self.dlat * ys
        lon = self.lons[0] + self.dlon * xs
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d, lon2d

    def full_latlon(self):
        lon2d, lat2d = np.meshgrid(self.lons, self.lats)
        return lat2d, lon2d


def cos_solar_zenith(lat_deg, lon_deg, when_utc: _dt.datetime) -> np.ndarray:
    """
    cos(güneş zenit açısı). NOAA Solar Calculator yaklaşımı (hata < ~0.1°).
    lat_deg, lon_deg: aynı şekilli numpy dizileri (derece). when_utc: naive UTC datetime.
    """
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.asarray(lon_deg, dtype=np.float64)
    doy = when_utc.timetuple().tm_yday
    hour = when_utc.hour + when_utc.minute / 60.0 + when_utc.second / 3600.0
    days_in_year = 366 if _is_leap(when_utc.year) else 365
    g = 2.0 * np.pi / days_in_year * (doy - 1 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    time_offset = eqtime + 4.0 * lon  # dakika (UTC için timezone terimi yok)
    tst = hour * 60.0 + time_offset
    ha = np.deg2rad(tst / 4.0 - 180.0)
    cz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return np.clip(cz, -1.0, 1.0)


def _is_leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def parse_sat_timestamp(ts) -> _dt.datetime:
    """H5 'timestamps' alanı: b'YYYYMMDDHHMMSS' veya str."""
    if isinstance(ts, (bytes, np.bytes_)):
        ts = ts.decode("utf-8")
    return _dt.datetime.strptime(str(ts)[:14], "%Y%m%d%H%M%S")


def floor_to_10min(t: _dt.datetime) -> _dt.datetime:
    """Uydu anını kapsayan 10 dakikalık AWOS penceresinin başlangıcı."""
    return t.replace(minute=(t.minute // 10) * 10, second=0, microsecond=0)


def nearest_10min(t: _dt.datetime) -> _dt.datetime:
    base = floor_to_10min(t)
    if (t - base) >= _dt.timedelta(minutes=5):
        base += _dt.timedelta(minutes=10)
    return base


def padded_window(field: np.ndarray, yc: int, xc: int, half: int, fill=0.0) -> np.ndarray:
    """2D alandan (yc, xc) merkezli pencere; grid dışı 'fill' ile doldurulur."""
    ny, nx = field.shape
    y0, y1 = yc - half, yc + half + 1
    x0, x1 = xc - half, xc + half + 1
    out = np.full((2 * half + 1, 2 * half + 1), fill, dtype=field.dtype)
    sy0, sy1 = max(y0, 0), min(y1, ny)
    sx0, sx1 = max(x0, 0), min(x1, nx)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = field[sy0:sy1, sx0:sx1]
    return out
