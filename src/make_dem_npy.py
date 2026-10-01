"""
make_dem_npy.py
===============
Herhangi bir DEM GeoTIFF'ini (SRTM, GMTED2010, Copernicus GLO-30/90, ASTER; herhangi bir CRS/çözünürlük)
uydu grid'ine (800 x 1000, EPSG:4326, piksel merkezleri lon 25.01 + 0.02*i, lat 42.995 - 0.01*j)
ortalama alarak yeniden örnekler ve metre cinsinden float32 .npy yazar.

Gereksinim: pip install rasterio
Kullanım:
    python src/make_dem_npy.py --dem_tif data/dem/copernicus_glo90_turkiye.tif --out data/dem_800x1000.npy
    (birden fazla karo varsa önce gdal_merge / rasterio.merge ile birleştirin ya da --dem_tif'i birden çok kez verin)
Sonra: update_h5_aux_channels.py ... --dem data/dem_800x1000.npy
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from grid_utils import LatLonGrid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem_tif", nargs="+", required=True, help="Bir ya da daha çok DEM GeoTIFF (karolar birleştirilir)")
    ap.add_argument("--out", default="data/dem_800x1000.npy")
    ap.add_argument("--sample_nc", default=None, help="Grid için örnek .nc (yoksa varsayılan Türkiye grid'i)")
    ap.add_argument("--sea_level_fill", type=float, default=0.0, help="Veri olmayan / deniz pikselleri için değer")
    args = ap.parse_args()

    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.merge import merge
        from rasterio.transform import from_origin
        from rasterio.warp import reproject
    except ImportError:
        raise SystemExit("rasterio gerekli: pip install rasterio")

    grid = LatLonGrid.from_nc(args.sample_nc) if args.sample_nc else LatLonGrid.default_turkiye()
    # piksel merkezleri -> sol/üst kenar
    left = grid.lons[0] - grid.dlon / 2.0
    top = grid.lats[0] - grid.dlat / 2.0          # dlat negatif -> top = lat0 + 0.005
    dst_transform = from_origin(left, top, abs(grid.dlon), abs(grid.dlat))
    dst = np.full(grid.shape, np.nan, dtype=np.float32)

    srcs = [rasterio.open(p) for p in args.dem_tif]
    if len(srcs) == 1:
        src_arr = srcs[0].read(1).astype(np.float32)
        src_transform, src_crs, nodata = srcs[0].transform, srcs[0].crs, srcs[0].nodata
    else:
        src_arr, src_transform = merge(srcs)
        src_arr = src_arr[0].astype(np.float32)
        src_crs, nodata = srcs[0].crs, srcs[0].nodata
    if nodata is not None:
        src_arr[src_arr == nodata] = np.nan
    print(f"Kaynak DEM: {src_arr.shape}, CRS {src_crs}, nodata {nodata}")

    reproject(source=src_arr, destination=dst,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs="EPSG:4326",
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.average)
    for s in srcs:
        s.close()

    n_nan = int(np.isnan(dst).sum())
    dst[np.isnan(dst)] = args.sea_level_fill
    dst[dst < -500] = args.sea_level_fill
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, dst.astype(np.float32))
    print(f"Yazıldı: {args.out}  shape={dst.shape}  min={dst.min():.0f} max={dst.max():.0f} m  "
          f"(veri olmayan/deniz {n_nan} px -> {args.sea_level_fill})")
    # hızlı doğrulama: bilinen noktalar
    checks = {"Ankara Esenboğa (~950 m)": (40.128, 32.995), "Erciyes zirve (~3900 m)": (38.532, 35.447), "İzmir kıyı (~0-50 m)": (38.42, 27.14)}
    for name, (lat, lon) in checks.items():
        y, x = grid.latlon_to_index(lat, lon)
        if 0 <= y < grid.ny and 0 <= x < grid.nx:
            print(f"   {name}: {dst[y, x]:.0f} m")


if __name__ == "__main__":
    main()
