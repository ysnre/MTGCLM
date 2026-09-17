import os
import glob
import hdf5plugin
from satpy import Scene
import xarray as xr
import numpy as np
from collections import defaultdict
from pyresample import create_area_def
import netCDF4

# Giriş ve çıkış klasörleri
input_dir = "F:\\mtg"
output_dir = "./turkiye_kesitler"
os.makedirs(output_dir, exist_ok=True)

# Dosyaları al
mtg_files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))

# Kullanılacak kanallar
channels = [
    "vis_04","vis_05","vis_06", "vis_08", 
    "vis_09", "nir_13", "nir_16", "nir_22", 
    "ir_38", "ir_87", "ir_97", "ir_105", 
    "ir_123", "ir_133", "wv_63", "wv_73"
]

# 1️⃣ Her gün ve tarama için dosyaları grupla
grouped_by_day_and_scan = defaultdict(lambda: defaultdict(list))
for file in mtg_files:
    basename = os.path.basename(file)
    date_str = basename.split("_")[4]       # yıl ay gün saat dakika saniye gibi (örneğin: 20231001120000)
    scan_id = basename.split("_")[-2]       # 0045 gibi
    grouped_by_day_and_scan[date_str][scan_id].append(file)

# 2️⃣ Her günün her taraması için işleme başla
for day, scans in grouped_by_day_and_scan.items():
    for scan_id, files in scans.items():
        try:
            print(f"İşleniyor: {day} | Tarama: {scan_id} ({len(files)} swath)")

            scn = Scene(reader="fci_l1c_nc", filenames=files)

            available = set(scn.available_dataset_names())
            loadable = list(available.intersection(channels))
            if not loadable:
                print(f" - Yüklenebilir kanal yok.")
                continue

            scn.load(loadable)

            area_def = create_area_def(
                "turkey",
                {'proj': 'latlong'},
                width=1000,
                height=800,
                area_extent=(25.0, 35.0, 45.0, 43.0)
            )

            resampled_scn = scn.resample(area_def)
            xr_ds = resampled_scn.to_xarray_dataset()

            if "crs" in xr_ds.coords:
                xr_ds = xr_ds.drop_vars("crs")
            for var in xr_ds.data_vars:
                xr_ds[var].attrs.clear()
            xr_ds.attrs.clear()

            out_path = os.path.join(output_dir, f"turkiye_kesit_{day}_{scan_id}.nc")
            xr_ds.to_netcdf(out_path)
            print(f" - Kaydedildi: {out_path}")

        except Exception as e:
            print(f"HATA ({day} - {scan_id}):", e)
