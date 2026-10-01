import os
import glob
# pyrefly: ignore [missing-import]
import numpy as np
from datetime import datetime, timedelta

from radar_loader import load_radar_metadata, get_radar_extent, load_radar_image, parse_radar_filename_time
from satellite_grid import get_satellite_grid
from resampler import resample_radar_to_satellite

RAW_MTG_DIR = "e:/belgeler/MTGCLM/data/raw_mtg"
RAW_RADAR_DIR = "e:/belgeler/MTGCLM/data/raw_radar"
RADAR_CSV = os.path.join(RAW_RADAR_DIR, "radar.csv")
OUTPUT_DIR = "e:/belgeler/MTGCLM/data/processed/radar_aligned"

def parse_mtg_time(filename: str) -> datetime:
    """
    Parses timestamp from MTG file.
    Example: turkiye_kesit_20260525000817_0001.nc -> 2026-05-25 00:08:17
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    time_str = parts[2] # 20260525000817
    return datetime.strptime(time_str, "%Y%m%d%H%M%S")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    print("Loading radar metadata...")
    radar_meta = load_radar_metadata(RADAR_CSV)
    
    # Get all satellite files
    mtg_files = glob.glob(os.path.join(RAW_MTG_DIR, "*.nc"))
    mtg_files.sort()
    
    # Pre-parse all radar files and their times to avoid doing it in the loop repeatedly
    print("Scanning radar files...")
    all_radar_pngs = glob.glob(os.path.join(RAW_RADAR_DIR, "*.png"))
    radar_info = []
    for png in all_radar_pngs:
        try:
            dt = parse_radar_filename_time(png)
            radar_info.append({"path": png, "time": dt})
        except Exception as e:
            print(f"Skipping {png}: {e}")
            
    # Time window: 5 minutes
    TIME_TOLERANCE = timedelta(minutes=5)
    
    # Load the target grid just once, assuming all MTG files have the same grid (they do)
    if not mtg_files:
        print("No MTG files found.")
        return
        
    print(f"Initializing target grid from {mtg_files[0]}...")
    target_def = get_satellite_grid(mtg_files[0])
    target_shape = target_def.shape # (800, 1000)
    
    # Process all MTG files
    for mtg_file in mtg_files:
        mtg_time = parse_mtg_time(mtg_file)
        basename = os.path.basename(mtg_file)
        out_name = basename.replace('.nc', '.npy')
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        print(f"\nProcessing {basename} (Time: {mtg_time})")
        
        # Find matching radars
        matching_radars = []
        for r in radar_info:
            if abs(r['time'] - mtg_time) <= TIME_TOLERANCE:
                matching_radars.append(r)
                
        print(f"Found {len(matching_radars)} radar images within +/- 5 mins.")
        
        # Initialize composite matrix
        composite_radar = np.zeros(target_shape, dtype=np.uint8)
        
        if not matching_radars:
            # Save zeros if no radars found
            np.save(out_path, composite_radar)
            continue
            
        for r in matching_radars:
            png_path = r['path']
            oa = os.path.basename(png_path).replace('.png', '')
            if oa not in radar_meta.index:
                print(f"  Warning: Metadata for {oa} not found. Skipping.")
                continue
                
            try:
                extent = get_radar_extent(radar_meta, oa)
                radar_array = load_radar_image(png_path)
                
                # Resample
                resampled = resample_radar_to_satellite(radar_array, extent, target_def)
                
                # Composite using Maximum Reflectivity
                composite_radar = np.maximum(composite_radar, resampled)
            except Exception as e:
                print(f"  Error processing {oa}: {e}")
                
        # Save output
        np.save(out_path, composite_radar)
        print(f"Saved {out_path}. Max value in composite: {composite_radar.max()}")

if __name__ == "__main__":
    main()
