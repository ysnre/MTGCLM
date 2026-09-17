import os
import sys
import glob
import time
import bisect
import numpy as np
import multiprocessing
from datetime import datetime, timedelta
from pymongo import MongoClient
import xarray as xr
from pyresample.geometry import SwathDefinition
from tqdm import tqdm

# Adjust path to import custom modules if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from resampler import resample_radar_to_satellite, resample_radar_to_satellite_aeqd
from radar_loader import load_radar_image

# Directories defined by user
RAW_MTG_DIR = "F:/turkiye_kesitler"
RAW_RADAR_DIR = "F:/radar/ppi"
OUTPUT_DIR = "F:/radar/radar_aligned"

# Worker global variables
target_def = None
mongo_col = None

def get_satellite_grid_local(nc_file_path: str) -> SwathDefinition:
    """Loads satellite grid coordinate mesh grid."""
    ds = xr.open_dataset(nc_file_path)
    lons_1d = ds['x'].values
    lats_1d = ds['y'].values
    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d, indexing='xy')
    return SwathDefinition(lons=lons_2d, lats=lats_2d)

def init_worker(sample_nc_path):
    """Initializes target grid and MongoDB connection once per worker process."""
    global target_def, mongo_col
    target_def = get_satellite_grid_local(sample_nc_path)
    client = MongoClient("mongodb://radar_user:radar@192.168.10.196:27017/radar?authSource=radar")
    mongo_col = client.radar.radar

def parse_mtg_time(filename: str) -> datetime:
    """Parses timestamp from MTG file."""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    time_str = parts[2] # YYYYMMDDHHMMSS
    return datetime.strptime(time_str, "%Y%m%d%H%M%S")

def process_single_satellite(task_args):
    """Processes a single satellite file by resampling and compositing its matching radars."""
    mtg_file_path, matching_radar_paths = task_args
    basename = os.path.basename(mtg_file_path)
    out_name = basename.replace('.nc', '.npy')
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    # Skip if already exists
    if os.path.exists(out_path):
        return basename, "Skipped (Exists)", len(matching_radar_paths), 0.0
        
    try:
        t_start = time.time()
        target_shape = target_def.shape # (800, 1000)
        composite_radar = np.zeros(target_shape, dtype=np.uint8)
        
        for png_path in matching_radar_paths:
            oa = os.path.basename(png_path).replace('.png', '')
            
            # Retrieve center and radius from MongoDB
            doc = mongo_col.find_one({"oa": oa})
            if doc is None:
                continue
                
            lt = doc.get('lt')
            ln = doc.get('ln')
            r = doc.get('r')
            if lt is None or ln is None or r is None:
                continue # Skip if center or radius is missing
                
            # Load and resample using Azimuthal Equidistant projection
            radar_array = load_radar_image(png_path)
            resampled = resample_radar_to_satellite_aeqd(
                radar_array, float(lt), float(ln), float(r), target_def
            )
            
            # Composite
            composite_radar = np.maximum(composite_radar, resampled)
            
        # Save output
        np.save(out_path, composite_radar)
        elapsed = time.time() - t_start
        return basename, "Success", len(matching_radar_paths), elapsed
    except Exception as e:
        return basename, f"Error: {str(e)}", len(matching_radar_paths), 0.0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode on first 5 files.")
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    print("Scanning satellite files...")
    mtg_files = glob.glob(os.path.join(RAW_MTG_DIR, "*.nc"))
    mtg_files.sort()
    num_mtg = len(mtg_files)
    print(f"Found {num_mtg} MTG files in {RAW_MTG_DIR}")
    
    if num_mtg == 0:
        print("No MTG files found. Exiting.")
        return
        
    print("Scanning radar files...")
    t0 = time.time()
    all_radar_pngs = glob.glob(os.path.join(RAW_RADAR_DIR, "*.png"))
    
    # Parse and index radar files
    radar_list = []
    for p in all_radar_pngs:
        b = os.path.basename(p)
        try:
            # YYMMDDHHMMSS format
            dt = datetime(2000 + int(b[3:5]), int(b[5:7]), int(b[7:9]), int(b[9:11]), int(b[11:13]), int(b[13:15]))
            radar_list.append((dt.timestamp(), p))
        except Exception:
            pass
            
    radar_list.sort(key=lambda x: x[0])
    radar_timestamps = [r[0] for r in radar_list]
    print(f"Loaded and indexed {len(radar_list)} radar files in {time.time() - t0:.2f} seconds.")
    
    # Prepare task arguments
    print("Preparing task mappings using binary search...")
    tasks = []
    skipped_existing = 0
    
    for mtg_file in mtg_files:
        basename = os.path.basename(mtg_file)
        out_name = basename.replace('.nc', '.npy')
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # Check if already processed
        if os.path.exists(out_path):
            skipped_existing += 1
            continue
            
        try:
            mtg_time = parse_mtg_time(mtg_file)
            ts_sat = mtg_time.timestamp()
            
            # Find matching radars within +/- 5 minutes (300 seconds)
            left_idx = bisect.bisect_left(radar_timestamps, ts_sat - 300)
            right_idx = bisect.bisect_right(radar_timestamps, ts_sat + 300)
            matching_paths = [radar_list[i][1] for i in range(left_idx, right_idx)]
            
            tasks.append((mtg_file, matching_paths))
        except Exception as e:
            print(f"Error preparing task for {basename}: {e}")
            
    print(f"Total MTG files: {num_mtg}")
    print(f"Already processed: {skipped_existing}")
    print(f"Tasks to run: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All files are already processed. Nothing to do!")
        return

    if args.test:
        print("Running in TEST mode (limiting to first 5 tasks)...")
        tasks = tasks[:5]
        
    # Set number of workers (leave some cores for the system)
    num_cpus = multiprocessing.cpu_count()
    num_workers = max(1, min(num_cpus - 2, 12)) # Let's use 12 workers to be safe and efficient
    print(f"Using {num_workers} parallel workers.")
    
    # Start the multiprocessing pool
    t_start_pool = time.time()
    errors = []
    
    # We recycle workers after 500 tasks to prevent potential memory leaks
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(mtg_files[0],), maxtasksperchild=500) as pool:
        results = pool.imap_unordered(process_single_satellite, tasks)
        
        # Display progress using tqdm
        for basename, status, num_radars, elapsed in tqdm(results, total=len(tasks), desc="Aligning Radar"):
            if "Error" in status:
                errors.append((basename, status))
                
    total_elapsed = time.time() - t_start_pool
    print(f"\nProcessing finished in {total_elapsed / 60:.2f} minutes.")
    print(f"Processed: {len(tasks) - len(errors)}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nFirst 10 errors:")
        for b, err in errors[:10]:
            print(f"  {b}: {err}")

if __name__ == "__main__":
    # Windows needs freeze_support for multiprocessing
    multiprocessing.freeze_support()
    main()
