import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
from tqdm import tqdm

# Adjust path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from awos_matcher import load_stations, match_stations_to_grid
from observation_loader import load_observations
from satellite_grid import get_satellite_grid

RAW_MTG_DIR = "F:/turkiye_kesitler"
RAW_OBS_DIR = "e:/belgeler/MTGCLM/data/raw_observation"
OUTPUT_DIR = "e:/belgeler/MTGCLM/data/processed/awos_aligned"

STATION_CSV = os.path.join(RAW_OBS_DIR, "station_list.csv")
SYNOP_CSV = os.path.join(RAW_OBS_DIR, "synop_all.csv")
METAR_CSV = os.path.join(RAW_OBS_DIR, "metar_all.csv")

def parse_mtg_time(filename: str) -> datetime:
    """Parses timestamp from MTG file."""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    time_str = parts[2] # YYYYMMDDHHMMSS
    return datetime.strptime(time_str, "%Y%m%d%H%M%S")

# Worker initialization
obs_shared = None

def init_worker(obs_df_pickled):
    global obs_shared
    obs_shared = obs_df_pickled

def process_single_file(task_args):
    mtg_file, out_path = task_args
    
    # Skip if already exists
    if os.path.exists(out_path):
        return True
        
    try:
        mtg_time = parse_mtg_time(mtg_file)
        
        # Filter observations within +/- 9 minutes
        # We access the shared dataframe directly
        time_diff = np.abs(obs_shared['validity'] - mtg_time)
        matched_obs = obs_shared[time_diff <= timedelta(minutes=9)].copy()
        
        # Rename y_idx and x_idx to y, x
        matched_obs = matched_obs.rename(columns={'y_idx': 'y', 'x_idx': 'x'})
        
        # Save to CSV
        matched_obs.to_csv(out_path, index=False)
        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(mtg_file)}: {e}")
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    mtg_files = sorted(glob.glob(os.path.join(RAW_MTG_DIR, "*.nc")))
    if not mtg_files:
        print("No MTG files found.")
        return
        
    print(f"Initializing target grid from {mtg_files[0]}...")
    target_def = get_satellite_grid(mtg_files[0])
    
    print("Loading and mapping stations to grid...")
    stations = load_stations(STATION_CSV)
    mapped_stations = match_stations_to_grid(stations, target_def)
    station_map = mapped_stations[['wmoid', 'y_idx', 'x_idx']].copy()
    
    print("Loading 6-month observations (SYNOP + METAR)...")
    obs_df = load_observations(SYNOP_CSV, METAR_CSV)
    
    # Merge observations with their mapped grid indices
    obs_mapped = pd.merge(obs_df, station_map, on='wmoid', how='inner')
    print(f"Loaded {len(obs_mapped)} observations matched with stations.")
    
    # Prepare task arguments
    tasks = []
    for mtg_file in mtg_files:
        basename = os.path.basename(mtg_file)
        out_name = basename.replace('.nc', '_awos.csv')
        out_path = os.path.join(OUTPUT_DIR, out_name)
        tasks.append((mtg_file, out_path))
        
    print(f"Total tasks: {len(tasks)}")
    
    # Use multiprocessing to speed up matching and I/O
    num_workers = min(multiprocessing.cpu_count(), 12)
    print(f"Using {num_workers} parallel workers...")
    
    # We pass obs_mapped to initializer to share it across worker processes
    pool = multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(obs_mapped,))
    
    # Run tasks with a progress bar
    results = []
    for res in tqdm(pool.imap_unordered(process_single_file, tasks, chunksize=50), total=len(tasks), desc="Aligning AWOS"):
        results.append(res)
        
    pool.close()
    pool.join()
    
    success_count = sum(1 for r in results if r)
    print(f"\nProcessing finished. Success: {success_count}/{len(tasks)}")

if __name__ == "__main__":
    main()
