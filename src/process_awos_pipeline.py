import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from awos_matcher import load_stations, match_stations_to_grid
from observation_loader import load_observations
from satellite_grid import get_satellite_grid
from process_radar_pipeline import parse_mtg_time

RAW_MTG_DIR = "e:/belgeler/MTGCLM/data/raw_mtg"
RAW_OBS_DIR = "e:/belgeler/MTGCLM/data/raw_observation"
OUTPUT_DIR = "e:/belgeler/MTGCLM/data/processed/awos_aligned"

STATION_CSV = os.path.join(RAW_OBS_DIR, "station_list.csv")
SYNOP_CSV = os.path.join(RAW_OBS_DIR, "synop.csv")
METAR_CSV = os.path.join(RAW_OBS_DIR, "metar.csv")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    mtg_files = glob.glob(os.path.join(RAW_MTG_DIR, "*.nc"))
    mtg_files.sort()
    
    if not mtg_files:
        print("No MTG files found.")
        return
        
    print(f"Initializing target grid from {mtg_files[0]}...")
    target_def = get_satellite_grid(mtg_files[0])
    
    print("Loading and mapping stations to grid...")
    stations = load_stations(STATION_CSV)
    mapped_stations = match_stations_to_grid(stations, target_def)
    
    # We only need wmoid, y_idx, x_idx
    station_map = mapped_stations[['wmoid', 'y_idx', 'x_idx']].copy()
    
    print("Loading observations...")
    obs_df = load_observations(SYNOP_CSV, METAR_CSV)
    
    # Merge observations with their mapped grid indices
    # This adds y_idx and x_idx to each observation based on wmoid
    obs_mapped = pd.merge(obs_df, station_map, on='wmoid', how='inner')
    print(f"Merged {len(obs_mapped)} observations with valid station coordinates.")
    
    # User specified 9 minutes time tolerance
    TIME_TOLERANCE = timedelta(minutes=9)
    
    # Process all files
    for mtg_file in mtg_files:
        mtg_time = parse_mtg_time(mtg_file)
        basename = os.path.basename(mtg_file)
        out_name = basename.replace('.nc', '_awos.csv')
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        print(f"\nProcessing {basename} (Time: {mtg_time})")
        
        # Filter observations within +/- 9 minutes
        time_diff = np.abs(obs_mapped['validity'] - mtg_time)
        matched_obs = obs_mapped[time_diff <= TIME_TOLERANCE].copy()
        
        print(f"Found {len(matched_obs)} observations within +/- 9 mins.")
        
        # We can drop the validity column for the output or keep it. Let's keep it to show when it was observed.
        # But we must rename y_idx and x_idx to y, x as requested
        matched_obs = matched_obs.rename(columns={'y_idx': 'y', 'x_idx': 'x'})
        
        # Save to CSV
        matched_obs.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
