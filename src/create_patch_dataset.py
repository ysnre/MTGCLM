import os
import sys
import h5py
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from datetime import datetime

# Adjust path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local paths
RAW_MTG_DIR = "F:/turkiye_kesitler"
PROCESSED_RADAR_DIR = "F:/radar/radar_aligned"
PROCESSED_AWOS_DIR = "e:/belgeler/MTGCLM/data/processed/awos_aligned"
OUTPUT_H5_PATH = "F:/radar/mtg_patch_dataset.h5"

def parse_mtg_time(filename: str) -> str:
    """Parses timestamp from MTG file name."""
    basename = os.path.basename(filename)
    return basename.split('_')[2]

def get_channel_names(nc_path: str) -> list:
    """Gets list of channel variable names from NetCDF using h5py."""
    with h5py.File(nc_path, 'r') as f:
        # Exclude coordinate and reference variables
        vars_list = [v for v in f.keys() if v not in ['x', 'y', 'time', 'spatial_ref', 'crs']]
    return vars_list

def load_features_fast(nc_path: str, radar_path: str, channel_names: list, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Fast load and normalization of features using h5py instead of xarray."""
    # Load satellite (16 channels)
    with h5py.File(nc_path, 'r') as f:
        sat_channels = [f[var][:] for var in channel_names]
    sat_arr = np.stack(sat_channels, axis=0) # (16, 800, 1000)
    
    # Load radar (1 channel)
    radar_arr = np.load(radar_path) # (800, 1000)
    radar_arr = np.expand_dims(radar_arr, axis=0) # (1, 800, 1000)
    
    # Combine (17, 800, 1000)
    features = np.concatenate([sat_arr, radar_arr], axis=0).astype(np.float32)
    
    # Normalize
    for c in range(17):
        features[c] = (features[c] - means[c]) / stds[c]
        
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

def get_consolidated_label(cloud_coverage: int, radar_val: float) -> int:
    """Consolidated Ground Truth logic (excluding radar to prevent target leakage)."""
    if 1 <= cloud_coverage <= 8:
        return 1
    elif cloud_coverage == 0:
        return 0
    else:
        return -1

def compute_normalization_stats_fast(samples: list, channel_names: list) -> tuple:
    """Computes means and stds for normalization from first 5 sample files."""
    print("Computing normalization stats from sample files...")
    sample_size = min(5, len(samples))
    
    sat_data_list = []
    radar_data_list = []
    
    for i in range(sample_size):
        s = samples[i]
        with h5py.File(s['nc_path'], 'r') as f:
            sat_channels = [f[var][:] for var in channel_names]
        sat_arr = np.stack(sat_channels, axis=0)
        sat_data_list.append(sat_arr)
        
        radar_arr = np.load(s['radar_path'])
        radar_data_list.append(radar_arr)
        
    all_sat = np.concatenate(sat_data_list, axis=1) # (16, sample_size*800, 1000)
    all_radar = np.concatenate(radar_data_list, axis=0) # (sample_size*800, 1000)
    
    means = np.zeros(17)
    stds = np.ones(17)
    
    for c in range(16):
        means[c] = np.nanmean(all_sat[c])
        stds[c] = np.nanstd(all_sat[c]) + 1e-6
        
    means[16] = np.nanmean(all_radar)
    stds[16] = np.nanstd(all_radar) + 1e-6
    
    print("Stats computed successfully.")
    return means, stds

def main():
    print("Scanning directory for files...")
    nc_files = sorted(glob.glob(os.path.join(RAW_MTG_DIR, "*.nc")))
    
    samples = []
    for nc_file in nc_files:
        basename = os.path.basename(nc_file)
        timestamp = parse_mtg_time(nc_file)
        radar_file = basename.replace('.nc', '.npy')
        awos_file = basename.replace('.nc', '_awos.csv')
        
        radar_path = os.path.join(PROCESSED_RADAR_DIR, radar_file)
        awos_path = os.path.join(PROCESSED_AWOS_DIR, awos_file)
        
        if os.path.exists(radar_path) and os.path.exists(awos_path):
            samples.append({
                "nc_path": nc_file,
                "radar_path": radar_path,
                "awos_path": awos_path,
                "timestamp": timestamp
            })
            
    print(f"Found {len(samples)} matched timestamps out of {len(nc_files)} NetCDF files.")
    if not samples:
        print("No matched samples found.")
        return
        
    # Get channel names from first sample
    channel_names = get_channel_names(samples[0]['nc_path'])
    print(f"Detected {len(channel_names)} satellite channels: {channel_names}")
    
    # Compute normalization stats
    means, stds = compute_normalization_stats_fast(samples, channel_names)
    
    # Load AWOS observations into a dictionary mapped by sample index
    print("Loading AWOS ground observations...")
    patch_samples_by_ts = {}
    total_raw_observations = 0
    for idx, s in enumerate(tqdm(samples, desc="Loading AWOS")):
        df = pd.read_csv(s['awos_path'])
        obs_list = []
        for _, row in df.iterrows():
            obs_list.append({
                "wmoid": int(row['wmoid']),
                "y": int(row['y']),
                "x": int(row['x']),
                "cloud_coverage": int(row['cloud_coverage'])
            })
        patch_samples_by_ts[idx] = obs_list
        total_raw_observations += len(obs_list)
        
    print(f"Total raw observations loaded: {total_raw_observations}")
    
    # Remove existing partial H5 file if any
    if os.path.exists(OUTPUT_H5_PATH):
        try:
            os.remove(OUTPUT_H5_PATH)
        except Exception as e:
            print(f"Warning: could not remove existing file: {e}")
            
    # Create HDF5 File
    h5_file = h5py.File(OUTPUT_H5_PATH, 'w')
    
    # We remove GZIP compression completely for maximum extraction speed.
    # The user can compress it with 7-Zip afterwards in minutes.
    patches_ds = h5_file.create_dataset(
        'patches', 
        shape=(0, 17, 33, 33), 
        maxshape=(None, 17, 33, 33), 
        dtype=np.float32,
        chunks=(2048, 17, 33, 33)
    )
    
    labels_ds = h5_file.create_dataset(
        'labels', 
        shape=(0,), 
        maxshape=(None,), 
        dtype=np.int64,
        chunks=(2048,)
    )
    
    timestamps_ds = h5_file.create_dataset(
        'timestamps', 
        shape=(0,), 
        maxshape=(None,), 
        dtype=h5py.special_dtype(vlen=str),
        chunks=(2048,)
    )
    
    wmoids_ds = h5_file.create_dataset(
        'wmoids', 
        shape=(0,), 
        maxshape=(None,), 
        dtype=np.int64,
        chunks=(2048,)
    )
    
    # Save normalization stats as metadata attributes
    h5_file.attrs['means'] = means
    h5_file.attrs['stds'] = stds
    
    # Buffer to hold items before writing to disk
    buffer_patches = []
    buffer_labels = []
    buffer_timestamps = []
    buffer_wmoids = []
    
    write_idx = 0
    patch_size = 33
    pad = patch_size // 2
    
    print("\nExtracting patches and writing to HDF5...")
    for idx, s in enumerate(tqdm(samples, desc="Processing files")):
        obs_list = patch_samples_by_ts[idx]
        if not obs_list:
            continue
            
        timestamp = s['timestamp']
        
        # Load and normalize features (fast)
        try:
            features = load_features_fast(s['nc_path'], s['radar_path'], channel_names, means, stds)
        except Exception as e:
            print(f"\nError loading {timestamp}: {e}. Skipping.")
            continue
            
        padded_features = np.pad(features, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
        
        # Reconstruct raw radar
        raw_radar = features[16] * stds[16] + means[16]
        
        for obs in obs_list:
            y = obs['y']
            x = obs['x']
            cc = obs['cloud_coverage']
            raw_radar_val = raw_radar[y, x]
            
            # Label
            label = get_consolidated_label(cc, raw_radar_val)
            if label == -1:
                continue
                
            # Cut patch
            y_padded = y + pad
            x_padded = x + pad
            patch = padded_features[:, y_padded - pad : y_padded + pad + 1, x_padded - pad : x_padded + pad + 1]
            
            buffer_patches.append(patch)
            buffer_labels.append(label)
            buffer_timestamps.append(timestamp)
            buffer_wmoids.append(obs['wmoid'])
            
            # Write to H5 in blocks of 2048 to maximize throughput
            if len(buffer_patches) >= 2048:
                curr_size = write_idx + len(buffer_patches)
                
                # Resize
                patches_ds.resize(curr_size, axis=0)
                labels_ds.resize(curr_size, axis=0)
                timestamps_ds.resize(curr_size, axis=0)
                wmoids_ds.resize(curr_size, axis=0)
                
                # Write
                patches_ds[write_idx:curr_size] = np.stack(buffer_patches, axis=0)
                labels_ds[write_idx:curr_size] = np.array(buffer_labels, dtype=np.int64)
                timestamps_ds[write_idx:curr_size] = np.array(buffer_timestamps, dtype=object)
                wmoids_ds[write_idx:curr_size] = np.array(buffer_wmoids, dtype=np.int64)
                
                write_idx = curr_size
                buffer_patches.clear()
                buffer_labels.clear()
                buffer_timestamps.clear()
                buffer_wmoids.clear()
                
    # Flush remaining buffer
    if buffer_patches:
        curr_size = write_idx + len(buffer_patches)
        patches_ds.resize(curr_size, axis=0)
        labels_ds.resize(curr_size, axis=0)
        timestamps_ds.resize(curr_size, axis=0)
        wmoids_ds.resize(curr_size, axis=0)
        
        patches_ds[write_idx:curr_size] = np.stack(buffer_patches, axis=0)
        labels_ds[write_idx:curr_size] = np.array(buffer_labels, dtype=np.int64)
        timestamps_ds[write_idx:curr_size] = np.array(buffer_timestamps, dtype=object)
        wmoids_ds[write_idx:curr_size] = np.array(buffer_wmoids, dtype=np.int64)
        
        write_idx = curr_size
        
    h5_file.close()
    
    print(f"\nSuccessfully finished! Created dataset at {OUTPUT_H5_PATH}")
    print(f"Total raw observations: {total_raw_observations} -> Kept valid observations: {write_idx} ({write_idx/total_raw_observations*100.0:.2f}%)")

if __name__ == "__main__":
    main()
