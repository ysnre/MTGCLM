import os
import h5py
import numpy as np
import datetime
from awos_scalar_loader import build_awos_lookup, round_to_nearest_10min

def update_h5_with_scalars(h5_path: str, awos_dir: str):
    """
    Reads all timestamps and wmoids from the HDF5 file, looks up the corresponding
    AWOS observations, and adds them directly as a new 'scalars' dataset.
    This enables highly optimized data loading without needing CSVs during training.
    """
    print(f"Opening {h5_path} to add AWOS scalars...")
    
    with h5py.File(h5_path, 'a') as f:
        if 'scalars' in f:
            print("The 'scalars' dataset already exists in the HDF5 file. Deleting old dataset...")
            del f['scalars']
            
        timestamps = f['timestamps'][:]
        wmoids = f['wmoids'][:]
        num_samples = len(timestamps)
        
        required_keys = set()
        keys_list = []
        
        print("Extracting and rounding timestamps...")
        for i in range(num_samples):
            wmoid = wmoids[i]
            ts_str = timestamps[i].decode('utf-8')
            
            try:
                # Format: YYYYMMDDHHMMSS
                dt = datetime.datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                rounded_dt = round_to_nearest_10min(dt)
                
                key = (wmoid, rounded_dt.year, rounded_dt.month, rounded_dt.day, rounded_dt.hour, rounded_dt.minute)
                required_keys.add(key)
                keys_list.append(key)
            except Exception as e:
                print(f"Error parsing timestamp {ts_str}: {e}")
                keys_list.append(None)
                
        print(f"Found {len(required_keys)} unique (wmoid, time) combinations needed.")
        
        # Build lookup table by scanning the large CSV once
        lookup = build_awos_lookup(required_keys, awos_dir)
        
        print("Building scalar arrays...")
        scalars = np.zeros((num_samples, 3), dtype=np.float32)
        
        missing_count = 0
        for i in range(num_samples):
            key = keys_list[i]
            if key and key in lookup:
                scalars[i] = lookup[key]
            else:
                missing_count += 1
                # The build_awos_lookup function already guarantees all required_keys are in lookup 
                # (using median imputation for missing). This else block is just a safeguard.
                scalars[i] = [0.0, 0.0, 0.0] # Fallback to normalized mean (0)
                
        print(f"Scalar arrays built. Safeguard missing count: {missing_count}")
        
        # Save to HDF5
        print("Saving 'scalars' dataset to HDF5...")
        f.create_dataset(
            'scalars', 
            data=scalars, 
            compression="gzip", 
            compression_opts=4,
            chunks=(1024, 3)
        )
        print("HDF5 file updated successfully with AWOS scalars!")

if __name__ == "__main__":
    from config import IS_COLAB, BASE_DIR
    
    # Path configuration
    awos_dir = "e:/belgeler/MTGCLM/data/raw_awos_observation"
    h5_path = "F:/radar/mtg_patch_dataset.h5"
    
    if IS_COLAB:
        print("This script is meant to be run locally before uploading to Colab.")
    elif os.path.exists(h5_path):
        update_h5_with_scalars(h5_path, awos_dir)
    else:
        print(f"HDF5 file not found at {h5_path}. Please check the path.")
