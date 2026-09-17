import os
import glob
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

def main():
    h5_path = "/content/mtg_patch_dataset.h5"
    if not os.path.exists(h5_path):
        # Fallback to Windows local path if running locally
        h5_path = "F:/radar/mtg_patch_dataset.h5"
        
    if not os.path.exists(h5_path):
        print(f"Error: H5 dataset not found at {h5_path}")
        return

    # Find the AWOS directory
    awos_dirs = [
        "/content/awos_aligned",
        "/content/drive/MyDrive/MTGCLM/data/processed/awos_aligned",
        "/content/drive/MyDrive/MTGCLM/awos_aligned",
        "e:/belgeler/MTGCLM/data/processed/awos_aligned"
    ]
    
    awos_dir = None
    for d in awos_dirs:
        if os.path.exists(d) and len(glob.glob(os.path.join(d, "*.csv"))) > 0:
            awos_dir = d
            break
            
    if awos_dir is None:
        print("Error: Could not find AWOS directory with CSV files.")
        return
        
    print(f"Using AWOS CSV directory: {awos_dir}")
    print(f"Reading H5 file from: {h5_path}")
    
    # 1. Load all CSV files and build (timestamp, wmoid) -> cloud_coverage mapping
    awos_dict = {}
    csv_files = glob.glob(os.path.join(awos_dir, "*.csv"))
    print(f"Loading {len(csv_files)} AWOS CSV files...")
    
    for csv_path in tqdm(csv_files, desc="Parsing CSVs"):
        basename = os.path.basename(csv_path)
        # Extract timestamp: e.g. MSG_FCI_20260216T120000Z_..._awos.csv -> 20260216T120000Z
        parts = basename.split('_')
        if len(parts) >= 3:
            timestamp = parts[2]
        else:
            continue
            
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                wmoid = int(row['wmoid'])
                cc = int(row['cloud_coverage'])
                awos_dict[(timestamp, wmoid)] = cc
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            
    print(f"Mapped {len(awos_dict)} (timestamp, wmoid) observations.")

    # 2. Open H5 file and load metadata
    print("Opening H5 file...")
    with h5py.File(h5_path, 'r+') as f:
        # Load timestamps and wmoids
        timestamps_ds = f['timestamps']
        wmoids_ds = f['wmoids']
        
        num_samples = len(timestamps_ds)
        print(f"Found {num_samples} samples in H5 file.")
        
        # Read arrays
        print("Reading timestamps and wmoids into memory...")
        timestamps = timestamps_ds[:]
        wmoids = wmoids_ds[:]
        
        # 3. Compute new labels
        print("Computing new labels...")
        new_labels = np.empty(num_samples, dtype=np.int64)
        
        for i in tqdm(range(num_samples), desc="Updating labels"):
            ts = timestamps[i]
            ts_str = ts.decode('utf-8') if isinstance(ts, bytes) else ts
            wmoid = wmoids[i]
            
            cc = awos_dict.get((ts_str, wmoid), -1)
            
            # Label logic based ONLY on AWOS cloud_coverage (0-8)
            # 1 (Cloudy) if 1 <= cc <= 8
            # 0 (Clear) if cc == 0
            # -1 (Ignore) otherwise
            if 1 <= cc <= 8:
                new_labels[i] = 1
            elif cc == 0:
                new_labels[i] = 0
            else:
                new_labels[i] = -1
                
        # 4. Overwrite labels dataset
        print("Writing new labels to H5 file...")
        f['labels'][:] = new_labels
        
    print("Successfully updated H5 dataset labels in-place!")

if __name__ == "__main__":
    main()
