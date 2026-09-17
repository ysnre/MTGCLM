import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

class MTGCloudDataset(Dataset):
    def __init__(self, 
                 mtg_dir: str, 
                 radar_dir: str, 
                 awos_dir: str, 
                 mode: str = "patch", 
                 patch_size: int = 33, 
                 stats: dict = None):
        """
        PyTorch Dataset for MTG Cloud Masking.
        
        Args:
            mtg_dir (str): Directory containing raw satellite NetCDF files (*.nc).
            radar_dir (str): Directory containing aligned radar Numpy files (*.npy).
            awos_dir (str): Directory containing aligned AWOS CSV files (*_awos.csv).
            mode (str): "patch" for CNN localized training, or "full" for U-Net segmentation.
            patch_size (int): Size of the patch to cut in "patch" mode (must be odd).
            stats (dict): Pre-calculated normalization stats {"means": [...], "stds": [...]}.
                          If None, stats will be computed from a sample of the files.
        """
        self.mtg_dir = mtg_dir
        self.radar_dir = radar_dir
        self.awos_dir = awos_dir
        self.mode = mode.lower()
        self.patch_size = patch_size
        
        if self.mode == "patch" and patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer.")
            
        # In-memory cache to prevent redundant disk reads during batch generation
        self.feature_cache = {}
        self.cache_limit = 80
            
        # Match files by timestamp
        self.nc_files = sorted(glob.glob(os.path.join(mtg_dir, "*.nc")))
        self.samples = []
        
        # Build file index mapping
        for nc_path in self.nc_files:
            basename = os.path.basename(nc_path)
            # Find radar file
            radar_name = basename.replace('.nc', '.npy')
            radar_path = os.path.join(radar_dir, radar_name)
            # Find awos file
            awos_name = basename.replace('.nc', '_awos.csv')
            awos_path = os.path.join(awos_dir, awos_name)
            
            if os.path.exists(radar_path) and os.path.exists(awos_path):
                self.samples.append({
                    "nc_path": nc_path,
                    "radar_path": radar_path,
                    "awos_path": awos_path,
                    "timestamp": basename.split('_')[2]
                })
                
        print(f"Dataset initialized in '{self.mode}' mode with {len(self.samples)} timestamps.")
        
        # Calculate or assign normalization stats
        if stats:
            self.means = stats['means']
            self.stds = stats['stds']
        else:
            self._compute_normalization_stats()
            
        # If patch mode, index individual station observations
        if self.mode == "patch":
            self.patch_samples = []
            for ts_idx, s in enumerate(self.samples):
                df = pd.read_csv(s['awos_path'])
                for _, row in df.iterrows():
                    wmoid = int(row['wmoid'])
                    y = int(row['y'])
                    x = int(row['x'])
                    cloud_coverage = int(row['cloud_coverage'])
                    
                    self.patch_samples.append({
                        "ts_idx": ts_idx,
                        "wmoid": wmoid,
                        "y": y,
                        "x": x,
                        "cloud_coverage": cloud_coverage
                    })
            print(f"Total patch samples (observations): {len(self.patch_samples)}")
            
    def _compute_normalization_stats(self):
        """
        Computes mean and standard deviation for the 16 satellite channels and 1 radar channel
        using a subset of the dataset (up to 5 timestamps).
        """
        print("Computing normalization stats from sample files...")
        sample_size = min(5, len(self.samples))
        if sample_size == 0:
            # Fallback to defaults if no files found
            self.means = np.zeros(17)
            self.stds = np.ones(17)
            return
            
        sat_data_list = []
        radar_data_list = []
        
        for i in range(sample_size):
            s = self.samples[i]
            # Load sat
            ds = xr.open_dataset(s['nc_path'])
            # 16 channels
            sat_channels = [ds[var].values for var in ds.data_vars]
            sat_arr = np.stack(sat_channels, axis=0) # (16, 800, 1000)
            sat_data_list.append(sat_arr)
            
            # Load radar
            radar_arr = np.load(s['radar_path']) # (800, 1000)
            radar_data_list.append(radar_arr)
            
        # Stack to compute
        all_sat = np.concatenate(sat_data_list, axis=1) # (16, sample_size*800, 1000)
        all_radar = np.concatenate(radar_data_list, axis=0) # (sample_size*800, 1000)
        
        self.means = np.zeros(17)
        self.stds = np.ones(17)
        
        for c in range(16):
            self.means[c] = np.nanmean(all_sat[c])
            self.stds[c] = np.nanstd(all_sat[c]) + 1e-6
            
        self.means[16] = np.nanmean(all_radar)
        self.stds[16] = np.nanstd(all_radar) + 1e-6
        
        print("Stats computed successfully.")
        
    def _load_features(self, nc_path: str, radar_path: str) -> np.ndarray:
        """
        Loads and normalizes satellite and radar features.
        Returns a normalized (17, 800, 1000) float32 numpy array.
        Uses in-memory cache to prevent redundant NetCDF/Numpy reads.
        """
        if nc_path in self.feature_cache:
            return self.feature_cache[nc_path]
            
        # Load satellite (16 channels)
        ds = xr.open_dataset(nc_path)
        sat_channels = [ds[var].values for var in ds.data_vars]
        sat_arr = np.stack(sat_channels, axis=0) # (16, 800, 1000)
        
        # Load radar (1 channel)
        radar_arr = np.load(radar_path) # (800, 1000)
        radar_arr = np.expand_dims(radar_arr, axis=0) # (1, 800, 1000)
        
        # Combine (17, 800, 1000)
        features = np.concatenate([sat_arr, radar_arr], axis=0).astype(np.float32)
        
        # Apply Z-score normalization channel-wise
        for c in range(17):
            features[c] = (features[c] - self.means[c]) / self.stds[c]
            
        # Replace any NaNs or Infinities with 0.0 (the mean value)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Handle cache size
        if len(self.feature_cache) >= self.cache_limit:
            # Drop oldest key (FIFO)
            oldest_key = next(iter(self.feature_cache))
            self.feature_cache.pop(oldest_key)
            
        self.feature_cache[nc_path] = features
        return features

    def _get_consolidated_label(self, cloud_coverage: int, radar_val: float) -> int:
        """
        Applies the Consolidated Ground Truth logic to determine clear/cloudy label.
        - 1 (Cloudy): AWOS is between 1 and 8.
        - 0 (Clear): AWOS is 0.
        - -1 (Ignore): Invalid or missing.
        """
        # Radar is excluded from label definition to prevent target leakage
        # since it is used as an input feature in Scenario B.
        if 1 <= cloud_coverage <= 8:
            return 1
        elif cloud_coverage == 0:
            return 0
        else:
            return -1

    def __len__(self) -> int:
        if self.mode == "patch":
            return len(self.patch_samples)
        return len(self.samples)
        
    def __getitem__(self, idx: int):
        if self.mode == "patch":
            sample_info = self.patch_samples[idx]
            ts_idx = sample_info['ts_idx']
            s = self.samples[ts_idx]
            
            # Load full features
            features = self._load_features(s['nc_path'], s['radar_path'])
            
            # We need the raw radar value at the station to compute the label
            # Reconstruct raw radar from normalized features:
            # normalized_val = (raw_val - mean) / std -> raw_val = normalized_val * std + mean
            norm_radar_val = features[16, sample_info['y'], sample_info['x']]
            raw_radar_val = norm_radar_val * self.stds[16] + self.means[16]
            
            # Label
            label = self._get_consolidated_label(sample_info['cloud_coverage'], raw_radar_val)
            
            # Cut patch
            pad = self.patch_size // 2
            # Pad features (17, 800, 1000) -> (17, 800 + 2*pad, 1000 + 2*pad)
            padded_features = np.pad(features, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
            
            y_padded = sample_info['y'] + pad
            x_padded = sample_info['x'] + pad
            
            patch = padded_features[:, y_padded - pad : y_padded + pad + 1, x_padded - pad : x_padded + pad + 1]
            
            return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            
        else: # "full" mode
            s = self.samples[idx]
            features = self._load_features(s['nc_path'], s['radar_path'])
            
            # Build full-image label mask of shape (800, 1000)
            # Initialize with -1 (ignore)
            label_mask = np.full((800, 1000), -1, dtype=np.int64)
            
            # Load observations for this timestamp
            df = pd.read_csv(s['awos_path'])
            
            # Get raw radar channel to apply rules
            # Reconstruct raw radar array
            raw_radar = features[16] * self.stds[16] + self.means[16]
            
            for _, row in df.iterrows():
                y = int(row['y'])
                x = int(row['x'])
                cc = int(row['cloud_coverage'])
                r_val = raw_radar[y, x]
                
                label_mask[y, x] = self._get_consolidated_label(cc, r_val)
                
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label_mask, dtype=torch.long)
