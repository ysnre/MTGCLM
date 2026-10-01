import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class MTGH5Dataset(Dataset):
    """
    High-performance PyTorch Dataset that reads pre-extracted patches from a compressed HDF5 file.
    Opens the file lazily in __getitem__ to be compatible with PyTorch DataLoader multiprocessing (num_workers > 0).
    """
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        
        # Open file once in main thread to load metadata, length, and stats
        # Set chunk cache size to 512 MB to accommodate large 144.6 MB chunks
        with h5py.File(self.h5_path, 'r', rdcc_nbytes=512*1024*1024) as f:
            self.length = len(f['labels'])
            self.means = f.attrs['means'][:]
            self.stds = f.attrs['stds'][:]
            # Preload small metadata arrays into memory for fast splitting/indexing and class weights calculation
            self.timestamps = [ts.decode('utf-8') for ts in f['timestamps'][:]]
            self.wmoids = f['wmoids'][:]
            self.labels = f['labels'][:]
            
        self.h5_file = None  # Opened lazily per worker process
        self.patches_ds = None
        self.labels_ds = None
        self.scalars_ds = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self.h5_file is None:
            # Open the file in read-only mode with 512 MB chunk cache
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True, rdcc_nbytes=512*1024*1024)
            # Store dataset references to prevent open/close overhead
            self.patches_ds = self.h5_file['patches']
            self.labels_ds = self.h5_file['labels']
            if 'scalars' in self.h5_file:
                self.scalars_ds = self.h5_file['scalars']
            
        # Load patch and label directly from cached datasets
        patch = self.patches_ds[idx]
        label = self.labels_ds[idx]
        
        if self.scalars_ds is not None:
            scalar = self.scalars_ds[idx]
        else:
            scalar = np.zeros(3, dtype=np.float32) # Fallback if scalars are not yet baked in
            
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(scalar, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
