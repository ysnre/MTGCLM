import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset import MTGCloudDataset
from h5_dataset import MTGH5Dataset

def get_train_val_split(dataset, val_ratio: float = 0.2):
    """
    Splits the dataset temporally (by date) to prevent data leakage.
    Shuffles unique dates with a fixed seed for reproducibility and seasonal balance,
    ensuring both sets contain a representative mix of seasons (Spring, Summer, Autumn).
    
    Args:
        dataset: The initialized dataset.
        val_ratio (float): Fraction of dates to keep for validation.
        
    Returns:
        tuple: (train_dataset, val_dataset) as PyTorch Subsets.
    """
    if isinstance(dataset, MTGH5Dataset):
        timestamps = dataset.timestamps
        unique_dates = sorted(list(set([ts[:8] for ts in timestamps])))
        
        # Shuffle dates with a fixed seed to distribute seasons evenly
        rng = np.random.default_rng(seed=42)
        shuffled_dates = list(unique_dates)
        rng.shuffle(shuffled_dates)
        
        num_val_dates = max(1, int(len(shuffled_dates) * val_ratio))
        val_dates = set(shuffled_dates[:num_val_dates])
        train_dates = set(shuffled_dates[num_val_dates:])
        
        print(f"Temporal Split (H5 Dataset - Seed 42 Shuffle):")
        print(f"  Train dates: {sorted(list(train_dates))}")
        print(f"  Val dates:   {sorted(list(val_dates))}")
        
        train_indices = []
        val_indices = []
        for idx, ts in enumerate(timestamps):
            date = ts[:8]
            if date in train_dates:
                train_indices.append(idx)
            else:
                val_indices.append(idx)
    else:
        # Extract unique dates from the samples (YYYYMMDD)
        timestamps = [s['timestamp'] for s in dataset.samples]
        unique_dates = sorted(list(set([ts[:8] for ts in timestamps])))
        
        # Shuffle dates with a fixed seed to distribute seasons evenly
        rng = np.random.default_rng(seed=42)
        shuffled_dates = list(unique_dates)
        rng.shuffle(shuffled_dates)
        
        num_val_dates = max(1, int(len(shuffled_dates) * val_ratio))
        val_dates = set(shuffled_dates[:num_val_dates])
        train_dates = set(shuffled_dates[num_val_dates:])
        
        print(f"Temporal Split (Cloud Dataset - Seed 42 Shuffle):")
        print(f"  Train dates: {sorted(list(train_dates))}")
        print(f"  Val dates:   {sorted(list(val_dates))}")
        
        train_indices = []
        val_indices = []
        
        if dataset.mode == "patch":
            for idx, item in enumerate(dataset.patch_samples):
                ts_idx = item['ts_idx']
                date = dataset.samples[ts_idx]['timestamp'][:8]
                if date in train_dates:
                    train_indices.append(idx)
                else:
                    val_indices.append(idx)
        else: # "full" mode
            for idx, s in enumerate(dataset.samples):
                date = s['timestamp'][:8]
                if date in train_dates:
                    train_indices.append(idx)
                else:
                    val_indices.append(idx)
                
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    print(f"Split completed:")
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Val samples:   {len(val_subset)}")
    
    return train_subset, val_subset

import random
import torch
from torch.utils.data import Dataset as TorchDataset

class RandomSpatialTransforms:
    """
    Applies random horizontal and vertical flips to a PyTorch tensor patch [C, H, W].
    Preserves the center coordinate (e.g. at [C, 16, 16] for 33x33 patches),
    making it completely safe for point-based targets like AWOS.
    """
    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        # Random horizontal flip
        if random.random() > 0.5:
            patch = torch.flip(patch, dims=[2])  # flip width dimension
        # Random vertical flip
        if random.random() > 0.5:
            patch = torch.flip(patch, dims=[1])  # flip height dimension
        return patch

class AugmentedDataset(TorchDataset):
    """
    Wrapper around a PyTorch Dataset (or Subset) to apply online data augmentation
    only during the training phase.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def _apply(self, item):
        if len(item) == 3:
            patch, scalars, label = item
            if self.transform:
                patch = self.transform(patch)
            return patch, scalars, label
        patch, label = item
        if self.transform:
            patch = self.transform(patch)
        return patch, label

    def __getitem__(self, idx: int) -> tuple:
        return self._apply(self.dataset[idx])

    def __getitems__(self, indices):
        """Alttaki dataset toplu okumayı destekliyorsa ona devret (bkz. MTGH5Dataset.__getitems__).
        Artırma (flip) örneklere ÇAĞRILDIKLARI SIRAYLA uygulanır; yani rastgele sayı üretecinin
        tüketim sırası tek tek okumayla birebir aynıdır, sonuçlar değişmez."""
        fn = getattr(self.dataset, "__getitems__", None)
        items = fn(indices) if callable(fn) else [self.dataset[i] for i in indices]
        return [self._apply(it) for it in items]


def _check_split_matches_h5(split, dataset, h5_path: str, split_file: str) -> None:
    """
    Bölme dosyasının BU H5 ile eşleştiğini doğrular. En sık hata: kompakt H5 için üretilmiş
    split'i büyük kaynak H5'e (ya da tersi) uygulamak -> sessizce yanlış örnekler okunur.
    """
    n = len(dataset)
    labels = np.asarray(dataset.labels)
    okta = None if getattr(dataset, "okta", None) is None else np.asarray(dataset.okta)
    problems = []
    for name in split.files:
        idx = split[name].astype(np.int64)
        if len(idx) == 0:
            continue
        if idx.max() >= n:
            problems.append(f"{name}: en büyük indeks {idx.max()} >= H5 satır sayısı {n}")
            continue
        if name == "test_partial":
            if okta is not None:
                bad = int(((okta[idx] < 1) | (okta[idx] > 4)).sum())
                if bad:
                    problems.append(f"test_partial: {bad}/{len(idx)} örnek 1-4 okta aralığı dışında")
        else:
            bad = int((labels[idx] < 0).sum())
            if bad:
                problems.append(f"{name}: {bad}/{len(idx)} örneğin etiketi -1 (geçersiz)")
    if problems:
        raise SystemExit(
            "\nHATA: bölme dosyası bu HDF5 ile uyuşmuyor!\n"
            f"  H5    : {h5_path}  ({n} satır)\n"
            f"  split : {split_file}\n  - " + "\n  - ".join(problems) +
            "\n\nOlası sebep: kompakt H5 (mtg_train_v2.h5) icin uretilmis '*_compact.npz' dosyasini "
            "kaynak H5'e uygulamak ya da tersi. --h5 ile dogru dosyayi verin "
            "(veya MTGCLM_H5 ortam degiskenini ayni kabukta ayarlayin).\n"
        )
    print(f"  Bölme doğrulandı: {n} satırlık H5 ile uyumlu.")

def get_split_dataloaders(h5_path: str,
                          split_file: str,
                          batch_size: int = 64,
                          num_workers: int = 0,
                          use_aux: bool = False,
                          aux_channels=None,
                          shuffle_train: bool = False,
                          f16: bool = False) -> dict:
    """
    make_splits.py ile üretilen .npz bölme dosyasından DataLoader sözlüğü üretir:
        {"train", "val", "test_time", "test_station", "test_both", "test_partial"}  (boş alt kümeler atlanır)
    Eğitim alt kümesi RandomSpatialTransforms ile sarılır. Diğerleri sıralı okunur.
    shuffle_train=False: sıralı H5 okuması; karıştırma train.py'deki buffered_shuffle_generator'da
    (ya da önceden karıştırılmış H5 ile) yapılır.
    """
    print(f"Initializing MTGH5Dataset from {h5_path} (use_aux={use_aux})...")
    full_dataset = MTGH5Dataset(h5_path, use_aux=use_aux, aux_channels=aux_channels, f16=f16)
    if f16:
        print("  Veri yolu: float16 (float32'ye cevirme ve aux z-skoru GPU'da)")
    print(f"  Kanal sayısı: {full_dataset.num_channels} (uydu/radar {full_dataset.base_channels} + aux {full_dataset.aux_names})")
    split = np.load(split_file)
    _check_split_matches_h5(split, full_dataset, h5_path, split_file)
    loaders = {}
    for name in ("train", "val", "test_time", "test_station", "test_both", "test_partial"):
        if name not in split.files or len(split[name]) == 0:
            continue
        idx = split[name].astype(np.int64).tolist()
        subset = Subset(full_dataset, idx)
        kw = dict(num_workers=num_workers, pin_memory=num_workers > 0)
        if num_workers > 0:
            kw.update(persistent_workers=True, prefetch_factor=4)
        if name == "train":
            ds = AugmentedDataset(subset, transform=RandomSpatialTransforms())
            loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle_train, **kw)
        else:
            loaders[name] = DataLoader(subset, batch_size=batch_size, shuffle=False, **kw)
        print(f"  {name:13s}: {len(idx)} örnek")
    return loaders


def get_dataloaders(mtg_dir: str = None,
                    radar_dir: str = None,
                    awos_dir: str = None,
                    h5_path: str = None,
                    mode: str = "patch",
                    patch_size: int = 33,
                    batch_size: int = 32,
                    val_ratio: float = 0.2,
                    num_workers: int = 0,
                    shuffle: bool = True,
                    use_aux: bool = False,
                    aux_channels=None) -> tuple:
    """
    Creates PyTorch DataLoaders for training and validation.

    Returns:
        tuple: (train_loader, val_loader)
    """
    if h5_path is not None:
        print(f"Initializing MTGH5Dataset from {h5_path}...")
        full_dataset = MTGH5Dataset(h5_path, use_aux=use_aux, aux_channels=aux_channels)
    else:
        # Initialize full dataset
        full_dataset = MTGCloudDataset(
            mtg_dir=mtg_dir,
            radar_dir=radar_dir,
            awos_dir=awos_dir,
            mode=mode,
            patch_size=patch_size
        )
    
    # Split
    train_subset, val_subset = get_train_val_split(full_dataset, val_ratio)
    
    # Wrap train subset with spatial augmentations for training
    train_dataset = AugmentedDataset(train_subset, transform=RandomSpatialTransforms())
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test dataloader creation
    mtg_dir = "e:/belgeler/MTGCLM/data/raw_mtg"
    radar_dir = "e:/belgeler/MTGCLM/data/processed/radar_aligned"
    awos_dir = "e:/belgeler/MTGCLM/data/processed/awos_aligned"
    
    train_loader, val_loader = get_dataloaders(
        mtg_dir=mtg_dir,
        radar_dir=radar_dir,
        awos_dir=awos_dir,
        mode="patch",
        batch_size=16
    )
    
    # Pull one batch
    batch = next(iter(train_loader))
    if len(batch) == 3:
        features, scalars, labels = batch
        print(f"\nBatch extracted:")
        print(f"  Features shape: {features.shape} (expected: [Batch, 17, Patch, Patch])")
        print(f"  Scalars shape:  {scalars.shape} (expected: [Batch, 3])")
        print(f"  Labels shape:   {labels.shape} (expected: [Batch])")
        print(f"  Label distribution in batch: {np.bincount(labels.numpy())}")
    else:
        features, labels = batch
        print(f"\nBatch extracted:")
        print(f"  Features shape: {features.shape} (expected: [Batch, 17, Patch, Patch])")
        print(f"  Labels shape:   {labels.shape} (expected: [Batch])")
        print(f"  Label distribution in batch: {np.bincount(labels.numpy())}")
