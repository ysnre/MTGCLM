import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

# Auto-detect Google Colab
IS_COLAB = os.path.exists('/content')

if IS_COLAB:
    # Google Colab paths
    BASE_DIR = "/content/drive/MyDrive/MTGCLM"
    NUM_WORKERS = 0  # Set to 0 to avoid HDF5 multiprocessing deadlocks
    
    # Read from fast local SSD on Colab
    RAW_MTG_DIR = "/content/turkiye_kesitler"
    PROCESSED_RADAR_DIR = "/content/radar_aligned"
    PROCESSED_AWOS_DIR = "/content/awos_aligned"
else:
    # Local Windows paths
    BASE_DIR = "e:/belgeler/MTGCLM"
    NUM_WORKERS = 0
    
    RAW_MTG_DIR = os.path.join(BASE_DIR, "data/raw_mtg")
    PROCESSED_RADAR_DIR = os.path.join(BASE_DIR, "data/processed/radar_aligned")
    PROCESSED_AWOS_DIR = os.path.join(BASE_DIR, "data/processed/awos_aligned")

# Model Training Settings
PATCH_SIZE = 33
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

# Path to pre-extracted HDF5 patch dataset (Scenario 3)
H5_PATH = os.environ.get("MTGCLM_H5") or ("/content/mtg_patch_dataset.h5" if IS_COLAB else "F:/radar/mtg_patch_dataset.h5")

# Hardware selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- Configuration loaded ---")
print(f"Platform detected: {'Google Colab' if IS_COLAB else 'Local Windows'}")
print(f"Base Directory:    {BASE_DIR}")
print(f"Compute Device:    {DEVICE}")
print(f"----------------------------")