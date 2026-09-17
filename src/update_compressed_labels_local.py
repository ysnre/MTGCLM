import h5py
import os

def main():
    src_path = "F:/radar/mtg_patch_dataset.h5"
    dst_path = "F:/radar/mtg_patch_dataset_compressed.h5"
    
    if not os.path.exists(src_path):
        print(f"Source file {src_path} not found.")
        return
    if not os.path.exists(dst_path):
        print(f"Destination compressed file {dst_path} not found.")
        return
        
    print("Updating labels in compressed dataset in-place (takes 1 second)...")
    with h5py.File(src_path, 'r') as src:
        labels = src['labels'][:]
        
    with h5py.File(dst_path, 'r+') as dst:
        # Overwrite the labels dataset
        dst['labels'][:] = labels
        
    print("Successfully updated compressed dataset labels in-place!")

if __name__ == "__main__":
    main()
