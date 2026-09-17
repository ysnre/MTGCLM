import h5py
import time
import os
import numpy as np

src_path = "F:/radar/mtg_patch_dataset.h5"
dst_path = "F:/radar/mtg_patch_dataset_compressed.h5"

def main():
    if not os.path.exists(src_path):
        print(f"Source file {src_path} not found.")
        return
        
    print("Compressing HDF5 dataset to resolve Google Colab storage limits...")
    print(f"Source: {src_path}")
    print(f"Destination: {dst_path}")
    
    t0 = time.time()
    
    with h5py.File(src_path, "r") as src:
        with h5py.File(dst_path, "w") as dst:
            # Copy file attributes (normalization stats)
            for name, value in src.attrs.items():
                dst.attrs[name] = value
                
            # Copy datasets
            for key in src.keys():
                print(f"\nProcessing dataset: {key}")
                shape = src[key].shape
                dtype = src[key].dtype
                
                if key == 'patches':
                    # Create target compressed dataset
                    dst_ds = dst.create_dataset(
                        key,
                        shape=shape,
                        dtype=dtype,
                        chunks=(2048, 17, 33, 33),
                        compression="gzip",
                        compression_opts=4
                    )
                    
                    # Copy in memory-safe chunks of 10,000 samples (uses ~740 MB RAM)
                    block_size = 10000
                    num_samples = shape[0]
                    for start in range(0, num_samples, block_size):
                        end = min(start + block_size, num_samples)
                        print(f"  Compressing patches: {start}/{num_samples} ({start/num_samples*100.0:.1f}%)")
                        dst_ds[start:end] = src[key][start:end]
                else:
                    # Create target compressed dataset for labels, timestamps, wmoids
                    dst_ds = dst.create_dataset(
                        key,
                        shape=shape,
                        dtype=dtype,
                        chunks=True,
                        compression="gzip",
                        compression_opts=4
                    )
                    # Small datasets can be copied at once
                    dst_ds[:] = src[key][:]
                    print(f"  Copied {key} successfully.")

    elapsed = time.time() - t0
    src_size = os.path.getsize(src_path) / (1024**3)
    dst_size = os.path.getsize(dst_path) / (1024**3)
    
    print(f"\nFinished compression successfully in {elapsed/60:.2f} minutes!")
    print(f"Original size:  {src_size:.2f} GB")
    print(f"Compressed size: {dst_size:.2f} GB (Ratio: {src_size/dst_size:.2f}x)")
    print(f"You can now upload 'mtg_patch_dataset_compressed.h5' directly to Google Drive!")

if __name__ == "__main__":
    main()
