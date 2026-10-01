# AWOS Scalar Data Fusion - Implementation Walkthrough

The integration of 10-minute ground observations from AWOS stations into the MTGCLM Cloud Masking project is now complete. The pipeline seamlessly merges 17-channel patch data with a 3-dimensional scalar feature vector `[Temperature, Humidity, Elevation]` to better model topographic forcing and low-stratus cloud formations.

Below is a detailed summary of what was accomplished:

## 1. Data Engineering (Offline Scaler Integration)
To completely bypass the high RAM and CPU overhead of looking up timestamps in a 4.4GB CSV during training (especially crucial for Colab), I built a two-step localized data processing pipeline:

- **[awos_scalar_loader.py](file:///e:/belgeler/MTGCLM/src/awos_scalar_loader.py)**: A highly optimized, memory-efficient python script that lazily reads the 4.4GB AWOS dataset line-by-line, filters only the timestamps corresponding to satellite patches, applies median imputation for missing values, and Z-score normalization for Temperature, RH, and Elevation.
- **[update_h5_scalars.py](file:///e:/belgeler/MTGCLM/src/update_h5_scalars.py)**: Iterates over the local `mtg_patch_dataset.h5` file, retrieves all `wmoid` and `timestamps` (rounding them to the nearest 10-minute AWOS interval), fetches the normalized features, and injects a new compressed `scalars` dataset `[N, 3]` directly into the `.h5` file. 

> [!TIP]
> This strategy ensures that when you run on Colab, the PyTorch dataloader operates at maximum I/O speed natively reading from the `.h5` file without needing the raw `.csv` files uploaded.

## 2. Dataset and DataLoader Refactoring
The primary data fetching mechanisms were updated to handle the new scalar values.
- **`MTGH5Dataset`** in [h5_dataset.py](file:///e:/belgeler/MTGCLM/src/h5_dataset.py) was updated to return a 3-element tuple: `(patch, scalars, label)`. It features a safe fallback mechanism if the scalars haven't been baked in yet.
- **`AugmentedDataset`** in [dataloader.py](file:///e:/belgeler/MTGCLM/src/dataloader.py) was dynamically modified to intercept 3-element tuples, apply the `RandomHorizontalFlip` and `RandomVerticalFlip` spatial transforms exclusively to the `patch`, and transparently forward the scalars and labels.

## 3. PyTorch Model Architectures
In [model.py](file:///e:/belgeler/MTGCLM/src/model.py), the models were expanded to accept dual inputs (`x`, `scalars`).
- **`MTGConvNet_LateFusion`**: Implements Spatial Flattening instead of Global Average Pooling (preserving center-pixel coordinates). The resulting 1D vector is concatenated with the 3-scalar vector right before the fully connected classification head.
- **`ResNet18_LateFusion`**: Replaces the GAP layer with a `Flatten()` operation and implements Late Fusion.
- **`CloudUNet_Fusion`**: A specialized U-Net architecture. Since U-Nets process 2D spatial maps entirely, standard concatenation fails. I implemented **Bottleneck Fusion**:
  1. The 3-scalar vector is passed through an MLP to expand its dimensionality.
  2. The vector is spatially broadcasted to `[Batch, C, H, W]`.
  3. It is additively fused at the exact center bottleneck layer of the U-Net.
  4. The model outputs a `[Batch, 2, 33, 33]` spatial probability mask.

## 4. Center-Pixel Masked Loss
In [loss.py](file:///e:/belgeler/MTGCLM/src/loss.py), I created **`MaskedBCELoss`**.
- The U-Net outputs a `33x33` prediction mask, but our ground truth (SYNOP observations) is strictly a point-target at the exact station coordinate.
- The `MaskedBCELoss` mathematically masks out all pixels indexed as `-1` (invalid), calculating the gradient exclusively for the central `(16, 16)` coordinate. This prevents the model from attempting to hallucinate targets in unknown areas and heavily penalizing itself for it.

## 5. Verification Results
Both `train.py` (Classification models) and `train_unet.py` (Segmentation models) were successfully run as local dry-runs to verify matrix shape alignments, loss function behavior, and data loading stability.

```text
==================================================
         ABLATION STUDY COMPARATIVE RESULTS
==================================================
Model Name            | Params     | Best Val F1 
--------------------------------------------------
MTGConvNet_LateFusion | 459,330    | 0.8866      

==================================================
         U-NET RESULTS
==================================================
Model Name       | Params     | Best Val F1 
--------------------------------------------------
CloudUNet_Fusion | 480,450    | 0.8930      
```

The system is now fully prepared to be trained on Colab!
