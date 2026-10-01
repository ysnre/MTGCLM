# Task List

- `[x]` **Phase 1: Data Engineering**
  - `[x]` Create `src/awos_scalar_loader.py` to parse CSVs, normalize data, and match by time and wmoid.
  - `[x]` Create `src/update_h5_scalars.py` to bake `[Temperature, Humidity, Elevation]` vectors directly into the local `.h5` file (addressing Colab integration).

- `[x]` **Phase 2: Dataset and DataLoader Update**
  - `[x]` Modify `src/h5_dataset.py` to lazily read the new `scalars` dataset from the HDF5 file.
  - `[x]` Modify `src/dataloader.py` so `RandomSpatialTransforms` and subset wrappers properly handle the 3-element tuple `(patch, scalars, labels)`.

- `[x]` **Phase 3: PyTorch Model Architectures**
  - `[x]` Update `src/model.py` with `MTGConvNet_LateFusion`.
  - `[x]` Update `src/model.py` with `ResNet18_LateFusion`.
  - `[x]` Update `src/model.py` with `CloudUNet_Fusion` (Bottleneck MLP + Broadcasting, dual output head).

- `[x]` **Phase 4: Custom Loss and Training Loop**
  - `[x]` Create `src/loss.py` with `MaskedBCELoss` (ignoring `-1` index).
  - `[x]` Update `src/train.py` to support `MTGConvNet_LateFusion` and `ResNet18_LateFusion` (keeping MTGConvNet separated).
  - `[x]` Update `src/train_unet.py` to support `CloudUNet_Fusion` and `MaskedBCELoss` (keeping U-Net separated).

- `[ ]` **Phase 5: Verification**
  - `[ ]` Run `src/update_h5_scalars.py` locally to populate the HDF5 file.
  - `[ ]` Run `train.py` dry-run locally.
  - `[ ]` Run `train_unet.py` dry-run locally.
