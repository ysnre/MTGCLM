# Goal Description

Integrate 10-minute ground observations from AWOS stations (Temperature, Relative Humidity, and Elevation) into the existing MTGCLM deep learning pipeline as a 3-dimensional scalar feature vector. This involves fusing these features with 17-channel satellite/radar data using Late Fusion for classification models (MTGConvNet, ResNet18) and Bottleneck Fusion for semantic segmentation models (U-Net). The models and training logic will be refactored to support these multi-modal inputs, ensuring robust Colab compatibility and strict adherence to the project's data leakage prevention rules.

## User Review Required

> [!IMPORTANT]
> The HDF5 dataset (`MTGH5Dataset` located in `src/h5_dataset.py`) currently reads only `patches` and `labels`. To fetch AWOS data dynamically during training, we must extract the timestamp and WMOID for each sample from the H5 file metadata and look them up in the AWOS scalar loader. Is this dynamic lookup during training acceptable, or do you plan to pre-extract these 3 scalar values directly into the HDF5 file in a future step? I will implement the dynamic lookup for now as requested.

## Open Questions

> [!WARNING]
> 1. The instructions mention updating `MTGH5Dataset` in `src/dataset.py`, but it is currently located in `src/h5_dataset.py`. I will update it in `src/h5_dataset.py`. Is that correct?
> 2. For the `MaskedBCELoss`, the target labels currently provided by the dataloader are `0` (Clear), `1` (Cloud), and `-1` (Ignore). Should the loss use PyTorch's `BCEWithLogitsLoss` internally, and should it handle the `-1` ignore index dynamically?
> 3. You mentioned "colab'da çalışacağım şekilde düzenle" (Organize for Colab). The paths in `config.py` already support Colab. Should I ensure that the new `RAW_AWOS_OBSERVATION` directory is added to `config.py` with a Colab-friendly path (e.g., `/content/drive/MyDrive/MTGCLM/data/raw_awos_observation`)?

## Proposed Changes

---

### Phase 1: Data Engineering (AWOS Scalar Loader)
Create the AWOS scalar data processor.

#### [NEW] [awos_scalar_loader.py](file:///e:/belgeler/MTGCLM/src/awos_scalar_loader.py)
- Implement `AWOSScalarLoader` class.
- Load `stationlist_awos.csv` to get static `elevation`.
- Load `observation_awos.csv` and parse datetime.
- Merge dynamic (Temp, RH) and static (Elevation) data.
- Handle missing values (NaN, -9999) with median imputation.
- Apply Z-score normalization.
- Provide `get_features(wmoid, timestamp)` with +/- 5 minutes tolerance.

#### [MODIFY] [config.py](file:///e:/belgeler/MTGCLM/src/config.py)
- Add Colab-compatible paths for the raw AWOS observation CSV files.

---

### Phase 2: Dataset and DataLoader Update
Refactor datasets to output the new 3-element tuple and apply spatial augmentations.

#### [MODIFY] [h5_dataset.py](file:///e:/belgeler/MTGCLM/src/h5_dataset.py)
- Modify `MTGH5Dataset.__getitem__` to lookup `wmoid` and `timestamp`, fetch the 3-scalar vector from `AWOSScalarLoader`, and return `(patch, scalars, label)`.

#### [MODIFY] [dataloader.py](file:///e:/belgeler/MTGCLM/src/dataloader.py)
- Update `RandomSpatialTransforms` (Random Horizontal and Vertical Flips).
- Ensure the collate function or standard PyTorch batched loading handles the 3-element tuple properly.

---

### Phase 3: PyTorch Model Architectures
Expand models to accept the scalar feature vector.

#### [MODIFY] [model.py](file:///e:/belgeler/MTGCLM/src/model.py)
- `MTGConvNet_Flat`: Standard 17-channel CNN without scalars (Spatial Flattening).
- `MTGConvNet_LateFusion`: Concat 3-scalar vector to the flattened 1D spatial features before FC layers.
- `ResNet18_LateFusion`: ResNet18 adapted for 17 channels, Flattened, concatenated with scalars.
- `CloudUNet_Fusion`: Bottleneck fusion. Pass scalars through MLP, broadcast to `[B, C, H, W]` to match bottleneck, and add/concat. Output both `[B, 1, 33, 33]` mask and `[B, 1]` center-pixel score.

---

### Phase 4: Custom Loss and Training Loop
Implement training scripts.

#### [NEW] [loss.py](file:///e:/belgeler/MTGCLM/src/loss.py)
- Implement `MaskedBCELoss` that calculates binary cross-entropy exclusively at the `(16, 16)` spatial coordinate, ignoring index `-1`.

#### [MODIFY] [train.py](file:///e:/belgeler/MTGCLM/src/train.py)
- Add `--model` argument parsing (`mtg_flat`, `mtg_late_fusion`, `resnet18_fusion`, `unet_fusion`).
- Modify the loop: `for features, scalars, labels in loader:`.
- Route the `scalars` to the models that require them.
- Maintain 5-Fold Cross Validation loop and logging of Accuracy, Precision, Recall, and F1-Score.

## Verification Plan

### Automated Tests
- Run a dry-run execution of `train.py` with each of the 4 models (`mtg_flat`, `mtg_late_fusion`, `resnet18_fusion`, `unet_fusion`) for 1 batch to ensure tensor shapes align and backpropagation works correctly without crashing.

### Manual Verification
- Review the generated loss curves and verify that the `MaskedBCELoss` properly isolates the `(16, 16)` pixel.
- Verify Colab compatibility by checking that paths fallback gracefully.
