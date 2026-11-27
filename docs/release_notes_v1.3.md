# MTGCLM v1.3 — release notes

Station-trained cloud mask for Meteosat Third Generation FCI over Türkiye. This release corresponds to the submitted
manuscript. Zenodo archive (all versions): https://doi.org/10.5281/zenodo.24112649 — this version:
https://doi.org/10.5281/zenodo.24560405

## Contents of this release
- Full pipeline from raw inputs to the compact training set (`src/`), training and evaluation with crash-safe
  orchestration (`src/run_all.py`), and the v3 analysis scripts (`analyze_preds.py`, `channel_importance.py`,
  `error_analysis.py`, `label_sensitivity.py`, `parallax_shift.py`, `references_by_period.py`).
- Revision analyses (contingency scores, multiplicity correction, reference-calibrated baseline, observer ceiling,
  AWOS co-location, IDW leave-one-out, normalisation shift, snow/VZA breakdown); see README, "Revizyon analizleri".
- Full-scene inference (`infer_scene.py`; the U-Net trained with centre-pixel jitter runs in a single pass) and cloud
  occurrence frequency (`batch_scenes.py`, `cloud_frequency.py`).
- Split definitions (`splits/`) and per-run results (`artifacts_v3/*.json`, `artifacts_v3/analysis/*.csv`).
- Row index of all samples without data values (`make_sample_index.py`) and Zenodo packaging (`make_zenodo_package.py`).

## Zenodo record (v1.3)
- Row index (FCI scene time, station, grid position, split set), split files, run results and analysis tables,
  cloud occurrence frequency maps.
- Weights: `best_model_B_full_s1__medium_flat.pth` (CNN, all inputs), `best_model_D_unet_s1__mtgunet.pth`
  (centre-supervised U-Net), `best_model_D_unet_jit_s1__mtgunet.pth` (U-Net with centre-pixel jitter, used for full scenes).

## Data
No original data values are redistributed. FCI Level-1c and FCI CLM are available free of charge from the EUMETSAT
Data Store; under the EUMETSAT Data Policy non-hourly Level-1 data may not be redistributed, so FCI patches are rebuilt
from the downloaded files with `src/coknc.py` and `src/create_patch_dataset.py` using the row index. Radar and
SYNOP/METAR/AWOS observations belong to the Turkish State Meteorological Service (MGM) and are available from MGM on request.
Contains modified EUMETSAT Meteosat data (2025–2026).

## Reproducibility
Seeds 1–3, deterministic PyTorch settings (`src/run_utils.py`); see README, section "v3 sonuçlarını yeniden üretme".
