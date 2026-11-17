# MTGCLM v1.0 — release notes (draft)

Station-trained cloud mask for Meteosat Third Generation FCI over Türkiye. Archive: https://doi.org/10.5281/zenodo.24112650

## Contents of this release
- Full pipeline from raw inputs to the compact training set (`src/`), training and evaluation with crash-safe
  orchestration (`src/run_all.py`), and the v3 analysis scripts (`analyze_preds.py`, `channel_importance.py`,
  `error_analysis.py`, `label_sensitivity.py`, `parallax_shift.py`, `references_by_period.py`).
- Full-scene inference (`infer_scene.py`, phase-aligned U-Net) and cloud occurrence frequency (`batch_scenes.py`,
  `cloud_frequency.py`).
- Split definitions (`splits/`) and per-run results (`artifacts_v3/*.json`, `artifacts_v3/analysis/*.csv`).
- Two-layer public sample (`make_public_sample.py`) and Zenodo packaging (`make_zenodo_package.py`).

## Assets attached to the release
- `best_model_B_full_s1__medium_flat.pth` — CNN (Medium_Flat), all inputs, seed 1
- `best_model_D_unet_s1__mtgunet.pth` — U-Net, all inputs, seed 1

## Data
Both layers of the stratified sample are in the Zenodo record: layer (a) (FCI channels, static fields, SYNOP/METAR
labels) and layer (b) (radar and AWOS-derived fields, shared with the permission of the Turkish State Meteorological
Service). The raw radar and surface observations are available from the Turkish State Meteorological Service on request.

## Reproducibility
Seeds 1–3, deterministic PyTorch settings (`src/run_utils.py`); see README, section "v3 sonuçlarını yeniden üretme".
