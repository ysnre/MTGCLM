# Graph Report - MTGCLM  (2026-09-17)

## Corpus Check
- 63 files · ~96,234 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: (none) 2, .npz 2, .ipynb 1)

## Summary
- 352 nodes · 642 edges · 17 communities (15 shown, 2 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 82 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Bölme, H5 ve Eğitim Betikleri
- Lat/Lon Grid ve Aux Kanallar
- AWOS Piksel Eşleme ve Uydu Resample
- v2 Proje Mimarisi ve Modeller
- Radar Kompozit Hattı ve Skill
- Etiket, Kayıp ve Model Seçimi
- Zaman/SZA ve Dedup
- DataLoader ve Augmentasyon
- Büyük Ölçek Radar İşleme
- Konsolide Etiket Dataset'i (v1)
- Patch Dataset Oluşturma
- AWOS Skaler Yükleyici
- Ablasyon Sonuçları ve Grafikleri
- İstasyon Yama Görselleri
- Eğitim Döngüsü Yardımcıları

## God Nodes (most connected - your core abstractions)
1. `main()` - 17 edges
2. `LatLonGrid` - 16 edges
3. `MTGH5Dataset` - 14 edges
4. `MTGCloudDataset` - 13 edges
5. `run_experiment()` - 11 edges
6. `get_dataloaders()` - 9 edges
7. `run_experiment()` - 9 edges
8. `AugmentedDataset` - 8 edges
9. `main()` - 8 edges
10. `get_split_dataloaders()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Raster Aux Channels (K1, 22 channels)` --semantically_similar_to--> `CloudUNet_Fusion`  [INFERRED] [semantically similar]
  CLAUDE.md → src/model.py
- `MTGCLM_MONGO_URI env variable` --shares_data_with--> `init_worker()`  [INFERRED]
  .agent/skills/radar-skill/SKILL.md → src/process_radar_pipeline_large.py
- `MaskedBCELoss` --conceptually_related_to--> `Label Policy clear=0 / cloudy>=5 okta (K2)`  [INFERRED]
  src/loss.py → CLAUDE.md
- `ResNet18 validation instability (F1 collapse, loss spikes)` --conceptually_related_to--> `ResNet18_LateFusion`  [INFERRED]
  artifacts/ablation_study_f1.png → src/model.py
- `Consolidated Ground Truth (AWOS+Radar consensus labels)` --rationale_for--> `MTGCloudDataset`  [EXTRACTED]
  .agent/steps/implementation_plan_dataset.md → src/dataset.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **AWOS station pixel matching pipeline** — src_process_awos_pipeline, src_awos_matcher, src_observation_loader [EXTRACTED 1.00]
- **Radar PNG to MTG grid alignment pipeline** — src_process_radar_pipeline, src_radar_loader, src_resampler, src_satellite_grid [EXTRACTED 1.00]
- **Leakage-safe labeling and evaluation decisions** — claude_label_policy, claude_dedup_policy, claude_leakage_free_split, claude_balanced_accuracy_selection [INFERRED 0.85]
- **v1 scalar late fusion models** — src_model_mtgconvnet_latefusion, src_model_resnet18_latefusion, src_model_cloudunet_fusion, src_update_h5_scalars [EXTRACTED 1.00]
- **v2 pipeline aux -> labels -> splits -> export** — src_update_h5_aux_channels, src_update_h5_labels_v2, src_make_splits, src_export_compact_h5 [EXTRACTED 1.00]
- **Architecture ablation validation curves** — artifacts_ablation_study_f1, artifacts_ablation_study_loss, artifacts_ablation_results [INFERRED 0.85]

## Communities (17 total, 2 thin omitted)

### Community 0 - "Bölme, H5 ve Eğitim Betikleri"
Cohesion: 0.09
Nodes (27): argparse, Week-block + Station Holdout Split (K4), datetime, h5py, json, matplotlib, matplotlib_pyplot, numpy (+19 more)

### Community 1 - "Lat/Lon Grid ve Aux Kanallar"
Cohesion: 0.07
Nodes (30): LatLonGrid, Düzenli enlem/boylam grid'i. Satır = enlem (y), sütun = boylam (x)., turkiye_kesit_*.nc dosyalarındaki grid (800 x 1000)., (lat, lon) -> (y_km, x_km) referans noktasına göre., En yakın (y, x) piksel indeksleri (grid dışına taşabilir; clip edilmez)., (yc, xc) merkezli (2*half+1)^2 pencerenin lat/lon 2D dizileri (grid dışı lineer…, main(), IDWInterpolator (+22 more)

### Community 2 - "AWOS Piksel Eşleme ve Uydu Resample"
Cohesion: 0.09
Nodes (31): Skill: AWOS Pixel Match, Skill: Satellite Resample (Satpy), AWOS Pixel Match Implementation Plan, +/-9 min AWOS time tolerance, collections, glob, hdf5plugin, netcdf4 (+23 more)

### Community 3 - "v2 Proje Mimarisi ve Modeller"
Cohesion: 0.07
Nodes (23): Raster Aux Channels (K1, 22 channels), Colab Training Workflow, Full-map 800x1000 Cloud Mask, MTGCLM Project (Cloud Mask over Turkey), v1 Late Fusion Implementation Plan, Python Dependencies (torch, h5py, xarray, pyresample...), CloudUNet_Fusion, MTGConvNet (+15 more)

### Community 4 - "Radar Kompozit Hattı ve Skill"
Cohesion: 0.10
Nodes (24): Rule_radar (manual trigger), MTGCLM_MONGO_URI env variable, Radar MongoDB product metadata (PPI, ll bbox), radar-skill, MTG RS/AI engineer agent persona, Walkthrough (dataset copy), Maximum Reflectivity compositing, Radar-to-MTG grid alignment walkthrough (+16 more)

### Community 5 - "Etiket, Kayıp ve Model Seçimi"
Cohesion: 0.09
Nodes (23): Val Balanced Accuracy Model Selection (K5), Label Policy clear=0 / cloudy>=5 okta (K2), MaskedBCELoss, Tensor, Args: preds: [Batch, 2, H, W] (raw logits from U-Net) targets: [Batch, H, W]…, Custom Loss for U-Net Cloud Masking. Computes Binary Cross Entropy strictly for…, compute_metrics(), format_metrics() (+15 more)

### Community 6 - "Zaman/SZA ve Dedup"
Cohesion: 0.12
Nodes (23): Duplicate Sample Dedup (K3), cos_solar_zenith(), floor_to_10min(), _is_leap(), nearest_10min(), padded_window(), parse_sat_timestamp(), datetime (+15 more)

### Community 7 - "DataLoader ve Augmentasyon"
Cohesion: 0.10
Nodes (17): AugmentedDataset, get_dataloaders(), get_split_dataloaders(), get_train_val_split(), Tensor, RandomSpatialTransforms, Wrapper around a PyTorch Dataset (or Subset) to apply online data augmentation…, make_splits.py ile üretilen .npz bölme dosyasından DataLoader sözlüğü üretir:… (+9 more)

### Community 8 - "Büyük Ölçek Radar İşleme"
Cohesion: 0.12
Nodes (22): bisect, multiprocessing, pymongo, pyresample_geometry, pyresample_kd_tree, get_satellite_grid_local(), init_worker(), main() (+14 more)

### Community 9 - "Konsolide Etiket Dataset'i (v1)"
Cohesion: 0.16
Nodes (9): Consolidated Ground Truth (AWOS+Radar consensus labels), AI Dataset & DataLoader Plan, MTGCloudDataset, Dataset, ndarray, Loads and normalizes satellite and radar features. Returns a normalized (17,…, PyTorch Dataset for MTG Cloud Masking. Args: mtg_dir (str): Directory…, Applies the Consolidated Ground Truth logic to determine clear/cloudy label. -… (+1 more)

### Community 10 - "Patch Dataset Oluşturma"
Cohesion: 0.22
Nodes (12): compute_normalization_stats_fast(), get_channel_names(), get_consolidated_label(), load_features_fast(), main(), parse_mtg_time(), ndarray, Parses timestamp from MTG file name. (+4 more)

### Community 11 - "AWOS Skaler Yükleyici"
Cohesion: 0.24
Nodes (10): csv, build_awos_lookup(), parse_station_elevations(), datetime, Parses the stationlist CSV and returns a dict mapping wmoid -> elevation., Reads the 4.4GB AWOS observation CSV line by line. Only stores the data for…, Rounds a datetime object to the nearest 10-minute interval., round_to_nearest_10min() (+2 more)

### Community 12 - "Ablasyon Sonuçları ve Grafikleri"
Cohesion: 0.28
Nodes (9): ablation_results.json, Validation F1-Score Comparison between Architectures (15 epochs; Flat models stable ~0.83-0.84, Medium_GAP slow start 0.11, ResNet18 collapse to 0.18 at epoch 6), Flatten heads converge faster/more stable than GAP heads, ResNet18 validation instability (F1 collapse, loss spikes), Validation Loss Comparison between Architectures (ResNet18 loss unstable, spikes to ~38; ConvNets ~0.5-1), Classification Ablation Results (K6), MTGConvNet_Flat, MTGConvNet Architecture Diagrams (6 variants) (+1 more)

### Community 13 - "İstasyon Yama Görselleri"
Cohesion: 0.47
Nodes (6): IR10.5 (normalized) + radar patches with labels, 10 stations, 2026 Feb-May, Binary cloud label (Bulutlu 1 / Acik 0), Station patch visualization grid (VIS0.6/IR10.5/WV6.3/Radar, 10 airports, 2025-03-01), 33x33 station-centred patch (center pixel 16,16), Rainy-case station patches (high radar dBZ, all labeled cloudy), WV 6.3um panels uniform (~-70C) - suspected channel/scaling issue

### Community 14 - "Eğitim Döngüsü Yardımcıları"
Cohesion: 0.33
Nodes (6): buffered_shuffle_generator(), calculate_metrics(), ndarray, Calculates Accuracy, Precision, Recall, and F1-Score from predictions and…, Highly optimized generator that loads batches sequentially (fast HDF5 read),…, train_one_epoch()

## Ambiguous Edges - Review These
- `Rule_radar (manual trigger)` → `radar-skill`  [AMBIGUOUS]
  .agent/rules/Rule_radar.md · relation: conceptually_related_to

## Knowledge Gaps
- **12 isolated node(s):** `Rule_radar (manual trigger)`, `Skill: AWOS Pixel Match`, `Skill: Satellite Resample (Satpy)`, `MTG RS/AI engineer agent persona`, `AI Dataset & DataLoader Plan` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 135 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Rule_radar (manual trigger)` and `radar-skill`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `MTGCloudDataset` connect `Konsolide Etiket Dataset'i (v1)` to `Bölme, H5 ve Eğitim Betikleri`, `AWOS Piksel Eşleme ve Uydu Resample`, `Radar Kompozit Hattı ve Skill`, `DataLoader ve Augmentasyon`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `LatLonGrid` connect `Lat/Lon Grid ve Aux Kanallar` to `Zaman/SZA ve Dedup`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `MTGH5Dataset` connect `DataLoader ve Augmentasyon` to `Bölme, H5 ve Eğitim Betikleri`, `Lat/Lon Grid ve Aux Kanallar`, `Etiket, Kayıp ve Model Seçimi`, `Eğitim Döngüsü Yardımcıları`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `main()` (e.g. with `cos_solar_zenith()` and `LatLonGrid`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `LatLonGrid` (e.g. with `main()` and `load_dem()`) actually correct?**
  _`LatLonGrid` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `MTGH5Dataset` (e.g. with `get_dataloaders()` and `get_split_dataloaders()`) actually correct?**
  _`MTGH5Dataset` has 6 INFERRED edges - model-reasoned connections that need verification._