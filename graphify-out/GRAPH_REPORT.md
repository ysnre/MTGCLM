# Graph Report - MTGCLM  (2026-09-17)

## Corpus Check
- 60 files · ~86,787 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: (none) 2, .npz 2, .ipynb 1)

## Summary
- 363 nodes · 665 edges · 15 communities (13 shown, 2 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.86)
- Token cost: 70,601 input · 0 output

## Community Hubs (Navigation)
- AWOS Piksel Eşleme ve Uydu Resample
- v2 Veri Hattı, Bölme ve Eğitim
- Lat/Lon Grid ve Aux Kanallar
- DataLoader ve Ablasyon
- Etiket, Kayıp ve Model Seçimi
- Aux Kanallar ve Modeller
- Radar Kompozit Hattı ve Skill
- Büyük Ölçek Radar İşleme
- Konsolide Etiket Dataset'i (v1)
- Zaman/SZA Yardımcıları
- Patch Dataset Oluşturma
- AWOS Skaler Yükleyici
- İstasyon Yama Görselleri

## God Nodes (most connected - your core abstractions)
1. `main()` - 17 edges
2. `LatLonGrid` - 16 edges
3. `MTGH5Dataset` - 14 edges
4. `MTGCloudDataset` - 13 edges
5. `get_split_dataloaders()` - 11 edges
6. `run_experiment()` - 11 edges
7. `get_dataloaders()` - 9 edges
8. `run_experiment()` - 9 edges
9. `AugmentedDataset` - 8 edges
10. `main()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Raster Aux Channels (K1, 22 channels)` --semantically_similar_to--> `CloudUNet_Fusion`  [INFERRED] [semantically similar]
  CLAUDE.md → src/model.py
- `get_split_dataloaders()` --conceptually_related_to--> `mtg_train_v2.h5 (compact, pre_shuffled)`  [AMBIGUOUS]
  src/dataloader.py → walkthrough_v2.md
- `get_split_dataloaders()` --shares_data_with--> `splits/split_v1_compact.npz`  [EXTRACTED]
  src/dataloader.py → walkthrough_v2.md
- `MaskedBCELoss` --conceptually_related_to--> `Label Policy clear=0 / cloudy>=5 okta (K2)`  [INFERRED]
  src/loss.py → CLAUDE.md
- `MTGCLM_MONGO_URI env variable` --shares_data_with--> `init_worker()`  [INFERRED]
  .agent/skills/radar-skill/SKILL.md → src/process_radar_pipeline_large.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **AWOS station pixel matching pipeline** — src_process_awos_pipeline, src_awos_matcher, src_observation_loader [EXTRACTED 1.00]
- **Radar PNG to MTG grid alignment pipeline** — src_process_radar_pipeline, src_radar_loader, src_resampler, src_satellite_grid [EXTRACTED 1.00]
- **Leakage-safe labeling and evaluation decisions** — claude_label_policy, claude_dedup_policy, claude_leakage_free_split, claude_balanced_accuracy_selection [INFERRED 0.85]
- **v1 scalar late fusion models** — src_model_mtgconvnet_latefusion, src_model_resnet18_latefusion, src_model_cloudunet_fusion, src_update_h5_scalars [EXTRACTED 1.00]
- **v2 data prep pipeline aux->labels->splits->export** — src_make_dem_npy, src_update_h5_aux_channels, src_update_h5_labels_v2, src_make_splits, src_export_compact_h5 [EXTRACTED 1.00]
- **Training stack on compact H5** — src_train, src_train_unet, src_dataloader_get_split_dataloaders, src_h5_dataset_mtgh5dataset, src_config [INFERRED 0.85]

## Communities (15 total, 2 thin omitted)

### Community 0 - "AWOS Piksel Eşleme ve Uydu Resample"
Cohesion: 0.07
Nodes (38): Skill: AWOS Pixel Match, Skill: Satellite Resample (Satpy), AWOS Pixel Match Implementation Plan, +/-9 min AWOS time tolerance, collections, glob, hdf5plugin, multiprocessing (+30 more)

### Community 1 - "v2 Veri Hattı, Bölme ve Eğitim"
Cohesion: 0.06
Nodes (37): argparse, Colab Training Workflow, Duplicate Sample Dedup (K3), Week-block + Station Holdout Split (K4), MTGCLM Project (Cloud Mask over Turkey), datetime, h5py, json (+29 more)

### Community 2 - "Lat/Lon Grid ve Aux Kanallar"
Cohesion: 0.06
Nodes (35): bisect, scipy_spatial, LatLonGrid, Düzenli enlem/boylam grid'i. Satır = enlem (y), sütun = boylam (x)., turkiye_kesit_*.nc dosyalarındaki grid (800 x 1000)., (lat, lon) -> (y_km, x_km) referans noktasına göre., En yakın (y, x) piksel indeksleri (grid dışına taşabilir; clip edilmez)., (yc, xc) merkezli (2*half+1)^2 pencerenin lat/lon 2D dizileri (grid dışı lineer… (+27 more)

### Community 3 - "DataLoader ve Ablasyon"
Cohesion: 0.06
Nodes (37): Classification Ablation Results (K6), AugmentedDataset, get_dataloaders(), get_split_dataloaders(), get_train_val_split(), Tensor, RandomSpatialTransforms, Wrapper around a PyTorch Dataset (or Subset) to apply online data augmentation… (+29 more)

### Community 4 - "Etiket, Kayıp ve Model Seçimi"
Cohesion: 0.08
Nodes (28): Val Balanced Accuracy Model Selection (K5), Label Policy clear=0 / cloudy>=5 okta (K2), MaskedBCELoss, Tensor, Args: preds: [Batch, 2, H, W] (raw logits from U-Net) targets: [Batch, H, W]…, Custom Loss for U-Net Cloud Masking. Computes Binary Cross Entropy strictly for…, compute_metrics(), format_metrics() (+20 more)

### Community 5 - "Aux Kanallar ve Modeller"
Cohesion: 0.09
Nodes (16): Raster Aux Channels (K1, 22 channels), Full-map 800x1000 Cloud Mask, v1 Late Fusion Implementation Plan, CloudUNet_Fusion, MTGConvNet, MTGConvNet_LateFusion, MTGResNet18, MTGUNet (+8 more)

### Community 6 - "Radar Kompozit Hattı ve Skill"
Cohesion: 0.11
Nodes (23): Rule_radar (manual trigger), Radar MongoDB product metadata (PPI, ll bbox), radar-skill, MTG RS/AI engineer agent persona, Walkthrough (dataset copy), Maximum Reflectivity compositing, Radar-to-MTG grid alignment walkthrough, pil (+15 more)

### Community 7 - "Büyük Ölçek Radar İşleme"
Cohesion: 0.13
Nodes (19): MTGCLM_MONGO_URI env variable, pymongo, get_satellite_grid_local(), init_worker(), main(), parse_mtg_time(), process_single_satellite(), datetime (+11 more)

### Community 8 - "Konsolide Etiket Dataset'i (v1)"
Cohesion: 0.16
Nodes (9): Consolidated Ground Truth (AWOS+Radar consensus labels), AI Dataset & DataLoader Plan, MTGCloudDataset, Dataset, ndarray, Loads and normalizes satellite and radar features. Returns a normalized (17,…, PyTorch Dataset for MTG Cloud Masking. Args: mtg_dir (str): Directory…, Applies the Consolidated Ground Truth logic to determine clear/cloudy label. -… (+1 more)

### Community 9 - "Zaman/SZA Yardımcıları"
Cohesion: 0.20
Nodes (13): cos_solar_zenith(), floor_to_10min(), _is_leap(), nearest_10min(), padded_window(), parse_sat_timestamp(), datetime, ndarray (+5 more)

### Community 10 - "Patch Dataset Oluşturma"
Cohesion: 0.22
Nodes (12): compute_normalization_stats_fast(), get_channel_names(), get_consolidated_label(), load_features_fast(), main(), parse_mtg_time(), ndarray, Parses timestamp from MTG file name. (+4 more)

### Community 11 - "AWOS Skaler Yükleyici"
Cohesion: 0.24
Nodes (10): csv, build_awos_lookup(), parse_station_elevations(), datetime, Parses the stationlist CSV and returns a dict mapping wmoid -> elevation., Reads the 4.4GB AWOS observation CSV line by line. Only stores the data for…, Rounds a datetime object to the nearest 10-minute interval., round_to_nearest_10min() (+2 more)

### Community 12 - "İstasyon Yama Görselleri"
Cohesion: 0.47
Nodes (6): IR10.5 (normalized) + radar patches with labels, 10 stations, 2026 Feb-May, Binary cloud label (Bulutlu 1 / Acik 0), Station patch visualization grid (VIS0.6/IR10.5/WV6.3/Radar, 10 airports, 2025-03-01), 33x33 station-centred patch (center pixel 16,16), Rainy-case station patches (high radar dBZ, all labeled cloudy), WV 6.3um panels uniform (~-70C) - suspected channel/scaling issue

## Ambiguous Edges - Review These
- `get_split_dataloaders()` → `mtg_train_v2.h5 (compact, pre_shuffled)`  [AMBIGUOUS]
  walkthrough_v2.md · relation: conceptually_related_to
- `Rule_radar (manual trigger)` → `radar-skill`  [AMBIGUOUS]
  .agent/rules/Rule_radar.md · relation: conceptually_related_to

## Knowledge Gaps
- **17 isolated node(s):** `Rule_radar (manual trigger)`, `Skill: AWOS Pixel Match`, `Skill: Satellite Resample (Satpy)`, `Radar MongoDB product metadata (PPI, ll bbox)`, `MTG RS/AI engineer agent persona` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 140 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `get_split_dataloaders()` and `mtg_train_v2.h5 (compact, pre_shuffled)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Rule_radar (manual trigger)` and `radar-skill`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `MTGCloudDataset` connect `Konsolide Etiket Dataset'i (v1)` to `AWOS Piksel Eşleme ve Uydu Resample`, `DataLoader ve Ablasyon`, `Radar Kompozit Hattı ve Skill`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `LatLonGrid` connect `Lat/Lon Grid ve Aux Kanallar` to `Zaman/SZA Yardımcıları`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `MTGH5Dataset` connect `DataLoader ve Ablasyon` to `AWOS Piksel Eşleme ve Uydu Resample`, `Etiket, Kayıp ve Model Seçimi`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `main()` (e.g. with `cos_solar_zenith()` and `LatLonGrid`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `LatLonGrid` (e.g. with `main()` and `load_dem()`) actually correct?**
  _`LatLonGrid` has 5 INFERRED edges - model-reasoned connections that need verification._