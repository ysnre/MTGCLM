# Graph Report - MTGCLM  (2026-09-17)

## Corpus Check
- 68 files · ~100,792 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: .npz 2, (none) 1, .ipynb 1)

## Summary
- 399 nodes · 756 edges · 19 communities (17 shown, 2 thin omitted)
- Extraction: 87% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 93 edges (avg confidence: 0.87)
- Token cost: 288,446 input · 0 output

## Community Hubs (Navigation)
- Eski Eğitim Döngüsü (backup)
- Lat/Lon Grid Yardımcıları
- AWOS Piksel Eşleme ve Uydu Resample
- v2 Proje Mimarisi ve U-Net
- Metrikler ve Model Seçimi
- Bölme ve H5 Veri Hazırlama
- Zaman/SZA ve Dedup
- Radar Kompozit Hattı
- v2 DataLoader ve Augmentasyon
- Büyük Ölçek Radar İşleme
- Eski DataLoader (backup)
- Konsolide Etiket Dataset'i
- Yardımcı Modüller
- AWOS Skaler Yükleyici
- Ablasyon Sonuçları ve Grafikleri
- Etiket Politikası ve Maskeli Kayıp
- İstasyon Yama Görselleri

## God Nodes (most connected - your core abstractions)
1. `MTGH5Dataset` - 18 edges
2. `main()` - 17 edges
3. `LatLonGrid` - 16 edges
4. `MTGCloudDataset` - 14 edges
5. `get_dataloaders()` - 11 edges
6. `MTGH5Dataset` - 11 edges
7. `run_experiment()` - 11 edges
8. `get_dataloaders()` - 10 edges
9. `run_experiment()` - 9 edges
10. `run_experiment()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Raster Aux Channels (K1, 22 channels)` --semantically_similar_to--> `CloudUNet_Fusion`  [INFERRED] [semantically similar]
  CLAUDE.md → src/model.py
- `MaskedBCELoss` --conceptually_related_to--> `Label Policy clear=0 / cloudy>=5 okta (K2)`  [INFERRED]
  src/loss.py → CLAUDE.md
- `ResNet18 validation instability (F1 collapse, loss spikes)` --conceptually_related_to--> `ResNet18_LateFusion`  [INFERRED]
  artifacts/ablation_study_f1.png → src/model.py
- `Consolidated Ground Truth (AWOS+Radar consensus labels)` --rationale_for--> `MTGCloudDataset`  [EXTRACTED]
  .agent/steps/implementation_plan_dataset.md → src/dataset.py
- `AI Dataset & DataLoader Plan` --references--> `MTGCloudDataset`  [EXTRACTED]
  .agent/steps/implementation_plan_dataset.md → src/dataset.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Radar PNG to MTG grid alignment pipeline** — src_process_radar_pipeline, src_radar_loader, src_resampler, src_satellite_grid [EXTRACTED 1.00]
- **AWOS station pixel matching pipeline** — src_process_awos_pipeline, src_awos_matcher, src_observation_loader [EXTRACTED 1.00]
- **v2 pipeline aux -> labels -> splits -> export** — src_update_h5_aux_channels, src_update_h5_labels_v2, src_make_splits, src_export_compact_h5 [EXTRACTED 1.00]
- **v1 scalar late fusion models** — src_model_mtgconvnet_latefusion, src_model_resnet18_latefusion, src_model_cloudunet_fusion, src_update_h5_scalars [EXTRACTED 1.00]
- **Leakage-safe labeling and evaluation decisions** — claude_label_policy, claude_dedup_policy, claude_leakage_free_split, claude_balanced_accuracy_selection [INFERRED 0.85]
- **Architecture ablation validation curves** — artifacts_ablation_study_f1, artifacts_ablation_study_loss, artifacts_ablation_results [INFERRED 0.85]

## Communities (19 total, 2 thin omitted)

### Community 0 - "Eski Eğitim Döngüsü (backup)"
Cohesion: 0.10
Nodes (34): config, dataloader, matplotlib, matplotlib_pyplot, os, buffered_shuffle_generator(), calculate_metrics(), main() (+26 more)

### Community 1 - "Lat/Lon Grid Yardımcıları"
Cohesion: 0.07
Nodes (31): scipy_spatial, LatLonGrid, Düzenli enlem/boylam grid'i. Satır = enlem (y), sütun = boylam (x)., turkiye_kesit_*.nc dosyalarındaki grid (800 x 1000)., (lat, lon) -> (y_km, x_km) referans noktasına göre., En yakın (y, x) piksel indeksleri (grid dışına taşabilir; clip edilmez)., (yc, xc) merkezli (2*half+1)^2 pencerenin lat/lon 2D dizileri (grid dışı lineer…, main() (+23 more)

### Community 2 - "AWOS Piksel Eşleme ve Uydu Resample"
Cohesion: 0.09
Nodes (30): Skill: AWOS Pixel Match, Skill: Satellite Resample (Satpy), AWOS Pixel Match Implementation Plan, +/-9 min AWOS time tolerance, collections, glob, hdf5plugin, netcdf4 (+22 more)

### Community 3 - "v2 Proje Mimarisi ve U-Net"
Cohesion: 0.08
Nodes (18): Raster Aux Channels (K1, 22 channels), Colab Training Workflow, Full-map 800x1000 Cloud Mask, MTGCLM Project (Cloud Mask over Turkey), v1 Late Fusion Implementation Plan, Python Dependencies (torch, h5py, xarray, pyresample...), CloudUNet_Fusion, MTGConvNet (+10 more)

### Community 4 - "Metrikler ve Model Seçimi"
Cohesion: 0.08
Nodes (28): Val Balanced Accuracy Model Selection (K5), compute_metrics(), format_metrics(), ndarray, metrics.py — ikili bulut maskesi için genişletilmiş metrikler (inceleme A1).…, Mann-Whitney U ile ROC-AUC (bağlı skorlar ortalama sıra ile)., _roc_auc(), buffered_shuffle_generator() (+20 more)

### Community 5 - "Bölme ve H5 Veri Hazırlama"
Cohesion: 0.12
Nodes (16): argparse, Week-block + Station Holdout Split (K4), datetime, h5_dataset, h5py, json, numpy, random (+8 more)

### Community 6 - "Zaman/SZA ve Dedup"
Cohesion: 0.12
Nodes (23): Duplicate Sample Dedup (K3), cos_solar_zenith(), floor_to_10min(), _is_leap(), nearest_10min(), padded_window(), parse_sat_timestamp(), datetime (+15 more)

### Community 7 - "Radar Kompozit Hattı"
Cohesion: 0.11
Nodes (23): Rule_radar (manual trigger), Radar MongoDB product metadata (PPI, ll bbox), radar-skill, MTG RS/AI engineer agent persona, Walkthrough (dataset copy), Maximum Reflectivity compositing, Radar-to-MTG grid alignment walkthrough, pil (+15 more)

### Community 8 - "v2 DataLoader ve Augmentasyon"
Cohesion: 0.10
Nodes (17): AugmentedDataset, get_dataloaders(), get_split_dataloaders(), get_train_val_split(), Tensor, TorchDataset, RandomSpatialTransforms, Wrapper around a PyTorch Dataset (or Subset) to apply online data augmentation… (+9 more)

### Community 9 - "Büyük Ölçek Radar İşleme"
Cohesion: 0.12
Nodes (22): bisect, multiprocessing, pymongo, pyresample_geometry, pyresample_kd_tree, get_satellite_grid_local(), init_worker(), main() (+14 more)

### Community 10 - "Eski DataLoader (backup)"
Cohesion: 0.11
Nodes (13): AugmentedDataset, get_dataloaders(), get_train_val_split(), Tensor, TorchDataset, RandomSpatialTransforms, Wrapper around a PyTorch Dataset (or Subset) to apply online data augmentation…, Creates PyTorch DataLoaders for training and validation. Returns: tuple:… (+5 more)

### Community 11 - "Konsolide Etiket Dataset'i"
Cohesion: 0.16
Nodes (9): Consolidated Ground Truth (AWOS+Radar consensus labels), AI Dataset & DataLoader Plan, MTGCloudDataset, Dataset, ndarray, Loads and normalizes satellite and radar features. Returns a normalized (17,…, PyTorch Dataset for MTG Cloud Masking. Args: mtg_dir (str): Directory…, Applies the Consolidated Ground Truth logic to determine clear/cloudy label. -… (+1 more)

### Community 12 - "Yardımcı Modüller"
Cohesion: 0.22
Nodes (12): compute_normalization_stats_fast(), get_channel_names(), get_consolidated_label(), load_features_fast(), main(), parse_mtg_time(), ndarray, Parses timestamp from MTG file name. (+4 more)

### Community 13 - "AWOS Skaler Yükleyici"
Cohesion: 0.24
Nodes (10): csv, build_awos_lookup(), parse_station_elevations(), datetime, Parses the stationlist CSV and returns a dict mapping wmoid -> elevation., Reads the 4.4GB AWOS observation CSV line by line. Only stores the data for…, Rounds a datetime object to the nearest 10-minute interval., round_to_nearest_10min() (+2 more)

### Community 14 - "Ablasyon Sonuçları ve Grafikleri"
Cohesion: 0.24
Nodes (10): ablation_results.json, Validation F1-Score Comparison between Architectures (15 epochs; Flat models stable ~0.83-0.84, Medium_GAP slow start 0.11, ResNet18 collapse to 0.18 at epoch 6), Flatten heads converge faster/more stable than GAP heads, ResNet18 validation instability (F1 collapse, loss spikes), Validation Loss Comparison between Architectures (ResNet18 loss unstable, spikes to ~38; ConvNets ~0.5-1), test_plot (trivial matplotlib test line, no project data), Classification Ablation Results (K6), MTGConvNet_Flat (+2 more)

### Community 15 - "Etiket Politikası ve Maskeli Kayıp"
Cohesion: 0.33
Nodes (5): Label Policy clear=0 / cloudy>=5 okta (K2), MaskedBCELoss, Tensor, Args: preds: [Batch, 2, H, W] (raw logits from U-Net) targets: [Batch, H, W]…, Custom Loss for U-Net Cloud Masking. Computes Binary Cross Entropy strictly for…

### Community 16 - "İstasyon Yama Görselleri"
Cohesion: 0.47
Nodes (6): IR10.5 (normalized) + radar patches with labels, 10 stations, 2026 Feb-May, Binary cloud label (Bulutlu 1 / Acik 0), Station patch visualization grid (VIS0.6/IR10.5/WV6.3/Radar, 10 airports, 2025-03-01), 33x33 station-centred patch (center pixel 16,16), Rainy-case station patches (high radar dBZ, all labeled cloudy), WV 6.3um panels uniform (~-70C) - suspected channel/scaling issue

## Ambiguous Edges - Review These
- `Rule_radar (manual trigger)` → `radar-skill`  [AMBIGUOUS]
  .agent/rules/Rule_radar.md · relation: conceptually_related_to
- `Validation F1-Score Comparison between Architectures (15 epochs; Flat models stable ~0.83-0.84, Medium_GAP slow start 0.11, ResNet18 collapse to 0.18 at epoch 6)` → `test_plot (trivial matplotlib test line, no project data)`  [AMBIGUOUS]
  artifacts/test_plot.png · relation: conceptually_related_to

## Knowledge Gaps
- **13 isolated node(s):** `Rule_radar (manual trigger)`, `Skill: AWOS Pixel Match`, `Skill: Satellite Resample (Satpy)`, `Radar MongoDB product metadata (PPI, ll bbox)`, `MTG RS/AI engineer agent persona` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 156 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Rule_radar (manual trigger)` and `radar-skill`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Validation F1-Score Comparison between Architectures (15 epochs; Flat models stable ~0.83-0.84, Medium_GAP slow start 0.11, ResNet18 collapse to 0.18 at epoch 6)` and `test_plot (trivial matplotlib test line, no project data)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `MTGCloudDataset` connect `Konsolide Etiket Dataset'i` to `AWOS Piksel Eşleme ve Uydu Resample`, `Bölme ve H5 Veri Hazırlama`, `Radar Kompozit Hattı`, `v2 DataLoader ve Augmentasyon`, `Eski DataLoader (backup)`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `LatLonGrid` connect `Lat/Lon Grid Yardımcıları` to `Zaman/SZA ve Dedup`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `MTGH5Dataset` connect `v2 DataLoader ve Augmentasyon` to `Eski Eğitim Döngüsü (backup)`, `Lat/Lon Grid Yardımcıları`, `Bölme ve H5 Veri Hazırlama`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `MTGH5Dataset` (e.g. with `get_train_val_split()` and `train_one_epoch()`) actually correct?**
  _`MTGH5Dataset` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `main()` (e.g. with `cos_solar_zenith()` and `LatLonGrid`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._