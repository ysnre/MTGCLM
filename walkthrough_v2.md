# MTGCLM v2 — Raster yardımcı kanallar, yeni etiket politikası, sızıntısız bölme

Bu sürüm `walkthrough.md`'deki skaler geç-füzyon hattının yerine geçer. Uydu yamaları (`patches`)
yeniden çıkarılmaz; H5'e yeni datasetler eklenir ve eğitim betikleri bunları kullanır.

## Yeni / değişen dosyalar

| Dosya | Durum | Ne yapar |
|---|---|---|
| `src/grid_utils.py` | yeni | Düzenli lat/lon grid, km projeksiyonu, cos(SZA), 10 dk pencere yardımcıları |
| `src/update_h5_aux_channels.py` | yeni | `aux[N,5,33,33]` float16: t2m, rh2m (1982 AWOS → IDW), elev, cos_sza, radar_cov; `station_y/x`, `aux_flags` |
| `src/update_h5_labels_v2.py` | yeni | Parametrik etiket (clear=0, cloudy≥5), ham `okta`, `obs_time`, çift örnek işaretleme (`is_dup`), `labels_prev` yedeği |
| `src/make_splits.py` | yeni | Hafta-blok zaman bölmesi + istasyon ayrımı → `train/val/test_time/test_station/test_both` (.npz + .json) |
| `src/export_compact_h5.py` | yeni | Yalnızca bölmedeki örnekler, global permütasyon, float16 → `mtg_train_v2.h5` (+ `<split>_compact.npz`) |
| `src/make_dem_npy.py` | yeni | DEM GeoTIFF → 800×1000 `.npy` (metre), ortalama ile yeniden örnekleme |
| `src/metrics.py` | yeni | Balanced Acc, MCC, F1(clear/cloudy), AUC, karışıklık matrisi, majority baseline |
| `src/h5_dataset.py` | değişti | `use_aux=True` → aux kanalları normalize edilip 17 kanala eklenir (`num_channels`) |
| `src/dataloader.py` | değişti | `get_split_dataloaders()` (bölme dosyasından), `get_dataloaders(use_aux=...)` |
| `src/train.py`, `src/train_unet.py` | değişti | `--split --aux --seed --epochs --limit_batches --results`; kanal sayısı veri setinden; model seçimi **val balanced accuracy**; test kümeleri sonuç JSON'una yazılır; `torch.load(weights_only=False)` |
| `src/config.py` | değişti | `MTGCLM_H5` ortam değişkeni H5 yolunu geçersiz kılar |

Eski dosyaların yedeği: `src/_backup_20260916/`.

## Çalıştırma sırası (yerel, Colab'a çıkmadan önce)

```bat
cd e:\belgeler\MTGCLM

:: 0) DEM (bir kez): herhangi bir DEM GeoTIFF'ini grid'e örnekle  (pip install rasterio)
python src\make_dem_npy.py --dem_tif data/dem/<dem>.tif --out data/dem_800x1000.npy

:: 1) Yardımcı kanallar (~20-30 dk; 4.4 GB CSV bir kez taranır; kesilirse aynı komut kaldığı yerden sürer)
python src\update_h5_aux_channels.py --h5 F:/radar/mtg_patch_dataset.h5 ^
    --awos_dir data/raw_awos_observation --station_csv data/raw_observation/station_list.csv ^
    --sample_nc data/raw_mtg/turkiye_kesit_20260525000817_0001.nc ^
    --radar_csv data/raw_radar/radar.csv --radar_dir F:/radar/raw_radar --dem data/dem_800x1000.npy

:: 2) Etiketler (önce --dry_run ile dağılıma bakın; okta 9 kararı RH'ye göre -> aux'tan sonra çalışmalı)
python src\update_h5_labels_v2.py --h5 F:/radar/mtg_patch_dataset.h5 --awos_aligned_dir data/processed/awos_aligned --dry_run
python src\update_h5_labels_v2.py --h5 F:/radar/mtg_patch_dataset.h5 --awos_aligned_dir data/processed/awos_aligned

:: 3) Bölme
python src\make_splits.py --h5 F:/radar/mtg_patch_dataset.h5 --out splits/split_v1 --seed 42 --holdout_stations 30

:: 4) Colab için kompakt, ön-karıştırılmış H5 (~19 GB; 395k+partial örnek, float16)
python src\export_compact_h5.py --src F:/radar/mtg_patch_dataset.h5 --split splits/split_v1.npz --out F:/radar/mtg_train_v2.h5 --seed 42

:: 5) Yerel dry-run (kompakt dosyayla)
set MTGCLM_H5=F:/radar/mtg_train_v2.h5
python src\train.py --model mtg_flat --split splits/split_v1_compact.npz --aux --epochs 1 --limit_batches 30
python src\train_unet.py --model unet --split splits/split_v1_compact.npz --aux --epochs 1 --limit_batches 20
```

`make_splits` ayrıca `test_partial` (1–4 okta, etiketsiz) alt kümesini yazar; eğitim betikleri bu kümede okta başına
ortalama bulut olasılığını raporlar (beklenti: okta ile monoton artış, 0 ve ≥5 okta arasında kalma).

Sıra önemli: **aux → labels → splits** (etiket betiği sis kararı için `rh2m_center`'ı, bölme ise geçerli etiket ve `is_dup` bilgisini kullanır).
Etiket politikasını değiştirirseniz (`--cloudy_min 1` gibi) bölmeyi yeniden üretin.

## Colab

```bash
# mtg_train_v2.h5 /content'e kopyalandıktan sonra; splits/split_v1_compact.npz de repo ile gelir
export MTGCLM_H5=/content/mtg_train_v2.h5
# aşağıdaki komutlarda splits/split_v1.npz yerine splits/split_v1_compact.npz kullanın
# Modalite ablasyonu (aynı split, 3 seed):
python src/train.py --model mtg_flat --split splits/split_v1.npz --seed 1 --results artifacts/sat_radar_s1.json
python src/train.py --model mtg_flat --split splits/split_v1.npz --seed 1 --aux --results artifacts/sat_radar_aux_s1.json
python src/train.py --model mtg_flat --split splits/split_v1.npz --seed 1 --aux --aux_channels cos_sza,elev,radar_cov --results artifacts/no_awos_s1.json
python src/train_unet.py --model unet --split splits/split_v1.npz --seed 1 --aux --results artifacts/unet_aux_s1.json
```

Uydu-only (radar kanalı hariç) ablasyonu için `MTGH5Dataset`'e kanal seçme parametresi eklemek gerekir
(henüz yok; istenirse `base_channels` alt kümesi olarak eklenebilir).

## Notlar / bilinçli kararlar

- `aux` ham fiziksel birimlerde (°C, %, m) saklanır; normalizasyon `aux_means/aux_stds` attrs ile veri setinde yapılır.
  `cos_sza` ve `radar_cov` ham bırakılır.
- `radar_cov` 0/1 maske değil, **1 − d/r** yakınlığı (radar üstünde 1, menzil kenarında 0); `--radar_dir` verilirse
  o anda PNG'si olan radarlar sayılır.
- AWOS zaman eşlemesi artık **floor** (uyduyu kapsayan 10 dk penceresi); eski davranış `--time_match nearest`.
- Sıcaklıkta yükseklik düzeltmesi `--lapse_rate 6.5` K/km (0 = kapalı). DEM yoksa yükseklik alanı
  istasyon yüksekliklerinden IDW ile üretilir — kaba; makale için gerçek DEM verin.
- Etiket: `--okta9 rh` varsayılan: N=9 (gök görünmüyor) + istasyonda RH ≥ 90 % → sis/alçak stratus → cloudy; RH düşükse ignore. Eşik `--fog_rh`.
- `is_dup`: aynı (istasyon, gözlem zamanı) gözleminin uyduya en yakın olmayan kopyası → etiket −1.
  ±9 dk tolerans nedeniyle geçerli örnek sayısı yaklaşık yarıya iner; bu gerçek N'dir.
- Model seçimi artık val **balanced accuracy** ile (F1(cloudy) çoğunluk sınıfını ödüllendiriyordu).
  Sonuç JSON'unda `best_f1` alanı geriye uyumluluk için korunur ama balanced accuracy içerir.
- Kompakt H5 `pre_shuffled=1` attr'ı taşır; `train.py` bunu görünce `buffered_shuffle_generator`'ı atlar (dosya zaten rastgele sıralı).
- Test kümeleri yalnızca eğitim bitince, en iyi val modeliyle bir kez değerlendirilir ve JSON'a yazılır.
