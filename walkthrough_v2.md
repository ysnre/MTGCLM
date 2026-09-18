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
| `src/make_cv_splits.py` | yeni | Bloklu K-katlı CV bölmeleri (`cv5_fold0..4.npz`) |
| `src/aggregate_results.py` | yeni | Koşu JSON'larını toplar → ortalama ± s.s. tablo (Markdown/CSV/LaTeX) + grafikler |
| `src/baselines.py` | yeni | Karşılaştırma modelleri: majority, kanal eşiği, çift eşik, logreg, RF, HGB, kNN |
| `src/run_all.py` | yeni | Çökmeye dayanıklı koşu yöneticisi: matrisi tanımlar, tamamlananları atlar, log/manifest tutar |
| `src/make_public_sample.py` | yeni | Paylaşılabilir küçük örneklem H5 + bölme (GitHub/Zenodo) |
| `README.md`, `.gitignore`, `LICENSE`, `CITATION.cff`, `requirements.txt` | yeni | Depo iskeleti + veri bulunabilirliği beyanı |
| `src/metrics.py` | yeni | Balanced Acc, MCC, F1(clear/cloudy), AUC, karışıklık matrisi, majority baseline |
| `src/h5_dataset.py` | değişti | `use_aux=True` → aux kanalları normalize edilip 17 kanala eklenir (`num_channels`) |
| `src/dataloader.py` | değişti | `get_split_dataloaders()` (bölme dosyasından), `get_dataloaders(use_aux=...)` |
| `src/train.py`, `src/train_unet.py` | değişti | `--h5 --split --aux --seed --epochs --limit_batches --results`; mimari kaydı Shallow/Medium/Deep × GAP/Flatten (6 varyant, `--model arch`); kanal sayısı veri setinden; model seçimi **val balanced accuracy**; test kümeleri sonuç JSON'una yazılır; `torch.load(weights_only=False)` |
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

:: 5) Yerel dry-run (kompakt dosyayla) -- H5 yolunu HER ZAMAN --h5 ile verin
python src\train.py --h5 F:/radar/mtg_train_v2.h5 --model mtg_flat --split splits/split_v1_compact.npz --aux --epochs 1 --limit_batches 30
python src\train_unet.py --h5 F:/radar/mtg_train_v2.h5 --model unet --split splits/split_v1_compact.npz --aux --epochs 1 --limit_batches 20
```

> `--h5` verilmezse `MTGCLM_H5` ortam değişkeni, o da yoksa `config.H5_PATH` kullanılır. Kompakt H5 için
> üretilen `*_compact.npz` bölmesini kaynak H5'e (ya da tersini) uygulamak sessiz bir hata olurdu;
> `get_split_dataloaders` artık indeks aralığını ve etiket geçerliliğini kontrol edip uyuşmazlıkta durur
> ("Bölme doğrulandı: N satırlık H5 ile uyumlu." satırını görmelisiniz).

`make_splits` ayrıca `test_partial` (1–4 okta, etiketsiz) alt kümesini yazar; eğitim betikleri bu kümede okta başına
ortalama bulut olasılığını raporlar (beklenti: okta ile monoton artış, 0 ve ≥5 okta arasında kalma).

Sıra önemli: **aux → labels → splits** (etiket betiği sis kararı için `rh2m_center`'ı, bölme ise geçerli etiket ve `is_dup` bilgisini kullanır).
Etiket politikasını değiştirirseniz (`--cloudy_min 1` gibi) bölmeyi yeniden üretin.

## Colab

```bash
# mtg_train_v2.h5 /content'e kopyalandıktan sonra; splits/split_v1_compact.npz de repo ile gelir
# Modalite ablasyonu (aynı split, 3 seed):
H5=/content/mtg_train_v2.h5; SP=splits/split_v1_compact.npz
python src/train.py --h5 $H5 --split $SP --model mtg_flat --seed 1 --results artifacts/sat_radar_s1.json
python src/train.py --h5 $H5 --split $SP --model mtg_flat --seed 1 --aux --results artifacts/sat_radar_aux_s1.json
python src/train.py --h5 $H5 --split $SP --model mtg_flat --seed 1 --aux --aux_channels cos_sza,elev,radar_cov --results artifacts/no_awos_s1.json
python src/train_unet.py --h5 $H5 --split $SP --model unet --seed 1 --aux --results artifacts/unet_aux_s1.json
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
- `train.py --h5` / `train_unet.py --h5` H5 yolunu açıkça belirler; bölme–H5 uyuşmazlığı başlangıçta yakalanır.
- Test kümeleri yalnızca eğitim bitince, en iyi val modeliyle bir kez değerlendirilir ve JSON'a yazılır.


## Çapraz doğrulama, karşılaştırma ve raporlama (danışman talepleri)

```bash
H5=F:/radar/mtg_train_v2.h5; SP=splits/split_v1_compact.npz

# 1) Bloklu 5-katlı CV (tek bölme yerine; katlar 7 günlük bloklardan oluşur)
python src/make_cv_splits.py --h5 $H5 --out splits/cv5 --folds 5 --block_days 7 --seed 42
for k in 0 1 2 3 4; do
  python src/train.py --h5 $H5 --split splits/cv5_fold$k.npz --model mtg_flat --aux --seed 1 \
      --results artifacts/CV_full_f${k}_s1.json
done

# 2) Klasik karşılaştırma modelleri (aynı bölme, aynı metrikler)
python src/baselines.py --h5 $H5 --split $SP --aux \
    --models majority,threshold,threshold2,logreg,rf,hgb --results artifacts/Z_baselines_s1.json

# 3) Toplu rapor: ortalama ± standart sapma, Markdown + CSV + LaTeX + grafikler
python src/aggregate_results.py --dir artifacts --out artifacts/summary --plot
python src/aggregate_results.py --dir artifacts --pattern "CV_*.json" --out artifacts/cv_summary

# 4) Paylaşılabilir örneklem (GitHub release / Zenodo)
python src/make_public_sample.py --src $H5 --split $SP --out data/mtg_sample.h5 --n 2000
```

Raporlanan metrikler: balanced accuracy, MCC, sınıf bazında precision / recall / F1, accuracy,
ROC-AUC, karışıklık matrisi ve çoğunluk sınıfı referansı. `aggregate_results.py` dosya adındaki
`_f<kat>_s<seed>` kalıbını okuyup kat ve seed koşularını tek satırda ortalar.

Baseline'ların eşikleri **yalnızca train** üzerinde optimize edilir; test kümesine bakılmaz.
Baseline sonuçları CNN koşularıyla aynı JSON şemasını kullandığı için tek tabloda birleşir.


## Mimari ablasyonu (v1 ile aynı 6 varyant)

`--model arch` altı varyantı sırayla koşar; parametre sayıları v1 koşularıyla birebir aynıdır
(aux kullanıldığında her varyant +1.440 parametre alır: 5 ek giriş kanalı × 32 filtre × 3×3).

| Deney | conv_blocks | Havuzlama | Parametre (17 kanal) |
|---|---|---|---|
| Shallow_GAP | [32, 64] | GAP | 27.906 |
| Shallow_Flat | [32, 64] | Flatten | 285.954 |
| Medium_GAP | [32, 64, 128] | GAP | 106.114 |
| Medium_Flat | [32, 64, 128] | Flatten | 228.994 |
| Deep_GAP | [32, 64, 128, 256] | GAP | 409.986 |
| Deep_Flat | [32, 64, 128, 256] | Flatten | 459.138 |

```bash
python src/train.py --h5 $H5 --split $SP --model arch --aux --seed 1 --results artifacts/ARCH_s1.json
python src/train.py --h5 $H5 --split $SP --model resnet18 --aux --seed 1 --results artifacts/RESNET_s1.json
```

Tek varyant için `--model medium_flat` gibi kısa adlar ya da doğrudan deney adı (`--model Medium_GAP`)
kullanılabilir. `--model mtg_flat` geriye uyumluluk için Deep_Flat'i çalıştırır.

> Dikkat: `train.py`, sonuç JSON'unda **adı zaten bulunan** deneyi atlar (kesintiden sonra devam
> edebilmek için). Bir deneyi yeniden koşmak istediğinizde ya yeni bir `--results` dosyası verin
> ya da eski JSON'u silin.


## Colab çökmelerine karşı kalıcı kurulum

Colab oturumu çökünce `/content` altındaki her şey silinir. Bu yüzden **checkpoint, en iyi model
ve sonuç JSON'ları Drive'a yazılır**; `--outdir` ile verilen dizin kalıcı olmalıdır.

Üç seviyede kaldığı yerden devam:

| Seviye | Nasıl | Dosya |
|---|---|---|
| Koşu | tamamlanan koşu işaretlenir, tekrar çalıştırılmaz | `<outdir>/status/<koşu>.done` |
| Deney | `--model arch` içindeki 6 varyanttan bitenler atlanır | sonuç JSON'undaki deney adları |
| Epoch | her epoch sonunda model+optimizer+scheduler+history yazılır | `<outdir>/checkpoints/checkpoint_<ad>.pth` |

Tüm yazmalar **atomiktir** (`.tmp` → `os.replace`): yazma anında çökme dosyayı bozmaz. Bozuk bir
checkpoint yine de okunamazsa uyarı basılıp o deney sıfırdan başlar, diğerleri etkilenmez.

### Colab hücreleri

```python
# 1. HÜCRE — kurulum (her oturumda, çökme sonrası dahil aynen çalıştırılır)
from google.colab import drive; drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/MTGCLM'
OUT   = f'{DRIVE}/artifacts'          # KALICI çıktı dizini
H5    = '/content/mtg_train_v2.h5'    # hızlı yerel kopya

import os, shutil
os.makedirs(OUT, exist_ok=True)
src = f'{DRIVE}/mtg_train_v2.h5'
need = (not os.path.exists(H5)) or os.path.getsize(H5) != os.path.getsize(src)
if need:
    print('H5 kopyalanıyor (20-40 dk)...'); shutil.copyfile(src, H5); print('bitti')
else:
    print('H5 zaten yerelde ve boyutu doğru, kopyalama atlandı')

if not os.path.exists('/content/MTGCLM'):
    !git clone https://github.com/<kullanıcı>/MTGCLM.git /content/MTGCLM
%cd /content/MTGCLM
!git pull --ff-only
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

```python
# 2. HÜCRE — koşular. Çökerse SADECE bu hücreyi tekrar çalıştırın.
!python src/run_all.py --h5 {H5} --outdir {OUT}     --split splits/split_v1_compact.npz --cv_prefix splits/cv5     --seeds 1,2,3 --arch_model medium_flat --stage all
```

```python
# 3. HÜCRE — durum ve rapor
!python src/run_all.py --h5 {H5} --outdir {OUT} --stage all --dry_run   # ne kaldı?
!python src/aggregate_results.py --dir {OUT} --out {OUT}/summary --plot
```

Mimari ablasyonunun kazananını gördükten sonra `--arch_model` değerini güncelleyip aynı komutu
tekrar verin; biten aşamalar atlanır, yalnızca modalite/CV koşuları yeni mimariyle çalışır
(bu durumda `<outdir>/status/` altındaki `A_*`, `B_*`, `C_*`, `CV_*` işaretlerini silin).

Aşamayı tek tek koşmak isterseniz: `--stage arch`, `--stage modality,unet`, `--stage cv,baselines`.
Her koşunun tam çıktısı `<outdir>/logs/<koşu>.log` dosyasında, özet durum
`<outdir>/run_manifest.json` içinde tutulur.

> Çökme sıklığını azaltmak için: sekmeyi açık bırakın (Pro+ arka plan yürütme sunar), ve
> `--stage` ile işi 2-3 saatlik parçalara bölün. Yine de çökerse hiçbir şey kaybolmaz.
