# CLAUDE.md — MTGCLM

Bu dosya, Claude Code / Cowork oturumlarının projeyi sıfırdan anlamadan kaldığı yerden
devam edebilmesi için tutulur. **Yeni bir karar alındığında (model seçimi, veri bölme,
etiket politikası, eğitim komutu) bu dosyaya birkaç satır ekleyin.**

Son güncelleme: 2026-09-18

---

## 1. Proje özeti

MTG (Meteosat Third Generation) FCI uydu verisi + TMM C-bandı radar + yer gözlemleriyle
**Türkiye üzerinde bulut maskesi** üretmek. İki hat var:

- **Sınıflandırma (nokta):** İstasyon merkezli 33×33 yamadan istasyonun bulutlu/bulutsuz
  durumunu tahmin eden `MTGConvNet` (Shallow/Medium/Deep × GAP/Flatten = 6 varyant) ve
  17 kanala uyarlanmış `ResNet18`.
- **Segmentasyon (hedef ürün):** `MTGUNet` / `CloudUNet_Fusion` ile piksel bazlı tahmin;
  eğitimde yalnız merkez (16,16) pikselde denetim (`MaskedBCELoss`).
  **Nihai hedef ürün tam harita (full-map) bulut maskesidir** — 800×1000 sahne tek geçişte.

### Veri
- Uydu grid'i: lon 25.01–44.99 (0.02°), lat 42.995–35.005 (0.01°) → **800×1000**, 1 km kabulü.
- Dönem: Mar–Nis 2025, Ara 2025 – Haz 2026 (Tem–Kas verisi yok).
- 17 kanal = 16 FCI + 1 radar; + 5 aux = **22 kanal**. VIS kanalları gece ~0 (NaN değil).
- Radar PNG'lerinde dBZ değeri **mavi kanala** yazılı.
- AWOS `SAAT` alanı **UTC**; `DAKIKA_BASLANGICI` 10 dk periyodun **başlangıcı**.
- METAR `BKN` → 5 okta.
- Etiket kaynağı: 147 insanlı sinoptik istasyon + 79 havaalanı (METAR). Yardımcı alanlar
  1982 otomatik istasyondan (AWOS) IDW ile.

### Eğitim veri seti (`mtg_train_v2.h5`, ~19 GB, float16, ön-karıştırılmış)
```
patches      [N,17,33,33] float16   16 FCI + radar (kanal bazında z-skor)
aux          [N, 5,33,33] float16   t2m, rh2m, elev, cos_sza, radar_cov (ham birimlerde)
labels       [N] int64              0 = açık (0 okta), 1 = bulutlu (≥5 okta), -1 = yok say
okta         [N] int8               ham bulut miktarı (0–9)
wmoids, timestamps, station_y/x, obs_time, obs_dt_sec, is_dup, source_index
attrs: means, stds, aux_channels, aux_means, aux_stds, label_policy, pre_shuffled
```
N ≈ 400.209 (bölmedeki örnekler + test_partial).

### Yollar
- Kod: `E:\belgeler\MTGCLM` — **git deposu**, remote `https://github.com/ysnre/MTGCLM.git`, dal `main`
- Ana (kaynak) H5: `F:/radar/mtg_patch_dataset.h5` (n_total 1.477.101)
- Kompakt/eğitim H5: `F:/radar/mtg_train_v2.h5` → Colab'da `/content/mtg_train_v2.h5`
- H5 yolu önceliği: **`--h5` argümanı** > `MTGCLM_H5` ortam değişkeni > `config.H5_PATH`.
  Betikleri çağırırken **her zaman `--h5` verin.**
- MongoDB URI ortam değişkeninden okunur (kodda gömülü değil).

---

## 2. Önemli kararlar (karar günlüğü)

### K1 — Nokta skalerleri yerine **raster yardımcı (aux) kanallar** (15 Eyl 2026)
Geç füzyon 3 skaleri yalnız etiketli 226 istasyonun kendi AWOS'undan alıyordu; 1982
istasyonun yoğunluğu kullanılmıyordu ve skaler girdi **tam harita üretimine izin vermiyor**.
Karar: t2m, rh2m (IDW), elev/DEM, cos_sza, radar_cov → `aux[N,5,33,33]` float16 olarak H5'e
yazılır, `patch` ile kanal ekseninde birleşir → **17 + 5 = 22 kanal**.
`*_LateFusion` ve `CloudUNet_Fusion` yalnızca ablasyon için kalır.

### K2 — Etiket politikası
`clear = 0 okta`, `cloudy ≥ 5 okta` (BKN/OVC), `1–4 okta → -1 (ignore)`; ham `okta` da H5'te.
`--okta9 rh` varsayılan: N=9 (gök görünmüyor) + istasyonda RH ≥ %90 → sis/alçak stratus →
cloudy; RH düşükse ignore (`--fog_rh`).

### K3 — Çift örnek (dedup)
Aynı (istasyon, gözlem zamanı) için uyduya en yakın olmayan kopya `is_dup` → etiket −1.
±9 dk tolerans nedeniyle geçerli örnek sayısı ~yarıya iner; **gerçek N budur.**

### K4 — Veri bölme (splits) mantığı — SIZINTI ÖNLEME
İki bölme rejimi bir arada tutuluyor:

**(a) Tek bölme `splits/split_v1*` (ana rapor):** hafta-blok zaman bölmesi (block_days=7,
70/15/15, seed 42) **+ istasyon ayrımı** (30 holdout istasyon) → `train / val / test_time /
test_station / test_both` + etiketsiz `test_partial` (1–4 okta).
- train 218.648 | val 52.033 | test_time 51.704 | test_station 49.228 | test_both 11.655 | test_partial 16.941
- holdout istasyonlar: 17022, 17026, 17056, 17066, 17069, 17077, 17080, 17084, 17088, 17096,
  17099, 17116, 17119, 17120, 17124, 17129, 17150, 17170, 17180, 17200, 17221, 17239, 17246,
  17256, 17282, 17290, 17297, 17298, 17371, 17380
- Kompakt H5 ile **`splits/split_v1_compact.npz`** kullanılır (aynı sayılar, yeniden indekslenmiş).

**(b) Bloklu 5-katlı CV `splits/cv5_fold0..4.npz` (danışman talebi):** ardışık günler 7 günlük
bloklara ayrılıp katlara dağıtılır; aynı sinoptik durum hem eğitimde hem testte olmaz.
Kat başına ≈ train 252–258 bin, val 48–54 bin, test_time 73–78 bin (`splits/cv5_cv.json`).
Kat bazında bulutlu oranı %36–%74 arasında değişiyor — **majority baseline'ı kat kat raporlayın.**

- **Sıra önemli: aux → labels → splits.** Etiket politikası değişirse bölmeyi yeniden üretin.
- Model seçimi **val**'de, raporlama yalnız **test**'te.
- **Bölme–H5 uyuşmazlığı artık yakalanıyor:** `get_split_dataloaders` indeks aralığını ve etiket
  geçerliliğini doğrular; başlangıçta "Bölme doğrulandı: N satırlık H5 ile uyumlu." satırını görmelisiniz.
  (Kompakt bölmeyi kaynak H5'e uygulamak eskiden sessiz hataydı.)

### K5 — Model seçim metriği
**Val balanced accuracy** (F1(cloudy) çoğunluk sınıfını ödüllendiriyordu). Sonuç JSON'undaki
`best_f1` alanı geriye uyumluluk için duruyor ama **balanced accuracy içerir**.
Raporlama: Balanced Acc, MCC, sınıf bazında precision/recall/F1, accuracy, ROC-AUC,
karışıklık matrisi + majority referansı (`src/metrics.py`).

### K6 — Mimari ablasyonu ve "hangi model en iyi?"
`train.py --model arch` altı varyantı sırayla koşar (parametre sayıları v1 ile birebir aynı;
aux açıkken her varyant **+1.440** parametre alır: 5 ek kanal × 32 filtre × 3×3):

| Deney | conv_blocks | Havuzlama | Parametre (17 kanal) |
|---|---|---|---|
| Shallow_GAP | [32, 64] | GAP | 27.906 |
| Shallow_Flat | [32, 64] | Flatten | 285.954 |
| Medium_GAP | [32, 64, 128] | GAP | 106.114 |
| Medium_Flat | [32, 64, 128] | Flatten | 228.994 |
| Deep_GAP | [32, 64, 128, 256] | GAP | 409.986 |
| Deep_Flat | [32, 64, 128, 256] | Flatten | 459.138 |

Kısa adlar (`--model medium_flat`) veya doğrudan deney adı (`--model Medium_GAP`) çalışır.
`--model mtg_flat` geriye uyumluluk için **Deep_Flat**'i koşar.

**v1 (arşiv) sonuçları** — `archive_v1/artifacts/ablation_results.json`, 16 FCI + radar,
**eski etiket/bölme**, metrik o zamanki F1:

| Model | best_f1 | params |
|---|---|---|
| ResNet18 | **0.8432** | 11.221.442 |
| Medium_Flat | 0.8400 | 228.994 |
| Shallow_Flat | 0.8397 | 285.954 |
| Deep_Flat | 0.8388 | 459.138 |
| Shallow_GAP | 0.8370 | 27.906 |
| Medium_GAP | 0.8355 | 106.114 |

Okuma: **Flatten > GAP** (konumsal bilgi önemli); derinlik kazandırmıyor; ResNet18 ~40×
parametreye rağmen +0.003 → fiyat/performansta **Medium_Flat** önde.

> ⚠️ **v2 (aux + yeni etiket + sızıntısız bölme) koşuları henüz yapılmadı** — `artifacts/` içinde
> sonuç JSON'u yok. Yukarıdaki tablo referans/arşivdir; v2 ARCH koşuları bitince bu bölümü
> güncelleyin. Eski `unet_results.json` da 1 epoch / 20 batch dry-run'dır, sonuç değildir.

### K7 — Kazanan mimari otomatik seçilir
`run_all.py --arch_model auto` (varsayılan) `ARCH_*.json` dosyalarını okur, her varyantın
seed'ler arası **ortalama val balanced accuracy**'sini hesaplar, en iyisini modalite/CV/U-Net
koşularında kullanır ve seçim tablosunu basar. Mimari ablasyonu koşulmadıysa net hata verip durur.
Sonradan başka bir mimari zorlanırsa (`--arch_model Medium_Flat`) eski mimariyle tamamlanmış
koşuları fark edip hangi `.done` işaretlerinin silinmesi gerektiğini söyler — **sessizce karışık
sonuç üretmez.**

### K8 — Çökmeye dayanıklı koşu altyapısı (Colab)
Colab oturumu çökünce `/content` silinir; bu yüzden checkpoint, en iyi model ve sonuç JSON'ları
**Drive'a** (`--outdir`) yazılır. Üç seviyede devam:

| Seviye | Nasıl | Dosya |
|---|---|---|
| Koşu | tamamlanan koşu bir daha çalışmaz | `<outdir>/status/<koşu>.done` |
| Deney | `--model arch` içinde biten varyantlar atlanır | sonuç JSON'undaki deney adları |
| Epoch | model+optimizer+scheduler+history yazılır | `<outdir>/checkpoints/checkpoint_<ad>.pth` |

Tüm yazmalar **atomik** (`.tmp` → `os.replace`). Bozuk checkpoint okunamazsa uyarı basılır, o
deney sıfırdan başlar, diğerleri etkilenmez. Loglar `<outdir>/logs/<koşu>.log`, özet
`<outdir>/run_manifest.json`.

> Dikkat: `train.py` sonuç JSON'unda **adı zaten bulunan** deneyi atlar. Yeniden koşmak için
> ya yeni bir `--results` dosyası verin ya da eski JSON'u silin (`run_all.py --force` da var).

### K9 — Karşılaştırma modelleri (baselines)
`src/baselines.py`: majority, tek kanal eşiği (`threshold`), çift eşik (`threshold2`), logreg,
random forest, HGB, kNN — **aynı bölme, aynı metrikler, aynı JSON şeması** (tek tabloda birleşir).
Eşikler **yalnız train** üzerinde optimize edilir, test'e bakılmaz.

### K10 — Depo / yayın düzeni (17–18 Eyl 2026)
- GitHub: `https://github.com/ysnre/MTGCLM.git`, MIT lisans, `README.md`, `CITATION.cff`.
- `.gitignore`: veri (`*.h5 *.nc *.npy *.tif`, `data/raw_*`, `data/processed/`, `data/dem/`),
  ağırlıklar (`*.pth`), `scratch/`, `sonuç1/`, `graphify-out/`, `.claude/`, PNG çıktıları **dışarıda**;
  **`splits/*.npz|json` ve `artifacts/*.json` PAYLAŞILIR.**
- v1 sonuçları/ağırlıkları `archive_v1/` altına taşındı.
- Veri bulunabilirliği: MTG FCI → EUMETSAT Data Store (açık); radar + yer gözlemleri MGM'ye ait,
  paylaşılmıyor. Türetilmiş 19 GB H5 → **Zenodo** (DOI bekleniyor). ~2.000 yamalık örneklem
  `src/make_public_sample.py` ile üretilir.
- DEM kaynağı: **Copernicus DEM GLO-90** (açık); ham DEM dosyaları depo dışında.

### K11 — Diğer bilinçli kararlar
- `aux` ham fiziksel birimlerde (°C, %, m); normalizasyon `aux_means/aux_stds` attrs ile dataset
  içinde. `cos_sza` ve `radar_cov` ham bırakılır.
- `radar_cov` 0/1 maske değil, **1 − d/r** yakınlığı (radar üstünde 1, menzil kenarında 0).
- AWOS zaman eşlemesi **floor** (uyduyu kapsayan 10 dk penceresi); eski davranış `--time_match nearest`.
- Sıcaklıkta yükseklik düzeltmesi `--lapse_rate 6.5` K/km (0 = kapalı).
- Kompakt H5 `pre_shuffled=1` attr'ı taşır; `train.py` görünce `buffered_shuffle_generator`'ı atlar.
- `torch.load(..., weights_only=False, map_location=DEVICE)`.
- Test kümeleri yalnız eğitim bitince, en iyi val modeliyle **bir kez** değerlendirilir.
- Eğitimler Colab'da; yerelde `--epochs 1 --limit_batches N` dry-run. T4'te epoch ≈ 3 dk.

---

## 3. Komutlar

### 3.1 Veri hazırlama (yerel, tek seferlik) — sıra: aux → labels → splits → export
```bat
cd e:\belgeler\MTGCLM

:: 0) DEM (bir kez)  (pip install rasterio)
python src\make_dem_npy.py --dem_tif data/dem/<dem>.tif --out data/dem_800x1000.npy

:: 1) Yardımcı kanallar (~20-30 dk; kesilirse aynı komut kaldığı yerden sürer)
python src\update_h5_aux_channels.py --h5 F:/radar/mtg_patch_dataset.h5 ^
    --awos_dir data/raw_awos_observation --station_csv data/raw_observation/station_list.csv ^
    --sample_nc data/raw_mtg/turkiye_kesit_20260525000817_0001.nc ^
    --radar_csv data/raw_radar/radar.csv --radar_dir F:/radar/raw_radar --dem data/dem_800x1000.npy

:: 2) Etiketler (önce --dry_run; okta 9 kararı RH'ye bağlı → aux'tan SONRA)
python src\update_h5_labels_v2.py --h5 F:/radar/mtg_patch_dataset.h5 --awos_aligned_dir data/processed/awos_aligned --dry_run
python src\update_h5_labels_v2.py --h5 F:/radar/mtg_patch_dataset.h5 --awos_aligned_dir data/processed/awos_aligned

:: 3) Bölmeler
python src\make_splits.py    --h5 F:/radar/mtg_patch_dataset.h5 --out splits/split_v1 --seed 42 --holdout_stations 30
python src\make_cv_splits.py --h5 F:/radar/mtg_train_v2.h5      --out splits/cv5 --folds 5 --block_days 7 --seed 42

:: 4) Colab için kompakt, ön-karıştırılmış H5 (~19 GB, float16)
python src\export_compact_h5.py --src F:/radar/mtg_patch_dataset.h5 --split splits/split_v1.npz --out F:/radar/mtg_train_v2.h5 --seed 42

:: 5) Yerel dry-run -- H5 yolunu HER ZAMAN --h5 ile verin
python src\train.py      --h5 F:/radar/mtg_train_v2.h5 --model mtg_flat --split splits/split_v1_compact.npz --aux --epochs 1 --limit_batches 30
python src\train_unet.py --h5 F:/radar/mtg_train_v2.h5 --model unet     --split splits/split_v1_compact.npz --aux --epochs 1 --limit_batches 20
```

### 3.2 Colab — tek komutla tüm matris (önerilen)
```python
# 1. HÜCRE — kurulum (her oturumda, çökme sonrası dahil aynen)
from google.colab import drive; drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/MTGCLM'
OUT   = f'{DRIVE}/artifacts'          # KALICI çıktı dizini
H5    = '/content/mtg_train_v2.h5'    # hızlı yerel kopya
import os, shutil
os.makedirs(OUT, exist_ok=True)
src = f'{DRIVE}/mtg_train_v2.h5'
if (not os.path.exists(H5)) or os.path.getsize(H5) != os.path.getsize(src):
    print('H5 kopyalanıyor (20-40 dk)...'); shutil.copyfile(src, H5); print('bitti')
if not os.path.exists('/content/MTGCLM'):
    !git clone https://github.com/ysnre/MTGCLM.git /content/MTGCLM
%cd /content/MTGCLM
!git pull --ff-only
```
```python
# 2. HÜCRE — koşular. Çökerse SADECE bu hücreyi tekrar çalıştırın.
!python src/run_all.py --h5 {H5} --outdir {OUT} \
    --split splits/split_v1_compact.npz --cv_prefix splits/cv5 \
    --seeds 1,2,3 --arch_model auto --stage all
```
```python
# 3. HÜCRE — durum ve rapor
!python src/run_all.py --h5 {H5} --outdir {OUT} --stage all --dry_run   # ne kaldı?
!python src/aggregate_results.py --dir {OUT} --out {OUT}/summary --plot
```

`run_all.py` aşamaları: **`arch, resnet, modality, unet, cv, baselines`** (`--stage all` ya da
virgülle alt küme). Diğer argümanlar: `--cv_folds 5 --seeds 1,2,3 --epochs --limit_batches
--dry_run --force --retries`. İşi 2–3 saatlik parçalara bölmek için `--stage arch`,
`--stage modality,unet`, `--stage cv,baselines`.

### 3.3 Elle koşmak isterseniz
```bash
H5=/content/mtg_train_v2.h5; SP=splits/split_v1_compact.npz

# mimari ablasyonu + resnet
python src/train.py --h5 $H5 --split $SP --model arch     --aux --seed 1 --results artifacts/ARCH_s1.json
python src/train.py --h5 $H5 --split $SP --model resnet18 --aux --seed 1 --results artifacts/RESNET_s1.json

# modalite ablasyonu (aynı split, 3 seed)
python src/train.py --h5 $H5 --split $SP --model mtg_flat --seed 1                                            --results artifacts/sat_radar_s1.json
python src/train.py --h5 $H5 --split $SP --model mtg_flat --seed 1 --aux                                      --results artifacts/sat_radar_aux_s1.json
python src/train.py --h5 $H5 --split $SP --model mtg_flat --seed 1 --aux --aux_channels cos_sza,elev,radar_cov --results artifacts/no_awos_s1.json
python src/train_unet.py --h5 $H5 --split $SP --model unet --aux --seed 1 --results artifacts/unet_aux_s1.json

# bloklu 5-katlı CV
for k in 0 1 2 3 4; do
  python src/train.py --h5 $H5 --split splits/cv5_fold$k.npz --model mtg_flat --aux --seed 1 \
      --results artifacts/CV_full_f${k}_s1.json
done

# baseline'lar + toplu rapor + paylaşılabilir örneklem
python src/baselines.py --h5 $H5 --split $SP --aux --models majority,threshold,threshold2,logreg,rf,hgb --results artifacts/Z_baselines_s1.json
python src/aggregate_results.py --dir artifacts --out artifacts/summary --plot
python src/aggregate_results.py --dir artifacts --pattern "CV_*.json" --out artifacts/cv_summary
python src/make_public_sample.py --src $H5 --split $SP --out data/mtg_sample.h5 --n 2000
```
`aggregate_results.py` dosya adındaki **`_f<kat>_s<seed>`** kalıbını okuyup kat/seed koşularını
tek satırda ortalar (ortalama ± s.s.; Markdown + CSV + LaTeX + grafik).

---

## 4. Dosya haritası

| Dosya | Ne yapar |
|---|---|
| `src/grid_utils.py` | Düzenli lat/lon grid, km projeksiyonu, cos(SZA), 10 dk pencere |
| `src/update_h5_aux_channels.py` | `aux[N,5,33,33]` (t2m, rh2m, elev, cos_sza, radar_cov) + `station_y/x`, `aux_flags` |
| `src/update_h5_labels_v2.py` | Parametrik etiket, ham `okta`, `obs_time`, `is_dup`, `labels_prev` |
| `src/make_splits.py` | Hafta-blok + istasyon ayrımı → 5 küme (+ `test_partial`) |
| `src/make_cv_splits.py` | Bloklu K-katlı CV bölmeleri (`cv5_fold0..4.npz`) |
| `src/export_compact_h5.py` | Bölmedeki örnekler, global permütasyon, float16 → `mtg_train_v2.h5` |
| `src/make_dem_npy.py` | DEM GeoTIFF → 800×1000 `.npy` (metre) |
| `src/make_public_sample.py` | Paylaşılabilir küçük örneklem H5 + bölme (GitHub/Zenodo) |
| `src/metrics.py` | Balanced Acc, MCC, F1(clear/cloudy), AUC, karışıklık matrisi, majority |
| `src/baselines.py` | majority, threshold, threshold2, logreg, rf, hgb, knn |
| `src/aggregate_results.py` | Koşu JSON'ları → ortalama ± s.s. tablo (MD/CSV/LaTeX) + grafik |
| `src/run_all.py` | Çökmeye dayanıklı koşu yöneticisi (aşamalar, `.done`, manifest, log) |
| `src/h5_dataset.py` | `use_aux=True` → aux normalize edilip kanallara eklenir |
| `src/dataloader.py` | `get_split_dataloaders()` (+ bölme–H5 doğrulaması), `get_dataloaders(use_aux=...)` |
| `src/model.py` | `MTGConvNet_Flat`, `*_LateFusion`, `ResNet18_LateFusion`, `CloudUNet_Fusion`, `MTGUNet` |
| `src/loss.py` | `MaskedBCELoss` — yalnız (16,16) pikselinde, `-1` yok sayılır |
| `src/train.py`, `src/train_unet.py` | Eğitim; `--h5 --split --aux --aux_channels --seed --epochs --limit_batches --results --outdir --model` |
| `src/config.py` | Colab/yerel yollar, `MTGCLM_H5`, PATCH_SIZE 33, BATCH_SIZE 64, LR 1e-3 |
| `README.md`, `LICENSE`, `CITATION.cff`, `.gitignore` | Depo iskeleti + veri bulunabilirliği beyanı |

Arşiv: `archive_v1/` (v1 sonuçları + ağırlıkları), `src/_backup_20260916/` (v1 skaler füzyon kodu).
Akış dokümanı: `walkthrough_v2.md` (v1: `walkthrough.md`). Görev listesi: `task.md`.
`implementation_plan.md` v1 geç-füzyon planıdır — **arşiv**, güncel hattı temsil etmez.

---

## 5. Açık işler / sıradaki adımlar

- [ ] **v2 koşu matrisini Colab'da çalıştır** (`run_all.py --stage all`) → ARCH → kazanan mimari →
      modality / unet / cv / baselines; sonra §2/K6 tablosunu v2 sayılarıyla güncelle.
- [ ] Gerçek Copernicus GLO-90 DEM'i indirip `make_dem_npy.py` ile üret (IDW yükseklik kaba).
- [ ] Uydu-only (radar hariç) ablasyonu için `MTGH5Dataset`'e kanal seçme parametresi ekle
      (`base_channels` alt kümesi) — henüz yok.
- [ ] Mümkünse EUMETSAT FCI CLM / NWC SAF CMa ile aynı istasyon-anlarında karşılaştırma.
- [ ] U-Net tam harita için hedef-piksel jitter (49×49 çıkar → 33×33 rastgele kırp).
- [ ] `infer_scene.py`: tek NetCDF sahne + radar + aux → 800×1000 olasılık haritası (GeoTIFF/PNG).
- [ ] Kanal permütasyon önemi / Integrated Gradients (17→22 kanal için asıl bilimsel soru).
- [ ] Metrikleri gündüz/gece ve mevsim kırılımıyla raporla.
- [ ] Zenodo DOI al, README'ye ekle; eğitilmiş ağırlıkları release olarak yayımla.
- [ ] Makale limitations: Tem–Kas verisi yok; kar–bulut karışıklığı; 1 km piksel vs yarımküre
      gözlem ölçek farkı; etiket eşiği duyarlılığı (0/≥5 vs 0/≥1).

---

## 6. Referans dokümanlar (Cowork projesi "MTGCLM")

Cowork projesinde **özel proje talimatı (custom instructions) tanımlı değil**; bilgi tabanı
şu üç dokümandan oluşuyor:

- `claude/mtgclm_inceleme_2026-09-15.md` — akademik + teknik inceleme raporu
- `claude/mtgclm_eylem_plani_2026-09-15.md` — 5 fazlı eylem planı (kararların kaynağı)
- `claude/mtgclm_walkthrough_v2.md` — v2 akışı (yereldeki `walkthrough_v2.md` güncel sürümdür)

Yereldeki diğer notlar: `MTG Cloud Phase Web Service Development.md`,
`Visualizing MTGConvNet Architecture2.md`.

---

## 7. Çalışma kuralları (Claude için)

- Türkçe yanıt ver.
- Veri sızıntısı kurallarına **sıkı** uy: bölme dosyası dışında örnek kullanma, model seçimini
  val'de yap, test'e yalnız bir kez dokun.
- Betiklere **`--h5` ver**; yolları sabit kodlama (`config.py` / env / argparse).
- H5'i yeniden çıkarma (`create_patch_dataset.py`) — yamalar sabit, yalnız dataset eklenir.
- Ağır eğitim Colab'da, `run_all.py` üzerinden, `--outdir` Drive altında; yerelde dry-run.
- Depoya veri/ağırlık commit etme; `splits/*.npz|json` ve `artifacts/*.json` paylaşılır.
- Yeni bir karar alındığında bu dosyanın §2'sine ekle.
