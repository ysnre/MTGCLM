# CLAUDE.md — MTGCLM

Bu dosya, Claude Code / Cowork oturumlarının projeyi sıfırdan anlamadan kaldığı yerden
devam edebilmesi için tutulur. **Yeni bir karar alındığında (model seçimi, veri bölme,
etiket politikası, eğitim komutu) bu dosyaya birkaç satır ekleyin.**

Son güncelleme: 2026-09-17

---

## 1. Proje özeti

MTG (Meteosat Third Generation) FCI uydu verisi + radar + yer gözlemleriyle **Türkiye
üzerinde bulut maskesi** üretmek. İki hat var:

- **Sınıflandırma (nokta):** İstasyon merkezli 33×33 yamadan istasyonun bulutlu/bulutsuz
  durumunu tahmin eden `MTGConvNet` (Shallow/Medium/Deep × Flatten/GAP ablasyonu) ve
  17 kanala uyarlanmış `ResNet18`.
- **Segmentasyon (hedef ürün):** `MTGUNet` / `CloudUNet_Fusion` ile piksel bazlı tahmin;
  eğitimde yalnız merkez (16,16) pikselde denetim (`MaskedBCELoss`).
  **Nihai hedef ürün tam harita (full-map) bulut maskesidir** — 800×1000 sahne tek geçişte.

### Veri
- Uydu grid'i: lon 25.01–44.99 (0.02°), lat 42.995–35.005 (0.01°) → **800×1000**, 1 km kabulü.
- 17 kanal = 16 FCI + 1 radar. VIS kanalları gece ~0 (NaN değil) — makalede belirtilecek.
- Radar PNG'lerinde dBZ değeri **mavi kanala** yazılı.
- AWOS `SAAT` alanı **UTC**; `DAKIKA_BASLANGICI` 10 dk periyodun **başlangıcı**.
- METAR `BKN` → 5 okta.
- Etiket kaynağı: 147 insanlı sinoptik istasyon + 79 havaalanı. Geç füzyon/raster için
  ayrıca 1982 otomatik istasyonun T/RH verisi.
- 942.944 yama–istasyon eşleşmesinin 788.086'sı AWOS CSV'de bulundu (~%16 eksik, medyanla dolduruldu).

### Yollar
- Kod + `awos_aligned`: `E:\belgeler\MTGCLM`
- Ana H5: `F:/radar/mtg_patch_dataset.h5`
- Colab/kompakt H5: `F:/radar/mtg_train_v2.h5` → Colab'da `/content/mtg_train_v2.h5`
- `src/config.py` Colab'ı otomatik algılar; **`MTGCLM_H5` ortam değişkeni H5 yolunu geçersiz kılar.**

---

## 2. Önemli kararlar (karar günlüğü)

### K1 — Nokta skalerleri yerine **raster yardımcı (aux) kanallar** (15 Eyl 2026)
Geç füzyon 3 skaleri yalnız etiketli 226 istasyonun kendi AWOS'undan alıyordu; 1982
istasyonun yoğunluğu kullanılmıyordu ve skaler girdi **tam harita üretimine izin vermiyor**.
Karar: T2m, RH2m (1982 istasyondan IDW), DEM/elev, cos(SZA), radar_cov → `aux[N,5,33,33]`
float16 olarak H5'e yazılır, `patch` ile kanal ekseninde birleşir → **17 + 5 = 22 kanal**.
Sonuç: tam harita çıkarımı mümkün, eksik veri komşudan dolar, `*_LateFusion` ve
`CloudUNet_Fusion` yalnızca ablasyon için kalır.

### K2 — Etiket politikası
`clear = 0 okta`, `cloudy ≥ 5 okta` (BKN/OVC), `1–4 okta → -1 (ignore)`.
Ham `okta` da H5'e yazılır (ileride yeniden etiketleme için).
`--okta9 rh` varsayılan: N=9 (gök görünmüyor) + istasyonda RH ≥ %90 → sis/alçak stratus →
cloudy; RH düşükse ignore (`--fog_rh`).

### K3 — Çift örnek (dedup)
Aynı (istasyon, gözlem zamanı) için uyduya en yakın olmayan kopya `is_dup` → etiket −1.
±9 dk tolerans nedeniyle geçerli örnek sayısı ~yarıya iner; **gerçek N budur.**

### K4 — Veri bölme (splits) mantığı — SIZINTI ÖNLEME
- **Hafta-blok zaman bölmesi** (block_days=7, 70/15/15, seed 42) **+ istasyon ayrımı**
  (30 holdout istasyon) → beş küme: `train / val / test_time / test_station / test_both`
  (+ etiketsiz `test_partial`, 1–4 okta).
- Mevcut bölme: `splits/split_v1.npz` (+ `.json` meta, `_compact.npz` kompakt H5 için)
  - train 218.648 | val 52.033 | test_time 51.704 | test_station 49.228 | test_both 11.655 | test_partial 16.941
  - h5 `mtg_patch_dataset.h5` (n_total 1.477.101), seed 42, drop_dups=true
  - holdout istasyonlar: 17022, 17026, 17056, 17066, 17069, 17077, 17080, 17084, 17088,
    17096, 17099, 17116, 17119, 17120, 17124, 17129, 17150, 17170, 17180, 17200, 17221,
    17239, 17246, 17256, 17282, 17290, 17297, 17298, 17371, 17380
- **Sıra önemli: aux → labels → splits.** Etiket politikası değişirse bölmeyi yeniden üretin.
- Model seçimi **val**'de, raporlama yalnız **test**'te.

### K5 — Model seçim metriği
Artık **val balanced accuracy** (F1(cloudy) çoğunluk sınıfını ödüllendiriyordu).
Sonuç JSON'undaki `best_f1` alanı geriye uyumluluk için duruyor ama **balanced accuracy içerir**.
Raporlama metrikleri: Balanced Acc, MCC, F1(clear/cloudy), macro-F1, ROC-AUC,
karışıklık matrisi + majority baseline (`src/metrics.py`).

### K6 — Hangi model en iyi? (mevcut durum)
Sınıflandırma ablasyonu (16 FCI + radar, **eski** etiket/bölme, `artifacts/ablation_results.json`):

| Model | best_f1 | params |
|---|---|---|
| ResNet18 | **0.8432** | 11.221.442 |
| Medium_Flat | 0.8400 | 228.994 |
| Shallow_Flat | 0.8397 | 285.954 |
| Deep_Flat | 0.8388 | 459.138 |
| Shallow_GAP | 0.8370 | 27.906 |
| Medium_GAP | 0.8355 | 106.114 |

Okuma: **Flatten > GAP** (konumsal bilgi önemli); derinlik artışı kazandırmıyor;
ResNet18 ~40× parametreye rağmen yalnız +0.003 → **fiyat/performansta Medium_Flat en iyi**,
ResNet18 referans/üst sınır olarak tutulur.
`artifacts/unet_results.json` yalnızca **1 epoch, 20 batch dry-run**tır — gerçek sonuç değildir,
karşılaştırmada kullanmayın.

> ⚠️ Yukarıdaki ablasyon **v2 öncesi** etiket/bölme ile alındı. v2 (aux kanallar + yeni
> etiket + sızıntısız bölme) sonuçları geldiğinde bu tabloyu güncelleyin.

### K7 — Diğer bilinçli kararlar
- `aux` ham fiziksel birimlerde (°C, %, m) saklanır; normalizasyon `aux_means/aux_stds`
  attrs ile dataset içinde yapılır. `cos_sza` ve `radar_cov` ham bırakılır.
- `radar_cov` 0/1 maske değil, **1 − d/r** yakınlığı (radar üstünde 1, menzil kenarında 0).
- AWOS zaman eşlemesi **floor** (uyduyu kapsayan 10 dk penceresi); eski davranış `--time_match nearest`.
- Sıcaklıkta yükseklik düzeltmesi `--lapse_rate 6.5` K/km (0 = kapalı). DEM yoksa yükseklik
  IDW ile üretilir — kaba; makale için **gerçek DEM** verin.
- Kompakt H5 `pre_shuffled=1` attr'ı taşır; `train.py` bunu görünce `buffered_shuffle_generator`'ı atlar.
- `torch.load(..., weights_only=False, map_location=DEVICE)`.
- Eğitimler **Colab**'da; yerelde yalnız dry-run. Colab Pro+ (arka plan yürütme, 24 sa oturum)
  ~21 koşu × 1.5–2.5 sa ≈ 40–50 sa T4 için önerildi; iş IO-sınırlı olduğundan T4/L4 A100'den ekonomik.

---

## 3. Eğitim / veri hazırlama komutları

Sıra: **aux → labels → splits → export → dry-run → Colab**

```bat
cd e:\belgeler\MTGCLM

:: 0) DEM (bir kez)  (pip install rasterio)
python src\make_dem_npy.py --dem_tif data/dem/<dem>.tif --out data/dem_800x1000.npy

:: 1) Yardımcı kanallar (~20-30 dk; kesilirse aynı komut kaldığı yerden sürer)
python src\update_h5_aux_channels.py --h5 F:/radar/mtg_patch_dataset.h5 ^
    --awos_dir data/raw_awos_observation --station_csv data/raw_observation/station_list.csv ^
    --sample_nc data/raw_mtg/turkiye_kesit_20260525000817_0001.nc ^
    --radar_csv data/raw_radar/radar.csv --radar_dir F:/radar/raw_radar --dem data/dem_800x1000.npy

:: 2) Etiketler (önce --dry_run ile dağılıma bakın; okta 9 kararı RH'ye bağlı → aux'tan SONRA)
python src\update_h5_labels_v2.py --h5 F:/radar/mtg_patch_dataset.h5 --awos_aligned_dir data/processed/awos_aligned --dry_run
python src\update_h5_labels_v2.py --h5 F:/radar/mtg_patch_dataset.h5 --awos_aligned_dir data/processed/awos_aligned

:: 3) Bölme
python src\make_splits.py --h5 F:/radar/mtg_patch_dataset.h5 --out splits/split_v1 --seed 42 --holdout_stations 30

:: 4) Colab için kompakt, ön-karıştırılmış H5 (~19 GB, float16)
python src\export_compact_h5.py --src F:/radar/mtg_patch_dataset.h5 --split splits/split_v1.npz --out F:/radar/mtg_train_v2.h5 --seed 42

:: 5) Yerel dry-run
set MTGCLM_H5=F:/radar/mtg_train_v2.h5
python src\train.py --model mtg_flat --split splits/split_v1_compact.npz --aux --epochs 1 --limit_batches 30
python src\train_unet.py --model unet --split splits/split_v1_compact.npz --aux --epochs 1 --limit_batches 20
```

### Colab
```bash
export MTGCLM_H5=/content/mtg_train_v2.h5
# kompakt dosyayla çalışırken splits/split_v1.npz yerine splits/split_v1_compact.npz kullanın

# Modalite ablasyonu (aynı split, 3 seed):
python src/train.py --model mtg_flat        --split splits/split_v1.npz --seed 1        --results artifacts/sat_radar_s1.json
python src/train.py --model mtg_flat        --split splits/split_v1.npz --seed 1 --aux  --results artifacts/sat_radar_aux_s1.json
python src/train.py --model mtg_flat --split splits/split_v1.npz --seed 1 --aux --aux_channels cos_sza,elev,radar_cov --results artifacts/no_awos_s1.json
python src/train_unet.py --model unet       --split splits/split_v1.npz --seed 1 --aux  --results artifacts/unet_aux_s1.json
```

Ortak argümanlar: `--split --aux --aux_channels --seed --epochs --limit_batches --results`.
Kanal sayısı veri setinden okunur (`num_channels`).
Uydu-only (radar hariç) ablasyonu için `MTGH5Dataset`'e kanal seçme parametresi eklenmeli (henüz yok).

---

## 4. Dosya haritası (v2)

| Dosya | Ne yapar |
|---|---|
| `src/grid_utils.py` | Düzenli lat/lon grid, km projeksiyonu, cos(SZA), 10 dk pencere yardımcıları |
| `src/update_h5_aux_channels.py` | `aux[N,5,33,33]` float16 (t2m, rh2m, elev, cos_sza, radar_cov) + `station_y/x`, `aux_flags` |
| `src/update_h5_labels_v2.py` | Parametrik etiket, ham `okta`, `obs_time`, `is_dup`, `labels_prev` yedeği |
| `src/make_splits.py` | Hafta-blok + istasyon ayrımı → 5 küme (+ `test_partial`) `.npz`/`.json` |
| `src/export_compact_h5.py` | Bölmedeki örnekler, global permütasyon, float16 → `mtg_train_v2.h5` |
| `src/make_dem_npy.py` | DEM GeoTIFF → 800×1000 `.npy` (metre) |
| `src/metrics.py` | Balanced Acc, MCC, F1(clear/cloudy), AUC, karışıklık matrisi, majority baseline |
| `src/h5_dataset.py` | `use_aux=True` → aux normalize edilip kanallara eklenir |
| `src/dataloader.py` | `get_split_dataloaders()`, `get_dataloaders(use_aux=...)` |
| `src/model.py` | `MTGConvNet_Flat`, `MTGConvNet_LateFusion`, `ResNet18_LateFusion`, `CloudUNet_Fusion`, `MTGUNet` |
| `src/loss.py` | `MaskedBCELoss` — yalnız (16,16) pikselinde, `-1` yok sayılır |
| `src/train.py`, `src/train_unet.py` | Eğitim; model seçimi val balanced accuracy; test kümeleri JSON'a |
| `src/config.py` | Colab/yerel yollar, `MTGCLM_H5` override, PATCH_SIZE 33, BATCH_SIZE 64, LR 1e-3 |

Eski (v1 skaler geç-füzyon) dosyaların yedeği: `src/_backup_20260916/`.
Akış dokümanı: `walkthrough_v2.md` (v1: `walkthrough.md`). Görev listesi: `task.md`.

---

## 5. Açık işler / sıradaki adımlar

- [ ] Gerçek DEM GeoTIFF temin edip `make_dem_npy.py` ile üret (şu an IDW yükseklik kaba).
- [ ] v2 hattını uçtan uca koş (aux → labels → splits → export) ve ablasyon tablosunu yenile.
- [ ] Baseline'lar: majority, IR10.5 BT eşiği, VIS0.6+IR; mümkünse EUMETSAT FCI CLM / NWC SAF CMa
      aynı istasyon-anlarında.
- [ ] U-Net tam harita için hedef-piksel jitter (49×49 çıkar → 33×33 rastgele kırp).
- [ ] `infer_scene.py`: tek NetCDF sahne + radar + aux → 800×1000 olasılık haritası (GeoTIFF/PNG).
- [ ] Kanal permütasyon önemi / Integrated Gradients (17→22 kanal için asıl bilimsel soru).
- [ ] Metrikleri gündüz/gece ve mevsim kırılımıyla raporla.
- [ ] Makale limitations: Tem–Kas verisi yok; kar–bulut karışıklığı; 1 km piksel vs yarımküre
      gözlem ölçek farkı; etiket eşiği duyarlılığı (0/≥5 vs 0/≥1).

---

## 6. Referans dokümanlar (Cowork projesi "MTGCLM")

Cowork projesinde **özel proje talimatı (custom instructions) tanımlı değil**; bilgi tabanı
şu üç dokümandan oluşuyor:

- `claude/mtgclm_inceleme_2026-09-15.md` — akademik + teknik inceleme raporu
- `claude/mtgclm_eylem_plani_2026-09-15.md` — 5 fazlı eylem planı (bu dosyadaki kararların kaynağı)
- `claude/mtgclm_walkthrough_v2.md` — v2 akışı (yereldeki `walkthrough_v2.md` ile aynı)

Yereldeki diğer notlar: `implementation_plan.md` (v1 geç-füzyon planı, arşiv),
`MTG Cloud Phase Web Service Development.md`, `Visualizing MTGConvNet Architecture2.md`.

---

## 7. Çalışma kuralları (Claude için)

- Türkçe yanıt ver.
- Veri sızıntısı kurallarına **sıkı** uy: bölme dosyası dışında örnek kullanma, model seçimini
  val'de yap, test'e yalnız bir kez dokun.
- Yolları sabit kodlama; `config.py` / `MTGCLM_H5` / argparse kullan.
- H5'i yeniden çıkarma (`create_patch_dataset.py`) — yamalar sabit, yalnız dataset eklenir.
- Ağır eğitim Colab'da; yerelde `--epochs 1 --limit_batches N` dry-run.
- Yeni bir karar alındığında bu dosyanın §2'sine ekle.
