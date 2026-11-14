# CLAUDE.md — MTGCLM

Bu dosya, Claude Code / Cowork oturumlarının projeyi sıfırdan anlamadan kaldığı yerden
devam edebilmesi için tutulur. **Yeni bir karar alındığında (model seçimi, veri bölme,
etiket politikası, eğitim komutu) bu dosyaya birkaç satır ekleyin.**

Son güncelleme: 2026-11-15

---

## 1. Proje özeti

MTG (Meteosat Third Generation) FCI uydu verisi + TMM C-bandı radar + yer gözlemleriyle
**Türkiye üzerinde bulut maskesi** üretmek. İki hat var:

- **Sınıflandırma (nokta):** İstasyon merkezli 33×33 yamadan istasyonun bulutlu/bulutsuz
  durumunu tahmin eden `MTGConvNet` (Shallow/Medium/Deep × GAP/Flatten = 6 varyant) ve
  22 kanala uyarlanmış `ResNet18`.
- **Segmentasyon (hedef ürün):** `MTGUNet` ile piksel bazlı tahmin; eğitimde yalnız merkez
  (16,16) pikselde denetim (`MaskedBCELoss`).
  **Nihai hedef ürün tam harita (full-map) bulut maskesidir** — 800×1000 sahne tek geçişte.

> **Durum (5 Eki 2026): v2 koşu matrisi tamamlandı.** ARCH/RESNET/modalite/U-Net/CV/baseline
> sonuçları `artifacts/` altında; özet `artifacts/summary.md|csv|tex` + grafikler. Detay §2/K6, K12.
>
> **Durum (23 Eki 2026): v3 tamamlandı (K17, K18);** analiz `artifacts_v3/analysis/` (analyze_preds.py).
> **(16 Eki 2026): v3 hazır.** Etiket politikası "verify" (`splits/split_v2_verify.npz`,
> `splits/cv5_verify_*`), hızlı yükleyici, örnek bazlı tahmin kaydı. Koşular yerel PC (CPU) + Colab (ResNet-CIFAR),
> çıktı `artifacts_v3/`. Detay §2/K16, komutlar §3.4.

### Veri
- Uydu grid'i: lon 25.01–44.99 (0.02°), lat 42.995–35.005 (0.01°) → **800×1000**. Piksel ≈ **1,73 km (D–B) × 1,11 km (K–G)** (39°K), 33×33 yama ≈ **57 × 37 km**
  ("1 km kabulü" yanlıştı; mesafe hesapları `grid_utils.to_km` ile zaten km cinsinden — yeniden koşu gerekmez, bkz. K13).
- Dönem: Mar–Nis 2025, Ara 2025 – Haz 2026 (Tem–Kas verisi yok).
- 17 kanal = 16 FCI + 1 radar; + 5 aux = **22 kanal**. VIS kanalları gece ~0 (NaN değil).
- Radar PNG'lerinde dBZ değeri **mavi kanala** yazılı.
- AWOS `SAAT` alanı **UTC**; `DAKIKA_BASLANGICI` 10 dk periyodun **başlangıcı**.
- METAR `BKN` → 5 okta.
- Etiket kaynağı: 147 insanlı sinoptik istasyon + 79 havaalanı (METAR). Yardımcı alanlar
  1982 otomatik istasyondan (AWOS) IDW ile.
- DEM: **gerçek Copernicus GLO-90** verisiyle üretildi → `data/dem_800x1000.npy` (artık IDW
  yaklaşıklığı değil; eski uyarı kapandı).

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
- v2 koşu çıktıları: `artifacts/*.json` (sonuç), `artifacts/summary.*` (toplu rapor),
  `artifacts/checkpoints/*.pth` (en iyi ağırlıklar), `artifacts/logs/*.log`,
  `artifacts/run_manifest.json`, `artifacts/status/*.done` + `artifacts/status/arch_model.txt`.

---

## 2. Önemli kararlar (karar günlüğü)

### K1 — Nokta skalerleri yerine **raster yardımcı (aux) kanallar** (15 Eyl 2026)
Geç füzyon 3 skaleri yalnız etiketli 226 istasyonun kendi AWOS'undan alıyordu; 1982
istasyonun yoğunluğu kullanılmıyordu ve skaler girdi **tam harita üretimine izin vermiyor**.
Karar: t2m, rh2m (IDW), elev/DEM, cos_sza, radar_cov → `aux[N,5,33,33]` float16 olarak H5'e
yazılır, `patch` ile kanal ekseninde birleşir → **17 + 5 = 22 kanal**.
`*_LateFusion` ve `CloudUNet_Fusion` 12 Eki 2026'da arşive taşındı (v2 verisiyle çalışmıyorlardı; bkz. K13).

> **5 Eki 2026 güncellemesi (bkz. K12):** v2 sonuçları AWOS T/RH raster kanallarının (t2m, rh2m)
> test_time'da ölçülebilir bir katkısı olmadığını, hatta hafif düşürdüğünü gösterdi. K1'in temel
> gerekçesi (tam harita çıkarımı) hâlâ geçerli, ama "AWOS yoğunluğu ablasyonda da kazandırır"
> varsayımı **doğrulanmadı** — makalede nötr/negatif bulgu olarak raporlanmalı.

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
- Çoğunluk sınıfı oranı (test_time) ≈ %54.3 cloudy (`majority_acc`); test_station %71.1,
  test_both %58.4 — bölmeler dengesiz, **BalAcc/MCC her zaman accuracy ile birlikte okunmalı.**

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

### K6 — Mimari ablasyonu ve "hangi model en iyi?" — **v2 sonuçlarıyla güncellendi (5 Eki 2026)**
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

**v2 ARCH sonucu (3 seed ortalaması, val balanced accuracy — `artifacts/ARCH_s1..3.json`):**

| Mimari | val BalAcc (ort.) | s.s. |
|---|---|---|
| **Medium_Flat** | **0.9045** | 0.0004 |
| Deep_Flat | 0.9032 | 0.0007 |
| Deep_GAP | 0.9028 | 0.0004 |
| Medium_GAP | 0.9024 | 0.0011 |
| Shallow_Flat | 0.9004 | 0.0009 |
| Shallow_GAP | 0.8951 | 0.0010 |

→ **Medium_Flat kazandı**, `artifacts/status/arch_model.txt` bunu doğruluyor; v1'deki
fiyat/performans sezgisini (Flatten > GAP, derinlik marjinal) v2'de de teyit ediyor. Bütün
modalite/CV/baseline koşuları bu mimariyle yapıldı (bkz. K7, K12).

> v1 (arşiv) tablosu — `archive_v1/artifacts/ablation_results.json`, eski etiket/bölme, eski
> F1 metriği — artık yalnız tarihsel referans; **güncel karşılaştırma için K12'ye bakın.**
> Eski `unet_results.json` da 1 epoch/20 batch dry-run'dır, sonuç değildir.

### K7 — Kazanan mimari otomatik seçilir
`run_all.py --arch_model auto` (varsayılan) `ARCH_*.json` dosyalarını okur, her varyantın
seed'ler arası **ortalama val balanced accuracy**'sini hesaplar, en iyisini modalite/CV/U-Net
koşularında kullanır ve seçim tablosunu basar. Mimari ablasyonu koşulmadıysa net hata verip durur.
Sonradan başka bir mimari zorlanırsa (`--arch_model Medium_Flat`) eski mimariyle tamamlanmış
koşuları fark edip hangi `.done` işaretlerinin silinmesi gerektiğini söyler — **sessizce karışık
sonuç üretmez.** (5 Eki koşusunda seçilen: **Medium_Flat**.)

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
`<outdir>/run_manifest.json`. **5 Eki 2026'da tüm matris (22 koşu) bu altyapıyla sorunsuz
tamamlandı** — `run_manifest.json`'da 22 kayıt, hepsi `.done`.

> Dikkat: `train.py` sonuç JSON'unda **adı zaten bulunan** deneyi atlar. Yeniden koşmak için
> ya yeni bir `--results` dosyası verin ya da eski JSON'u silin (`run_all.py --force` da var).

### K9 — Karşılaştırma modelleri (baselines)
`src/baselines.py`: majority, `threshold_ir_105` (tek kanal eşiği), `threshold2_ir_105+vis_06`
(çift eşik), `logreg_aux`, `rf_aux`, `hgb_aux` — **aynı bölme, aynı metrikler, aynı JSON şeması**
(tek tabloda birleşir). Eşikler **yalnız train** üzerinde optimize edilir, test'e bakılmaz.
v2 sonucu: ortalama BalAcc 0.806 ± 0.166 (test_time) — modeller arası büyük fark var, bkz. K12.

### K10 — Depo / yayın düzeni (17–18 Eyl 2026)
- GitHub: `https://github.com/ysnre/MTGCLM.git`, MIT lisans, `README.md`, `CITATION.cff`.
- `.gitignore`: veri (`*.h5 *.nc *.npy *.tif`, `data/raw_*`, `data/processed/`, `data/dem/`),
  ağırlıklar (`*.pth`), `scratch/`, `sonuç1/`, `graphify-out/`, `.claude/`, PNG çıktıları **dışarıda**;
  **`splits/*.npz|json` ve `artifacts/*.json` PAYLAŞILIR.**
- v1 sonuçları/ağırlıkları `archive_v1/` altına taşındı.
- Veri bulunabilirliği: MTG FCI → EUMETSAT Data Store (açık); radar + yer gözlemleri MGM'ye ait,
  paylaşılmıyor. Türetilmiş 19 GB H5 → **Zenodo** (DOI bekleniyor). ~2.000 yamalık örneklem
  `src/make_public_sample.py` ile üretilir (henüz çalıştırılmadı).
- DEM kaynağı: **Copernicus DEM GLO-90** (açık, artık yerelde gerçek veriyle üretildi).
- Not: yerel repoda şu an **tek commit** var ("İlk commit", 1 Eki 2026) — önceki ayrıntılı commit
  geçmişi bu squash ile tek işleme indirildi; `git log` geçmişine güvenmeyin, bu dosyaya güvenin.

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
  **(16 Eki güncellemesi, K16):** epoch'un ~3 dk'sı GPU değil örnek-örnek H5 okumasıydı. v3'te eğitim yerel PC'de
  (CPU, bölme RAM'de) + Colab'da (yalnız ResNet-CIFAR); `train*.py` varsayılanı her platformda 15 epoch / tüm veri.

### K12 — **v2 koşu matrisi sonuçları ve bulgular (5 Eki 2026)** — `artifacts/summary.md|csv|tex`

Koşu matrisi (hepsi Medium_Flat mimarisiyle, `splits/split_v1_compact.npz`, aksi belirtilmedikçe
test_time üzerinde):

| Deney | Ne test ediyor | aux | aux_channels |
|---|---|---|---|
| `A_sat_s{1,2,3}` | sadece uydu+radar (17 kanal) | hayır | — |
| `B_full_s{1,2,3}` | + tüm aux (22 kanal, AWOS t2m/rh2m dahil) | evet | tümü |
| `C_noawos_s{1,2,3}` | + aux ama **AWOS'suz** (cos_sza, elev, radar_cov) | evet | `cos_sza,elev,radar_cov` |
| `D_unet_s{1,2,3}` | `MTGUNet`, tam aux | evet | tümü |
| `RESNET_s1` | `ResNet18`, tam aux | evet | tümü |
| `CV_f0..4_s1` | Medium_Flat+aux, bloklu 5-katlı CV | evet | tümü |
| `Z_baselines_s1` | klasik modeller (K9) | evet | tümü |

**test_time (görülmemiş haftalar, n=51.704, majority_acc 0.543):**

| Deney | BalAcc | MCC | F1(cloudy) | Parametre |
|---|---|---|---|---|
| A_sat (uydu+radar) | 0.935 ± 0.001 | 0.868 ± 0.003 | 0.938 ± 0.002 | 228.994 |
| B_full (+AWOS aux) | 0.933 ± 0.001 | 0.864 ± 0.003 | 0.936 ± 0.001 | 230.434 |
| C_noawos (+aux, AWOS'suz) | **0.937 ± 0.001** | **0.872 ± 0.002** | 0.939 ± 0.001 | 229.858 |
| D_unet | 0.936 ± 0.001 | 0.869 ± 0.002 | 0.939 ± 0.001 | 473.314 |
| ResNet18 | 0.936 | 0.870 | 0.940 | 11.237.122 |
| Z_baselines (ort. 6 model) | 0.806 ± 0.166 | 0.611 ± 0.331 | 0.845 ± 0.098 | — |

**test_station (görülmemiş 30 istasyon, n=49.228, majority_acc 0.711):**
A_sat 0.918 | B_full 0.918 | C_noawos 0.918 | **D_unet 0.923** (en iyi) | ResNet18 0.914

**test_both (n=11.655, majority_acc 0.584):**
A_sat 0.946 | B_full 0.942 | **C_noawos 0.946** | D_unet 0.943 | ResNet18 0.939

**Bloklu 5-katlı CV (test_time, kat ortalaması):** BalAcc 0.913 ± 0.021, MCC 0.812 ± 0.040 —
tek-bölmeli test_time'dan (0.933–0.937) **daha düşük ve daha değişken**; katlar arası majority_acc
%36–74 arasında salınıyor. **Tekli bölme sonucu tek başına yeterli değil, CV farkı makalede
raporlanmalı.**

**Önemli bulgu — AWOS raster aux'unun katkısı belirsiz/negatif:** test_time'da B_full (AWOS dahil)
A_sat'tan (hiç aux yok) ve C_noawos'tan (AWOS'suz aux) **daha düşük**; test_station'da üçü eşit;
test_both'ta C_noawos en iyi. **AWOS T/RH raster kanalının modele ölçülebilir bir katkısı
görünmüyor** — K1'in "1982 istasyonun yoğunluğu ablasyonda da kazandırır" beklentisi test
kümelerinde doğrulanmadı (val'de hafif kazanç var: B_full val BalAcc 0.904 > A_sat 0.893 —
olası val/test tutarsızlığı veya overfitting; makalede dikkatle tartışılmalı).

**U-Net (segmentasyon) sınıflandırma modelleriyle rekabetçi**, test_station'da en iyisi — tam
harita hedefi için umut verici bir sinyal.

**Baseline'lar (Z_baselines) çok daha düşük ve yüksek varyanslı** (model başına BalAcc 0.6–0.95
arası dağılıyor) — derin modellerin klasik yaklaşımlara üstünlüğü net; ayrıntı için
`artifacts/Z_baselines_s1.json`'daki tek tek model satırlarına bakın (özet tabloda ortalanmış).

`_check.json` (`Medium_Flat`, aux, 1 epoch/50 batch, n=3.200) **yerel sağlık kontrolü**dür,
Colab'a çıkmadan önceki dry-run — gerçek sonuç olarak sayılmaz.

**Parçalı bulut tutarlılığı (test_partial, etiketsiz 1–4 okta, P(cloudy) ortalaması):** tüm
deneylerde okta ile **monotonik artış** var (okta1 ≈0.10–0.13 → okta4 ≈0.54–0.62) — modelin ara
bulut yoğunluğunu da tutarlı sıraladığının dolaylı kanıtı.

### K13 — 12 Eki 2026 kararları ve düzeltmeler
- **Grid:** 0,02°×0,01° ≈ 1,73×1,11 km; yama ≈ 57×37 km. Makalede "1 km" yazılmayacak. Yeniden koşu gerekmez.
- **Paralaks düzeltilmemiş:** `coknc.py` satpy'yi yalnız `fci_l1c_nc` okuma + `resample` için kullanıyor; satpy
  paralaks düzeltmesini varsayılan olarak uygulamaz (CTH + açıkça tanımlanmış modifier gerekir). Girdiler ve CLM
  düzeltmesiz geometride; CLM karşılaştırması merkez piksele ek olarak alan bazlı yapılacak.
- **v1 füzyon modelleri arşivlendi:** `CloudUNet_Fusion`, `MTGConvNet_LateFusion`, `ResNet18_LateFusion` →
  `src/_backup_20260916/model_v1_fusion_classes.py`; ağırlıkları → `archive_v1/v1_fusion_weights/`.
  Late fusion ablasyonu gerekirse yeni bir MLP/FiLM sınıfı yazılır (doğrudan birleştirme ölçek dengesizliği taşır).
- **ResNet18 iki giriş katmanıyla (danışman talebi: model karşılaştırması):** `MTGResNet18(stem="imagenet"|"cifar")`.
  Deneyler `ResNet18` (ImageNet stem; RESNET_s1 sonucu korunur, 11.237.122 parametre) ve `ResNet18_CIFAR`
  (3×3 adım 1, maxpool yok; 11.180.802 parametre). `run_all.py --stage resnet` → `RESNET_CIFAR_s{1,2,3}`;
  `--resnet_imagenet_seeds` varsayılanı `1,2,3` → ImageNet stem'e de 2 seed eklenir (s1 zaten var); iki stem ± ile raporlanır.
- **Koşuya özgü checkpoint adları:** `checkpoints/best_model_<results-dosyası>__<deney>.pth` (train.py, train_unet.py);
  JSON `config.best_model_file` alanını taşır. 18–20 Eyl koşularının ağırlıkları geri gelmez (her adın yalnız son koşusu var).
- **FCI CLM:** `F:/mtg_fci_clm`, 28.104 dosya. Mar–Nis 2025 dosyaları `C` (commissioning / ön-operasyonel),
  Ara 2025–Haz 2026 `O` (operasyonel) → ayrı raporlanacak. Dizi yönü: satır 0 güney, sütun 0 batı (netCDF `x`
  işareti pyproj geos ile ters; indeksler satpy alanıyla hesaplanır, gün/gece deseniyle doğrulandı). İstasyonlarda
  CLM pikseli ≈ 2,95×3,56 km, VZA 51,6–64,7°. `src/extract_fci_clm.py` + `data/clm_station_index.npz` →
  `data/processed/clm_station/clm_YYYYMMDD.npz` (merkez, 3×3, r5/r10/r20/r30 km sınıf sayımları).
  "Cloud contaminated" sınıfı çok sık (tek sahnede Türkiye kutusunun %30–40'ı) → ikili eşleme duyarlılığı raporlanacak.
- **NWC SAF Cloud Type (MGM, PNG):** `F:/nwcsaf`, 15 dk, MSG tabanlı. 12 renkli kayıpsız PNG; bulutsuz kara/deniz
  beyaz (birleşik). Mercator; köşeler (MGM, 14 Eki): sol-üst 49.203243 K 12.497211 D, sağ-alt 31.269161 K 46.958802 D
  (CLM ile bağımsız tahminimizle ±0,1° uyumlu). Çıkarım: `src/extract_nwcsaf_ct.py` → `data/processed/nwcsaf_station/`.
- **K12 düzeltmeleri:** baseline satırı (0,806) majority/eşik ile ortalanmış — `hgb_aux` (0,932 / 0,924 / 0,942)
  CNN'lerle başa baş; "derin modellerin üstünlüğü net" ifadesi geçersiz. METAR "açık" etiketlerinin tamamı NSC/CAVOK
  (SKC kodu ICAO'da 2008'den beri yok); SYNOP ile çakışan çiftlerde %13,3'ü N≥5. Ayrıntı: proje dokümanları (5–12 Eki).

### K14 — 14 Eki 2026: referans ürünler aynı örneklerde, METAR doğrulaması, AWOS bilgisi
- Kompakt H5'te **132 istasyon** var (CLAUDE.md'deki 226 = etiket kaynağı istasyon sayısı); METAR-only 32 havaalanının
  örneği yok. Etiketlerin %55'i METAR; **açık etiketlerin %61–68'i METAR (NSC/CAVOK)**. Okta-9 payı %0,2–0,3 (önemsiz).
- **Referanslar aynı split örneklerinde** (`src/eval_clm_on_splits.py`, `src/eval_nwc_and_metar.py`; eşik/alan yalnız
  train'de): test_time BalAcc — CLM merkez 0,869 / r30 0,884; NWC CT merkez 0,878 / r30 0,893; modellerimiz 0,933–0,937.
  test_station: CLM 0,841/0,853, NWC 0,824/0,865, modeller 0,918–0,923. val (zor haftalar): CLM 0,785/0,801, NWC 0,797/0,810.
  Yalnız SYNOP etiketlerinde referanslar belirgin daha iyi (test_time CLM r30 0,904, NWC r30 0,919) → adil kıyas SYNOP-only.
- **METAR NSC doğrulaması:** havaalanlarında SYNOP 3 saatlik → METAR-açıkların ~%67'si aynı saatte SYNOP'suz. Doğrulanabilenlerde
  N=0 %57, N1–4 %27, N5–8 %16. Doğrulanmış NSC'ler referanslarla SYNOP açıkları kadar temiz; "NSC ama SYNOP bulutlu"
  vakaların NWC tipi çoğunlukla yüksek yarı saydam (sirrüs) + yüksek opak.
- **AWOS T2m bilgisi gerçek:** tek kanal AUC (SYNOP) BT10.5 0,893 → BT10.5−T2m 0,932; HGB'de (100k alt küme) t2m+rh2m
  her alt kümede +0,013…+0,030 BalAcc (gece daha fazla). CNN'de B−C ≈ 0 → CNN bilgiyi uydu bağlamından zaten çıkarıyor
  olabilir ya da kullanamıyor; CV + SYNOP-only ile incelenecek.
- `run_all.py --resnet_imagenet_seeds` varsayılanı `1,2,3`.

### K15 — 16 Eki 2026: insan-okur infografik ve 3 çalışma kuralı
- `docs/mtgclm_infografik.html`: projeyi tek sayfada özetleyen, bağımsız (harici bağımlılıksız), açık/koyu temalı
  HTML infografik. Kaynak sayılar `artifacts/summary.md`, `clm_on_splits.json`, `nwc_metar_on_splits.json`'dan.
- `.agent/rules/` altına 3 kalıcı çalışma kuralı yazıldı: (1) dosya başına İngilizce commit komutları üretme,
  (2) CLAUDE.md güncel tutma, (3) infografiği her değişiklikte güncelleme. (`.agent/` .gitignore'da, yerelde durur.)

### K16 — 16 Eki 2026: v3 hazırlığı — etiket politikası "verify", hızlı yükleyici, tahmin kaydı, yerel PC
- **Etiket politikası (kullanıcı kararı): yalnız doğrulanmış NSC.** METAR "açık" (gözlem dakikası ≠ 0, etiket 0) yalnız
  aynı istasyonun aynı (yuvarlanmış) saatteki SYNOP'u N=0 ise tutulur; SYNOP'un tamamı ve METAR bulutlu aynen kalır.
  `src/make_label_policy.py` → `splits/split_v2_verify.npz` + `splits/cv5_verify_fold{0..4}.npz` (+ `.json` istatistik).
  **Kompakt H5 değişmedi**: politika yalnız örnek çıkarır, etiket değiştirmez → K4'teki "etiket → bölme → H5" yeniden
  üretimi gerekmedi. METAR açık 82.650 → 16.808 tutuldu. train 218.648 → **179.326** (açık %33,8 → %19,3), val 52.033 → 43.152,
  test_time 51.704 → 40.868, test_station 49.228 → 43.865, test_both 11.655 → 10.215, test_partial aynen 16.941.
- **"Colab GPU %1" kök nedeni:** `MTGH5Dataset.__getitem__` örnek-örnek okuma (0,88 ms/örnek ≈ 3,2 dk/epoch yalnız okuma) +
  `num_workers=0`. Çözüm `src/fast_loader.py`, `--loader memory|stream|auto|legacy` (varsayılan auto). Bölme indeksleri
  artan sıralı → batch'ler eski yolla **birebir aynı örneklerden** (mini H5'te tensör eşitliği test edildi); yalnız
  çevirme RNG'si artık (seed, epoch)'a bağlı `torch.Generator`. `--cache_dir`: bölme satırlarının `.npy` önbelleği
  (~16 GB, bir kez yazılır; Colab'da şart, yerelde NVMe'ye isteğe bağlı).
- **Örnek bazlı tahmin kaydı:** `<outdir>/preds/<koşu>__<deney>.npz` — val + test_* için satır, etiket, olasılık (en iyi val
  modeli), `last_<küme>_prob` (son epoch), test_partial. Sonuç JSON'unda yeni anahtarlar: `test.<küme>_synop` (yalnız SYNOP,
  referanslarla adil kıyas), `test.<küme>_last`, `test.val_best(_synop)`; `config.env` (git commit + dirty, torch, cihaz,
  iş parçacığı, yükleyici), `config.label_policy`, `config.preds_file`, `config.wall_minutes`; `history.epoch_time`.
- **Tekrarlanabilirlik:** `src/run_utils.py` → seed'ler, `cudnn.deterministic`, `use_deterministic_algorithms(warn_only)`,
  `CUBLAS_WORKSPACE_CONFIG`; checkpoint RNG durumlarını da saklar → kesintiye uğrayıp devam eden koşu kesintisizle
  birebir aynı (CPU'da doğrulandı). Kapatmak için `--no_deterministic`.
- **Sessiz varsayılan düzeltildi:** `train.py`/`train_unet.py` yerelde (Colab dışı) sessizce 3 epoch/30 batch (U-Net 2/20)
  koşuyordu. Artık her platformda 15 epoch / tüm veri; hızlı deneme `--epochs 1 --limit_batches 50`.
- **run_all v3:** varsayılan split `split_v2_verify`, CV `cv5_verify`, mimari `medium_flat` (K6 kazananı; `auto` hâlâ var),
  `--stage v3` = modality, unet, resnet, cv, baselines (arch hariç). Modaliteler A_sat, B_full, C_noawos ve yeni **C_t2m**
  (C + yalnız t2m, K14). CV artık 4 modalite × 5 kat: `CV_<A|B|C|Ct2m>_f<k>_s1`. Baselines tüm veriyle (`--max_train 0
  --max_eval 0`, blok okuma, SYNOP-only + tahmin kaydı). Süzgeç `--only/--skip` (ön ek), `--threads`, `--loader`,
  `--cache_dir` (CV'ye yalnız `--cache_cv` ile). `aggregate_results.py` `*_synop` alt kümelerini de tablolar.
- **Donanım (kullanıcı, 16 Eki):** i7-12700, 31,6 GB RAM, **NVIDIA GPU yok** (Intel UHD 770 eğitimde kullanılmaz), NVMe 1 TB +
  USB 3.0 4 TB disk, conda base Python 3.11.5. **Plan:** Medium_Flat modalite + CV, U-Net, ResNet-ImageNet ve baselines yerel
  CPU'da; ResNet-CIFAR Colab T4'te. Çıktı `artifacts_v3/` (v2 `artifacts/` karşılaştırma için korunur).
- **Hız:** `src/bench_speed.py` (sentetik batch, veri gerekmez). Bulut ortamının 2 çekirdeğinde koşu başına: Medium_Flat ~76 dk,
  U-Net ~6,3 sa, ResNet-ImageNet ~5,6 sa, ResNet-CIFAR ~26 sa.
  **i7-12700 ölçümü (20 Eki, `bench_speed.py`, dk/koşu, 8/12/20 iş parçacığı):** Medium_Flat 25/24/25, U-Net 162/129/147,
  ResNet-ImageNet 258/252/265, ResNet-CIFAR 862/665/624 → **`--threads 12`** (fiziksel çekirdek sayısı; 20 yalnız CIFAR'da
  hafif hızlı). **Güncel iş bölümü:** yerel CPU = modalite (12 × ~24 dk) + CV (20 × ~29 dk) + baselines ≈ 15 sa;
  Colab T4 = U-Net × 3 + ResNet-ImageNet × 3 + ResNet-CIFAR × 3 (`--stage unet,resnet`).
- **1 epoch gerçek deneme (20 Eki):** boş RAM yalnız 6,4 GB → auto `stream` seçti; H5 USB diskten akınca epoch 372 s (hesap ~90 s,
  geri kalanı her epoch tüm dosyayı iki kez okumak, ~140 MB/s). Çözüm: **H5'i NVMe'ye kopyala** (`C:/mtg/mtg_train_v2.h5`, `--h5` ile ver)
  — CV katları dahil her bölme için çalışır; ek önbellek gerekmez. (1 epoch skoru, verify etiketleri: val 0,938, test_time 0,953 —
  v2 sayılarıyla doğrudan kıyaslanamaz, etiket kümesi farklı.)
- **NVMe ile tekrar (22 Eki):** boş RAM 18,2 GB → train+val RAM'e (10,7 GB, 38 s, ~504 MB/s), testler NVMe'den akış.
  **Epoch 102 s** (USB akışında 372 s). Kayıp/metrikler USB akışlı koşuyla birebir aynı → memory/stream eşdeğerliği gerçek
  veride de doğrulandı. Medium_Flat koşusu ≈ 26 dk.
- **Ortam (20 Eki):** Windows'ta `python` conda dışındaki Python 3.14'ü buluyordu → torch import'ta çöküyordu. Çözüm: ayrı
  conda ortamı `mtgclm` (Python 3.11, torch CPU); her pencerede `conda activate mtgclm`.
- Ayrıntı: proje dokümanı `claude/mtgclm_run_hazirlik_2026-10-16.md`.

### K17 — 23 Eki 2026: v3 koşuları — Windows kodlama hatası ve ön sonuçlar
- **Durum:** Colab 9/9 tamam (U-Net, ResNet-ImageNet, ResNet-CIFAR × 3; T4; U-Net ~39 dk, ResNet ~34 dk, CIFAR ~89 dk/koşu).
  Yerel 33 koşunun **28'i eğitimi (15 epoch) bitirdikten sonra çöktü**: `run_all` çıktıyı boruya aldığında Windows kodlaması cp1254
  oluyor, `test_partial` satırındaki "p̄" karakteri yazılamıyordu → JSON ve preds yazılmadı. test_partial'ı boş olan CV fold 3 ve
  baselines etkilenmedi. **Eğitim kaybı yok:** checkpoint'ler 15. epoch'ta duruyor; düzeltilmiş kodla aynı komut yalnız
  değerlendirmeyi yapar (sonuçlar kesintisiz koşuyla birebir aynı — container'da cp1254 ile yeniden üretilip doğrulandı).
- **Düzeltme:** `train*.py` ve `baselines.py` stdout/stderr'i `errors="backslashreplace"` ile açar; "p̄" → "p_ort"; `run_all.py`
  alt süreçleri `PYTHONIOENCODING=utf-8` ile başlatıp UTF-8 okur; ✓/✗ → [OK]/[HATA].
- **Ön sonuçlar (log'lardan; değerlendirme deterministik, tekrar koşunca aynı çıkacak), verify etiketleri, test_time BalAcc:**
  A_sat 0,958, B_full 0,956, C_noawos 0,959, C_t2m 0,958, U-Net 0,957, ResNet-IN 0,958, ResNet-CIFAR 0,957 (s.s. ≤ 0,002);
  **hgb_aux 0,959**, rf 0,954, logreg 0,948, eşikler 0,78–0,80. Yalnız-SYNOP test_time: modeller 0,954–0,955 (CLM r30 0,904,
  NWC r30 0,919 aynı SYNOP satırlarında — verify yalnız METAR satırı çıkardığı için SYNOP alt kümesi v2 ile aynı).
  test_station: tüm derin modeller 0,937–0,941; HGB 0,940. → K12'deki tablo deseni korunuyor: ~0,96 tavan, HGB ≈ CNN.
- **AWOS bulgusu CV'de netleşiyor:** kat bazında eşleştirilmiş farklar (test_time BalAcc) B−C +0,0013/+0,0044/+0,0005/+0,0050/+0,0013
  (5/5 pozitif, ort. +0,0025); C_t2m−C 5/5 pozitif (ort. +0,0011); B−A ort. +0,0028 (4/5). Tek bölmede B < A görünmesi (K12) küçük
  test kümesine özgü olabilir. Örnek bazlı McNemar / blok bootstrap ile sınanacak (`analyze_preds.py`).

### K18 — 23 Eki 2026: v3 tamamlandı — anlamlılık analizi (`src/analyze_preds.py`)
- Yerel değerlendirme tekrarı 28/28 başarılı (1–5 dk/koşu); 41 koşu + baselines: hepsi 15 epoch, `label_policy=verify`, preds dosyası var.
  Colab koşuları git `4a69c7b` (kod a29b27b ile aynı), yerel eğitim `b68d201`, değerlendirme tekrarı `e2ff6d2` (yalnız çıktı düzeltmesi).
- `src/analyze_preds.py`: örnek bazlı tahminlerden ana tablo (BalAcc/MCC ± s.s., %95 blok-bootstrap GA, Brier, ECE, son epoch),
  eşleştirilmiş farklar (gün blokları; test_station istasyon blokları), CV kat + birleşik farklar, kırılımlar (gündüz/alacakaranlık/
  gece — güneş yüksekliği istasyon konumundan; SYNOP/METAR; yükseklik; ay; okta başına yakalama) ve `--refs` ile CLM/NWC'ye karşı
  aynı kapsanan örneklerde kıyas (eşik yalnız train'de: CLM r30 ≥ 0,45, NWC r30 ≥ 0,50). Yalnız numpy + pandas; ~20 s.
  McNemar yalnız CSV'de (kümelenme nedeniyle aşırı iyimser); çıkarım blok-bootstrap'a dayanır.
- **Bulgular (test_time, verify):** tüm derin modeller 0,957–0,959, HGB 0,959; model çiftleri arası farkların GA'ları sıfırı içeriyor
  (HGB − B_full +0,003 [+0,000, +0,006] sınırda). test_station 0,937–0,941. Gece (BalAcc ~0,94) gündüzden (~0,97) zor.
- **AWOS:** CV birleşik B − C +0,0027 [+0,0015, +0,0041] (5/5 kat), C_t2m − C +0,0013 [+0,0004, +0,0022] (5/5), SYNOP-only de
  aynı yönde; test_station'da B − C +0,0031 [+0,0009, +0,0059]. Tek bölmenin test_time'ında fark yok. → K1/K12'deki "AWOS katkısı
  yok" yargısı **düzeltildi**: küçük ama tutarlı pozitif katkı (özellikle CV ve görülmemiş istasyonlar).
- **Referanslar (yalnız SYNOP, aynı örnekler):** test_time model 0,953–0,955 vs CLM r30 0,908, NWC r30 0,923 (NWC kapsamında model
  0,947–0,951; fark +0,023…+0,026, GA sıfırı dışlıyor); test_station model 0,932–0,936 vs NWC r30 0,903, CLM r30 0,896.

### K19 — 30 Eki 2026: kod haritası (ikinci infografik)
- `docs/mtgclm_kod_haritasi.html`: bütün `src/*.py` ve `notebooks/*.ipynb` dosyalarının, sınıflarının, metotlarının ve
  fonksiyonlarının birbirine bağlantısı. Üç görünüm: **Modül haritası** (iş akışı sütunları; import, fonksiyon çağrısı,
  alt süreçle betik çalıştırma, veri okur/yazar okları), **Fonksiyon ağı** (seçilen fonksiyonu çağıranlar / çağırdıkları),
  **Dizin** (aranabilir, sıralanabilir tüm sınıf ve fonksiyonlar). Harici bağımlılık yok, açık/koyu tema.
- **Otomatik üretilir:** `python src/make_code_map.py` (stdlib `ast`; ~1 sn). Modül grupları, açıklamalar ve "eski" işaretleri
  `MODULE_INFO`'da; dizeden çıkarılamayan veri bağlantıları `EXTRA_DATA`'da. Elle düzenlenmez.
- Kural 3 genişletildi: kod değişikliğinden sonra kod haritası yeniden üretilir ve ayrı commit edilir.
- İlk üretim: 52 dosya, 14 sınıf, 252 fonksiyon/metot, 297 çağrı bağlantısı. Statik çözümleme sınırı: nesne üzerinden
  dolaylı çağrılar (ör. `model(x)`, `loader.dataset`) görünmez; veri okur/yazar yönü sezgiseldir.

### K20 — 30 Eki 2026: makale hedefi, şekil kuralı, yazım kararları
- **Hedef dergiler:** Theoretical and Applied Climatology (TAAC, birincil) ve Meteorology and Atmospheric Physics
  (MAP, yedek); ikisi de Springer Wien, hibrit (abonelik yolu ücretsiz). Tek makale gövdesi, dergiye özel kısımlar
  LaTeX'te `\ifTAAC` anahtarıyla. TAAC için bulut sıklığı haritaları zorunlu; MAP için kanal önem analizi zorunlu.
- **Ana model:** tam harita ürünü **U-Net** (`D_unet`); nokta tablolarında **CNN, tüm girdiler** (`B_full`) odak satır.
- **Zaman:** ≥ 1 ay → yol haritasındaki S1–S6'nın tamamı ilk gönderime girer; yalnız P20 (U-Net jitter) opsiyonel.
- **Yazım:** LaTeX (Springer Nature şablonu) + İngiliz yazımı (centre, colour); ondalık ayırıcı nokta.
- **Şekil kuralı:** `docs/paper/sekil_stil_kurali.md` + `paper/figures/springer_fig.py` + `springer_taac_map.mplstyle`;
  Kural 4 (`.agent/rules/rule_figure_style.md`). Demo: referans kıyası şekli (yalnız SYNOP, kontrol raporu OK).
- **Makale dokümanları:** `docs/paper/` (yol haritası, bölüm iş listesi, eksik promptlar P0–P20, literatür).
- **Veri politikası (araştırma notu):** WMO Unified Data Policy (Res. 1, Cg-Ext 2021) çekirdek (core) veriyi serbest ve
  kısıtsız değişime bağlar; çekirdek yüzey verisi GBON ve WIGOS El Kitabı'nda tanımlanan gözlemlerdir (ör. RBSN
  istasyonlarının ana sinoptik saatleri). 1982 AWOS istasyonunun çoğu ve radar hacim verisi açıkça çekirdek listede
  görünmüyor → büyük olasılıkla "recommended" (koşullu olabilir); türetilmiş veri kaynağının koşulunu taşır.
  Karar: SYNOP/METAR türevleri UDP gereği paylaşılabilir; radar ve AWOS türevleri için MGM'den kısa yazılı teyit
  alınır (Springer gönderiminde kurum onayı zaten beyan ediliyor). Ayrıntı: `docs/paper/makale_yol_haritasi.md` §5.
- **Not:** CLAUDE.md §5'teki "49×49 çıkar → 33×33 kırp" jitter maddesi §7'deki "yamaları yeniden çıkarma" kuralıyla
  çelişiyor; P20 bunun yerine 33×33 içinde 25×25 rastgele kırpma öneriyor.

### K21 — 14 Kas 2026: P11 + P12a — şekil altyapısı doğrulandı, Fig. 1/4/5/6/10 ve Tab. 2–5 (+S1–S2) üretildi
- **P11:** `springer_fig` import OK. `check_figure()` artık çözümlenen yazı tipini de denetler: Arial/Helvetica değilse rapor
  OK vermez (Linux önizlemesinde Liberation Sans → "yalnız önizleme"). **Son sürüm Arial'lı Windows'ta** `python paper/figures/make_all.py` ile.
- **Betikler (`paper/figures/`):** `figdata.py` (ortak yollar/meta; `analyze_preds.load_meta` yeniden kullanılır), `make_fig01_study_area.py`,
  `make_fig04_main_results.py`, `make_fig05_references.py`, `make_fig06_breakdowns.py`, `make_fig10_calibration.py`, `make_tables.py`,
  `make_all.py`. Her şekil `FigXX_caption.txt` üretir; tablolar `paper/tables/*.tex|csv`.
- **Fig. 6:** `breakdowns.csv` aydınlanma × ay çapraz tablosunu içermiyor → betik tahmin dosyalarından **aynı tanımla** hesaplıyor,
  marjinalleri `breakdowns.csv` ile birebir doğruluyor (uyuşmazsa durur) ve `artifacts_v3/analysis/breakdowns_illum_month.csv`'ye yazıyor.
- **Tablo düzeni (yol haritasına göre):** Tab. 3 ana tablo, Tab. 4 modalite farkları (tek bölme + CV birleşik; P1 satırları `\todo`),
  Tab. 5 hesaplama maliyeti (sahne çıkarımı P4'te `\todo`), Tab. S1 mimari çiftleri, Tab. S2 CV kat başına BalAcc.
  **İstisna:** eşleştirilmiş farklar ve seed s.s. 4 ondalık (3 ondalık sıfırı dışlayan GA'ları "±0.000" yapıyordu).
- **Fig. 1:** istasyon türü gözlem dakikasından: 90 sinoptik (yalnız tam saat) + 42 havalimanı (:20/:50 METAR da var) = 132; 30 holdout.
  Radar kapsamı `radar_circle_masks` ile 19 radarın azami menzilinden (nominal; zamanla değişen `radar_cov` değil). Natural Earth
  10 m dosyaları `data/natural_earth/` (GitHub nvkelso aynası; .gitignore'da).
- **Gözlem (P14 için):** Fig. 10a'da bütün modellerin güvenilirlik eğrisi köşegenin **üstünde** — orta olasılıklarda bulutluluğu
  olduğundan düşük tahmin ediyorlar (ECE 0.024–0.047). 1–4 okta örneklerinde ortalama olasılık okta ile monoton artıyor (0.10 → 0.58).
- Doğrulama: Tab. 2 sayıları bölme dosyasından, B_full test_time BalAcc ve bir Fig. 6 hücresi (test_station, gece, Ara 2025: 0.838, n = 565)
  bağımsız kodla yeniden hesaplandı — birebir.

### K22 — 14 Kas 2026: P13 — LaTeX iskeleti (`paper/`), makale klasörleri depo dışında
- **Kullanıcı kararı:** makale yazımıyla ilgili hiçbir şey GitHub'a gitmez; `paper/` ve `docs/paper/` .gitignore'da. Bu klasörlerdeki
  dosyalar için commit komutu üretilmez. (Code availability beyanı buna göre uyarlanacak — `backmatter.tex`'te `\todo`.)
- **Şablon:** Springer Nature `sn-jnl` (Aralık 2024, zip `docs/paper/`'de; açılmış kopya `paper/template_sn/`). **Kaynak stili `sn-basic`
  (author-year):** TAAC kılavuzu "name and year in parentheses", alfabetik liste ve Springer Basic biçimini istiyor (doğrulandı, 14 Kas).
- **Yapı:** `main.tex` (`\ifTAAC`; MAP için `\def\MAPversion{}`, gönderim için `\def\FINALversion{}` → `\todo` gizli, satır no yok),
  `preamble.tex` (ortak makrolar: `\todo`, `\fig{stem}{genişlik}{label}` altyazıyı `figures/<stem>_caption.tex`'ten okur, `\figpending`),
  `frontmatter.tex` (başlık/öz/anahtar kelime dergiye göre; yazar bloğu `declarations_taac.md`'den), `sections/01–06` + `sections/results/*`
  (Bulgular sırası dergiye göre `04_results.tex`'te), `backmatter.tex` (beyanlar), `supplementary.tex` (Tab. S1–S3, ek şekiller).
  Tablo numaraları yol haritasıyla aynı: Tab. 1 kanallar (P14), 2 bölmeler, 3 ana, 4 ablasyon, 5 maliyet (MAP ana metin; TAAC'ta Tab. S3).
- **Araçlar:** `python tools/build.py [--journal taac|map|both] [--final]` (pdflatex + bibtex, latexmk/Perl gerekmez; sayfa, tanımsız
  atıf/etiket, taşan kutu, kalan `\todo` raporu), `python tools/count_chars.py [--journal map]` (100 000 sınır / 80 000 hedef; öz, metin,
  altyazılar, tablolar, beyanlar, kaynaklar), `tools/captions2tex.py` (`_caption.txt` → `.tex`, kalın panel harfi; `make_all.py` sonunda da çalışır).
- **refs.bib:** bağlandı (`\bibliography{../docs/paper/refs}`); 134 girdinin hepsine `verified = {no}` eklendi (yedek `refs.bib.bak_20261114`).
  Derlemeyi bozan veri düzeltildi: Huckle2009 bozuk karakter, μ → `{\textmu}`, Shao2017 `journal = {Remote Sensing}`. T1 fontenc eklendi.
  **Bozuk kayıt kaldı:** `Bechini2007` (başlık dosya adı, dergi yok) → P16'da düzelt/çıkar. Liu2020 konu dışı (bulut bilişim) → P16.
- `make_tables.py` sn-jnl tablo düzenine geçirildi (`\footnotetext`, `\botrule`, iki satırlı başlık); Tab. 4/S1 "çift başına grup, küme başına
  satır" düzeni (metin genişliği 31 pc). Fig. 4/6 altyazılarında panel harfinden sonra büyük harf (betik + `.txt` birlikte düzeltildi).
- İlk derleme: TAAC 12 + 3 sayfa, MAP 13 + 2 sayfa; tanımsız referans 0, taşan kutu 0; 58 `\todo`. Şekil dosya adları (FigXX) sabit, LaTeX
  numarası dergiye göre değişir (TAAC: sahne 7, sıklık 8; MAP: kanal önemi 6 … kalibrasyon 9) → gönderimde dosyalar yeniden adlandırılır.

### K23 — 15 Kas 2026: P14 — Bölüm 2–4 taslakları, sayı denetimi, **kanal sırası sorunu**
- `paper/sections/02_data.tex` (section2 md'den dönüştürüldü), `03_methods.tex`, `results/*.tex` yazıldı. Her sayı satır sonunda
  `% src: <csv>[anahtarlar] <sütun>` ile kaynağına bağlı (satır no yerine anahtar: P1 satır ekleyince kaymaz); `% ref:` = tasarım sabiti.
  **`python paper/tools/verify_numbers.py`** bütün kaynakları CSV hücreleriyle karşılaştırır (ilk çalıştırmada 2 yanlış GA'yı yakaladı);
  şu an 178 değer, 0 hata. Ek kaynak CSV'ler `paper/tools/derived_facts.py` → `artifacts_v3/analysis/{dataset_facts,cv_summary,
  okta_consistency,baselines_extra}.csv`.
- **Bölüm 2 düzeltmeleri (md'deki eski bilgiler):** METAR payı %55 → **%43,8**, açıkların METAR payı %61–68 → **%17,0–28,1**
  (K14 sayıları verify öncesi etiket kümesindendi); okta-9 %0,2–0,3 → %0,35; "altı güneş kanalı" → **sekiz** (5 VIS + 3 NIR gece ~0);
  radar "5 dk döngü" → uydu zamanının ±5 dk'sı içindeki tekil radar görüntülerinin piksel bazında maksimumu, `radar_cov` etkin
  radarlar üzerinde maksimum; bölme dışındaki 65 842 satır "kopya" değil, **doğrulanmamış METAR açık** raporları; 132 istasyon
  (90 yalnız SYNOP + 42 METAR'lı havalimanı) eklendi. md dosyası değiştirilmedi.
- **Kanal sırası (açık sorun):** `create_patch_dataset.get_channel_names()` sırayı İLK NetCDF'in h5py anahtar sırasından alıyor; sıra
  dosyadan dosyaya değişiyor (`data/raw_mtg`'de 3 farklı sıra). Kanallar adla okunduğu için **veri tutarlı**, ama sıra kaydedilmemiş.
  `baselines.CHANNEL_NAMES` başka bir dosyanın sırasını varsayıyor ve gece istatistikleriyle çelişiyor ("vis_06" = kanal 13 gece ~249 K).
  → Eşik modellerinin kanal adları (Fig. 4 altyazısı "10.5 µm / 0.6 µm") ve **P3 kanal önemi** doğrulanana kadar güvenilmez.
  Doğrulama: `python src\verify_channel_order.py --h5 C:/mtg/mtg_train_v2.h5 --nc_dir F:/turkiye_kesitler --n 200` (salt okuma).
- Bulgu notları: HGB − B_full test_time +0,0030 [+0,0003, +0,0062] GA sıfırı dışlıyor (K18'deki "[+0,000" yuvarlamaydı), ama B_full
  test_time'da en düşük ağ; HGB − A_sat anlamsız. RESNET_CIFAR − A_sat test_both −0,0051 [−0,0104, −0,0006] anlamlı. Referans kıyasında
  tek anlamsız fark logreg − NWC r30 (test_both). Güvenilirlik eğrileri köşegen üstünde → dengeli sınıf ağırlıkları (train.py) ile tutarlı.
- Taslak: TAAC 18 + 3 sayfa, MAP 19 + 2; ~35 700 karakter (kaynaklar hariç); 56 `\todo` (liste: P14 yanıtı / CLAUDE.md §5).

### K24 — 15 Kas 2026: H5 kanal sırası doğrulandı; çift eşik modelinin adı düzeltildi
- `src/verify_channel_order.py` (200 örnek, ham `F:/turkiye_kesitler` ile): **gerçek H5 sırası** `ir_87, vis_09, wv_73, ir_105, nir_22,
  ir_123, wv_63, vis_08, ir_97, ir_38, vis_06, nir_16, nir_13, ir_133, vis_05, vis_04` (+ 16 = radar). 8 IR kanalı %99–100 oy;
  güneş kanalları %70–93 çünkü gece hepsi ≈0 (ayırt edilemez) → betik artık yalnız ayırt edici eşleşmeleri oylar; teyit için yeniden koşulacak.
- `baselines.CHANNEL_NAMES` bu sırayla değiştirildi (eskisi yalnız indeks 3'te tutuyordu). Etki: **tek eşik (ir_105) doğruydu**;
  **"threshold2_ir_105+vis_06" aslında ir_105 + ir_133 kullandı** → sonuç geçerli, adı yanlıştı. JSON/preds anahtarları değiştirilmedi
  (kayıt); makalede "IR10.5 ve IR13.3" yazılıyor (Fig. 4 altyazısı, Yöntem, Bulgular). Sinir ağları/HGB/RF/LogReg etkilenmez (tüm kanallar).
- P3 (kanal önemi) bu sırayı kullanmalı. Not: `baselines.py --ch2 vis_06` artık gerçek VIS0.6'yı seçer (önceki koşudan farklı model).

---

## 3. Komutlar

### 3.1 Veri hazırlama (yerel, tek seferlik) — sıra: aux → labels → splits → export
```bat
cd e:\belgeler\MTGCLM

:: 0) DEM (bir kez; Copernicus GLO-90 indirildi, zaten üretildi: data/dem_800x1000.npy)
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
python src\train.py      --h5 F:/radar/mtg_train_v2.h5 --model medium_flat --split splits/split_v1_compact.npz --aux --epochs 1 --limit_batches 50
python src\train_unet.py --h5 F:/radar/mtg_train_v2.h5 --model unet        --split splits/split_v1_compact.npz --aux --epochs 1 --limit_batches 20
```

### 3.2 Colab — tek komutla tüm matris (çalıştırıldı, 5 Eki 2026 — tamamlandı)
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

**Sonuçları yerele almak için:** Drive'daki `artifacts/` klasörünü (JSON + summary + checkpoints)
`E:\belgeler\MTGCLM\artifacts\`'e indirin — yerelde zaten mevcut (bu CLAUDE.md onları referans alıyor).

### 3.3 Elle koşmak isterseniz (tek tek deney adları gerçek koşuda kullanılan adlardır)
```bash
H5=/content/mtg_train_v2.h5; SP=splits/split_v1_compact.npz

# mimari ablasyonu + resnet
python src/train.py --h5 $H5 --split $SP --model arch     --aux --seed 1 --results artifacts/ARCH_s1.json
python src/train.py --h5 $H5 --split $SP --model resnet18 --aux --seed 1 --results artifacts/RESNET_s1.json

# modalite ablasyonu (kazanan mimari = medium_flat, 3 seed)
python src/train.py --h5 $H5 --split $SP --model medium_flat --seed 1                                                   --results artifacts/A_sat_s1.json
python src/train.py --h5 $H5 --split $SP --model medium_flat --seed 1 --aux                                             --results artifacts/B_full_s1.json
python src/train.py --h5 $H5 --split $SP --model medium_flat --seed 1 --aux --aux_channels cos_sza,elev,radar_cov       --results artifacts/C_noawos_s1.json
python src/train_unet.py --h5 $H5 --split $SP --model unet --aux --seed 1                                               --results artifacts/D_unet_s1.json

# bloklu 5-katlı CV
for k in 0 1 2 3 4; do
  python src/train.py --h5 $H5 --split splits/cv5_fold$k.npz --model medium_flat --aux --seed 1 \
      --results artifacts/CV_f${k}_s1.json
done

# baseline'lar + toplu rapor + paylaşılabilir örneklem (henüz üretilmedi)
python src/baselines.py --h5 $H5 --split $SP --aux --models majority,threshold,threshold2,logreg,rf,hgb --results artifacts/Z_baselines_s1.json
python src/aggregate_results.py --dir artifacts --out artifacts/summary --plot
python src/aggregate_results.py --dir artifacts --pattern "CV_*.json" --out artifacts/cv_summary
python src/make_public_sample.py --src $H5 --split $SP --out data/mtg_sample.h5 --n 2000
```
`aggregate_results.py` dosya adındaki **`_f<kat>_s<seed>`** kalıbını okuyup kat/seed koşularını
tek satırda ortalar (ortalama ± s.s.; Markdown + CSV + LaTeX + grafik) — gerçek çıktı
`artifacts/summary.md|csv|tex` + `artifacts/summary_{balanced_accuracy,mcc,f1_cloudy}.png`.

### 3.4 v3 — yerel PC (Windows, CPU) + Colab (ResNet-CIFAR) — K16
```bat
cd /d E:\belgeler\MTGCLM

:: 0) Ortam kontrolü (conda base)
python -c "import torch, h5py, sklearn, psutil, tqdm; print(torch.__version__, torch.get_num_threads())"

:: 1) Etiket politikası bölmeleri (yapıldı, 16 Eki; yeniden üretmek gerekirse)
python src\make_label_policy.py --h5 F:/radar/mtg_train_v2.h5 --policy verify

:: 2) Hız ölçümü (~3-5 dk) -> en iyi iş parçacığı sayısını seç
python src\bench_speed.py --threads 8,12,20

:: 3) 1 epoch gerçek deneme (yükleme MB/s ve epoch süresi log'da)
python src\train.py --h5 F:/radar/mtg_train_v2.h5 --split splits/split_v2_verify.npz --model medium_flat --aux ^
    --epochs 1 --loader auto --threads 12 --outdir artifacts_bench --results artifacts_bench/BENCH_B_s1.json

:: 4) v3 matrisi (yerel; ResNet-CIFAR hariç). Kesilirse aynı komut kaldığı yerden sürer.
python src\run_all.py --h5 C:/mtg/mtg_train_v2.h5 --outdir artifacts_v3 --stage modality,cv,baselines ^
    --loader auto --threads 12
::    isteğe bağlı: --cache_dir C:/mtg_cache  (NVMe'de >= 20 GB boş yer varsa; yüklemeyi hızlandırır)

:: 5) Rapor
python src\aggregate_results.py --dir artifacts_v3 --out artifacts_v3/summary --plot
python src\analyze_preds.py --dir artifacts_v3 --out artifacts_v3/analysis --refs
```
```python
# Colab — U-Net + iki ResNet (9 koşu). 1. hücre §3.2 ile aynı (Drive, H5 kopyası, git pull).
!python src/run_all.py --h5 {H5} --outdir {DRIVE}/artifacts_v3 --stage unet,resnet \
    --loader auto --cache_dir /content/cache
# Bitince Drive'daki artifacts_v3/{D_unet,RESNET}*.json + preds/ dosyalarını yerel artifacts_v3/'e kopyalayın.
```

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
| `src/make_dem_npy.py` | DEM GeoTIFF → 800×1000 `.npy` (metre) — Copernicus GLO-90 ile çalıştırıldı |
| `src/make_public_sample.py` | Paylaşılabilir küçük örneklem H5 + bölme (henüz çalıştırılmadı) |
| `src/metrics.py` | Balanced Acc, MCC, F1(clear/cloudy), AUC, karışıklık matrisi, majority |
| `src/baselines.py` | majority, threshold_ir_105, threshold2_ir_105+vis_06, logreg_aux, rf_aux, hgb_aux (K16: blok okuma, SYNOP-only, tahmin kaydı) |
| `src/aggregate_results.py` | Koşu JSON'ları → ortalama ± s.s. tablo (MD/CSV/LaTeX) + grafik |
| `src/run_all.py` | Çökmeye dayanıklı koşu yöneticisi (aşamalar, `.done`, manifest, log); v3: `--stage v3 --only --skip --loader --threads --cache_dir` |
| `src/h5_dataset.py` | `use_aux=True` → aux normalize edilip kanallara eklenir; meta (labels, okta, obs_time, ...) |
| `src/dataloader.py` | `get_split_dataloaders(loader=...)` (+ bölme–H5 doğrulaması), `get_dataloaders(use_aux=...)` |
| `src/fast_loader.py` | Hızlı yükleyici (K16): blok okuma, `memory/stream/auto`, `.npy` önbellek, batch çevirme |
| `src/run_utils.py` | Tekrarlanabilirlik (seed, deterministik, RNG checkpoint), ortam bilgisi, SYNOP-only metrik, tahmin kaydı |
| `src/make_label_policy.py` | METAR NSC politikası (keep/verify/ignore) → `split_v2_<politika>.npz`, `cv5_<politika>_fold*.npz` |
| `src/analyze_preds.py` | v3 örnek bazlı analiz: blok-bootstrap GA, eşleştirilmiş farklar, CV, kırılımlar, CLM/NWC kıyası (K18) |
| `src/bench_speed.py` | Model başına eğitim/değerlendirme hızı (sentetik), koşu süresi tahmini, iş parçacığı seçimi |
| `src/model.py` | `MTGConvNet` (Shallow/Medium/Deep × GAP/Flatten), `MTGResNet18` (stem: imagenet / cifar), `MTGUNet` |
| `src/extract_fci_clm.py` | FCI CLM → istasyon çevresi sınıf sayımları (merkez, 3×3, r5–r30 km); indeks `data/clm_station_index.npz` |
| `src/extract_nwcsaf_ct.py` | NWC SAF CT PNG (MGM, Mercator) → istasyon çevresi bulut tipi sayımları |
| `src/eval_clm_on_splits.py`, `src/eval_nwc_and_metar.py` | Referansları split örneklerinde değerlendirme; METAR NSC doğrulaması; bulut tipi kırılımı |
| `src/clm_vs_synop_quicklook.py` | CLM ↔ ham SYNOP ön bakış (merkez vs alan) |
| `docs/mtgclm_infografik.html` | Proje geneli insan-okur HTML infografik (her anlamlı değişiklikte güncellenir, K15) |
| `docs/mtgclm_kod_haritasi.html` | Kod haritası: dosya/sınıf/fonksiyon bağlantıları (otomatik, `src/make_code_map.py`; K19) |
| `src/make_code_map.py` | Kodu `ast` ile tarayıp kod haritasını üretir; `MODULE_INFO` (grup/açıklama/eski), `EXTRA_DATA` |
| `src/loss.py` | `MaskedBCELoss` — yalnız (16,16) pikselinde, `-1` yok sayılır |
| `src/train.py`, `src/train_unet.py` | Eğitim; `--h5 --split --aux --aux_channels --seed --epochs --limit_batches --results --outdir --model` + K16: `--loader --mem_splits --cache_dir --threads --no_deterministic --no_last_preds` |
| `src/config.py` | Colab/yerel yollar, `MTGCLM_H5`, PATCH_SIZE 33, BATCH_SIZE 64, LR 1e-3 |
| `paper/main.tex`, `preamble.tex`, `frontmatter.tex`, `backmatter.tex`, `sections/`, `supplementary.tex` | Makale LaTeX iskeleti (sn-jnl, sn-basic; `\ifTAAC`) — K22 |
| `paper/tools/verify_numbers.py`, `derived_facts.py` | `% src:` yorumlarını CSV hücreleriyle denetler; metin için ek kaynak CSV'leri üretir — K23 |
| `src/verify_channel_order.py` | Kompakt H5 kanal sırasını ham NetCDF ile eşleştirir (salt okuma) — K23 |
| `paper/tools/build.py`, `count_chars.py`, `captions2tex.py` | Derleme (TAAC/MAP, taslak/gönderim), TAAC karakter sayımı, altyazı dönüşümü — K22 |
| `paper/figures/make_*.py`, `figdata.py`, `make_all.py` | Makale şekilleri (Fig. 1, 4, 5, 6, 10) ve tabloları (Tab. 2–5, S1–S2); `make_all.py` hepsini üretip kontrol raporlarını özetler (K21) |
| `README.md`, `LICENSE`, `CITATION.cff`, `.gitignore` | Depo iskeleti + veri bulunabilirliği beyanı |

Arşiv: `archive_v1/` (v1 sonuçları + ağırlıkları — not: alt klasör adı yerelde `artifacts 06 2026`
olarak değişmiş görünüyor, muhtemelen kazara; `archive_v1/sonuç1` ve `sonuç09` da var), `src/_backup_20260916/`
(v1 skaler füzyon kodu). Akış dokümanı: `walkthrough_v2.md` (v1: `walkthrough.md`). Görev listesi:
`task.md` (hâlâ v1 fazlarını gösteriyor, v2 için güncellenmedi). `implementation_plan.md` v1
geç-füzyon planıdır — **arşiv**, güncel hattı temsil etmez. `notebooks/clmcl.ipynb` — keşif defteri.

---

## 5. Açık işler / sıradaki adımlar

- [x] ~~v2 koşu matrisini Colab'da çalıştır~~ — **tamamlandı (5 Eki 2026)**, sonuçlar K12'de.
- [x] ~~Gerçek Copernicus GLO-90 DEM'i indirip üret~~ — **tamamlandı**, `data/dem_800x1000.npy`.
- [ ] **K12'deki AWOS-aux bulgusunu doğrula/açıkla**: neden B_full (AWOS dahil) C_noawos'tan
      (AWOS'suz) düşük? IDW enterpolasyon kalitesi, normalizasyon, ya da gerçekten bilgi
      katkısı yok mu — ek analiz/ablasyon (farklı IDW parametreleri, AWOS yoğunluk haritası
      görselleştirme) gerekebilir.
- [ ] Bloklu CV'nin tek-bölme test_time'dan daha düşük/değişken çıkmasını makalede tartış
      (K12) — hangi haftalar/katlar düşürüyor, incele.
- [ ] Uydu-only (radar hariç) ablasyonu için `MTGH5Dataset`'e kanal seçme parametresi ekle
      (`base_channels` alt kümesi) — henüz yok.
- [ ] FCI CLM karşılaştırması: çıkarım `src/extract_fci_clm.py` ile (K13); aynı örneklerde merkez + alan bazlı, C/O dönemleri ayrı.
- [x] ~~NWC SAF CT projeksiyonu~~ — MGM köşeleri alındı, çıkarım yapıldı (K13/K14).
- [x] ~~METAR NSC politikası~~ — **verify** seçildi, `split_v2_verify` + `cv5_verify` üretildi; H5 yeniden üretimi gerekmedi (K16).
- [x] ~~Örnek bazlı tahmin kaydı, SYNOP-only metrik, determinizm, hızlı yükleyici~~ — K16.
- [x] ~~v3 matrisini koş~~ — 22–23 Eki 2026 tamamlandı (K17, K18).
- [x] ~~`src/analyze_preds.py`~~ — yazıldı ve gerçek veride denendi (K18).
- [x] ~~AWOS-t2m'nin CNN'deki etkisi~~ — CV'de küçük ama anlamlı pozitif (K18).
- [x] ~~P11 şekil altyapısı + P12a (Fig. 1, 4, 5, 6, 10; Tab. 2–5, S1–S2)~~ — K21. **Kalan:** Arial'lı makinede `make_all.py` ile son üretim.
- [x] ~~P13 LaTeX iskeleti~~ — K22.
- [ ] **Ertelenen işler (unutma):** P1 sonrası Tab. 4'e radarsız satırlar + Fig. 4 yeniden; P4 sonrası Tab. 5'e sahne başına çıkarım süresi;
      Tab. 2 dipnotundaki kalibrasyon şekli atfı çapraz referansa; gönderimde FigXX dosyalarını dergiye göre yeniden adlandır;
      `Bechini2007`/`Liu2020` (P16); Code availability metnini depo dışı makale klasörlerine göre uyarla; eşleştirilmiş farklar 4 ondalık (K21 istisnası).
- [x] ~~P14 Bölüm 2–4 taslakları~~ — K23.
- [x] ~~Kanal sırasını doğrula~~ — K24 (IR kesin; güneş kanalları için güncellenmiş betikle bir kez daha koş, `TAMAM` beklenir).
- [ ] Bölüm 2.2: "5–30 km komşuluk" ifadesi — sonuçlarda yalnız merkez ve 30 km var (5–20 km ekle ya da cümleyi daralt).
- [ ] Analiz raporunu makale bulgularına dönüştür; gece hataları (alçak bulut/sis) için ayrıntı.
- [ ] U-Net tam harita için hedef-piksel jitter (49×49 çıkar → 33×33 rastgele kırp).
- [ ] `infer_scene.py`: tek NetCDF sahne + radar + aux → 800×1000 olasılık haritası (GeoTIFF/PNG) — henüz yok.
- [ ] Kanal permütasyon önemi / Integrated Gradients (17→22 kanal için asıl bilimsel soru).
- [ ] Metrikleri gündüz/gece ve mevsim kırılımıyla raporla.
- [ ] `src/make_public_sample.py` ile paylaşılabilir örneklem (`data/mtg_sample.h5`) üret — henüz çalıştırılmadı.
- [ ] Zenodo DOI al, README'ye ekle; eğitilmiş ağırlıkları (artık `artifacts/checkpoints/*.pth`
      içinde) release olarak yayımla.
- [ ] `task.md`'yi v2 fazlarıyla güncelle (hâlâ v1 checklist'i gösteriyor).
- [ ] Makale limitations: Tem–Kas verisi yok; kar–bulut karışıklığı; ~1,7×1,1 km piksel / ~3×3,5 km CLM pikseli vs yarımküre
      gözlem ölçek farkı; etiket eşiği duyarlılığı (0/≥5 vs 0/≥1); AWOS-aux'un belirsiz katkısı.

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
- Ağır eğitim `run_all.py` üzerinden: v3'te yerel PC (CPU, `--loader auto`, `--outdir artifacts_v3`) + Colab (ResNet-CIFAR,
  `--outdir` Drive altında, `--cache_dir /content/cache`). Bkz. K16, §3.4.
- Depoya veri/ağırlık commit etme; `splits/*.npz|json` ve `artifacts/*.json` paylaşılır.
- Mimari/modalite tartışırken **Medium_Flat**'in kazanan mimari olduğunu ve AWOS-aux'un
  test kümelerinde net bir katkısı görünmediğini unutma (K6, K12) — v1 sayılarını güncel
  karşılaştırma için kullanma.
- Yeni bir karar alındığında bu dosyanın §2'sine ekle (Kural 2 / `.agent/rules/rule_update_claude_md.md`).
- İş paketi bitince dosya başına ayrı İngilizce commit komutları üret, çalıştırma (Kural 1 / `.agent/rules/rule_commit_per_file.md`).
- Anlamlı her değişiklikte `docs/mtgclm_infografik.html`'i güncelle; her kod değişikliğinden sonra `python src/make_code_map.py`
  ile `docs/mtgclm_kod_haritasi.html`'i yeniden üret (Kural 3 / `.agent/rules/rule_update_infographic.md`).
- `.agent/` ve `archive_v1/` .gitignore'da; bu dosyalar için commit komutu verme, yerelde kalır.
