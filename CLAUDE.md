# CLAUDE.md — MTGCLM

Bu dosya, Claude Code / Cowork oturumlarının projeyi sıfırdan anlamadan kaldığı yerden
devam edebilmesi için tutulur. **Yeni bir karar alındığında (model seçimi, veri bölme,
etiket politikası, eğitim komutu) bu dosyaya birkaç satır ekleyin.**

Son güncelleme: 2026-11-24

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
  1982 otomatik istasyondan (AWOS) IDW ile. **Kompakt H5'te örneği olan istasyon: 132** (90 yalnız SYNOP + 42 METAR'lı
  havalimanı; K14, K21) — makaledeki sayı budur.
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
  **Güncelleme (18 Kas 2026):** "tek commit" notu eskidi — 1 Eki'den beri dosya başına commit'lerle depoda ~196 commit var
  (Kural 1). Yine de karar gerekçesi için kaynak bu dosyadır.

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
  **Güncelleme (18 Kas 2026, K44):** çelişki kapandı — P20 yapılacak (kullanıcı kararı), yöntem 33×33 içinde 25×25 kırpma; H5 yeniden çıkarılmaz.

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

### K25 — 15 Kas 2026: P1–P3 kodu hazır (radarsız ablasyon, HGB/doku kontrolü, kanal önemi) — gece koşusu bekliyor
- **Tek kaynak kanal sırası:** `src/channels.py` (`FCI_CHANNELS`, `BASE_CHANNELS`, `WAVELENGTH`, P3 `GROUPS`); `baselines.py` buradan alır.
- **P1 — `train.py --base_channels`** (`fci` = 16 FCI, radarsız; virgüllü adlar; varsayılan `all`). Seçim yükleyicide değil
  `prep_features`'ta (aux z-skorundan sonra `index_select`) yapılır → önbellek/yükleyici değişmez. **Doğrulandı:** argümansız koşu ile
  17 kanalın açıkça seçildiği koşu mini H5'te bit bit aynı tahmin üretiyor. JSON `config.base_channels` (ad listesi | "all").
  `run_all.py` modaliteleri: `A_norad` (16 FCI), `B_norad` (16 FCI + t2m, rh2m, elev, cos_sza; radar_cov yok); CV varyantı `Bnorad`.
- **P2 — `train.py --spatial_shuffle`:** örnek başına 33×33 piksel konumları karıştırılır, merkez sabit, tüm kanallarda aynı permütasyon,
  eğitim ve değerlendirmede (birim testleri: merkez korunur, kanal başına değer kümesi korunur). Modalite `B_full_shuf`.
  **`baselines.py --feature_sets center1,center5,current,texture`** (texture = current + yüzdelik 10/90 + merkez−ortalama + Sobel);
  `current` eski özniteliklerle bit bit aynı ve `hgb_current` olasılıkları eski `hgb_aux` ile birebir (mini H5). Aşama `hgbfs` → `Z_hgbfs_s1`.
- **P3 — `src/channel_importance.py`:** B_full s1–3 checkpoint'leri, test_time; permütasyon önemi (22 kanal + 10 grup × 5 tekrar) ve IG
  (5000 tabakalı örnek, 32 adım, taban 0, hedef logit farkı). **Güvenlik kontrolü:** yeniden hesaplanan olasılıklar kayıtlı tahminlerle
  karşılaştırılır, fark > 1e-3 ise durur (mini testte fark 0). Aşama `importance` → `analysis/channel_importance*.csv`, `ig_summary.csv`.
  Şekil: `paper/figures/make_fig07_channel_importance.py` (make_all.py'de; çıktı yoksa atlanır).
- **`analyze_preds.py`:** yeni deneyler ORDER sonuna (mevcut tablo sırası değişmez), çiftler A_sat−A_norad, B_full−B_norad,
  B_full−B_full_shuf, hgb_texture−hgb_current, hgb_current−hgb_center1, hgb_center5−hgb_center1, B_full−hgb_texture; CV B−Bnorad;
  radar kapsamı kırılımı (`--aux_h5` bir kez → `data/processed/radar_cov_center.npy`). **Yeni koşu yokken mevcut 6 CSV birebir yeniden
  üretildi** (bootstrap ağırlıkları deney döngüsünden önce çekiliyor → makaledeki sayılar değişmez).
- **Koşu (~8,5–9 sa; bölünebilir, kaldığı yerden sürer):**
  `python src\run_all.py --h5 C:/mtg/mtg_train_v2.h5 --outdir artifacts_v3 --stage modality,cv,hgbfs,importance
  --modalities A_norad,B_norad,B_full_shuf --cv_variants Bnorad --loader auto --threads 12`
  Sonra: `python src\analyze_preds.py --dir artifacts_v3 --out artifacts_v3/analysis --refs --aux_h5 C:/mtg/mtg_train_v2.h5`,
  `python paper\tools\verify_numbers.py` (mevcut 178 sayı hâlâ 0 hata olmalı), `python paper\figures\make_all.py`.

### K26 — 15 Kas 2026: P1–P3 gece koşusu bitti; kanal önemi permütasyonu düzeltildi, şekil/tablo süzgeci
- **P1 (radar):** A_sat − A_norad test_time −0,0006 [−0,0016, +0,0003] (anlamsız), test_both −0,0024 [−0,0040, −0,0005];
  B_full − B_norad test_time +0,0007 [+0,00002, +0,0015]; CV B − Bnorad +0,0009 [+0,0003, +0,0014] (4/5 kat). → Radarın katkısı
  aux varken küçük pozitif, aux yokken yok. Radar kapsam dışı örnek yalnız test_station'da (n = 964).
- **P2 (uzamsal yapı):** B_full − B_full_shuf +0,0016 [+0,00002, +0,0033] (test_time), test_station +0,0021 (anlamlı).
  HGB: hgb_texture test_time 0,962; texture − current +0,0025 (anlamlı); current − center1 +0,0146; B_full − hgb_texture
  −0,0055 [−0,0093, −0,0023] → doku öznitelikli HGB CNN'i test_time'da geçiyor; makalede dürüstçe raporlanacak.
- **P3 tasarım hatası düzeltildi:** tüm kümede permütasyon gece örneklerine gündüz yansıması taşıyordu (gece NIR1.6 kaybı 0,18 —
  fiziksel değil). `channel_importance.py --perm_within illum` (varsayılan): permütasyon gündüz/alacakaranlık/gece sınıfı içinde,
  assert ile korunur, json'a `perm_within` yazılır. **importance aşaması yeniden koşulmalı** (`--stage importance --force`).
- **Şekil/tablo:** `figdata.MAIN_EXPS` (MODEL_ORDER'ın ilk 12'si: ana modeller + B_norad + A_norad) — Fig. 4/5/6 ve Tab. 3 yalnız bunları
  çizer; B_full_shuf ve hgb_* öznitelik kümeleri yeni **Tab. S4** (`tabS4_spatial_texture`). Tab. 4'e A_sat−A_norad ve B_full−B_norad
  (CV: B − Bnorad). `springer_fig.COLORS/LABELS` yeni deneyleri içerir. `MTGCLM_FIG_OUT` ortam değişkeni şekilleri önizleme klasörüne
  yönlendirir (son şekillerin üzerine yazmadan sınama).
- **P3 yeniden koşuldu (aydınlanma içi, 76 dk):** gündüz IR8.7 0,304, IR12.3 0,225, IR3.8 0,177, NIR1.6 0,158, NIR2.2 0,115 (VIS ≤ 0,010);
  gece IR3.8 0,381, IR8.7 0,205, T2m 0,150, IR10.5 0,106; radar ≤ 0,004, statik ≤ 0,001. Gece NIR kaybı artık 0. **IG gece NIR1.6'ya %12
  pay veriyor** — sıfır tabanın karanlık yansımadan uzaklığı ("gece" bilgisi); yorum permütasyona dayanır (metinde ve altyazıda yazıldı).
  T2m modelde önemli ama ablasyonda küçük → bilgi büyük ölçüde IR'de de var (BT10.5−T2m, K14).
- **Metin:** 03_methods (§3.5 radar/doku/karıştırma, §3.7 kanal önemi), results/inputs (radar; TAAC'ta önem özeti → Supp. Fig. S1),
  models (HGB öznitelik kümeleri, karıştırma), importance.tex (MAP). refs.bib'e He2016, Breiman2001, Kunsch1989, Sundararajan2017.
  verify_numbers 249/0; TAAC 20+5 s., MAP 22+3 s., 48 \todo.

### K27 — 15 Kas 2026: P4 — `src/infer_scene.py` (tam harita çıkarımı) — gerçek sahnelerde doğrulandı
- Girdiler eğitimle birebir: FCI kanalları **adla** (`channels.py` sırası), H5 `means/stds` z-skoru, NaN→0, float16 yuvarlaması;
  radar `radar_aligned/<sahne>.npy`; aux `update_h5_aux_channels.py` fonksiyonları ve parametreleri **H5 `aux_params`'tan**
  (k=8, p=2, 250 km, 6,5 K/km, floor; radar_cov o anda PNG'si olan radarlar — eğitimde `--radar_dir f:/radar/ppi`). Grid dışı:
  eğitimdeki yama dolgusu (uydu/radar 0, elev/radar_cov 0, t2m/rh2m/cos_sza ekstrapole konumda). AWOS anları
  `data/processed/awos_scene_cache/` altında önbelleklenir (4,4 GB CSV tek geçişte taranır).
- **U-Net havuzlama fazı:** U-Net yalnız merkez pikselde (16,16; 4'e bölünen konum) denetlendi → tek geçişte 4 piksellik faz deseni
  çıkıyor (v2 ağırlığıyla 25 May 2026 sahnesi: tek geçiş − faz-hizalı ort. |fark| 0,077, maskede %3 uyuşmazlık, 4 periyotlu güç
  oranı 1,9 → 0,001). Varsayılan `--unet_mode aligned`: 16 geçiş, her piksel eğitim fazında. Boyut 4k+1 olmalı (up1 output_padding=1).
- CNN (B_full) kayan pencere 33×33 adım 1; bulut ortamında (2 çekirdek) 320 bin piksel 148 s.
- **Tutarlılık testi (`--check`):** sahte H5 + preds ile mekanik doğrulandı (girdi farkı 0, harita = yama olasılığı). Gerçek doğrulama
  kullanıcı makinesinde F:/turkiye_kesitler + C:/mtg H5 ile yapılacak.
- **Ölçek gözlemi (makale için):** 25 May 2026 12 UTC'de kümülüs alanında RGB'de çok sayıda açık boşluk varken model Anadolu'nun
  çoğunu bulutlu veriyor — ürün "piksel bulut maskesi" değil, ~57×37 km bağlamdan **istasyonun ≥5 okta bildirme olasılığı**.
  P5/P6 ve Tartışma'da bu ayrım açıkça yazılmalı.
- Çıktılar `artifacts_v3/scenes/`: `.nc` (h5py ile NetCDF4/CF), `_prob.tif` (EPSG:4326, PIL ile), `.png`, `.json`, `timing.csv`.
  quality_flag: 1 uydu NaN, 2 radar dosyası yok, 4 AWOS doldurma, 8 radar kapsamı dışı, 16 statik radar_cov.
- **Gerçek doğrulama (15 Kas, MSSPC, 3 test sahnesi: 2026-05-03 11:58 ve 03:48, 2025-12-31 11:58; 298 istasyon satırı):** girdi farkı **0**
  (22 kanal), yeniden üretilen yamadan olasılık ↔ kayıtlı preds ≤ 4e-7; tam harita ↔ kayıtlı: CNN birebir, U-Net medyan ≤ 8e-5
  (p95 ≤ 0,043, en büyük 0,16 — 33 piksel dışındaki gerçek bağlam). **v3 U-Net tek geçişte kullanılamaz:** tek fazlı geçiş maskenin
  %38–54'ünü değiştiriyor (tek indeksli fazlarda ort. |fark| ~0,4–0,6) → faz-hizalı mod zorunlu; P20 (jitter) ile kalıcı çözülebilir.
  Süre (12 iş parçacığı, tek model): U-Net 14 s (16 geçiş), CNN kayan pencere 167 s; girdi hazırlığı ~2 s; tepe bellek ~2 GB.
  Özet: `src/summarize_scenes.py` → `artifacts_v3/scenes/scenes_summary.csv` (verify_numbers bu klasörü de okur). Tab. 5 sahne
  sütunu ve Yöntem §3.8 (P4 kısmı) yazıldı.

### K28 — 15 Kas 2026: P5 — sahne seçimi, ürün eşlemesi ve Fig. 8 kodu (gerçek koşu bekliyor)
- `src/select_scenes.py`: test haftalarından (test_time ∪ test_both, ≥ 25 etiketli istasyon) 5 sahne **görüntüye bakmadan**,
  önceden yazılı puanlarla: konveksiyon (yama radar maks. ≥ 128 mavi kanal), kar (Ara–Mar gündüz, açık, ≥ 1500 m, VIS0.6 ≥ %30,
  NIR1.6/VIS0.6 ≤ 0,4), sirrüs (NWC CT 11–15), gece alçak bulut/sis (bulutlu ve okta 9 ya da RH ≥ %90), gündüz açık (açık oranı).
  Aynı gün iki kez seçilmez. Çıktı `scenes_selected.csv` + `scenes_candidates.csv`.
  **Radar ölçeği varsayımı:** radar_aligned değerleri 3,2'lik adımlarla 0–192 (≈ 0–80 dBZ doğrusal ise 128 ≈ 40 dBZ); MGM PNG ölçeği teyit edilmeli.
- `src/build_scene_pack.py`: FCI doğal renk (R NIR1.6, G VIS0.8, B VIS0.6, SZA normalize) / gece IR10.5; U-Net olasılığı; FCI CLM
  (2,3 → bulutlu; 1,8 → açık; diğer geçersiz) ve NWC CT (5–15 bulutlu) 800×1000'e en yakın piksel. **CLM ızgara eşlemesi satpy'siz:**
  geos (sweep y) `col = floor(X/2000,02 + 2784)`, `row = floor(Y/2000,02 + 2784)` — `clm_station_index.npz` ile 279/279 birebir.
- `paper/figures/make_fig08_scenes.py`: 5 satır × 4 sütun (a–t), 174 × ~120 mm; istasyon dolu = bulutlu, boş = açık (şekille);
  `MTGCLM_SCENE_PACK`/`MTGCLM_SCENES_CSV` ile test girdisi; make_all'a eklendi (paket yoksa atlanır). Sahte paketle düzen sınandı.
- **İlk gerçek koşu (15 Kas):** konveksiyon 14 May 11:58, kar 12 Mar 08:58, sirrüs 13 May 05:58, gece 31 Ara 17:58, gündüz açık 11 Mar 06:58 —
  ama 11 Mar 06:58'de **FCI CLM yok** (o gün 09:50'de başlıyor). Seçime ön koşul eklendi: CLM tekrar döngüsü var ve NWC istasyonların
  ≥ %90'ında var (veri varlığı; görüntüye bakılmadı). Yeni gündüz açık adayı 15 Haz 2026 06:58 (diğer dört sahne değişmiyor).
  İlk gözlemler: kar sahnesinde CLM ve NWC karlı dağları büyük ölçüde bulutlu sayıyor (CLM'de kar/buz sınıfı 8 hiç yok), model açık;
  sirrüs sahnesinde NWC tüm Anadolu'yu yarı saydam bulutlu, gözlemcilerin yarısı açık bildiriyor (model istasyonlarla uyumlu).
  `build_scene_pack.py` artık `artifacts_v3/scenes/scene_pack_summary.csv` yazar (istasyonlarda model/CLM/NWC doğruluğu, piksel uyumu).

### K29 — 16 Kas 2026: 82 bozuk NWC SAF CT karesi — referans sayıları düzeltildi
- 82 PNG'nin (18 354 karenin %0,45'i) **tamamı tek sınıf 15** (kar/buz üstü yarı saydam; denizler dahil) → fiziksel değil, bozuk kare.
  P5 sirrüs ölçütü tam da bu karelerden birini (13 May 2026 05:45) seçti; NWC referans metrikleri de bu kareleri "bulutlu" sayıyordu.
- `extract_nwcsaf_ct.py`: `degenerate_frames()` (istasyonların ≥ %95'i sınıf 15), `is_degenerate_image()`; `--flag_only` mevcut gün
  dosyalarında `ok=False` + `degenerate` dizisi (uygulandı). `select_scenes.py` ve `build_scene_pack.py` bu kareleri veri yok sayar.
- `analyze_preds.py --refs` yeniden koşuldu: **yalnız references.csv değişti** (diğer CSV'ler sayısal olarak birebir). Yalnız-SYNOP
  test_time: NWC merkez 0,907 → 0,911, r30 0,923 → 0,927; kapsam %84,1 → %83,6; CNN − NWC r30 +0,023 → +0,019 [+0,013, +0,024];
  test_station NWC r30 0,903 → 0,904, fark +0,030 → +0,029; test_both r30 farkı +0,020 → +0,017 [+0,007, +0,026]; lojistik regresyon
  istisnası sürüyor (+0,006 [−0,005, +0,017]). products.tex ve infografik güncellendi; verify_numbers 251/0. Fig. 5 yeniden üretilmeli.
  (v2 dönemi `artifacts/nwc_metar_on_splits.json` tarihsel; yeniden koşulmadı.)

### K30 — 16 Kas 2026: P6 — bulut sıklığı altyapısı (`batch_scenes.py`, `cloud_frequency.py`, run_all `climatology`)
- Dönem = h5meta'daki 183 etiketli gün × 8 nominal saat (00…21 UTC) = 1 464 zaman; en yakın FCI sahnesi (≤ 10 dk). U-Net faz-hizalı,
  **3 seed ortalaması (kullanıcı kararı)**; sahne başına ~45 s → ~18 sa. Sahne başına `data/processed/cloud_freq/scenes/cf_<ts>.npz`
  (prob_u8, flag, CLM ikili) + `manifest.csv`; kaldığı yerden sürer.
- `cloud_frequency.py`: sıklık = bulutlu/geçerli (olasılık ≥ 0,5); CLM ile kıyas aynı sahne-piksel geçerlilik maskesinde (joint);
  tüm dönem / ay / saat NetCDF'leri `artifacts_v3/climatology/`, `cloud_freq_summary.csv` (sahne sayıları, kara ortalamaları,
  istasyon doğrulaması: SYNOP gözlenen sıklık vs model/CLM, aynı sahne zamanları), `station_frequency.csv`. Ay başına gün sayısı
  eşit değil (Ara 2025: 8 gün, Nis 2025: 11 gün) — şekilde ve metinde sahne sayısı verilmeli; "klimatoloji" denmeyecek.

### K31 — 16 Kas 2026: P6 sonuçları — bulut oluşum sıklığı, istasyon doğrulaması, Fig. 9
- Koşu (MSSPC, 1 103 dk): 1 464 nominal zaman → 1 437 sahne (27'sinde FCI yok), 1 sahne radar yok → **1 436 geçerli**, 1 420'sinde CLM
  (C 290 / O 1 131 sahne). Sahne başına medyan 43,6 s (3 seed).
- Türkiye maskesi: kara ve bir AWOS istasyonuna ≤ 25 km; kıyı ≤ 30 km deniz; doğu yaylaları ≥ 38,5° D ve ≥ 1500 m; iç kesim kalan
  (`cloud_frequency.py region_masks`, `regions.npz`). Model sıklığı (Türkiye) 0,609; en bulutlu Şub 2026 0,793, en açık Haz 2026 0,325;
  günlük döngü tepe 12 UTC 0,732, en düşük 21 UTC 0,517.
- **İstasyon doğrulaması** (SYNOP, sahne zamanlarında, 0/≥5 okta; `--station_only` ile alt kümeler eklendi): test haftaları 131 istasyon,
  19 333 rapor: model r 0,973, bias −0,026, MAE 0,031; CLM r 0,936, −0,035, 0,051. Görülmemiş istasyonlar: model r 0,955, CLM 0,923.
  "all" alt kümesi eğitim örneklerini de içerir (iyimser) — makalede test alt kümeleri kullanıldı.
- **CLM − model (aynı sahne-piksel):** Türkiye −0,012; iç kesim −0,032 (15 UTC −0,090); doğu yaylaları +0,059 — yalnız gündüz
  (06–12 UTC +0,130…+0,163) ve kar mevsiminde (Ara +0,139, Oca +0,138; Mar 2025 +0,189, commissioning dönemi), Nis–Haz ~0 → CLM karı
  bulut sayıyor (Fig. 8 kar sahnesiyle tutarlı). P7 C/O ayrımı bu Mar 2025 değerini ayrıca ele almalı.
- `paper/figures/make_fig09_cloud_frequency.py` (a tüm dönem, b CLM − model, c Oca, d Haz, e günlük döngü bölgeler; Türkiye dışı beyaz
  örtülü; renk ölçeği 0,1–0,9). TAAC ana metin Fig. 9, MAP Supplementary Fig. S1 (`\figfreq` makrosu). make_all'a eklendi.
  Yöntem §3.8 ve Bulgular "Full-scene" alt bölümü yazıldı; verify_numbers 290/0; TAAC 22+5, MAP 24+4 sayfa, 41 \todo.
- U-Net haritasında ızgara kenarında (25° D, 35° K) ince dolgu şeritleri var; Türkiye maskesi dışında kalıyor.

### K32 — 16 Kas 2026: P7 — FCI CLM dönem (C/O) ve "cloud contaminated" duyarlılığı
- `src/references_by_period.py` → `artifacts_v3/analysis/references_by_period.csv` (yalnız SYNOP, CLM kapsamı; B_full ve D_unet; gün blokları,
  test_station istasyon blokları; all_test = üç kümenin birleşimi). C dosyaları yalnız test_station'da (test haftaları O döneminde).
- **Fark C'den kaynaklanmıyor:** all_test, 30 km: CNN − CLM C +0,042 [+0,023, +0,062] (n 6 745), O +0,041 [+0,032, +0,052] (n 50 265).
  C döneminde model de daha düşük (0,920 vs 0,949) — Mar–Nis 2025 zor bir dönem.
- Eşleme: contaminated = açık → CLM r30 O'da 0,866 (fark +0,083); dışarıda bırak → fark +0,040 [+0,029, +0,053] (değişmez).
  r30 eşikleri (train): bulutlu 0,45, açık 0,15, dışarıda 0,25.
- Ek tablo `tabS5_clm_period` (TAAC S5 / MAP S4), products.tex'e 3 cümle. verify_numbers 303/0.

### K33 — 16 Kas 2026: P8 — gece / sis / alçak bulut / kar hata analizi
- `src/export_center_values.py` → `data/processed/center_values.npz` (400 209 satır; istasyon pikselinde 17 temel kanal ham birim, 5 aux,
  yama radar maks.). `src/error_analysis.py` (B_full s1–3 olasılık ortalaması ≥ 0,5; test_time + test_station birlikte) →
  `error_analysis.csv`, `error_cases.csv` (10 vaka, tohum 0), `error_summary.json/.csv`.
- Kaçırma gece %8,3, gündüz %3,7; yanlış alarm %5,1 / %3,3. **Gece kaçırmaları sis/alçak bulut değil:** NWC "çok alçak/alçak" payı
  kaçırmalarda %6, doğru yakalananlarda %20; kaçırmaların %50'sinde NWC "açık"; RH ≥ %90 payı %12 vs %33. Kaçırmalar yarı saydam
  bulutta (ince %20,8, orta %16,6) ve kuru havada (RH < %50: %11,1). RH ≥ %95'te açık raporların %13,7'si bulutlu → sığ sis/pus olabilir.
- **Kar yanlış alarm üretmiyor:** kar imzalı gündüz açık raporlarda %1,2 (n 169) vs tüm gündüz açık %3,3. Yüksek istasyonlardaki
  (> 1500 m) yanlış alarm (kış %7,4, ilkbahar %10,8) kara bağlı değil; az örnek (311, 139). → CLM'nin kar sorunu modelde yok.
- Ek şekil `paper/figures/make_figS_errors.py` (TAAC S3 / MAP S2), breakdowns.tex'te iki P8 \todo'su metne dönüştü. verify_numbers 323/0.

### K34 — 16 Kas 2026: P9 etiket tanımı duyarlılığı ve P10 paralaks kaydırması
- **P9** (`src/predict_rows.py` — kayıtlı ağırlıklarla 17 639 doğrulanmamış METAR açık satırı, güvenlik kontrolü 512 satır;
  `src/label_sensitivity.py` → `label_sensitivity.csv`; ek tablo `tabS6_label_sensitivity` TAAC S6 / MAP S5). Görülmemiş haftalar, tüm etiketler:
  verify CNN 0,956 (CLM r30 0,912, NWC r30 0,928; fark +0,044 / +0,021); keep CNN 0,935 (0,886 / 0,899; +0,047 / +0,028);
  0/≥1 okta CNN 0,844 (0,806 / 0,810; +0,039 / +0,014; eşik 0,5, yeniden seçilmedi — val'de 1–4 okta yok). Sıralama değişmiyor.
  `artifacts*/preds_extra/` .gitignore'a eklendi. ResNet ağırlıkları yerelde yok → P9'da yok.
- **P10** (`src/parallax_shift.py`, analitik, küresel Dünya, MTG-I1 0°; satpy bu ortamda yok): 132 istasyonda VZA 51,6–64,4°;
  kayma 6 km CTH'de 7,6–12,5 km (medyan 9,2), 10 km'de en çok 20,9 km (≈ 10,9 px K–G, 10,3 px D–B) → 33×33 yama (16 px yarı) ve
  30 km içinde. Ek tablo `tabS7_parallax` (TAAC S7 / MAP S6), ek şekil `make_figS_parallax.py` (TAAC S4 / MAP S3); 02_data.tex'teki
  "5–30 km" ertelenen işi çözüldü (metin "merkez ve 30 km" olarak düzeltildi). verify_numbers 343/0.

### K35 — 16 Kas 2026: P15 — Giriş, Tartışma, Sonuç ve iki öz yazıldı
- `01_introduction.tex` (7 paragraf; ¶1 ve ¶7 \ifTAAC), `05_discussion.tex` (5.1 ürünler, 5.2 model ailesi, 5.3 AWOS/radar, 5.4 gece;
  TAAC 5.5 izleme / MAP 5.6 kanal fiziği; 5.7 sınırlılıklar), `06_conclusions.tex`, frontmatter özleri (TAAC 231, MAP 215 kelime).
- **Atıf kuralı:** refs.bib'de henüz `verified = {yes}` yok (P16). Tam metni okunmamış kaynaklar `\citeu{}` / `\citetu{}` ile
  (preamble; taslakta turuncu "v", build.py sayar: 16). P16'da doğrulananlar `\citep`'e çevrilecek, desteklemeyenler kaldırılacak.
  Eksik kaynaklar `\todo`: Holmlund 2021 (BAMS, MTG), Ackerman 1998 / Stengel 2014, satpy, FCI L1c, CLM ATBD, NWC SAF CT, Copernicus DEM.
  MAP çapaları (10.1007/s00703-…) bib'de yok.
- Karakter: toplam 70 038 (sınır 100 000); TAAC 28 + 8, MAP 29 + 8 sayfa; verify_numbers 346/0; kalan \todo 17.

### K36 — 16 Kas 2026: P16 (1. tur) — atıf yapılan 20 kaynağın tam metin doğrulaması
- PDF klasörü: `E:\belgeler\OneDrive\OneDrive - mgm.gov.tr\uydu ile ilgili makale` (kök 100 + `oa_pdf/` 40). Makaledeki 20 `\citeu`
  kaynağının hepsi bulundu ve okundu → `\citep`; refs.bib'de `verified = {yes}` (20; yedek `refs.bib.bak_20261116`).
  Notlar: `docs/paper/literature_notes.md` (ne yaptılar / ana sayı / desteklediği cümle).
- **Üç cümle daraltıldı:** Derrien2005 yalnız eşik testlerini anlatıyor (gece/kar/sis zayıflığı iddiası yok → gece kısmı Kotarba2017'ye);
  Wang_2023 derin ağla kıyas yapmıyor (→ "RF yalnız spektral öznitelikle yüksek doğruluk"); Ezber_2022/Youn2026 "sığ sis" değil
  (→ "İstanbul çevresinde sık sis/pus").
- PDF'i olmayan çapalar: rs12030365 (paralaks+radar), amt-19-6193-2026, peerj.20495 (FCI kar), rs15245662 (MTG yağış),
  MAP'in 6 çapası (10.1007/s00703-…), BAMS 2021 MTG (Holmlund). Yöntem kaynakları (He, Breiman, Künsch, Sundararajan) `verified = {no}`.
- **2. tur:** Ozkaya_2022 (Veri, MGM radar), Bozzo_2024 (Giriş ¶2), Hadizadeh2019 (Giriş ¶4), Kotarba2016 (TAAC 5.5) ve yeni
  `NWCSAF_ATBD_2019` (Veri, referans ürünler) eklendi → 25 doğrulanmış kaynak. Borna_2024, Kamat_2025 okundu, eklenmedi.

### K37 — 17 Kas 2026: P16 (3. tur) — eksik PDF'ler eklendi, kalan atıf \todo'ları kapandı
- Kullanıcının eklediği 11 PDF okundu (DOI'ler PDF'ten). refs.bib'e 9 makale/rapor + 4 yazılım/veri girdisi (`verified = {yes}`;
  yedek `refs.bib.bak_20261117`; eski eksik `Ackerman2006` girdisi doğrulanmışla değiştirildi). Notlar `literature_notes.md` 3. tur.
- Yerleşim: Holmlund2021 + DAdderio2023 (Giriş ¶2), Ackerman2006 = MOD35 ATBD v5.0 (Giriş fizik ¶ MAP + Bulgular/önem),
  Balaj2026 (Giriş ML ¶), LiS2024 (Giriş radar ¶), Mowla2026 (Yöntem/önem), Bielinski2020 + Kotarba2026 (Veri/paralaks; Sonuç),
  Kuter2026 (TAAC Tartışma: doğu yaylalarında kar → FCI H43 kar maskesiyle sınama), satpy, EUMETSAT_FCIL1c (EO:EUM:DAT:0662),
  EUMETSAT_FCICLM (EO:EUM:DAT:0678), CopernicusDEM (10.5270/ESA-c5d3d65).
- Kullanılmayan MAP çapaları: Shukla 2026 (tavan yüksekliği), Xiang 2022 (CloudSat girdap), Sukanya 2019 (bulut radarı) — cümle desteği yok.
- Açık: satpy sürümü (`\todo`, coknc.py ortamı), Balaj2026 tam yazar sırası (bib'de "and others"). Derleme 0 tanımsız, 0 \citeu,
  kalan \todo 12; verify_numbers 346/0.

### K38 — 17 Kas 2026: P17 — iki katmanlı açık örneklem ve Zenodo paketi (kod hazır, gerçek veride koşu bekliyor)
- `src/make_public_sample.py` yeniden yazıldı: tabakalı (küme × sınıf × gündüz/gece; merkez cos_sza > 0) örneklem, iki dosya —
  `<önek>_core.h5` (katman a: patches[:, :16] FCI + aux elev, cos_sza + etiket/meta) ve `<önek>_mgm.h5` (katman b: radar
  [N,1,33,33] + t2m, rh2m, radar_cov + aux_flags). Ortak `row` (kompakt H5 satırı); `--merge` tam 17+5 biçimini geri kurar,
  `--check` (bölme/etiket, katman eşleşmesi, kaynakla 50 satır birebir). Sentetik H5'te merge birebir; core dosyayla
  `train.py --aux` 1 epoch uçtan uca koştu (18 kanal).
- `src/make_zenodo_package.py`: örneklem (a) + splits + artifacts_v3 JSON + analysis + 2 ağırlık (B_full_s1, D_unet_s1) +
  klimatoloji NetCDF → `zenodo_package/` + MANIFEST.csv (sha256) + metadata.json + README; (b) yalnız `--include_mgm`.
- README: "v3 sonuçlarını yeniden üretme" (çıktı → komut tablosu) ve "Açık örneklemle hızlı deneme". **Sapma:** P17'deki
  "her şekil/tablo için tek komut" paper/ betiklerini GitHub'a taşırdı (kullanıcı istemiyor) → yalnız src/ çıktıları listelendi.
  Etiket önerisi `v1.0-taac` yerine `v1.0` (aynı nedenle). Sürüm notu taslağı `docs/release_notes_v1.0.md`.
- .gitignore: `zenodo_package/`, `data/public/`, `artifacts_sample/`.
- **Gerçek veride koşuldu (17 Kas):** `data/public/mtg_sample_core.h5` 2038 satır, 32 MB (+ `_mgm.h5` 9 MB), `--check` OK;
  `zenodo_package/` 126 dosya (a: 30 dosya 39 MB, türetilmiş: 95 dosya 69 MB), katman (b) pakette yok. satpy sürümü 0.48.0
  (coknc.py ortamı). .gitignore'a `artifacts_v3/scenes/*.png`, `climatology/regions.npz`, `analysis/parallax_grid_6km.npz` eklendi.

### K39 — 17 Kas 2026: P18 — kapak mektupları ve önerilen hakemler
- `docs/paper/cover_letter_taac.md`, `cover_letter_map.md` (İngilizce, ≤ 1 sayfa; 3 katkı, kapsam uyumu, veri dönemi kısıtı,
  veri/kod erişimi, çıkar çatışması yok). Sayılar özetlerden (verify_numbers geçen frontmatter).
- `docs/paper/suggested_reviewers.md`: Kotarba (CBK PAN), Kuma (MISU), Karlsson (SMHI), Tian (PNNL), Casella (CNR-ISAC); e-postalar
  PDF'lerin yazışma satırından. Önerilmeyen: Wojciechowska (Kotarba ortak yazarı), Kuter (H SAF–MGM kurumsal çakışma riski),
  Bozzo (EUMETSAT, CLM üreticisi), Alves (doğrudan rakip).
- P16 kapanış: satpy 0.48.0 (22 Nis 2024) metne ve bib'e; Balaj2026 tam yazar listesi (kullanıcı, yayıncı sayfası).
  Backmatter: örneklem adı/boyutu yazıldı, "Code availability" todo'su yeniden ifadeyle kapandı → kalan \todo 10.

### K40 — 17 Kas 2026: P19 — gönderim öncesi kontrol raporu
- `docs/paper/p19_kontrol_2026-11-17.md`. Engelleyenler: Şekil 2 (veri akışı) ve 3 (mimari) hâlâ `\figpending`; MAP
  importance.tex:25'te "K14" final PDF'e sızıyor; 12 `\todo`. MAP'te Tablo 5, Tablo 3'ten önce atıflanıyor; ek materyal
  ilk atıf sırası karışık. Açılmayan kısaltmalar: NWC SAF, SEVIRI, CT, METAR (+AUC MAP). FigS_errors etiket çakışması,
  FigS_parallax yazı tipi → Windows'ta make_all. Öz/karakter/ondalık/Türkiye/beyan başlıkları OK.

### K41 — 17 Kas 2026: P19 düzeltmeleri
- **Şekil 2 ve 3 çizildi:** `paper/figures/make_fig02_flow.py` (veri akışı + etiket politikası) ve `make_fig03_arch.py`
  (Medium_Flat CNN; U-Net, kayıp yalnız merkezde), yardımcı `diagram_utils.py`; make_all'a eklendi.
- **K14 sızıntısı kapandı:** AUC yeni `src/single_channel_auc.py` → `artifacts_v3/analysis/single_channel_auc.csv` ile güncel
  etiketlerde yeniden hesaplandı. Eğitim kümesi SYNOP: BT10.5 − T2m 0,930, BT10.5 0,901 (eski K14: 0,932/0,893).
- **Zayıf CV haftaları (K12 açık işi):** `src/cv_weeks.py` → `cv_weeks.csv`. Mar–Nis 2025 0,941 / Ara 2025–Haz 2026 0,950;
  en zayıf dönemler neredeyse tamamen bulutlu: Şub 2026 0,917 (%93 bulutlu), Nis 2025 0,925, 2026-W05 0,835 (%97).
- **Diğer:** Tablo 1'e doğal çözünürlük + radar/aux satırları; MAP'e bootstrap/CV denklemleri (Denk. 1–2); scikit-learn 1.9.1;
  NWC SAF/DEM teşekkürü; kısaltmalar açıldı (SEVIRI, MSG, CALIOP, MODIS, EUMETSAT, NWC SAF, CT, METAR, AUC; özetlerde NWC SAF).
  İsteğe bağlı kalibrasyon todo'su kaldırıldı.
- **Atıf sırası:** ek materyal numaraları `preamble.tex`'teki makrolarla (`\STparallax`, `\SFerrors`, …) ilk atıf sırasına bağlandı.
  `supplementary.tex` dergiye göre aynı sırada; MAP maliyet tablosu Yöntem'de (Tablo 3).
- **Gönderim dosyaları:** `paper/tools/export_figures.py` → `paper/submission/<dergi>/Fig<N>.pdf|tif`, `ESM_FigS<N>` +
  `figure_map.csv` (.aux'tan numara okur). Kaynak adları iki dergide ortak olduğu için yeniden adlandırılmadı.
- `FigS_errors` panel harfleri eksen dışına (çakışma giderildi). Kalan 2 `\todo` kullanıcıda: MGM teyidi + Zenodo concept DOI,
  S.K. katkısı/tez beyanı. TAAC özeti 247 kelime (sınır 250). **Son adım: final derleme `main_final`.**

### K42 — 17 Kas 2026: katman (b) izni ve son todo'lar
- **MGM katman (b) için sözlü onay verdi** (yazılı onay kullanıcıda). Data availability: iki katman da Zenodo'da
  (`mtg_sample_core.h5` 32 MB, `mtg_sample_mgm.h5` 9 MB); ham radar/yer gözlemleri talep üzerine MGM'den. README, sürüm notu,
  kapak mektupları, `make_zenodo_package.py` (artık `--include_mgm` ile koşulur) ve declarations taslağı güncellendi.
- Çalışma tezden türemiyor; S.K. doktora danışmanı (yazar katkısında "Supervision: S.K." yeterli, ek beyan yok).
- Tablo 2 dipnotu kaynakta (`make_tables.py`) düzeltildi — make_all tabloyu yeniden yazınca eski todo geri gelmişti.
- Fig02/03 betikleri altyazıyı `fd.FIG_DIR`'e yazıyor (önceden göreli yol; make_all kökten koşunca hata veriyordu).
- Derleme: 0 `\todo`, 0 tanımsız, 0 taşan kutu, verify_numbers 356/0. Kalan: Fig02/03'ü Arial ile üret, final derleme, export.

### K43 — 17 Kas 2026 akşam: final derleme ve gönderim dosyaları (K42'deki kalan işler kapandı)
- Fig02/Fig03 Arial'lı Windows'ta üretildi (`_check.txt` OK); `paper/main_final.pdf` ve `main_MAP_final.pdf` derlendi; `paper/submission/taac`
  (Fig1–8 + ESM_Fig1–4) ve `paper/submission/map` (Fig1–9 + ESM_Fig1–3) + `figure_map.csv` dışa aktarıldı. 0 `\todo`, verify_numbers 356/0.
  (Kayıt 18 Kas'ta dosya zaman damgalarından geriye dönük yazıldı.)

### K44 — 18 Kas 2026: hakem öz-değerlendirmesi, revizyon planı, ön-kayıt, EUMETSAT CLM doğrulama raporu, P20 kararı
- `docs/paper/hakem_degerlendirmesi_MTGCLM.md` (3 kritik + 15 ikincil itiraz, 15 jüri sorusu) ve `docs/paper/ret_riski_yol_haritasi_MTGCLM.md` (R1–R15,
  E1–E12) madde madde değerlendirildi → `docs/paper/madde_madde_degerlendirme_2026-11-18.md`; takvim `docs/paper/calisma_plani_revizyon_2026-11-18.md`
  (R16–R19 ve N1–N5 eklendi). **Ön-kayıt `docs/revizyon_onkayit.md`** (docs/paper .gitignore'da olduğu için docs/ altında; commit'i zaman damgası).
- **Model–referans farkı etiket tanımına duyarsız** (`label_sensitivity.csv`): CLM r30 +0,044 / +0,047 / +0,039, NWC r30 +0,021 / +0,028 / +0,014
  (verify / keep / 0–≥1 okta) → Kritik 2'ye ana savunma.
- **EUMETSAT FCI L2 doğrulama raporu (EUM2045564 v1A, 21 Kas 2025)** okundu: CLM yalnız SEVIRI NWC SAF CMa ve MODIS maskelerine karşı
  (13 Ağu–13 Eki 2025) doğrulanmış — **yer gözlemi yok** (ACTRIS yalnız OCA CTH için). Ortalama POD %93, FPR %9,6, doğruluk %92 (SEVIRI'ye göre);
  yarı kurak yüzeylerde POD ~%50'ye iniyor. **Kar/buz testi kapalı** → karlı açık pikseller bulutlu sınıflanabilir (bilinen sorun; NIR2.2 kullanılmıyor)
  → doğu yaylaları kış CLM fazlası (K31) ve kar sahnesi (K28) için bağımsız kaynak; "CLM'de sınıf 8 hiç yok" gözleminin açıklaması.
  Ayrıca: alacakaranlıkta alçak bulutlar "contaminated" sınıflanabilir (K32 eşleme duyarlılığıyla ilgili). NWC SAF eğitim sunumu (Haz 2025)
  NWC/GEO CMA'nın SYNOP/SHIP ile doğrulandığını söylüyor (uydu adı belirtilmeden) → Öz'ün ilk cümlesi "operasyonel CLM yer gözlemiyle
  doğrulanmadı" biçiminde daraltılacak; NWC/GEO MTG-I day-1 doğrulama raporu okununca kesinleşecek.
- **Kullanıcı kararları (18 Kas):** P20 (U-Net jitter) yapılacak; MGM yazılı izin süreci başlatıldı.
- **Olgusal düzeltme gereken iki cümle:** 03_methods "each test set is evaluated once" (K14→K16 kronolojisiyle çelişiyor) ve TAAC özetinin ilk cümlesi.
  AI beyanı (03_methods sonu) taslak yazımını kapsamıyor → genişletilecek (Springer: LLM kullanımı Methods'ta belgelenir).
- **18 Kas akşam — N1 uygulandı:** NWC/GEO MTG-I Day-1 User Manual (NWC/CDOP3/MTG/MFL/SCI/UM/Cloud 1.1.1, 30 May 2025) okundu:
  NWC/GEO CMA'nın MTG doğrulaması **SYNOP/SHIP ile tam diskte, POD %85,9** (VR: NWC/CDOP4/MTG/MFL/SCI/VR/Cloud v1.0; PDF henüz klasörde yok);
  gece karlı zemin algılanmıyor. → Eski Öz cümlesi ("not yet evaluated against surface observations") yanlıştı. TAAC özetinin ilk cümlesi
  "validated only at full-disc scale, not against a national station network" oldu (250 kelime, sınırda); Giriş ¶2'ye iki ajans doğrulaması
  ve CLM'de kar testinin kapalı olduğu eklendi. refs.bib: `EUMETSAT_CLMVR2025`, `NWCSAF_UMMTG2025` (verified=yes; yedek `refs.bib.bak_20261118`).
  verify_numbers 356/0; TAAC 31+7, MAP 32+6 sayfa, 0 tanımsız, 0 \todo.
- **Zenodo (18–19 Kas):** v1.0 yayımlandı, **katman (b) `mtg_sample_mgm.h5` yüklenmedi** (MGM yazılı izni bekleniyor). Sürüm DOI
  10.5281/zenodo.24112650 (Code availability: makalenin sürümü), **concept DOI 10.5281/zenodo.24112649** (Data availability, README,
  CITATION.cff, kapak mektupları). Data availability metni "katman (b) yazılı onaydan sonra aynı kayda eklenecek" olarak düzeltildi;
  README'deki "19 GB H5 Zenodo'da" ifadesi yanlıştı, düzeltildi (tam H5 paylaşılmaz). Yayımlanan pakette README katman (b)'yi "included"
  diye anlatıyor → v1.1'de (katman b eklenince) doğru olur.
  **Düzeltme (19 Kas):** v1.0'a örneklem hiç yüklenmemiş, açıklama boş (kullanıcı). → **v1.1** paketi `zenodo_package/v1.1/` (128 dosya,
  107,6 MB; katman (b) yok) — `make_zenodo_package.py` artık `--version` alır, metadata'da `doi` alanı yok (yeni sürüm DOI'sini Zenodo atar),
  README concept DOI'yi verir. **v1.1 yayımlandı (19 Kas): DOI 10.5281/zenodo.24368365** → Code availability ve CITATION.cff'e işlendi. Katman (b) → v1.2.
- **YÖK taraması (19 Kas, 15 tez):** çakışan çalışma yok; ayrıntı `docs/paper/literatur_tarama_kaydi_2026-11-19.md`.
- **NWC/GEO MTG-I day-1 VR (NWC/CDOP4/MTG/MFL/SCI/VR/Cloud 1.0.1, 30 May 2025) okundu:** CMA v2025, 27 gün (Eki 2024–Şub 2025), tam disk
  SYNOP/SHIP; bulutlu N > 5 okta, açık N < 3 okta (3–5 dışarıda), 7×7 kutu; POD %85,9 / FAR %6,4 (gündüz 83,6/4,6, gece 86,7/8,6). Raporun
  kendisi "this is not a proper validation" diyor; bölgesel kırılım yok. EUMETSAT CLM VR'de CLM bölümünde yer gözlemi yok (ACTRIS yalnız OCA).
- **Literatür taraması (18–19 Kas; Scopus 14, IEEE 5, EBSCO 2×50 kayıt):** kontrol makaleleri (Alves_2026, Kuter2026) bulundu → sorgu çalışıyor;
  FCI bulut maskesini yer gözlemiyle değerlendiren hakemli çalışma yok. Kayıt `docs/paper/literatur_tarama_kaydi_2026-11-19.md`; CSV'ler `docs/paper/`.
- **Uyarı (18 Kas):** depo herkese açık; `docs/` altındaki iç ön-inceleme belgeleri (hakem öz-değerlendirmesi, ret riski, madde madde değerlendirme,
  revizyon planı) push edildi → depodan çıkarılıp `docs/paper/` (gitignore) altına taşınması önerildi; geçmişten silme kararı kullanıcıda.

### K45 — 19 Kas 2026: revizyon Gün 1 — R3, R4, N2–N4 (metin; yeni analiz yok)
- **R3:** `docs/paper/karar_kronolojisi.md` (karar · tarih · kanıt · küme · K). 03_methods "each test set is evaluated once" (2 yer) kaldırıldı;
  §3.2 sonuna paragraf: analiz ön-kayıtlı değildi, verify politikası test sonuçlarından sonra, P1–P3 ablasyonları HGB ≈ CNN görüldükten sonra
  eklendi; mimari, epoch ve eşikler yalnız val/train'de; tek bölme skoru ön-belirlenmiş değil, CV ve Tablo S(label) bağımlılığı gösterir.
- **R4:** AI beyanı (03_methods, Software and hardware; declarations_taac.md) → "Claude, Anthropic … to draft the manuscript text …"; yazarlar
  çalışmayı tasarladı, analizleri yönetti, her şeyi doğruladı. Code availability seçenek (a): "analysis tables … one command per output";
  şekil/tablo dizgi betikleri talep üzerine (K22 ile uyumlu). Author contributions değiştirilmedi.
- **N2:** 82 bozuk NWC karesi Veri §2.6'ya (18 354'ten 82, %0,45; tümü sınıf 15; ölçüt istasyonların ≥ %95'i); products.tex atıfla kısaldı.
- **N3:** README'ye ve Zenodo README şablonuna `threshold2_ir_105+vis_06` = IR10.5 + IR13.3 notu (v1.2'de Zenodo'ya girer).
- **N4:** BKN = 5–7 okta (alt sınır); 1–4 okta dışlaması NWC SAF doğrulama pratiğiyle (3–5 dışarıda) uyumlu + Tablo S(label) atfı; r30 yarıçapı
  km cinsinden, ürünün kendi ızgarasında; mimari v2 etiketleriyle seçildi, v3'te aileler benzer.
- **Not:** satır ortasına `% ref:` yorumu koymak cümlenin geri kalanını siler (18 Kas'ta Data availability'de oldu, 19 Kas düzeltildi) →
  yorumu satır SONUNA koy, metne yeni satırda devam et.
- verify_numbers 356/0; TAAC 31+7, MAP 32+6 sayfa; toplam 81 143 karakter (sınır 100 000, hedef 80 000 → hafif aşım).

### K46 — 19 Kas 2026: P20 kodu (U-Net hedef-piksel jitter) + "neden 0 / ≥5 okta" metni + R20 ön-kaydı
- **`train_unet.py --center_jitter [--jitter_crop 25]`:** eğitimde 33×33 yamadan örnek başına rastgele 25×25 pencere (`center_jitter_crop`);
  etiket istasyonun penceredeki gerçek konumuna (8..16) yazılır → 16 havuzlama fazının hepsi görülür; kayıp (`MaskedBCELoss`) değişmedi.
  Pencereler (seed, epoch)'a bağlı ayrı `torch.Generator`'dan (devam eden koşu aynı pencereleri görür; küresel RNG'ye dokunmaz).
  Doğrulama/test değişmedi (33×33, merkez). JSON config: `center_jitter`, `jitter_crop`. Epoch satırına genel ilerleme % ve kalan süre eklendi.
- **Testler `src/test_center_jitter.py`:** 6 birim testi + mini H5'te bütünleşik: jitter KAPALI yeni kod == P20 öncesi kod (`3f1008a`) bit bit
  (kayıp geçmişi + ağırlıklar), jitter AÇIK uçtan uca. Bulut ortamında hepsi geçti (torch 2.15 CPU).
- **run_all:** yeni aşama `unetjit` → `D_unet_jit_s{1,2,3}` (D_unet dosyalarına dokunmaz). Değerlendirme (Gün 4): `infer_scene.py --exp D_unet_jit
  --artifact_check` (tek geçiş vs faz-hizalı), analyze_preds'e D_unet_jit − D_unet. Fig. 9 kuralı ön-kayıtta.
- **Metin (02_data §2.4):** sınıf sınırlarının gerekçesi — 0 okta tüm gökte bulutu dışlayan tek rapor; 5 okta METAR'da BKN'nin (5–7) alt
  sınırı; ürünler aynı kuralla ikilileştirilmiyor, merkez piksel + 30 km bulutlu oranı (eşik train'de) — NWC SAF doğrulamasının kutu-oranı
  kuralına benzer. TAAC 32+7 sayfa.
- **R20 ön-kayda eklendi (17:22 TSİ, hiçbir analiz koşulmadan):** bulutlu ≥ 6 okta duyarlılığı, yalnız SYNOP.

### K47 — 19 Kas 2026: P20 eğitimi bitti (D_unet_jit × 3, yerel CPU) — ölçüt (ii) sağlandı, ölçüt (i) sahne testi bekliyor
- `run_all --stage unetjit`: 3 koşu × 15 epoch, ~101–103 dk/koşu (i7-12700, 12 iş parçacığı); git `a51df12` (s2/s3 "dirty" = yalnız koşu
  sırasında değişen artifacts_v3/run_manifest.json; kod aynı). Log'larda hata/uyarı yok. En iyi epoch 4 / 2 / 8 (val eğrisi 2. epoch'tan
  sonra ~0,940–0,943'te düz; D_unet 0,943–0,945). Val kaybı D_unet'ten yüksek (0,156–0,165 vs 0,148–0,151): eğitim 25×25, doğrulama 33×33
  — beklenen pencere farkı.
- **Ön-kayıt ölçütü (ii) (eşdeğer-olmama, alt GA > −0,005):** D_unet_jit − D_unet test_time −0,0011 [−0,0024, +0,00004]; test_station
  −0,0010 [−0,0024, +0,0005]; test_both −0,0008 [−0,0036, +0,0017] → **sağlandı** (farklar sıfırdan ayırt edilemiyor). BalAcc: test_time
  0,9563 vs 0,9574. Hesap `scratch/p20_pairs.py` → `scratch/p20_analysis/` (analyze_preds, ORDER'a geçici ekleme; yayımlanmış CSV'lere
  dokunulmadı, D_unet satırları birebir aynı).
- **Ölçüt (i)** (tek geçiş ↔ faz-hizalı maske uyuşmazlığı < %5; D_unet'te %38–54): kullanıcı makinesinde `infer_scene.py --exp D_unet_jit
  --check --artifact_check` (K27'nin 3 sahnesi) bekleniyor. Not: D_unet Colab T4'te, D_unet_jit CPU'da eğitildi (donanım farkı).
- **Ölçüt (i) sağlandı (19 Kas, 3 sahne):** tek geçiş ↔ faz-hizalı ikili maske uyuşmazlığı **%0,27 / %0,14 / %0,16** (D_unet %38 / %54 / %39);
  ort. |fark| 0,002–0,003 (D_unet 0,34–0,42); 4 piksellik periyot gücü tek geçişte 0,005–0,014 (D_unet 0,85–1,54) → faz deseni kalktı.
  `--check`: girdi farkı 0, istasyonlarda doğruluk D_unet ile aynı ya da yüksek. → **Ön-kayıt kuralı: Fig. 9 jitter'lı modelle yeniden üretilir.**
- `scenes_summary.csv` yeniden üretildi: P5 sahne paketi koşularının (7 D_unet sahnesi, `--check` olmadan) satırları da girdi → o satırlarda
  tutarlılık sütunları boş (NaN, beklenen). Tab. 5 `timing.csv`'den yalnız D_unet satırlarını kullanır; verify_numbers 356/0.
- **`batch_scenes.py --unet_mode single`** eklendi (yalnız adında "jit" geçen modelle; D_unet'te reddeder); ilerleme satırına % eklendi.
  Koşu: `--exp D_unet_jit --unet_mode single --out data/processed/cloud_freq_jit`, sonra `cloud_frequency.py --src data/processed/cloud_freq_jit
  --out artifacts_v3/climatology_jit`. Eski `cloud_freq/` ve `climatology/` korunur; Fig. 9'un kaynağı sonuçlar incelendikten sonra değişecek.

### K48 — 20 Kas 2026: bulut sıklığı jitter'lı U-Net ile yeniden üretildi; ürün modeli = D_unet_jit (tek geçiş)
- `batch_scenes --exp D_unet_jit --unet_mode single` → `data/processed/cloud_freq_jit/` (1 437 sahne ok, 27 sahnesiz, 0 hata; medyan 7,3 s/sahne,
  eski 43,6 s). `cloud_frequency.py` → iklim. **Klasörler yer değiştirdi:** `artifacts_v3/climatology/` = jitter'lı (ürün), eski 16 geçişli
  `artifacts_v3/climatology_D_unet_aligned/` (arşiv). CLM tarafı birebir aynı (en büyük fark 2e-7).
- **Sonuç farkları:** model her bölgede biraz daha bulutlu (Türkiye 0,609 → 0,626; doğu yaylaları +0,029, Haziran +0,05); istasyon
  doğrulaması iyileşti (test haftaları bias −0,026 → −0,016, MAE 0,031 → 0,027, r 0,973 → 0,975; görülmemiş istasyonlar r 0,955 → 0,958).
  **CLM − model doğu yaylaları +0,059 → +0,029** (yarıya indi); gündüz 06–12 UTC +0,106…+0,141, Ara +0,107, Oca +0,105; gece −0,049…−0,001;
  Şub 2026'dan sonra −0,074…+0,023; Mar 2025 +0,168. Türkiye −0,012 → −0,030, iç kesim −0,032 → −0,046.
  → Ön-kayıt R10 kuralına göre (tüm varyantlarda > +0,03 değil; taban zaten 0,029) **"CLM karı bulut sayıyor" hipotez düzeyine indi**:
  metin "gündüz-kış imzası, ürünün kapalı kar testiyle (EUMETSAT_CLMVR2025) uyumlu; istasyon ağı piksel düzeyinde doğrulayamaz" diyor.
- **Metin:** Yöntem §3.8 (merkez denetimli U-Net'in faz bağımlılığı → jitter eğitimi; tek geçiş %0,14–0,27; nokta skoru farkı −0,0011
  [−0,0024, +0,0000]; tablolar merkez denetimli U-Net'i raporlar), fullmap.tex iklim paragrafları, Tartışma §5.5, Sonuç, TAAC özeti
  ("cloudier on winter days over …", 250 kelime). analyze_preds: D_unet_jit ORDER sonuna + çift D_unet_jit − D_unet (eski CSV'ler birebir;
  references.csv yalnız iki sütun kazandı). verify_numbers 363/0; TAAC 32+7, MAP 33+6.
- `build_scene_pack.py`: dosya eşleşmesi tam ad (D_unet artık D_unet_jit'i yakalamaz) + `--unet_mode single`.
- **Bekleyen (kullanıcı, Windows):** Fig. 8 sahne paketi D_unet_jit ile (infer_scene 5 sahne, single → build_scene_pack) + `make_all.py`
  (Fig. 8, Fig. 9). Sonra fullmap.tex ilk paragrafındaki sahne doğrulukları güncellenecek. Zenodo v1.1'deki iklim dosyaları eski modelden → v1.2'de yenilenir.

### K49 — 20 Kas 2026: revizyon Gün 2 analizleri — R6, R8, R17, R20 (ön-kayıtlı; yeniden eğitim yok)
- **Fig. 8 ve Fig. 9 jitter'lı modelle yeniden üretildi** (kullanıcı, Windows, `_check` OK). Sahne paketinde istasyon doğrulukları değişmedi
  (yalnız bulutlu alan oranları), fullmap.tex ilk paragrafı geçerli. Altyazılar "U-Net trained with centre-pixel jitter, mean of three seeds,
  single pass" oldu.
- **R6 `src/revision_metrics.py` → `analysis/contingency.csv`** (a/b/c/d, POD, POD_açık, POFD, FAR, bias, CSI, KSS, HSS; KSS/HSS için blok GA).
  **KSS = 2·BalAcc − 1** (birebir) → BalAcc sonuçları Hanssen–Kuipers diline doğrudan çevrilir. test_time SYNOP: B_full POD 0,938, FAR 0,022,
  bias 0,958, KSS 0,907; CLM r30 POD 0,913, FAR 0,063, bias 0,975, KSS 0,815; NWC r30 POD 0,916, FAR 0,050, KSS 0,855 → modelin üstünlüğü
  hem daha az kaçırma hem daha az yanlış alarmdan. Modeller bulutu hafif az tahmin ediyor (bias ~0,96).
- **R17 → `analysis/unverified_metar_clear.csv`:** doğrulanmamış METAR açıklarında (yalnız açık etiket) doğru açık oranı test_time B_full 0,877
  [0,822, 0,925] vs CLM r30 0,798, NWC r30 0,816 (aynı satırlar 0,872); test_station 0,828 vs 0,667 / 0,666. Model bu satırlarda da önde.
- **R20 → `analysis/label_ge6.csv`:** SYNOP, ≥ 6 okta bulutlu, 5 okta dışarıda: BalAcc test_time 0,970; model − CLM r30 +0,043 [+0,028, +0,060],
  − NWC r30 +0,020 [+0,014, +0,026]; tüm kümelerde ve D_unet'te pozitif, GA sıfırı dışlıyor → **ön-kayıt kuralı: "etiket tanımından bağımsız"
  ifadesi kalır.**
- **R8 `src/inference_robustness.py` → `analysis/multiplicity.csv`, `block_length_sensitivity.csv`:** p = 2(k+1)/(B+1); BH (birincil), Holm.
  **Aile 1** (model − referans, 48): hepsi BH ve Holm sonrası anlamlı; 1/2/3/5 günlük örtüşen bloklarda hepsi sıfırı dışlıyor (en düşük alt GA
  +0,0115). **Aile 3** (CV, 10): 7'si BH ve Holm sonrası anlamlı (AWOS B − C, Ct2m − C, B − A, B − Bnorad; yalnız SYNOP B − Bnorad düşüyor).
  **Aile 2** (tek bölme, 108): GA'sı sıfırı dışlayan 30 farkın **14'ü BH sonrası kalıyor, Holm sonrası 0**. BH'de düşenler: hgb_aux − B_full,
  B_full − B_norad, B_full − B_full_shuf, hgb_texture − hgb_current, A_sat − A_norad (test_both), RESNET_CIFAR − A_sat (test_both),
  C_noawos − A_sat (test_station). Kalanlar: hgb öznitelik kümeleri, B_full − hgb_texture (test_time tümü), B_full − C_noawos ve C_t2m − C_noawos
  (test_station). Blok uzunluğu GA'yı test_time'da 1 → 5 gün %44 genişletiyor; RESNET_CIFAR − A_sat, B_full − B_full_shuf, hgb_aux − B_full
  bloğa duyarlı. P20 farkı (D_unet_jit − D_unet) test_time'da her L'de alt GA > −0,0025.
- **Metin etkisi (Gün 5/6):** inputs.tex ve models.tex'te "significant/significantly" yalnız BH sonrası kalanlar için; Yöntem §3.6'ya BH + Holm
  ve blok uzunluğu cümlesi; ek tablolar (kontenjans, ≥6 okta, doğrulanmamış METAR, çoklu karşılaştırma).

### K50 — 20 Kas 2026: revizyon Gün 3 — R12 referans-kalibre kıyas, R11 gözlemci tavanı (ön-kayıtlı)
- **R12 `src/reference_calibrated_baseline.py` → `analysis/reference_calibrated.csv` (+ `_info.json`):** CLM (6 grup × 4 geçerli sınıf + geçerli
  oran + merkez durumu, 34 öz.) ve NWC CT (6 grup × 16 sınıf + merkez sınıf, 112 öz.) oranlarıyla HGB/LR, yalnız train SYNOP (~91 bin),
  `baselines.py` ayarları, arama yok, eşik 0,5. sklearn 1.9.1 (Windows; VM'deki 1.7.2 ile 4. ondalıkta aynı). Süre ~1 dk.
- **Birincil (test_time SYNOP, CLM+NWC ortak kapsam, n 18 992):** REF_cal (HGB) 0,941; **CNN − REF_cal +0,0057 [+0,0024, +0,0089]**,
  U-Net +0,0080 [+0,0048, +0,0111] → ön-kayıt kuralı: GA sıfırı dışlıyor, "kalibrasyonun ötesinde ek bilgi" ifadesi kalır, ama küçük.
  **Ayrıştırma:** aynı satırlarda ham NWC r30 0,928, ham CLM r30 0,899; CNN 0,947. NWC'ye karşı farkın ~%70'i (+0,013 / +0,019) kalibrasyonla
  kapanıyor (REF_cal ≈ NWC-only kalibre 0,940); CLM'yi kalibre etmek yalnız +0,004 kazandırıyor (0,903) → CLM farkı (+0,044) ürün içeriğinden.
  İkincil: test_station HGB farkı GA sıfırı içeriyor (+0,0045 [−0,0008, +0,0100]); test_both sınırda (+0,0071 [−0,0003, +0,0129]); LR'ye karşı
  hepsi pozitif. Sensör kuşağı (NWC CT SEVIRI tabanlı) bu analizle ayrılamaz.
- **R11 `src/observer_ceiling.py` → `analysis/observer_ceiling.csv`, `_confusion.csv`, `_okta.csv`:** SYNOP HH:00 ↔ METAR HH−1:50, 42 havalimanı,
  42 804 çift. METAR→SYNOP ikili uyum BalAcc 0,956 [0,945, 0,966] (tüm çiftler), 0,974 (test kümeleri); FAR ≈ 0 (SYNOP açık/METAR bulutlu
  yalnız 11 çift), uyuşmazlık neredeyse tamamen SYNOP ≥ 5 / METAR NSC (2 551) → büyük ölçüde NSC tanımı (yüksek bulut METAR'da raporlanmaz),
  gözlemci gürültüsü değil. Aynı çiftlerde B_full 0,952 vs METAR 0,972 (test_time −0,019 [−0,032, −0,005]) → tavan iyimser; model ona ulaşmıyor.
  Okta başına (test_time SYNOP, 3 tohum ort.): okta 0 yanlış alarm %3,0; kaçırma okta 5 %16,7, 6 %6,5, 7 %1,3, 8 %0,7 → hatalar sınıf sınırında.
- **Karar Noktası 1 (dergi):** kullanıcıda. R12 TAAC'ın ana iddiasını korur ama NWC'ye karşı üstünlüğün çoğu hizalamadan; Öz/Tartışma buna göre.

### K51 — 20 Kas 2026: revizyon Gün 4 — R9, R18, R13, R10 (ön-kayıtlı; yeniden eğitim yok)
- **R9 `src/awos_colocation.py` → `analysis/awos_colocation*.csv`, `awos_pixel_distance.csv`, `okta9_rh.csv`:** 30 holdout istasyonunun
  **hepsinin kendi AWOS'u var** (aynı wmoid; 24'ü ≤ 1 km, 6'sı listeler arası 1–3 km koordinat farkı). Ön-kayıtlı kıyas: B_full − C_noawos
  test_station ≤ 1 km grubu +0,0023 [+0,0001, +0,0046], > 1 km grubu +0,0053 [+0,0009, +0,0113] → "yalnız ko-lokasyonlu grupta" koşulu
  sağlanmadı, "transfer edilemez" yazılmıyor. **Ama** kendi AWOS'u olmayan holdout istasyonu yok → test_station AWOS katkısını yalnız
  enstrümanlı noktalarda ölçüyor. Türkiye piksellerinde en yakın AWOS medyan 9,7 km (%95 20,1 km; ≤ 1 km payı %0,7) → Sınırlılıklar'a.
  okta 9 test satırlarının %0,3'ü (114 / 153); B_full doğru %95,6 / %90,2, C_noawos %94,7 / %91,5 → RH bağlaşımı önemsiz.
- **R18 `src/idw_loo.py` → `analysis/idw_loo.csv` (+ `_info.json`, önbellek `data/processed/idw_loo_sample.npz`):** 288 an (9 ay × 8 saat ×
  4 gün), 1 861 AWOS, aux ile aynı IDW, kendisi dışarıda. t2m RMSE 1,83 K (bias +0,01; etiket istasyonları 1,55 K); yükseklik düzeltmesiz
  2,44 K (etiket ist. bias −0,82 K) → 6,5 K/km düzeltmesi işe yarıyor. Komşuya |Δz| ≥ 600 m'de 2,53 K. rh2m RMSE %12,7 (etiket ist. %11,0,
  bias +2,3, gündüz +4,1). Hata komşu mesafesiyle neredeyse değişmiyor (≤ 5 km 1,81 K, > 20 km 1,89 K) → yerel etkiler baskın.
- **R13 `src/normalization_shift.py` → `analysis/normalization_shift*.csv`:** istasyon pikseli, ay × aydınlanma, train σ. **Kural (a) tetiklendi:**
  Haz 2026'da 8 IR kanalının hepsinde 3 aydınlanma sınıfında |kayma| 0,69–1,52σ (sıcak yaz; diğer aylar < 0,5σ). Haziran eğitimde var
  (9 948 satır); B_full Haz BalAcc test_time 0,974 (en iyi ay), test_station 0,919 (aylar 0,875–0,958). Tanısal (b) aynı ay içinde
  test − train IR en büyük 0,89σ (WV6.3, Ara 2025). **Yeniden eğitim kararı kullanıcıda** (öneri: hayır; Tem–Kas yokluğu Sınırlılıklar'da).
- **R10 `src/cloud_freq_sensitivity.py` → `climatology/cloud_freq_sensitivity.csv` (+ `_info.json`):** D_unet_jit, doğu yaylaları, CLM − model
  (joint), tüm dönem: eşik 0,4 / 0,5 / 0,6 → +0,004 / +0,029 / +0,061; val bias = 1 (eşik 0,176) −0,053; izotonik ≥ 0,5 (eşik 0,100) −0,083;
  izotonik ortalama −0,057; yarıçap 15 / 50 km ±0,007. **Kural: hipotez** (en küçük −0,089). Val'de bulutlu oranı 0,838 (verify politikası
  METAR açıkları çıkarıyor) → val kalibrasyonu sahne önceline uymuyor; not edilecek. Tanısal kış gündüzü (Ara–Mar, 06–12 UTC) tüm
  varyantlarda +0,12…+0,24 → fark kar mevsimi gündüzünde sağlam, tüm dönem ortalamasında değil. Metin zaten hipotez düzeyinde (K48).

### K52 — 20 Kas 2026: dergi kararı ve tekrarlanabilirlik düzeltmeleri
- **Karar Noktası 1 (kullanıcı):** birincil dergi **TAAC** (ilk karar süresi en kısa); kapsam dışı reddine karşı MAP sürümü de gönderime
  hazır tutulur (`\ifTAAC` iki sürüm). Öz/Tartışma R12'ye göre: "station-scale", NWC'ye karşı farkın çoğu hizalamadan.
- R10 Windows'ta sklearn 1.9.1 ile yeniden koşuldu: sayılar aynı (hipotez; −0,089…+0,062).
- R9'un AWOS kayıt sayısı önbelleği ilk koşuda bir awk komutuyla üretilmişti (depoda değil). Betiğin kendi sayımıyla
  (`awos_colocation.awos_counts`) yeniden üretildi: 555 istasyonda sayı ~%1 farklı (awk boş olmayan dizeyi, pandas sayısal değeri sayıyor),
  yeterli kayıtlı AWOS 1 849 → 1 844; awos_colocation.csv ve istasyon tablosunun kayıt sayısı dışındaki sütunları birebir aynı.
- README'ye "Revizyon analizleri" tablosu (R6–R20, P20); §7'ye tekrarlanabilirlik kuralı.

### K53 — 20 Kas 2026: R13 ADIM 3 — z-skor sabitleri eğitim kümesinden yeniden eğitim (fark yok)
- **Bulgu:** H5 `means/stds` (create_patch_dataset, ilk 5 sahne) eğitim kümesinden IR'de 1,3–1,8σ sapıyor (istasyon pikseli; IR3.8
  eğitimde 270,8 ± 20,0 K, sabit 286,3 ± 8,7 K) → girdi z'si IR'de ort. ≈ −1,4, s.s. ≈ 2. Eğitim, test ve tam harita aynı sabitleri
  kullandığı için uyumsuzluk yok; afin fark ilk conv + BN ile kuramsal olarak telafi edilir. Kullanıcı ADIM 3'ü onayladı; ön kayda
  (`docs/revizyon_onkayit.md` §3, 20 Kas 17:10) birincil ölçütle koşudan önce eklendi.
- **`train.py --renorm train`:** temel 17 kanal için yeni z = a·z + b (a = σ_eski/σ_yeni, b = (μ_eski − μ_yeni)/σ_yeni); sabitler eğitim
  kümesinden, ön-karıştırılmış H5'te 20 rastgele ardışık blok × 2048 satır (tohum 0, model tohumundan bağımsız), önbellek
  `<outdir>/norm_stats/train_<split>.json`; JSON `config.renorm`, `config.renorm_stats`. Dolgu (NaN→0) aynı fiziksel değerde kalır.
  `--renorm none` (varsayılan) eski kodla bit bit aynı tahmin (mini H5'te doğrulandı); `train` ile eğitimdeki yeni z ort. |≤ 0,04|, s.s. 0,97–1,01.
- `run_all.py` modalitesi `B_full_trainnorm`; `analyze_preds.py` ORDER/LABEL sonuna (varsayılan çiftlere EKLENMEDİ → R8 aileleri değişmez).
  Birincil fark ayrı çağrıyla: `analyze_preds.py --pairs B_full_trainnorm-B_full --out artifacts_v3/analysis/r13`.
- **Sonuç (20 Kas, Windows, 3 × ~50 dk; koşu 17:24'te, ön kayıt commit'i 17:11'de):** sabitler 18 413 train satırından; eski z'nin eğitim
  ortalaması −1,82…+1,13, s.s. oranı 1,27–3,99. **B_full_trainnorm − B_full test_time −0,0004 [−0,0011, +0,0002]** (yalnız SYNOP −0,0005
  [−0,0015, +0,0004]); test_station +0,0002 [−0,0005, +0,0008]; test_both −0,0001 [−0,0014, +0,0012]. Tohum başına test_time 0,9557 / 0,9564 /
  0,9557. **Ön kayıt kuralı: "sonuçlar normalizasyon sabitlerine duyarsız"** → Yöntem'e bir cümle + ek tablo satırı; ana tablo değişmez.
  JSON'larda git `b24b716` + dirty (çalışma ağacındaki kod sonraki `9a7e37e` commit'iyle aynı). summary.* yalnız 7 satır ekledi, eski satırlar birebir.

### K54 — 20 Kas 2026: revizyon Gün 5 — R7 kar ve VZA kırılımı (ön-kayıtlı)
- **`src/snow_vza_breakdown.py` → `analysis/snow_vza.csv`, `snow_vza_diff.csv`, `snow_vza_stations.csv`:** verify test kümeleri + birleşimi
  ("tüm test", gün blokları), B_full / D_unet / D_unet_jit (tohum başına) ve CLM / NWC (merkez, r30; ortak kapsamda fark). n < 300 → belirsiz.
- **Kar imzalı gündüz açık** (VIS0.6 ≥ %30, NIR1.6 ≤ 0,4·VIS0.6; n 185–199 → kural gereği "belirsiz" işaretli): yanlış alarm B_full %1,2,
  CLM r30 %99,5, NWC r30 %4,9.
- **Kış gündüzü (Ara–Mar), istasyon > 1500 m** (n 2 963–2 997, açık 336–356): BalAcc B_full 0,946, CLM r30 0,540, NWC r30 0,916; açıkların
  yanlış alarm oranı 0,064 / 0,914 / 0,147. B_full − CLM r30 +0,406 [+0,372, +0,435], − NWC r30 +0,029 [+0,004, +0,061]. Kıyas: kış ≤ 1500 m
  CLM 0,924; ilkbahar–yaz > 1500 m CLM 0,927 (model ile fark yok, +0,002 [−0,023, +0,023]) → CLM sorunu kar mevsimine özgü.
  Açık raporlar 9 istasyondan (17096 171 / 356); CLM merkez yanlış alarm 9 istasyonun 8'inde ≥ 0,87, model ≤ 0,13 → tek istasyona dayanmıyor.
  **İstasyon düzeyinde bulgu** (R10'un harita düzeyi "hipotez" kararı ayrı; ikisi birlikte yazılacak).
- **VZA (istasyon dilimleri < 55,6° / 55,6–58,5° / ≥ 58,5°; istasyonlar 51,6–64,4°):** B_full 0,943 / 0,959 / 0,947; CLM r30 0,917 / 0,915 / 0,859;
  NWC r30 0,917 / 0,927 / 0,919. CLM en yüksek VZA diliminde düşüyor (dilim doğuyla karışık; betimsel).
- **Teyit (21 Kas, `.agent/rules/rule_fci_clm.md` ölçeğiyle):** eşleme kurala uygun (1 ve 8 = açık, 2 ve 3 = bulutlu). Ama ~7,8 milyon
  istasyon-merkez CLM değerinde ve r30 alanlarında **sınıf 8 (kar/buz, açık) hiç yok** → ürün kar sınıfı üretmiyor. EUMETSAT doğrulama
  raporu (EUM2045564; `uydu ile ilgili makale/FCI_Level_2_product_validation_report…pdf`) bunu yazıyor: kar/buz, toz ve kül testleri
  kapalı, "no detection of snow/ice", karlı açık pikseller bulutlu sınıflanabilir (bilinen anomali, düzeltme bekliyor; NIR2.2 kullanılmıyor).
  Kış > 1500 m açık raporlarda CLM merkez durumu O döneminde %77 opak (3), %18 kısmi (2), %6 açık (1); C döneminde (Mar 2025) %39 / %23 / %38.
  Aynı istasyonlarda Nis–Haz açıklarda %81–84 açık → konum/indeks hatası yok. **Sonuç:** karışıklık bizim eşlememizden değil, ürünün
  belgelenmiş sınırlılığından; metinde EUMETSAT raporuna atıfla yazılır.

### K55 — 21 Kas 2026: keşifsel CLM kar duyarlılığı; makale Word çıktısı (Mendeley alanlı)
- **Keşifsel (ön kayıtta yok; `docs/revizyon_onkayit.md` §3, 21 Kas 09:40, yorum kuralı yok) `src/clm_snow_exclusion.py` →
  `analysis/clm_snow_exclusion.csv`:** model − referans farkı, kış (Ara–Mar) > 1500 m satırları çıkarılarak. Yalnız SYNOP, test_time:
  CNN − CLM r30 +0,045 → +0,041 (kış gündüzü > 1500 m hariç, 267 satır) → +0,040 (tüm kış > 1500 m hariç); test_station +0,040 → +0,035 →
  +0,034; GA'lar sıfırı dışlamaya devam ediyor. NWC farkı değişmiyor (+0,019). → CLM'ye üstünlüğün yalnız ~%10–15'i kar satırlarından;
  asıl fark başka koşullarda da var.
- **Word (kullanıcı kararı):** LaTeX ana kaynak kalır (verify_numbers yalnız .tex'i denetler); Word her seferinde üretilir.
  `paper/tools/build_docx.py` (+ `mendeley_fields.lua`, `springer-basic-author-date.csl`) → `paper/submission/word/{TAAC,MAP}_{manuscript,ESM}.docx`.
  Düzleştirme (\input, \ifTAAC, makrolar, \ref numaraları son LaTeX derlemesinin `*_final.aux`'ından) → pandoc ≥ 3.1 + citeproc
  (Springer Basic author-date) → Lua süzgeci her atıfı Mendeley alan koduna (`ADDIN CSL_CITATION {…itemData…}`) ve kaynakçayı
  `ADDIN Mendeley Bibliography CSL_BIBLIOGRAPHY` alanına çevirir. Kullanıcı Mendeley web + Mendeley Cite (Word eklentisi) kullanıyor:
  Cite, eski eklenti alanlarını dönüştürmeyi önerir; önce refs.bib Mendeley kitaplığına aktarılmalı. Biçim: Times New Roman 12 pt, 1,5 aralık,
  satır numaraları, tablolar içeriğe göre sütun genişliği (9 pt). Basit satır içi matematik ($-$, $>$, $p$) Unicode metne çevrilir.
- **Sıra:** önce `python tools/build.py --journal both --final` (aux güncellensin), sonra `python tools/build_docx.py --journal both`.
  Windows'ta bir kez: `pip install pypandoc_binary python-docx` (pandoc 3.x ile gelir). İlk üretim: atıf alanı TAAC 44 / MAP 39,
  kaynakça 43 / 37 (LaTeX .bbl ile aynı), tablo 4 / 5, şekil 8 / 9; çözülmemiş \ref 0. `paper/` .gitignore'da → commit yok.

### K56 — 22 Kas 2026: revizyon Gün 5 metin — R1 (Öz, Sonuç) ve R2 (ürün tanımı)
- **TAAC Öz (249 kelime):** tek skor yerine aralık (görülmemiş haftalar 0,956; bloklu CV katları 0,931–0,959); "her etiket tanımında"
  ifadesi; R12 cümlesi (kalibre edilince NWC'ye karşı fark 0,007, CLM'ye karşı 0,044); CLM kar cümlesi (kar testi kapalı, > 1500 m'de
  kış gündüzü açıkların %91'i bulutlu); doğu yaylaları harita iddiası Öz'den çıkarıldı (R10: hipotez). **MAP Öz (249):** "çoklu karşılaştırmaya
  göre düzeltilmiş" eklendi; uzamsal düzen katkısı (0,0016) BH'de düştüğü için çıkarıldı; CV aralığı ve R12 cümlesi eklendi.
- **Sonuç:** CV aralığı, R12 ayrıştırması (NWC farkı çoğunlukla hizalama; CLM farkı kalibrasyonla açıklanmıyor, kar ancak küçük bir kısmı),
  TAAC paragrafı R7 istasyon kanıtı + R10 belirsizliği + Kuter2026 ile.
- **R2:** Yöntem §3.8 başına ürün tanımı paragrafı (çıktı = o pikseldeki bir istasyonun ≥ 5 okta bildirme olasılığı; istasyon ölçekli bulut
  oluşum ürünü, 1,7 × 1,1 km piksel bulut örtüsü değil; "cloud mask" = 0,5 eşiğindeki ikili biçim). Fig. 9 altyazısına aynı ayrım
  (make_fig09 betiği + txt + tex). Bulgular 4.x (fullmap) ve Tartışma 5.5: "istasyon ağı doğrulayamaz" → R7 istasyon kanıtı + R10 eşik
  duyarlılığı. Başlık alternatifleri kullanıcıya sunuldu (seçim kullanıcıda).
- Eski sürümler `paper/_prev/*_20261122.tex`. verify_numbers 369/0; LaTeX 0 tanımsız; Word yeniden üretildi. Toplam karakter 84 976 (hedef 80 000).

### K57 — 23 Kas 2026: revizyon Gün 5 metin — R16, R19, R12/R13/R18 cümleleri
- **Tartışma 5.1:** R12 kalibrasyon paragrafı (NWC CT 0,928 → 0,940, CLM 0,899 → 0,903, ikisi birlikte 0,941; CNN 0,947, fark 0,006
  [0,002, 0,009]); NWC farkı çoğunlukla hizalama, CLM farkı ürünün kendisinde; keşifsel kar analizi (0,045 → 0,040, "ön kayıtta yok").
- **R16 (5.2):** "Neden derin ağ?" paragrafı — istasyonda HGB + doku en az CNN kadar iyi (BH sonrası anlamlı); ağın gerekçesi ürün (U-Net tek
  geçişte harita, komşuluk özniteliklerini öğreniyor); öneri: nokta doğrulamada HGB referans, haritada U-Net.
- **R19 Sınırlılıklar** genişletildi: Haziran IR kayması (1,5σ, skor düşmüyor), gözlemci uyumu (42 havalimanı, 0,956, NSC tanımı), holdout'ların
  hepsi kendi AWOS'lu + piksel–AWOS medyan 9,7 km (%95 20,1 km), val kümesinin bulutlu ağırlığı, yayımlanmış DL maskeleri yeniden eğitilmedi.
- **Veri 2.x:** IDW dışarıda-bırak (t2m 1,83 K / düzeltmesiz 2,44 K, rh2m %12,7). **Yöntem 3.4:** normalizasyon sabitleri (5 sahne, IR'de 1,8σ'ya
  kadar sapma) ve yeniden eğitim sonucu (−0,0004 [−0,0011, +0,0002]).
- verify_numbers 387/0; LaTeX 0 tanımsız (TAAC 33, MAP 34 sayfa); Word yeniden üretildi. Eski sürümler `paper/_prev/*_20261123.tex`.

### K58 — 23 Kas 2026: R8 metne işlendi — "significant" yalnız BH sonrası
- **Yöntem §3.6:** p = 2(k+1)/(B+1); üç ön-kayıtlı aile (model–ürün 48, tek bölme 108, CV 10); Benjamini–Hochberg (birincil) ve Holm;
  "significant" = BH sonrası anlamlı, aksi halde "nominal"; örtüşen 1–5 günlük bloklarda model–ürün sonuçları aynı. refs.bib'e Benjamini1995,
  Holm1979 (verified = {no}, yöntem kaynağı).
- **Bulgular:** "nominal" oldu: hgb_aux − B_full (test_time), RESNET_CIFAR − A_sat (test_both), hgb_texture − hgb_current, B_full − B_full_shuf,
  C_noawos − A_sat ve C_t2m − C_noawos (test_station), A_sat − A_norad (test_both), B_full − B_norad (test_time). "Significant" kalan:
  B_full − C_noawos (test_station), B_full − hgb_texture (test_time), bütün CV farkları (tümü), model–ürün 48 farkın hepsi (products.tex'e cümle).
  Tartışma 5.2: uzamsal yapı "yalnız nominal".
- verify_numbers 387/0; LaTeX TAAC 34, MAP 34 sayfa; Word yeniden üretildi. Eski sürümler `paper/_prev/*_20261123b.tex`.

### K59 — 23 Kas 2026: revizyon ek tabloları (7 yeni) ve ek materyal numaraları
- **`paper/figures/make_tables_revision.py`** (make_all'a eklendi) → `paper/tables/`: tabS8_contingency (R6), tabS9_label_checks (R20 + R17),
  tabS10_calibrated (R12 + keşifsel kar), tabS11_multiplicity (R8: aileler, BH/Holm, örtüşen blok), tabS12_robustness (R13, P20, R18, R11, R9),
  tabS13_snow_vza (R7; n < 300 hücresi † ile), tabS14_freq_sensitivity (R10). Dosya adları kaynak; makaledeki numara ilk atıf sırasından.
- **Numaralama:** preamble.tex `\ST…` makroları ilk atıf sırasına göre yeniden yazıldı. TAAC: S1 paralaks, S2 sağlamlık, S3 etiket duyarlılığı,
  S4 etiket denetimleri, S5 maliyet, S6 çoklu karşılaştırma, S7 model çiftleri, S8 doku, S9 CV, S10 kontenjans, S11 CLM dönemi, S12 kar/VZA,
  S13 sıklık duyarlılığı, S14 kalibre referans. MAP: aynı sıra, maliyet yok (ana metinde Tablo). supplementary.tex aynı sırada.
  Sıra kontrolü: build_docx düzleştirme fonksiyonlarıyla ilk `\ST…` geçişi (sıra değişirse yeniden hesaplanmalı).
- Metin: "Supplementary Material" geçen yerler numaralandı; products.tex'e kontenjans cümlesi, breakdowns.tex'e VZA cümlesi (CNN 0,943–0,959,
  CLM en yüksek VZA diliminde 0,859), Yöntem 3.1'e ≥6 okta / doğrulanmamış METAR atfı.
- verify_numbers 390/0; LaTeX 0 tanımsız, 0 taşan kutu (ESM TAAC 11, MAP 10 sayfa); Word yeniden üretildi. Toplam karakter 89 967.

### K60 — 24 Kas 2026: karakter kısaltması (hedef 80 000) tamamlandı
- Yöntem (23,3 → 19,0 kB), Tartışma (12,5 → 9,5 kB), Veri (13,4 → 10,8 kB), Giriş ve results/fullmap kısaltıldı; yalnız ifade
  sıkıştırıldı, sayı/atıf/etiket çıkarılmadı (iki sahne cümlesi tek cümlede birleşti). Yedekler `paper/_prev/*_20261124.tex`.
- Sonuç: TAAC 89 967 → **79 910** karakter (ana metin 60,5k → ~50,5k), MAP 83 634; öz 249 kelime. verify_numbers 390/0;
  final derleme TAAC 31 + 11, MAP 32 + 10 sayfa, 0 tanımsız, 0 taşan kutu, 0 \todo; Word (build_docx) iki dergi için yeniden üretildi.

### K61 — 24 Kas 2026: danışman atıfları, Word biçim denetimi, revizyon soru–cevap dosyası
- **Danışman çalışmaları** (`docs/paper/hocalar/*.csv`, yalnız başlık + öz okundu → refs.bib `verified = {abstract}`; yedek `refs.bib.bak_20261124`):
  SekerErtugrul2025 (Kodaz; kanal tabanlı füzyon + U-Net → Yöntem/girdiler), Dinsel2026 (Kodaz; rastgele kırpma → §3.8 jitter),
  Karadag2018 (Erdaş; derin özniteliklerin kaymaya duyarlılığı → §3.8), Kocer2017 (Koçer, öz-atıf; CNN doğruluk–hız → Tartışma §5.2),
  Jama2025 (Hacıbeyoğlu; uzamsal/kanal dikkat → Sınırlılıklar). Konu dışı görünebilir → hakem isterse kaldırılır (soru–cevap G4).
- **Word:** TAAC kılavuzu Word ister, dergi şablonu yok (≥ 11 pt, ≥ 1,5 aralık, satır no, otomatik sayfa no, şekiller metin içinde,
  "do not use field functions"). build_docx'e alt bilgide sayfa numarası eklendi. Mendeley alanları gönderimden önce Mendeley Cite ile düz metne çevrilmeli.
- `docs/paper/revizyon_soru_cevap_2026-11-24.md`: olası hakem soruları (A–G) + İngilizce yanıt taslakları + dikkat edilecek riskler (H).
- Derleme: TAAC 32 + 11, MAP 33 + 10 sayfa, 0 tanımsız; verify_numbers 390/0; toplam ~80 400 karakter.

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
| `src/make_public_sample.py` | İki katmanlı tabakalı örneklem (a: FCI+statik+etiket, b: MGM radar/AWOS) + bölme; `--merge`, `--check` (K38) |
| `src/make_zenodo_package.py` | Zenodo yükleme paketi + MANIFEST (sha256) + metadata.json (K38) |
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
| `src/infer_scene.py` | P4 tam harita çıkarımı: U-Net faz-hizalı / CNN kayan pencere → NetCDF + GeoTIFF + PNG; `--check` H5/preds tutarlılığı (K27) |
| `src/select_scenes.py` | P5 sahne seçimi (ölçütle, görüntüsüz) → `scenes_selected.csv` (K28) |
| `src/build_scene_pack.py` | P5 sahne paketi: FCI görüntüsü + model + CLM + NWC CT aynı ızgarada (K28) |
| `src/batch_scenes.py` | P6 toplu tam sahne çıkarımı (3 saatte bir, U-Net + CLM; kaldığı yerden) (K30) |
| `src/cloud_frequency.py` | P6 bulut oluşum sıklığı NetCDF + istasyon doğrulaması (K30) |
| `src/references_by_period.py` | P7 FCI CLM C/O dönemi ve "contaminated" eşleme duyarlılığı (K32) |
| `src/export_center_values.py` | Kompakt H5 → istasyon pikseli ham değerleri (`center_values.npz`) (K33) |
| `src/error_analysis.py` | P8 hata analizi: gece, nem, NWC sınıfı, okta, kar (K33) |
| `src/predict_rows.py` | Kayıtlı v3 ağırlıklarıyla H5 satırlarına tahmin (P9 keep ekleri; güvenlik kontrollü) (K34) |
| `src/label_sensitivity.py` | P9 etiket tanımı duyarlılığı (verify / keep / 0–≥1 okta) (K34) |
| `src/parallax_shift.py` | P10 paralaks kaydırması (istasyon × CTH, ızgara 6 km) (K34) |
| `src/summarize_scenes.py` | Sahne JSON'ları → `scenes_summary.csv` (tutarlılık, faz deseni, süre; torch gerekmez) |
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
| `src/revision_metrics.py` | Revizyon R6/R17/R20: kontenjans metrikleri, doğrulanmamış METAR açıkları, ≥6 okta — K49 |
| `src/inference_robustness.py` | Revizyon R8: BH/Holm, örtüşen blok bootstrap — K49 |
| `src/reference_calibrated_baseline.py` | Revizyon R12: CLM/NWC oranlarıyla kalibre kıyas modeli (HGB/LR) — K50 |
| `src/observer_ceiling.py` | Revizyon R11: SYNOP–METAR gözlemci uyumu, okta başına hata — K50 |
| `src/awos_colocation.py` | Revizyon R9: holdout ↔ AWOS ko-lokasyonu, piksel–AWOS mesafesi, okta-9 — K51 |
| `src/idw_loo.py` | Revizyon R18: t2m/rh2m IDW leave-one-out — K51 |
| `src/normalization_shift.py` | Revizyon R13: ay × aydınlanma kanal kayması (σ) — K51 |
| `src/cloud_freq_sensitivity.py` | Revizyon R10: doğu yaylaları sıklık farkının eşik/kalibrasyon/maske duyarlılığı — K51 |
| `src/snow_vza_breakdown.py` | Revizyon R7: kar imzası / kış yüksek istasyon / VZA kırılımı, istasyon başına yanlış alarm — K54 |
| `src/clm_snow_exclusion.py` | Keşifsel: kış > 1500 m satırları olmadan model − CLM/NWC farkı — K55 |
| `paper/tools/build_docx.py` | LaTeX → Word (.docx), Mendeley atıf alanlarıyla; TAAC/MAP ana metin + ESM (paper/ depoda değil) — K55 |
| `src/test_center_jitter.py` | P20 jitter birim testleri + P20 öncesi kodla bit bit eşitlik (mini H5) — K46 |
| `src/train.py`, `src/train_unet.py` | Eğitim; `--h5 --split --aux --aux_channels --seed --epochs --limit_batches --results --outdir --model` + K16: `--loader --mem_splits --cache_dir --threads --no_deterministic --no_last_preds` |
| `src/config.py` | Colab/yerel yollar, `MTGCLM_H5`, PATCH_SIZE 33, BATCH_SIZE 64, LR 1e-3 |
| `paper/main.tex`, `preamble.tex`, `frontmatter.tex`, `backmatter.tex`, `sections/`, `supplementary.tex` | Makale LaTeX iskeleti (sn-jnl, sn-basic; `\ifTAAC`) — K22 |
| `paper/tools/verify_numbers.py`, `derived_facts.py` | `% src:` yorumlarını CSV hücreleriyle denetler; metin için ek kaynak CSV'leri üretir — K23 |
| `src/channels.py` | Doğrulanmış H5 kanal sırası (tek kaynak), dalga boyları, P3 grupları — K24/K25 |
| `src/channel_importance.py` | P3: permütasyon önemi + Integrated Gradients, kayıtlı tahminlerle güvenlik kontrolü — K25 |
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
- [x] ~~**K12'deki AWOS-aux bulgusunu doğrula/açıkla**~~ — K18 (CV'de küçük pozitif). Kalan: IDW doğrulaması → revizyon R18.
  _Eski metin:_ **K12'deki AWOS-aux bulgusunu doğrula/açıkla**: neden B_full (AWOS dahil) C_noawos'tan
      (AWOS'suz) düşük? IDW enterpolasyon kalitesi, normalizasyon, ya da gerçekten bilgi
      katkısı yok mu — ek analiz/ablasyon (farklı IDW parametreleri, AWOS yoğunluk haritası
      görselleştirme) gerekebilir.
- [x] ~~Bloklu CV'nin tek-bölme test_time'dan daha düşük/değişken çıkmasını makalede tartış~~ — K41 (`cv_weeks.csv`).
  _Eski metin:_ Bloklu CV'nin tek-bölme test_time'dan daha düşük/değişken çıkmasını makalede tartış
      (K12) — hangi haftalar/katlar düşürüyor, incele.
- [x] ~~Uydu-only (radar hariç) ablasyonu için kanal seçimi~~ — K25 (`train.py --base_channels`; koşu bekliyor).
- [x] ~~FCI CLM karşılaştırması: çıkarım `src/extract_fci_clm.py` ile (K13); aynı örneklerde merkez + alan bazlı, C/O dönemleri ayrı.~~ — K14, K18, K32.
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
- [x] ~~Bölüm 2.2: "5–30 km komşuluk" ifadesi~~ — K34'te "merkez ve 30 km" olarak düzeltildi.
- [x] ~~Analiz raporunu makale bulgularına dönüştür; gece hataları~~ — K33, K35.
- [ ] **P20 (revizyon R14):** kod + testler hazır (K46); kalan: 3 tohum eğitimi (`--stage unetjit`) ve değerlendirme. U-Net hedef-piksel jitter — **33×33 içinde 25×25 rastgele kırpma**, kayıp gerçek konumda; H5 yeniden
      çıkarılmaz (eski "49×49 çıkar" maddesi K20/K44 ile değişti). Fig. 9'un yeniden üretim kuralı `docs/revizyon_onkayit.md`'de.
- [x] ~~`infer_scene.py`~~ — yazıldı ve gerçek sahnelerde doğrulandı (K27).
- [x] ~~Kanal permütasyon önemi / Integrated Gradients~~ — K26 (aydınlanma içi permütasyon).
- [x] ~~Metrikleri gündüz/gece ve mevsim kırılımıyla raporla~~ — K18, K21 (Fig. 6).
- [x] ~~Açık örneklemi üret ve Zenodo paketini hazırla~~ (K38).
- [ ] `make_zenodo_package.py --include_mgm` ile yeniden paketle ve Zenodo'ya yükle (MGM sözlü onayı K42; **yazılı izin süreci 18 Kas'ta başlatıldı**).
- [ ] Zenodo kaydını **yayımla** (DOI 10.5281/zenodo.24112650 yalnız rezerve, taslak kayıt — yayımlanana kadar çözümlenmez);
      yayımlanınca oluşan **concept DOI**'yi makale, README, CITATION.cff, kapak mektupları ve release notlarına yaz (K44).
- [ ] **Revizyon (K44):** `docs/paper/calisma_plani_revizyon_2026-11-18.md` takvimi; ön-kayıt `docs/revizyon_onkayit.md`.
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
- `claude/mtgclm_madde_madde_degerlendirme_2026-11-18.md`, `claude/mtgclm_revizyon_calisma_plani_2026-11-18.md` — hakem raporu
  ve ret riski yol haritasının madde madde değerlendirmesi ve revizyon takvimi (K44). Yerel kopyaları `docs/paper/` altında (depo dışı).

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
- Mimari/modalite tartışırken **Medium_Flat**'in kazanan mimari olduğunu (K6) ve AWOS-aux'un **küçük ama
  tutarlı pozitif** katkısını unutma: CV birleşik B − C +0,0027 [+0,0015, +0,0041] (5/5 kat), test_station +0,0031;
  tek bölmenin test_time'ında fark yok (K18; K12'deki "katkı yok" yargısı düzeltildi) — v1 sayılarını güncel
  karşılaştırma için kullanma.
- Yeni bir karar alındığında bu dosyanın §2'sine ekle (Kural 2 / `.agent/rules/rule_update_claude_md.md`).
- İş paketi bitince dosya başına ayrı İngilizce commit komutları üret, çalıştırma (Kural 1 / `.agent/rules/rule_commit_per_file.md`).
- Anlamlı her değişiklikte `docs/mtgclm_infografik.html`'i güncelle; her kod değişikliğinden sonra `python src/make_code_map.py`
  ile `docs/mtgclm_kod_haritasi.html`'i yeniden üret (Kural 3 / `.agent/rules/rule_update_infographic.md`).
- **Tekrarlanabilirlik kuralı (20 Kas):** makaleye ya da CLAUDE.md'ye giren her sayı depodaki bir `src/` betiğinin yazdığı bir dosyadan
  gelir; geçici `python -c` / awk denetimleri yalnız kontrol içindir, kaynak sayılmaz. Bulut ortamında (VM) koşulan analizler
  kullanıcının `mtgclm` ortamında bir kez yeniden koşulur; README "Revizyon analizleri" tablosuna her yeni betik eklenir.
- FCI CLM `cloud_state` sınıf tablosu `.agent/rules/rule_fci_clm.md`'de (0 işlenmedi, 1 açık, 2 kısmi, 3 opak, 4–7 toz/kül, 8 açık-kar/buz, 9 tanımsız);
  CLM ile ilgili her analizden önce okunur. Mevcut ürün sürümü 8 üretmiyor (K54).
- `.agent/` ve `archive_v1/` .gitignore'da; bu dosyalar için commit komutu verme, yerelde kalır.
