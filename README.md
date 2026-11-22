# MTGCLM — Meteosat Third Generation ile Türkiye üzerinde bulut maskeleme

MTG-I1 FCI'ın 16 spektral kanalı, TMM C-bandı radar kompozitleri ve yer gözlemlerini birleştirerek
Türkiye üzerinde bulut varlığını tahmin eden derin öğrenme hattı. Etiketler 147 insanlı sinoptik
istasyon ve 79 havaalanı (METAR) gözleminden; yardımcı meteorolojik alanlar 1982 otomatik
gözlem istasyonundan (AWOS) enterpolasyonla üretilir.

> Durum: v3 sonuçları tamamlandı (`artifacts_v3/`). Akış için `walkthrough_v2.md`, kararlar için `CLAUDE.md`.

## İçindekiler

- [Kurulum](#kurulum)
- [Veri](#veri)
- [Hattı yeniden üretme](#hattı-yeniden-üretme)
- [Eğitim ve değerlendirme](#eğitim-ve-değerlendirme)
- [Çapraz doğrulama](#çapraz-doğrulama)
- [Karşılaştırma modelleri](#karşılaştırma-modelleri)
- [v3 sonuçlarını yeniden üretme](#v3-sonuçlarını-yeniden-üretme)
- [Açık örneklemle hızlı deneme](#açık-örneklemle-hızlı-deneme)
- [Veri bulunabilirliği](#veri-bulunabilirliği)
- [Atıf](#atıf)
- [Lisans](#lisans)

## Kurulum

```bash
git clone https://github.com/ysnre/MTGCLM.git
cd MTGCLM
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.x. GPU zorunlu değil ama eğitim için önerilir (T4 üzerinde epoch ≈ 3 dk).

## Veri

| Kaynak | İçerik | Dönem | Erişim |
|---|---|---|---|
| EUMETSAT MTG-I1 FCI | 16 kanal, Türkiye kesiti 800×1000 (0,02°×0,01°) | Mar–Nis 2025, Ara 2025–Haz 2026 | EUMETSAT Data Store (ücretsiz kayıt) |
| TMM C-bandı radar | 19 radar, PPI yansıtırlık (dBZ), 5 dk | aynı | Meteoroloji Genel Müdürlüğü |
| SYNOP + METAR | `cloud_coverage` (okta), 226 istasyon | aynı | Meteoroloji Genel Müdürlüğü |
| AWOS | 10 dk sıcaklık ve bağıl nem, 1982 istasyon | aynı | Meteoroloji Genel Müdürlüğü |
| DEM | Yükseklik, grid'e örneklenmiş | — | Copernicus DEM GLO-90 (açık) |

Türetilmiş veri seti (`mtg_train_v2.h5`, ~19 GB): 400.209 yama × 22 kanal × 33×33, float16,
global permütasyonla karıştırılmış. Yapısı:

```
patches      [N,17,33,33] float16   16 FCI kanalı + radar (kanal bazında z-skor)
aux          [N, 5,33,33] float16   t2m, rh2m, elev, cos_sza, radar_cov
labels       [N] int64              0 = açık (0 okta), 1 = bulutlu (≥5 okta), -1 = yok sayılan
okta         [N] int8               ham bulut miktarı (0–9)
wmoids, timestamps, station_y/x, obs_time, obs_dt_sec, is_dup, source_index
attrs: means, stds, aux_channels, aux_means, aux_stds, label_policy, pre_shuffled
```

## Hattı yeniden üretme

Ham veriden başlayarak (ayrıntılar ve parametreler için `walkthrough_v2.md`):

```bash
python src/process_radar_pipeline_large.py     # radar PNG → uydu grid'inde kompozit .npy
python src/process_awos_pipeline_large.py      # SYNOP/METAR → uydu anlarına hizalı CSV
python src/create_patch_dataset.py             # 33×33 yamalar → mtg_patch_dataset.h5
python src/make_dem_npy.py --dem_tif <dem>.tif --out data/dem_800x1000.npy
python src/update_h5_aux_channels.py --h5 <h5> --awos_dir <...> --dem data/dem_800x1000.npy
python src/update_h5_labels_v2.py  --h5 <h5> --awos_aligned_dir data/processed/awos_aligned
python src/make_splits.py          --h5 <h5> --out splits/split_v1 --seed 42 --holdout_stations 30
python src/export_compact_h5.py    --src <h5> --split splits/split_v1.npz --out mtg_train_v2.h5
```

Tüm adımlar deterministiktir (seed 42); ara dosyalar silinirse aynı çıktılar yeniden üretilir.

## Eğitim ve değerlendirme

```bash
H5=mtg_train_v2.h5; SP=splits/split_v1_compact.npz
python src/train.py      --h5 $H5 --split $SP --model mtg_flat --aux --seed 1 --results artifacts/full_s1.json
python src/train_unet.py --h5 $H5 --split $SP --model unet     --aux --seed 1 --results artifacts/unet_s1.json
python src/aggregate_results.py --dir artifacts --out artifacts/summary --plot
```

Değerlendirme alt kümeleri sızıntıya karşı ayrılmıştır: `test_time` (görülmemiş haftalar),
`test_station` (görülmemiş istasyonlar), `test_both` (ikisi birden), `test_partial` (1–4 okta,
etiketsiz tutarlılık kontrolü). Raporlanan metrikler: balanced accuracy, MCC, sınıf bazında
precision/recall/F1, ROC-AUC, karışıklık matrisi ve çoğunluk sınıfı referansı.

## Çapraz doğrulama

```bash
python src/make_cv_splits.py --h5 $H5 --out splits/cv5 --folds 5 --block_days 7 --seed 42
for k in 0 1 2 3 4; do
  python src/train.py --h5 $H5 --split splits/cv5_fold$k.npz --model mtg_flat --aux --seed 1 \
      --results artifacts/cv_f${k}_s1.json
done
python src/aggregate_results.py --dir artifacts --pattern "cv_*.json" --out artifacts/cv_summary
```

Katlar rastgele değil **bloklu**: ardışık günler 7 günlük bloklara ayrılır ve bloklar katlara
dağıtılır, böylece aynı sinoptik durum hem eğitimde hem testte yer almaz.

## Karşılaştırma modelleri

```bash
python src/baselines.py --h5 $H5 --split $SP --aux \
    --models majority,threshold,threshold2,logreg,rf,hgb --results artifacts/baselines.json
```

Çoğunluk sınıfı, tek ve çift kanal parlaklık sıcaklığı eşikleri (klasik operasyonel yaklaşım),
lojistik regresyon, rastgele orman ve gradyan artırma; hepsi aynı bölme ve metriklerle.
Derin modeller tarafında MTGConvNet (Shallow/Medium/Deep × GAP/Flatten), 17 kanala uyarlanmış
ResNet-18 ve U-Net karşılaştırılır.

> **Anahtar adı uyarısı.** `artifacts_v3` sonuç dosyalarında ve Zenodo kaydında çift eşik modeli
> `threshold2_ir_105+vis_06` adını taşır, ama bu koşu **IR10.5 ve IR13.3** kanallarını kullandı
> (kanal sırası sonradan doğrulandı; adı kayıt için değiştirilmedi). Makalede "IR10.5 ve IR13.3" yazılıdır.
> Güncel kodda `--ch2 vis_06` gerçek VIS0.6'yı seçer, yani farklı bir modeldir.

## v3 sonuçlarını yeniden üretme

Güncel sonuçlar `artifacts_v3/` altındadır (etiket politikası *verify*: `splits/split_v2_verify.npz`, bloklu CV
`splits/cv5_verify_fold{0..4}.npz`). Her çıktı tek komutla yeniden üretilir; `--h5` her zaman kompakt veri setini gösterir.

| Çıktı | Komut |
|---|---|
| Modalite, U-Net, ResNet, CV, karşılaştırma modelleri (`artifacts_v3/*.json`, `preds/`) | `python src/run_all.py --h5 <h5> --outdir artifacts_v3 --stage v3 --threads 12` |
| Radarsız ve uzamsal karıştırma ablasyonları; HGB öznitelik setleri | `python src/run_all.py --h5 <h5> --outdir artifacts_v3 --stage modality,cv --modalities A_norad,B_norad,B_full_shuf --cv_variants Bnorad` ; `--stage hgbfs` |
| Toplu tablo | `python src/aggregate_results.py --dir artifacts_v3 --out artifacts_v3/summary --plot` |
| Ana tablo, eşleştirilmiş farklar, CV, kırılımlar, referanslarla kıyas (`analysis/*.csv`) | `python src/analyze_preds.py --dir artifacts_v3 --out artifacts_v3/analysis --refs` |
| Kanal önemi (permütasyon + integrated gradients) | `python src/channel_importance.py --h5 <h5> --dir artifacts_v3 --threads 12` |
| FCI CLM dönem (C/O) ve "cloud contaminated" duyarlılığı | `python src/references_by_period.py` |
| Gece / sis / kar hata analizi | `python src/export_center_values.py --h5 <h5>` → `python src/error_analysis.py` |
| Etiket tanımı duyarlılığı (keep, ≥1 okta) | `python src/predict_rows.py --h5 <h5> --threads 12` → `python src/label_sensitivity.py` |
| Paralaks kaydırması | `python src/parallax_shift.py` |
| Tam sahne haritaları | `python src/select_scenes.py --h5 <h5>` → `python src/infer_scene.py --h5 <h5> --time <YYYYMMDDhhmm> --check` → `python src/summarize_scenes.py --dir artifacts_v3/scenes` |
| Bulut oluşum sıklığı (NetCDF) | `python src/run_all.py --h5 <h5> --outdir artifacts_v3 --stage climatology` (`batch_scenes.py` + `cloud_frequency.py`) |

**Revizyon analizleri** (ön kayıt: `docs/revizyon_onkayit.md`; hepsi yukarıdaki çıktıları okur, yeniden eğitim yoktur; varsayılan
yollar depo köküne göredir):

| Analiz | Komut | Çıktı |
|---|---|---|
| U-Net hedef-piksel jitter (P20 / R14) | `python src/run_all.py --h5 <h5> --outdir artifacts_v3 --stage unetjit` | `D_unet_jit_s*.json`, `preds/` |
| Kontenjans metrikleri, doğrulanmamış METAR açıkları, ≥ 6 okta (R6, R17, R20) | `python src/revision_metrics.py --dir artifacts_v3` | `analysis/contingency.csv`, `unverified_metar_clear.csv`, `label_ge6.csv` |
| Çoklu karşılaştırma (BH / Holm) ve blok uzunluğu (R8) | `python src/inference_robustness.py --dir artifacts_v3` | `analysis/multiplicity.csv`, `block_length_sensitivity.csv` |
| Referans ürünlerin aynı etiketlerle kalibrasyonu (R12) | `python src/reference_calibrated_baseline.py --dir artifacts_v3` | `analysis/reference_calibrated.csv` |
| SYNOP–METAR gözlemci uyumu (R11) | `python src/observer_ceiling.py --dir artifacts_v3` | `analysis/observer_ceiling*.csv` |
| Holdout ↔ AWOS ko-lokasyonu, okta-9 (R9) | `python src/awos_colocation.py --dir artifacts_v3` | `analysis/awos_colocation*.csv`, `awos_pixel_distance.csv`, `okta9_rh.csv` |
| AWOS IDW leave-one-out (R18) | `python src/idw_loo.py` | `analysis/idw_loo.csv` |
| Ay × aydınlanma kanal kayması (R13) | `python src/normalization_shift.py` | `analysis/normalization_shift*.csv` |
| Z-skor sabitleri eğitim kümesinden yeniden eğitim (R13 ADIM 3) | `python src/run_all.py --h5 <h5> --outdir artifacts_v3 --stage modality --modalities B_full_trainnorm` → `python src/analyze_preds.py --dir artifacts_v3 --out artifacts_v3/analysis/r13 --pairs B_full_trainnorm-B_full` | `B_full_trainnorm_s*.json`, `analysis/r13/pairs.csv` |
| Kar ve görüş açısı (VZA) kırılımı (R7) | `python src/snow_vza_breakdown.py` | `analysis/snow_vza*.csv` |
| Keşifsel: kış > 1500 m satırları olmadan model − CLM/NWC farkı | `python src/clm_snow_exclusion.py` | `analysis/clm_snow_exclusion.csv` |
| Bulut sıklığı duyarlılığı (R10) | `python src/cloud_freq_sensitivity.py` | `climatology/cloud_freq_sensitivity.csv` |

R9 ve R18 ham AWOS dosyasını (`data/raw_awos_observation/`) okur; ilk koşu istasyon kayıt sayılarını
`data/processed/awos_station_counts.csv`, R18 örneklemini `data/processed/idw_loo_sample.npz` olarak önbelleğe alır.
`center_values.npz` için önce `src/export_center_values.py --h5 <h5>` çalıştırılmalıdır.

Referans ürün çıkarımları (FCI CLM, NWC SAF CT) ham ürün dosyalarını gerektirir: `src/extract_fci_clm.py`,
`src/extract_nwcsaf_ct.py`, ardından `src/eval_clm_on_splits.py` ve `src/eval_nwc_and_metar.py`.

## Açık örneklemle hızlı deneme

`src/make_public_sample.py` kompakt veri setinden tabakalı (küme × sınıf × gündüz/gece) ~2.000 satırlık bir örneklem
üretir, iki dosya hâlinde: `mtg_sample_core.h5` (katman a: 16 FCI kanalı, yükseklik, cos SZA, etiketler) ve
`mtg_sample_mgm.h5` (katman b: radar, `radar_cov`, t2m, rh2m). Satır sırası ortaktır; `--merge` tam biçimi geri kurar.

```bash
python src/make_public_sample.py --h5 <h5> --split splits/split_v2_verify.npz --out data/public/mtg_sample --n 2000
python src/train.py --h5 data/public/mtg_sample_core.h5 --split data/public/mtg_sample_split.npz \
    --model medium_flat --aux --epochs 3 --outdir artifacts_sample
python src/make_zenodo_package.py --out zenodo_package --sample data/public/mtg_sample   # Zenodo yükleme paketi
```

## Veri bulunabilirliği

MTG FCI verisi EUMETSAT Data Store üzerinden ücretsiz kayıtla herkese açıktır. Radar ve yer
gözlemleri Meteoroloji Genel Müdürlüğü'ne aittir ve kurum izniyle kullanılmıştır; ham hâlleri bu
depoda paylaşılmamaktadır.

Bu depoda bulunanlar: veri setini ham kaynaklardan yeniden üretmek için gereken **tüm kod**,
bölme dosyaları (`splits/*.npz`, ~2 MB) ve koşu sonuçları (`artifacts/*.json`). Tam eğitim veri seti
(`mtg_train_v2.h5`, 19 GB) MGM verisi içerdiği için **paylaşılmaz**; yerine ~2.000 satırlık tabakalı bir örneklem Zenodo'da arşivlenir:

> **Zenodo DOI (tüm sürümler):** [10.5281/zenodo.24112649](https://doi.org/10.5281/zenodo.24112649)

Türetilmiş veri seti iki katman hâlinde paylaşılır: **katman (a)** 16 FCI kanalı, statik alanlar
(yükseklik, cos SZA) ve SYNOP/METAR etiketleri; **katman (b)** radar kanalı, `radar_cov`, t2m ve rh2m — MGM verisinden
türetilmiştir. Zenodo kaydının güncel sürümünde katman (a) örneklemi (`mtg_sample_core.h5`), bölmeler, koşu sonuçları, analiz tabloları,
iki model ağırlığı ve bulut oluşum sıklığı haritaları bulunur. Katman (b) örneklemi (`mtg_sample_mgm.h5`, 9 MB) MGM'nin yazılı
onayından sonra aynı kayda yeni sürüm olarak eklenecek (`src/make_zenodo_package.py --out zenodo_package/v1.2 --version 1.2 --include_mgm`); o zamana kadar MGM izniyle
yazarlardan istenebilir. Ham radar ve yer gözlemleri talep üzerine MGM'den temin edilir.

Eğitilmiş model ağırlıkları (`best_model_*.pth`) sürüm (release) eklerinde yayımlanır.

## Atıf

Bu kodu ya da veri setini kullanırsanız `CITATION.cff` dosyasındaki bilgileri kullanın.

## Lisans

Kod MIT lisansı altındadır (`LICENSE`). Veri lisansları ilgili kurumlara aittir; EUMETSAT
verisi EUMETSAT Data Policy, radar ve yer gözlemleri MGM koşullarına tabidir.
