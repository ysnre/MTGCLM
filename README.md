# MTGCLM — Meteosat Third Generation ile Türkiye üzerinde bulut maskeleme

MTG-I1 FCI'ın 16 spektral kanalı, TMM C-bandı radar kompozitleri ve yer gözlemlerini birleştirerek
Türkiye üzerinde bulut varlığını tahmin eden derin öğrenme hattı. Etiketler 147 insanlı sinoptik
istasyon ve 79 havaalanı (METAR) gözleminden; yardımcı meteorolojik alanlar 1982 otomatik
gözlem istasyonundan (AWOS) enterpolasyonla üretilir.

> Durum: geliştirme aşamasında. Sonuçlar ve sürümler için `walkthrough_v2.md` dosyasına bakın.

## İçindekiler

- [Kurulum](#kurulum)
- [Veri](#veri)
- [Hattı yeniden üretme](#hattı-yeniden-üretme)
- [Eğitim ve değerlendirme](#eğitim-ve-değerlendirme)
- [Çapraz doğrulama](#çapraz-doğrulama)
- [Karşılaştırma modelleri](#karşılaştırma-modelleri)
- [Veri bulunabilirliği](#veri-bulunabilirliği)
- [Atıf](#atıf)
- [Lisans](#lisans)

## Kurulum

```bash
git clone https://github.com/<kullanıcı>/MTGCLM.git
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

## Veri bulunabilirliği

MTG FCI verisi EUMETSAT Data Store üzerinden ücretsiz kayıtla herkese açıktır. Radar ve yer
gözlemleri Meteoroloji Genel Müdürlüğü'ne aittir ve kurum izniyle kullanılmıştır; ham hâlleri bu
depoda paylaşılmamaktadır.

Bu depoda bulunanlar: veri setini ham kaynaklardan yeniden üretmek için gereken **tüm kod**,
bölme dosyaları (`splits/*.npz`, ~2 MB) ve koşu sonuçları (`artifacts/*.json`). Türetilmiş
veri seti (`mtg_train_v2.h5`, 19 GB) GitHub boyut sınırlarını aştığından Zenodo'da arşivlenir;
DOI yayımlandığında buraya eklenecektir. Ayrıca doğrulama ve hızlı deneme için ~2.000 yamalık
bir örneklem (`mtg_sample.h5`, ~100 MB) `src/make_public_sample.py` ile üretilebilir.

Eğitilmiş model ağırlıkları (`best_model_*.pth`) sürüm (release) eklerinde yayımlanır.

## Atıf

Bu kodu ya da veri setini kullanırsanız `CITATION.cff` dosyasındaki bilgileri kullanın.

## Lisans

Kod MIT lisansı altındadır (`LICENSE`). Veri lisansları ilgili kurumlara aittir; EUMETSAT
verisi EUMETSAT Data Policy, radar ve yer gözlemleri MGM koşullarına tabidir.
