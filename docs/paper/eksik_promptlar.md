# MTGCLM — Eksiklerin tamamlanması için hazır promptlar

Sürüm 1 · 2026-10-30. Hedef: Claude Code / Cowork oturumu, çalışma dizini `E:\belgeler\MTGCLM`.
Her promptun başına **P0**'ı ekleyin (ya da oturumun ilk mesajı olarak bir kez verin). Promptlar birbirinden
bağımsızdır; bağımlılıklar "Önkoşul" satırında yazılıdır. Süreler i7-12700 CPU, `--threads 12`, H5 NVMe'de
(`C:/mtg/mtg_train_v2.h5`) varsayımıyla CLAUDE.md K16'dan türetilmiştir.

Önerilen sıra: P11 → (P1, P2, P3 gece koşusu) → P4 → P5 → P6 → P7–P10 → P12 → P13 → P14 → P15 → P16 → P17 → P18 → P19.

---

## P0 — Oturum açılışı (her oturumda bir kez)

```
CLAUDE.md'yi baştan sona oku; özellikle §2 K4 (sızıntı), K11, K13, K16–K18 ve §7 çalışma kurallarına uy.
.agent/rules/ altındaki kuralları uygula (dosya başına İngilizce commit komutu üret ama çalıştırma;
CLAUDE.md'yi güncelle; infografiği güncelle). Makale hedefi: Theoretical and Applied Climatology (birincil)
ve Meteorology and Atmospheric Physics (yedek), ikisi de Springer. Şekil üretirken docs/paper/sekil_stil_kurali.md
kuralına ve paper/figures/springer_fig.py yardımcılarına uy. Betikleri her zaman --h5 ile çağır, H5 yolu
C:/mtg/mtg_train_v2.h5. Varsayılan bölme splits/split_v2_verify.npz, CV splits/cv5_verify_fold{k}.npz,
çıktı artifacts_v3/. Model seçimi yalnız val'de; test kümelerine yalnız eğitim bitince bir kez dokun.
Uzun koşuları run_all.py üzerinden başlat. Türkçe yanıt ver. Bir şey belirsizse koşuyu başlatmadan önce sor.
Kararlar (CLAUDE.md K20): tam harita ürünü U-Net (D_unet), nokta tablolarında odak model CNN tüm girdiler (B_full);
makale metni LaTeX + İngiliz yazımı (centre, colour, -ise); makale dokümanları docs/paper/ altında.
```

---

## P1 — Radarsız (uydu-only) ablasyon

```
Amaç: radar kanalının ve radar_cov'un katkısını ölçmek (radar MGM'ye ait, Türkiye dışına taşınamıyor).

1) src/h5_dataset.py ve src/fast_loader.py'ye base_channels parametresi ekle (17 temel kanal içinden alt küme,
   isimle: FCI kanal adları + "radar"). Varsayılan = tüm 17 kanal → mevcut koşular birebir aynı kalmalı;
   bunu mini H5 üzerinde tensör eşitliği testiyle doğrula. Normalizasyon istatistikleri seçilen kanallara göre
   alt kümelensin. train.py ve train_unet.py'ye --base_channels argümanı ekle; config'e ve sonuç JSON'una yaz.
2) run_all.py'ye yeni modaliteler ekle:
   - A_norad: 16 FCI kanalı, aux yok
   - B_norad: 16 FCI kanalı + aux (t2m, rh2m, elev, cos_sza), radar_cov HARİÇ
   Her biri 3 seed, Medium_Flat, split_v2_verify. Ayrıca CV_Bnorad_f0..4_s1 (cv5_verify).
   Süre tahmini: 6 × ~26 dk + 5 × ~29 dk ≈ 5 sa. Gece koşusu olarak başlat: --stage modality,cv --only A_norad,B_norad,CV_Bnorad.
3) Bitince analyze_preds.py'yi yeniden çalıştır ve eşleştirilmiş farklara şu çiftleri ekle:
   A_sat − A_norad (radarın saf katkısı), B_full − B_norad (aux varken radarın katkısı), CV birleşik B − B_norad.
   Gündüz/gece ve radar kapsamı içi/dışı (radar_cov > 0 / = 0) kırılımlarını da üret.
4) Çıktı: artifacts_v3/analysis/ altında güncel pairs.csv, cv_pairs.csv, breakdowns.csv; kısa özet
   (radar katkısı, GA, hangi koşullarda büyük). CLAUDE.md'ye K19 olarak işle.
Kabul ölçütü: mevcut A/B koşularının sonuçları değişmemeli; yeni JSON'larda config.base_channels doğru.
```

## P2 — HGB kontrollü kıyas ve uzamsal doku testi

```
Amaç: "HGB ≈ CNN" bulgusunu kontrollü sınamak — uzamsal doku CNN'e bir şey katıyor mu?

1) src/baselines.py'de load_features'a öznitelik seti seçeneği ekle (--feature_set):
   - center1: yalnız merkez piksel (16,16), 22 kanal
   - center5: merkez 5×5 ortalaması (mevcut özniteliklerin bir alt kümesi)
   - current: mevcut set (yama ortalaması + merkez 5×5 + aux) → hgb_aux ile birebir aynı sonucu vermeli
   - texture: current + her kanal için yama std, 10/90 yüzdelik, merkez–yama farkı, Sobel gradyan büyüklüğü ortalaması
   HGB'yi bu dört setle tüm veride koş (hgb_center1, hgb_center5, hgb_current, hgb_texture). Eşik/hiperparametre yalnız train/val'de.
2) CNN uzamsal karıştırma kontrolü: train.py'ye --spatial_shuffle bayrağı ekle. Her örnekte 33×33 piksellerin
   konumları rastgele permüte edilsin (tüm kanallara aynı permütasyon; merkez piksel (16,16) yerinde kalsın).
   Bu, spektral dağılımı korur, uzamsal yapıyı yok eder. B_full_shuf_s{1,2,3} olarak koş (~1,3 sa).
3) analyze_preds.py ile eşleştirilmiş farklar: B_full − B_full_shuf, hgb_texture − hgb_current,
   hgb_current − hgb_center1, B_full − hgb_texture. test_time, test_station, test_both; gündüz/gece kırılımı.
4) Kısa yorum: uzamsal doku hangi koşulda (gece? kenar bulutları? parçalı bulut test_partial?) katkı veriyor.
Kabul ölçütü: hgb_current == mevcut hgb_aux (aynı seed, aynı sayılar).
```

## P3 — Kanal önem analizi (permütasyon + Integrated Gradients)

```
Amaç: hangi girdiler bulut kararını taşıyor; fiziksel yorum (MAP için zorunlu, TAAC'ta ek).

Yeni betik src/channel_importance.py (yalnız numpy, torch, pandas; captum KULLANMA, IG'yi elle yaz):
1) Permütasyon önemi: B_full_s{1,2,3} en iyi val checkpoint'leri, test_time üzerinde. Her kanal (22) ve
   fiziksel gruplar için: VIS/NIR (0.4–1.6 µm), 2.2 µm, 3.8 µm, su buharı (6.3/7.3), pencere IR (8.7/10.5/12.3),
   9.7 ozon, 13.3 CO2, radar, aux-AWOS (t2m, rh2m), aux-statik (elev, cos_sza, radar_cov). Permütasyon kanal
   bazında örnekler arası karıştırma (yama içi yapı korunur), 5 tekrar. Metrik: BalAcc düşüşü ± s.s.
   Gündüz (güneş yüksekliği > 10°), alacakaranlık, gece ayrı (analyze_preds.py'deki tanımı kullan).
   Kanal adlarını H5 attrs'tan oku; sabit kodlama yapma.
2) Integrated Gradients: 5.000 örneklik tabakalı alt küme (gündüz/gece × sınıf), taban çizgisi = normalize
   uzayda sıfır (kanal ortalaması), 32 adım. Kanal başına |IG| toplamı, merkez 5×5 ve yama geneli ayrı.
   Tamlık (completeness) kontrolü: sum(IG) ≈ f(x) − f(baseline), sapma raporlansın.
3) Çıktılar: artifacts_v3/analysis/channel_importance.csv, ig_summary.csv ve yayın şekli
   paper/figures/make_fig07_channel_importance.py ile Fig07 (174 mm, iki panel: a permütasyon gündüz/gece,
   b IG gündüz/gece; kanallar dalga boyuna göre sıralı; renkler springer_fig.COLORS'tan değil, gruplara göre
   tek bir sıralı/kategorik palet; sekil_stil_kurali.md §4).
4) Fiziksel tutarlılık notu: gece 3.8 µm ve 10.5 µm'nin, gündüz VIS 0.6/NIR 1.6'nın baskın çıkıp çıkmadığı;
   t2m katkısının BT10.5 − T2m kontrastıyla ilişkisi (K14).
Süre: permütasyon 22 kanal × 5 tekrar × 3 seed çıkarım ≈ 2–3 sa CPU; IG ≈ 30 dk.
```

## P4 — Tam harita çıkarımı: `infer_scene.py`

```
Amaç: tek bir zaman adımı için 800×1000 bulut olasılık haritası ve ikili maske üretmek.

1) src/infer_scene.py: girdi = zaman damgası (UTC). İçeride:
   - FCI NetCDF'i coknc.py'deki okuma + resample ile aynı grid'e getir (aynı kanal sırası, aynı ölçekleme).
   - Radar PNG → dBZ (mavi kanal, K11), radar_cov raster'ı.
   - aux: t2m/rh2m için o zamana ait AWOS gözlemlerinden IDW (update_h5_aux_channels.py'deki fonksiyonları
     yeniden kullan; floor zaman eşlemesi, lapse rate 6.5 K/km), elev = data/dem_800x1000.npy, cos_sza grid'den.
   - Normalizasyon: H5 attrs means/stds ve aux_means/aux_stds — eğitimle birebir aynı.
   - Model: U-Net (D_unet_s1, tam evrişimli, tek geçiş, kenarda yansıtmalı dolgu). Seçenek olarak Medium-Flat
     B_full ile kayan pencere (33×33, adım 1, batch'li) — hız raporlansın. Varsayılan U-Net.
   - Çıktı: NetCDF (lat, lon, cloud_prob, cloud_mask, kalite bayrağı: girdi eksikliği), GeoTIFF (EPSG:4326) ve PNG önizleme.
     İkili eşik val'de seçilen eşik (0.5 değilse JSON'dan oku).
2) Tutarlılık testi: istasyon piksellerinde infer_scene çıktısı ile preds/ dosyasındaki aynı örnek olasılığı
   karşılaştır (aynı zaman, aynı istasyon) → mutlak fark dağılımı; medyan < 1e-3 beklenir. Fark büyükse
   aux üretimi veya normalizasyonda uyumsuzluk vardır, durup raporla.
3) Kenar artefaktı kontrolü: 33 piksellik ızgara deseninde süreksizlik var mı (U-Net merkez-piksel denetimiyle
   eğitildi). Varsa P20'yi öner.
4) Süre: sahne başına CPU süresi ve bellek; Tab. 5 için kaydet.
Önkoşul: yok. Çıktı dizini: artifacts_v3/scenes/.
```

## P5 — Nitel sahne paneli (Fig. 8)

```
Amaç: modelin tam harita çıktısını operasyonel ürünlerle yan yana göstermek.

1) Yalnız TEST haftalarından (split_v2_verify test_time günleri) 5 sahne seç; seçimi otomatik ve gerekçeli yap:
   gündüz geniş açık alan, gece alçak bulut/sis (istasyonda okta 9 veya RH ≥ %90), karla kaplı dağlık alan
   (Doğu Anadolu, kış), ince sirrüs (NWC CT yarı saydam sınıfı), derin konveksiyon (radar > 40 dBZ).
   Seçim ölçütlerini ve seçilen zamanları scenes_selected.csv'ye yaz. Sahneleri ben görmeden "güzel" diye seçme.
2) Her sahne için 4 sütun: FCI RGB (gündüz: doğal renk; gece: 10.5 µm BT, ters gri), model olasılığı (cividis 0–1),
   FCI CLM (ikili eşlemeyle), NWC SAF CT (ikili). İstasyon gözlemleri üstüne işaret (açık = içi boş daire,
   bulutlu = dolu daire; renk değil şekil). Ortak kıyı çizgisi ve sınırlar.
3) paper/figures/make_fig08_scenes.py, springer_fig ile 174 mm, en fazla 234 mm yükseklik; satır başlıkları
   (zaman UTC + sahne türü) şeklin solunda, panel harfleri a–t. Kind="combination" (600 dpi TIFF).
Önkoşul: P4.
```

## P6 — Bulut sıklığı haritaları (TAAC için zorunlu, Fig. 9)

```
Amaç: Türkiye için bulut oluşum sıklığının aylık ve günün saatine göre dağılımı; CLM ile karşılaştırma.

1) infer_scene.py'yi toplu çalıştır: tüm dönemde 3 saatte bir (00, 03, …, 21 UTC) sahne. Eksik girdili
   sahneleri kalite bayrağıyla dışla ve sayısını raporla. Kesilirse kaldığı yerden devam etsin.
   Önce süreyi tahmin et (sahne sayısı × P4 süresi) ve 6 saati aşıyorsa bana sor.
2) Sıklık = bulutlu sahne sayısı / geçerli sahne sayısı, piksel başına: (a) ay başına (Mar, Nis 2025; Ara 2025–Haz 2026),
   (b) günün saati başına (8 dilim), (c) tüm dönem. Aynı hesap FCI CLM için (aynı zamanlar, aynı geçerlilik maskesi).
3) Doğrulama: istasyon noktalarında model sıklığı ile SYNOP sıklığı (aynı zamanlar) dağılım grafiği + bias.
4) Şekil: make_fig09_cloud_frequency.py, 174 mm: a tüm dönem model sıklığı, b CLM − model farkı (RdBu_r, 0'da
   merkezli), c–d seçili iki ay (Ocak, Haziran), e günlük döngü (bölge ortalamaları: kıyı, iç, doğu dağlık).
   Ek şekil olarak tüm aylar ve tüm saatler.
5) NetCDF olarak kaydet (artifacts_v3/climatology/cloud_freq_*.nc) — Zenodo'ya da gidebilir.
Not: 7 aylık dönem bir klimatoloji değildir; metinde "cloud occurrence frequency during the study period" ifadesi kullanılacak.
Önkoşul: P4.
```

## P7 — FCI CLM commissioning (C) / operasyonel (O) dönem ayrımı

```
eval_clm_on_splits.py ve analyze_preds.py --refs çıktılarını dosya adındaki C/O bayrağına göre iki döneme ayır
(Mar–Nis 2025 = C, Ara 2025–Haz 2026 = O; K13). Her dönem için yalnız SYNOP, kapsam eşleşmiş örneklerde:
CLM merkez ve r30 BalAcc, model BalAcc, fark ve gün-blok bootstrap GA. Ayrıca ikili eşleme duyarlılığı:
"cloud contaminated" sınıfı bulutlu / açık / dışarıda bırakılmış üç seçenek. Çıktı: references_by_period.csv ve
2–3 cümlelik özet. Model–CLM farkının yalnız C döneminden kaynaklanıp kaynaklanmadığını açıkça söyle.
```

## P8 — Gece / sis / alçak bulut hata analizi

```
B_full_s{1,2,3} preds dosyalarından test_time ve test_station hatalarını incele:
1) Hata oranı: güneş yüksekliği dilimleri (5° adım), RH dilimleri, okta (0, 5, 6, 7, 8, 9), yükseklik kuşakları,
   ay; gece için ayrıca NWC CT sınıfına göre (aynı zamanlı eşleşmede).
2) Yanlış "açık" (FN) ve yanlış "bulutlu" (FP) ayrı; gece FN'lerde alçak bulut/sis payı.
3) Kar: Ara–Mar, yükseklik > 1500 m, gündüz FP oranı.
4) 10 temsilî hata vakası (rastgele, tohum sabit) için merkez piksel kanal değerleri ve istasyon gözlemi tablosu.
Çıktı: error_analysis.csv, kısa Türkçe özet, ek şekil (Supplementary). Yeni eğitim yok.
```

## P9 — Etiket politikası duyarlılığı (yeniden eğitimsiz)

```
v3 modellerini (verify etiketleriyle eğitilmiş) iki ek etiket tanımıyla değerlendir, yeniden eğitim yapma:
(a) keep: METAR NSC/CAVOK doğrulanmadan açık kabul (split_v1 test kümelerindeki ek METAR satırları),
(b) okta eşiği: mevcut tanım 0 / ≥5; alternatif 0 / ≥1 tanımında test_partial (1–4 okta) satırları bulutlu
    sayılarak test kümelerine eklenir (model ve referanslar aynı satırlarda; ikili eşik val'de seçilen eşik).
Aynı satırlarda CLM ve NWC de değerlendirilsin. Tablo: etiket tanımı × model/referans × BalAcc (GA).
Amaç: yer gerçeği tanımının skoru ve model–referans farkını ne kadar kaydırdığını göstermek. Satır kümeleri
arasındaki farkı (n) açıkça yaz. Çıktı: label_sensitivity.csv.
```

## P10 — Paralaks kaydırma büyüklüğü

```
Amaç: düzeltme yapılmadığı için kaydırmanın büyüklüğünü sayısal vermek (sınırlılık + r30 gerekçesi).
1) Her istasyon için MTG-FCI (0° boylam alt nokta) görüş geometrisinden VZA ve azimut (K13: VZA 51,6–64,7°).
2) Bulut tepe yüksekliği 1, 3, 6, 10 km için yatay kaydırma (km) ve piksel cinsinden (0,02° × 0,01°) — satpy'nin
   paralaks fonksiyonu varsa onu kullan, yoksa küresel Dünya geometrisiyle analitik hesap; ikisi tutarlı olmalı.
3) Tablo: istasyon dağılımı (min/medyan/maks) × CTH. Harita: 6 km için kaydırma büyüklüğü (Supplementary).
4) Yorum: kaydırmanın 33×33 yama (≈57 × 37 km) ve r30 alanı içinde kalıp kalmadığı.
Çıktı: parallax_shift.csv, ek şekil. Eğitim yok.
```

## P11 — Şekil stil altyapısını repoya kur

```
Stil altyapısı 30 Eki 2026'da repoya kuruldu (K20): docs/paper/sekil_stil_kurali.md,
paper/figures/springer_taac_map.mplstyle, paper/figures/springer_fig.py, .agent/rules/rule_figure_style.md.
Dosyaların yerinde olduğunu ve `python -c "import sys; sys.path.insert(0,'paper/figures'); import springer_fig"`
komutunun çalıştığını doğrula; Arial yazı tipinin matplotlib tarafından bulunduğunu kontrol et. Ardından demo olarak Fig. 5'i (referans kıyası, references.csv,
yalnız SYNOP) paper/figures/make_fig05_references.py ile üret ve check raporunu göster.
.gitignore: paper/figures/*.pdf|tif|png çıktıları dışarıda, betikler ve stil dosyaları içeride.
```

## P12 — Yayın şekilleri ve tablolar (mevcut sonuçlardan)

```
paper/figures/ altında, her biri bağımsız çalışan betiklerle üret (girdi yalnız artifacts_v3/ ve data/):
- make_fig01_study_area.py — DEM altlığı (gist_earth, açık ton), SYNOP (daire) / METAR (üçgen) istasyonları,
  30 holdout istasyon (içi boş, kalın kenar), radar kapsamı (radar_cov = 0 sınırı kontur), il sınırları yok,
  ülke sınırları ve kıyı çizgisi; 174 mm; cartopy PlateCarree; 5° enlem-boylam ızgarası.
- make_fig04_main_results.py — üç test kümesi, modeller BalAcc ± %95 GA (nokta + çizgi), çoğunluk sınıfı referans çizgisi.
- make_fig06_breakdowns.py — gündüz/alacakaranlık/gece × ay ısı haritası (hücre değerleri yazılı) + yükseklik kuşakları.
- make_fig10_calibration.py — güvenilirlik eğrisi + test_partial okta başına ortalama olasılık.
- make_tables.py — Tab. 2–5'i LaTeX (booktabs) ve CSV olarak; sayılar 3 ondalık; GA köşeli parantez.
Renkler ve etiketler springer_fig.COLORS/LABELS'tan; her betik save_figure() ile PDF+TIFF+PNG ve check raporu üretir.
Tüm check raporları OK olmadan bitirme. Şekil başlığı yazma; altyazı metnini her betik figXX_caption.txt olarak üretsin
(n, küme, etiket alt kümesi, GA yöntemi altyazıda).
```

## P13 — LaTeX iskeleti (TAAC + MAP tek kaynak)

```
paper/ altında Springer Nature LaTeX şablonu (sn-jnl.cls, sn-article.tex; resmi şablonu Springer'ın sayfasından
indirmemi iste, kendin uydurma) ile iskelet kur:
- main.tex: \newif\ifTAAC \TAACtrue anahtarı; başlık, öz, anahtar kelimeler, Giriş ¶1 ve son ¶, Tartışma vurgusu
  ve kapak mektubu dergiye göre iki sürüm. Gövde bölümleri sections/*.tex (ortak).
- Kaynak stili: derginin submission guidelines sayfasındaki "References" bölümüne göre (author-year ise sn-basic).
  Önce sayfayı kontrol etmemi iste; varsayım yapma.
- refs.bib: literatur_konumlandirma.md'deki DOI'lerden Crossref ile BibTeX üret; her girdiye "verified = {no}"
  notu düş (P16'da doğrulanınca yes).
- figures/ ve tables/ P12 çıktılarına bağlansın; Makefile / latexmk komutu; karakter sayımı betiği
  (texcount + tablo/altyazı/kaynak dahil, TAAC 100.000 sınırına karşı).
```

## P14 — Veri, Yöntem ve Bulgular taslağı (İngilizce)

```
paper/sections/ altına 02_data.tex, 03_methods.tex, 04_results.tex taslaklarını İngilizce yaz.
Kaynak: CLAUDE.md (K1–K18 + yeni K'ler), artifacts_v3/analysis/analysis.md ve CSV'ler,
makale_bolumleri_yapilacaklar.md §2–§4. Kurallar:
- Her sayı bir CSV hücresinden gelsin; sayının yanına LaTeX yorumu olarak kaynağını yaz (% src: main_table.csv r7 BalAcc).
- "1 km" yazma; piksel ≈ 1.73 × 1.11 km. Paralaks düzeltmesinin uygulanmadığını açıkça yaz.
- Bulgular TAAC sırasıyla (bölüm tablosu); MAP sırası için \ifTAAC ile alt bölüm sırası değişsin.
- İddia dili ölçülü: "significantly higher (95% block-bootstrap CI excludes zero)" gibi; "outperforms" yalnız GA sıfırı dışlıyorsa.
- Yapay zekâ destekli yazım/kod araçlarının kullanımını 03_methods.tex sonunda bir cümleyle beyan et.
Bitince eksik sayı/şekil gerektiren yerleri \todo{} ile işaretle ve listesini ver.
```

## P15 — Giriş, Tartışma, Sonuç (iki dergi sürümü)

```
literatur_konumlandirma.md (§3 yakın çalışmalar, §4 katkı ifadesi, §5 Giriş iskeleti, §8 dergiye özgü atıflar)
ve makale_bolumleri_yapilacaklar.md §1, §5, §6'ya göre 01_introduction.tex, 05_discussion.tex, 06_conclusions.tex yaz.
\ifTAAC ile: Giriş ¶1 ve son ¶, Tartışma 5.5 (TAAC: bulut sıklığı ve izleme) / 5.6 (MAP: kanal fiziği).
Yalnız verified = {yes} kaynaklara atıf yap; diğerlerini \todo{verify} ile bırak.
Öz (150–250 kelime) iki sürüm; kelime sayısını yaz.
```

## P16 — Literatür tam metin doğrulaması

```
refs.bib'deki her kaynak için (öncelik: literatur_konumlandirma.md §3 ve §8) tam metni aç (açık erişim ya da
kurum erişimi; bulamazsan bana listele). Her biri için 3 satır not: (1) ne yaptılar (veri, bölge, yöntem, etiket kaynağı),
(2) ana sayı, (3) bizim makalede hangi cümleyi destekliyor. Desteklemiyorsa o atıfı kaldır. Notları
docs/paper/literature_notes.md'ye yaz ve refs.bib'de verified = {yes} yap. Ayrıca TR Dizin ve YÖK Tez'de
"bulut maskesi uydu Türkiye" araması yapmam için arama terimlerini öner.
```

## P17 — Tekrarlanabilirlik paketi

```
1) src/make_public_sample.py'yi çalıştır (--n 2000, split_v2_verify, tabakalı: küme × sınıf × gündüz/gece).
   Veri politikası docs/paper/makale_yol_haritasi.md §5'e göre iki katmanlı: katman (a) FCI + statik alanlar
   (elev, cos_sza) + SYNOP/METAR etiketleri; katman (b) radar kanalı, radar_cov, t2m, rh2m. Her iki katmanı ayrı
   dosya olarak üret (mtg_sample_core.h5, mtg_sample_mgm.h5). MGM teyidinin gelip gelmediğini bana sor; gelmediyse
   yalnız (a) Zenodo paketine girsin ve README'de (b)'nin talep üzerine verildiği yazsın.
2) Zenodo yükleme paketi: örneklem, splits/*.npz|json, artifacts_v3/*.json, artifacts_v3/analysis/*.csv,
   checkpoints (B_full_s1, D_unet_s1), climatology NetCDF (P6). metadata.json taslağı.
3) README: "Reproducing the paper" bölümü (her şekil ve tablo için tek komut). CITATION.cff güncelle.
4) Git etiketi önerisi v1.0-taac; release notları taslağı. Commit komutlarını dosya başına üret, çalıştırma.
```

## P18 — Kapak mektubu ve önerilen hakemler

```
İki kapak mektubu yaz (TAAC ve MAP; İngilizce, ≤ 1 sayfa): katkı (3 madde), kapsamla uyum gerekçesi (TAAC: applied
meteorology / cloud monitoring; MAP: cloudy-atmosphere processes and statistical techniques), veri dönemi kısıtının
dürüst açıklaması, veri/kod erişimi, çıkar çatışması yok beyanı. Önerilen 5 hakem: literatur_konumlandirma.md
yakın çalışmalarının yazarlarından, ortak yazarlık veya kurumsal çakışma olmayan; her biri için gerekçe ve
kurum e-postası kaynağı. E-postaları uydurma; bulamadığın yerde boş bırak.
```

## P19 — Gönderim öncesi son kontrol

```
paper/ derlenmiş PDF'i üzerinde: öz kelime sayısı (150–250), TAAC karakter sayımı (≤ 100.000, her şey dahil),
tüm şekillerin check raporu OK, şekil/tablo atıf sırası, kısaltmaların ilk geçişte açılması, ondalık ayırıcı nokta,
"Türkiye" yazım tutarlılığı, kaynakların derginin biçiminde olması ve DOI'lerin çözülmesi, beyanların eksiksizliği,
Supplementary dosyasının numaralandırması (Fig. S1…). Bulguları madde madde listele; düzeltmeyi bana bırak.
```

## P20 — (Opsiyonel) U-Net hedef-piksel jitter

```
Yalnız P4'te kenar/ızgara artefaktı görülürse. CLAUDE.md §7 yamaların yeniden çıkarılmasını yasaklar
(create_patch_dataset.py çalıştırılmaz), bu yüzden 49×49 seçeneği yok. Mevcut 33×33 yamalar içinde çalış:
eğitimde 25×25 rastgele kırpma (hedef piksel merkezden ±4 piksel kayar), MaskedBCELoss hedef konumu buna göre
kaysın; değerlendirmede merkez kırpma. D_unet_jit_s{1,2,3}; tam harita artefaktı ve nokta metrikleri
D_unet ile karşılaştırılsın. Başlamadan önce süre tahmini ver ve onay iste (U-Net CPU'da ~2 sa/koşu; Colab T4 ~39 dk).
```
