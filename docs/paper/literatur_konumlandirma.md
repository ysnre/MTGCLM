# MTGCLM — Literatür taraması ve konumlandırma

Tarih: 2026-10-27 · Güncelleme: 2026-10-30 (TAAC/MAP odaklı) · Kaynak kod/sonuçlar: `E:\belgeler\MTGCLM` (CLAUDE.md K1–K18, `artifacts_v3/`)
Üretilen dosyalar: `taac_map_relevant.csv`, `literature_screened.csv`, `literature_core_table.csv`, `key_papers.csv`,
`literature_landscape.png`, `lit_stats.json`, `crossref_raw.csv`, `arxiv_candidates.csv`

---

## 1. Tarama yöntemi (makalede "Methods — literature review" olarak yazılabilir)

- Kaynak: **Crossref REST API** (`api.crossref.org/works`), `query.bibliographic`, tür = dergi makalesi,
  yayın tarihi ≥ 2012-01-01. 26 ayrı sorgu dizesi (uydu bulut maskesi, derin öğrenme bulut tespiti,
  SEVIRI/ABI/AHI/MODIS/VIIRS/FCI, NWC SAF doğrulaması, yer gözlemiyle doğrulama, radar–uydu füzyonu,
  gece bulut tespiti, kar–bulut ayrımı, Türkiye).
- Ek kaynak: arXiv API (3 sorgu) — konuya özgü önbaskı neredeyse yok (9 kayıt, çoğu ilgisiz).
- Ham sonuç: **1.142 benzersiz kayıt** (DOI bazında tekilleştirilmiş).
- Eleme: "cloud computing" türü sahte eşleşmeler kural tabanlı atıldı; kalan kayıtların başlık+özeti
  LLM ile yapılandırılmış alanlara çevrildi (ilgi düzeyi, görev, sensör, platform, yöntem, **etiket/yer
  gerçeği kaynağı**, bölge, yer gözlemi kullanımı, radar kullanımı). **249 kayıt** elendi,
  bunların **153'ü** "core" (uydu görüntüsünden bulut/açık ayrımı), **54'ü** geostasyoner platformlu.
- **Sınırlılık (makalede belirtilmeli):** Crossref kayıtlarının bir kısmında özet yok; bu kayıtlarda
  sınıflama yalnız başlıktan yapıldı. Etiket kaynağı yalnız **68** core kaydında (geostasyonerde 31)
  güvenle belirlenebildi; aşağıdaki oranlar bu alt küme içindir. Nihai makale için bu çekirdek
  listenin tam metinle elle doğrulanması gerekir — tarama, taramanın kendisi bir iddia değil,
  **boşluğun gösterimi**dir.

---

## 2. Manzara: literatür nerede duruyor

![Literatür manzarası: etiket kaynağına göre taranmış bulut tespiti çalışmaları (sol) ve yıllara göre platform dağılımı (sağ)]({{artifact:art_41438dc8-a26b-4961-80f7-f0b24a300d30}})

| Yer gerçeği kaynağı | Tüm platformlar | Geostasyoner |
|---|---|---|
| Operasyonel uydu bulut maskesi ürünü (MOD35, CLM, vb.) | 28 | 15 |
| Uzay tabanlı aktif algılayıcı (CALIOP, CloudSat, DARDAR) | 20 | 7 |
| **Yer istasyonu gözlemi (SYNOP/METAR/seylometre)** | **9** | **6** |
| İnsan/elle görüntü etiketleme | 6 | 0 |
| Yer tabanlı bulut radarı | 3 | 3 |
| Tüm gökyüzü kamerası | 2 | 0 |

Üç gözlem:

1. **Baskın paradigma kendi kendini referans alıyor.** Etiket kaynağı belirlenebilen çalışmaların
   neredeyse yarısı **başka bir uydu bulut maskesi ürününü** yer gerçeği sayıyor. Bu, mevcut ürünün
   hatalarını (özellikle gece, sis, kar, ince sirrüs) yeni modele miras bırakıyor.
2. **İkinci paradigma uzay tabanlı aktif algılayıcı** (CALIOP/CloudSat). Bilimsel olarak güçlü ama
   kesişim dar (dar yörünge izi), zamansal örnekleme seyrek ve CALIPSO/CloudSat misyonları artık
   sona erdi — yeni nesil geostasyoner sensörler için ölçeklenebilir bir etiket kaynağı değil.
3. **Yer gözlemi, özellikle insanlı sinoptik okta gözlemi, eğitim etiketi olarak neredeyse hiç
   kullanılmıyor.** Bulunan 9 çalışmada yer gözlemi hemen daima **doğrulama** amaçlı; eğitim etiketi
   olarak kullanıldığı tek örnek bulut *tipi* sınıflandırması (ACP 2023, CERES geniş bantlı radyasyon,
   küresel) — bulut maskesi değil, görüntüleyici değil.

### MTG-FCI boşluğu

Taranan 1.142 kaydın **22'si** MTG/FCI'dan söz ediyor (2'si ilgisiz "FCI" kısaltması). İçerikleri:
Lightning Imager (6), IRS sounder (3), aerosol/kar/yangın/yağış ürünleri, uydu mühendisliği,
genel misyon tanıtımı. **Bulut maskesi** bağlamında tek kayıt:

- `10.1088/2515-7620/ae8153` (Environmental Research Communications, 2026) — Madeira Adası üzerinde
  58.760 FCI sahnesiyle bulut örtüsü **nowcasting**'i. Burada FCI CLM **girdi/hedef olarak tüketiliyor**;
  CLM'in kendisi bağımsız bir yer gerçeğine karşı sorgulanmıyor.

Yani: **FCI CLM ürünü literatürde kullanılmaya başlandı, ama bağımsız yer gerçeğine karşı hiç
doğrulanmadı ve FCI için yer gözlemiyle eğitilmiş bir alternatif maske hiç üretilmedi.** MTGCLM'in
boşluğu tam olarak budur.

---

## 3. En yakın önceki çalışmalar (tam metinle doğrulanmalı — `key_papers.csv`)

| # | Çalışma | Neden önemli | MTGCLM ile ilişki |
|---|---|---|---|
| 1 | `10.1088/2515-7620/ae8153` — MTG cloud mask nowcasting (ERC 2026) | FCI CLM kullanan tek bulut maskesi çalışması | **En yakın rakip.** Giriş'te mutlaka alıntılanmalı; farkı net belirtilmeli: onlar CLM'i veri olarak alıyor, biz CLM'i **değerlendiriyor ve yerine geçiyoruz** |
| 2 | `10.5194/acp-23-523-2023` — ML of cloud types (ACP 2023) | İnsan gözlemci raporlarını (GTS) eğitim etiketi yapan tek çalışma | Metodolojik öncül: "insan gözlemi → ML etiketi" yaklaşımının literatürdeki dayanağı |
| 3 | `10.1002/joc.5011` — SYNOP ile MODIS bulut miktarı tutarsızlığı (IJC 2017) | Yer gözlemi ↔ uydu maskesi ölçek/yorum uyuşmazlığının nicel analizi | **Etiket politikası bölümünün temel referansı** (0/≥5 okta eşiği, 1–4 okta ignore kararı) |
| 4 | `10.1002/2015jd023277` — MODIS maske yorumunun bulut miktarına etkisi (JGR 2015) | Maske sınıflarının bulut miktarına çevrilmesindeki belirsizlik | FCI CLM'in "cloud contaminated" sınıfının ikili eşlemesi (K13) için doğrudan dayanak |
| 5 | `10.1002/joc.8141` — Polonya bulut tipi: MODIS vs SYNOP (IJC 2023) | Bölgesel uydu–SYNOP karşılaştırmasının güncel örneği | Yöntem ve sunum şablonu; hedef dergi sinyali (IJC bu işi basıyor) |
| 6 | `10.1002/joc.4539` — Bölgesel MODIS bulut klimatolojisi (IJC 2015) | Bölgesel yüksek çözünürlüklü maske ürünü + SYNOP doğrulaması | Türkiye ölçeğinde bölgesel ürün savunması için emsal |
| 7 | `10.3390/rs15041141` — AHI gerçek zamanlı bulut örtüsü, PV (RS 2023) | İstasyon bulut miktarını doğrulama ölçütü yapıyor, operasyonel maskeyle kıyaslıyor | Bizim kıyas tasarımımızın (model vs operasyonel ürün, aynı istasyonlar) en yakın örneği |
| 8 | `10.1038/s41467-023-44666-1` — PV için bulut örtüsü nowcasting (Nat. Comm. 2024) | Geostasyoner + istasyon gözlemi, yüksek görünürlüklü örnek | Uygulama motivasyonu (güneş enerjisi) ve yöntem gerekçesi |
| 9 | `10.1029/2024jd041512` — GOES-16 + seylometre sığ kümülüs (JGR 2024) | Yer tabanlı nokta gözlemiyle piksel eşleme sorunları | Nokta↔piksel eşleme ve paralaks tartışması |
| 10 | `10.1109/tgrs.2022.3153129` — Himawari-8 derin öğrenme maske/sınıflama (TGRS 2022) | Geostasyoner görüntüleyici + derin öğrenme maskesinin referans uygulaması | Mimari/metrik karşılaştırma referansı |
| 11 | `10.1080/01431161.2019.1624863` — MSG IR3.9 ile ML maskeleme doğrulaması (IJRS 2019) | SEVIRI üzerinde ML maske, Orta Doğu denizi | Bölgesel ML maske emsali; coğrafi olarak en yakın |
| 12 | `10.1080/2150704x.2016.1249296` — NARX ile SEVIRI maske öngörüsü (IJRS 2016) | Meteosat ailesi + sinir ağı | Tarihsel bağlam |
| 13 | `10.3390/atmos14091410` — CloudSat etiketli RF ile AHI bulut sınıflaması (2023) | Klasik ML'in geostasyonerde CNN'e rakip olduğu örnek | **HGB≈CNN bulgumuzun literatürdeki karşılığı** — tartışma bölümünün dayanağı |
| 14 | `10.3390/rs12030365` — Paralaks kayması düzeltmesi: geostasyoner + radar (RS 2020) | Geostasyoner piksel ile yer radarı arasındaki paralaks kaydırması | **Sınırlılıklar bölümünün zorunlu referansı** (K13: düzeltme uygulanmadı, VZA 51,6–64,7°) |
| 15 | `10.5194/amt-19-6193-2026` — Derin konveksiyon için paralaks düzeltme CTH (AMT 2026) | Güncel paralaks düzeltme yaklaşımı | Aynı; "gelecek çalışma" olarak önerilebilir |
| 16 | `10.3390/rs18183202` — HAF-Net: uydudan radar yansıtırlığı rekonstrüksiyonu (RS 2026) | Uydu ↔ yer radarı ilişkisini **ters yönde** kuruyor | Radar kanalı kullanımımızın literatürdeki tek yakın akrabası |
| 17 | `10.5194/amt-17-961-2024` — AI ile 3B bulut tomografisi (AMT 2024) | SEVIRI + CloudSat, U-Net | Segmentasyon hattımızın emsali |
| 18 | `10.3390/rs16162989` — SEVIRI Optimal Cloud Analysis CDR (RS 2024) | EUMETSAT operasyonel ürün zinciri | Operasyonel ürün bağlamı |
| 19 | `10.7717/peerj.20495` — FCI ile kar örtüsü haritalama, ilk değerlendirme (PeerJ 2026) | FCI tabanlı ilk ürün değerlendirmelerinden biri | **Kar–bulut karışıklığı** sınırlılığımız için referans; ayrıca "FCI ilk değerlendirme" türü makalenin basılabildiğinin kanıtı |
| 20 | `10.3390/rs15245662` — MTG Day-2 yağış oranı için derin öğrenme (RS 2023) | MTG için ML tabanlı ürün geliştirmenin emsali | "MTG + derin öğrenme" hattının açık olduğunu gösteriyor |

---

## 4. Boşluk → katkı ifadesi (makalenin omurgası)

**Boşluk (üç katmanlı, her biri taramayla desteklenmiş):**

1. MTG-FCI için yer gözlemine dayalı, bağımsız bir bulut maskesi **yok**; FCI CLM kullanılmaya
   başlandı ama hiç bağımsız doğrulanmadı.
2. Uydu bulut maskesi literatüründe yer gerçeği baskın olarak **başka bir uydu ürünü** ya da
   artık sona ermiş **uzay tabanlı aktif algılayıcılar**; insanlı sinoptik okta gözlemi eğitim
   etiketi olarak kullanılmıyor.
3. Yer tabanlı **meteorolojik radar yansıtırlığı**, geostasyoner görüntüleyiciyle bulut tespiti
   için **girdi kanalı** olarak kullanılmamış (mevcut radar kullanımları ya uzay radarı etiketi ya da
   ters yön rekonstrüksiyon).

**Önerilen tek cümlelik katkı ifadesi:**

> Meteosat Third Generation FCI'ın 16 kanalını, yer tabanlı C-bandı meteorolojik radar yansıtırlığı ve
> yüzey yardımcı alanlarıyla birleştirerek, **insanlı sinoptik okta gözlemleriyle eğitilmiş** ilk bölgesel
> bulut maskesini sunuyor; aynı gözlem satırlarında EUMETSAT FCI CLM ve NWC SAF Cloud Type ürünlerinden
> istatistiksel olarak anlamlı biçimde daha yüksek dengeli doğruluk elde ediyor ve yer gerçeği tanımının
> (METAR NSC doğrulaması, okta eşiği) sonuçları ne kadar kaydırdığını nicel olarak gösteriyoruz.

Bu ifade, elde **hazır olan** sonuçlarla doğrudan desteklenebilir (K18: test_time yalnız-SYNOP
model 0,953–0,955 vs CLM r30 0,908, NWC r30 0,923; GA sıfırı dışlıyor). Ablasyon ve mimari
karşılaştırması bu ifadenin **destekleyicisi**, ana savı değil — HGB≈CNN riski böylece etkisizleşiyor.

**Bilerek savunulan ikincil (nötr) bulgular** — ayrı alt başlıklar hâline getirilmeli:

- Kanal istatistikleriyle beslenen gradyan artırma (HGB) CNN'lerle başa baş; nokta tahmininde uzamsal
  doku marjinal. `10.3390/atmos14091410` bu bulguyu literatürde yalnız bırakmıyor.
- AWOS kaynaklı T2m/RH raster kanallarının katkısı küçük ama CV'de tutarlı pozitif
  (birleşik B − C +0,0027 [+0,0015, +0,0041]); tek bölmede görünmüyor → tek bölmeli raporlamanın
  yanıltıcılığı üzerine metodolojik not.
- Bloklu CV, tek bölmeli test'ten sistematik olarak daha düşük ve daha değişken.

---

## 5. Giriş bölümü iskeleti (paragraf → alıntı eşlemesi)

1. **Bulut maskesi neden temel ürün** — radyasyon bütçesi, NWP asimilasyonu, güneş enerjisi,
   yağış/konveksiyon izleme. Alıntı: `10.1038/s41467-023-44666-1`, `10.3390/rs15041141`,
   `10.3390/rs16162989`.
2. **MTG-FCI yeni nesil geostasyoner gözlem** — 16 kanal, 10 dk tam disk; ürün ekosistemi yeni
   kuruluyor. Alıntı: BAMS 2021 MTG tanıtımı, `10.7717/peerj.20495`, `10.3390/rs15245662`,
   `10.1088/2515-7620/ae8153`.
3. **Eşik tabanlı operasyonel maskelerin sınırları** — gece, sis/alçak stratus, kar, ince sirrüs;
   "cloud contaminated" ara sınıfların ikili eşlemesi. Alıntı: `10.1002/2015jd023277`, `10.1002/joc.5011`.
4. **ML tabanlı maskeler ve etiket sorunu** — operasyonel ürünü etiket alan çalışmalar hataları
   miras alıyor; aktif algılayıcı etiketleri dar kesişimli ve misyonlar sona erdi.
   Alıntı: `10.1109/tgrs.2022.3153129`, `10.5194/amt-17-961-2024`, `10.3390/atmos14091410`.
5. **Yer gözlemi alternatifi ve sınırları** — yarımküre gözlemi ↔ piksel ölçek uyuşmazlığı, insan
   gözlemci değişkenliği, istasyon seyrekliği. Alıntı: `10.1002/joc.5011`, `10.1002/joc.8141`,
   `10.1029/2024jd041512`, `10.5194/acp-23-523-2023`.
6. **Radar füzyonu** — yer radarının bulut tespitinde kullanılmamış bir bilgi kaynağı olması.
   Alıntı: `10.3390/rs18183202`, `10.3390/rs12030365`.
7. **Boşluk + katkı listesi** (§4) + makalenin yapısı.

---

## 6. Hedef dergiler: TAAC ve MAP (güncellendi 2026-10-30)

Önceki sürümdeki geniş dergi listesi kapatıldı. İki Springer Wien kardeş dergisine paralel hazırlık yapılıyor:
**Theoretical and Applied Climatology (TAAC)** birincil, **Meteorology and Atmospheric Physics (MAP)** yedek
(sıra kullanıcı onayına bağlı; bkz. yol haritası §0).

| | TAAC | MAP |
|---|---|---|
| JIF 2025 / 5 yıllık | 3.2 / 3.4 | 2.0 / 2.4 |
| İlk karar (medyan) | 11 gün | 45 gün |
| JCR çeyreği (üçüncü taraf; JCR'den teyit edilmeli) | Q2 (39/87) | Q3–Q4 sınırı (83–87/114) |
| Yayın modeli | Hibrit (abonelik yolu ücretsiz) | Hibrit (abonelik yolu ücretsiz) |
| Öz | 150–250 kelime | 150–250 kelime |
| Uzunluk | Research Article ≤ 100.000 karakter (öz, anahtar kelime, tablo, şekil, altyazı ve kaynaklar dahil) | Sınır bu taramada bulunamadı — yazar kılavuzundan kontrol |
| Hakem süreci | — | Tek-kör (single-blind) |
| Taramada konu yoğunluğu (2018+; bulut/radar/uydu/ML; `taac_map_relevant.csv`) | 160 kayıt, 8'i Türkiye | 75 kayıt, 2'si Türkiye |

**Kapsam uyumu ve çerçeveleme farkı**

- **TAAC** kapsamı iklim, atmosfer ve meteoroloji; uygulamalı meteoroloji ve meso-iklim. Risk: ~7 aylık veri
  "klimatoloji değil" gerekçesiyle masa reddine yol açabilir. Önlem: (i) makaleyi uygulamalı meteoroloji /
  izleme ürünü olarak konumlandır, (ii) gündüz-gece, ay ve yükseklik kırılımlarını ana sonuçlara taşı,
  (iii) tam harita ürünüyle Türkiye için **bulut sıklığı haritaları** (aylık ve günün saatine göre) ekle,
  (iv) bulut örtüsünün güneşlenme süresi, güneş enerjisi ve iklim izleme için önemini Giriş'te TAAC
  makaleleriyle bağla (aşağıdaki §8).
- **MAP** kapsamı açık ve bulutlu atmosferde fiziksel süreçler ve meteorolojik verilere uygulanan
  matematiksel/istatistiksel teknikler. Çerçeve: (i) kanal önem analizinin **fiziksel yorumu**
  (pencere kanalları, 3,8 µm gece ayrımı, BT10,5 − T2m kontrastı), (ii) gece alçak bulut/sis
  hatalarının fiziği, (iii) radarın yağış üreten / üretmeyen bulutlarda rolü, (iv) blok-bootstrap,
  bloklu CV ve etiket politikası duyarlılığı gibi istatistiksel titizlik.

## 7. Taramanın bıraktığı işler

1. Çekirdek 20 makalenin **tam metni** okunup tablodaki alanlar (metrik, n, bölge, eşik) elle doğrulanmalı
   — LLM çıkarımı tarama içindir, alıntı için değil.
2. BAMS 2021 MTG misyon makalesi, EUMETSAT FCI CLM **ATBD/ürün doğrulama raporu** ve NWC SAF Cloud Type
   ürün dokümanı (gri literatür) künyeye eklenmeli — referans ürünleri kıyasladığımız için bunlar zorunlu.
3. Türkiye/MGM kaynaklı önceki bulut çalışmaları (ulusal dergiler, tez) ayrıca taranmalı; Crossref
   bunların çoğunu indekslemiyor.
4. `literature_core_table.csv` içindeki 153 çekirdek kaydın ~85'inde etiket kaynağı "unknown" —
   nihai boşluk iddiasını yayımlamadan önce en azından geostasyoner olanların özetleri tamamlanmalı.

---

## 8. Dergiye özgü atıf çapaları (TAAC / MAP; Crossref ISSN taraması)

Editör ve hakemler derginin kendi literatürüyle bağ kurulmasını bekler. Aşağıdakiler `taac_map_relevant.csv`
içinden konuya en yakın olanlardır; **tam metinleri okunmadan atıf yapılmamalı** (bazılarının Crossref'te özeti yok).

**TAAC (ISSN 1434-4483)**

| DOI | Konu | Kullanım yeri |
|---|---|---|
| `10.1007/s00704-024-04962-2` | Türkiye için CM SAF bulut kesri (CFC) ürünü + MLP ile aylık güneşlenme süresi tahmini | **En güçlü çapa**: Türkiye + EUMETSAT bulut ürünü + ML + istasyon; Giriş ¶1 ve ¶5 |
| `10.1007/s00704-024-05062-x` | Farklı atmosfer koşullarında bulutluluk derecesinin makine öğrenmesiyle değerlendirilmesi | Giriş ¶4 (ML ile bulutluluk) — tam metin okunmalı |
| `10.1007/s00704-024-05245-6` | FY-4A uydu verisi ile istasyon gözleminin füzyonu, yoğun sis tanıma | Giriş ¶5, Tartışma (gece/sis) |
| `10.1007/s00704-024-05103-5` | Avusturya günlük güneşlenme gridleri: istasyon + uydu birleşimi (1961–) | Giriş ¶1, tam harita ürününün iklim kullanımı |
| `10.1007/s00704-025-05985-z` | Yüksek rakımlı bölgede yer tabanlı lidar ve uydu ile bulut özellikleri | Yer tabanlı doğrulama, Tartışma (ölçek) |
| `10.1007/s00704-021-03518-y` | Alçak bulut ve sisin yüzey radyasyon akılarına etkisi | Giriş ¶1 (neden önemli) |
| `10.1007/s00704-026-06617-w` | İstanbul'da uzun dönem sis karakteristikleri | Tartışma (gece/sis, Türkiye) |
| `10.1007/s00704-022-03966-0` | İstanbul'da yoğun adveksiyon sisinin WRF duyarlılık benzetimleri | Tartışma (sis, Türkiye) |
| `10.1007/s00704-022-04034-3` | Radar yağış tahmini hatalarının analizi (Türkiye radar verisi) | Veri (TMM radar ağı), Türkiye radar literatürü |
| `10.1007/s00704-025-05650-5` | Uydu tabanlı radar yansıtırlığı ile fırtına nowcasting (Hindistan) | Radar–uydu ilişkisi |

**MAP (ISSN 1436-5065)**

| DOI | Konu | Kullanım yeri |
|---|---|---|
| `10.1007/s00703-026-01191-2` | Yer tabanlı tüm gökyüzü ve uydu görüntülerinde vision transformer ile konvektif bulut tanıma | **En güçlü çapa**: derin öğrenme + uydu + yer gözlemi |
| `10.1007/s00703-024-01028-w` | Yağış kalite kontrolünde yağış ölçer–radar–uydu verisinin birlikte kullanımı | Çoklu kaynak füzyonu gerekçesi |
| `10.1007/s00703-025-01110-x` | Havaalanlarında bulut tabanı yüksekliği farklılıkları | METAR/havaalanı gözlemi |
| `10.1007/s00703-026-01179-y` | Türkiye üzerinde açıklanabilir hibrit derin öğrenme ile yağışa dönüşebilir su buharı tahmini | Türkiye + açıklanabilir DL (kanal önem analizi) |
| `10.1007/s00703-022-00905-6` | CloudSat ile girdap yapısı analizi | Aktif algılayıcı bağlamı |
| `10.1007/s00703-019-00666-9` | Bulut radarı ile mikrofiziksel profil farkları | Radarın bulut fiziğindeki yeri |

Not: Bu dergilerde Türkiye çıkışlı çalışmalar ağırlıkla yağış, sıcaklık ve kuraklık üzerine; **Türkiye üzerinde
uydu bulut maskesi çalışması bulunamadı.** Bu, her iki dergi için de "bölgesel ilk" iddiasını destekliyor,
ancak iddia yazılmadan önce TR Dizin / YÖK Tez taraması yapılmalı (bkz. §7 madde 3).
