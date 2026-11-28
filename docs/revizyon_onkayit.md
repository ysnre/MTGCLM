# MTGCLM — Revizyon analizleri ön-kaydı (pre-registration)

**Tarih:** 18 Kasım 2026 · **Depo durumu:** `main` @ `469a476` (bu dosyanın commit'i ön-kaydın zaman damgasıdır)
**Dayanak:** gönderim öncesi **gayriresmî iç ön-inceleme** (dergi hakem süreci değil). Bu dosya, yeni test kümesi analizlerine
başlamadan önce yazarların kendi kendine verdiği bir taahhüttür.

## 0. Taahhütler

1. **Raporlama:** Aşağıdaki her analizin sonucu, lehte ya da aleyhte olsun, makalede ya da ek materyalde raporlanacak.
   Hiçbir analiz sonucuna bakıldıktan sonra sessizce bırakılmayacak.
2. **Değişmeyecekler:** Ana tablo (Tab. 3), ana bölme (`split_v2_verify`), ana etiket tanımı (verify) ve Öz'deki tek bölme
   skorları bu turda değişmez. Yeni sonuçlar ek tablo ve şekillere girer.
3. **Seçim yasağı:** Hiçbir eşik, hiperparametre ya da yarıçap test kümesine bakılarak seçilmeyecek. Seçim gerekiyorsa
   yalnız `train` ya da `val` kullanılacak.
4. **Sapma kaydı:** Bu plandan sapılırsa sapma, tarihi ve gerekçesiyle bu dosyanın sonundaki "Sapmalar" bölümüne eklenecek.
5. **Çıkarım:** Çıkarım blok-bootstrap ile yapılacak. Gün blokları kullanılacak; test_station'da istasyon blokları.
   Bootstrap tohumları `analyze_preds.py` ile aynı olacak.

## 1. Analizler, birincil metrikler ve yorum kuralları

| # | Analiz | Birincil metrik / küme | Önceden yazılmış yorum kuralı |
|---|---|---|---|
| R6 | Kontenjans metrikleri | POD(bulutlu), FAR, frekans bias, KSS, HSS; test_time / test_station / test_both × {tümü, SYNOP}; modeller + CLM/NWC (merkez, r30) | Betimsel; ek tablo. Ana metinde tek atıf. |
| R7 | Kar ve VZA kırılımı | BalAcc + FAR; kar imzalı gündüz açık raporlar (ölçüt `select_scenes.py`'den) ve VZA üçlükleri | n < 300 olan hücrelerde "fark yok" denmeyecek, yalnız "belirsiz". VZA sonuçları betimseldir (VZA boylamla karışık). |
| R8a | Çoklu karşılaştırma | p = (k+1)/(B+1); **Aile 1:** model–referans farkları (`references.csv`); **Aile 2:** tek bölme modalite ve ablasyon farkları (`pairs.csv`); **Aile 3:** CV farkları (`cv_pairs.csv`). Aile içinde BH q = 0,05 (birincil), Holm (duyarlılık). | Metinde "anlamlı" nitelemesi yalnız BH sonrası ayakta kalan farklarda kullanılacak. |
| R8b | Blok uzunluğu | Örtüşen (moving) blok bootstrap, 1 / 2 / 3 / 5 gün; Aile 1 ve 2'nin GA genişlikleri | Herhangi bir blok uzunluğunda sıfırı içeren fark "blok uzunluğuna duyarlı" diye işaretlenecek. |
| R9 | Holdout ↔ AWOS | En yakın AWOS mesafesi (≤ 1 / 2 / 5 km); B_full − C_noawos farkı ko-lokasyonlu ve olmayan holdout gruplarında | Fark yalnız ko-lokasyonlu grupta GA'sı sıfırı dışlıyorsa "AWOS katkısı transfer edilemez" yazılacak. |
| R10 | Sıklık duyarlılığı | Doğu yaylaları CLM − model farkı; eşik 0,4 / 0,5 / 0,6, val'de bias = 1 veren eşik ve val'de izotonik kalibrasyon; maske yarıçapı 15 / 25 / 50 km | "CLM karı bulut sayıyor" **bulgu** olarak kalır, ancak fark tüm varyantlarda > +0,03 ise. Aksi halde hipotez düzeyine iner. EUMETSAT doğrulama raporundaki (EUM2045564) kar testinin kapalı olduğu bilgisi bu kuraldan bağımsız olarak metne girer. |
| R11 | Gözlemci tavanı | SYNOP–METAR aynı saat ikili uyuşmazlığı → ima edilen BalAcc tavanı; okta 4–5 vs 0/8 hata oranı | Tavan iyimser kabul edilecek (aynı gözlemci olasılığı). Sayı desteklemiyorsa analiz yine raporlanacak. |
| R12 | Referans-kalibre kıyas | **Birincil:** HGB, CLM+NWC sınıf oranları (merkez, 3×3, r5–r30), yalnız train SYNOP ile eğitilir; değerlendirme test_time SYNOP, ortak kapsam. **İkincil:** LR; yalnız CLM; yalnız NWC. Ayar `baselines.py` varsayılanları, arama yok. | Fark CNN − REF_cal GA'sı sıfırı dışlıyorsa: "ürünlerin gözlemci ölçeğine kalibrasyonunun ötesinde ek bilgi". Dışlamıyorsa: "operasyonel ürünlere üstünlük büyük ölçüde hedef ve bölge hizalamasından kaynaklanıyor"; TAAC'ın ana iddiası buna göre daraltılır. Sensör kuşağı farkı bu analizle ayrılamaz (yazılacak). |
| R13 | Normalizasyon kayması | Ay × aydınlanma sınıfı içinde kanal ortalaması kayması, eğitim kümesi istatistiğine göre σ biriminde | Yalnız IR kanallarında \|kayma\| > 0,5σ olursa yeniden eğitim için kullanıcı kararı istenir. Güneş kanalları tanısal raporlanır. |
| R14 = P20 | U-Net hedef-piksel jitter | (i) Tek geçiş ↔ faz-hizalı ikili uyuşmazlık (3 test sahnesi, K27 ile aynı); (ii) D_unet_jit − D_unet test_time BalAcc, eşleştirilmiş, 3 tohum | Ek tabloya girer. **Fig. 9 jitter modeliyle yeniden üretilir, ancak** (i) < %5 VE (ii)'nin GA alt sınırı > −0,005 ise (eşdeğer-olmama sınırı). Aksi halde Fig. 9 faz-hizalı modelle kalır ve jitter sonucu ek tabloda raporlanır. |
| R17 | Doğrulanmamış METAR açıkları | P9 keep satırları (test kümeleri) üzerinde yalnız bu alt kümede BalAcc / POD / FAR; model ve referanslar | Betimsel; ek tabloya. |
| R20 | Bulutlu sınıf sınırı ≥ 6 okta (NWC SAF doğrulamasıyla uyumlu) | Yalnız SYNOP satırları (METAR'da BKN 5–7 ayrılamaz); 5 okta raporları dışarıda, açık = 0 okta; yeniden eğitim yok, eşik 0,5; test_time / test_station / test_both; modeller + CLM/NWC (merkez, r30) | Ek tabloya girer (etiket duyarlılığı tablosunun yanına). Model–referans farkının işareti değişirse Öz'deki "etiket tanımından bağımsız" ifadesi kaldırılır. |
| R18 | IDW doğrulaması | t2m / rh2m leave-one-out RMSE ve bias; yükseklik farkı sınıflarına göre | Betimsel; Veri bölümüne bir cümle, ek tabloya. |

## 2. Metin kararları (sonuçlardan bağımsız)

- Öz'ün ilk cümlesi daraltılacak. Operasyonel FCI CLM'nin şimdiye kadar başka uydu maskeleriyle (SEVIRI NWC SAF, MODIS;
  EUMETSAT EUM2045564) doğrulandığı, yer gözlemleriyle doğrulanmadığı belirtilecek. NWC/GEO MTG-I doğrulama raporu SYNOP içeriyorsa
  o da atıflanacak.
- 03_methods'taki "each test set is evaluated once" cümleleri, karar kronolojisi paragrafıyla (R3) değiştirilecek.
- AI beyanı, metnin taslağının da yapay zekâ yardımıyla hazırlandığını kapsayacak biçimde genişletilecek.

## 3. Sapmalar

- **18 Kas 2026, 18:57 (TSİ):** Yalnız "Dayanak" satırı netleştirildi (iç ön-inceleme olduğu yazıldı, iç belge adları çıkarıldı).
  Analiz listesi, metrikler ve yorum kuralları **değişmedi**. Bu satır yazılırken henüz hiçbir yeni analiz koşulmamıştı.
- **18 Kas 2026:** §2'deki Öz cümlesi kararı uygulandı (N1). Uygulanan biçim: "validated only at full-disc scale, not against a national
  station network". Kaynaklar: EUMETSAT_CLMVR2025 ve NWCSAF_UMMTG2025 (NWC/GEO CMA, MTG için SYNOP/SHIP ile tam disk, POD %85,9).
  Bu bir metin kararıdır, analiz sonucu değildir.
- **19 Kas 2026, 17:22 (TSİ):** R20 eklendi (≥ 6 okta duyarlılığı). Gerekçe: hakemlerin "ürün doğrulamaları > 5 okta kullanırken neden ≥ 5"
  sorusu. Eklendiği anda ön-kayıttaki analizlerin hiçbiri koşulmamıştı.
- **20 Kas 2026, 17:10 (TSİ):** R13 ADIM 3 eklendi (kullanıcı onayı). R13 tanısı (ay × aydınlanma, eğitim σ'sı) Haz 2026 IR
  kanallarında kuralı tetikledi; ek denetimde z-skor sabitlerinin (H5 `means/stds`, ilk 5 sahneden) eğitim kümesi istatistiğinden
  IR'de 1,3–1,8σ saptığı görüldü. **ADIM 3:** B_full (Medium_Flat + tüm aux), yalnız temel 17 kanalın z-skor sabitleri eğitim
  kümesinden yeniden hesaplanarak (afin dönüşüm; tabakalı olmayan, deterministik 20 blok × 2048 satırlık örnek, tohum 0; dolgu
  değerleri aynı fiziksel değerde kalır), diğer her şey aynı, tohum 1–3. **Birincil:** B_full_trainnorm − B_full test_time BalAcc,
  3 tohum ortalaması, eşleştirilmiş gün-blok bootstrap (analyze_preds). **İkincil:** test_station, test_both, yalnız SYNOP, CV yok.
  **Yorum:** GA sıfırı içeriyorsa "sonuçlar normalizasyon sabitlerine duyarsız"; dışlıyorsa fark ek tabloda raporlanır ve Yöntem'de
  yazılır. Her durumda ana tablo değişmez, U-Net yeniden eğitilmez. Eklendiği anda bu koşu başlamamıştı.
- **21 Kas 2026, 09:40 (TSİ):** Ön kayıtta OLMAYAN keşifsel bir analiz eklendi (kullanıcı isteği): R7'de CLM'nin kış gündüzü > 1500 m
  satırlarında açıkların %91'ini bulutlu saydığı görüldükten sonra, model − CLM / NWC farkı bu satırlar çıkarılarak yeniden hesaplanacak
  (varyantlar: kış gündüzü > 1500 m hariç; tüm kış > 1500 m hariç). Sonuç görülerek tanımlandığı için **yorum kuralı yoktur**; ek tabloda
  "keşifsel" etiketiyle, ana sonuçlarla yan yana raporlanır ve ana tablo değişmez.
- **27 Kas 2026, 14:30 (TSİ) — ikinci iç hakem turu (B1–B3), koşudan önce yazıldı.** Gerekçe: 26 Kas iç değerlendirmesi
  (`docs/paper/hakem_degerlendirmesi_2026-11-26.md`). Üçü de **betimsel**; hiçbir model, eşik ya da etiket bu sonuçlara bakılarak
  değiştirilmeyecek; ana tablo değişmez. Tümü verify bölmesi, kayıtlı tahminler, gün blokları (test_station'da istasyon blokları), B = 2000, tohum 0.
  - **B1 kapsama yanlılığı** (`src/coverage_bias.py`): test kümelerinde (yalnız SYNOP ve tüm etiketler) CLM ve NWC SAF CT'nin kapsadığı ve
    kapsamadığı satırlar; ay, UTC saati, aydınlanma, yükseklik bandına göre kapsama oranı; kapsanan/kapsanmayan satırların bulutlu oranı ve
    B_full BalAcc'i. **Birincil:** test_time SYNOP, B_full BalAcc(NWC kapsanan) − BalAcc(kapsanmayan) farkı ve GA'sı. **Yorum:** fark GA'sı
    sıfırı dışlar ve |fark| ≥ 0,01 ise Sınırlılıklar'a "NWC kıyası kapsanan satırlarla sınırlı ve bu satırlar daha kolay/zor" cümlesi;
    aksi halde yalnız ek tablo.
  - **B2 silinen METAR açıkları** (`src/removed_metar_profile.py`): verify politikasının çıkardığı doğrulanmamış METAR açık satırlarının
    (bölme dışı) saat, ay, aydınlanma, istasyon dağılımı; aynı istasyon-saatte SYNOP varlığı; CLM r30 ve NWC r30'a göre bulutlu oranı
    (eşikler train'den, analyze_preds ile aynı), tutulan açık satırlarla (aynı kümelerde, METAR ve SYNOP ayrı) karşılaştırma.
    **Birincil:** silinen ile tutulan METAR açıklarında NWC r30 bulutlu oranı farkı (GA, istasyon blokları). **Yorum:** silinenlerde oran
    belirgin yüksekse (fark ≥ 0,05 ve GA sıfırı dışlıyor) "verify politikası zor açık vakaları ayıklıyor olabilir; keep sonuçları (Tablo S3)
    alt sınırdır" cümlesi Sınırlılıklar'a; benzer ya da düşükse betimsel kalır.
  - **B3 yükseklik ve kıyı kırılımı** (`src/surface_breakdown.py`): istasyon yüksekliği (< 500, 500–1000, 1000–1500, > 1500 m) ve kıyıya
    uzaklık (< 30 km / ≥ 30 km, `cloud_frequency.py` kıyı maskesi); B_full, D_unet_jit, CLM r30, NWC r30; n, BalAcc, POD, POFD, FAR;
    n < 300 belirsiz. **Yorum:** yalnız betimsel (ek tablo + bir cümle).
- **27 Kas 2026, 18:50 (TSİ) — C1 IMS kar testi, analizden ve betik yazılmadan önce (kullanıcı onayı).** Ayrıntı:
  `docs/paper/C1_ims_fizibilite_2026-11-27.md`. Veri: NOAA/USNIC IMS 1 km ASCII (NSIDC G02156, v1.3), dosya adındaki günün 00 UTC
  haritası (ilk veri satırı haritanın altı; 800 × 1000 ızgaraya en yakın komşu). Yalnız biçim/yön denetimi için bir dosyaya bakıldı
  (2025-03-01); hiçbir bulut sıklığı ile eşleştirme yapılmadı. Alan: doğu yaylaları maskesi (`regions.npz`); ikincil tüm Türkiye maskesi.
  Sahneler: kış gündüzü (Ara–Mar, 06–12 UTC), `data/processed/cloud_freq_jit/scenes`. Her sahne-pikselde CLM ve model (D_unet_jit) ikisi de
  geçerliyse IMS sınıfına göre **karlı (4)** ve **karsız kara (2)** olarak ayrılır. **Birincil:** Δ = (CLM − model)_karlı − (CLM − model)_karsız,
  model eşiği 0,5, gün blokları bootstrap (B = 2000, tohum 0). **Yorum kuralı:** Δ'nın GA'sı sıfırı dışlar, Δ ≥ 0,05 **ve** aynı sonuç
  frekans-eşleyen eşikle (val, 0,176) de sağlanırsa metinde "kış gündüzü CLM fazlası bağımsız bir kar analizinde karlı görünen piksellerde
  toplanıyor" yazılır (tüm dönem harita farkı keşifsel kalır); aksi halde bulgu keşifsel kalır, sonuç ek tabloda raporlanır.
  **İkincil (betimsel):** karsız piksellerde CLM − model; ertesi günün haritası; aylara göre; tüm Türkiye maskesi.
- **28 Kas 2026, 10:50 (TSİ) — B3 genişletmesi ve B4 ek tohumları, koşudan önce.** Gerekçe: iç tur listesinde B3 eğim/arazi örtüsü ve B4
  S3 (ek tohum) açık kalmıştı.
  - **B3b eğim** (`src/surface_breakdown.py --slope`): istasyon pikselinde DEM eğimi (3 × 3 merkezî fark, piksel boyutu 1,73 × 1,11 km),
    istasyonlar eğime göre **tertillere** ayrılır (sınırlar istasyonlardan, sonuçlara bakılmadan). Metrikler ve yorum B3 ile aynı: yalnız betimsel.
  - **B3c arazi örtüsü** (veri gelirse): istasyon çevresi (merkez piksel) baskın sınıf; sınıf başına n < 300 belirsiz. Yalnız betimsel.
  - **B4-S3 ek tohumlar:** A_sat, A_norad, B_full, B_norad, C_noawos için tohum 4–7 (toplam 7), ayrı klasör `artifacts_v3_seeds/`
    (ana tablo 3 tohumla değişmez, ön-kayıt madde 2'ye dokunulmaz). **Birincil:** `seed_nested_boot.py` iç içe GA'ları 7 tohumla,
    çiftler B_full − A_sat, B_full − B_norad, B_full − C_noawos, A_sat − A_norad; test_time ve test_station. **Yorum kuralı:** metindeki
    her girdi katkısı ifadesi 7 tohumlu iç içe GA'ya göre yazılır — GA sıfırı dışlıyorsa "tohumlardan bağımsız", içeriyorsa "eğitim
    varyansı içinde"; 3 tohumlu sonuçla çelişirse 7 tohumlu sonuç geçerli olur ve fark Yöntem'de belirtilir.
