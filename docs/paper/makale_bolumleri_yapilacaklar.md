# MTGCLM — Makale bölümleri için yapılacaklar (TAAC + MAP)

Sürüm 1 · 2026-10-30. Her bölüm için: içerik, kaynak dosya, hedef uzunluk, dergiye göre fark ve kontrol listesi.
Prompt numaraları (P#) `eksik_promptlar.md`'ye karşılık gelir.

**Uzunluk bütçesi.** TAAC'ta Research Article sınırı 100.000 karakterdir ve öz, anahtar kelimeler, tablolar,
şekiller, altyazılar ve kaynaklar **dahildir**. Hedef ≈ 7.500–8.500 kelime metin + 5 tablo + 9–10 şekil +
~60 kaynak, toplamda ≤ 80.000 karakter (pay bırakılır). Aynı bütçe MAP için de kullanılır.

| Bölüm | Hedef (kelime) | Ortak / dergiye özel |
|---|---|---|
| Öz | 200–240 | özel |
| 1 Giriş | 900–1.100 | ¶1 ve son ¶ özel |
| 2 Çalışma alanı ve veri | 900–1.000 | ortak |
| 3 Yöntem | 1.400–1.700 | ortak (MAP'te 3.6 daha uzun) |
| 4 Bulgular | 1.800–2.100 | ortak, sıra özel |
| 5 Tartışma | 1.100–1.400 | özel vurgu |
| 6 Sonuçlar | 250–350 | ortak |
| Beyanlar | 250–400 | ortak |

---

## Başlık, öz, anahtar kelimeler

- [ ] Başlık: TAAC ve MAP önerileri `makale_yol_haritasi.md` §0'da. "1 km" yazılmaz (K13: piksel ≈ 1,73 × 1,11 km).
- [ ] Öz 150–250 kelime; tanımsız kısaltma yok (FCI, MTG, CLM, NWC SAF ilk geçişte açılır); kaynak yok.
- [ ] Öz yapısı: konu ve önem → boşluk → veri/dönem/alan → yöntem (etiket kaynağı, girdiler, modeller, bölme rejimleri)
      → ana sayılar (test_time BalAcc, CLM/NWC'ye karşı fark ve GA) → önem.
- [ ] Sayılar `analysis/main_table.csv` ve `references.csv`'den kopyalanır; tek değer kuralı: aynı sayı metinde, tabloda, özde aynıdır.
- [ ] Anahtar kelimeler (4–6). TAAC: *cloud mask; Meteosat Third Generation; synoptic observations; cloud occurrence; Türkiye; machine learning*.
      MAP: *cloud detection; Flexible Combined Imager; weather radar; deep learning; explainability; block bootstrap*.

## 1 Giriş

- [ ] ¶1 (özel). TAAC: bulut örtüsünün iklim izleme, güneşlenme süresi ve güneş enerjisi için önemi; Türkiye bağlamı
      (`10.1007/s00704-024-04962-2`, `10.1007/s00704-024-05103-5`). MAP: bulut-radyasyon etkileşimi ve bulut tespitinin
      fiziksel temeli (spektral imzalar, gece ayrımı).
- [ ] ¶2 MTG-FCI: 16 kanal, 10 dk; ürün ekosistemi yeni (`literatur_konumlandirma.md` §5 ¶2).
- [ ] ¶3 Eşik tabanlı operasyonel maskelerin sınırları (gece, sis, kar, ince sirrüs; ara sınıfların ikili eşlemesi).
- [ ] ¶4 ML maskeleri ve etiket sorunu (ürün etiketi hataları miras alır; aktif algılayıcı kesişimi dar, misyonlar sona erdi).
- [ ] ¶5 Yer gözlemi alternatifi ve sınırları (yarımküre ↔ piksel ölçeği; gözlemci değişkenliği).
- [ ] ¶6 Radar füzyonu boşluğu.
- [ ] ¶7 (özel) Amaç + katkı listesi (3–4 madde) + makalenin yapısı. TAAC'ta katkı 1 = izleme ürünü güvenilirliği;
      MAP'te katkı 1 = girdi katkılarının fiziksel ve istatistiksel sınanması.
- [ ] Tüm atıflar tam metinle doğrulanmış olmalı (P16). Dergiye özgü atıf çapaları: `literatur_konumlandirma.md` §8.

## 2 Çalışma alanı ve veri

- [ ] 2.1 Alan ve grid: lon 25,01–44,99 (0,02°), lat 35,005–42,995 (0,01°), 800 × 1000; piksel ≈ 1,73 × 1,11 km (39°K); yama 33 × 33 ≈ 57 × 37 km. **Fig. 1** (P11).
- [ ] 2.2 MTG-FCI L1C: 16 kanal, yeniden örnekleme (satpy `fci_l1c_nc`), paralaks düzeltmesi **uygulanmadı** (K13) — açıkça yazılır.
- [ ] 2.3 TMM C-bandı radar kompoziti: dBZ kanalı, `radar_cov` = 1 − d/r yakınlığı (K11).
- [ ] 2.4 Yer gözlemleri: SYNOP (insanlı) ve METAR; dedup (±9 dk, en yakın kopya), okta eşlemesi (BKN → 5 okta), okta 9 + RH ≥ %90 kuralı. Kompakt veride 132 istasyon (K14).
- [ ] 2.5 Yardımcı alanlar: AWOS T2m/RH (1982 istasyon, IDW, 6,5 K/km lapse rate), Copernicus GLO-90 DEM, cos(SZA).
- [ ] 2.6 Referans ürünler: FCI CLM (`C` / `O` dönemleri ayrı, P7), NWC SAF CT (MSG tabanlı, Mercator, 15 dk); eşleme merkez ve r5–r30 km.
- [ ] 2.7 Dönem: Mar–Nis 2025, Ara 2025–Haz 2026; Tem–Kas yok → sınırlılık.
- [ ] **Tab. 1** kanal listesi; **Tab. 2** bölme istatistikleri.

## 3 Yöntem

- [ ] 3.1 Etiket politikası: 0 okta → açık, ≥ 5 okta → bulutlu, 1–4 → yok sayılır; METAR NSC yalnız aynı saatte SYNOP N = 0 ise (verify, K16). **Fig. 2** akış şeması.
- [ ] 3.2 Bölme rejimleri: hafta-blok (7 gün, 70/15/15) + 30 holdout istasyon → test_time / test_station / test_both; bloklu 5-katlı CV. Sızıntı önlemleri.
- [ ] 3.3 Modeller: MTGConvNet Medium-Flat (mimari ablasyonun kazananı), MTGUNet (merkez-piksel denetimi, `MaskedBCELoss`), ResNet-18 (iki stem), klasik modeller (eşik, lojistik regresyon, RF, HGB). **Fig. 3**.
- [ ] 3.4 Eğitim: 15 epoch, val BalAcc ile model seçimi, 3 seed, determinizm, test'e tek dokunuş.
- [ ] 3.5 Ablasyonlar: modalite (A/B/C/C_t2m + radarsız P1), HGB kontrollü kıyas (P2).
- [ ] 3.6 Değerlendirme ve istatistik: BalAcc, MCC, F1, Brier, ECE; gün-blok bootstrap (test_station'da istasyon-blok), eşleştirilmiş farklar; McNemar'ın neden kullanılmadığı (kümelenme). MAP sürümünde bootstrap blok seçimi ve CV birleşik fark hesabı formülle verilir.
- [ ] 3.7 Kanal önem analizi yöntemi (P3).
- [ ] 3.8 Tam harita çıkarımı ve bulut sıklığı hesabı (P4, P6).
- [ ] 3.9 Yazılım ve donanım; yapay zekâ destekli araç kullanımı beyanı (Springer Nature politikası gereği Yöntem'de).

## 4 Bulgular (önerilen sıra)

| Alt bölüm | İçerik | TAAC sırası | MAP sırası |
|---|---|---|---|
| Model karşılaştırması | Fig. 4, Tab. 3; CNN ≈ U-Net ≈ ResNet ≈ HGB | 1 | 1 |
| Operasyonel ürünlere karşı | Fig. 5; yalnız SYNOP + kapsam eşleşmiş; C/O ayrımı | 2 | 2 |
| Girdi katkıları | Tab. 4; AWOS (CV birleşik B − C +0,0027 [+0,0015, +0,0041]), radarsız | 4 | 3 |
| Kanal önemi | Fig. 7 | ek | 4 |
| Kırılımlar | Fig. 6; gündüz/gece, ay, yükseklik, SYNOP/METAR | 3 | 5 |
| Tam harita ve sıklık | Fig. 8, Fig. 9 | 5 | 6 (Fig. 9 ek) |
| Kalibrasyon ve okta tutarlılığı | Fig. 10 | ek | 7 |

- [ ] Her sayıda n, küme, etiket alt kümesi (tümü / SYNOP) ve GA verilir.
- [ ] Tek bölme ile bloklu CV farkı (CV daha düşük ve değişken) raporlanır.

## 5 Tartışma

- [ ] 5.1 Operasyonel ürünlere üstünlüğün kaynağı: etiket kaynağı mı, girdi füzyonu mu, bölgesel eğitim mi? (P1 ve P7 sonuçlarıyla.)
- [ ] 5.2 Model ailesi nötr bulgusu: spektral bilgi baskın; uzamsal doku katkısı (P2); derin modelin tam harita avantajı.
- [ ] 5.3 AWOS ve radarın katkısı: küçük ama CV'de tutarlı AWOS etkisi; radar katkısı (P1).
- [ ] 5.4 Gece, sis ve alçak bulut (P8); MAP'te fiziksel yorum daha derin.
- [ ] 5.5 (TAAC) Bulut sıklığı haritalarının iklim izleme açısından anlamı ve CLM ile fark deseni (P6).
- [ ] 5.6 (MAP) Kanal önemi ile spektral fiziğin tutarlılığı (P3).
- [ ] 5.7 Sınırlılıklar: Tem–Kas yok; kar–bulut; ölçek uyuşmazlığı (piksel ↔ CLM pikseli ≈ 3 × 3,5 km ↔ yarımküre gözlemi); paralaks (P10); etiket eşiği (P9); radar yalnız Türkiye; 132 istasyon.

## 6 Sonuçlar

- [ ] 4–5 cümle bulgu + 2 cümle operasyonel anlam + 1–2 cümle gelecek çalışma (paralaks düzeltmesi, Tem–Kas verisi, bulut fazı/tipi, MGM operasyonel entegrasyonu).

## Beyanlar ve ek dosyalar (Springer)

- [ ] Funding (proje no varsa) · Competing interests · Ethics approval: Not applicable · Consent: Not applicable
- [ ] Data availability: FCI → EUMETSAT Data Store; CLM → EUMETSAT; radar ve yer gözlemleri MGM'ye ait, talep üzerine/izinle; türetilmiş örneklem → Zenodo DOI
- [ ] Code availability: GitHub + sürüm etiketi + Zenodo DOI
- [ ] Author contributions (CRediT) · Acknowledgements (EUMETSAT, NWC SAF, Copernicus DEM, MGM birimleri)
- [ ] Supplementary Information: ek şekiller (TAAC'ta "ek" işaretliler), tam tablolar, CV kat tabloları

## Son kontrol (P19)

- [ ] Öz kelime sayısı · TAAC karakter sayımı · şekil kontrol raporları (`*_check.txt` hepsi `OK`) · şekiller metinde sırayla anılıyor
- [ ] Tüm kısaltmalar ilk geçişte açılmış · ondalık ayırıcı nokta · birimler SI · "Türkiye" yazımı tutarlı
- [ ] Kaynak listesi derginin biçiminde · DOI'ler çalışıyor · her atıf tam metinle doğrulanmış
