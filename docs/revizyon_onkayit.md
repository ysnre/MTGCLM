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
