# Radar Verilerinin MTG Uydu Izgarasına Hizalanması (Walkthrough)

Radar PNG verilerinin, MTG uydusunun (1000x800) Türkiye kesitine "En Yakın Komşu" (Nearest Neighbor) interpolasyonu ve Maksimum Reflektivite yöntemiyle başarıyla hizalanması işlemini tamamladık.

## Neler Yapıldı?

1. **`src/satellite_grid.py` Modülü:**
   - Örnek bir uydu `.nc` dosyasından 1000x800 boyutlarındaki `x` (boylam) ve `y` (enlem) matrisleri okundu.
   - `pyresample` kütüphanesi için hedeflenen `SwathDefinition` oluşturuldu.

2. **`src/radar_loader.py` Modülü:**
   - Radar koordinat bilgilerini almak için `data/raw_radar/radar.csv` dosyası okundu (`pandas` ile).
   - Radar PNG görüntülerini yüklemek ve Mavi (Blue - B) kanalını Numpy dizisi olarak çıkarmak için `Pillow` kütüphanesi entegre edildi.
   - Her dosyanın zaman damgası dosya isminden ayrıştırıldı (`AFY2605250000...` -> 2026-05-25 00:00).

3. **`src/resampler.py` Modülü:**
   - `pyresample.kd_tree.resample_nearest` kullanılarak orijinal radar boyutu ve Bounding Box üzerinden (`AreaDefinition`) hedef uydu matrisine dönüştürme mantığı kuruldu.

4. **`src/process_radar_pipeline.py` (Ana Çalıştırma Akışı):**
   - Belirlenen bir zaman penceresi (**+/- 5 dakika**) içerisinde tüm uyumlu radar istasyonları (Samsun, Bursa, Afyon vb.) tespit edildi.
   - Her radar parçası MTG uydu gridine çekildi.
   - Türkiye genelini elde etmek için radarların kesişim alanlarında **Maksimum Değer (Maximum Reflectivity)** birleştirme stratejisi (compositing) kullanıldı.

## Doğrulama Sonuçları

Örnek 3 netCDF dosyası ile süreç test edildi:
- İlk dosya (Saat 00:08) için **32 adet** uyumlu radar görüntüsü tespit edildi ve başarıyla hizalandı.
- Oluşturulan `.npy` dosyasının boyutunun tam olarak **(800, 1000)** olduğu doğrulandı.
- Örnek dosyada piksellerin yaklaşık **%6.93'ünün** yağış verisi (0'dan büyük reflektivite) içerdiği teyit edildi.

## AWOS İstasyonları Eşleştirme (Pixel Match) İşlemi

Radar hizalamasının ardından, Yer Gözlem İstasyonlarının (AWOS) uydu matrisine eşleştirilmesi işlemini de tamamladık.

1. **İstasyonların Matris İndekslerine Çevrilmesi:**
   - `station_list.csv` içerisindeki 279 adet istasyonun Enlem/Boylam (Lat/Lon) değerleri, `scipy.spatial.cKDTree` algoritması kullanılarak 1000x800'lük MTG uydu gridinde karşılık gelen **Y (satır)** ve **X (sütun)** indekslerine dönüştürüldü.
   - İndeks eşleştirme hassasiyeti ~0.005 derece sapma ile (~500 metre, yani 1km'lik pikselin tam içine düşecek şekilde) yüksek bir doğrulukla gerçekleşti.

2. **Gözlem Verilerinin İşlenmesi:**
   - Hem `synop.csv` (saatlik) hem de `metar.csv` (yarım saatlik) dosyaları okundu ve tek bir çatı altında birleştirildi. Toplamda **93,916 adet** geçerli gözlem kaydı oluşturuldu (Hatalı `-1` `cloud_coverage` değerleri elendi).

3. **Zaman Toleransı ve CSV Çıktıları:**
   - Her bir uydu (.nc) dosyası için zaman damgası okundu.
   - İstediğiniz gibi **+/- 9 dakikalık** zaman toleransı ile o ana ait eşleşen AWOS gözlemleri tespit edildi.
   - Bulunan gözlemler, `(validity, wmoid, cloud_coverage, y, x)` sütunlarına sahip **CSV** dosyaları olarak `data/processed/awos_aligned/` klasörüne kaydedildi. 
   - Örneğin, `00:08`, `00:18` ve `00:28` uydu saatleri için sırasıyla 18, 39 ve 39 adet istasyonun o anki bulutluluk (`cloud_coverage`) değerleri başarıyla indekslendi.

## Sonraki Adımlar
Bu aşama ile uydunun her bir pikseline karşılık gelen hem **Radar (Feature)** hem de **AWOS (Label)** verilerini konumsal ve zamansal olarak aynı düzleme çekmiş olduk. AI eğitim modeli için veri hazırlık aşamasının en zor bölümü bitti. 

Şimdi isterseniz geri kalan tüm 10 günlük veriyi toplu olarak çalıştırıp sonuçlandırabiliriz veya modeli eğitecek olan Dataset/Dataloader oluşturma kısmına geçebiliriz.
