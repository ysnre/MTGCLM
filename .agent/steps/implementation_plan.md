# AWOS Pixel Match Skill - Implementation Plan

## Hedef (Goal)
Yer gözlem istasyonlarının (AWOS) sağladığı meteorolojik gözlemlerin (SYNOP/METAR) enlem/boylam koordinatlarını kullanarak, bu istasyonların 1000x800 boyutlarındaki MTG uydu matrisinde tam olarak hangi X (sütun) ve Y (satır) indekslerine denk geldiğini (Pixel Matching) tespit etmek. Bu işlem, yapay zeka modelinin doğrulanması (ground truth) ve eğitimi (labeling) için kritik öneme sahiptir.

## Open Questions

> [!IMPORTANT]
> **1. Çıktı Formatı:**
> AWOS eşleştirmesi sonucunda, istasyonların 1000x800 matris üzerindeki (y, x) piksel indekslerini ve o piksellerdeki `cloud_coverage` (bulut örtüsü) değerini elde edeceğiz. 
> Çıktı olarak her uydu zamanı için bir **CSV** dosyası mı (Sütunlar: `y, x, wmoid, cloud_coverage`), yoksa 1000x800 boyutunda sadece istasyon piksellerinin dolu olduğu (diğer yerlerin -1 veya nan olduğu) bir **Numpy (.npy)** matrisi mi kaydedelim? 
> *(Not: Model eğitiminde doğrudan piksel tabanlı örneklem alınacağı için CSV formatı veri kümesi oluşturmada daha performanslı ve pratik olabilir.)*

> [!IMPORTANT]
> **2. Zaman Toleransı:**
> Uydu görüntüleri 10 dakikada bir üretilirken, SYNOP verileri genelde saatlik, METAR verileri ise yarım saatlik periyotlarla gelir. Uydu saatine karşılık gelen gözlemi seçmek için eşleştirme toleransımız **+/- kaç dakika** olmalıdır? (Örn: +/- 15 dakika veya +/- 30 dakika).

## Önerilen Mimari (Proposed Changes)

Mevcut projeye aşağıdaki yeni modüller eklenecektir:

### 1. AWOS Konum Eşleştirici

#### [NEW] `src/awos_matcher.py`
- `data/raw_observation/station_list.csv` dosyasını okuyup istasyon listesini ve koordinatlarını yükleyen bir fonksiyon yazılacaktır.
- `scipy.spatial.cKDTree` algoritması kullanılarak, istasyonların enlem/boylam noktalarının MTG uydu gridinde (1000x800 lats/lons) matematiksel olarak en yakın olduğu (Y, X) piksel indekslerini hesaplayan `match_stations_to_grid()` fonksiyonu oluşturulacaktır.

### 2. Gözlem Verisi Yükleyici

#### [NEW] `src/observation_loader.py`
- `synop.csv` ve `metar.csv` dosyalarını okuyacak.
- `validity` (zaman) sütunlarını `datetime` formatına dönüştürecek.
- Hatalı veya eksik (`cloud_coverage == -1`) gözlemleri filtreleyecek bir mekanizma sağlayacaktır.

### 3. AWOS Entegrasyon Akışı

#### [NEW] `src/process_awos_pipeline.py`
- Radar akışına (Pipeline) benzer şekilde çalışacaktır.
- Her bir `turkiye_kesit_*.nc` uydu dosyası için uydu zaman damgası alınacak.
- Gözlem verilerinde (SYNOP/METAR) uydu zamanına belirlenen tolerans aralığında en yakın olan kayıtlar filtrelenecektir.
- Seçilen gözlemler, `awos_matcher.py` ile önceden hesaplanan (Y, X) indeksleriyle birleştirilerek nihai sonuçlar kaydedilecektir (`data/processed/awos_aligned/` klasörüne).

## Doğrulama Planı (Verification Plan)
1. Seçilen örnek bir uydunun saatiyle uyuşan gözlem kayıtları yazdırılarak zaman toleransının doğruluğu test edilecektir.
2. Bir istasyonun bilinen koordinatları, hesaplanan (Y, X) pikselleriyle harita veya grid üzerinde görsel olarak (veya manuel değer kontrolüyle) çapraz kontrol edilecektir.
