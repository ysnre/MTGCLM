# AI Dataset & DataLoader - Güncellenmiş Uygulama Planı

## Hedef (Goal)
Lokal olarak hizalanmış MTG Uydu verileri (16 kanal), Radar verileri (1 kanal) ve AWOS Gözlemlerini (Etiketler) kullanarak, Pytorch tabanlı bir Derin Öğrenme modelini eğitebilmek için özel bir `torch.utils.data.Dataset` ve `DataLoader` yapısı oluşturmak.

---

## 1. Literatür Analizi ve Doğrulama Standartları
Bulut maskesi (cloud mask) literatüründe yer gözlemleri (SYNOP/METAR) ile uydu verilerinin karşılaştırılmasında en büyük zorluk **mekansal temsil (spatial mismatch)** sorunudur. Yer gözlemcisi kubbe şeklinde gökyüzünü görürken, uydu düz bir pikseli görür. 
- **Parçalı Bulutluluk (1-7 Okta):** Bulutun tam olarak hangi pikselin üzerinde olduğu bilinemediği için hata payı en yüksek olan aralıktır. EUMETSAT ve NWCSAF gibi uluslararası kuruluşların doğrulama (validation) çalışmalarında, **belirsizliği önlemek için orta okta değerleri (3-5 veya 1-7 arası) eğitim ve doğrulamadan tamamen çıkarılır.**
- **Net Durumlar (0 ve 8 Okta):** 0 Okta (kesin açık) ve 8/9 Okta (kesin kapalı/sis) durumları uzamsal olarak en kararlı ve tutarlı etiketlerdir.

---

## 2. Konsolide Doğruluk (Consolidated Ground Truth) ve Etiket Stratejisi
Meteorolojide tek başına %100 kusursuz bir bulut maskesi kaynağı yoktur. En doğru etiket, bağımsız gözlem kaynaklarının (AWOS ve Radar) birleştirilmesiyle (Consensus) elde edilir:

- **Sınıf 1 (Bulutlu - Cloudy) Teyidi:**
  - AWOS **8 Okta** (gökyüzü tamamen kapalı) ise VEYA
  - İlgili istasyon pikselindeki radar yansıması **> 0** (yağış varsa kesin buluttur) ise VEYA
  - **Yeni Kriter (Tereddüt Durumu):** Uydu ve radarın kesin bir bulut tespiti yapamadığı (radar yansıması 0 olan kuru bulutlar veya uydunun zayıf sinyal aldığı durumlar), ancak gözlemcinin parçalı bulut (**1 ila 7 Okta**) rapor ettiği durumlar. Bu durumda, bulutu kaçırmamak (güvenli/muhafazakar yaklaşım) adına bu alanları da **Bulut Var (1)** olarak kabul edebiliriz.
- **Sınıf 0 (Açık - Clear) Teyidi:**
  - AWOS **0 Okta** (gökyüzü tamamen açık) ise VE radar yansıması **0** ise.
- **Ignore (-1) (Belirsiz / Maskelenen):**
  - Gözlemcinin parçalı bulut (1-7 Okta) bildirdiği ancak uydunun ve radarın "aşırı net bir şekilde pürüzsüz açık gökyüzü" gösterdiği, konumsal kaymanın çok bariz olduğu durumlar (İstenirse bu durumlar elenir veya uzamsal 3x3 komşuluk kontrolüyle elenebilir).

Bu yeni kriterin avantajı, uydunun tek başına görmekte zorlandığı ince/yağışsız bulutları (örn: yüksek sirüs veya alçak stratus) yer gözlemi teyidiyle modele öğretebilmemizdir. Modeli bulut tespiti konusunda daha duyarlı (sensitive) hale getirir.

Bu yöntemle, modelin gürültülü (noisy) veya hatalı piksellerden öğrenmesini engelleyip, sadece konumsal ve fiziksel olarak teyit edilmiş pikselleri eğitiyoruz.

---

## 3. Önerilen Kod Mimarisi

### [NEW] `src/dataset.py`
`MTGCloudDataset` sınıfı aşağıdaki özelliklerle yazılacaktır:
- **Girdi:** 16 kanal MTG verisi + 1 kanal resampled Radar verisi (Toplam: 17 kanal tensor).
- **Etiket Haritalama (Label Mapping):**
  ```python
  # Okta -> Sınıf dönüşüm sözlüğü
  LABEL_MAP = {
      0: 0,   # Clear
      8: 1,   # Cloudy
      9: 1,   # Cloudy (Sis)
      # 1-7 arası değerler bu sözlükte yer almayacak ve varsayılan olarak -1 (ignore) dönecek
  }
  ```
- **Patch ve Full-Image Modu:**
  - `mode="patch"`: `(y, x)` etrafından `patch_size` boyutunda yama keser. Girdi: `[17, patch_size, patch_size]`, Etiket: `0` veya `1`. (Eğer etiket -1 ise bu örnek eğitim kümesinden elenir).
  - `mode="full"`: Tüm Türkiye matrisini döner. Girdi: `[17, 800, 1000]`, Etiket: `[800, 1000]` (İstasyon olan yerler 0 veya 1, olmayan yerler -1).

- **Normalizasyon:**
  - Uydu kanalları kendi kanallarına göre normalleştirilir (Mean-Std veya Min-Max).
  - Radar verisi 0-255 arasından 0.0-1.0 arasına normalize edilir.

### [NEW] `src/dataloader.py`
- Eğitim / Validasyon ayrımını (Train/Val split) zamansal (temporal) olarak ayıracak fonksiyonlar eklenecektir (örn: verinin %80'i train, %20'si val günleri şeklinde). Bu, verideki zamansal bağımlılıktan kaynaklı veri sızıntısını (data leakage) önler.

---

## 4. Doğrulama Planı (Verification Plan)
1. DataLoader'dan alınan batch'lerdeki etiket dağılımı kontrol edilecek (Sadece 0 ve 1 değerleri olduğu, 1-7 oktalı verilerin -1 olarak etiketlendiği doğrulanacak).
2. `mode="patch"` için tensor şekillerinin `[Batch, 17, 33, 33]` olduğu teyit edilecek.
3. `mode="full"` için tensor şekillerinin `[Batch, 17, 800, 1000]` olduğu teyit edilecek.
