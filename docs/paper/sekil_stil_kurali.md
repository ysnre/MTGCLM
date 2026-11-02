# MTGCLM — Makale şekil ve görsel kuralı (TAAC / MAP, Springer)

Sürüm 1 · 2026-10-30 · Uygulayan dosyalar: `springer_taac_map.mplstyle` (matplotlib stili) ve
`springer_fig.py` (boyut, renk, etiket, kayıt ve otomatik kontrol). Repoya yerleşimi: `docs/paper/` ve
`paper/figures/` (P11).

TAAC ve MAP aynı yayınevinin (Springer Wien) standart şekil kurallarını uygular; bu kural ikisi için **tektir**.
Dergi değiştirilirse şekil yeniden üretilmez. Gönderimden önce derginin "Artwork and Illustrations Guidelines"
bölümü bir kez kontrol edilmelidir.

Bu belgede **ZORUNLU** maddeler Springer gereksinimi veya makale içi tutarlılık şartıdır; **ÖNERİ** maddeleri
gerekçeli istisnaya açıktır (istisna şekil betiğinin başına yorum olarak yazılır).

---

## 1. Boyut — ZORUNLU

| Ad | Genişlik | Kullanım |
|---|---|---|
| `sub` | 39 mm | küçük ek panel (nadiren) |
| `single` | 84 mm | tek sütun: kalibrasyon eğrisi, tek panel |
| `onehalf` | 129 mm | 1,5 sütun: 2 panel |
| `double` | 174 mm | tam genişlik: harita, çok panelli sonuç şekilleri |

- Yükseklik ≤ 234 mm.
- Şekil **son baskı boyutunda** üretilir (`new_figure(width, height_mm)`). Sonradan büyütme/küçültme yok.
- `bbox_inches="tight"` **kullanılmaz** (son genişliği değiştirir). Yerleşim `constrained_layout` ile yapılır.

## 2. Yazı — ZORUNLU

- Yazı tipi: Arial (yoksa Helvetica / Liberation Sans). Gölge, dış çizgi vb. efekt yok.
- Boyut basamakları: **8 pt** (tik etiketleri, lejant, şekil içi açıklama) · **9 pt** (eksen adları, panel üst
  başlıkları) · **10 pt kalın** (yalnız panel harfleri). 8 pt altı yok.
- Ondalık ayırıcı **nokta** (İngilizce makale). Türkçe yerel ayar kapalıdır (`axes.formatter.use_locale: False`).
  Binlik ayırıcı: boşluk (`22 506`) veya hiç; virgül kullanılmaz.
- Eksi işareti gerçek eksi (−).
- Birimler SI ve parantez içinde: `Brightness temperature (K)`, `Elevation (m)`, `Reflectivity (dBZ)`.
- Yazı kontrastı arka plana göre ≥ 4,5:1.

## 3. Başlık, panel harfi, altyazı — ZORUNLU

- **Şeklin içinde şekil başlığı yok.** Başlık ve açıklama altyazıda (caption) yazılır; altyazılar metin dosyasında
  durur. (`check_figure` eksen başlığı bulursa uyarır.)
- Panelin neyi gösterdiğini belirten kısa üst etiket (ör. "Unseen weeks") eksen başlığı olarak değil,
  `ax.text(0, 1.03, ...)` ile 9 pt verilebilir. En fazla 4–5 kelime.
- Panel harfleri: **küçük harf, kalın, 10 pt**, eksen kutusunun sol üstünde, dışarıda (`panel_label(ax, "a")`).
  Metinde "Fig. 4a", altyazıda "**a** Unseen weeks, **b** …".
- Şekiller metinde sırayla anılır (Fig. 1, 2, …); ekler Fig. S1, S2, ….
- Altyazı içeriği (her şekil betiği `figXX_caption.txt` üretir): ne gösterildiği, **n**, küme, etiket alt kümesi
  (tümü / yalnız SYNOP), belirsizlik yöntemi (%95 gün-blok bootstrap; test_station'da istasyon-blok), sabit tutulan
  her şey, tüm kısaltmaların açılımı.

## 4. Renk — ZORUNLU (eşleme) / ÖNERİ (palet)

- Palet: Okabe-Ito (renk körlüğü güvenli). Kırmızı–yeşil karşıtlığı yok.
- **Bir varlığın rengi bütün şekillerde aynıdır.** Eşleme `springer_fig.COLORS`'tadır; yeni varlık eklenirse önce oraya eklenir.

| Varlık | Renk | Not |
|---|---|---|
| CNN, tüm girdiler (`B_full`) — **odak model** | `#0072B2` koyu mavi | |
| CNN, uydu + radar (`A_sat`) | `#56B4E9` açık mavi | |
| CNN, AWOS'suz (`C_noawos`) | `#009E73` yeşil-turkuaz | |
| CNN, AWOS'suz + T2m (`C_t2m`) | `#CC79A7` pembe-mor | |
| CNN, yalnız uydu (`A_norad`, P1) | `#999999` gri | |
| U-Net | `#000000` siyah | |
| ResNet-18 ImageNet / CIFAR stem | `#6A3D9A` / `#B294C7` | |
| Gradyan artırma / RF / lojistik | `#E69F00` / `#F0C35E` / `#F5DDA0` | turuncu ailesi |
| FCI CLM | `#D55E00` vermilyon | referans |
| NWC SAF CT | `#8C510A` kahve | referans |

- Merkez piksel ve 30 km alan gibi alt varyantlar **renkle değil işaret şekliyle** ayrılır (kare / daire / üçgen).
- Sürekli değerler: bulut olasılığı `cividis` (0–1, sabit ölçek); işaretli farklar `RdBu_r` **0'da merkezli**
  (`TwoSlopeNorm(vcenter=0)`), simetrik sınır; DEM altlığı `gist_earth` açık tonlarda; IR parlaklık sıcaklığı ters gri.
- İkili bulut maskesi haritası: bulutlu `#F2F2F2`, açık kara `#D9C7A1`, açık deniz `#24476B`, veri yok beyaz + tarama.
- Gri tonlamada da ayırt edilebilirlik kontrol edilir (baskı siyah-beyaz olabilir).

## 5. Çizgi ve işaret — ZORUNLU

- Tüm çizgiler ≥ 0,3 pt (0,1 mm). Varsayılan: veri çizgisi 1,0 pt, eksen 0,6 pt, ızgara 0,4 pt.
- Hata çubuğu: %95 GA, uç çizgisi yok (`capsize 0`), çizgi 0,8 pt.
- Saçılım işareti 3–4 pt; yoğun bulutlarda saydamlık 0,3–0,5.
- Üst ve sağ çerçeve kapalı; ızgara varsayılan kapalı (gerekirse yalnız y, açık gri).

## 6. Grafik türü seçimi — ÖNERİ

| Veri | Tür |
|---|---|
| Modeller × test kümeleri, tek metrik + GA | nokta + GA çizgisi (çubuk grafik değil); y ekseni veriye göre kırpılır, kırpma açıkça işaretlenir |
| Model − referans farkı | dumbbell (iki nokta + bağlantı) + Δ [GA] etiketi (demo: `FigX_demo_reference_comparison`) |
| Kategoriler × kategoriler (gündüz/gece × ay) | değerleri hücreye yazılmış ısı haritası (≤ 200 hücre) |
| Kalibrasyon | güvenilirlik eğrisi + altında örnek histogramı |
| Mekânsal alan | harita (§7) |
| Kanal önemi | yatay nokta grafiği, kanallar dalga boyuna göre sıralı, gündüz/gece yan yana |

- Seed'ler arası yayılım küçükse (s.s. ≤ 0,002) ortalama + GA; seed noktaları yalnız ek şekilde.
- Çoğunluk sınıfı başarımı gibi referans değerler ince gri kesikli çizgi, doğrudan etiketli.

## 7. Haritalar — ZORUNLU

- Projeksiyon: veri grid'i düzenli enlem-boylam → `cartopy.crs.PlateCarree()`. Kapsam sabit:
  `MAP_EXTENT = (25, 45, 35, 43)` (lon_min, lon_max, lat_min, lat_max).
- Kıyı çizgisi ve ülke sınırları Natural Earth 10 m, 0,4 pt, koyu gri. İl sınırları yok (gerekmedikçe).
- Enlem-boylam ızgarası 5° aralıklı, etiketler sol ve altta, `25°E`, `40°N` biçiminde.
- Raster veri `pcolormesh` veya `imshow(extent=…, origin=…)` ile **yeniden örneklemeden** çizilir; grid yönü
  CLAUDE.md'deki tanımla (lat 42,995 → 35,005) uyumlu olmalı — kuzey yukarıda olduğunu kıyı çizgisiyle doğrula.
- Renk çubuğu: yatay, haritanın altında, etiket + birim; çok panelli haritalarda **ortak ve sabit ölçek**.
- İstasyon işaretleri: SYNOP daire, METAR üçgen, holdout içi boş + kalın kenar. Açık/bulutlu gözlem şekille
  (içi boş / dolu) gösterilir, renkle değil.
- Ölçek çubuğu ve kuzey oku gerekmez (enlem-boylam ızgarası var).
- FCI CLM ve NWC SAF CT kendi geometrilerinden **aynı 800 × 1000 grid'e** getirilip çizilir; yöntem altyazıda.

## 8. Dosya biçimi ve çözünürlük — ZORUNLU

`save_figure(fig, name, kind=...)` aynı anda üretir:

| Dosya | Amaç |
|---|---|
| `name.pdf` | vektör, yazı tipleri gömülü (`pdf.fonttype 42`) — LaTeX'e bu girer |
| `name.tif` | LZW sıkıştırmalı; `kind="line"` 1200 dpi, `"combination"` 600 dpi (grafik + yazı, harita + renk çubuğu), `"halftone"` 300 dpi (yalnız görüntü) |
| `name_preview.png` | 200 dpi önizleme |
| `name_check.txt` | otomatik kontrol raporu; **`OK` değilse şekil teslim edilmez** |

- Renk modu RGB. Harita ve görüntü içeren şekillerde `pcolormesh(..., rasterized=True)` ile PDF boyutu küçültülür
  (yazılar vektör kalır).
- Dosya adları: `Fig01_study_area`, `Fig04_main_results`, …, ekler `FigS01_…`.

## 9. Tablolar — ZORUNLU

- Tablolar görüntü değil, düzenlenebilir metin (LaTeX `booktabs`). Dikey çizgi yok.
- Ondalık: BalAcc/MCC 3 ondalık, farklar işaretli 3 ondalık (`+0.045`), GA köşeli parantezde `[+0.033, +0.061]`.
- Her tablonun altında n ve kısaltma açılımları.

## 10. Terminoloji — ZORUNLU (şekil, tablo ve metinde aynı)

Kod adları şekle yazılmaz; `springer_fig.LABELS` kullanılır:
`Unseen weeks` (test_time) · `Unseen stations` (test_station) · `Unseen weeks and stations` (test_both) ·
`CNN, all inputs` (B_full) · `U-Net` · `Gradient boosting` (hgb_aux) · `FCI CLM` · `NWC SAF CT` ·
`centre pixel` / `30 km area` · `Balanced accuracy` · `MCC` · `Brier score` · `ECE`.
**İngiliz yazımı** kullanılır (K20): `centre`, `colour`, `analyse`, `-ise`. Şekil, tablo ve metinde aynı.

**Odak model (K20):** nokta sonuçlarını gösteren şekillerde `B_full` (CNN, all inputs) koyu mavi ve en belirgin
seridir; tam harita ürünlerini gösteren şekillerde (Fig. 8–9) ürün modeli `D_unet` (U-Net) kullanılır ve altyazıda
belirtilir.

## 11. Şekil betiği iskeleti

```python
# paper/figures/make_fig05_references.py
import pandas as pd
from springer_fig import new_figure, panel_label, COLORS, LABELS, save_figure

df = pd.read_csv("../../artifacts_v3/analysis/references.csv")   # tek veri kaynağı
fig, axes = new_figure("double", height_mm=70, ncols=3, sharey=True)
...                                                               # çizim
for ax, letter in zip(axes, "abc"):
    panel_label(ax, letter)
problems = save_figure(fig, "Fig05_references", kind="combination")
assert not problems, problems
open("figures/Fig05_references_caption.txt", "w", encoding="utf-8").write(
    "Fig. 5 Balanced accuracy of the station-trained CNN and operational products on identical "
    "SYNOP-labelled samples. ... n = ...; 95% day-block bootstrap confidence intervals.")
```

Kurallar: her betik tek bir şekil üretir; yalnız `artifacts_v3/` ve `data/` okur; rastgelelik varsa sabit tohum;
`plt.savefig` değil `fig.savefig` (yardımcı bunu yapar).

## 12. Gönderim öncesi şekil kontrol listesi

- [ ] Bütün `*_check.txt` dosyaları `OK`
- [ ] PDF'te yazı tipleri gömülü (`pdffonts Fig05_references.pdf` → hepsi `emb yes`)
- [ ] %100 yakınlaştırmada, baskı boyutunda okunuyor; en küçük yazı 8 pt
- [ ] Gri tonlama ve döteranopi benzetiminde seriler ayırt ediliyor
- [ ] Aynı varlık her şekilde aynı renk ve aynı adla
- [ ] Altyazıda n, küme, etiket alt kümesi, GA yöntemi, kısaltmalar
- [ ] Metindeki sayılar şekildeki ve tablodaki sayılarla aynı (tek değer kuralı)
- [ ] Yazım tercihi (İngiliz / Amerikan) tutarlı
