# MTGCLM — Makale yol haritası (TAAC + MAP paralel hazırlık)

Sürüm 2 · 2026-10-30 · Kaynak: `E:\belgeler\MTGCLM\CLAUDE.md` (K1–K18), `artifacts_v3/analysis/`
Eşlik eden dosyalar: `makale_bolumleri_yapilacaklar.md` (bölüm bölüm iş listesi), `eksik_promptlar.md`
(her eksik için hazır prompt), `sekil_stil_kurali.md` + `springer_taac_map.mplstyle` + `springer_fig.py`
(şekil kuralı), `literatur_konumlandirma.md` (literatür ve konumlandırma).

Durum etiketleri: **[VAR]** hazır · **[KISMİ]** veri var, sunuma dönüştürülmeli · **[EKSİK]** üretilmeli ·
**[KARAR]** kullanıcı kararı.

---

## 0. Strateji: tek gövde, iki çerçeve

TAAC ve MAP aynı yayınevi (Springer Wien), aynı şekil kuralları, aynı öz sınırı (150–250 kelime) ve aynı
LaTeX şablonu ailesini kullanır. Bu yüzden **tek bir makale gövdesi** yazılır; dergiye göre değişen yalnız
**başlık, öz, Giriş'in 1. ve son paragrafı, Tartışma'nın vurgusu, anahtar kelimeler ve kapak mektubu**dur.
Bu kısımlar LaTeX'te `\ifTAAC ... \else ... \fi` anahtarıyla iki sürüm olarak tutulur (bkz. P13).

| | TAAC (birincil) | MAP (yedek) |
|---|---|---|
| Ana soru | Yer gözlemiyle eğitilmiş bir FCI bulut maskesi Türkiye üzerinde operasyonel ürünlerden daha güvenilir bir bulut izleme ürünü sağlıyor mu? | FCI'ın spektral kanalları, radar ve yüzey alanları bulut/açık ayrımına fiziksel olarak nasıl katkı veriyor, ve bu katkı istatistiksel olarak nasıl sınanır? |
| Vurgu | Uygulamalı meteoroloji, bulut sıklığının mekânsal/zamansal dağılımı, ürün güvenilirliği | Fiziksel yorum (kanal önemi, gece/sis), istatistiksel yöntem (blok-bootstrap, bloklu CV, etiket duyarlılığı) |
| Zorunlu ek içerik | Bulut sıklığı haritaları (aylık, günün saati), gündüz-gece ve mevsim kırılımı ana sonuçlarda | Kanal önem analizi + fiziksel yorum, gece hata fiziği |
| Uzunluk | ≤ 100.000 karakter (her şey dahil) → hedef ≤ 80.000 | Kılavuzdan kontrol; aynı hedef |
| Başlık önerisi | *A station-trained cloud mask for Meteosat Third Generation FCI over Türkiye: evaluation against operational products and implications for cloud monitoring* | *What drives cloud detection with MTG-FCI? Spectral, radar and surface contributions to a station-trained cloud mask over Türkiye* |

**Ana sav (her iki dergi):** insanlı sinoptik okta gözlemiyle eğitilen model, aynı istasyon-zaman
satırlarında FCI CLM ve NWC SAF CT'den anlamlı biçimde yüksek dengeli doğruluk veriyor
(yalnız SYNOP, test_time: CNN tüm girdiler − CLM 30 km +0,045 [+0,033, +0,061]; − NWC 30 km
+0,023 [+0,015, +0,032]; `references.csv`). Model ailesi karşılaştırması (CNN ≈ U-Net ≈ ResNet ≈ HGB)
ikincil, dürüst nötr bulgu olarak sunulur.

---

### 0.1 Alınan kararlar (2026-10-30; CLAUDE.md K20)

| Konu | Karar | Sonucu |
|---|---|---|
| Ana model | Tam harita ürünü **U-Net**; nokta tablolarında **CNN, tüm girdiler** odak satır | Fig. 8–9 U-Net; Fig. 4–5, Tab. 3–4 CNN odaklı. Yöntem'de "U-Net, CNN ile istatistiksel olarak eşdeğer nokta başarımı verdiği ve tam evrişimli olduğu için ürün modeli seçildi" gerekçesi |
| Zaman | ≥ 1 ay | S1–S6'nın tamamı ilk gönderime; yalnız P20 opsiyonel; bir iç okuma turu eklenir |
| Yazım | LaTeX + İngiliz yazımı | `centre`, `colour`, `analyse`; `-ise` / `-ize` seçimi tek tip (öneri: `-ise`) |
| Veri paylaşımı | WMO UDP'ye dayanılarak paylaşım; radar ve AWOS için MGM'den kısa yazılı teyit | §5'teki iki katmanlı beyan |
| Şekil kuralı | Repoya kuruldu | `docs/paper/sekil_stil_kurali.md`, `paper/figures/`, Kural 4 |

## 1. Zaman kısıtına göre sıralanmış iş akışı

Kritik yol: **P4 tam harita çıkarımı** (TAAC'ın bulut sıklığı haritaları ve her iki derginin nitel şekli buna bağlı).
CPU koşuları gece boyunca paralel bırakılır; gündüz yazım yapılır.

| Sprint | İçerik | Prompt | Süre (tahmin) | İki dergi için |
|---|---|---|---|---|
| **S1 — mevcut sonuçlarla** | Şekil stil kuralını repoya kur; F4–F8 ve T2–T5'i mevcut CSV'lerden üret; Veri + Yöntem taslağı; LaTeX iskeleti | P11, P12, P13, P14 | 2–3 gün | ortak |
| **S2 — gece koşuları** (S1 ile paralel) | Radarsız ablasyon (3 seed + 5 kat CV); HGB kontrollü kıyas + uzamsal karıştırma CNN; kanal önem analizi | P1, P2, P3 | CPU ~8–10 sa (2 gece) | ortak; P3 MAP için zorunlu |
| **S3 — tam harita** | `infer_scene.py`; nitel sahne paneli; bulut sıklığı haritaları | P4, P5, P6 | 3–4 gün | P6 TAAC için zorunlu |
| **S4 — ek analizler** | CLM C/O dönem ayrımı; gece/sis hata analizi; etiket politikası duyarlılığı; paralaks kaydırma tahmini | P7, P8, P9, P10 | 1–2 gün | ortak |
| **S5 — metin** | Bulgular, Giriş, Tartışma, Sonuç; iki dergi sürümü; literatür tam metin doğrulaması | P14, P15, P16 | 4–5 gün | sürüm ayrımı burada |
| **S6 — gönderim** | Tekrarlanabilirlik paketi; beyanlar; kapak mektupları; son kontrol | P17, P18, P19 | 1–2 gün | dergiye özel |

Toplam: **~2,5–3 hafta** yoğun çalışma. Zaman daha da kısıtlıysa **en küçük gönderilebilir paket**:
S1 + P1 + P3 + P4/P5 (2–3 sahne) + P6 (yalnız TAAC) + S5 + S6 (~10–12 iş günü). P2, P7–P10 hakem turunda
"revizyonda eklendi" olarak verilebilir; bu durumda Tartışma'da sınırlılık olarak açıkça anılmalı.

---

## 2. Deney ve analiz eksikleri

| # | İş | Neden | Dergi | Durum | Prompt |
|---|---|---|---|---|---|
| 1.1 | Radarsız (uydu-only) ablasyon: `A_norad` (16 FCI), `B_norad` (16 FCI + aux, `radar_cov` hariç) | Radar MGM'ye ait, paylaşılmıyor; ürünün taşınabilirliği | ikisi | [EKSİK] | P1 |
| 1.2 | HGB kontrollü kıyas: HGB merkez-1×1 / 5×5 / +doku öznitelikleri; CNN uzamsal karıştırılmış yama | "Derinlik gerekli mi?" sorusunun kontrollü yanıtı | ikisi | [EKSİK] | P2 |
| 1.3 | Kanal önem analizi: permütasyon (grup ve tekil) + Integrated Gradients, gündüz/gece ayrı | Fiziksel yorum | **MAP zorunlu**, TAAC önerilir | [EKSİK] | P3 |
| 1.4 | `infer_scene.py`: 800×1000 olasılık + ikili maske (GeoTIFF/NetCDF + PNG) | Beyan edilen ürün | ikisi | [EKSİK] | P4 |
| 1.5 | Nitel sahne paneli (4–6 sahne: gündüz/gece, sis, kar, sirrüs, konveksiyon) + CLM/NWC yan yana | F9 | ikisi | [EKSİK] | P5 |
| 1.6 | Bulut sıklığı haritaları (aylık; günün saatine göre) + CLM ile fark haritası | Klimatolojik boyut | **TAAC zorunlu** | [EKSİK] | P6 |
| 1.7 | FCI CLM C (Mar–Nis 2025) / O (Ara 2025–Haz 2026) ayrımı | Ön-operasyonel veriyle kıyas eleştirisi | ikisi | [EKSİK] | P7 |
| 1.8 | Gece/sis/alçak bulut hata analizi (NWC bulut tipi kırılımı, RH, okta 9) | Gece BalAcc ~0,94 vs gündüz ~0,97 | ikisi; MAP'te daha derin | [KISMİ] | P8 |
| 1.9 | Etiket politikası duyarlılığı (verify vs keep; yeniden eğitimsiz değerlendirme) | Yer gerçeği tanımının etkisi | ikisi | [KISMİ] | P9 |
| 1.10 | Paralaks kaydırma büyüklüğü (VZA 51,6–64,7°, CTH 1/3/6/10 km) | Düzeltme yapılmadı (K13) | ikisi | [EKSİK] | P10 |
| 1.11 | U-Net hedef-piksel jitter | Tam harita kenar artefaktı (yalnız P4'te sorun görülürse) | opsiyonel | [EKSİK] | P20 |

Elde hazır olanlar (yeniden koşulmaz): mimari ablasyonu (6 × 3 seed), modalite ablasyonu
(A/B/C/C_t2m × 3 seed), U-Net, iki ResNet stem'i, bloklu 5-katlı CV (4 modalite × 5 kat), 6 klasik
model, blok-bootstrap GA, eşleştirilmiş farklar, Brier/ECE, gündüz-alacakaranlık-gece / SYNOP-METAR /
yükseklik / ay kırılımları, CLM ve NWC kıyası.

---

## 3. Şekil ve tablo seti

Tüm şekiller `sekil_stil_kurali.md`'ye uyar (Springer: 84 / 129 / 174 mm genişlik, Arial 8–10 pt, başlık
şeklin içinde değil altyazıda, panel harfleri küçük ve kalın). Her şekil `paper/figures/make_figXX.py`
betiğinden üretilir.

| # | İçerik | Genişlik | Kaynak | Durum | TAAC | MAP |
|---|---|---|---|---|---|---|
| Fig. 1 | Çalışma alanı: DEM altlığı, SYNOP/METAR istasyonları, 30 holdout istasyon, radar kapsamı | 174 mm | DEM, istasyon listesi, `radar_cov` | [EKSİK] | ✓ | ✓ |
| Fig. 2 | Veri akışı ve etiket politikası şeması | 174 mm | — | [EKSİK] | ✓ | ✓ |
| Fig. 3 | Model mimarisi (Medium-Flat CNN + U-Net) | 174 mm | `model.py` | [KISMİ] | ek | ✓ |
| Fig. 4 | Ana sonuç: üç test kümesinde BalAcc, %95 blok-bootstrap GA | 174 mm | `main_table.csv` | [VAR→stil] | ✓ | ✓ |
| Fig. 5 | Referans kıyası (yalnız SYNOP): model vs CLM/NWC, merkez ve 30 km | 174 mm | `references.csv` | [VAR→stil] örnek üretildi | ✓ | ✓ |
| Fig. 6 | Kırılımlar: gündüz/alacakaranlık/gece × ay × yükseklik | 174 mm | `breakdowns.csv` | [KISMİ] | ✓ (ana) | ✓ |
| Fig. 7 | Kanal önemi (permütasyon + IG), gündüz/gece | 129–174 mm | P3 | [EKSİK] | ek | ✓ (ana) |
| Fig. 8 | Nitel tam harita sahneleri + CLM/NWC | 174 mm | P4/P5 | [EKSİK] | ✓ | ✓ |
| Fig. 9 | Bulut sıklığı haritaları + CLM farkı | 174 mm | P6 | [EKSİK] | ✓ (ana) | ek |
| Fig. 10 | Kalibrasyon (güvenilirlik eğrisi) + okta tutarlılığı (`test_partial`) | 84–129 mm | `reliability_test_time.png`, sonuç JSON | [KISMİ] | ek | ✓ |
| Tab. 1 | Girdi kanalları (16 FCI + radar + 5 aux): dalga boyu, çözünürlük, kaynak | — | — | [EKSİK] | ✓ | ✓ |
| Tab. 2 | Bölme istatistikleri (N, bulutlu oranı, çoğunluk sınıfı, istasyon/hafta) | — | `splits/*.json` | [VAR] | ✓ | ✓ |
| Tab. 3 | Ana karşılaştırma: BalAcc/MCC/Brier/ECE ± s.s. + GA | — | `main_table.csv` | [VAR] | ✓ | ✓ |
| Tab. 4 | Ablasyon: modalite eşleştirilmiş farkları (tek bölme + CV birleşik) + radarsız | — | `pairs.csv`, `cv_pairs.csv`, P1 | [KISMİ] | ✓ | ✓ |
| Tab. 5 | Hesaplama maliyeti: parametre, eğitim süresi, sahne başına çıkarım süresi | — | JSON `wall_minutes`, `bench_speed.py`, P4 | [KISMİ] | ek | ✓ |

TAAC'ta "ek" işaretliler Supplementary Information'a gider (karakter sınırı); MAP'te ana metinde kalabilir.

---

## 4. Metin

Bölüm bölüm yapılacaklar: `makale_bolumleri_yapilacaklar.md`. Özet:

1. Başlık, öz (150–250 kelime, tanımsız kısaltma ve kaynak yok), 4–6 anahtar kelime — **dergiye özel**
2. Giriş — ¶1 ve son ¶ dergiye özel, gövde ortak
3. Veri — ortak
4. Yöntem — ortak; MAP sürümünde istatistik alt bölümü daha ayrıntılı
5. Bulgular — ortak sıra; TAAC'ta kırılım ve sıklık haritaları öne, MAP'te kanal önemi öne
6. Tartışma — dergiye özel vurgu
7. Sonuçlar — ortak
8. Beyanlar (Springer): Funding, Competing interests, Ethics approval (N/A), Data availability, Code
   availability, Author contributions (CRediT); yapay zekâ araçlarının kullanımı Yöntem'de belirtilir

---

## 5. Tekrarlanabilirlik paketi ve veri politikası

**WMO Unified Data Policy (Res. 1, Cg-Ext 2021) araştırma notu**

- Politika iki kategori tanımlar: **core** (değişimi zorunlu, "free and unrestricted") ve **recommended**
  (değişimi önerilir, kullanım koşulu konabilir). Core verinin değişimi, verinin kaynağından bağımsız olarak zorunludur.
- Core yüzey verisi: GBON gözlemleri ve WIGOS El Kitabı'nda (WMO-No. 1160) tanımlanan diğer gözlemler; RBSN
  istasyonlarının 00/06/12/18 UTC gözlemleri ve CLIMAT raporları core sayılır.
- Türetilmiş veri, kaynak verinin core/recommended koşulunu taşır.
- Politika WMO üyesi hükümetlere ve **uluslararası değişime** yöneliktir; MGM'nin ulusal veri dağıtım düzenlemesini
  kendiliğinden ortadan kaldırmaz.
- **Bulunamayan:** 1982 AWOS istasyonunun çoğu ve radar hacim verisi core listede açıkça görünmüyor; büyük
  olasılıkla recommended. Bu turda teyit edemedim.

**Sonuç ve öneri**

1. SYNOP (GBON/RBSN istasyonları) ve METAR türevleri: paylaşılabilir, UDP'ye atıfla.
2. Radar ve AWOS türevleri (radar kanalı, `radar_cov`, `t2m`, `rh2m`): yazar MGM personeli olduğu için kurum içi kısa
   bir yazılı teyit (e-posta yeterli) alınması önerilir. Springer gönderim şartı zaten yayının kurumdaki yetkili
   makamlarca (açık ya da örtük) onaylandığını varsayar; teyit bu beyanı belgelendirir.
3. Teyit gelene kadar Zenodo paketi iki katmanlı hazırlanır: (a) FCI + statik alanlar + SYNOP/METAR etiketleri hemen
   paylaşılır; (b) radar ve AWOS türevleri teyit sonrası eklenir.

**Veri bulunabilirlik beyanı taslağı (İngilizce, iki sürüm)**

> *(teyit alındıysa)* MTG-FCI Level-1c data and the FCI Cloud Mask product are available from the EUMETSAT Data Store.
> The derived training dataset (FCI, radar and surface-derived fields, station labels), data splits and model
> predictions are available at Zenodo (DOI: …). Surface observations and weather radar data were provided by the
> Turkish State Meteorological Service (MGM); SYNOP observations are core data under the WMO Unified Data Policy
> (Resolution 1, Cg-Ext 2021).
>
> *(teyit alınmadıysa)* … The derived dataset without radar and automatic-station fields is available at Zenodo
> (DOI: …). Weather radar and automatic weather station data are available from MGM upon reasonable request.

1. `src/make_public_sample.py` ile ~2.000 yamalık örneklem (`data/mtg_sample.h5`).
2. Zenodo: örneklem veya türetilmiş H5 (yukarıdaki iki katmana göre) + `splits/*.npz` + `artifacts_v3/*.json` + analiz CSV'leri → DOI.
3. GitHub sürüm etiketi `v1.0-taac` (veya `v1.0-map`) + ağırlıklar release eki.
4. README'de "her şekil/tabloyu üreten komut" bölümü; `CITATION.cff` güncel.

---

## 6. Gönderim paketi (dergiye özel)

| | TAAC | MAP |
|---|---|---|
| Kapak mektubu | Uygulamalı meteoroloji / bulut izleme vurgusu; 7 aylık verinin neden yeterli olduğu (operasyonel ürün doğrulaması) | Fiziksel süreç + istatistiksel teknik vurgusu |
| Önerilen hakemler (3–5) | Uydu–SYNOP bulut karşılaştırması yazarları, TAAC'ta bulut/güneşlenme yazarları | Geostasyoner bulut fiziği, radar–uydu yazarları |
| Ek dosya | Supplementary Information (ek şekiller, tam tablolar) | Aynı |
| Son kontrol | Karakter sayımı ≤ 100.000 (öz + metin + tablo + altyazı + kaynak) | Kılavuz sınırları |

---

## 7. Kritik riskler

1. **TAAC masa reddi (kapsam):** 7 aylık veri. Önlem: P6 bulut sıklığı haritaları + kırılımlar ana sonuçlarda + kapak mektubu.
   Reddedilirse aynı dosyalarla MAP'e geçiş: başlık/öz/Giriş ¶1/Tartışma anahtarı değiştirilir.
2. **"Neden derin öğrenme?"** HGB 0,959 ≈ CNN 0,957–0,959. Önlem: ana sav ürün doğrulaması; P2 kontrollü
   kıyası ve tam harita üretiminin hesaplama avantajı (P4, Tab. 5) Tartışma'da.
3. **Paralaks düzeltmesi yok:** P10 ile kaydırma büyüklüğü sayısal olarak verilir; r30 alan eşlemesinin bu kaydırmayı kapsadığı gösterilir.
4. **CLM'in ön-operasyonel dönemi:** P7 ile ayrı raporlanır.
5. **Veri paylaşımı:** radar ve yer gözlemi MGM'ye ait. Veri bulunabilirlik beyanı ve örneklem paketi gönderimden önce hazır olmalı.
