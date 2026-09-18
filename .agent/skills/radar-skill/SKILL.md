---
name: radar-skill
description: Radar verilerini uydu matrisinin üzerine tam çakıştırma yeteneği
---

# Radar Skill

png dosyasına kaydedilen radar verilerini pyresample veya rioxarray kullanarak uydu matrisinin (200x200) üzerine tam çakıştırma yeteneği

## When to use this skill

- radar verilerini uydu verileri ile çakıştırmak istendiğinde
- Radar verilerini polar koordinattan Mercator projeksiyonuna çakıştırmak istendiğinde

## How to use it

Bağlantı dizesi `MTGCLM_MONGO_URI` ortam değişkeninden okunur (repoya yazmayın).

 tablonun örnek çıktısı:
{
    "dt": 1735063582061,
    "e": 0,
    "h": 0,
    "ll": "22.56073570251465 32.8768196105957 
          46.11445617675781 44.64813232421875",
    "ln": 0,
    "lt": 0,
    "oa": "CMP241224180622",
    "pt": "PPI",
    "r": 0,
    "rn": "CMP",
    "sX": 0,
    "sY": 0,
    "sZ": 0,
    "wc": 1,
    "wm": 1,
    "xS": 1072,
    "yS": 690,
    "zS": 1
  },


- tablonun açıklaması:
"dt": Tarih ve saat, Unix milisaniye formatı,
"e": Radarın yer ile yaptığı açı derecesi,
"h": Radarın deniz seviyesinden yüksekliği,
"ll": Radar ürününün haritaya yerleştirilmesi için belirlenen köşe koordinatları,
"ln": Radarın bulunduğu konumun boylam değeri,
"lt": Radarın bulunduğu konumun enlem değeri,
"oa": Radar verisinden üretilen ürünün dosya adı,
"pt": Radardan elde edilen ürün tipi, (Plan Position Indicator (PPI) kullanılmaktadır),
"r": Radar tarama alanı yarıçapı (km),
"rn": Radar adı,
"sX": Radardan elde edilen ürünün X ekseni çözünürlüğü,
"sY": Radardan elde edilen ürünün Y ekseni çözünürlüğü,
"sZ": Radardan elde edilen ürünün Z ekseni çözünürlüğü, 
"wc": Mercator resim oluşturuldu (0-1), 
"wm": Harita oluşturuldu (0-1),
"xS": Resim eni,
"yS": Resim boyu,
"zS": Resim yüksekliği.
PPI görüntü iki boyutlu olduğu için sZ ve sZ değerleri kullanılmamaktadır.


