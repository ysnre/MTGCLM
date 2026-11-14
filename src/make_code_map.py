"""
make_code_map.py — kod haritası infografiğini üretir (K19, 30 Eki 2026)
======================================================================
src/*.py ve notebooks/*.ipynb dosyalarını Python'un kendi ayrıştırıcısıyla (ast) tarar ve projenin bütün
parçalarının birbirine nasıl bağlandığını gösteren tek dosyalık, etkileşimli bir HTML sayfası yazar:

  - modüller (dosyalar), sınıflar, metotlar, fonksiyonlar; imza, docstring'in ilk satırı, satır numarası
  - proje içi import'lar (fonksiyon içindeki tembel import'lar dahil)
  - fonksiyon / sınıf düzeyinde çağrılar (hangi fonksiyon hangisini çağırıyor; modül içi ve modüller arası)
  - alt süreçle çalıştırılan betikler (ör. run_all.py → train.py)
  - okunan / yazılan veri dosyaları (dize sabitleri ve argparse varsayılanlarından sezgisel)
  - komut satırı argümanları (argparse), dış kütüphaneler

Sayfa: docs/mtgclm_kod_haritasi.html — harici bağımlılık yok, veri sayfaya gömülü, açık/koyu tema.
Görünümler: Modül haritası (iş akışı sütunları + bağlantılar), Fonksiyon ağı (seçilen fonksiyonu çağıranlar /
çağırdıkları), Dizin (aranabilir tüm sınıf ve fonksiyonlar).

Kullanım (proje kökünden; her kod değişikliğinden sonra — Kural 3):
    python src/make_code_map.py
    python src/make_code_map.py --out docs/mtgclm_kod_haritasi.html
Yeni bir modül eklendiğinde MODULE_INFO'ya grubunu ve tek satırlık açıklamasını ekleyin; eklenmezse sayfada
"Sınıflandırılmamış" grubunda uyarıyla görünür. Yalnız standart kütüphane kullanır (Python ≥ 3.9).
"""
import argparse
import ast
import glob
import json
import os
import re
import subprocess
import sys
import time

for _stream in (sys.stdout, sys.stderr):           # Windows cp1254 borusu (K17)
    try:
        _stream.reconfigure(errors="backslashreplace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------- gruplar (iş akışı sırası)
GROUPS = [
    ("ham", "1 · Ham veri hazırlama", "Uydu, radar, yer gözlemi ve DEM'i ortak 800×1000 ızgaraya getirir", "#3d9be9"),
    ("h5", "2 · H5 veri seti", "33×33 yamaları çıkarır; aux kanalları, etiketleri ekler; kompakt H5'i üretir", "#22b8a7"),
    ("bolme", "3 · Bölme ve etiket politikası", "Sızıntısız train/val/test bölmeleri, CV katları, METAR NSC politikası", "#9a7cf0"),
    ("yukleme", "4 · Veri yükleme", "H5'ten batch üretimi: yapılandırma, Dataset, hızlı yükleyici", "#3fbf73"),
    ("model", "5 · Model, kayıp, metrik", "Ağ mimarileri, merkez piksel kaybı, değerlendirme metrikleri", "#e0a21f"),
    ("egitim", "6 · Eğitim", "Derin modellerin ve klasik modellerin eğitimi, hız ölçümü", "#f07a45"),
    ("kosu", "7 · Koşu yönetimi", "Deney matrisini çökmeye dayanıklı biçimde sırayla çalıştırır", "#ec5a72"),
    ("analiz", "8 · Analiz ve rapor", "Koşu sonuçlarını tablolar; anlamlılık, kırılım ve referans analizleri", "#d16fcf"),
    ("referans", "9 · Referans ürünler", "FCI CLM ve NWC SAF Cloud Type çıkarımı ve aynı örneklerde kıyas", "#6f93bf"),
    ("arac", "10 · Araçlar ve defterler", "Kod haritası, Colab defterleri, boş/yer tutucu dosyalar", "#8b97a5"),
]
# modül -> (grup, tek satırlık açıklama, eski ise gerekçe)
MODULE_INFO = {
    # 1 ham
    "coknc": ("ham", "MTG FCI L1c dosyalarını satpy ile okuyup Türkiye kesitlerini NetCDF olarak yazar (betik)", None),
    "satellite_grid": ("ham", "Örnek NetCDF'ten uydu ızgarasını (pyresample SwathDefinition) okur", None),
    "grid_utils": ("ham", "Düzenli enlem/boylam ızgarası, km projeksiyonu, cos(SZA), 10 dk pencere yardımcıları", None),
    "resampler": ("ham", "Radar görüntüsünü uydu ızgarasına en yakın komşu ile yeniden örnekler", None),
    "radar_loader": ("ham", "Radar PNG + radar.csv meta verisini okur; dBZ mavi kanalda", None),
    "process_radar_pipeline": ("ham", "Radarı uydu zamanlarına hizalayıp ızgaraya taşır (küçük veri sürümü)",
                               "Büyük veri için process_radar_pipeline_large.py kullanılıyor"),
    "process_radar_pipeline_large": ("ham", "Radar hizalama, büyük veri / çok süreçli sürüm (MongoDB meta)", None),
    "observation_loader": ("ham", "SYNOP + METAR gözlemlerini okuyup birleştirir", None),
    "awos_matcher": ("ham", "İstasyonları uydu ızgarasındaki en yakın piksele eşler", None),
    "process_awos_pipeline": ("ham", "Gözlemleri uydu zamanları ve piksellerle hizalar (küçük veri sürümü)",
                              "Büyük veri için process_awos_pipeline_large.py kullanılıyor"),
    "process_awos_pipeline_large": ("ham", "Gözlem hizalama, büyük veri / çok süreçli sürüm", None),
    "make_dem_npy": ("ham", "DEM GeoTIFF'i (Copernicus GLO-90) 800×1000 .npy ızgarasına çevirir", None),
    "awos_scalar_loader": ("ham", "v1 geç füzyon için istasyon AWOS skalerlerini okur",
                           "v1 skaler füzyon; v2'de raster aux kanallar kullanılıyor (K1)"),
    # 2 h5
    "create_patch_dataset": ("h5", "Hizalanmış uydu + radar + gözlemden 33×33 yamaları çıkarıp ana H5'i yazar", None),
    "update_h5_aux_channels": ("h5", "Ana H5'e 5 raster yardımcı kanal ekler: t2m, rh2m, elev, cos_sza, radar_cov", None),
    "update_h5_labels_v2": ("h5", "Parametrik etiket politikası, ham okta, obs_time, çift örnek işareti", None),
    "export_compact_h5": ("h5", "Bölmedeki örnekleri karıştırıp float16 kompakt H5'e (mtg_train_v2.h5) yazar", None),
    "make_public_sample": ("h5", "Kompakt H5'ten paylaşılabilir küçük örneklem üretir", None),
    "update_h5_labels": ("h5", "Etiketleri ilk politika ile yazar (v1)", "update_h5_labels_v2.py ile değiştirildi"),
    "update_h5_scalars": ("h5", "H5'e v1 AWOS skalerlerini ekler", "v1 skaler füzyon (K1)"),
    "compress_h5": ("h5", "Ana H5'in sıkıştırılmış kopyasını yazar (eski Colab depolama çözümü)",
                    "Kompakt H5 (export_compact_h5.py) ile gereksizleşti"),
    "update_compressed_labels_local": ("h5", "Sıkıştırılmış H5'teki etiketleri yerinde günceller",
                                       "compress_h5.py ile birlikte eskidi"),
    # 3 bölme
    "make_splits": ("bolme", "Hafta-blok + istasyon ayrımlı tek bölme (train/val/test_time/test_station/test_both)", None),
    "make_cv_splits": ("bolme", "Bloklu 5-katlı çapraz doğrulama bölmeleri", None),
    "make_label_policy": ("bolme", "METAR NSC politikası (keep/verify/ignore) → split_v2_verify, cv5_verify", None),
    # 4 yükleme
    "config": ("yukleme", "Colab/yerel yollar, H5 yolu, batch boyutu, öğrenme hızı, cihaz", None),
    "h5_dataset": ("yukleme", "MTGH5Dataset: H5 meta verisi + örnek-örnek okuma; aux normalizasyonu", None),
    "fast_loader": ("yukleme", "Hızlı yükleyici: blok okuma, RAM / akış / önbellek modları, batch çevirme (K16)", None),
    "dataloader": ("yukleme", "get_split_dataloaders: bölme dosyasından yükleyiciler; legacy ↔ hızlı yükleyici", None),
    "dataset": ("yukleme", "MTGCloudDataset: NetCDF'ten doğrudan okuyan ilk yükleyici",
                "H5 hattından önceki yol; yalnız get_dataloaders(h5_path=None) kullanır"),
    # 5 model
    "model": ("model", "MTGConvNet (6 varyant), MTGResNet18 (ImageNet / CIFAR girişi), MTGUNet", None),
    "loss": ("model", "MaskedBCELoss: U-Net'te yalnız merkez piksel (16,16) denetimi", None),
    "metrics": ("model", "Dengeli doğruluk, MCC, F1, AUC, karışıklık matrisi, çoğunluk referansı", None),
    # 6 eğitim
    "train": ("egitim", "Sınıflandırma modellerinin eğitimi (MTGConvNet, ResNet18); test, tahmin kaydı", None),
    "train_unet": ("egitim", "U-Net eğitimi (merkez piksel denetimi); test, tahmin kaydı", None),
    "run_utils": ("egitim", "Tekrarlanabilirlik, RNG checkpoint, ortam bilgisi, SYNOP-only metrik, tahmin kaydı", None),
    "baselines": ("egitim", "Klasik modeller: çoğunluk, eşikler, lojistik regresyon, RF, HGB", None),
    "bench_speed": ("egitim", "Sentetik batch'le model başına eğitim hızı ve koşu süresi tahmini", None),
    # 7 koşu
    "run_all": ("kosu", "Deney matrisi (aşamalar), .done işaretleri, log, manifest; train/train_unet/baselines'ı çalıştırır", None),
    # 8 analiz
    "aggregate_results": ("analiz", "Koşu JSON'larını ortalama ± s.s. tablolara (MD/CSV/LaTeX) ve grafiklere toplar", None),
    "analyze_preds": ("analiz", "Örnek bazlı tahminlerden blok-bootstrap, CV, kırılımlar, CLM/NWC kıyası (K18)", None),
    "plot_results": ("analiz", "v1 ablasyon sonuç grafiği", "aggregate_results.py ile değiştirildi"),
    # 9 referans
    "extract_fci_clm": ("referans", "FCI CLM netCDF → istasyon çevresi sınıf sayımları (merkez, 3×3, r5–r30)", None),
    "extract_nwcsaf_ct": ("referans", "NWC SAF Cloud Type PNG → istasyon çevresi bulut tipi sayımları", None),
    "eval_clm_on_splits": ("referans", "CLM'yi model örneklerinde değerlendirir; load_clm_for paylaşılır", None),
    "eval_nwc_and_metar": ("referans", "NWC değerlendirmesi, METAR NSC doğrulaması, bulut tipi kırılımı (betik)", None),
    "clm_vs_synop_quicklook": ("referans", "CLM ↔ ham SYNOP ön bakış (merkez vs alan)", None),
    # 10 araç
    "verify_compact_h5": ("arac", "Yeniden üretilen kompakt H5'i h5meta_train_v2.npz ile birebir karşılaştırır (12 Kas 2026)", None),
    "verify_channel_order": ("arac", "Kompakt H5 kanal sırasını ham NetCDF ile eşleştirip doğrular (salt okuma; K23)", None),
    "make_code_map": ("arac", "Bu sayfayı üretir: kodu tarar, bağlantıları çıkarır, HTML yazar", None),
    "__init__": ("arac", "Paket işaretçisi (boş)", None),
    "data_fusion": ("arac", "Boş dosya (yer tutucu)", "İçeriği yok"),
    "nb:clmcl": ("arac", "Colab defteri: Drive, H5 kopyası, depo klonu, run_all çağrısı", None),
    "nb:check_data": ("arac", "Veri kontrol defteri (boş)", "İçeriği yok"),
}

# ----------------------------------------------------------------------------- veri düğümleri (sıra önemli)
DATA_NODES = [
    ("sonuc", "Koşu sonuçları (artifacts*/)", r"^artifacts|[\\/]artifacts|artifacts_|ablation_results|unet_results"),
    ("fci_l1c", "MTG FCI L1c (ham)", r"F:\\+mtg$|F:/mtg$|[\\/]mtg[\\/]?$"),
    ("kesit", "Türkiye kesitleri NetCDF", r"turkiye_kesit|raw_mtg|\.nc$"),
    ("radar_hiz", "Hizalanmış radar", r"radar_aligned"),
    ("radar_ham", "Radar PNG + radar.csv", r"raw_radar|radar\.csv|[\\/]radar[\\/]?$"),
    ("awos", "AWOS gözlemleri", r"raw_awos|awos_aligned"),
    ("gozlem", "SYNOP / METAR / istasyon listesi", r"raw_observation|synop[_.]|metar\.csv|station_list"),
    ("dem", "DEM (Copernicus GLO-90)", r"(^|[\\/_])dem([_./\\]|$)"),
    ("h5_ana", "Ana H5 (mtg_patch_dataset.h5)", r"mtg_patch_dataset"),
    ("h5_kompakt", "Kompakt H5 (mtg_train_v2.h5)", r"mtg_train_v2|mtg_train\b"),
    ("h5_ornek", "Paylaşılabilir örneklem H5", r"mtg_sample"),
    ("h5meta", "H5 meta (h5meta_train_v2.npz)", r"h5meta"),
    ("bolme", "Bölme dosyaları (splits/)", r"splits[\\/]|split_v\d|cv5"),
    ("clm_ist", "CLM istasyon çıkarımları", r"clm_station"),
    ("clm_ham", "FCI CLM netCDF (ham)", r"mtg_fci_clm"),
    ("nwc_ist", "NWC istasyon çıkarımları", r"nwcsaf_station"),
    ("nwc_ham", "NWC SAF CT PNG (ham)", r"nwcsaf"),
    ("cache", "Yükleyici önbelleği (.npy)", r"mtg_cache|/content/cache"),
]
# Kodda dize sabiti olarak geçmeyen (argümandan gelen) bilinen veri bağlantıları — elle eklenir
EXTRA_DATA = {
    "make_label_policy": [("bolme", "okur+yazar"), ("h5_kompakt", "okur")],
    "run_all": [("sonuc", "yazar"), ("h5_kompakt", "okur")],
    "train": [("h5_kompakt", "okur"), ("bolme", "okur")],
    "train_unet": [("h5_kompakt", "okur"), ("bolme", "okur")],
    "h5_dataset": [("h5_kompakt", "okur")],
    "fast_loader": [("h5_kompakt", "okur"), ("cache", "okur+yazar")],
    "analyze_preds": [("sonuc", "okur+yazar")],
}
WRITE_FUNCS = {"savez", "savez_compressed", "save", "to_csv", "savefig", "dump", "to_netcdf", "write", "open_memmap",
               "copyfile", "imwrite"}
WRITE_FLAG = re.compile(r"out|result|dst|save|cache", re.I)
WRITE_NAME = re.compile(r"OUT|DST|DEST|SAVE", re.I)


def classify_data(s):
    if len(s) > 200 or "\n" in s or s.count(" ") >= 3:
        return None
    if re.fullmatch(r"artifacts\w*", s):
        return "sonuc"
    if not re.search(r"[\\/]|\.(h5|npz|npy|nc|csv|json|png|tif)\b", s):
        return None
    for nid, _, pat in DATA_NODES:
        if re.search(pat, s):
            return nid
    return None


def first_line(doc, n=220):
    if not doc:
        return ""
    lines = [l.strip() for l in doc.strip().splitlines()]
    out = []
    for l in lines:
        if not l or set(l) <= set("=-~"):
            if out:
                break
            continue
        out.append(l)
        if len(" ".join(out)) > n:
            break
    t = " ".join(out)
    return t if len(t) <= n else t[:n - 1] + "…"


def unparse(node, n=160):
    try:
        t = ast.unparse(node)
    except Exception:
        t = ""
    t = re.sub(r"\s+", " ", t)
    return t if len(t) <= n else t[:n - 1] + "…"


# ----------------------------------------------------------------------------- Python modülü
def parse_py(path, mid, modnames):
    src = open(path, encoding="utf-8", errors="replace").read()
    rec = base_record(mid, os.path.relpath(path, ROOT).replace("\\", "/"), len(src.splitlines()))
    if not src.strip():
        return rec, []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        rec["warn"] = f"ayrıştırılamadı: {e}"
        return rec, []
    rec["doc"] = first_line(ast.get_docstring(tree))
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n

    alias_mod, alias_sym, proj, ext = {}, {}, set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                parts = a.name.split(".")
                top = parts[1] if parts[0] == "src" and len(parts) > 1 else parts[0]
                if top in modnames:
                    proj.add(top); alias_mod[a.asname or top] = top
                else:
                    ext.add(parts[0])
        elif isinstance(n, ast.ImportFrom):
            parts = (n.module or "").split(".")
            top = parts[1] if parts[0] == "src" and len(parts) > 1 else parts[0]
            if n.level and not n.module:
                for a in n.names:
                    if a.name in modnames:
                        proj.add(a.name); alias_mod[a.asname or a.name] = a.name
            elif top in modnames:
                proj.add(top)
                for a in n.names:
                    alias_sym[a.asname or a.name] = (top, a.name)
            elif parts[0]:
                ext.add(parts[0])
    proj.discard(mid)
    rec["imports"] = sorted(proj)
    rec["ext"] = sorted(e for e in ext if e not in ("__future__",))

    local_funcs, local_classes = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local_funcs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            local_classes[node.name] = {f.name for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def sym(node, qual):
        return {"name": node.name, "qual": qual, "sig": f"{node.name}({unparse(node.args)})",
                "doc": first_line(ast.get_docstring(node), 200), "line": node.lineno,
                "lines": (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rec["functions"].append(sym(node, f"{mid}.{node.name}"))
        elif isinstance(node, ast.ClassDef):
            c = {"name": node.name, "qual": f"{mid}.{node.name}", "bases": [unparse(b, 60) for b in node.bases],
                 "doc": first_line(ast.get_docstring(node), 200), "line": node.lineno,
                 "lines": (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1, "methods": []}
            for f in node.body:
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    c["methods"].append(sym(f, f"{mid}.{node.name}.{f.name}"))
            rec["classes"].append(c)

    def resolve(call, cls):
        f = call.func
        if isinstance(f, ast.Name):
            n = f.id
            if n in local_funcs or n in local_classes:
                return f"{mid}.{n}"
            if n in alias_sym:
                m, s = alias_sym[n]
                return f"{m}.{s}"
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            v = f.value.id
            if v in alias_mod:
                return f"{alias_mod[v]}.{f.attr}"
            if v in ("self", "cls") and cls and f.attr in local_classes.get(cls, ()):
                return f"{mid}.{cls}.{f.attr}"
            if v in local_classes and f.attr in local_classes[v]:
                return f"{mid}.{v}.{f.attr}"
        return None

    calls = []

    def collect(nodes, src_qual, cls=None):
        cnt = {}
        for top in nodes:
            for n in ast.walk(top):
                if isinstance(n, ast.Call):
                    t = resolve(n, cls)
                    if t and t != src_qual:
                        cnt[t] = cnt.get(t, 0) + 1
        for t, k in cnt.items():
            calls.append([src_qual, t, k])

    module_level = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            collect([node], f"{mid}.{node.name}")
        elif isinstance(node, ast.ClassDef):
            for f in node.body:
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    collect([f], f"{mid}.{node.name}.{f.name}", node.name)
        else:
            module_level.append(node)
    collect(module_level, f"{mid}.<modül>")

    # argparse, alt süreç, veri, __main__
    runs, data = set(), {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument":
            flags = [a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            kw = {k.arg: k.value for k in n.keywords if k.arg}
            h = kw.get("help")
            rec["args"].append({"flag": ", ".join(flags),
                                "help": first_line(h.value, 160) if isinstance(h, ast.Constant) and isinstance(h.value, str) else "",
                                "default": unparse(kw["default"], 60) if "default" in kw else ""})
        if isinstance(n, ast.If) and "__name__" in unparse(n.test) and "__main__" in unparse(n.test):
            rec["entry"] = True
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and not skip_const(n, parents, mid):
            s = n.value
            if s.endswith(".py"):
                b = os.path.basename(s.replace("\\", "/"))[:-3]
                if b in modnames and b != mid:
                    runs.add(b)
            nid = classify_data(s)
            if nid:
                mode = data_mode(n, parents)
                prev = data.get(nid)
                data[nid] = "okur+yazar" if prev and prev != mode else mode
    for nid, mode in EXTRA_DATA.get(mid, []):
        prev = data.get(nid)
        data[nid] = mode if not prev or prev == mode else "okur+yazar"
    rec["runs"] = sorted(runs)
    rec["data"] = [{"id": k, "mode": v} for k, v in sorted(data.items())]
    return rec, calls


def skip_const(node, parents, mid):
    """Docstring, help/description metni ve bu betiğin kendi desenleri veri yolu sayılmaz."""
    if mid == "make_code_map":
        return True
    p = parents.get(node)
    if isinstance(p, ast.Expr):
        return True
    if isinstance(p, ast.keyword) and p.arg in ("help", "description", "epilog", "title"):
        return True
    if isinstance(p, ast.JoinedStr):
        return False
    return False


def data_mode(node, parents):
    p = parents.get(node)
    depth = 0
    while p is not None and depth < 6:
        if isinstance(p, ast.Call):
            fname = p.func.attr if isinstance(p.func, ast.Attribute) else (p.func.id if isinstance(p.func, ast.Name) else "")
            if fname == "add_argument":
                flags = " ".join(a.value for a in p.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
                return "yazar" if WRITE_FLAG.search(flags) else "okur"
            if fname in WRITE_FUNCS:
                return "yazar"
            if fname in ("open", "File"):
                modes = [a.value for a in p.args[1:2] if isinstance(a, ast.Constant)] + \
                        [k.value.value for k in p.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
                if any(isinstance(m, str) and m[:1] in "wa" or m == "r+" for m in modes):
                    return "yazar"
        if isinstance(p, ast.Assign):
            names = " ".join(unparse(t, 80) for t in p.targets)
            if WRITE_NAME.search(names):
                return "yazar"
            return "okur"
        p = parents.get(p)
        depth += 1
    return "okur"


def base_record(mid, path, lines):
    g, desc, legacy = MODULE_INFO.get(mid, ("diger", "", None))
    return {"id": mid, "file": path, "group": g, "desc": desc, "legacy": legacy, "lines": lines, "doc": "",
            "imports": [], "ext": [], "runs": [], "data": [], "args": [], "classes": [], "functions": [],
            "entry": False, "kind": "py", "warn": "" if mid in MODULE_INFO else
            "MODULE_INFO'da yok: src/make_code_map.py'ye grup ve açıklama ekleyin"}


# ----------------------------------------------------------------------------- defterler
def parse_ipynb(path, mid, modnames):
    rec = base_record(mid, os.path.relpath(path, ROOT).replace("\\", "/"), 0)
    rec["kind"] = "ipynb"
    try:
        nb = json.load(open(path, encoding="utf-8"))
    except Exception:
        return rec, []
    code = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            code.append("".join(c.get("source", [])))
    txt = "\n".join(code)
    rec["lines"] = len(txt.splitlines())
    imps, runs, data = set(), set(), {}
    for m in re.finditer(r"^\s*(?:from|import)\s+(?:src\.)?(\w+)", txt, re.M):
        if m.group(1) in modnames:
            imps.add(m.group(1))
    for m in re.finditer(r"([\w./\\-]+)\.py\b", txt):
        b = os.path.basename(m.group(1).replace("\\", "/"))
        if b in modnames:
            runs.add(b)
    for m in re.finditer(r"['\"]([^'\"\n]{3,200})['\"]|(\S+\.(?:h5|npz|json|csv))", txt):
        s = m.group(1) or m.group(2)
        nid = classify_data(s)
        if nid:
            data[nid] = "okur"
    rec["imports"], rec["runs"] = sorted(imps), sorted(runs)
    rec["data"] = [{"id": k, "mode": v} for k, v in sorted(data.items())]
    rec["entry"] = bool(code)
    return rec, []


# ----------------------------------------------------------------------------- ana
def build(root):
    pys = sorted(p for p in glob.glob(os.path.join(root, "src", "*.py")))
    nbs = sorted(glob.glob(os.path.join(root, "notebooks", "*.ipynb")))
    modnames = {os.path.splitext(os.path.basename(p))[0] for p in pys}
    modules, calls = [], []
    for p in pys:
        r, c = parse_py(p, os.path.splitext(os.path.basename(p))[0], modnames)
        modules.append(r); calls += c
    for p in nbs:
        r, c = parse_ipynb(p, "nb:" + os.path.splitext(os.path.basename(p))[0], modnames)
        modules.append(r)
    # yalnız var olan sembollere giden çağrılar (sınıf örnekleme dahil)
    known = set()
    for m in modules:
        known.update(f["qual"] for f in m["functions"])
        for c in m["classes"]:
            known.add(c["qual"]); known.update(f["qual"] for f in c["methods"])
    calls = [c for c in calls if c[1] in known]
    groups = [{"id": g, "name": n, "desc": d, "color": col} for g, n, d, col in GROUPS]
    if any(m["group"] == "diger" for m in modules):
        groups.append({"id": "diger", "name": "Sınıflandırılmamış", "desc": "MODULE_INFO'ya eklenmemiş yeni dosyalar", "color": "#c0603a"})
    used = {d["id"] for m in modules for d in m["data"]}
    data_nodes = [{"id": i, "name": n} for i, n, _ in DATA_NODES if i in used]
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        git = ""
    return {"generated": time.strftime("%Y-%m-%d %H:%M"), "git": git, "groups": groups, "modules": modules,
            "calls": calls, "data_nodes": data_nodes}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "mtgclm_kod_haritasi.html"))
    ap.add_argument("--json", default=None, help="İsteğe bağlı: çıkarılan grafiği ayrıca JSON olarak yaz")
    a = ap.parse_args()
    data = build(a.root)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("/*__DATA__*/null", payload)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    os.replace(tmp, a.out)
    if a.json:
        json.dump(data, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    nf = sum(len(m["functions"]) + sum(len(c["methods"]) for c in m["classes"]) for m in data["modules"])
    nc = sum(len(m["classes"]) for m in data["modules"])
    warn = [m["id"] for m in data["modules"] if m["warn"]]
    print(f"Yazıldı: {a.out}  ({len(data['modules'])} modül, {nc} sınıf, {nf} fonksiyon/metot, {len(data['calls'])} çağrı bağlantısı)")
    if warn:
        print("Uyarı — sınıflandırılmamış ya da ayrıştırılamayan:", ", ".join(warn))


TEMPLATE = r"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MTGCLM Kod Haritası</title>
<meta name="description" content="MTGCLM deposundaki bütün modüllerin, sınıfların ve fonksiyonların birbirine nasıl bağlandığını gösteren etkileşimli harita (otomatik üretilir).">
<style>
  :root{
    --bg:#0c1726; --bg2:#122236; --card:#15273d; --card2:#1b3047; --hover:#21395a;
    --ink:#eaf2fb; --ink2:#a9c0da; --mut:#7e9ab6; --line:#24405f;
    --sky:#49a7f0; --sky2:#7cc5ff; --teal:#35d6c3; --amber:#f5b733; --rose:#ff6b7f; --green:#52d08a;
    --e-import:#7f97b2; --e-call:#49a7f0; --e-run:#ff8a5c; --e-data:#52d08a;
  }
  @media (prefers-color-scheme: light){
    :root:not([data-theme="dark"]){
      --bg:#eef4fb; --bg2:#e4edf7; --card:#ffffff; --card2:#f4f8fd; --hover:#e8f0fa;
      --ink:#132233; --ink2:#38506b; --mut:#5f7793; --line:#d4e2f0;
      --sky2:#1f7fd1; --e-import:#8aa0b8; --e-call:#2a86d8; --e-run:#e0672e; --e-data:#2fa866;
    }
  }
  *{box-sizing:border-box}
  body{margin:0; background:linear-gradient(180deg,var(--bg),var(--bg2)); color:var(--ink);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; -webkit-font-smoothing:antialiased; min-height:100vh}
  a{color:var(--sky2)}
  code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.92em}
  .wrap{max-width:1600px; margin:0 auto; padding:0 16px}
  header{padding:26px 0 10px}
  .eyebrow{letter-spacing:.14em; text-transform:uppercase; font-size:12px; font-weight:700; color:var(--sky2)}
  h1{font-size:clamp(24px,4vw,38px); font-weight:800; margin:6px 0 6px; letter-spacing:-.02em}
  .lede{color:var(--ink2); max-width:900px; margin:0}
  .stats{display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 4px}
  .stat{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:8px 14px}
  .stat b{font-size:20px; letter-spacing:-.01em} .stat span{color:var(--mut); font-size:12.5px; margin-left:6px}
  .bar{position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 0}
  .bar .wrap{display:flex; flex-wrap:wrap; gap:10px; align-items:center}
  .tabs{display:flex; gap:4px; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:3px}
  .tabs button{border:0; background:transparent; color:var(--ink2); padding:7px 12px; border-radius:7px; font:inherit; font-size:14px; cursor:pointer}
  .tabs button.on{background:var(--hover); color:var(--ink); font-weight:650}
  input[type=search]{background:var(--card); color:var(--ink); border:1px solid var(--line); border-radius:9px; padding:8px 11px; font:inherit; font-size:14px; min-width:220px; flex:1; max-width:340px}
  .tg{display:flex; flex-wrap:wrap; gap:10px; font-size:13px; color:var(--ink2)}
  .tg label{display:inline-flex; gap:5px; align-items:center; cursor:pointer; user-select:none}
  .sw{display:inline-block; width:18px; height:0; border-top:2px solid; vertical-align:middle}
  .zoom{display:flex; gap:4px; margin-left:auto}
  .zoom button,.btn{background:var(--card); color:var(--ink); border:1px solid var(--line); border-radius:8px; padding:6px 10px; font:inherit; font-size:13px; cursor:pointer}
  .zoom button:hover,.btn:hover{background:var(--hover)}
  main{display:grid; grid-template-columns:minmax(0,1fr) 400px; gap:14px; padding:14px 0 30px}
  @media (max-width:1000px){ main{grid-template-columns:1fr} }
  .viewport{background:var(--card2); border:1px solid var(--line); border-radius:14px; overflow:auto; height:calc(100vh - 150px); min-height:460px; cursor:grab; position:relative}
  .viewport.drag{cursor:grabbing}
  .panel{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; height:calc(100vh - 150px); min-height:460px; overflow:auto}
  @media (max-width:1000px){ .viewport,.panel{height:auto; min-height:380px} .viewport{height:70vh} }
  .panel h2{font-size:19px; margin:0 0 4px; word-break:break-word}
  .panel h3{font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--mut); margin:16px 0 6px}
  .panel p{margin:4px 0; color:var(--ink2); font-size:14px}
  .chip{display:inline-flex; align-items:center; gap:5px; font-size:12.5px; padding:2px 9px; border-radius:999px; border:1px solid var(--line); background:var(--card2); color:var(--ink2); margin:2px 3px 2px 0; cursor:pointer; white-space:nowrap}
  .chip:hover{background:var(--hover); color:var(--ink)}
  .chip .dot{width:8px; height:8px; border-radius:50%}
  .chip.plain{cursor:default}
  .badge{display:inline-block; font-size:11px; font-weight:700; padding:1px 7px; border-radius:6px; margin-left:4px; vertical-align:middle}
  .b-old{background:rgba(245,183,51,.18); color:var(--amber)} .b-cli{background:rgba(73,167,240,.16); color:var(--sky2)}
  .b-warn{background:rgba(255,107,127,.16); color:var(--rose)}
  .fn{border:1px solid var(--line); border-radius:9px; padding:7px 9px; margin:6px 0; background:var(--card2); cursor:pointer}
  .fn:hover{background:var(--hover)}
  .fn .s{font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; color:var(--ink); word-break:break-word}
  .fn .d{font-size:12.5px; color:var(--mut); margin-top:2px}
  .fn .c{font-size:11.5px; color:var(--ink2); margin-top:3px}
  details{margin:4px 0} summary{cursor:pointer; color:var(--ink2); font-size:13.5px}
  table.args{width:100%; border-collapse:collapse; font-size:12.5px} table.args td{border-top:1px solid var(--line); padding:4px 4px; vertical-align:top}
  table.args td:first-child{font-family:ui-monospace,Menlo,Consolas,monospace; white-space:nowrap; color:var(--ink)}
  .hint{color:var(--mut); font-size:13.5px}
  svg text{fill:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .node rect.box{fill:var(--card); stroke:var(--line); stroke-width:1}
  .node:hover rect.box{fill:var(--hover)}
  .node{cursor:pointer}
  .node.sel rect.box{stroke:var(--ink); stroke-width:2}
  .node.old rect.box{fill:var(--card2); stroke-dasharray:4 3}
  .node.dim{opacity:.18}
  .node .t{font-size:13px; font-weight:650} .node .m{font-size:11px; fill:var(--mut)}
  .colh{font-size:12.5px; font-weight:750; fill:var(--ink2)} .cold{font-size:10.5px; fill:var(--mut)}
  .edge{fill:none; stroke-width:1.3; opacity:.38}
  .edge.import{stroke:var(--e-import)} .edge.call{stroke:var(--e-call)} .edge.run{stroke:var(--e-run); stroke-dasharray:6 4; stroke-width:1.8}
  .edge.data{stroke:var(--e-data); stroke-dasharray:2 3}
  .edge.on{opacity:.95; stroke-width:2.2} .edge.off{opacity:.04}
  .dnode rect{fill:var(--card); stroke:var(--e-data); stroke-width:1; rx:14}
  .dnode text{font-size:11.5px}
  .lane{font-size:12.5px; font-weight:750; fill:var(--ink2)}
  .fnode rect{fill:var(--card); stroke:var(--line)} .fnode{cursor:pointer} .fnode:hover rect{fill:var(--hover)}
  .fnode.center rect{stroke:var(--ink); stroke-width:2}
  .fnode .t{font-size:12.5px; font-weight:650; font-family:ui-monospace,Menlo,Consolas,monospace} .fnode .m{font-size:11px; fill:var(--mut)}
  .idx{width:100%; border-collapse:collapse; font-size:13px}
  .idx th{position:sticky; top:0; background:var(--card2); text-align:left; font-size:11.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); padding:8px; cursor:pointer; border-bottom:1px solid var(--line)}
  .idx td{padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top}
  .idx tr:hover td{background:var(--hover); cursor:pointer}
  .idx td.n{text-align:right; font-variant-numeric:tabular-nums}
  .idx .s{font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px}
  footer{color:var(--mut); font-size:12.5px; padding:0 0 30px}
  .legend{display:flex; flex-wrap:wrap; gap:8px; margin-top:10px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow">MTGCLM · kod haritası</div>
  <h1>Hangi dosya, hangi fonksiyon, kimi çağırıyor?</h1>
  <p class="lede">Depodaki bütün Python modülleri, sınıfları ve fonksiyonları; aralarındaki import, çağrı, betik çalıştırma ve veri dosyası
  bağlantıları. Sütunlar iş akışının sırasını izler: ham veriden H5'e, bölmeye, eğitime, analize. Bir modüle tıklayın: bağlantıları
  vurgulanır ve içeriği sağda açılır. Bir fonksiyona tıklayın: onu çağıranlar ve onun çağırdıkları görünür.
  Genel proje özeti için <a href="mtgclm_infografik.html">proje infografiği</a>.</p>
  <div class="stats" id="stats"></div>
  <div class="legend" id="legend"></div>
</header>
</div>
<div class="bar"><div class="wrap">
  <div class="tabs" role="tablist">
    <button class="on" data-view="map">Modül haritası</button>
    <button data-view="fn">Fonksiyon ağı</button>
    <button data-view="idx">Dizin</button>
  </div>
  <input type="search" id="q" placeholder="Ara: modül, sınıf, fonksiyon…" aria-label="Ara">
  <div class="tg" id="toggles">
    <label><input type="checkbox" data-e="import" checked><span class="sw" style="border-color:var(--e-import)"></span>import</label>
    <label><input type="checkbox" data-e="call"><span class="sw" style="border-color:var(--e-call)"></span>fonksiyon çağrısı</label>
    <label><input type="checkbox" data-e="run" checked><span class="sw" style="border-color:var(--e-run);border-top-style:dashed"></span>betik çalıştırır</label>
    <label><input type="checkbox" data-e="data"><span class="sw" style="border-color:var(--e-data);border-top-style:dotted"></span>veri okur/yazar</label>
    <label><input type="checkbox" id="showOld" checked>eski dosyalar</label>
  </div>
  <div class="zoom"><button id="zo" title="Uzaklaş">−</button><button id="zf" title="Sığdır">Sığdır</button><button id="zi" title="Yakınlaş">+</button></div>
</div></div>
<div class="wrap">
<main>
  <div class="viewport" id="vp"><svg id="svg" xmlns="http://www.w3.org/2000/svg"></svg><div id="idxwrap" style="display:none"></div></div>
  <aside class="panel" id="panel"></aside>
</main>
<footer id="foot"></footer>
</div>
<script>
const D = /*__DATA__*/null;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const GROUP = Object.fromEntries(D.groups.map(g => [g.id, g]));
const GORDER = Object.fromEntries(D.groups.map((g, i) => [g.id, i]));
const MOD = Object.fromEntries(D.modules.map(m => [m.id, m]));
const DNODE = Object.fromEntries(D.data_nodes.map(d => [d.id, d]));
const fileName = m => m.kind === "ipynb" ? m.file.split("/").pop() : m.id + ".py";

// ---- sembol dizini ve çağrı grafı
const SYM = {};
D.modules.forEach(m => {
  m.functions.forEach(f => SYM[f.qual] = {...f, kind: "fonksiyon", mod: m.id});
  m.classes.forEach(c => { SYM[c.qual] = {...c, kind: "sınıf", mod: m.id, sig: c.name + (c.bases.length ? "(" + c.bases.join(", ") + ")" : "")};
    c.methods.forEach(f => SYM[f.qual] = {...f, kind: "metot", mod: m.id, cls: c.name, name: c.name + "." + f.name}); });
});
const OUT = {}, IN = {};
D.calls.forEach(([s, t, n]) => {
  const mod = s.split(".")[0];
  if (!SYM[s]) SYM[s] = {qual: s, name: "<modül düzeyi>", kind: "modül kodu", mod, sig: "", doc: "Fonksiyon dışındaki kod: betik gövdesi ve __main__ bloğu", line: 1};
  (OUT[s] = OUT[s] || []).push([t, n]); (IN[t] = IN[t] || []).push([s, n]);
});
const symMod = q => SYM[q] ? SYM[q].mod : q.split(".")[0];

// ---- modül düzeyi kenarlar
const EDGES = [];
D.modules.forEach(m => {
  m.imports.forEach(t => MOD[t] && EDGES.push({type: "import", s: m.id, t}));
  m.runs.forEach(t => MOD[t] && EDGES.push({type: "run", s: m.id, t}));
  m.data.forEach(d => { if (d.mode !== "okur") EDGES.push({type: "data", s: m.id, t: "d:" + d.id, mode: "yazar"});
                        if (d.mode !== "yazar") EDGES.push({type: "data", s: "d:" + d.id, t: m.id, mode: "okur"}); });
});
const callPairs = {};
D.calls.forEach(([s, t, n]) => { const a = symMod(s), b = symMod(t); if (a !== b) callPairs[a + ">" + b] = (callPairs[a + ">" + b] || 0) + n; });
Object.entries(callPairs).forEach(([k, n]) => { const [s, t] = k.split(">"); EDGES.push({type: "call", s, t, n}); });

// ---- durum
const st = {view: "map", sel: null, fn: null, hist: [], scales: {map: 1, fn: 1}, q: "", edges: {import: true, call: false, run: true, data: false}, old: true, sort: ["mod", 1]};

// ---- başlık
(function head() {
  const nf = Object.values(SYM).filter(s => s.kind === "fonksiyon" || s.kind === "metot").length;
  const nc = Object.values(SYM).filter(s => s.kind === "sınıf").length;
  const nl = D.modules.reduce((a, m) => a + m.lines, 0);
  $("#stats").innerHTML = [[D.modules.length, "dosya"], [nc, "sınıf"], [nf, "fonksiyon / metot"], [D.calls.length, "çağrı bağlantısı"],
    [EDGES.filter(e => e.type === "import").length, "import"], [nl.toLocaleString("tr-TR"), "satır kod"]]
    .map(([n, l]) => `<div class="stat"><b>${n}</b><span>${l}</span></div>`).join("");
  $("#legend").innerHTML = D.groups.map(g => `<span class="chip plain" title="${esc(g.desc)}"><span class="dot" style="background:${g.color}"></span>${esc(g.name)}</span>`).join("")
    + `<span class="chip plain"><span class="badge b-cli">▶</span> komut satırından çalışır</span><span class="chip plain"><span class="badge b-old">eski</span> kullanımdan kalkmış</span>`;
  $("#foot").innerHTML = `Otomatik üretildi: ${esc(D.generated)}${D.git ? " · git " + esc(D.git) : ""} · <code>python src/make_code_map.py</code> ile yeniden üretin, elle düzenlemeyin.
    Çağrı bağlantıları statik çözümlemedir: doğrudan isimle ya da modül adıyla yapılan çağrıları yakalar; nesne üzerinden dolaylı çağrılar (ör. <code>model(x)</code>) görünmez.
    Veri okur/yazar yönü dize sabitlerinden sezgiseldir.`;
})();

// ---- yerleşim: modül haritası
const CW = 196, CH = 50, GX = 70, GY = 10, TOP = 64, LEFT = 20;
let LAY = null;
function layout() {
  const groups = D.groups.filter(g => D.modules.some(m => m.group === g.id && (st.old || !m.legacy)));
  const pos = {}; let maxY = 0;
  groups.forEach((g, ci) => {
    const mods = D.modules.filter(m => m.group === g.id && (st.old || !m.legacy))
      .sort((a, b) => (!!a.legacy - !!b.legacy) || (b.entry - a.entry) || a.id.localeCompare(b.id));
    mods.forEach((m, ri) => { pos[m.id] = {x: LEFT + ci * (CW + GX), y: TOP + ri * (CH + GY), w: CW, h: CH, col: ci}; });
    maxY = Math.max(maxY, TOP + mods.length * (CH + GY));
  });
  // veri şeridi
  const laneY = maxY + 70, dpos = {};
  const want = D.data_nodes.map(d => {
    const xs = D.modules.filter(m => pos[m.id] && m.data.some(x => x.id === d.id)).map(m => pos[m.id].x + CW / 2);
    return {d, x: xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0, has: xs.length > 0};
  }).filter(o => o.has).sort((a, b) => a.x - b.x);
  const DW = 176, DH = 30;
  let rowEnd = [-1e9, -1e9];
  want.forEach(o => {
    let r = rowEnd[0] <= rowEnd[1] ? 0 : 1;
    let x = Math.max(o.x - DW / 2, rowEnd[r] + 12, LEFT);
    if (x - (o.x - DW / 2) > 120) { const r2 = 1 - r; const x2 = Math.max(o.x - DW / 2, rowEnd[r2] + 12, LEFT); if (x2 < x) { r = r2; x = x2; } }
    rowEnd[r] = x + DW;
    dpos["d:" + o.d.id] = {x, y: laneY + 18 + r * (DH + 14), w: DW, h: DH};
  });
  const W = LEFT * 2 + groups.length * (CW + GX), H = laneY + 18 + 2 * (DH + 14) + 30;
  LAY = {groups, pos, dpos, W: Math.max(W, ...Object.values(dpos).map(p => p.x + p.w + LEFT)), H, laneY};
}
function edgePath(a, b, type, k) {
  if (type === "data") {
    const isDataTarget = !!b.isData;
    const p1 = isDataTarget ? {x: a.x + a.w / 2, y: a.y + a.h} : {x: a.x + a.w / 2, y: a.y};
    const p2 = isDataTarget ? {x: b.x + b.w / 2, y: b.y} : {x: b.x + b.w / 2, y: b.y + b.h};
    const my = (p1.y + p2.y) / 2;
    return `M${p1.x},${p1.y} C${p1.x},${my} ${p2.x},${my} ${p2.x},${p2.y}`;
  }
  if (Math.abs(a.x - b.x) < 1) {               // aynı sütun: sağdan kavis
    const x = a.x + a.w, y1 = a.y + a.h / 2, y2 = b.y + b.h / 2, bend = 26 + Math.min(40, Math.abs(y2 - y1) / 6);
    return `M${x},${y1} C${x + bend},${y1} ${x + bend},${y2} ${x},${y2}`;
  }
  const fwd = b.x > a.x;
  const x1 = fwd ? a.x + a.w : a.x, x2 = fwd ? b.x : b.x + b.w, y1 = a.y + a.h / 2, y2 = b.y + b.h / 2;
  const dx = Math.max(40, Math.abs(x2 - x1) * 0.45) * (fwd ? 1 : -1);
  return `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
}
function matches(m, q) {
  if (!q) return true;
  q = q.toLowerCase();
  if (m.id.toLowerCase().includes(q) || (m.desc || "").toLowerCase().includes(q)) return true;
  return m.functions.some(f => f.name.toLowerCase().includes(q)) || m.classes.some(c => c.name.toLowerCase().includes(q) || c.methods.some(f => f.name.toLowerCase().includes(q)));
}
function drawMap() {
  layout();
  const {pos, dpos, groups, W, H, laneY} = LAY;
  const all = {...pos}; Object.entries(dpos).forEach(([k, v]) => all[k] = {...v, isData: true});
  const sel = st.sel, q = st.q.trim();
  const nb = new Set();
  if (sel) EDGES.forEach(e => { if (e.s === sel) nb.add(e.t); if (e.t === sel) nb.add(e.s); });
  let s = `<defs>${["import", "call", "run", "data"].map(t => `<marker id="ar-${t}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="var(--e-${t})"/></marker>`).join("")}</defs>`;
  groups.forEach((g, ci) => {
    const x = LEFT + ci * (CW + GX);
    s += `<rect x="${x}" y="14" width="${CW}" height="4" rx="2" fill="${g.color}"/><text class="colh" x="${x}" y="36">${esc(g.name)}</text>`;
    s += `<text class="cold" x="${x}" y="52">${esc(g.desc.length > 34 ? g.desc.slice(0, 33) + "…" : g.desc)}<title>${esc(g.desc)}</title></text>`;
  });
  if (Object.keys(dpos).length && (st.edges.data || sel)) s += `<text class="lane" x="${LEFT}" y="${laneY}">Veri dosyaları</text>`;
  // kenarlar
  let es = "";
  EDGES.forEach(e => {
    const a = all[e.s], b = all[e.t];
    if (!a || !b) return;
    const touches = sel && (e.s === sel || e.t === sel);
    if (!st.edges[e.type] && !touches) return;
    if (e.type === "data" && !st.edges.data && !touches) return;
    const cls = sel ? (touches ? "on" : "off") : "";
    es += `<path class="edge ${e.type} ${cls}" d="${edgePath(a, b, e.type)}" marker-end="url(#ar-${e.type})"><title>${esc(edgeTitle(e))}</title></path>`;
  });
  s += es;
  // veri düğümleri
  if (st.edges.data || sel) Object.entries(dpos).forEach(([k, p]) => {
    const d = DNODE[k.slice(2)], dim = sel && !nb.has(k) && sel !== k;
    s += `<g class="dnode${dim ? " node dim" : ""}" data-id="${k}" transform="translate(${p.x},${p.y})"><rect width="${p.w}" height="${p.h}"/><text x="12" y="19">${esc(d.name.length > 27 ? d.name.slice(0, 26) + "…" : d.name)}</text><title>${esc(d.name)}</title></g>`;
  });
  // modül kartları
  D.modules.forEach(m => {
    const p = pos[m.id]; if (!p) return;
    const g = GROUP[m.group];
    const nfn = m.functions.length + m.classes.reduce((a, c) => a + c.methods.length, 0);
    const dim = (sel && sel !== m.id && !nb.has(m.id)) || (q && !matches(m, q));
    const meta = `${m.lines} sat.` + (m.classes.length ? ` · ${m.classes.length} sınıf` : "") + (nfn ? ` · ${nfn} fonk.` : "");
    const name = fileName(m);
    s += `<g class="node${sel === m.id ? " sel" : ""}${m.legacy ? " old" : ""}${dim ? " dim" : ""}" data-id="${m.id}" tabindex="0" transform="translate(${p.x},${p.y})">
      <rect class="box" width="${p.w}" height="${p.h}" rx="9"/><rect width="5" height="${p.h}" rx="2" fill="${g.color}"/>
      <text class="t" x="14" y="21">${esc(name.length > (m.entry || m.warn ? 20 : 23) ? name.slice(0, m.entry || m.warn ? 19 : 22) + "…" : name)}</text>
      <text class="m" x="14" y="38">${esc(meta)}</text>
      ${m.entry ? `<text x="${p.w - 12}" y="20" text-anchor="end" font-size="11" fill="var(--sky2)">▶</text>` : ""}
      ${m.legacy ? `<text x="${p.w - 12}" y="38" text-anchor="end" font-size="10.5" fill="var(--amber)">eski</text>` : ""}
      ${m.warn ? `<text x="${p.w - 26}" y="20" text-anchor="end" font-size="11" fill="var(--rose)">!</text>` : ""}
      <title>${esc(name + " — " + (m.desc || m.doc || ""))}</title></g>`;
  });
  setSvg(s, W, H);
}
function edgeTitle(e) {
  const nm = id => id.startsWith("d:") ? DNODE[id.slice(2)].name : fileName(MOD[id]);
  return {import: `${nm(e.s)} → ${nm(e.t)} (import eder)`, call: `${nm(e.s)} → ${nm(e.t)}: ${e.n} çağrı`, run: `${nm(e.s)} → ${nm(e.t)} (alt süreçle çalıştırır)`,
          data: e.mode === "yazar" ? `${nm(e.s)} yazar → ${nm(e.t)}` : `${nm(e.t)} okur ← ${nm(e.s)}`}[e.type];
}
function setSvg(s, W, H) {
  const svg = $("#svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const sc = st.scales[st.view] || 1;
  svg.setAttribute("width", Math.round(W * sc)); svg.setAttribute("height", Math.round(H * sc));
  svg.innerHTML = s; svg.dataset.w = W; svg.dataset.h = H;
}

// ---- fonksiyon ağı
function drawFn() {
  const q = st.fn;
  if (!q || !SYM[q]) { setSvg(`<text x="30" y="60" class="hint" fill="var(--mut)" font-size="15">Sağdaki listeden ya da Dizin'den bir fonksiyon seçin.</text>`, 900, 120); return; }
  const callers = (IN[q] || []).slice().sort(byMod), callees = (OUT[q] || []).slice().sort(byMod);
  const FW = 250, FH = 40, GAPY = 8, X0 = 20, XC = 380, X1 = 740;
  const rows = Math.max(callers.length, callees.length, 1);
  const H = Math.max(200, 70 + rows * (FH + GAPY) + 30), W = X1 + FW + 30;
  const cy = Math.max(70, 70 + (rows * (FH + GAPY)) / 2 - FH);
  let s = `<defs><marker id="ar-f" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--e-call)"/></marker></defs>`;
  s += `<text class="lane" x="${X0}" y="40">Çağıranlar (${callers.length})</text><text class="lane" x="${XC}" y="40">Seçili</text><text class="lane" x="${X1}" y="40">Çağırdıkları (${callees.length})</text>`;
  const node = (qq, x, y, center, n) => {
    const S = SYM[qq] || {name: qq, mod: symMod(qq), kind: "?"};
    const g = GROUP[(MOD[S.mod] || {}).group] || {color: "#888"};
    const label = (S.name || qq).length > 30 ? (S.name || qq).slice(0, 29) + "…" : (S.name || qq);
    return `<g class="fnode${center ? " center" : ""}" data-q="${esc(qq)}" transform="translate(${x},${y})"><rect width="${FW}" height="${FH}" rx="8"/><rect width="5" height="${FH}" rx="2" fill="${g.color}"/>
      <text class="t" x="13" y="17">${esc(label)}</text><text class="m" x="13" y="32">${esc(fileName(MOD[S.mod] || {id: S.mod}))} · ${esc(S.kind)}${n > 1 ? " · ×" + n : ""}</text><title>${esc((S.sig || S.name) + (S.doc ? "\n" + S.doc : ""))}</title></g>`;
  };
  callers.forEach(([c, n], i) => { const y = 60 + i * (FH + GAPY);
    s += `<path class="edge call on" d="M${X0 + FW},${y + FH / 2} C${X0 + FW + 60},${y + FH / 2} ${XC - 60},${cy + FH / 2} ${XC},${cy + FH / 2}" marker-end="url(#ar-f)"/>` + node(c, X0, y, false, n); });
  callees.forEach(([c, n], i) => { const y = 60 + i * (FH + GAPY);
    s += `<path class="edge call on" d="M${XC + FW},${cy + FH / 2} C${XC + FW + 60},${cy + FH / 2} ${X1 - 60},${y + FH / 2} ${X1},${y + FH / 2}" marker-end="url(#ar-f)"/>` + node(c, X1, y, false, n); });
  s += node(q, XC, cy, true, 0);
  if (!callers.length) s += `<text x="${X0}" y="80" fill="var(--mut)" font-size="12.5">Proje içinden doğrudan çağıran yok${SYM[q].kind !== "modül kodu" && MOD[SYM[q].mod].entry ? " (komut satırı / __main__ girişi olabilir)" : ""}</text>`;
  if (!callees.length) s += `<text x="${X1}" y="80" fill="var(--mut)" font-size="12.5">Proje içi fonksiyon çağırmıyor</text>`;
  setSvg(s, W, H);
}
const byMod = (a, b) => (GORDER[(MOD[symMod(a[0])] || {}).group] - GORDER[(MOD[symMod(b[0])] || {}).group]) || a[0].localeCompare(b[0]);

// ---- dizin
function drawIdx() {
  const q = st.q.trim().toLowerCase();
  const rows = Object.values(SYM).filter(s => s.kind !== "modül kodu").map(s => ({s, mod: s.mod, kind: s.kind, name: s.name,
    ins: (IN[s.qual] || []).length, outs: (OUT[s.qual] || []).length, line: s.line || 0, lines: s.lines || 0}))
    .filter(r => !q || r.name.toLowerCase().includes(q) || r.mod.toLowerCase().includes(q) || (r.s.doc || "").toLowerCase().includes(q));
  const [k, dir] = st.sort;
  rows.sort((a, b) => { const x = a[k], y = b[k]; return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * dir || a.name.localeCompare(b.name); });
  const th = (key, label) => `<th data-k="${key}">${label}${st.sort[0] === key ? (st.sort[1] > 0 ? " ▲" : " ▼") : ""}</th>`;
  $("#idxwrap").innerHTML = `<table class="idx"><thead><tr>${th("kind", "Tür")}${th("name", "Ad")}${th("mod", "Modül")}${th("lines", "Satır")}${th("ins", "← çağıran")}${th("outs", "→ çağırdığı")}<th>Açıklama</th></tr></thead><tbody>` +
    rows.map(r => `<tr data-q="${esc(r.s.qual)}"><td>${esc(r.kind)}</td><td class="s">${esc(r.name)}</td><td><span class="dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${GROUP[(MOD[r.mod] || {}).group]?.color || "#888"};margin-right:6px"></span>${esc(fileName(MOD[r.mod]))}:${r.line}</td><td class="n">${r.lines}</td><td class="n">${r.ins}</td><td class="n">${r.outs}</td><td>${esc(r.s.doc || "")}</td></tr>`).join("") +
    `</tbody></table>` + (rows.length ? "" : `<p class="hint" style="padding:12px">Eşleşen yok.</p>`);
}

// ---- yan panel
function chipMod(id) { const m = MOD[id]; if (!m) return ""; return `<span class="chip" data-mod="${esc(id)}"><span class="dot" style="background:${GROUP[m.group].color}"></span>${esc(fileName(m))}</span>`; }
function fnRow(f) {
  const i = (IN[f.qual] || []).length, o = (OUT[f.qual] || []).length;
  return `<div class="fn" data-q="${esc(f.qual)}"><div class="s">${esc(f.sig || f.name)}</div>${f.doc ? `<div class="d">${esc(f.doc)}</div>` : ""}<div class="c">satır ${f.line} · ${f.lines} satır · ← ${i} çağıran · → ${o} çağırdığı</div></div>`;
}
function panelModule(id) {
  if (id && id.startsWith("d:")) {
    const d = DNODE[id.slice(2)];
    const rd = D.modules.filter(m => m.data.some(x => x.id === d.id && x.mode !== "yazar")), wr = D.modules.filter(m => m.data.some(x => x.id === d.id && x.mode !== "okur"));
    $("#panel").innerHTML = `<h2>${esc(d.name)}</h2><p>Veri düğümü — kodda bu dosyaya/klasöre işaret eden dize sabitlerinden bulundu (sezgisel).</p>
      <h3>Yazanlar</h3>${wr.map(m => chipMod(m.id)).join("") || '<p class="hint">—</p>'}<h3>Okuyanlar</h3>${rd.map(m => chipMod(m.id)).join("") || '<p class="hint">—</p>'}`;
    return;
  }
  const m = MOD[id];
  if (!m) { $("#panel").innerHTML = overview(); return; }
  const g = GROUP[m.group];
  const usedBy = D.modules.filter(x => x.imports.includes(m.id)).map(x => x.id);
  const runBy = D.modules.filter(x => x.runs.includes(m.id)).map(x => x.id);
  const callOut = EDGES.filter(e => e.type === "call" && e.s === m.id), callIn = EDGES.filter(e => e.type === "call" && e.t === m.id);
  let h = `<h2>${esc(fileName(m))}${m.entry ? '<span class="badge b-cli">▶ çalıştırılabilir</span>' : ""}${m.legacy ? '<span class="badge b-old">eski</span>' : ""}</h2>
    <p><span class="chip plain"><span class="dot" style="background:${g.color}"></span>${esc(g.name)}</span> <code>${esc(m.file)}</code> · ${m.lines} satır</p>
    <p>${esc(m.desc || m.doc || "Açıklama yok.")}</p>${m.doc && m.desc ? `<p class="hint">Docstring: ${esc(m.doc)}</p>` : ""}
    ${m.legacy ? `<p><span class="badge b-old">eski</span> ${esc(m.legacy)}</p>` : ""}${m.warn ? `<p><span class="badge b-warn">!</span> ${esc(m.warn)}</p>` : ""}`;
  h += `<h3>Kullandığı proje modülleri (import)</h3>${m.imports.map(chipMod).join("") || '<p class="hint">—</p>'}`;
  h += `<h3>Onu import edenler</h3>${usedBy.map(chipMod).join("") || '<p class="hint">—</p>'}`;
  if (m.runs.length || runBy.length) h += `<h3>Betik çalıştırma</h3>${m.runs.length ? "<p>Çalıştırdıkları: " + m.runs.map(chipMod).join("") + "</p>" : ""}${runBy.length ? "<p>Onu çalıştıranlar: " + runBy.map(chipMod).join("") + "</p>" : ""}`;
  if (callOut.length || callIn.length) h += `<h3>Modüller arası çağrılar</h3>${callOut.length ? "<p>Çağırdığı modüller: " + callOut.map(e => chipMod(e.t).replace(/<\/span>$/, ` ×${e.n}</span>`)).join("") + "</p>" : ""}${callIn.length ? "<p>Onu çağıran modüller: " + callIn.map((e, i) => chipMod(e.s).replace(/<\/span>$/, ` ×${e.n}</span>`)).join("") + "</p>" : ""}`;
  if (m.data.length) h += `<h3>Veri dosyaları</h3>` + m.data.map(d => `<span class="chip" data-mod="d:${d.id}">${d.mode === "yazar" ? "✎ yazar" : d.mode === "okur" ? "⤓ okur" : "⤓✎"} · ${esc(DNODE[d.id].name)}</span>`).join("");
  if (m.args.length) h += `<h3>Komut satırı argümanları (${m.args.length})</h3><details><summary>göster</summary><table class="args">${m.args.map(a => `<tr><td>${esc(a.flag)}</td><td>${esc(a.help)}${a.default ? ` <span class="hint">[${esc(a.default)}]</span>` : ""}</td></tr>`).join("")}</table></details>`;
  if (m.classes.length) h += `<h3>Sınıflar (${m.classes.length})</h3>` + m.classes.map(c => `<div class="fn" data-q="${esc(c.qual)}"><div class="s">class ${esc(c.name)}${c.bases.length ? "(" + esc(c.bases.join(", ")) + ")" : ""}</div>${c.doc ? `<div class="d">${esc(c.doc)}</div>` : ""}<div class="c">satır ${c.line} · ${c.methods.length} metot · ← ${(IN[c.qual] || []).length} kullanan</div></div>
      <details><summary>${c.methods.length} metot</summary>${c.methods.map(fnRow).join("")}</details>`).join("");
  if (m.functions.length) h += `<h3>Fonksiyonlar (${m.functions.length})</h3>` + m.functions.map(fnRow).join("");
  const top = m.id + ".<modül>";
  if (OUT[top]) h += `<h3>Modül düzeyi kod</h3>${fnRow({...SYM[top], sig: "<modül düzeyi>", lines: "–"})}`;
  if (m.ext.length) h += `<h3>Dış kütüphaneler</h3>${m.ext.map(e => `<span class="chip plain">${esc(e)}</span>`).join("")}`;
  $("#panel").innerHTML = h;
}
function panelFn(q) {
  const S = SYM[q]; if (!S) { $("#panel").innerHTML = overview(); return; }
  const m = MOD[S.mod];
  const list = arr => arr.length ? arr.slice().sort(byMod).map(([x, n]) => { const T = SYM[x] || {name: x, mod: symMod(x)};
    return `<div class="fn" data-q="${esc(x)}"><div class="s">${esc(T.name)}${n > 1 ? " ×" + n : ""}</div><div class="c">${esc(fileName(MOD[T.mod] || {id: T.mod}))} · ${esc(T.kind || "")}</div></div>`; }).join("") : '<p class="hint">—</p>';
  $("#panel").innerHTML = `${st.hist.length ? '<button class="btn" id="back">← geri</button>' : ""}<h2 class="mono">${esc(S.name)}</h2>
    <p>${chipMod(S.mod)} <span class="hint">${esc(S.kind)} · satır ${S.line || "–"}${S.lines ? " · " + S.lines + " satır" : ""}</span></p>
    ${S.sig ? `<p class="mono" style="color:var(--ink)">${esc(S.sig)}</p>` : ""}<p>${esc(S.doc || "Docstring yok.")}</p>
    ${S.methods ? `<h3>Metotlar</h3>${S.methods.map(fnRow).join("")}` : ""}
    <h3>Çağıranlar (${(IN[q] || []).length})</h3>${list(IN[q] || [])}<h3>Çağırdıkları (${(OUT[q] || []).length})</h3>${list(OUT[q] || [])}
    <p class="hint" style="margin-top:14px">Modülün tüm içeriği için: ${chipMod(S.mod)}</p>`;
  const b = $("#back"); if (b) b.onclick = () => { st.fn = st.hist.pop(); render(); };
}
function overview() {
  const hubs = D.modules.map(m => ({m, n: D.modules.filter(x => x.imports.includes(m.id)).length})).sort((a, b) => b.n - a.n).slice(0, 6);
  const topFn = Object.keys(IN).map(q => ({q, n: IN[q].length})).sort((a, b) => b.n - a.n).slice(0, 8);
  return `<h2>Nasıl okunur?</h2><p>Sütunlar iş akışının adımları. Kartın solundaki renk grubunu, <b>▶</b> komut satırından çalıştırılabildiğini,
    kesikli çerçeve ve <b>eski</b> kullanımdan kalktığını gösterir. Okların yönü “kullanır” demektir: A → B, A'nın B'yi import ettiği, çağırdığı ya da çalıştırdığıdır.</p>
    <p>Bir karta tıklayın: yalnız o modülün bağlantıları kalır, içeriği burada açılır. Fonksiyon satırına tıklayınca <b>Fonksiyon ağı</b> açılır.
    Üstteki kutularla bağlantı türlerini açıp kapatın; boş alanı sürükleyerek kaydırın.</p>
    <h3>En çok import edilen modüller</h3>${hubs.map(h => chipMod(h.m.id).replace(/<\/span>$/, ` · ${h.n}</span>`)).join("")}
    <h3>En çok çağrılan fonksiyonlar</h3>${topFn.map(t => `<div class="fn" data-q="${esc(t.q)}"><div class="s">${esc(SYM[t.q]?.name || t.q)}</div><div class="c">${esc(fileName(MOD[symMod(t.q)]))} · ← ${t.n} çağıran</div></div>`).join("")}
    <h3>Ana giriş noktaları</h3>${["run_all", "train", "train_unet", "baselines", "analyze_preds", "aggregate_results", "make_code_map"].filter(x => MOD[x]).map(chipMod).join("")}`;
}

// ---- çizim ve olaylar
function render() {
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.view === st.view));
  const isIdx = st.view === "idx";
  $("#svg").style.display = isIdx ? "none" : ""; $("#idxwrap").style.display = isIdx ? "" : "none";
  $("#toggles").style.visibility = st.view === "map" ? "visible" : "hidden";
  $(".zoom").style.visibility = isIdx ? "hidden" : "visible";
  $("#panel").scrollTop = 0;
  if (st.view === "map") { drawMap(); panelModule(st.sel); }
  else if (st.view === "fn") { drawFn(); panelFn(st.fn); }
  else { drawIdx(); st.fn ? panelFn(st.fn) : ($("#panel").innerHTML = overview()); }
}
function openFn(q) { if (st.fn && st.fn !== q) st.hist.push(st.fn); st.fn = q; st.view = "fn"; render(); $("#vp").scrollTo(0, 0); }
function selectMod(id) { st.sel = st.sel === id ? null : id; st.view = "map"; render(); }
document.addEventListener("click", e => {
  const t = e.target.closest("[data-view],[data-mod],[data-q],.node,.dnode,th[data-k]");
  if (!t || (vpDrag.moved && t.closest("#vp"))) return;
  if (t.dataset.view) { st.view = t.dataset.view; render(); return; }
  if (t.dataset.k) { st.sort = [t.dataset.k, st.sort[0] === t.dataset.k ? -st.sort[1] : (["ins", "outs", "lines"].includes(t.dataset.k) ? -1 : 1)]; drawIdx(); return; }
  if (t.dataset.mod) { if (t.dataset.mod.startsWith("d:")) { st.sel = t.dataset.mod; st.view = "map"; render(); } else { st.sel = t.dataset.mod; st.view = "map"; render(); } return; }
  if (t.dataset.q) { openFn(t.dataset.q); return; }
  if (t.dataset.id) { selectMod(t.dataset.id); }
});
document.addEventListener("keydown", e => { if (e.key === "Enter" && e.target.classList && e.target.classList.contains("node")) selectMod(e.target.dataset.id); if (e.key === "Escape") { st.sel = null; render(); } });
$("#svg").addEventListener("click", e => { if (!e.target.closest(".node,.dnode,.fnode") && !vpDrag.moved && st.view === "map" && st.sel) { st.sel = null; render(); } });
$("#q").addEventListener("input", e => { st.q = e.target.value; if (st.view === "fn") st.view = "idx"; render(); });
document.querySelectorAll("#toggles input[data-e]").forEach(i => i.addEventListener("change", () => { st.edges[i.dataset.e] = i.checked; drawMap(); }));
$("#showOld").addEventListener("change", e => { st.old = e.target.checked; drawMap(); });
function zoom(f) { const vp = $("#vp"), svg = $("#svg"), v = st.view, old = st.scales[v] || 1;
  const cx = (vp.scrollLeft + vp.clientWidth / 2) / old, cy = (vp.scrollTop + vp.clientHeight / 2) / old;
  const sc = st.scales[v] = Math.min(2, Math.max(0.3, f === "fit" ? Math.min(1, (vp.clientWidth - 10) / (+svg.dataset.w)) : old * f));
  svg.setAttribute("width", Math.round(svg.dataset.w * sc)); svg.setAttribute("height", Math.round(svg.dataset.h * sc));
  if (f !== "fit") { vp.scrollLeft = cx * sc - vp.clientWidth / 2; vp.scrollTop = cy * sc - vp.clientHeight / 2; } }
$("#zi").onclick = () => zoom(1.2); $("#zo").onclick = () => zoom(1 / 1.2); $("#zf").onclick = () => zoom("fit");
const vpDrag = {on: false, moved: false};
$("#vp").addEventListener("pointerdown", e => { if (e.button !== 0 || st.view === "idx") return; vpDrag.on = true; vpDrag.moved = false; vpDrag.x = e.clientX; vpDrag.y = e.clientY; vpDrag.sl = $("#vp").scrollLeft; vpDrag.stp = $("#vp").scrollTop; });
window.addEventListener("pointermove", e => { if (!vpDrag.on) return; const dx = e.clientX - vpDrag.x, dy = e.clientY - vpDrag.y; if (Math.abs(dx) + Math.abs(dy) > 4) { vpDrag.moved = true; $("#vp").classList.add("drag"); } $("#vp").scrollLeft = vpDrag.sl - dx; $("#vp").scrollTop = vpDrag.stp - dy; });
window.addEventListener("pointerup", () => { vpDrag.on = false; $("#vp").classList.remove("drag"); setTimeout(() => vpDrag.moved = false, 0); });
render();
if (window.innerWidth < 1000) { st.scales.map = 0.75; render(); }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
