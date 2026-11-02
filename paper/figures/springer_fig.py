"""MTGCLM — Springer (TAAC / MAP) sekil yardimcilari.

Kural dokumani: sekil_stil_kurali.md. Her sekil betigi su iskeleti izler:

    from springer_fig import new_figure, panel_label, COLORS, LABELS, save_figure
    fig, axes = new_figure("double", height_mm=80, ncols=2)
    ...
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    save_figure(fig, "Fig04_main_results")     # PDF + TIFF + PNG onizleme + kontrol raporu

Bu modul matplotlib disinda bagimlilik istemez. Harita sekilleri icin cartopy ayrica gerekir.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

STYLE_FILE = Path(__file__).with_name("springer_taac_map.mplstyle")

MM = 1 / 25.4  # inch per mm
WIDTHS_MM = {"sub": 39.0, "single": 84.0, "onehalf": 129.0, "double": 174.0}
MAX_HEIGHT_MM = 234.0
MIN_FONT_PT = 8.0
MIN_LINE_PT = 0.3

# ---------------------------------------------------------------------------
# Varlik -> renk eslemesi. Butun sekillerde AYNI varlik AYNI renk (renk threading).
# Okabe-Ito paleti; kirmizi/yesil karsitligi yok.
# ---------------------------------------------------------------------------
COLORS = {
    # bizim modeller
    "B_full": "#0072B2",        # odak model (tum girdiler) - koyu mavi
    "A_sat": "#56B4E9",         # uydu + radar - acik mavi
    "C_noawos": "#009E73",      # AWOS'suz aux - yesil-turkuaz
    "C_t2m": "#CC79A7",         # AWOS'suz + t2m - pembe-mor
    "A_norad": "#999999",       # radarsiz (eklenecek) - gri
    "D_unet": "#000000",        # U-Net - siyah
    "RESNET": "#6A3D9A",        # ResNet-18 ImageNet stem - mor
    "RESNET_CIFAR": "#B294C7",  # ResNet-18 CIFAR stem - acik mor
    # klasik
    "hgb_aux": "#E69F00",       # gradyan artirma - turuncu
    "rf_aux": "#F0C35E",
    "logreg_aux": "#F5DDA0",
    "threshold": "#BBBBBB",
    # operasyonel referanslar
    "CLM": "#D55E00",           # EUMETSAT FCI CLM - vermilyon
    "NWC": "#8C510A",           # NWC SAF Cloud Type - kahve
    # siniflar ve isaretler
    "cloudy": "#F2F2F2",
    "clear_land": "#D9C7A1",
    "clear_sea": "#24476B",
    "neutral": "#7F7F7F",
}

# Sekillerde gorunen okunur adlar (kod adi sekle yazilmaz).
LABELS = {
    "B_full": "CNN, all inputs",
    "A_sat": "CNN, satellite + radar",
    "C_noawos": "CNN, no AWOS fields",
    "C_t2m": "CNN, no AWOS + T2m",
    "A_norad": "CNN, satellite only",
    "D_unet": "U-Net",
    "RESNET": "ResNet-18 (ImageNet stem)",
    "RESNET_CIFAR": "ResNet-18 (CIFAR stem)",
    "hgb_aux": "Gradient boosting",
    "rf_aux": "Random forest",
    "logreg_aux": "Logistic regression",
    "CLM": "FCI CLM",
    "CLM_merkez": "FCI CLM (centre pixel)",
    "CLM_r30": "FCI CLM (30 km area)",
    "NWC": "NWC SAF CT",
    "NWC_merkez": "NWC SAF CT (centre pixel)",
    "NWC_r30": "NWC SAF CT (30 km area)",
    "test_time": "Unseen weeks",
    "test_station": "Unseen stations",
    "test_both": "Unseen weeks and stations",
    "val": "Validation",
}

# Haritalar icin sabitler (veri grid'i duzenli enlem-boylam).
MAP_EXTENT = (25.0, 45.0, 35.0, 43.0)  # lon_min, lon_max, lat_min, lat_max
MAP_GRID_STEP_DEG = 5
CMAP_PROB = "cividis"        # bulut olasiligi 0-1
CMAP_DIFF = "RdBu_r"         # isaretli fark, 0'da merkezli (TwoSlopeNorm)
CMAP_TERRAIN = "gist_earth"  # DEM altligi (yalniz F1)


def use_style() -> None:
    """Springer stilini yukler. Her sekil betiginin basinda bir kez cagirin."""
    plt.style.use(str(STYLE_FILE))


def new_figure(width: str = "single", height_mm: float = 60.0, nrows: int = 1, ncols: int = 1, **kw):
    """Son baski boyutunda sekil olusturur. width: sub | single | onehalf | double."""
    if width not in WIDTHS_MM:
        raise ValueError(f"width {width!r}; izinli: {list(WIDTHS_MM)}")
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"yukseklik {height_mm} mm > {MAX_HEIGHT_MM} mm (Springer siniri)")
    use_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTHS_MM[width] * MM, height_mm * MM), **kw)
    fig._springer_width = width  # kontrol icin
    return fig, axes


def panel_label(ax, letter: str, dx_pt: float = -4.0, dy_pt: float = 2.0) -> None:
    """Springer panel harfi: kucuk harf, kalin, eksen kutusunun sol ustunde, disarida."""
    trans = mpl.transforms.offset_copy(ax.transAxes, fig=ax.figure, x=dx_pt, y=dy_pt, units="points")
    ax.text(0.0, 1.0, letter.lower(), transform=trans, fontsize=10, fontweight="bold",
            ha="right", va="bottom")


def check_figure(fig) -> list[str]:
    """Son kontrol: genislik, yukseklik, minimum yazi boyutu, cizgi kalinligi, yazi cakismasi."""
    problems = []
    w_mm, h_mm = (v / MM for v in fig.get_size_inches())
    if not any(abs(w_mm - v) < 0.5 for v in WIDTHS_MM.values()):
        problems.append(f"genislik {w_mm:.1f} mm standart degil ({WIDTHS_MM})")
    if h_mm > MAX_HEIGHT_MM:
        problems.append(f"yukseklik {h_mm:.1f} mm > {MAX_HEIGHT_MM}")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    for t in texts:
        if t.get_fontsize() < MIN_FONT_PT - 1e-6:
            problems.append(f"yazi {t.get_fontsize():.1f} pt < {MIN_FONT_PT}: {t.get_text()[:30]!r}")
    for ln in fig.findobj(mpl.lines.Line2D):
        if ln.get_visible() and ln.get_linestyle() not in ("None", "none", "") and 0 < ln.get_linewidth() < MIN_LINE_PT:
            problems.append(f"cizgi {ln.get_linewidth():.2f} pt < {MIN_LINE_PT}")
    if any(t.get_text().strip() for ax in fig.axes for t in [ax.title, ax._left_title, ax._right_title]):
        problems.append("eksen basligi var: Springer'da sekil basligi altyaziya (caption) yazilir")
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    ticks = {t for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels()}
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            if a in ticks and b in ticks:
                continue
            if ba.overlaps(bb):
                problems.append(f"cakisma: {a.get_text()[:20]!r} / {b.get_text()[:20]!r}")
    return problems


def save_figure(fig, name: str, outdir: str = "figures", kind: str = "combination") -> list[str]:
    """PDF (vektor, gomulu yazi tipi) + TIFF (kind'e gore dpi) + PNG onizleme yazar.

    kind: 'line' (1200 dpi), 'combination' (600 dpi; grafik + yazi, harita + renk cubugu),
          'halftone' (300 dpi; yalniz goruntu).
    Donus: kontrol sorunlari listesi (bos = temiz)."""
    dpi = {"line": 1200, "combination": 600, "halftone": 300}[kind]
    os.makedirs(outdir, exist_ok=True)
    problems = check_figure(fig)
    base = os.path.join(outdir, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".tif", dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base + "_preview.png", dpi=200)
    with open(base + "_check.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(problems) if problems else "OK")
    return problems
