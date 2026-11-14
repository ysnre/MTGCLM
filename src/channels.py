"""Kompakt H5 kanal sirasi — tek kaynak (K24, 15 Kas 2026).

patches[:, 0:16] FCI kanallari, patches[:, 16] radar; aux sirasi H5 attrs['aux_channels'] (t2m, rh2m, elev, cos_sza,
radar_cov). Sira src/verify_channel_order.py ile ham NetCDF'ten dogrulandi (artifacts_v3/channel_order.json, 400 ornek,
tum kanallarda %100 oy). create_patch_dataset.py sirayi ilk NetCDF'in h5py anahtar sirasindan aldigi icin bu liste
ELLE degistirilmez; veri yeniden uretilirse dogrulama betigi tekrar kosulur.
"""
FCI_CHANNELS = ["ir_87", "vis_09", "wv_73", "ir_105", "nir_22", "ir_123", "wv_63", "vis_08",
                "ir_97", "ir_38", "vis_06", "nir_16", "nir_13", "ir_133", "vis_05", "vis_04"]
BASE_CHANNELS = FCI_CHANNELS + ["radar"]           # patches dataset'inin 17 kanali
AUX_CHANNELS = ["t2m", "rh2m", "elev", "cos_sza", "radar_cov"]

# Dalga boyu (um) — sekillerde siralama ve gruplama icin
WAVELENGTH = {"vis_04": 0.44, "vis_05": 0.51, "vis_06": 0.64, "vis_08": 0.86, "vis_09": 0.91, "nir_13": 1.38,
              "nir_16": 1.61, "nir_22": 2.25, "ir_38": 3.80, "wv_63": 6.30, "wv_73": 7.35, "ir_87": 8.70,
              "ir_97": 9.66, "ir_105": 10.50, "ir_123": 12.30, "ir_133": 13.30}

# P3 fiziksel gruplari (eksik promptlar P3 tanimi)
GROUPS = {
    "VIS/NIR 0.4-1.6 um": ["vis_04", "vis_05", "vis_06", "vis_08", "vis_09", "nir_13", "nir_16"],
    "NIR 2.2 um": ["nir_22"],
    "IR 3.8 um": ["ir_38"],
    "Water vapour 6.3/7.3 um": ["wv_63", "wv_73"],
    "IR window 8.7/10.5/12.3 um": ["ir_87", "ir_105", "ir_123"],
    "Ozone 9.7 um": ["ir_97"],
    "CO2 13.3 um": ["ir_133"],
    "Radar": ["radar"],
    "AWOS (T2m, RH2m)": ["t2m", "rh2m"],
    "Static (elevation, cos SZA, radar coverage)": ["elev", "cos_sza", "radar_cov"],
}


def parse_base_channels(spec):
    """'all' / None -> None (tum 17 kanal); 'fci' -> 16 FCI; aksi halde virgulle ad listesi. Doner: indeks listesi."""
    if spec in (None, "", "all"):
        return None
    names = FCI_CHANNELS if spec == "fci" else [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in BASE_CHANNELS]
    if bad:
        raise SystemExit(f"Bilinmeyen kanal {bad}. Secenekler: {', '.join(BASE_CHANNELS)} veya 'fci'/'all'")
    return sorted(BASE_CHANNELS.index(n) for n in names)
