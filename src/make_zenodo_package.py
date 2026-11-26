"""
make_zenodo_package.py
======================
Zenodo kaydı (concept DOI 10.5281/zenodo.24112649; v1.0 = 10.5281/zenodo.24112650) için yükleme paketini tek klasörde toplar (P17):

  index/       satır dizinleri (make_sample_index.py): FCI sahne zamanı, istasyon, ızgara konumu, küme — v1.3'ten beri
               (K63). FCI kanal değerleri, radar/AWOS alanları ve etiketler dağıtılmaz:
               EUMETSAT Data Policy Art. 6.2 (saatlik olmayan Level-1 = Recommended, yeniden dağıtım yasak) ve MGM
               verisi için yalnız kullanım izni.
  sample/      (yalnız --include_sample ile; v1.0–v1.2 düzeni) açık örneklem <önek>_core.h5 + _split.npz|json,
               --include_mgm ile <önek>_mgm.h5
  splits/      splits/*.npz|json (v1, v2_verify, cv5, cv5_verify)
  results/     artifacts_v3/*.json (koşu sonuçları), artifacts_v3/analysis/*.csv|json|md
  checkpoints/ seçili ağırlıklar (varsayılan B_full_s1, D_unet_s1)
  climatology/ artifacts_v3/climatology/*.nc + *.csv (bulut oluşum sıklığı, P6)

Dosyalar kopyalanır (kaynak değişmez); MANIFEST.csv (yol, bayt, sha256, katman), metadata.json (Zenodo
"deposit metadata" taslağı) ve README.md yazılır. Paket klasörü depoya girmez (.gitignore: zenodo_package/).

Kullanım:
    python src/make_zenodo_package.py --out zenodo_package/v1.3 --version 1.3 \
        --index data/public/mtg_index.csv,data/public/mtg_sample_index.csv
    python src/make_zenodo_package.py --dry_run
Yeni sürümün DOI'sini Zenodo atar; metadata.json bu yüzden "doi" alanı taşımaz. README tüm sürümleri gösteren
concept DOI'yi verir.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil

CONCEPT_DOI = "10.5281/zenodo.24112649"  # tüm sürümler; v1.0 = 10.5281/zenodo.24112650
REPO = "https://github.com/ysnre/MTGCLM"
EXCLUDE_ANALYSIS = {"error_cases.csv"}  # örnek bazlı FCI kanal değerleri ve AWOS alanları içerir (K63)
CKPT_DEFAULT = ("best_model_B_full_s1__medium_flat.pth,best_model_D_unet_s1__mtgunet.pth,"
                "best_model_D_unet_jit_s1__mtgunet.pth")

README = """# MTGCLM — derived data, results and weights

Station-trained cloud mask for Meteosat Third Generation FCI over Türkiye. Code: {repo}
Version {version}; all versions of this record: https://doi.org/{doi}

## Contents
| Folder | Content |
|---|---|
| `index/` | Row index of the dataset: FCI scene time, station, grid position and split membership of every sample (no data values) |
| `splits/` | Split definitions (row indices of the compact dataset): single split and blocked 5-fold CV |
| `results/` | Per-run result files (`*.json`) and the analysis tables (`analysis/*.csv`) |
| `checkpoints/` | Trained weights: CNN with all inputs (`B_full_s1`), centre-supervised U-Net (`D_unet_s1`) and the U-Net trained with centre-pixel jitter used for full scenes (`D_unet_jit_s1`) |
| `climatology/` | Cloud occurrence frequency from 3-hourly single-pass maps of the jitter-trained U-Net (NetCDF) and summaries |

`MANIFEST.csv` lists every file with its size and SHA-256 checksum. On Zenodo, each folder is uploaded as a zip archive
of the same name.

## Data and how to rebuild the patches
This record contains no satellite, radar or station data values.
* **FCI:** the Level-1c data (EUMETSAT, EO:EUM:DAT:0662) are available without charge from the EUMETSAT Data Store.
  Non-hourly Level-1 data are "Recommended" data under the EUMETSAT Data Policy, whose original numerical data may not be
  redistributed; the patches are therefore not included. `index/` gives the FCI scene (`fci_scene`, UTC) and grid position
  (`station_y`, `station_x` on the 800 × 1000 grid, lon 25.01–44.99° E in 0.02° steps, lat 42.995–35.005° N in 0.01°
  steps) of every sample; `src/coknc.py` and `src/create_patch_dataset.py` in the repository rebuild the patches from the
  downloaded files.
* **Radar, AWOS fields and cloud labels (SYNOP/METAR):** data of the Turkish State Meteorological Service (TSMS/MGM),
  available from TSMS on request.
* **Derived products in this record** (splits, results, analysis tables, model weights, cloud occurrence frequency maps)
  contain no original numerical data. Contains modified EUMETSAT Meteosat data (2025–2026).
* Earlier versions 1.0–1.2 of this record are superseded; please use this version.

## Note on a result key
The double-threshold baseline is stored under the key `threshold2_ir_105+vis_06`; it actually uses the IR10.5 and IR13.3
channels (the channel order was verified after the run, and the key was kept unchanged for the record).

## Licence
Code: MIT (repository). This record: CC BY 4.0, except where the source data policies above apply.
"""


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="zenodo_package")
    ap.add_argument("--sample", default="data/public/mtg_sample", help="make_public_sample.py çıktı öneki")
    ap.add_argument("--index", default="", help="virgülle ayrılmış dizin CSV'leri (make_sample_index.py) → index/")
    ap.add_argument("--include_sample", action="store_true", help="eski düzen: FCI yamalı açık örneklem (v1.0–v1.2)")
    ap.add_argument("--include_mgm", action="store_true", help="Katman (b)'yi de ekle (MGM yeniden dağıtım izniyle)")
    ap.add_argument("--results_dir", default="artifacts_v3")
    ap.add_argument("--checkpoints", default=CKPT_DEFAULT, help="virgülle ayrılmış dosya adları (checkpoints/ altında)")
    ap.add_argument("--version", default="1.1", help="Zenodo sürüm etiketi (ör. 1.1)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    R = args.results_dir
    plan = []  # (kaynak, hedef, katman)
    pre = args.sample
    for f in [c.strip() for c in args.index.split(",") if c.strip()]:
        plan.append((f, "index/" + os.path.basename(f), "index"))
    if args.include_sample:
        for suf in ("_core.h5", "_split.npz", "_split.json"):
            plan.append((pre + suf, "sample/" + os.path.basename(pre + suf), "a"))
    if args.include_sample and args.include_mgm:
        plan.append((pre + "_mgm.h5", "sample/" + os.path.basename(pre + "_mgm.h5"), "b"))
    for f in sorted(glob.glob("splits/*.npz") + glob.glob("splits/*.json")):
        plan.append((f, "splits/" + os.path.basename(f), "a"))
    for f in sorted(glob.glob(os.path.join(R, "*.json"))):
        if os.path.basename(f) not in ("run_manifest.json",):
            plan.append((f, "results/" + os.path.basename(f), "derived"))
    for f in sorted(glob.glob(os.path.join(R, "analysis", "*.csv")) + glob.glob(os.path.join(R, "analysis", "*.json"))
                    + glob.glob(os.path.join(R, "analysis", "*.md"))):
        if os.path.basename(f) in EXCLUDE_ANALYSIS:
            continue
        plan.append((f, "results/analysis/" + os.path.basename(f), "derived"))
    for n in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        plan.append((os.path.join(R, "checkpoints", n), "checkpoints/" + n, "derived"))
    for f in sorted(glob.glob(os.path.join(R, "climatology", "*.nc")) + glob.glob(os.path.join(R, "climatology", "*.csv"))):
        plan.append((f, "climatology/" + os.path.basename(f), "derived"))

    missing = [s for s, _, _ in plan if not os.path.exists(s)]
    total = sum(os.path.getsize(s) for s, _, _ in plan if os.path.exists(s))
    print(f"{len(plan)} dosya, {total / 1e6:.1f} MB; eksik: {len(missing)}")
    for m in missing:
        print("  EKSİK:", m)
    if args.dry_run:
        for s, d, l in plan:
            print(f"  [{l:7s}] {d}")
        return
    if missing:
        raise SystemExit("Eksik dosyalar var; önce üretin (ör. make_public_sample.py) ya da --dry_run ile bakın.")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for s, d, l in plan:
        dst = os.path.join(args.out, d)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(s, dst)
        rows.append({"path": d, "bytes": os.path.getsize(dst), "sha256": sha256(dst), "layer": l})
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README.format(repo=REPO, doi=CONCEPT_DOI, version=args.version))
    rows.append({"path": "README.md", "bytes": os.path.getsize(os.path.join(args.out, "README.md")),
                 "sha256": sha256(os.path.join(args.out, "README.md")), "layer": "doc"})

    cff = {}
    try:
        import yaml  # isteğe bağlı
        with open("CITATION.cff", encoding="utf-8") as f:
            cff = yaml.safe_load(f)
    except Exception:
        pass
    creators = [{"name": f"{a.get('family-names', '')}, {a.get('given-names', '')}",
                 "affiliation": a.get("affiliation", ""), "orcid": a.get("orcid", "").rsplit("/", 1)[-1]}
                for a in cff.get("authors", [])] or [{"name": "Koçer, Sabri"}, {"name": "Er, Yasin"}]
    meta = {"metadata": {
        "title": "MTGCLM: derived data, results and trained weights for a station-trained FCI cloud mask over Türkiye",
        "upload_type": "dataset", "version": args.version, "creators": creators,
        "description": "Row index (FCI scene time, station, grid position, split) of the MTGCLM dataset, split "
                       "definitions, per-run results, analysis tables, trained weights and cloud occurrence frequency "
                       "maps. No satellite, radar or station data values are included. See README.md.",
        "access_right": "open", "license": "cc-by-4.0",
        "keywords": ["cloud mask", "Meteosat Third Generation", "FCI", "Türkiye", "SYNOP", "deep learning"],
        "related_identifiers": [{"identifier": REPO, "relation": "isSupplementTo", "scheme": "url"}],
        "notes": "FCI Level-1c data: EUMETSAT Data Store (EO:EUM:DAT:0662). Radar, AWOS and SYNOP/METAR data: "
                 "Turkish State Meteorological Service, on request. Contains modified EUMETSAT Meteosat data (2025-2026).",
    }}
    with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "layer"])
        w.writeheader(); w.writerows(rows)
    print(f"Yazıldı: {args.out}/ ({len(rows)} dosya, {sum(r['bytes'] for r in rows) / 1e6:.1f} MB) "
          f"+ MANIFEST.csv, metadata.json. FCI yamaları: {'EKLENDİ' if args.include_sample else 'yok'}.")


if __name__ == "__main__":
    main()
