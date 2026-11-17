"""
make_zenodo_package.py
======================
Zenodo kaydı (DOI 10.5281/zenodo.24112650) için yükleme paketini tek klasörde toplar (P17):

  sample/      açık örneklem katmanı (a): <önek>_core.h5 + _split.npz + _split.json   (make_public_sample.py)
               katman (b) <önek>_mgm.h5 yalnız --include_mgm ile (MGM teyidi gelene kadar EKLEMEYİN)
  splits/      splits/*.npz|json (v1, v2_verify, cv5, cv5_verify)
  results/     artifacts_v3/*.json (koşu sonuçları), artifacts_v3/analysis/*.csv|json|md
  checkpoints/ seçili ağırlıklar (varsayılan B_full_s1, D_unet_s1)
  climatology/ artifacts_v3/climatology/*.nc + *.csv (bulut oluşum sıklığı, P6)

Dosyalar kopyalanır (kaynak değişmez); MANIFEST.csv (yol, bayt, sha256, katman), metadata.json (Zenodo
"deposit metadata" taslağı) ve README.md yazılır. Paket klasörü depoya girmez (.gitignore: zenodo_package/).

Kullanım:
    python src/make_zenodo_package.py --out zenodo_package --sample data/public/mtg_sample
    python src/make_zenodo_package.py --out zenodo_package --sample data/public/mtg_sample --dry_run
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil

DOI = "10.5281/zenodo.24112650"
REPO = "https://github.com/ysnre/MTGCLM"
CKPT_DEFAULT = "best_model_B_full_s1__medium_flat.pth,best_model_D_unet_s1__mtgunet.pth"

README = """# MTGCLM — derived data, results and weights

Station-trained cloud mask for Meteosat Third Generation FCI over Türkiye. Code: {repo}
(this record: https://doi.org/{doi}).

## Contents
| Folder | Content |
|---|---|
| `sample/` | Stratified sample of the training/evaluation patches (layer a; see below) and its split file |
| `splits/` | Split definitions (row indices of the compact dataset): single split and blocked 5-fold CV |
| `results/` | Per-run result files (`*.json`) and the analysis tables (`analysis/*.csv`) |
| `checkpoints/` | Trained weights: CNN with all inputs (`B_full_s1`) and U-Net (`D_unet_s1`) |
| `climatology/` | Cloud occurrence frequency from 3-hourly full-scene U-Net maps (NetCDF) and summaries |

`MANIFEST.csv` lists every file with its size and SHA-256 checksum.

## Data layers
* **Layer (a), open:** the 16 FCI channels (EUMETSAT, EO:EUM:DAT:0662), terrain elevation (Copernicus DEM GLO-90),
  the cosine of the solar zenith angle, and the SYNOP/METAR cloud labels. FCI data are subject to the EUMETSAT data
  policy; surface synoptic reports are exchanged under the WMO Unified Data Policy.
* **Layer (b), on request:** the radar reflectivity channel, the radar coverage field and the interpolated 2 m
  temperature and relative humidity. These derive from data of the Turkish State Meteorological Service (TSMS/MGM)
  and are {mgm_state}.

Both layers carry the same row order (`row`); `python src/make_public_sample.py --merge <prefix> --out full.h5`
rebuilds the full 17 + 5-channel format used by the code.

## Quick start
```bash
git clone {repo} && cd MTGCLM && pip install -r requirements.txt
python src/make_public_sample.py --check sample/{prefix}
python src/train.py --h5 sample/{prefix}_core.h5 --split sample/{prefix}_split.npz --model medium_flat --aux --epochs 3
```

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
    ap.add_argument("--include_mgm", action="store_true", help="Katman (b)'yi de ekle (yalnız MGM teyidinden sonra)")
    ap.add_argument("--results_dir", default="artifacts_v3")
    ap.add_argument("--checkpoints", default=CKPT_DEFAULT, help="virgülle ayrılmış dosya adları (checkpoints/ altında)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    R = args.results_dir
    plan = []  # (kaynak, hedef, katman)
    pre = args.sample
    for suf in ("_core.h5", "_split.npz", "_split.json"):
        plan.append((pre + suf, "sample/" + os.path.basename(pre + suf), "a"))
    if args.include_mgm:
        plan.append((pre + "_mgm.h5", "sample/" + os.path.basename(pre + "_mgm.h5"), "b"))
    for f in sorted(glob.glob("splits/*.npz") + glob.glob("splits/*.json")):
        plan.append((f, "splits/" + os.path.basename(f), "a"))
    for f in sorted(glob.glob(os.path.join(R, "*.json"))):
        if os.path.basename(f) not in ("run_manifest.json",):
            plan.append((f, "results/" + os.path.basename(f), "derived"))
    for f in sorted(glob.glob(os.path.join(R, "analysis", "*.csv")) + glob.glob(os.path.join(R, "analysis", "*.json"))
                    + glob.glob(os.path.join(R, "analysis", "*.md"))):
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
    prefix = os.path.basename(pre)
    mgm_state = ("included in this record with the permission of TSMS" if args.include_mgm else
                 "not included; they are available from TSMS on request (contact the corresponding author)")
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README.format(repo=REPO, doi=DOI, prefix=prefix, mgm_state=mgm_state))
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
        "upload_type": "dataset", "doi": DOI, "creators": creators,
        "description": "Stratified sample of FCI patches (open layer), split definitions, per-run results, analysis "
                       "tables, trained weights and cloud occurrence frequency maps of the MTGCLM project. See README.md.",
        "access_right": "open", "license": "cc-by-4.0",
        "keywords": ["cloud mask", "Meteosat Third Generation", "FCI", "Türkiye", "SYNOP", "deep learning"],
        "related_identifiers": [{"identifier": REPO, "relation": "isSupplementTo", "scheme": "url"}],
        "notes": "Layer (b) (radar and AWOS-derived fields, TSMS) " + ("included." if args.include_mgm else "available on request."),
    }}
    with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "layer"])
        w.writeheader(); w.writerows(rows)
    print(f"Yazıldı: {args.out}/ ({len(rows)} dosya, {sum(r['bytes'] for r in rows) / 1e6:.1f} MB) "
          f"+ MANIFEST.csv, metadata.json. Katman (b): {'EKLENDİ' if args.include_mgm else 'eklenmedi'}.")


if __name__ == "__main__":
    main()
