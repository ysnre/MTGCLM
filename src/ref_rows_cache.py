"""Kompakt H5 satırları için FCI CLM ve NWC SAF CT ikili tahminleri (merkez, r30) — önbellek (27 Kas, B1–B3).

analyze_preds.reference_preds ile BİREBİR aynı kural (r30 eşikleri yalnız verify train SYNOP+METAR etiketlerinde seçilir);
farkı: verify bölmesindeki test kümeleriyle sınırlı kalmayıp v1_compact bölmesinin bütün satırlarına (verify politikasının
çıkardığı doğrulanmamış METAR açıkları dahil) uygulanır. Çıktı: data/processed/ref_rows_all.npz (CLM_merkez, CLM_r30,
NWC_merkez, NWC_r30; NaN = kapsam yok) + _info.json. Diğer betikler get_refs() ile okur.
    python src/ref_rows_cache.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_preds import SUBSETS, load_meta, reference_preds  # noqa: E402

CACHE = "data/processed/ref_rows_all.npz"


class _Split(dict):
    @property
    def files(self):
        return list(self.keys())


def build(meta, verify="splits/split_v2_verify.npz", v1="splits/split_v1_compact.npz", clm_dir="data/processed/clm_station",
          nwc_dir="data/processed/nwcsaf_station", out=CACHE):
    sv, s1 = np.load(verify), np.load(v1)
    allrows = np.unique(np.concatenate([s1[k] for k in s1.files if k != "test_partial"]))
    sp = _Split(train=sv["train"], **{SUBSETS[0]: allrows})
    refs, info = reference_preds(meta, sp, clm_dir, nwc_dir)
    np.savez_compressed(out, rows=allrows, **{k: v for k, v in refs.items()})
    info = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in info.items()}
    info["n_rows"] = int(len(allrows))
    json.dump(info, open(out.replace(".npz", "_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return refs


def get_refs(meta, out=CACHE):
    if os.path.exists(out):
        z = np.load(out)
        return {k: z[k] for k in ("CLM_merkez", "CLM_r30", "NWC_merkez", "NWC_r30")}
    return build(meta, out=out)


if __name__ == "__main__":
    m = load_meta("data/processed/h5meta_train_v2.npz", None, "data/raw_observation/station_list.csv")
    r = build(m)
    print({k: int(np.isfinite(v).sum()) for k, v in r.items()})
