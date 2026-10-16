"""
make_label_policy.py
====================
METAR NSC/CAVOK "açık" etiketleri için politika uygular ve mevcut bölme dosyalarından yeni bölme
dosyaları TÜRETİR (K16, 16 Eki 2026). Kompakt H5 değişmez: politika yalnız örnek ÇIKARIR, etiket
değiştirmez; bu yüzden etiket → bölme → H5 yeniden üretimine gerek yoktur.

Politikalar:
  keep    : bugünkü etiketler (kopya; karşılaştırma için)
  verify  : METAR "açık" (gözlem dakikası != 0 ve etiket 0) yalnız AYNI istasyonun AYNI (yuvarlanmış)
            saatteki SYNOP'u N=0 ise tutulur; aksi halde (SYNOP yok / N=1..9) çıkarılır.
  ignore  : METAR "açık" örneklerin tamamı çıkarılır.
Her politikada: SYNOP etiketlerinin tamamı ve METAR bulutlu (BKN/OVC/VV) etiketleri aynen kalır;
test_partial (etiketsiz 1-4 okta) aynen kopyalanır. CLM/NWC etiket temizlemede KULLANILMAZ (döngüsellik).

METAR/SYNOP ayrımı: H5 'obs_time' dakikası 0 ise SYNOP, değilse METAR (:20/:50) — eval_nwc_and_metar.py ile aynı.

Kullanım (proje kökünden):
    python src/make_label_policy.py --h5 F:/radar/mtg_train_v2.h5 --policy verify
Çıktı (varsayılan):
    splits/split_v2_verify.npz (+ .json)       <- splits/split_v1_compact.npz
    splits/cv5_verify_fold{0..4}.npz (+ .json)  <- splits/cv5_fold{0..4}.npz
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import pandas as pd

LABELED = ("train", "val", "test_time", "test_station", "test_both")


def load_meta(h5_path):
    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][:].astype(np.int64)
        wmoid = f["wmoids"][:].astype(np.int64)
        obs_time = f["obs_time"][:].astype(np.int64)
    ot = pd.to_datetime(obs_time, unit="s")
    return pd.DataFrame({
        "label": labels, "wmoid": wmoid,
        "is_metar": (ot.minute != 0),
        "obs_hour": pd.Series(ot).dt.round("h").dt.strftime("%Y%m%d%H").values,
    })


def synop_n(meta, synop_csv):
    syn = pd.read_csv(synop_csv, usecols=["validity", "wmoid", "cloud_coverage"]).dropna()
    syn = syn[syn.cloud_coverage >= 0]
    syn["wmoid"] = syn.wmoid.astype(np.int64)
    syn["obs_hour"] = pd.to_datetime(syn.validity).dt.strftime("%Y%m%d%H")
    syn = syn.drop_duplicates(["wmoid", "obs_hour"]).set_index(["wmoid", "obs_hour"]).cloud_coverage
    return syn.reindex(pd.MultiIndex.from_arrays([meta.wmoid, meta.obs_hour])).values


def keep_mask(meta, policy):
    nsc = meta.is_metar.values & (meta.label.values == 0)
    if policy == "keep":
        return np.ones(len(meta), bool)
    if policy == "ignore":
        return ~nsc
    if policy == "verify":
        return ~nsc | (meta.synop_N.values == 0)
    raise SystemExit(f"Bilinmeyen politika: {policy}")


def stats(meta, idx):
    l = meta.label.values[idx]
    m = meta.is_metar.values[idx]
    return {"n": int(len(idx)), "clear": int((l == 0).sum()), "cloudy": int((l == 1).sum()),
            "clear_ratio": round(float((l == 0).mean()), 4) if len(idx) else None,
            "metar_share": round(float(m.mean()), 4) if len(idx) else None,
            "synop_n": int((~m).sum())}


def derive(src_npz, out_npz, meta, keep, policy, h5_path):
    sp = np.load(src_npz)
    out, info = {}, {"source_split": os.path.basename(src_npz), "policy": policy, "h5": os.path.basename(h5_path),
                     "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "before": {}, "after": {}}
    for k in sp.files:
        idx = sp[k].astype(np.int64)
        if k in LABELED:
            new = idx[keep[idx]]
            info["before"][k] = stats(meta, idx)
            info["after"][k] = stats(meta, new)
        else:                      # test_partial vb. aynen
            new = idx
            info["after"][k] = {"n": int(len(idx))}
        out[k] = new
    np.savez_compressed(out_npz, **out)
    with open(os.path.splitext(out_npz)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  {os.path.basename(src_npz):24s} -> {out_npz}")
    for k in LABELED:
        if k in info["after"]:
            b, a = info["before"][k], info["after"][k]
            print(f"     {k:13s} {b['n']:7d} -> {a['n']:7d}   açık {b['clear']:6d} -> {a['clear']:6d} "
                  f"(açık oranı {b['clear_ratio']:.3f} -> {a['clear_ratio']:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--policy", default="verify", choices=["keep", "verify", "ignore"])
    ap.add_argument("--synop_csv", default="data/raw_observation/synop_all.csv")
    ap.add_argument("--split", default="splits/split_v1_compact.npz")
    ap.add_argument("--cv_prefix", default="splits/cv5", help="'' verilirse CV katları türetilmez")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--out_split", default=None, help="Varsayılan: splits/split_v2_<politika>.npz")
    ap.add_argument("--out_cv_prefix", default=None, help="Varsayılan: splits/cv5_<politika>")
    a = ap.parse_args()

    print(f"H5 meta okunuyor: {a.h5}")
    meta = load_meta(a.h5)
    meta["synop_N"] = synop_n(meta, a.synop_csv) if a.policy == "verify" else np.nan
    keep = keep_mask(meta, a.policy)
    nsc = meta.is_metar.values & (meta.label.values == 0)
    print(f"METAR açık (NSC/CAVOK): {int(nsc.sum())} | politika '{a.policy}' ile tutulan: {int((nsc & keep).sum())}")

    out_split = a.out_split or os.path.join(os.path.dirname(a.split) or ".", f"split_v2_{a.policy}.npz")
    derive(a.split, out_split, meta, keep, a.policy, a.h5)
    if a.cv_prefix:
        out_cv = a.out_cv_prefix or f"{a.cv_prefix}_{a.policy}"
        for k in range(a.cv_folds):
            src = f"{a.cv_prefix}_fold{k}.npz"
            if os.path.exists(src):
                derive(src, f"{out_cv}_fold{k}.npz", meta, keep, a.policy, a.h5)
    print("Bitti.")


if __name__ == "__main__":
    main()
