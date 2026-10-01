"""
make_splits.py
==============
H5 veri seti için sızıntısız, tekrar üretilebilir bölme dosyası üretir (.npz + .json).

Bölme mantığı (inceleme A6):
  * ZAMAN  : tarihler --block_days günlük bloklara ayrılır (varsayılan 7 = hafta), bloklar seed ile
             karıştırılır ve geçerli örnek sayısına göre test / val / train oranlarına greedy doldurulur.
             Ardışık günlerin sinoptik korelasyonu blok içinde kalır.
  * İSTASYON: --holdout_stations adet etiketli istasyon rastgele ayrılır (uzaysal genelleme testi).

Üretilen alt kümeler (H5 indeks dizileri):
  train         = train blokları  x görülen istasyonlar
  val           = val blokları    x görülen istasyonlar   (model seçimi / early stopping)
  test_time     = test blokları   x görülen istasyonlar   (görülmemiş zaman)
  test_station  = train blokları  x ayrılan istasyonlar   (görülmemiş istasyon, görülen zaman)
  test_both     = test blokları   x ayrılan istasyonlar   (ikisi de görülmemiş)
  test_partial  = test blokları   x görülen istasyonlar, 1-4 okta (etiket -1): parçalı bulut tutarlılık testi;
                  eğitimde kullanılmaz, yalnızca model olasılığının okta ile ilişkisi raporlanır

Her alt küme için sınıf dengesi ve "hep bulutlu" baseline'ı basılır.

Kullanım:
    python src/make_splits.py --h5 F:/radar/mtg_patch_dataset.h5 --out splits/split_v1 --seed 42
"""
import argparse
import datetime as dt
import json
import os

import h5py
import numpy as np


def summarize(name: str, labels: np.ndarray, idx: np.ndarray) -> dict:
    lab = labels[idx]
    if name == "test_partial":
        print(f"   {name:13s} n={len(idx):8d}  (1-4 okta, etiketsiz tutarlılık kümesi)")
        return {"n": int(len(idx))}
    v = lab[lab >= 0]
    n = len(v)
    p = float(v.mean()) if n else float("nan")
    info = {"n": int(n), "cloudy_ratio": p,
            "majority_acc": max(p, 1 - p) if n else float("nan"),
            "majority_f1_cloudy": 2 * p / (1 + p) if n else float("nan")}
    print(f"   {name:13s} n={n:8d}  cloudy={p:.3f}  majority: Acc={info['majority_acc']:.3f} "
          f"F1(cloudy)={info['majority_f1_cloudy']:.3f}")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="F:/radar/mtg_patch_dataset.h5")
    ap.add_argument("--out", default="splits/split_v1", help="Çıktı ön eki (.npz ve .json eklenir)")
    ap.add_argument("--val", type=float, default=0.15, help="Zaman bloklarının val payı (örnek oranı)")
    ap.add_argument("--test", type=float, default=0.15, help="Zaman bloklarının test payı (örnek oranı)")
    ap.add_argument("--block_days", type=int, default=7)
    ap.add_argument("--holdout_stations", type=int, default=30, help="Ayrılacak istasyon sayısı (0 = istasyon testi yok)")
    ap.add_argument("--min_station_samples", type=int, default=500, help="Ayrılmaya aday istasyon için asgari geçerli örnek")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_dups", action="store_true", default=True, help="is_dup==1 örnekleri hariç tut (varsayılan)")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        N = len(f["labels"])
        labels = f["labels"][:].astype(np.int64)
        wmoids = f["wmoids"][:].astype(np.int64)
        ts = f["timestamps"][:]
        is_dup = f["is_dup"][:] if "is_dup" in f else np.zeros(N, dtype=np.uint8)
        okta = f["okta"][:] if "okta" in f else np.full(N, -1, dtype=np.int8)
        policy = f.attrs.get("label_policy", "yok")
    dates = np.array([(t.decode("utf-8") if isinstance(t, (bytes, np.bytes_)) else str(t))[:8] for t in ts])
    print(f"H5: {N} örnek, etiket politikası: {policy}")

    valid = labels >= 0
    if args.drop_dups:
        valid &= is_dup == 0
    print(f"Geçerli örnek: {valid.sum()} / {N}")

    # --- zaman blokları
    uniq_dates = sorted(set(dates))
    d0 = dt.datetime.strptime(uniq_dates[0], "%Y%m%d")
    date_block = {d: (dt.datetime.strptime(d, "%Y%m%d") - d0).days // args.block_days for d in uniq_dates}
    block_of = np.array([date_block[d] for d in dates])
    blocks = np.array(sorted(set(block_of)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(blocks)
    counts = {b: int((valid & (block_of == b)).sum()) for b in blocks}
    total = sum(counts.values())
    test_blocks, val_blocks, train_blocks = [], [], []
    acc_t = acc_v = 0
    for b in blocks:
        if acc_t < args.test * total:
            test_blocks.append(b); acc_t += counts[b]
        elif acc_v < args.val * total:
            val_blocks.append(b); acc_v += counts[b]
        else:
            train_blocks.append(b)
    split_of_block = {}
    for b in train_blocks: split_of_block[b] = "train"
    for b in val_blocks: split_of_block[b] = "val"
    for b in test_blocks: split_of_block[b] = "test"
    time_split = np.array([split_of_block[b] for b in block_of])
    print(f"Zaman blokları ({args.block_days} gün): train {len(train_blocks)}, val {len(val_blocks)}, test {len(test_blocks)}")

    # --- istasyon ayrımı
    st_ids, st_counts = np.unique(wmoids[valid], return_counts=True)
    candidates = st_ids[st_counts >= args.min_station_samples]
    rng2 = np.random.default_rng(args.seed + 1)
    holdout = np.sort(rng2.choice(candidates, size=min(args.holdout_stations, len(candidates)), replace=False)) \
        if args.holdout_stations > 0 else np.array([], dtype=np.int64)
    is_holdout = np.isin(wmoids, holdout)
    print(f"İstasyon: {len(st_ids)} etiketli, {len(holdout)} ayrıldı: {holdout.tolist()}")

    subsets = {
        "train":        valid & (time_split == "train") & ~is_holdout,
        "val":          valid & (time_split == "val") & ~is_holdout,
        "test_time":    valid & (time_split == "test") & ~is_holdout,
        "test_station": valid & (time_split == "train") & is_holdout,
        "test_both":    valid & (time_split == "test") & is_holdout,
        "test_partial": (labels < 0) & (okta >= 1) & (okta <= 4) & (is_dup == 0) & (time_split == "test") & ~is_holdout,
    }
    print("Alt kümeler:")
    stats = {name: summarize(name, labels, np.flatnonzero(m)) for name, m in subsets.items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out + ".npz", **{k: np.flatnonzero(m).astype(np.int32) for k, m in subsets.items()})
    meta = {
        "h5": args.h5, "n_total": int(N), "seed": args.seed, "block_days": args.block_days,
        "val": args.val, "test": args.test, "label_policy": str(policy), "drop_dups": bool(args.drop_dups),
        "holdout_stations": holdout.tolist(),
        "dates": {s: sorted(d for d in uniq_dates if split_of_block[date_block[d]] == s) for s in ("train", "val", "test")},
        "stats": stats,
    }
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nYazıldı: {args.out}.npz ve {args.out}.json")


if __name__ == "__main__":
    main()
