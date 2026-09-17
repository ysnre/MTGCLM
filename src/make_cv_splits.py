"""
make_cv_splits.py
=================
Bloklu K-katlı çapraz doğrulama (blocked K-fold cross-validation) bölmeleri üretir.

Neden bloklu: ardışık uydu görüntüleri güçlü zamansal korelasyon taşır. Rastgele k-fold,
aynı sinoptik durumun hem eğitimde hem testte olmasına yol açar (sızıntı). Bu yüzden tarihler
--block_days günlük bloklara ayrılır, bloklar seed ile karıştırılıp K kata dağıtılır; bir kat
test, kalanlardan bir blok grubu val, geri kalanı train olur.

İsteğe bağlı --holdout_stations: her katta ayrıca N istasyon ayrılır (istasyon-dışı test).
Varsayılan 0 = istasyon ayrımı yok (tüm istasyonlar her katta kullanılır); makalede zamansal CV
ile uzaysal genellemeyi ayrı ayrı raporlamak daha temiz olur.

Çıktı: <out>_fold{k}.npz / .json  (train / val / test_time / test_partial [+ test_station, test_both])
ve <out>_cv.json (kat özetleri).

Kullanım (kompakt H5 üzerinde, yeniden dışa aktarma gerekmez):
    python src/make_cv_splits.py --h5 F:/radar/mtg_train_v2.h5 --out splits/cv5 --folds 5 --seed 42
Sonra her kat için:
    python src/train.py --h5 F:/radar/mtg_train_v2.h5 --split splits/cv5_fold0.npz --model mtg_flat --aux ^
        --seed 1 --results artifacts/cv5_f0_full_s1.json
"""
import argparse
import datetime as dt
import json
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="F:/radar/mtg_train_v2.h5")
    ap.add_argument("--out", default="splits/cv5")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block_days", type=int, default=7)
    ap.add_argument("--val_frac", type=float, default=0.15, help="Kat içi eğitim bloklarından val payı")
    ap.add_argument("--holdout_stations", type=int, default=0, help="Her katta ayrılacak istasyon sayısı (0 = yok)")
    ap.add_argument("--min_station_samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        N = len(f["labels"])
        labels = f["labels"][:].astype(np.int64)
        wmoids = f["wmoids"][:].astype(np.int64)
        okta = f["okta"][:] if "okta" in f else np.full(N, -1, dtype=np.int8)
        is_dup = f["is_dup"][:] if "is_dup" in f else np.zeros(N, dtype=np.uint8)
        ts = f["timestamps"][:]
        policy = str(f.attrs.get("label_policy", "yok"))
    dates = np.array([(t.decode("utf-8") if isinstance(t, (bytes, np.bytes_)) else str(t))[:8] for t in ts])
    valid = (labels >= 0) & (is_dup == 0)
    partial_pool = (labels < 0) & (okta >= 1) & (okta <= 4) & (is_dup == 0)
    print(f"H5: {args.h5}  N={N}, geçerli={valid.sum()}, partial havuzu={partial_pool.sum()}")
    print(f"Etiket politikası: {policy}")

    uniq_dates = sorted(set(dates))
    d0 = dt.datetime.strptime(uniq_dates[0], "%Y%m%d")
    date_block = {d: (dt.datetime.strptime(d, "%Y%m%d") - d0).days // args.block_days for d in uniq_dates}
    block_of = np.array([date_block[d] for d in dates])
    blocks = np.array(sorted(set(block_of)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(blocks)
    counts = np.array([int((valid & (block_of == b)).sum()) for b in blocks])

    # blokları K kata dengeli dağıt (greedy: en büyük bloğu en boş kata)
    order = np.argsort(-counts)
    fold_of_block, loads = {}, np.zeros(args.folds, dtype=np.int64)
    for i in order:
        k = int(np.argmin(loads))
        fold_of_block[int(blocks[i])] = k
        loads[k] += counts[i]
    print(f"{len(blocks)} blok ({args.block_days} gün) -> {args.folds} kat; kat başına örnek: {loads.tolist()}")

    summary = {"h5": os.path.abspath(args.h5), "folds": args.folds, "seed": args.seed,
               "block_days": args.block_days, "label_policy": policy, "fold_stats": {}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for k in range(args.folds):
        test_mask_block = np.array([fold_of_block[int(b)] == k for b in block_of])
        # kalan bloklardan val ayır (aynı katın seed'i ile, tekrar üretilebilir)
        rest = [b for b in blocks if fold_of_block[int(b)] != k]
        rng_k = np.random.default_rng(args.seed * 1000 + k)
        rest = list(rest); rng_k.shuffle(rest)
        rest_total = int((valid & ~test_mask_block).sum())
        val_blocks, acc = set(), 0
        for b in rest:
            if acc >= args.val_frac * rest_total:
                break
            val_blocks.add(int(b)); acc += int((valid & (block_of == b)).sum())
        val_mask_block = np.isin(block_of, list(val_blocks))
        train_mask_block = ~test_mask_block & ~val_mask_block

        if args.holdout_stations > 0:
            st, cnt = np.unique(wmoids[valid & train_mask_block], return_counts=True)
            cand = st[cnt >= args.min_station_samples]
            hold = np.sort(np.random.default_rng(args.seed * 2000 + k).choice(
                cand, size=min(args.holdout_stations, len(cand)), replace=False))
        else:
            hold = np.array([], dtype=np.int64)
        is_hold = np.isin(wmoids, hold)

        subsets = {
            "train":        valid & train_mask_block & ~is_hold,
            "val":          valid & val_mask_block & ~is_hold,
            "test_time":    valid & test_mask_block & ~is_hold,
            "test_partial": partial_pool & test_mask_block & ~is_hold,
        }
        if args.holdout_stations > 0:
            subsets["test_station"] = valid & train_mask_block & is_hold
            subsets["test_both"] = valid & test_mask_block & is_hold

        stats = {}
        parts = []
        for name, m in subsets.items():
            idx = np.flatnonzero(m)
            if name == "test_partial":
                stats[name] = {"n": int(len(idx))}
            else:
                v = labels[idx]
                p = float(v.mean()) if len(v) else float("nan")
                stats[name] = {"n": int(len(idx)), "cloudy_ratio": p, "majority_acc": max(p, 1 - p) if len(v) else float("nan")}
            parts.append(f"{name}={len(idx)}")
        path = f"{args.out}_fold{k}"
        np.savez_compressed(path + ".npz", **{n: np.flatnonzero(m).astype(np.int32) for n, m in subsets.items()})
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump({"fold": k, "h5": os.path.abspath(args.h5), "seed": args.seed,
                       "holdout_stations": hold.tolist(), "stats": stats,
                       "test_dates": sorted({d for d in uniq_dates if fold_of_block[date_block[d]] == k})}, f,
                      indent=2, ensure_ascii=False)
        summary["fold_stats"][f"fold{k}"] = stats
        print(f"  fold{k}: " + ", ".join(parts) +
              f"  (test cloudy={stats['test_time']['cloudy_ratio']:.3f})")

    with open(args.out + "_cv.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nYazıldı: {args.out}_fold0..{args.folds-1}.npz ve {args.out}_cv.json")


if __name__ == "__main__":
    main()
