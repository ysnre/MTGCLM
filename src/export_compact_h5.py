"""
export_compact_h5.py
====================
Büyük H5'ten yalnızca bölme dosyasındaki örnekleri (train/val/test_* + test_partial) alır,
GLOBAL permütasyonla karıştırır ve Colab için küçük, float16, ön-karıştırılmış bir H5 yazar.

Çıktı (mtg_train_v2.h5):
    patches  [M,17,33,33] float16   (normalize edilmiş, kaynaktakiyle aynı)
    aux      [M,5,33,33]  float16
    labels, okta, wmoids, timestamps, station_y, station_x, is_dup, source_index [M]
    attrs: means, stds, aux_channels, aux_means, aux_stds, label_policy, pre_shuffled=1, split_seed
  + <split>_compact.npz / .json : aynı alt kümeler, yeni dosyadaki konumlarla (artan sırada)

Dosya ön-karıştırılmış olduğu için eğitim döngüsü sıralı okur; buffered_shuffle_generator atlanır
(h5_dataset.pre_shuffled bayrağı). Boyut: ~395k örnek için ~17 GB (patches 15 + aux 2); --no_aux ile 15 GB.

İki geçiş:
  1) kaynak sıralı okunur, seçilen satırlar geçici contiguous H5'e (tmp) yazılır
  2) tmp'den permütasyon sırasıyla 2048'lik bloklar halinde okunup hedefe sıralı yazılır

Kullanım:
    python src/export_compact_h5.py --src F:/radar/mtg_patch_dataset.h5 --split splits/split_v1.npz ^
        --out F:/radar/mtg_train_v2.h5 --seed 42
"""
import argparse
import json
import os
import time

import h5py
import numpy as np

SUBSETS = ("train", "val", "test_time", "test_station", "test_both", "test_partial")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="F:/radar/mtg_patch_dataset.h5")
    ap.add_argument("--split", default="splits/split_v1.npz")
    ap.add_argument("--out", default="F:/radar/mtg_train_v2.h5")
    ap.add_argument("--split_out", default=None, help="Yeni bölme dosyası ön eki (varsayılan: <split>_compact)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--chunk_rows", type=int, default=256, help="Hedef chunk satır sayısı (256 x 17 x 33 x 33 x 2 B ≈ 9.5 MB)")
    ap.add_argument("--no_aux", action="store_true")
    ap.add_argument("--tmp", default=None, help="Geçici dosya (varsayılan: <out>.tmp.h5)")
    args = ap.parse_args()

    split = np.load(args.split)
    sel_parts = {k: split[k].astype(np.int64) for k in SUBSETS if k in split.files}
    sel = np.unique(np.concatenate(list(sel_parts.values())))
    M = len(sel)
    print(f"Seçilen örnek: {M} ({', '.join(f'{k}={len(v)}' for k, v in sel_parts.items())})")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(M)                 # yeni konum j -> seçilen sıradaki (sorted) örnek perm[j]
    new_pos = np.empty(M, dtype=np.int64)     # seçilen sıradaki i -> yeni konum
    new_pos[perm] = np.arange(M)
    src_to_new = {int(s): int(new_pos[i]) for i, s in enumerate(sel)}

    tmp_path = args.tmp or (args.out + ".tmp.h5")
    with h5py.File(args.src, "r") as src:
        N = len(src["labels"])
        C = src["patches"].shape[1]
        P = src["patches"].shape[2]
        use_aux = (not args.no_aux) and ("aux" in src)
        A = src["aux"].shape[1] if use_aux else 0
        print(f"Kaynak: N={N}, patches [{C},{P},{P}], aux={'var' if use_aux else 'yok'}")

        # ---------- 1. geçiş: sıralı okuma -> tmp (contiguous)
        t0 = time.time()
        with h5py.File(tmp_path, "w") as tmp:
            tp = tmp.create_dataset("patches", shape=(M, C, P, P), dtype=np.float16)
            ta = tmp.create_dataset("aux", shape=(M, A, P, P), dtype=np.float16) if use_aux else None
            src_chunk = src["patches"].chunks[0] if src["patches"].chunks else args.block
            step = max(src_chunk, args.block)
            wi = 0
            for start in range(0, N, step):
                end = min(start + step, N)
                lo = np.searchsorted(sel, start); hi = np.searchsorted(sel, end)
                if hi == lo:
                    continue
                rows = sel[lo:hi] - start
                blk = src["patches"][start:end]
                tp[wi:wi + len(rows)] = blk[rows].astype(np.float16)
                if use_aux:
                    ablk = src["aux"][start:end]
                    ta[wi:wi + len(rows)] = ablk[rows]
                wi += len(rows)
                if (start // step) % 20 == 0:
                    print(f"   1. geçiş {end}/{N}  ({time.time()-t0:.0f} s)")
            assert wi == M
        print(f"1. geçiş bitti ({time.time()-t0:.0f} s)")

        # ---------- meta
        meta = {}
        for name in ("labels", "okta", "wmoids", "station_y", "station_x", "is_dup", "obs_time", "obs_dt_sec"):
            if name in src:
                meta[name] = src[name][:][sel][perm]
        ts = src["timestamps"][:][sel][perm]
        attrs = {k: src.attrs[k] for k in src.attrs}

    # ---------- 2. geçiş: tmp -> hedef (permütasyon sırasıyla)
    t0 = time.time()
    with h5py.File(tmp_path, "r") as tmp, h5py.File(args.out, "w") as dst:
        dp = dst.create_dataset("patches", shape=(M, C, P, P), dtype=np.float16, chunks=(args.chunk_rows, C, P, P))
        da = dst.create_dataset("aux", shape=(M, A, P, P), dtype=np.float16, chunks=(args.chunk_rows, A, P, P)) if use_aux else None
        for start in range(0, M, args.block):
            end = min(start + args.block, M)
            want = perm[start:end]                       # tmp satırları
            order = np.argsort(want)
            sorted_rows = want[order]
            blk = tmp["patches"][sorted_rows]            # h5py: artan indeks gerekir
            out = np.empty_like(blk); out[order] = blk
            dp[start:end] = out
            if use_aux:
                ablk = tmp["aux"][sorted_rows]
                aout = np.empty_like(ablk); aout[order] = ablk
                da[start:end] = aout
            if (start // args.block) % 20 == 0:
                print(f"   2. geçiş {end}/{M}  ({time.time()-t0:.0f} s)")
        for name, arr in meta.items():
            dst.create_dataset(name, data=arr)
        dst.create_dataset("timestamps", data=np.array(ts, dtype=object), dtype=h5py.special_dtype(vlen=str))
        dst.create_dataset("source_index", data=sel[perm].astype(np.int64))
        for k, v in attrs.items():
            dst.attrs[k] = v
        dst.attrs["pre_shuffled"] = 1
        dst.attrs["split_seed"] = args.seed
        dst.attrs["source_file"] = os.path.abspath(args.src)
    os.remove(tmp_path)
    print(f"2. geçiş bitti ({time.time()-t0:.0f} s). Yazıldı: {args.out}  ({os.path.getsize(args.out)/1e9:.1f} GB)")

    # ---------- yeni bölme dosyası
    split_out = args.split_out or os.path.splitext(args.split)[0] + "_compact"
    new_split = {k: np.sort(np.array([src_to_new[int(s)] for s in v], dtype=np.int32)) for k, v in sel_parts.items()}
    np.savez_compressed(split_out + ".npz", **new_split)
    js = os.path.splitext(args.split)[0] + ".json"
    info = json.load(open(js, encoding="utf-8")) if os.path.exists(js) else {}
    info.update({"compact_h5": os.path.abspath(args.out), "source_split": os.path.abspath(args.split),
                 "n_compact": int(M), "pre_shuffled": True})
    with open(split_out + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Bölme: {split_out}.npz  (" + ", ".join(f"{k}={len(v)}" for k, v in new_split.items()) + ")")
    print("Colab: MTGCLM_H5=<out> ve --split <split_out>.npz ile çalıştırın.")


if __name__ == "__main__":
    main()
