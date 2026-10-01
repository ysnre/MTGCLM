"""
baselines.py
============
Derin öğrenme dışı karşılaştırma modelleri — hakemlerin "neden CNN?" sorusunun cevabı.
Hepsi AYNI bölme dosyasını, aynı metrikleri ve aynı sonuç JSON şemasını kullanır, böylece
aggregate_results.py çıktıları CNN koşularıyla tek tabloda birleştirir.

Modeller:
  majority      : her şeye çoğunluk sınıfı (referans taban)
  threshold     : tek kanal eşiği (varsayılan ir_105 = kanal 3); eşik ve yön train'de
                  balanced accuracy'yi maksimize edecek şekilde seçilir (klasik BT eşiği)
  threshold2    : iki kanal (ir_105 + vis_06) üzerinde ızgara aramalı çift eşik
  logreg        : merkez piksel + yama istatistikleri üzerinde lojistik regresyon
  rf            : rastgele orman (sklearn RandomForest)
  hgb           : histogram tabanlı gradyan artırma (sklearn HistGradientBoosting)
  knn           : k en yakın komşu (k=25), ölçeklenmiş öznitelikler

Öznitelikler (logreg/rf/hgb/knn): her kanal için merkez piksel değeri + 5x5 merkez ortalaması +
33x33 ortalaması ve standart sapması -> 4 x kanal sayısı. --aux ile aux kanalları da dahil edilir.

Kullanım:
    python src/baselines.py --h5 F:/radar/mtg_train_v2.h5 --split splits/split_v1_compact.npz ^
        --models majority,threshold,logreg,rf,hgb --aux --max_train 60000 --results artifacts/baselines.json
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_metrics, format_metrics

# create_patch_dataset.py'deki kanal sırası (get_channel_names, h5py key sırası):
CHANNEL_NAMES = ["nir_22", "ir_97", "vis_05", "ir_105", "nir_13", "ir_87", "vis_09", "ir_38",
                 "vis_04", "nir_16", "ir_133", "wv_73", "ir_123", "vis_06", "vis_08", "wv_63", "radar"]
SUBSETS = ("test_time", "test_station", "test_both")


def channel_index(name: str) -> int:
    if name.isdigit():
        return int(name)
    if name not in CHANNEL_NAMES:
        raise SystemExit(f"Bilinmeyen kanal '{name}'. Seçenekler: {', '.join(CHANNEL_NAMES)} veya indeks.")
    return CHANNEL_NAMES.index(name)


def load_features(h5, idx: np.ndarray, use_aux: bool, aux_idx, block: int = 4096):
    """idx (artan) -> X [n, 4*C] öznitelik, y [n], ayrıca merkez piksel ham değerleri [n, C]."""
    feats, centers, ys = [], [], []
    for s in range(0, len(idx), block):
        sel = idx[s:s + block]
        p = h5["patches"][sel].astype(np.float32)
        if use_aux:
            a = h5["aux"][sel].astype(np.float32)[:, aux_idx]
            p = np.concatenate([p, a], axis=1)
        c = p[:, :, 16, 16]
        m5 = p[:, :, 14:19, 14:19].mean(axis=(2, 3))
        mu = p.mean(axis=(2, 3))
        sd = p.std(axis=(2, 3))
        feats.append(np.concatenate([c, m5, mu, sd], axis=1))
        centers.append(c)
        ys.append(h5["labels"][sel])
    return np.concatenate(feats), np.concatenate(ys).astype(np.int64), np.concatenate(centers)


def best_threshold(values: np.ndarray, y: np.ndarray, n_grid: int = 200):
    """Balanced accuracy'yi maksimize eden (eşik, yön) — yön +1: değer >= eşik -> cloudy."""
    qs = np.quantile(values, np.linspace(0.01, 0.99, n_grid))
    best = (-1.0, qs[0], 1)
    for t in qs:
        for sign in (1, -1):
            pred = ((values >= t) if sign == 1 else (values < t)).astype(np.int64)
            tp = np.sum((pred == 1) & (y == 1)); tn = np.sum((pred == 0) & (y == 0))
            fp = np.sum((pred == 1) & (y == 0)); fn = np.sum((pred == 0) & (y == 1))
            ba = 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))
            if ba > best[0]:
                best = (ba, float(t), sign)
    return best[1], best[2], best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="F:/radar/mtg_train_v2.h5")
    ap.add_argument("--split", default="splits/split_v1_compact.npz")
    ap.add_argument("--models", default="majority,threshold,threshold2,logreg,rf,hgb")
    ap.add_argument("--aux", action="store_true", help="aux kanallarını da özniteliklere kat")
    ap.add_argument("--ch1", default="ir_105", help="threshold modeli için kanal")
    ap.add_argument("--ch2", default="vis_06", help="threshold2 için ikinci kanal")
    ap.add_argument("--max_train", type=int, default=60000, help="Klasik modeller için eğitim örneklemi")
    ap.add_argument("--max_eval", type=int, default=30000, help="Her test kümesi için değerlendirme örneklemi")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", default="artifacts/baselines.json")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rng = np.random.default_rng(args.seed)

    split = np.load(args.split)
    with h5py.File(args.h5, "r") as f:
        n_rows = len(f["labels"])
        aux_idx = None
        if args.aux:
            if "aux" not in f:
                raise SystemExit("--aux verildi ama H5'te 'aux' yok.")
            aux_idx = np.arange(f["aux"].shape[1])
        for name in split.files:
            if len(split[name]) and split[name].max() >= n_rows:
                raise SystemExit(f"Bölme dosyası bu H5 ile uyuşmuyor ({name}: max {split[name].max()} >= {n_rows}).")

        def sample(name, cap):
            idx = split[name].astype(np.int64)
            if cap and len(idx) > cap:
                idx = np.sort(rng.choice(idx, size=cap, replace=False))
            return idx

        t0 = time.time()
        tr_idx = sample("train", args.max_train)
        print(f"Öznitelik çıkarımı: train {len(tr_idx)}...")
        Xtr, ytr, Ctr = load_features(f, tr_idx, args.aux, aux_idx)
        evals = {}
        for s in SUBSETS:
            if s in split.files and len(split[s]):
                i = sample(s, args.max_eval)
                evals[s] = load_features(f, i, args.aux, aux_idx)
                print(f"  {s}: {len(i)} örnek")
        print(f"Öznitelikler hazır ({time.time()-t0:.0f} s), boyut {Xtr.shape}")

    results = {}

    def record(name, params, preds_probs):
        entry = {"num_params": params, "history": {}, "best_f1": float("nan"),
                 "best_val_balanced_accuracy": float("nan"), "test": {},
                 "config": {"model": name, "aux": args.aux, "split": args.split, "seed": args.seed,
                            "max_train": args.max_train, "baseline": True}}
        for s, (pred, prob) in preds_probs.items():
            m = compute_metrics(pred, evals[s][1], probs=prob)
            entry["test"][s] = m
            print(f"  [{s}] {format_metrics(m)}")
        results[name] = entry

    # ---------------- majority
    if "majority" in models:
        print("\n== majority ==")
        maj = int(np.round(ytr.mean()))
        record("majority", 1, {s: (np.full(len(evals[s][1]), maj), np.full(len(evals[s][1]), float(maj)))
                               for s in evals})

    # ---------------- tek kanal eşiği
    if "threshold" in models:
        ci = channel_index(args.ch1)
        print(f"\n== threshold ({args.ch1}, kanal {ci}) ==")
        t, sign, ba = best_threshold(Ctr[:, ci], ytr)
        print(f"  train'de seçilen eşik: {t:.3f}, yön {'>=' if sign==1 else '<'}, train BalAcc={ba:.3f}")
        out = {}
        for s in evals:
            v = evals[s][2][:, ci]
            pred = ((v >= t) if sign == 1 else (v < t)).astype(np.int64)
            out[s] = (pred, sign * v)
        record(f"threshold_{args.ch1}", 2, out)

    # ---------------- iki kanal eşiği
    if "threshold2" in models:
        c1, c2 = channel_index(args.ch1), channel_index(args.ch2)
        print(f"\n== threshold2 ({args.ch1} + {args.ch2}) ==")
        q1 = np.quantile(Ctr[:, c1], np.linspace(0.02, 0.98, 40))
        q2 = np.quantile(Ctr[:, c2], np.linspace(0.02, 0.98, 40))
        best = (-1, None)
        for t1 in q1:
            for s1 in (1, -1):
                a = (Ctr[:, c1] >= t1) if s1 == 1 else (Ctr[:, c1] < t1)
                for t2 in q2:
                    for s2 in (1, -1):
                        b = (Ctr[:, c2] >= t2) if s2 == 1 else (Ctr[:, c2] < t2)
                        pred = (a | b).astype(np.int64)
                        tp = np.sum((pred == 1) & (ytr == 1)); tn = np.sum((pred == 0) & (ytr == 0))
                        fp = np.sum((pred == 1) & (ytr == 0)); fn = np.sum((pred == 0) & (ytr == 1))
                        ba = 0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))
                        if ba > best[0]:
                            best = (ba, (t1, s1, t2, s2))
        t1, s1, t2, s2 = best[1]
        print(f"  train BalAcc={best[0]:.3f}  ({args.ch1} {'>=' if s1==1 else '<'} {t1:.3f}) VEYA "
              f"({args.ch2} {'>=' if s2==1 else '<'} {t2:.3f})")
        out = {}
        for s in evals:
            C = evals[s][2]
            a = (C[:, c1] >= t1) if s1 == 1 else (C[:, c1] < t1)
            b = (C[:, c2] >= t2) if s2 == 1 else (C[:, c2] < t2)
            pred = (a | b).astype(np.int64)
            out[s] = (pred, pred.astype(float))
        record(f"threshold2_{args.ch1}+{args.ch2}", 4, out)

    # ---------------- sklearn modelleri
    sk_models = [m for m in models if m in ("logreg", "rf", "hgb", "knn")]
    if sk_models:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise SystemExit("sklearn gerekli: pip install scikit-learn")
        builders = {
            "logreg": lambda: make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced")),
            "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                        class_weight="balanced", random_state=args.seed),
            "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                        class_weight="balanced", random_state=args.seed),
            "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
        }
        n_par = {"logreg": Xtr.shape[1] + 1, "rf": 300, "hgb": 300, "knn": len(Xtr)}
        for name in sk_models:
            print(f"\n== {name} ==")
            t0 = time.time()
            clf = builders[name]().fit(Xtr, ytr)
            print(f"  eğitildi ({time.time()-t0:.0f} s)")
            out = {}
            for s in evals:
                X = evals[s][0]
                prob = clf.predict_proba(X)[:, 1]
                out[s] = ((prob >= 0.5).astype(np.int64), prob)
            record(name + ("_aux" if args.aux else ""), n_par[name], out)

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    prev = {}
    if os.path.exists(args.results):
        try:
            prev = json.load(open(args.results, encoding="utf-8"))
        except Exception:
            pass
    prev.update(results)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False)
    print(f"\nYazıldı: {args.results}  ({', '.join(results)})")
    print("Tabloya eklemek için: python src/aggregate_results.py --dir artifacts")


if __name__ == "__main__":
    main()
