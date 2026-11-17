"""
single_channel_auc.py — Tek değişkenli ayırma gücü: istasyon pikselinde 10,5 µm parlaklık sıcaklığı (BT10.5) ile
BT10.5 − T2m (AWOS IDW 2 m sıcaklığı) bulutlu/açık etiketlerini ne kadar ayırıyor (ROC AUC, Mann–Whitney). K14'teki ön
analizin güncel etiket politikasıyla (split_v2_verify) tekrarı; makale §4 (kanal önemi) bu tabloya dayanır.

Girdi : data/processed/center_values.npz (export_center_values.py), data/processed/h5meta_train_v2.npz, bölme dosyası.
Çıktı : artifacts_v3/analysis/single_channel_auc.csv  (küme × etiket kaynağı × değişken → AUC, n)
Daha soğuk / daha büyük negatif fark = bulutlu yönü; AUC bu yönde hesaplanır.

    python src/single_channel_auc.py
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def auc(score, y):
    """y=1 için büyük skor. Bağlar ortalama sırayla."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s), float)
    r = np.arange(1, len(s) + 1, dtype=float)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    ranks[order] = (sums / cnt)[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centers", default=os.path.join(ROOT, "data/processed/center_values.npz"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/processed/h5meta_train_v2.npz"))
    ap.add_argument("--split", default=os.path.join(ROOT, "splits/split_v2_verify.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/analysis/single_channel_auc.csv"))
    a = ap.parse_args()
    c = np.load(a.centers, allow_pickle=True)
    m = np.load(a.meta, allow_pickle=True)
    sp = np.load(a.split)
    bn = [str(x) for x in c["base_names"]]; an = [str(x) for x in c["aux_names"]]
    bt = c["base"][:, bn.index("ir_105")].astype(float)
    t2 = c["aux"][:, an.index("t2m")].astype(float) + 273.15
    lab = m["labels"]
    synop = (m["obs_time"] // 60) % 60 == 0
    feats = {"BT10.5": -bt, "BT10.5 - T2m": -(bt - t2)}
    rows = []
    for s in ("train", "val", "test_time"):
        idx = sp[s].astype(np.int64)
        idx = idx[(lab[idx] >= 0) & np.isfinite(bt[idx]) & np.isfinite(t2[idx])]
        for src, mask in (("tümü", np.ones(len(idx), bool)), ("SYNOP", synop[idx])):
            ii = idx[mask]; y = lab[ii].astype(int)
            for name, f in feats.items():
                rows.append({"küme": s, "etiket": src, "değişken": name, "auc": round(float(auc(f[ii], y)), 4), "n": len(ii)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
