"""
metrics.py — ikili bulut maskesi için genişletilmiş metrikler (inceleme A1).

compute_metrics(preds, targets, probs=None) -> dict
    accuracy, balanced_accuracy, mcc, precision/recall (cloudy=1), specificity,
    f1_cloudy, f1_clear, f1_macro, roc_auc (probs verilirse), confusion [tn, fp, fn, tp], n,
    majority_acc / majority_f1_cloudy (aynı kümede 'hep bulutlu' baseline'ı).
-1 etiketli örnekler atılır. Ek bağımlılık yok (sklearn gerekmez).
"""
import numpy as np


def _roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U ile ROC-AUC (bağlı skorlar ortalama sıra ile)."""
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(preds, targets, probs=None) -> dict:
    preds = np.asarray(preds).astype(np.int64)
    targets = np.asarray(targets).astype(np.int64)
    mask = targets >= 0
    preds, targets = preds[mask], targets[mask]
    n = int(len(targets))
    empty = {k: float("nan") for k in ("accuracy", "balanced_accuracy", "mcc", "precision", "recall", "specificity",
                                        "f1_cloudy", "f1_clear", "f1_macro", "roc_auc", "majority_acc", "majority_f1_cloudy")}
    if n == 0:
        return {**empty, "confusion": [0, 0, 0, 0], "n": 0}

    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)               # cloudy duyarlılığı
    specificity = tn / (tn + fp + eps)          # clear duyarlılığı
    f1_cloudy = 2 * precision * recall / (precision + recall + eps)
    prec_clear = tn / (tn + fn + eps)
    f1_clear = 2 * prec_clear * specificity / (prec_clear + specificity + eps)
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0
    p = float(targets.mean())
    out = {
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "mcc": float(mcc),
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1_cloudy": f1_cloudy, "f1_clear": f1_clear, "f1_macro": 0.5 * (f1_cloudy + f1_clear),
        "roc_auc": _roc_auc(np.asarray(probs, dtype=np.float64)[mask], targets) if probs is not None else float("nan"),
        "confusion": [tn, fp, fn, tp], "n": n,
        "majority_acc": max(p, 1 - p), "majority_f1_cloudy": 2 * p / (1 + p) if p > 0 else 0.0,
    }
    return {k: (float(v) if isinstance(v, (np.floating, float, int)) and k not in ("n",) else v) for k, v in out.items()}


def format_metrics(m: dict) -> str:
    return (f"Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} MCC={m['mcc']:.4f} "
            f"F1c={m['f1_cloudy']:.4f} F1clear={m['f1_clear']:.4f} AUC={m['roc_auc']:.4f} "
            f"(P={m['precision']:.3f} R={m['recall']:.3f} Spec={m['specificity']:.3f}; "
            f"majority Acc={m['majority_acc']:.3f}/F1c={m['majority_f1_cloudy']:.3f}; n={m['n']})")
