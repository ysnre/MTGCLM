"""
run_utils.py — train.py ve train_unet.py'nin ortak tekrarlanabilirlik / kayıt yardımcıları (K16, 16 Eki 2026)

- setup_repro(seed, deterministic): tüm RNG'ler + cuDNN/deterministik algoritma ayarları
- rng_state()/set_rng_state(): checkpoint'ten devamda RNG akışı kesintisiz koşuyla aynı kalsın
- run_env(): git sürümü (+ commit edilmemiş değişiklik var mı), Python/torch sürümü, cihaz adı, iş parçacığı
- split_info(path): bölme dosyasının yanındaki .json'dan etiket politikası bilgisi
- rows_of(loader): yükleyicinin değerlendirdiği satır indeksleri (legacy Subset ve FastLoader için aynı)
- synop_metrics(...): yalnız SYNOP etiketlerinde (gözlem dakikası 0) metrik — referanslarla adil kıyas
- save_preds(path, dict): örnek bazlı tahminleri atomik olarak .npz'ye yazar
"""
import json
import os
import platform
import random
import subprocess
import sys

import numpy as np
import torch


def setup_repro(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # cudnn.benchmark kapalı: algoritma seçimini ölçerek yaptığı için tekrarlar birebir aynı çıkmaz.
        torch.backends.cudnn.benchmark = False
        if deterministic:
            torch.backends.cudnn.deterministic = True
    if deterministic:
        # CUBLAS_WORKSPACE_CONFIG, CUDA başlamadan önce ayarlanmalı (train*.py en üstte ayarlıyor).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def rng_state() -> dict:
    """Checkpoint'e yazılacak tüm RNG durumları (torch CPU/CUDA, numpy, random)."""
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "random": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict):
    try:
        torch.set_rng_state(st["torch"])
        np.random.set_state(st["numpy"])
        random.setstate(st["random"])
        if "cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda"])
    except Exception as e:
        print(f"  --> Uyarı: RNG durumu geri yüklenemedi ({e}); devam sonuçları kesintisiz koşudan az farklı olabilir.")


def _git(*args):
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def run_env(extra: dict = None) -> dict:
    dev = "cpu"
    if torch.cuda.is_available():
        try:
            dev = torch.cuda.get_device_name(0)
        except Exception:
            dev = "cuda"
    env = {
        "git_commit": _git("rev-parse", "--short", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": dev,
        "cpu": platform.processor() or platform.machine(),
        "threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "host": platform.node(),
    }
    if extra:
        env.update(extra)
    return env


def split_info(split_path: str) -> dict:
    """splits/<ad>.npz yanındaki <ad>.json: make_label_policy.py 'policy' alanını yazar."""
    if not split_path:
        return {}
    j = os.path.splitext(split_path)[0] + ".json"
    info = {"split_file": os.path.basename(split_path)}
    try:
        with open(j, encoding="utf-8") as f:
            d = json.load(f)
        info["label_policy"] = d.get("policy", "keep")
        info["source_split"] = d.get("source_split")
    except Exception:
        info["label_policy"] = "keep"
    return info


def rows_of(loader) -> np.ndarray:
    ds = loader.dataset
    while not hasattr(ds, "indices") and hasattr(ds, "dataset"):
        ds = ds.dataset
    return np.asarray(getattr(ds, "indices", []), dtype=np.int64)


def base_of(loader):
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def synop_metrics(base, rows, probs, labels, compute_metrics):
    """Yalnız SYNOP etiketli örnekler (obs_time dakikası 0). obs_time yoksa None."""
    ot = getattr(base, "obs_time", None)
    if ot is None or len(rows) == 0:
        return None
    m = (np.asarray(ot)[rows] // 60) % 60 == 0
    if m.sum() == 0:
        return None
    p = np.asarray(probs)[m]
    return compute_metrics((p >= 0.5).astype(np.int64), np.asarray(labels)[m], probs=p)


def save_preds(path: str, arrays: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
