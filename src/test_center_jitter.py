"""
test_center_jitter.py — P20 (U-Net hedef-piksel jitter) testleri
=================================================================
Birim testleri (veri gerekmez):
  1. İstasyon pikseli korunur: x[b, :, cy, cx] == features[b, :, 16, 16] (tüm kanallar, her örnek).
  2. Pencere içeriği doğru: x[b] == features[b, :, oy:oy+crop, ox:ox+crop] (ofsetler cy/cx'ten geri çıkarılır).
  3. Konumlar 8..16 aralığında ve 4x4 = 16 havuzlama fazının hepsi görülür.
  4. Aynı tohumlu üreteç aynı pencereleri verir; jitter kapalıyken küresel RNG tüketilmez.
  5. MTGUNet 25x25 girdide 25x25 çıktı verir; 33x33 değerlendirme yolu değişmez.
  6. Kayıp yalnız istasyonun penceredeki konumunda hesaplanır (diğer pikseller -1).

Bütünleşik test (--h5 ve --split verilirse; mini H5 ile ~1-2 dk):
  7. Jitter KAPALI yeni kod == eski kod (git HEAD) bit bit: aynı kayıp geçmişi ve aynı ağırlıklar.
  8. Jitter AÇIK koşu uçtan uca biter, config'e center_jitter/jitter_crop yazılır, preds dosyası oluşur.

Kullanım:
    python src/test_center_jitter.py
    python src/test_center_jitter.py --h5 data/processed/_mini_test.h5 --split <mini_split.npz> --old_script <eski train_unet.py>
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)


def unit_tests():
    from train_unet import center_jitter_crop
    from model import MTGUNet
    from loss import MaskedBCELoss

    torch.manual_seed(0)
    B, C, crop = 512, 22, 25
    feats = torch.randn(B, C, 33, 33)
    gen = torch.Generator().manual_seed(123)
    x, cy, cx = center_jitter_crop(feats, crop, gen)
    assert x.shape == (B, C, crop, crop), x.shape
    b = torch.arange(B)
    # 1) istasyon pikseli
    assert torch.equal(x[b, :, cy, cx], feats[:, :, 16, 16]), "istasyon pikseli korunmadı"
    # 2) pencere içeriği
    oy, ox = 16 - cy, 16 - cx
    for i in range(0, B, 37):
        assert torch.equal(x[i], feats[i, :, oy[i]:oy[i] + crop, ox[i]:ox[i] + crop]), f"pencere {i} yanlış"
    # 3) aralık ve fazlar
    assert int(cy.min()) >= 8 and int(cy.max()) <= 16 and int(cx.min()) >= 8 and int(cx.max()) <= 16
    phases = {(int(a) % 4, int(c) % 4) for a, c in zip(cy, cx)}
    assert len(phases) == 16, f"yalnız {len(phases)} faz görüldü"
    # 4) tekrarlanabilirlik ve küresel RNG
    x2, cy2, cx2 = center_jitter_crop(feats, crop, torch.Generator().manual_seed(123))
    assert torch.equal(x, x2) and torch.equal(cy, cy2) and torch.equal(cx, cx2)
    st = torch.get_rng_state()
    center_jitter_crop(feats, crop, torch.Generator().manual_seed(7))
    assert torch.equal(st, torch.get_rng_state()), "küresel RNG tüketildi"
    # 5) model boyutları
    net = MTGUNet(in_channels=C, num_classes=2).eval()
    with torch.no_grad():
        assert net(x[:4]).shape == (4, 2, crop, crop)
        assert net(feats[:4]).shape == (4, 2, 33, 33)
    # 6) kayıp yalnız istasyon konumunda
    labels = torch.randint(0, 2, (B,))
    sl = torch.full((B, crop, crop), -1, dtype=torch.long)
    sl[b, cy, cx] = labels
    assert int((sl != -1).sum()) == B
    out = torch.randn(B, 2, crop, crop, requires_grad=True)
    MaskedBCELoss()(out, sl).backward()
    g = out.grad.abs().sum(1)                       # [B, crop, crop]
    mask = torch.zeros_like(g, dtype=torch.bool)
    mask[b, cy, cx] = True
    assert float(g[~mask].abs().max()) == 0.0 and bool((g[mask] > 0).all()), "kayıp başka piksellere sızıyor"
    print("[OK] birim testleri (1-6)")


def run(script, h5, split, outdir, name, extra):
    res = os.path.join(outdir, f"{name}.json")
    cmd = [sys.executable, script, "--h5", h5, "--split", split, "--aux", "--seed", "1", "--epochs", "1",
           "--limit_batches", "3", "--outdir", outdir, "--results", res, "--loader", "memory", "--no_last_preds"] + extra
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, cwd=os.path.dirname(SRC), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"{name} başarısız")
    with open(res, encoding="utf-8") as f:
        r = json.load(f)["MTGUNet"]
    ck = torch.load(os.path.join(outdir, "checkpoints", f"best_model_{name}__mtgunet.pth"), map_location="cpu",
                    weights_only=False)
    return r, (ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck)


def integration_tests(h5, split, old_script):
    new_script = os.path.join(SRC, "train_unet.py")
    with tempfile.TemporaryDirectory() as td:
        r_old, w_old = run(old_script, h5, split, td, "old_off", [])
        r_new, w_new = run(new_script, h5, split, td, "new_off", [])
        assert r_old["history"]["train_loss"] == r_new["history"]["train_loss"], "kayıp geçmişi farklı"
        assert r_old["history"]["val_bal_acc"] == r_new["history"]["val_bal_acc"], "val BalAcc farklı"
        assert w_old.keys() == w_new.keys() and all(torch.equal(w_old[k], w_new[k]) for k in w_old), "ağırlıklar farklı"
        print("[OK] 7: jitter kapalıyken eski kodla bit bit aynı")
        r_j, _ = run(new_script, h5, split, td, "new_jit", ["--center_jitter", "--jitter_crop", "25"])
        cfg = r_j["config"]
        assert cfg["center_jitter"] is True and cfg["jitter_crop"] == 25
        assert r_j["history"]["train_loss"] != r_new["history"]["train_loss"], "jitter açıkken eğitim değişmedi"
        assert os.path.exists(os.path.join(td, cfg["preds_file"])), "preds dosyası yok"
        print("[OK] 8: jitter açık koşu uçtan uca; config ve preds yazıldı")


PRE_P20_COMMIT = "3f1008a"   # P20'den önceki son commit (19 Kas 2026)


def make_mini_split(h5, path):
    import h5py
    import numpy as np
    with h5py.File(h5, "r") as f:
        y = f["labels"][:]
    idx = np.where(y >= 0)[0]
    np.random.default_rng(0).shuffle(idx)
    n = len(idx)
    np.savez(path, train=np.sort(idx[:int(n * .6)]), val=np.sort(idx[int(n * .6):int(n * .8)]),
             test_time=np.sort(idx[int(n * .8):]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", help="Mini H5 (ör. data/processed/_mini_test.h5); verilirse bütünleşik testler de koşar")
    ap.add_argument("--split", help="Varsayılan: mini H5'ten 60/20/20 rastgele bölme üretilir")
    ap.add_argument("--old_script", help=f"Eski train_unet.py; varsayılan: git show {PRE_P20_COMMIT}:src/train_unet.py")
    a = ap.parse_args()
    unit_tests()
    if a.h5:
        with tempfile.TemporaryDirectory() as tmp:
            split = a.split or os.path.join(tmp, "mini_split.npz")
            if not a.split:
                make_mini_split(a.h5, split)
            old = a.old_script
            if not old:
                old = os.path.join(tmp, "train_unet_pre_p20.py")
                src = subprocess.run(["git", "show", f"{PRE_P20_COMMIT}:src/train_unet.py"], cwd=os.path.dirname(SRC),
                                     capture_output=True, text=True, encoding="utf-8").stdout
                if not src:
                    raise SystemExit(f"git show {PRE_P20_COMMIT} başarısız; --old_script verin")
                with open(old, "w", encoding="utf-8") as f:
                    f.write(src)
            integration_tests(a.h5, split, old)
    else:
        print("(bütünleşik testler atlandı: --h5 verin)")
