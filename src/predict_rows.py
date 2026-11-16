"""
predict_rows.py — Kayıtlı v3 modelleriyle (yeniden eğitim yok) H5'in istenen satırları için olasılık üretir (P9: "keep" etiket
politikasındaki, verify'da çıkarılmış doğrulanmamış METAR açık satırları).

Satır kümesi --rows ile: 'keep_extra' = split_v1_compact test kümeleri − split_v2_verify test kümeleri (küme adıyla).
Girdi hazırlığı train.py / train_unet.py ile aynı: yama float16 → float32, aux (config'teki aux_channels sırası) H5 attrs
aux_means/aux_stds ile z-skor; Medium_Flat / U-Net (merkez piksel) / softmax[:, 1].
Güvenlik: önce her ağırlık için kayıtlı tahmin dosyasındaki 512 test_time satırı yeniden hesaplanır; fark > 1e-4 ise durur.
Çıktı: artifacts_v3/preds_extra/<exp>_s<seed>__keep_extra.npz  (<küme>_rows, <küme>_prob)

    python src/predict_rows.py --h5 C:/mtg/mtg_train_v2.h5 --threads 12
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MTGConvNet, MTGResNet18, MTGUNet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = ("test_time", "test_station", "test_both")


def load_model(cfg_key, cfg, ckpt, in_ch):
    if cfg_key == "MTGUNet":
        net, unet = MTGUNet(in_channels=in_ch, num_classes=2), True
    elif cfg_key in ("ResNet18", "ResNet18_CIFAR"):
        net, unet = MTGResNet18(in_channels=in_ch, num_classes=2, stem="cifar" if "CIFAR" in cfg_key else "imagenet"), False
    elif cfg_key == "Medium_Flat":
        net, unet = MTGConvNet(in_channels=in_ch, num_classes=2, conv_blocks=[32, 64, 128], use_gap=False), False
    else:
        raise SystemExit(f"desteklenmeyen mimari {cfg_key}")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) and "model_state_dict" in sd else sd)
    return net.eval(), unet


def read_inputs(f, rows, aux_idx, am, asd, block=2048):
    out = []
    P, A = f["patches"], f["aux"] if aux_idx is not None else None
    for s in range(0, len(rows), block):
        r = rows[s:s + block]
        x = P[r].astype(np.float32)
        if aux_idx is not None:
            a = A[r][:, aux_idx].astype(np.float32)
            x = np.concatenate([x, (a - am) / asd], axis=1)
        out.append(x)
    return np.concatenate(out)


@torch.no_grad()
def predict(net, unet, x, batch=1024):
    ps = []
    for i in range(0, len(x), batch):
        o = net(torch.from_numpy(x[i:i + batch]))
        if unet:
            o = o[:, :, 16, 16]
        ps.append(torch.softmax(o, 1)[:, 1].numpy())
    return np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--dir", default=os.path.join(ROOT, "artifacts_v3"))
    ap.add_argument("--exps", default="B_full,A_sat,C_noawos,C_t2m,D_unet")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--split_keep", default=os.path.join(ROOT, "splits/split_v1_compact.npz"))
    ap.add_argument("--split_verify", default=os.path.join(ROOT, "splits/split_v2_verify.npz"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts_v3/preds_extra"))
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    k1, k2 = np.load(a.split_keep), np.load(a.split_verify)
    extra = {s: np.setdiff1d(k1[s], k2[s]) for s in SETS}
    print("Ek satır:", {s: len(v) for s, v in extra.items()})
    t0 = time.time()
    with h5py.File(a.h5, "r") as f:
        names = json.loads(f.attrs["aux_channels"])
        am_all = np.asarray(f.attrs["aux_means"], np.float32)
        as_all = np.asarray(f.attrs["aux_stds"], np.float32)
        cache = {}
        for exp in a.exps.split(","):
            for seed in a.seeds.split(","):
                jp = os.path.join(a.dir, f"{exp}_s{seed}.json")
                d = json.load(open(jp, encoding="utf-8"))
                key = next(iter(d))
                cfg = d[key]["config"]
                if cfg.get("base_channels") not in (None, "all") or cfg.get("spatial_shuffle"):
                    raise SystemExit(f"{exp}: alt kanal / karıştırma desteklenmiyor")
                if cfg.get("aux"):
                    sel = names if cfg.get("aux_channels") is None else [c for c in cfg["aux_channels"] if c in names]
                    aux_idx = np.array([names.index(c) for c in sel])
                    am, asd = am_all[aux_idx][:, None, None], as_all[aux_idx][:, None, None]
                else:
                    aux_idx, am, asd = None, None, None
                in_ch = 17 + (0 if aux_idx is None else len(aux_idx))
                net, unet = load_model(key, cfg, os.path.join(a.dir, "checkpoints", cfg["best_model_file"]), in_ch)
                # güvenlik kontrolü
                z = np.load(os.path.join(a.dir, cfg["preds_file"]))
                rr, pp = z["test_time_rows"][:512], z["test_time_prob"][:512]
                ck = tuple(aux_idx) if aux_idx is not None else None
                diff = float(np.abs(predict(net, unet, read_inputs(f, np.sort(rr), aux_idx, am, asd)) - pp[np.argsort(rr)]).max())
                print(f"  {exp} s{seed}: kayıtlı tahminle en büyük fark {diff:.2e}")
                if diff > 1e-4:
                    raise SystemExit("Girdi hazırlığı eğitimle uyuşmuyor — durduruldu.")
                res = {}
                for s, rows in extra.items():
                    if ck not in cache:
                        cache[ck] = {}
                    if s not in cache[ck]:
                        cache[ck][s] = read_inputs(f, rows, aux_idx, am, asd)
                    res[f"{s}_rows"] = rows.astype(np.int32)
                    res[f"{s}_prob"] = predict(net, unet, cache[ck][s]).astype(np.float32)
                stem = os.path.splitext(cfg["best_model_file"])[0].replace("best_model_", "")
                np.savez_compressed(os.path.join(a.out, f"{stem}__keep_extra.npz"), **res)
                print(f"    yazıldı ({time.time() - t0:.0f} s)")
    print("Bitti.")


if __name__ == "__main__":
    main()
