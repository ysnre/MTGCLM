"""
bench_speed.py — model başına eğitim hızı ölçümü (veri gerekmez; K16, 16 Eki 2026)
=================================================================================
Sentetik batch'lerle (64 x 22 x 33 x 33) her model için bir eğitim adımının (ileri + geri + AdamW) ve
bir değerlendirme adımının süresini ölçer; split_v2_verify boyutlarıyla koşu başına süreyi tahmin eder.
Amaç: hangi koşunun yerel CPU'da, hangisinin Colab GPU'da yapılacağına karar vermek ve CPU'da en iyi
iş parçacığı sayısını bulmak (i7-12700: 8 performans + 4 verimlilik çekirdeği; 8 ile 12 farklı çıkabilir).

Kullanım (proje kökünden):
    python src/bench_speed.py                       # varsayılan iş parçacığı
    python src/bench_speed.py --threads 8,12,20     # birkaç değer karşılaştır
    python src/bench_speed.py --models medium_flat,unet --seconds 15
Not: veri okuma bu ölçüme dahil değil (yeni yükleyicide bellek modunda ihmal edilebilir).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn

# split_v2_verify boyutları (make_label_policy.py çıktısı)
N_TRAIN, N_VAL, N_TEST = 179_326, 43_152, 40_868 + 43_865 + 10_215 + 16_941


def build(name, in_ch):
    from model import MTGConvNet, MTGUNet, MTGResNet18
    if name == "medium_flat":
        return MTGConvNet(in_channels=in_ch, num_classes=2, conv_blocks=[32, 64, 128], use_gap=False)
    if name == "unet":
        return MTGUNet(in_channels=in_ch, num_classes=2)
    if name == "resnet18":
        return MTGResNet18(in_channels=in_ch, num_classes=2, stem="imagenet")
    if name == "resnet18_cifar":
        return MTGResNet18(in_channels=in_ch, num_classes=2, stem="cifar")
    raise SystemExit(f"bilinmeyen model {name}")


def time_steps(model, x, y, train, seconds, device, unet):
    crit = nn.CrossEntropyLoss(ignore_index=-1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if unet:
        t = torch.full((x.shape[0], 33, 33), -1, dtype=torch.long, device=device); t[:, 16, 16] = y
        y = t

    def step():
        if train:
            model.train(); opt.zero_grad()
            loss = crit(model(x), y); loss.backward(); opt.step()
            return loss.item()
        model.eval()
        with torch.no_grad():
            return float(model(x).sum())

    for _ in range(3):
        step()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds or n < 5:
        step(); n += 1
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="medium_flat,unet,resnet18,resnet18_cifar")
    ap.add_argument("--threads", default="", help="virgülle, ör. 8,12 (boş = torch varsayılanı)")
    ap.add_argument("--seconds", type=float, default=8.0, help="her ölçüm için süre")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--channels", type=int, default=22)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name_dev = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    threads = [int(t) for t in a.threads.split(",") if t.strip()] or [torch.get_num_threads()]
    print(f"Cihaz: {name_dev} | torch {torch.__version__} | batch {a.batch} | {a.channels} kanal | {a.epochs} epoch")
    print(f"Tahmin tabanı: train {N_TRAIN}, val {N_VAL}, son değerlendirme {N_VAL + 2 * N_TEST} örnek (split_v2_verify)\n")
    print(f"{'model':16s} {'thr':>4s} {'eğitim ms/batch':>16s} {'değ. ms/batch':>14s} {'dk/epoch':>9s} {'dk/koşu':>8s}")
    for t in threads:
        if dev.type == "cpu":
            torch.set_num_threads(t)
        for m in [s.strip() for s in a.models.split(",") if s.strip()]:
            torch.manual_seed(0)
            model = build(m, a.channels).to(dev)
            x = torch.randn(a.batch, a.channels, 33, 33, device=dev)
            y = torch.randint(0, 2, (a.batch,), device=dev)
            tr = time_steps(model, x, y, True, a.seconds, dev, m == "unet")
            ev = time_steps(model, x, y, False, a.seconds / 2, dev, m == "unet")
            ep = (N_TRAIN / a.batch * tr + N_VAL / a.batch * ev) / 60
            run = a.epochs * ep + (N_VAL + 2 * N_TEST) / a.batch * ev / 60
            print(f"{m:16s} {t if dev.type == 'cpu' else '-':>4} {tr * 1000:16.1f} {ev * 1000:14.1f} {ep:9.1f} {run:8.0f}", flush=True)
            del model
    print("\nNot: CV katlarında train ~%20 daha büyük (~215 bin). Gerçek süre, ilk denemede log'daki 'Epoch .. Time' ile doğrulanmalı.")


if __name__ == "__main__":
    main()
