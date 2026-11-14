"""P3 — Kanal onemi: permutasyon onemi + Integrated Gradients (IG), gunduz / alacakaranlik / gece.

Model: CNN (Medium-Flat), tum girdiler (B_full), seed'ler 1-3; en iyi val checkpoint'leri
(<dir>/checkpoints/best_model_B_full_s<k>__medium_flat.pth). Kume: test_time (bolme dosyasindan; egitimde gorulmedi).

Guvenlik kontrolu (atlanamaz): her seed icin temel olasiliklar yeniden hesaplanir ve kayitli tahmin dosyasiyla
(<dir>/preds/B_full_s<k>__medium_flat.npz, test_time_prob) karsilastirilir. En buyuk mutlak fark > --tol ise betik durur:
normalizasyon, kanal sirasi ya da checkpoint egitimdekiyle ayni degildir ve sonuclar anlamsiz olurdu.

1) Permutasyon onemi: bir kanal (ya da fiziksel grup) tum test orneklerinde ornekler arasi karistirilir (yama ici yapi
   korunur; grupta ayni permutasyon tum kanallarina uygulanir), BalAcc dususu olculur. --repeats tekrar x seed.
2) IG: tabakali alt kume (aydinlanma x sinif), taban = normalize uzayda sifir, Riemann orta-nokta kurali --ig_steps adim,
   hedef = logit farki (bulutlu - acik). Kanal basina |IG| toplami (yama geneli ve merkez 5x5) ve paylari.
   Tamlik kontrolu: sum(IG) ~ f(x) - f(taban); goreli sapma raporlanir.
Kanal adlari src/channels.py (dogrulanmis H5 sirasi, K24) ve H5 attrs['aux_channels'].
Gunes yuksekligi tanimi analyze_preds.py ile ayni (gunduz >= 5 derece, gece <= -5 derece).

    python src\\channel_importance.py --h5 C:/mtg/mtg_train_v2.h5 --dir artifacts_v3 --threads 12
Cikti: <out>/channel_importance.csv (ozet), channel_importance_raw.csv, ig_summary.csv, channel_importance.json
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channels import BASE_CHANNELS, GROUPS  # noqa: E402

ILLUM = ("gündüz", "alacakaranlık", "gece")


def balacc(pred, y):
    y = y.astype(bool); pred = pred.astype(bool)
    return 0.5 * ((pred & y).sum() / max(y.sum(), 1) + (~pred & ~y).sum() / max((~y).sum(), 1))


def read_rows(h5, rows, aux_idx, block=4096):
    """Artan satirlari ardisik bloklarla okur -> float16 [n, 17 + A, 33, 33] (aux ham)."""
    out = np.empty((len(rows), h5["patches"].shape[1] + len(aux_idx), 33, 33), np.float16)
    k = 0
    for b in np.unique(rows // block):
        lo, hi = int(b) * block, min((int(b) + 1) * block, h5["labels"].shape[0])
        sel = rows[(rows >= lo) & (rows < hi)] - lo
        p = h5["patches"][lo:hi][sel]
        a = h5["aux"][lo:hi][sel][:, aux_idx]
        out[k:k + len(sel)] = np.concatenate([p, a], axis=1)
        k += len(sel)
    return out


class Normalizer:
    """train.py prep_features ile ayni: float32; aux z-skoru (cos_sza ve radar_cov icin ortalama 0, s.s. 1 -> ham)."""
    def __init__(self, n_base, aux_means, aux_stds):
        self.n_base = n_base
        self.m = torch.as_tensor(aux_means, dtype=torch.float32).view(1, -1, 1, 1)
        self.s = torch.as_tensor(aux_stds, dtype=torch.float32).view(1, -1, 1, 1)

    def __call__(self, x16):
        x = torch.from_numpy(np.ascontiguousarray(x16)).float()
        x[:, self.n_base:] = (x[:, self.n_base:] - self.m) / self.s
        return x


@torch.no_grad()
def predict(model, X, norm, batch, swap=None):
    """swap: (kanal_indeksleri, permutasyon) -> o kanallar X[perm] satirlarindan alinir."""
    out = np.empty(len(X), np.float32)
    for lo in range(0, len(X), batch):
        hi = min(lo + batch, len(X))
        x = norm(X[lo:hi])
        if swap is not None:
            ch, perm = swap
            xs = norm(X[perm[lo:hi]])
            x[:, ch] = xs[:, ch]
        out[lo:hi] = torch.softmax(model(x), 1)[:, 1].numpy()
    return out


def integrated_gradients(model, x, steps):
    """x: [n, C, H, W] normalize. Taban sifir. Doner: IG [n, C, H, W], f(x) - f(0) (logit farki)."""
    def f(z):
        o = model(z)
        return o[:, 1] - o[:, 0]
    total = torch.zeros_like(x)
    for k in range(steps):
        a = (k + 0.5) / steps
        z = (a * x).detach().requires_grad_(True)
        g, = torch.autograd.grad(f(z).sum(), z)
        total += g
    ig = x * total / steps
    with torch.no_grad():
        delta = f(x) - f(torch.zeros_like(x))
    return ig.detach(), delta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--split", default="splits/split_v2_verify.npz")
    ap.add_argument("--dir", default="artifacts_v3")
    ap.add_argument("--exp", default="B_full")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--set", default="test_time")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ig_n", type=int, default=5000)
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--meta", default="data/processed/h5meta_train_v2.npz")
    ap.add_argument("--stations", default="data/raw_observation/station_list.csv")
    ap.add_argument("--out", default=None, help="Varsayilan: <dir>/analysis")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--tol", type=float, default=1e-3, help="Kayitli tahminlerle izin verilen en buyuk mutlak fark")
    ap.add_argument("--limit", type=int, default=0, help="Yalniz deneme: test orneklerini bu sayiya indir")
    ap.add_argument("--skip_ig", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = a.out or os.path.join(a.dir, "analysis")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    from analyze_preds import load_meta
    from model import MTGConvNet
    meta = load_meta(a.meta or None, None if a.meta else a.h5, a.stations)   # --meta "" -> meta H5'ten
    rows = np.load(a.split)[a.set].astype(np.int64)
    if a.limit:
        rows = np.sort(np.random.default_rng(a.seed).choice(rows, size=min(a.limit, len(rows)), replace=False))
    with h5py.File(a.h5, "r") as f:
        if len(f["labels"]) != len(meta):
            raise SystemExit(f"H5 satir sayisi {len(f['labels'])} != meta {len(meta)}")
        aux_names = json.loads(f.attrs["aux_channels"])
        aux_idx = list(range(len(aux_names)))
        norm = Normalizer(int(f["patches"].shape[1]), np.asarray(f.attrs["aux_means"], np.float32)[aux_idx],
                          np.asarray(f.attrs["aux_stds"], np.float32)[aux_idx])
        print(f"{a.set}: {len(rows)} ornek okunuyor...")
        X = read_rows(f, rows, aux_idx)
        y = f["labels"][:][rows].astype(np.int64)
    names = BASE_CHANNELS + aux_names
    if X.shape[1] != len(names):
        raise SystemExit(f"Kanal sayisi {X.shape[1]} != ad sayisi {len(names)}")
    illum = meta.illum.values[rows]
    print(f"  {X.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s; kanallar: {', '.join(names)}")

    units = [(n, "channel", [names.index(n)]) for n in names]
    units += [(g, "group", [names.index(c) for c in chs if c in names]) for g, chs in GROUPS.items()]
    subsets = {"tümü": np.ones(len(rows), bool), **{il: illum == il for il in ILLUM}}

    raw, ig_rows, checks = [], [], []
    rng = np.random.default_rng(a.seed)
    for s in [int(x) for x in a.seeds.split(",")]:
        ck = os.path.join(a.dir, "checkpoints", f"best_model_{a.exp}_s{s}__medium_flat.pth")
        model = MTGConvNet(in_channels=len(names), num_classes=2, conv_blocks=[32, 64, 128], use_gap=False)
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
        model.eval()
        p0 = predict(model, X, norm, a.batch)
        pf = os.path.join(a.dir, "preds", f"{a.exp}_s{s}__medium_flat.npz")
        z = np.load(pf)
        ref_rows, ref_p = z[f"{a.set}_rows"].astype(np.int64), z[f"{a.set}_prob"]
        pos = np.searchsorted(ref_rows, rows)
        if not np.array_equal(ref_rows[pos], rows):
            raise SystemExit(f"{pf}: test satirlari uyusmuyor")
        dmax = float(np.abs(ref_p[pos] - p0).max())
        checks.append({"seed": s, "max_abs_diff_vs_saved_preds": dmax})
        print(f"seed {s}: kayitli tahminlerle en buyuk fark {dmax:.2e}")
        if dmax > a.tol:
            raise SystemExit(f"Guvenlik kontrolu basarisiz (fark {dmax:.3g} > {a.tol}): normalizasyon/kanal sirasi/checkpoint uyusmuyor")
        base = {k: balacc(p0[m] >= 0.5, y[m]) for k, m in subsets.items()}

        t1 = time.time()
        for name, kind, ch in units:
            for r in range(a.repeats):
                perm = rng.permutation(len(rows))
                p = predict(model, X, norm, a.batch, swap=(ch, perm))
                for k, m in subsets.items():
                    ba = balacc(p[m] >= 0.5, y[m])
                    raw.append({"seed": s, "birim": name, "tur": kind, "tekrar": r, "aydinlanma": k, "n": int(m.sum()),
                                "BalAcc_temel": base[k], "BalAcc": ba, "dusus": base[k] - ba})
        print(f"  permutasyon: {len(units)} birim x {a.repeats} tekrar, {time.time() - t1:.0f} s")

        if not a.skip_ig:
            t1 = time.time()
            sel = []
            per = max(1, a.ig_n // 6)
            for il in ILLUM:
                for c in (0, 1):
                    cand = np.flatnonzero((illum == il) & (y == c))
                    sel.append(rng.choice(cand, size=min(per, len(cand)), replace=False))
            sel = np.sort(np.concatenate(sel))
            att_p, att_c, comp = [], [], []
            for lo in range(0, len(sel), 256):
                b = sel[lo:lo + 256]
                ig, delta = integrated_gradients(model, norm(X[b]), a.ig_steps)
                att_p.append(ig.abs().sum(dim=(2, 3)).numpy())
                att_c.append(ig[:, :, 14:19, 14:19].abs().sum(dim=(2, 3)).numpy())
                comp.append(np.c_[ig.sum(dim=(1, 2, 3)).numpy(), delta.numpy()])
            att_p, att_c, comp = np.concatenate(att_p), np.concatenate(att_c), np.concatenate(comp)
            rel = np.abs(comp[:, 0] - comp[:, 1]) / np.maximum(np.abs(comp[:, 1]), 1e-6)
            checks[-1].update({"ig_n": int(len(sel)), "ig_completeness_rel_err_median": float(np.median(rel)),
                               "ig_completeness_rel_err_p90": float(np.percentile(rel, 90))})
            il_sel = illum[sel]
            for k in ("tümü",) + ILLUM:
                m = np.ones(len(sel), bool) if k == "tümü" else il_sel == k
                if not m.any():
                    continue
                tp, tc = att_p[m].mean(0), att_c[m].mean(0)
                for j, n in enumerate(names):
                    ig_rows.append({"seed": s, "aydinlanma": k, "n": int(m.sum()), "kanal": n,
                                    "absIG_yama": float(tp[j]), "absIG_merkez5": float(tc[j]),
                                    "pay_yama": float(tp[j] / tp.sum()), "pay_merkez5": float(tc[j] / tc.sum())})
            print(f"  IG: {len(sel)} ornek x {a.ig_steps} adim, {time.time() - t1:.0f} s; "
                  f"tamlik goreli sapma medyan {np.median(rel):.3g}")

    R = pd.DataFrame(raw)
    R.to_csv(os.path.join(out, "channel_importance_raw.csv"), index=False, encoding="utf-8")
    S = (R.groupby(["birim", "tur", "aydinlanma"])
         .agg(n=("n", "first"), BalAcc_temel=("BalAcc_temel", "mean"), dusus_ort=("dusus", "mean"),
              dusus_ss=("dusus", "std"), dusus_min=("dusus", "min"), dusus_max=("dusus", "max"),
              olcum=("dusus", "size")).reset_index().sort_values(["aydinlanma", "dusus_ort"], ascending=[True, False]))
    S.to_csv(os.path.join(out, "channel_importance.csv"), index=False, encoding="utf-8")
    if ig_rows:
        G = pd.DataFrame(ig_rows)
        G.groupby(["aydinlanma", "kanal"]).agg(n=("n", "first"), absIG_yama=("absIG_yama", "mean"),
                                               absIG_merkez5=("absIG_merkez5", "mean"), pay_yama=("pay_yama", "mean"),
                                               pay_yama_ss=("pay_yama", "std"), pay_merkez5=("pay_merkez5", "mean")) \
            .reset_index().to_csv(os.path.join(out, "ig_summary.csv"), index=False, encoding="utf-8")
    info = {"exp": a.exp, "set": a.set, "n": int(len(rows)), "seeds": a.seeds, "repeats": a.repeats, "ig_steps": a.ig_steps,
            "channels": names, "groups": GROUPS, "checks": checks, "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(out, "channel_importance.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1, ensure_ascii=False)
    print(f"Yazildi: {out}/channel_importance.csv, channel_importance_raw.csv, ig_summary.csv ({info['minutes']} dk)")


if __name__ == "__main__":
    main()
