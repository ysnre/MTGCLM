"""
update_h5_labels_v2.py
======================
H5 etiketlerini parametrik eşik politikasıyla yeniden yazar ve çift örnekleri işaretler.

Eklenen datasetler:
    okta        [N] int8   : eşleşen gözlemin ham bulut miktarı (0-9; -1 = gözlem bulunamadı)
    obs_time    [N] int64  : gözlem geçerlilik zamanı (unix s, UTC; -1 = yok)
    obs_dt_sec  [N] int16  : gözlem - uydu (saniye)
    is_dup      [N] uint8  : aynı (wmoid, obs_time) gözleminin uyduya EN YAKIN olmayan kopyası
    labels_prev [N] int64  : eski etiketlerin yedeği (ilk çalıştırmada)
    labels      [N] int64  : YENİ etiket  (0 = clear, 1 = cloudy, -1 = ignore)
Attrs:
    label_policy (JSON): clear_max, cloudy_min, okta9, dedup, ...

Varsayılan politika (inceleme A2): clear = 0 okta, cloudy >= 5 okta (BKN/OVC), 1-4 okta -> -1,
9 (gök görünmüyor): --okta9 rh -> istasyondaki RH >= --fog_rh (90%) ise sis kabul edilip cloudy, değilse ignore
(bunun için önce update_h5_aux_channels.py çalışmış olmalı; rh2m_center kullanılır).
Eski davranışa dönmek için: --clear_max 0 --cloudy_min 1 --okta9 ignore --keep_dups

Kullanım:
    python src/update_h5_labels_v2.py --h5 F:/radar/mtg_patch_dataset.h5 ^
        --awos_aligned_dir e:/belgeler/MTGCLM/data/processed/awos_aligned
    python src/update_h5_labels_v2.py ... --dry_run     (H5'e yazmadan istatistik)
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from grid_utils import parse_sat_timestamp

EPOCH = dt.datetime(1970, 1, 1)


def to_unix(t: dt.datetime) -> int:
    return int((t - EPOCH).total_seconds())


def load_aligned_observations(aligned_dir: str) -> dict:
    """
    awos_aligned/*.csv -> {(sat_ts_str, wmoid): [(obs_unix, okta), ...]}
    sat_ts_str = dosya adındaki YYYYMMDDHHMMSS.
    """
    files = sorted(glob.glob(os.path.join(aligned_dir, "*_awos.csv")))
    if not files:
        raise SystemExit(f"awos_aligned CSV bulunamadı: {aligned_dir}")
    print(f"{len(files)} hizalanmış gözlem dosyası okunuyor...")
    table = {}
    t0 = time.time()
    for n, path in enumerate(files):
        base = os.path.basename(path)
        try:
            sat_ts = base.split("_")[2][:14]
        except IndexError:
            continue
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    wmoid = int(row["wmoid"])
                    okta = int(float(row["cloud_coverage"]))
                    obs_t = dt.datetime.strptime(row["validity"][:19], "%Y-%m-%d %H:%M:%S")
                except (KeyError, ValueError):
                    continue
                table.setdefault((sat_ts, wmoid), []).append((to_unix(obs_t), okta))
        if (n + 1) % 5000 == 0:
            print(f"   {n+1}/{len(files)} dosya ({time.time()-t0:.0f} s)")
    print(f"   {len(table)} (uydu anı, istasyon) anahtarı yüklendi ({time.time()-t0:.0f} s)")
    return table


def apply_policy(okta: np.ndarray, clear_max: int, cloudy_min: int, okta9: str,
                 rh_center: np.ndarray = None, fog_rh: float = 90.0) -> np.ndarray:
    lab = np.full(okta.shape, -1, dtype=np.int64)
    lab[(okta >= 0) & (okta <= clear_max)] = 0
    lab[(okta >= cloudy_min) & (okta <= 8)] = 1
    if okta9 == "cloudy":
        lab[okta == 9] = 1
    elif okta9 == "clear":
        lab[okta == 9] = 0
    elif okta9 == "rh":
        # gök görünmüyor + yüksek nem = sis/alçak stratus -> uydudan bulut; düşük nem (kar, toz, karanlık) -> ignore
        fog = (okta == 9) & np.isfinite(rh_center) & (rh_center >= fog_rh)
        lab[fog] = 1
    return lab


def majority_stats(labels: np.ndarray) -> str:
    v = labels[labels >= 0]
    if len(v) == 0:
        return "geçerli etiket yok"
    p = v.mean()
    f1_all_cloudy = 2 * p / (1 + p)
    return (f"n={len(v)}, cloudy oranı={p:.3f}, 'hep bulutlu' baseline: Acc={max(p,1-p):.3f}, "
            f"F1(cloudy)={f1_all_cloudy:.3f}, BalancedAcc=0.500")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="F:/radar/mtg_patch_dataset.h5")
    ap.add_argument("--awos_aligned_dir", default="e:/belgeler/MTGCLM/data/processed/awos_aligned")
    ap.add_argument("--clear_max", type=int, default=0, help="<= bu okta 'clear' (varsayılan 0)")
    ap.add_argument("--cloudy_min", type=int, default=5, help=">= bu okta 'cloudy' (varsayılan 5 = BKN/OVC)")
    ap.add_argument("--okta9", choices=["ignore", "cloudy", "clear", "rh"], default="rh",
                    help="SYNOP N=9 (gök görünmüyor): rh = istasyondaki bağıl nem >= --fog_rh ise sis -> cloudy, "
                         "değilse ignore (H5'te rh2m_center gerekir: önce update_h5_aux_channels.py); ignore/cloudy/clear sabit")
    ap.add_argument("--fog_rh", type=float, default=90.0, help="okta9=rh için sis eşiği (%%)")
    ap.add_argument("--keep_dups", action="store_true", help="Çift örnekleri -1 yapma (yalnızca is_dup işaretle)")
    ap.add_argument("--dry_run", action="store_true", help="H5'e yazma, yalnızca istatistik bas")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.cloudy_min <= args.clear_max:
        raise SystemExit("cloudy_min > clear_max olmalı")

    table = load_aligned_observations(args.awos_aligned_dir)

    rh_center = None
    with h5py.File(args.h5, "r") as f:
        N = len(f["labels"]) if args.limit is None else min(len(f["labels"]), args.limit)
        ts_raw = f["timestamps"][:N]
        wmoids = f["wmoids"][:N].astype(np.int64)
        old_labels = f["labels"][:N]
        if args.okta9 == "rh":
            if "rh2m_center" in f:
                rh_center = f["rh2m_center"][:N].astype(np.float32)
            elif "aux" in f:
                rh_center = np.concatenate([f["aux"][s:min(s + 4096, N), 1, 16, 16].astype(np.float32)
                                            for s in range(0, N, 4096)])
            else:
                raise SystemExit("--okta9 rh için H5'te rh2m_center/aux yok: önce src/update_h5_aux_channels.py "
                                 "çalıştırın ya da --okta9 ignore|cloudy kullanın.")

    okta = np.full(N, -1, dtype=np.int8)
    obs_time = np.full(N, -1, dtype=np.int64)
    obs_dt = np.zeros(N, dtype=np.int16)
    n_multi = 0
    for i in range(N):
        ts = ts_raw[i].decode("utf-8") if isinstance(ts_raw[i], (bytes, np.bytes_)) else str(ts_raw[i])
        cands = table.get((ts[:14], int(wmoids[i])))
        if not cands:
            continue
        sat_unix = to_unix(parse_sat_timestamp(ts))
        if len(cands) > 1:
            n_multi += 1
        best = min(cands, key=lambda c: abs(c[0] - sat_unix))
        okta[i] = best[1]
        obs_time[i] = best[0]
        obs_dt[i] = int(np.clip(best[0] - sat_unix, -32000, 32000))
    print(f"Gözlem eşleşen örnek: {(okta >= 0).sum()}/{N}; aynı (uydu anı, istasyon) için >1 gözlem: {n_multi}")

    # --- çift tespiti: aynı (wmoid, obs_time) -> uyduya en yakın olan kalır
    is_dup = np.zeros(N, dtype=np.uint8)
    order = np.lexsort((np.abs(obs_dt.astype(np.int32)), obs_time, wmoids))  # wmoid, obs_time, |dt| sırası
    seen = set()
    for i in order:
        if obs_time[i] < 0:
            continue
        key = (int(wmoids[i]), int(obs_time[i]))
        if key in seen:
            is_dup[i] = 1
        else:
            seen.add(key)
    print(f"Çift örnek (aynı gözlemin ikinci kopyası): {is_dup.sum()} ({is_dup.mean()*100:.1f}%)")

    labels = apply_policy(okta, args.clear_max, args.cloudy_min, args.okta9, rh_center, args.fog_rh)
    if args.okta9 == "rh":
        n9 = int((okta == 9).sum()); n_fog = int(((okta == 9) & (labels == 1)).sum())
        print(f"okta 9: {n9} örnek; RH>={args.fog_rh:.0f}% ile sis->cloudy: {n_fog}, ignore: {n9 - n_fog}")
    if not args.keep_dups:
        labels[is_dup == 1] = -1

    # --- rapor
    print("\nOkta dağılımı (eşleşenler):")
    vals, cnts = np.unique(okta[okta >= 0], return_counts=True)
    for v, c in zip(vals, cnts):
        print(f"   okta {v}: {c:8d}  ({c/cnts.sum()*100:5.1f}%)")
    print(f"\nEski etiketler : clear={int((old_labels==0).sum())}, cloudy={int((old_labels==1).sum())}, ignore={int((old_labels==-1).sum())}")
    print(f"   {majority_stats(old_labels)}")
    print(f"Yeni etiketler : clear={int((labels==0).sum())}, cloudy={int((labels==1).sum())}, ignore={int((labels==-1).sum())}")
    print(f"   {majority_stats(labels)}")
    print(f"   politika: clear<= {args.clear_max}, cloudy>= {args.cloudy_min}, okta9={args.okta9}"
          f"{f' (fog_rh={args.fog_rh:.0f})' if args.okta9 == 'rh' else ''}, dedup={'hayır' if args.keep_dups else 'evet'}")

    if args.dry_run:
        print("\n(dry_run) H5 değiştirilmedi.")
        return

    with h5py.File(args.h5, "a") as f:
        if "labels_prev" not in f:
            f.create_dataset("labels_prev", data=f["labels"][:], dtype=np.int64)
            print("Eski etiketler 'labels_prev' olarak yedeklendi.")
        for name, arr in (("okta", okta), ("obs_time", obs_time), ("obs_dt_sec", obs_dt), ("is_dup", is_dup)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=arr)
        f["labels"][:N] = labels
        f.attrs["label_policy"] = json.dumps({
            "clear_max": args.clear_max, "cloudy_min": args.cloudy_min, "okta9": args.okta9, "fog_rh": args.fog_rh,
            "dedup": not args.keep_dups, "written": dt.datetime.utcnow().isoformat(timespec="seconds"),
        })
    print("\nH5 güncellendi: labels, okta, obs_time, obs_dt_sec, is_dup.")


if __name__ == "__main__":
    main()
