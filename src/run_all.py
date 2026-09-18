"""
run_all.py
==========
Colab çökmelerine dayanıklı koşu yöneticisi. Tüm deney matrisini tanımlar, HANGİLERİNİN
TAMAMLANDIĞINI çıktı dizininden okur ve yalnızca eksik olanları çalıştırır. Aynı komutu
çökme sonrası tekrar verdiğinizde kaldığı yerden devam eder.

Üç seviyede kurtarma:
  1. Koşu seviyesi : tamamlanan koşu `<outdir>/status/<koşu>.done` ile işaretlenir, atlanır.
  2. Deney seviyesi: bir koşu birden çok deney içeriyorsa (--model arch = 6 varyant),
                     train.py sonuç JSON'unda adı olan deneyi zaten atlar.
  3. Epoch seviyesi: train.py her epoch sonunda checkpoint yazar (<outdir>/checkpoints/),
                     yeniden başlatınca o epoch'tan devam eder.

ÖNEMLİ: --outdir mutlaka KALICI bir yer olmalı (Colab'da Google Drive altı).
/content altı oturum çökünce silinir.

Kullanım (Colab):
    python src/run_all.py --h5 /content/mtg_train_v2.h5 \
        --outdir /content/drive/MyDrive/MTGCLM/artifacts \
        --split splits/split_v1_compact.npz --cv_prefix splits/cv5 --stage all

    # yalnızca belirli aşama(lar):
    python src/run_all.py ... --stage arch
    python src/run_all.py ... --stage modality,unet
    # ne yapılacağını görmek için:
    python src/run_all.py ... --dry_run
"""
import argparse
import json
import os
import subprocess
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
STAGES = ["arch", "resnet", "modality", "unet", "cv", "baselines"]


def pick_best_arch(outdir: str):
    """ARCH_*.json dosyalarından val balanced accuracy'ye göre en iyi mimariyi seçer.
    Birden çok seed varsa ortalamaya bakar. Hiç sonuç yoksa None döner."""
    import glob
    from collections import defaultdict
    scores = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(outdir, "ARCH_*.json"))):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for exp, res in data.items():
            v = res.get("best_val_balanced_accuracy", res.get("best_f1"))
            if isinstance(v, (int, float)) and v == v:   # NaN değilse
                scores[exp].append(float(v))
    if not scores:
        return None, {}
    means = {e: sum(v) / len(v) for e, v in scores.items()}
    best = max(means, key=means.get)
    return best, {e: (means[e], len(scores[e])) for e in means}


def build_matrix(args):
    """(aşama, koşu_adı, komut_listesi) üçlüleri."""
    py = sys.executable
    tr = [py, os.path.join(SRC, "train.py")]
    un = [py, os.path.join(SRC, "train_unet.py")]
    base = ["--h5", args.h5, "--outdir", args.outdir]
    if args.epochs:
        base += ["--epochs", str(args.epochs)]
    if args.limit_batches:
        base += ["--limit_batches", str(args.limit_batches)]
    sp = ["--split", args.split]
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = []

    # 1) Mimari ablasyonu: 6 varyant tek koşuda, her seed ayrı
    for s in seeds:
        jobs.append(("arch", f"ARCH_s{s}",
                     tr + base + sp + ["--model", "arch", "--aux", "--seed", str(s),
                                       "--results", os.path.join(args.outdir, f"ARCH_s{s}.json")]))
    # 2) ResNet18 referansı
    for s in seeds[:1]:
        jobs.append(("resnet", f"RESNET_s{s}",
                     tr + base + sp + ["--model", "resnet18", "--aux", "--seed", str(s),
                                       "--results", os.path.join(args.outdir, f"RESNET_s{s}.json")]))
    # 3) Modalite ablasyonu (mimari --arch_model ile sabit)
    for s in seeds:
        jobs.append(("modality", f"A_sat_s{s}",
                     tr + base + sp + ["--model", args.arch_model, "--seed", str(s),
                                       "--results", os.path.join(args.outdir, f"A_sat_s{s}.json")]))
        jobs.append(("modality", f"B_full_s{s}",
                     tr + base + sp + ["--model", args.arch_model, "--aux", "--seed", str(s),
                                       "--results", os.path.join(args.outdir, f"B_full_s{s}.json")]))
        jobs.append(("modality", f"C_noawos_s{s}",
                     tr + base + sp + ["--model", args.arch_model, "--aux",
                                       "--aux_channels", "cos_sza,elev,radar_cov", "--seed", str(s),
                                       "--results", os.path.join(args.outdir, f"C_noawos_s{s}.json")]))
    # 4) U-Net
    for s in seeds:
        jobs.append(("unet", f"D_unet_s{s}",
                     un + base + sp + ["--model", "unet", "--aux", "--seed", str(s),
                                       "--results", os.path.join(args.outdir, f"D_unet_s{s}.json")]))
    # 5) Çapraz doğrulama
    if args.cv_prefix:
        for k in range(args.cv_folds):
            split_k = f"{args.cv_prefix}_fold{k}.npz"
            jobs.append(("cv", f"CV_f{k}_s{seeds[0]}",
                         tr + base + ["--split", split_k, "--model", args.arch_model, "--aux",
                                      "--seed", str(seeds[0]),
                                      "--results", os.path.join(args.outdir, f"CV_f{k}_s{seeds[0]}.json")]))
    # 6) Klasik karşılaştırma modelleri
    jobs.append(("baselines", f"Z_baselines_s{seeds[0]}",
                 [py, os.path.join(SRC, "baselines.py"), "--h5", args.h5, "--split", args.split, "--aux",
                  "--models", "majority,threshold,threshold2,logreg,rf,hgb", "--seed", str(seeds[0]),
                  "--results", os.path.join(args.outdir, f"Z_baselines_s{seeds[0]}.json")]))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--outdir", required=True, help="KALICI dizin (Colab'da Drive altı)")
    ap.add_argument("--split", default="splits/split_v1_compact.npz")
    ap.add_argument("--cv_prefix", default="splits/cv5", help="'' verilirse CV atlanır")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--arch_model", default="auto",
                    help="Modalite/CV koşularında kullanılacak mimari. 'auto' (varsayılan): mimari "
                         "ablasyonunun (ARCH_*.json) val balanced accuracy kazananı otomatik seçilir. "
                         "Ya da doğrudan bir ad: medium_flat, Deep_GAP, ...")
    ap.add_argument("--stage", default="all", help="all | " + " | ".join(STAGES) + " (virgülle birden çok)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit_batches", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true", help=".done işaretlerini yok say, hepsini yeniden koş")
    ap.add_argument("--retries", type=int, default=1, help="Başarısız koşu için yeniden deneme sayısı")
    args = ap.parse_args()

    # Alt süreçler proje kökünden çalıştığı için tüm yolları mutlaklaştır
    args.h5 = os.path.abspath(args.h5)
    args.outdir = os.path.abspath(args.outdir)
    args.split = os.path.abspath(args.split)
    if args.cv_prefix:
        args.cv_prefix = os.path.abspath(args.cv_prefix)

    stages = STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"Bilinmeyen aşama: {bad}. Seçenekler: {STAGES}")

    os.makedirs(args.outdir, exist_ok=True)
    status_dir = os.path.join(args.outdir, "status")
    log_dir = os.path.join(args.outdir, "logs")
    os.makedirs(status_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(args.h5):
        raise SystemExit(f"HDF5 bulunamadı: {args.h5}")

    # --- mimari seçimi: 'auto' ise mimari ablasyonu sonuçlarından kazananı bul
    needs_arch = any(st in stages for st in ("modality", "cv"))
    if args.arch_model == "auto":
        best, table = pick_best_arch(args.outdir)
        if best is None:
            if needs_arch:
                raise SystemExit(
                    "\n--arch_model auto: henüz ARCH_*.json yok, kazanan mimari seçilemiyor.\n"
                    "  Önce mimari ablasyonunu koşun:   --stage arch\n"
                    "  ya da mimariyi elle verin:        --arch_model medium_flat\n")
            args.arch_model = "medium_flat"   # yalnızca arch/resnet koşulacaksa önemsiz
        else:
            print("Mimari ablasyonu sonuçları (val balanced accuracy, seed ortalaması):")
            for e, (m, n) in sorted(table.items(), key=lambda kv: -kv[1][0]):
                mark = "  <-- seçilen" if e == best else ""
                print(f"   {e:<14s} {m:.4f}  ({n} seed){mark}")
            args.arch_model = best

    # --- mimari değiştiyse eski modalite/CV işaretleri geçersizdir
    stamp = os.path.join(status_dir, "arch_model.txt")
    prev = open(stamp).read().strip() if os.path.exists(stamp) else None
    if prev and prev != args.arch_model and needs_arch:
        stale = [f for f in os.listdir(status_dir)
                 if f.endswith(".done") and f.split("_")[0] in ("A", "B", "C", "CV")]
        if stale:
            raise SystemExit(
                f"\nMimari değişti: '{prev}' -> '{args.arch_model}'.\n"
                f"  {len(stale)} adet modalite/CV koşusu eski mimariyle tamamlanmış görünüyor.\n"
                f"  Bunları yeniden koşmak için işaretleri silin:\n"
                f"    rm {status_dir}/[ABC]_*.done {status_dir}/CV_*.done\n"
                f"  Eskileri korumak istiyorsanız --arch_model {prev} verin.\n")
    with open(stamp, "w") as f:
        f.write(args.arch_model)
    print(f"Modalite/CV mimarisi: {args.arch_model}")

    jobs = [j for j in build_matrix(args) if j[0] in stages]
    done = [j for j in jobs if os.path.exists(os.path.join(status_dir, j[1] + ".done")) and not args.force]
    todo = [j for j in jobs if j not in done]

    print("=" * 72)
    print(f"Çıktı dizini : {os.path.abspath(args.outdir)}")
    print(f"Aşamalar     : {', '.join(stages)}")
    print(f"Toplam koşu  : {len(jobs)}   tamamlanmış: {len(done)}   yapılacak: {len(todo)}")
    if done:
        print(f"  atlanan    : {', '.join(j[1] for j in done)}")
    print("=" * 72)
    if args.dry_run:
        for st, name, cmd in todo:
            print(f"[{st}] {name}\n    {' '.join(cmd)}")
        return

    manifest_path = os.path.join(args.outdir, "run_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            manifest = json.load(open(manifest_path, encoding="utf-8"))
        except Exception:
            manifest = {}

    failed = []
    for i, (stage, name, cmd) in enumerate(todo, 1):
        log_path = os.path.join(log_dir, name + ".log")
        print(f"\n[{i}/{len(todo)}] ({stage}) {name}  ->  {log_path}")
        t0 = time.time()
        ok = False
        for attempt in range(args.retries + 1):
            if attempt:
                print(f"    yeniden deneme {attempt}/{args.retries}...")
            with open(log_path, "a", encoding="utf-8", buffering=1) as lf:
                lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} deneme {attempt} =====\n")
                lf.write(" ".join(cmd) + "\n")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, cwd=os.path.dirname(SRC) or ".")
                for line in proc.stdout:
                    lf.write(line)
                    if any(k in line for k in ("Epoch ", "Val   Loss", "[test", "HATA", "Error", "Traceback",
                                               "Bölme doğrulandı", "Çalıştırılacak", "Finished", "Yazıldı")):
                        print("   ", line.rstrip())
                ok = proc.wait() == 0
            if ok:
                break
        dt = (time.time() - t0) / 60
        manifest[name] = {"stage": stage, "ok": ok, "minutes": round(dt, 1),
                          "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": " ".join(cmd)}
        with open(manifest_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        os.replace(manifest_path + ".tmp", manifest_path)
        if ok:
            open(os.path.join(status_dir, name + ".done"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"    ✓ bitti ({dt:.1f} dk)")
        else:
            failed.append(name)
            print(f"    ✗ BAŞARISIZ ({dt:.1f} dk) — ayrıntı: {log_path}")

    print("\n" + "=" * 72)
    print(f"Bitti. Başarılı: {len(todo) - len(failed)}/{len(todo)}")
    if failed:
        print(f"Başarısız: {', '.join(failed)}")
        print("Aynı komutu tekrar çalıştırın; tamamlananlar atlanır, bunlar kaldığı epoch'tan devam eder.")
    else:
        print("Tüm koşular tamam. Rapor için:")
        print(f"  python src/aggregate_results.py --dir {args.outdir} --out {args.outdir}/summary --plot")
    print("=" * 72)


if __name__ == "__main__":
    main()
