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

v3 matrisi (K16, 16 Eki 2026): etiket politikası "verify" (splits/split_v2_verify.npz, splits/cv5_verify_*),
hızlı yükleyici (--loader), örnek bazlı tahmin kaydı (<outdir>/preds/), SYNOP-only metrikler.
  modality : A_sat, B_full, C_noawos, C_t2m (= C + yalnız t2m) x seed'ler          (Medium_Flat)
  unet     : D_unet x seed'ler
  resnet   : RESNET_s* (ImageNet stem) + RESNET_CIFAR_s* (CIFAR stem)
  cv       : CV_<A|B|C|Ct2m>_f<k>_s<seed> — 5 kat x 4 modalite                       (Medium_Flat)
  baselines: Z_baselines (tüm veri; majority, eşikler, logreg, rf, hgb)
  arch     : 6 mimari varyantı (v2'de koşuldu, Medium_Flat kazandı; v3'te isteğe bağlı)
--stage v3 = modality,unet,resnet,cv,baselines (arch hariç); --stage all = hepsi.
--only / --skip: koşu adı ön ekleriyle süzme (ör. Colab'da --only RESNET_CIFAR, yerelde --skip RESNET_CIFAR).

Kullanım (yerel PC, Windows — bkz. CLAUDE.md §3.4):
    python src/run_all.py --h5 F:/radar/mtg_train_v2.h5 --outdir artifacts_v3 --stage v3 \
        --skip RESNET_CIFAR --loader memory --threads 8

Kullanım (Colab):
    python src/run_all.py --h5 /content/mtg_train_v2.h5 \
        --outdir /content/drive/MyDrive/MTGCLM/artifacts_v3 --stage resnet --only RESNET_CIFAR \
        --loader stream --cache_dir /content/cache

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
STAGES = ["arch", "modality", "unet", "resnet", "cv", "baselines"]
V3_STAGES = ["modality", "unet", "resnet", "cv", "baselines"]
# Modalite varyantları: ad -> train.py ek argümanları
MODALITIES = {
    "A_sat":    [],                                                        # 17 kanal: uydu + radar
    "B_full":   ["--aux"],                                                 # + tüm aux (t2m, rh2m, elev, cos_sza, radar_cov)
    "C_noawos": ["--aux", "--aux_channels", "cos_sza,elev,radar_cov"],     # aux, AWOS'suz
    "C_t2m":    ["--aux", "--aux_channels", "cos_sza,elev,radar_cov,t2m"], # AWOS'tan yalnız t2m (K14)
}
CV_SHORT = {"A": "A_sat", "B": "B_full", "C": "C_noawos", "Ct2m": "C_t2m"}


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
    """(aşama, koşu_adı, komut_listesi) üçlüleri — yürütme sırası bu listenin sırasıdır."""
    py = sys.executable
    tr = [py, os.path.join(SRC, "train.py")]
    un = [py, os.path.join(SRC, "train_unet.py")]
    base = ["--h5", args.h5, "--outdir", args.outdir, "--loader", args.loader]
    if args.num_workers is not None:
        base += ["--num_workers", str(args.num_workers)]
    if args.no_f16:
        base += ["--no_f16"]
    if args.epochs:
        base += ["--epochs", str(args.epochs)]
    if args.limit_batches:
        base += ["--limit_batches", str(args.limit_batches)]
    if args.threads:
        base += ["--threads", str(args.threads)]
    if args.mem_splits:
        base += ["--mem_splits", args.mem_splits]
    if args.no_last_preds:
        base += ["--no_last_preds"]
    # Önbellek yalnız ana bölmeye (CV katları için her kat ~15 GB ayrı önbellek demek; --cache_cv ile açılır)
    cache = ["--cache_dir", args.cache_dir] if args.cache_dir else []
    sp = ["--split", args.split] + cache
    seeds = [int(s) for s in args.seeds.split(",")]
    res = lambda name: ["--results", os.path.join(args.outdir, f"{name}.json")]
    jobs = []

    # 1) Mimari ablasyonu (isteğe bağlı): 6 varyant tek koşuda, her seed ayrı
    for s in seeds:
        jobs.append(("arch", f"ARCH_s{s}", tr + base + sp + ["--model", "arch", "--aux", "--seed", str(s)] + res(f"ARCH_s{s}")))
    # 2) Modalite ablasyonu (mimari --arch_model ile sabit) — seed sırasıyla: önce tüm modaliteler s1
    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    for s in seeds:
        for m in mods:
            jobs.append(("modality", f"{m}_s{s}",
                         tr + base + sp + ["--model", args.arch_model, "--seed", str(s)] + MODALITIES[m] + res(f"{m}_s{s}")))
    # 3) U-Net
    for s in seeds:
        jobs.append(("unet", f"D_unet_s{s}", un + base + sp + ["--model", "unet", "--aux", "--seed", str(s)] + res(f"D_unet_s{s}")))
    # 4) ResNet18 — iki giriş katmanı (danışman talebi: model karşılaştırması)
    for s in [int(x) for x in args.resnet_imagenet_seeds.split(",") if x.strip()]:
        jobs.append(("resnet", f"RESNET_s{s}", tr + base + sp + ["--model", "resnet18", "--aux", "--seed", str(s)] + res(f"RESNET_s{s}")))
    for s in seeds:
        jobs.append(("resnet", f"RESNET_CIFAR_s{s}",
                     tr + base + sp + ["--model", "resnet18_cifar", "--aux", "--seed", str(s)] + res(f"RESNET_CIFAR_s{s}")))
    # 5) Bloklu 5-katlı CV x modaliteler (K14: AWOS-t2m etkisi CV'de sınanır)
    if args.cv_prefix:
        s0 = seeds[0]
        for short in [v.strip() for v in args.cv_variants.split(",") if v.strip()]:
            m = CV_SHORT[short]
            for k in range(args.cv_folds):
                split_k = f"{args.cv_prefix}_fold{k}.npz"
                name = f"CV_{short}_f{k}_s{s0}"
                jobs.append(("cv", name, tr + base + ["--split", split_k, "--model", args.arch_model, "--seed", str(s0)]
                             + (cache if args.cache_cv else []) + MODALITIES[m] + res(name)))
    # 6) Klasik karşılaştırma modelleri — K16: tüm veriyle (örneklem yok)
    jobs.append(("baselines", f"Z_baselines_s{seeds[0]}",
                 [py, os.path.join(SRC, "baselines.py"), "--h5", args.h5, "--split", args.split, "--aux",
                  "--models", "majority,threshold,threshold2,logreg,rf,hgb", "--seed", str(seeds[0]),
                  "--max_train", "0", "--max_eval", "0"] + res(f"Z_baselines_s{seeds[0]}")))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--outdir", required=True, help="KALICI dizin (Colab'da Drive altı)")
    ap.add_argument("--split", default="splits/split_v2_verify.npz",
                    help="v3: split_v2_verify (METAR NSC yalnız SYNOP N=0 ile doğrulanırsa açık; K16). v2: split_v1_compact")
    ap.add_argument("--cv_prefix", default="splits/cv5_verify", help="'' verilirse CV atlanır (v2: splits/cv5)")
    ap.add_argument("--cv_variants", default="A,B,C,Ct2m", help="CV'de koşulacak modaliteler: A,B,C,Ct2m")
    ap.add_argument("--modalities", default="A_sat,B_full,C_noawos,C_t2m", help="modality aşamasının varyantları")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--arch_model", default="medium_flat",
                    help="Modalite/CV koşularında kullanılacak mimari. Varsayılan medium_flat (v2 ablasyonunun "
                         "kazananı, K6). 'auto': <outdir>/ARCH_*.json'dan val balanced accuracy kazananı seçilir.")
    ap.add_argument("--resnet_imagenet_seeds", default="1,2,3",
                    help="ImageNet-stem ResNet18 (RESNET_s*) için seed'ler. Varsayılan '1,2,3': s1 v2'de koşuldu "
                         "(atlanır), s2-s3 eklenir (~2,5 sa T4); böylece iki stem de ± ile raporlanır. Yalnız mevcut "
                         "sonuç için '1' verin.")
    ap.add_argument("--stage", default="v3", help="v3 (= " + ",".join(V3_STAGES) + ") | all | " + " | ".join(STAGES)
                    + " (virgülle birden çok)")
    ap.add_argument("--only", default="", help="Yalnız bu ön eklerle başlayan koşular (virgülle), ör. RESNET_CIFAR")
    ap.add_argument("--skip", default="", help="Bu ön eklerle başlayan koşuları atla (virgülle), ör. RESNET_CIFAR,D_unet")
    ap.add_argument("--loader", default="auto", choices=["auto", "memory", "stream", "legacy"],
                    help="Veri yükleyici (K16). Yerel PC (32 GB): memory. Colab (12,7 GB): stream + --cache_dir")
    ap.add_argument("--mem_splits", default=None, help="RAM'e alınacak kümeler, ör. train,val")
    ap.add_argument("--cache_dir", default=None,
                    help="Bölme satırları .npy önbelleği (~16 GB, bir kez yazılır). Colab: /content/cache; yerel: NVMe disk")
    ap.add_argument("--cache_cv", action="store_true", help="CV katları için de önbellek yaz (kat başına ~15 GB)")
    ap.add_argument("--threads", type=int, default=None, help="CPU eğitiminde torch iş parçacığı (bench_speed.py ile seçin)")
    ap.add_argument("--no_last_preds", action="store_true", help="Son epoch test tahminlerini kaydetme")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="DataLoader işçi sayısı. Colab 2 vCPU verdiği için 0 (varsayılan) bırakın.")
    ap.add_argument("--no_f16", action="store_true",
                    help="Eski veri yolu: float32 çevirme ve aux z-skoru CPU'da. ~%30 yavaş. "
                         "Tüm koşuları ARCH_s1 ile birebir aynı kod yolunda tutmak isterseniz.")
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
    if args.cache_dir:
        args.cache_dir = os.path.abspath(args.cache_dir)
    bad_m = [m for m in args.modalities.split(",") if m.strip() and m.strip() not in MODALITIES]
    bad_c = [v for v in args.cv_variants.split(",") if v.strip() and v.strip() not in CV_SHORT]
    if bad_m or bad_c:
        raise SystemExit(f"Bilinmeyen modalite {bad_m} / CV varyantı {bad_c}. Seçenekler: {list(MODALITIES)} / {list(CV_SHORT)}")

    if args.stage == "all":
        stages = STAGES
    elif args.stage == "v3":
        stages = V3_STAGES
    else:
        stages = [s.strip() for s in args.stage.split(",")]
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

    only = tuple(x.strip() for x in args.only.split(",") if x.strip())
    skip = tuple(x.strip() for x in args.skip.split(",") if x.strip())
    jobs = [j for j in build_matrix(args) if j[0] in stages
            and (not only or j[1].startswith(only)) and not (skip and j[1].startswith(skip))]
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
                                               "Bölme doğrulandı", "Çalıştırılacak", "Finished", "Yazıldı",
                                               "[loader]", "Ortam:", "SYNOP-only", "[son epoch]")):
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
