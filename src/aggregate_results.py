"""
aggregate_results.py
====================
artifacts/*.json koşu sonuçlarını toplar; deney x test kümesi bazında ortalama ± standart sapma
tablosu (Markdown, CSV, LaTeX) ve karşılaştırma grafikleri üretir.

Gruplama: dosya adı `<DENEY>_s<SEED>.json` ya da `<DENEY>_f<FOLD>_s<SEED>.json` kalıbına göre
ayrıştırılır; kalıba uymayan dosyalarda JSON içindeki `config` (seed, aux, split) kullanılır.
Aynı DENEY'in tüm seed/fold koşuları bir satırda toplanır (n = koşu sayısı).

Raporlanan metrikler (varsayılan): balanced_accuracy, mcc, f1_cloudy, f1_clear, precision_cloudy,
recall_cloudy, precision_clear, recall_clear, accuracy, roc_auc + majority_acc referansı.

Kullanım:
    python src/aggregate_results.py --dir artifacts --out artifacts/summary
    python src/aggregate_results.py --dir artifacts --subset test_time --plot
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

METRICS = ["balanced_accuracy", "mcc", "f1_cloudy", "f1_clear", "precision_cloudy", "recall_cloudy",
           "precision_clear", "recall_clear", "accuracy", "roc_auc"]
SHORT = {"balanced_accuracy": "BalAcc", "mcc": "MCC", "f1_cloudy": "F1(cloudy)", "f1_clear": "F1(clear)",
         "precision_cloudy": "P(cloudy)", "recall_cloudy": "R(cloudy)", "precision_clear": "P(clear)",
         "recall_clear": "R(clear)", "accuracy": "Acc", "roc_auc": "AUC"}
SUBSETS = ["test_time", "test_station", "test_both", "val"]
RUN_RE = re.compile(r"^(?P<exp>.+?)(?:_f(?P<fold>\d+))?_s(?P<seed>\d+)$")


def load_runs(directory: str, pattern: str):
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"  atlandı ({os.path.basename(path)}): {e}")
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        m = RUN_RE.match(stem)
        for model_name, res in data.items():
            if not isinstance(res, dict) or "num_params" not in res:
                continue
            cfg = res.get("config", {}) or {}
            exp = m.group("exp") if m else stem
            seed = int(m.group("seed")) if m and m.group("seed") else cfg.get("seed")
            fold = int(m.group("fold")) if m and m.group("fold") else None
            runs.append({
                "file": os.path.basename(path), "exp": exp, "model": model_name,
                "seed": seed, "fold": fold, "params": res.get("num_params"),
                "val_best": res.get("best_val_balanced_accuracy", res.get("best_f1")),
                "test": res.get("test", {}) or {}, "history": res.get("history", {}),
                "config": cfg,
            })
    return runs


def fmt(mean, std, n, digits=3):
    if not np.isfinite(mean):
        return "-"
    if n <= 1 or not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def aggregate(runs, subset: str):
    """exp -> {metric: (mean, std, n)} + params, majority"""
    by_exp = defaultdict(lambda: defaultdict(list))
    extra = {}
    for r in runs:
        src = r["test"].get(subset) if subset != "val" else None
        if subset == "val":
            vals = {"balanced_accuracy": r["val_best"]}
            h = r["history"]
            if h:
                i = int(np.nanargmax(h.get("val_bal_acc", [np.nan]))) if h.get("val_bal_acc") else -1
                for k_src, k_dst in (("val_f1", "f1_cloudy"), ("val_f1_clear", "f1_clear"), ("val_mcc", "mcc"),
                                     ("val_prec", "precision_cloudy"), ("val_rec", "recall_cloudy"),
                                     ("val_acc", "accuracy"), ("val_auc", "roc_auc")):
                    if h.get(k_src):
                        vals[k_dst] = h[k_src][i]
            src = vals
        if not src:
            continue
        for k in METRICS:
            v = src.get(k)
            if v is not None and np.isfinite(v):
                by_exp[r["exp"]][k].append(float(v))
        if "majority_acc" in src:
            by_exp[r["exp"]]["_majority"].append(float(src["majority_acc"]))
        if "n" in src:
            by_exp[r["exp"]]["_n"].append(int(src["n"]))
        extra.setdefault(r["exp"], {"params": r["params"], "model": r["model"],
                                    "seeds": set(), "folds": set()})
        if r["seed"] is not None:
            extra[r["exp"]]["seeds"].add(r["seed"])
        if r["fold"] is not None:
            extra[r["exp"]]["folds"].add(r["fold"])
    out = {}
    for exp, md in by_exp.items():
        row = {}
        for k in METRICS:
            v = md.get(k, [])
            row[k] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v)) \
                if v else (float("nan"), float("nan"), 0)
        row["_majority"] = float(np.mean(md["_majority"])) if md.get("_majority") else float("nan")
        row["_n"] = int(np.mean(md["_n"])) if md.get("_n") else 0
        row.update(extra[exp])
        out[exp] = row
    return out


def markdown_table(agg: dict, metrics, title: str) -> str:
    exps = sorted(agg)
    head = "| Deney | Koşu | Parametre | n | " + " | ".join(SHORT[m] for m in metrics) + " | Majority Acc |"
    sep = "|" + "---|" * (len(metrics) + 5)
    lines = [f"### {title}", "", head, sep]
    for e in exps:
        r = agg[e]
        nruns = max((r[m][2] for m in metrics), default=0)
        runs_desc = f"{nruns}"
        if r["folds"]:
            runs_desc += f" ({len(r['folds'])} kat × {max(1, len(r['seeds']))} seed)"
        elif r["seeds"]:
            runs_desc += f" ({len(r['seeds'])} seed)"
        cells = " | ".join(fmt(*r[m][:3]) for m in metrics)
        lines.append(f"| {e} | {runs_desc} | {r['params']:,} | {r['_n']:,} | {cells} | {r['_majority']:.3f} |")
    return "\n".join(lines) + "\n"


def latex_table(agg: dict, metrics, caption: str, label: str) -> str:
    exps = sorted(agg)
    cols = "l" + "r" * (len(metrics) + 1)
    lines = ["\\begin{table}[ht]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{cols}}}", "\\hline",
             "Model & " + " & ".join(SHORT[m] for m in metrics) + " & Majority \\\\", "\\hline"]
    for e in exps:
        r = agg[e]
        cells = " & ".join(fmt(*r[m][:3]).replace("±", "$\\pm$") for m in metrics)
        lines += [f"{e.replace('_', chr(92)+'_')} & {cells} & {r['_majority']:.3f} \\\\"]
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def csv_table(agg: dict, metrics, subset: str) -> str:
    rows = ["subset,experiment,model,params,n_samples,n_runs," +
            ",".join(f"{m}_mean,{m}_std" for m in metrics) + ",majority_acc"]
    for e in sorted(agg):
        r = agg[e]
        nruns = max((r[m][2] for m in metrics), default=0)
        vals = ",".join(f"{r[m][0]:.6f},{r[m][1]:.6f}" for m in metrics)
        rows.append(f"{subset},{e},{r['model']},{r['params']},{r['_n']},{nruns},{vals},{r['_majority']:.6f}")
    return "\n".join(rows) + "\n"


def partial_table(runs) -> str:
    """test_partial: okta başına ortalama bulut olasılığı (tüm koşuların ortalaması)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in runs:
        pr = r["test"].get("test_partial")
        if not isinstance(pr, dict):
            continue
        for k, v in pr.items():
            if isinstance(v, dict) and "mean_prob" in v:
                acc[r["exp"]][k].append(v["mean_prob"])
    if not acc:
        return ""
    keys = sorted({k for d in acc.values() for k in d if k.startswith("okta")},
                  key=lambda x: int(x.replace("okta", "")))
    lines = ["### Parçalı bulut tutarlılığı (test_partial): okta başına ortalama P(cloudy)", "",
             "| Deney | " + " | ".join(k.replace("okta", "okta ") for k in keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for e in sorted(acc):
        cells = " | ".join(f"{np.mean(acc[e][k]):.2f}" if acc[e].get(k) else "-" for k in keys)
        lines.append(f"| {e} | {cells} |")
    return "\n".join(lines) + "\n"


def make_plots(agg_by_subset: dict, out_prefix: str, metric: str = "balanced_accuracy"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subsets = [s for s in SUBSETS if s in agg_by_subset and agg_by_subset[s]]
    if not subsets:
        return
    exps = sorted({e for s in subsets for e in agg_by_subset[s]})
    x = np.arange(len(exps)); w = 0.8 / len(subsets)
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(exps)), 4.5), dpi=300)
    for i, s in enumerate(subsets):
        means = [agg_by_subset[s].get(e, {}).get(metric, (np.nan,))[0] for e in exps]
        stds = [agg_by_subset[s].get(e, {}).get(metric, (np.nan, np.nan))[1] for e in exps]
        stds = [0 if not np.isfinite(v) else v for v in stds]
        ax.bar(x + i * w - 0.4 + w / 2, means, w, yerr=stds, capsize=3, label=s)
    maj = [agg_by_subset[subsets[0]].get(e, {}).get("_majority", np.nan) for e in exps]
    ax.plot(x, maj, "k--", lw=1, label="majority baseline (test_time)")
    ax.set_xticks(x); ax.set_xticklabels(exps, rotation=20, ha="right")
    ax.set_ylabel(SHORT.get(metric, metric)); ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"Modalite / model karşılaştırması — {SHORT.get(metric, metric)} (ortalama ± s.s.)")
    fig.tight_layout(); fig.savefig(out_prefix + f"_{metric}.png"); plt.close(fig)
    print(f"Grafik: {out_prefix}_{metric}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="artifacts")
    ap.add_argument("--pattern", default="*.json")
    ap.add_argument("--out", default="artifacts/summary")
    ap.add_argument("--subset", default=None, help="Yalnızca bu alt küme (varsayılan: hepsi)")
    ap.add_argument("--metrics", default=None, help="Virgülle ayrılmış metrik listesi")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    metrics = args.metrics.split(",") if args.metrics else METRICS

    runs = load_runs(args.dir, args.pattern)
    if not runs:
        raise SystemExit(f"{args.dir}/{args.pattern} içinde sonuç JSON'u bulunamadı.")
    print(f"{len(runs)} koşu yüklendi: " + ", ".join(sorted({r['exp'] for r in runs})))

    subsets = [args.subset] if args.subset else SUBSETS
    md_parts, csv_parts, tex_parts, agg_by_subset = [], [], [], {}
    for s in subsets:
        agg = aggregate(runs, s)
        if not agg:
            continue
        agg_by_subset[s] = agg
        md_parts.append(markdown_table(agg, metrics, f"{s}"))
        csv_parts.append(csv_table(agg, metrics, s))
        tex_parts.append(latex_table(agg, metrics[:6], f"MTGCLM sonuçları ({s}), ortalama $\\pm$ standart sapma",
                                     f"tab:results_{s}"))
        print("\n" + md_parts[-1])
    pt = partial_table(runs)
    if pt:
        md_parts.append(pt); print("\n" + pt)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out + ".md", "w", encoding="utf-8").write(
        "# MTGCLM — sonuç özeti\n\n" + "\n".join(md_parts))
    open(args.out + ".csv", "w", encoding="utf-8").write(csv_parts[0].split("\n")[0] + "\n" +
        "\n".join("\n".join(c.split("\n")[1:]).strip() for c in csv_parts) + "\n")
    open(args.out + ".tex", "w", encoding="utf-8").write("\n".join(tex_parts))
    print(f"Yazıldı: {args.out}.md, {args.out}.csv, {args.out}.tex")
    if args.plot:
        for m in ("balanced_accuracy", "mcc", "f1_cloudy"):
            make_plots(agg_by_subset, args.out, m)


if __name__ == "__main__":
    main()
