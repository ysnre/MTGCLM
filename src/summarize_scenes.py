"""
summarize_scenes.py — P4: artifacts_v3/scenes/*.json -> scenes_summary.csv (tutarlılık testi, faz deseni, süre).
Yalnız standart kütüphane + pandas; torch gerekmez. Makaledeki sahne sayıları bu CSV'den gelir (verify_numbers).

    python src/summarize_scenes.py --dir artifacts_v3/scenes
"""
import argparse
import glob
import json
import os

import pandas as pd


def row(path):
    d = json.load(open(path, encoding="utf-8"))
    c = d.get("check", {}) or {}
    art = d.get("artifact", {}) or {}
    sec = d.get("seconds", {}) or {}
    n_seed = len(str(d.get("seeds", "1")).split(","))

    def g(k, s):
        v = c.get(k)
        return v.get(s) if isinstance(v, dict) else None
    inp = c.get("input_absdiff_max") or {}
    return {
        "scene": d["scene"]["sat_time"], "exp": d["exp"], "seeds": d["seeds"], "mode": d["mode"], "pad": d.get("pad"),
        "roi": d.get("roi") or "", "n_rows": c.get("n_rows"), "n_labelled": c.get("n_labelled"),
        "input_absdiff_max": max(inp.values()) if inp else None,
        "rebuilt_vs_saved_median": g("prob_rebuilt_vs_saved", "median"), "rebuilt_vs_saved_max": g("prob_rebuilt_vs_saved", "max"),
        "map_vs_saved_median": g("prob_map_vs_saved", "median"), "map_vs_saved_p95": g("prob_map_vs_saved", "p95"),
        "map_vs_saved_max": g("prob_map_vs_saved", "max"),
        "acc_map": c.get("acc_map_on_labels"), "acc_patch": c.get("acc_h5patch_on_labels_same_rows"),
        "single_vs_aligned_absdiff_mean": art.get("single_vs_aligned_absdiff_mean"),
        "single_vs_aligned_mask_disagreement": art.get("mask_disagreement_frac"),
        "period4_power_single": art.get("col_power_ratio_period4_single"),
        "period4_power_aligned": art.get("col_power_ratio_period4_aligned"),
        "inputs_s": sec.get("inputs"), "model_s": sec.get("model"), "model_s_per_seed": (sec.get("model") or 0) / n_seed,
        "total_s": d.get("seconds_total"), "peak_mem_mb": d.get("peak_mem_mb"), "threads": d.get("n_threads"),
        "cloud_frac_valid": d.get("cloud_frac_valid"), "valid_frac": d.get("valid_frac"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts_v3", "scenes"))
    ap.add_argument("--out", default=None, help="Varsayılan <dir>/scenes_summary.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "scene_*.json")))
    if not files:
        raise SystemExit(f"{a.dir}: sahne JSON'u yok")
    T = pd.DataFrame([row(p) for p in files]).sort_values(["exp", "scene"])
    out = a.out or os.path.join(a.dir, "scenes_summary.csv")
    T.to_csv(out, index=False, encoding="utf-8")
    full = T[T["roi"] == ""]
    print(f"{len(T)} sahne -> {out}")
    print(full[["scene", "exp", "mode", "n_rows", "input_absdiff_max", "map_vs_saved_median", "map_vs_saved_p95",
                "single_vs_aligned_mask_disagreement", "model_s_per_seed"]].to_string(index=False))
    pooled = full.groupby("exp").agg(scenes=("scene", "count"), rows=("n_rows", "sum"),
                                     map_med_max=("map_vs_saved_median", "max"), map_max=("map_vs_saved_max", "max"),
                                     model_s_per_seed=("model_s_per_seed", "median"))
    print(pooled.to_string())


if __name__ == "__main__":
    main()
