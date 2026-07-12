#!/usr/bin/env python3
"""
build_paper_assets.py
---------------------------------------------------------------
Aggregates the Jetson perf-per-watt JSONs (produced by benchmark_ppw.py) plus
the offline AUROC numbers into publication-ready TABLES (CSV + LaTeX/IEEEtran)
and FIGURES (PDF + PNG). Runs OFFLINE on the Legion — no Jetson needed.

Reads:
  results/ppw_<variant>_<precision>.json   (12 static engines, batch=1)
  results/ppw_*_dyn.json                    (batch sweep, one engine)
  bootstrap_diffs.csv (optional)            (ΔAUROC/CI/p from bootstrap_auroc_ci.py)

Writes to paper_assets/:
  table_main.{csv,tex}                 T1  perf + AUROC per variant×precision
  table_distillation.{csv,tex}         T2  AUROC + ΔAUROC vs baseline (+CI,p)
  table_precision_robustness.{csv,tex} T3  AUROC across FP32/FP16/INT8
  table_complexity.tex                 T4  teacher vs student params/GMACs
  table_batch_sweep.{csv,tex}          T5  batch 1..16
  table_deployment.{csv,tex}           T6  engine-load / peak RAM
  fig_pareto.{pdf,png}                 F1  AUROC vs energy-per-inference  ⭐
  fig_ppw_bar.{pdf,png}                F2  perf-per-watt by precision
  fig_batch_sweep.{pdf,png}            F3  throughput & latency vs batch
  fig_auroc_vs_precision.{pdf,png}     F5  AUROC across precisions per variant

Usage:
    python build_paper_assets.py --results results --out paper_assets
    python build_paper_assets.py --results results --bootstrap bootstrap_diffs.csv
"""
import argparse
import csv
import glob
import json
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# ============  EDIT ME: offline AUROC numbers (MACRO-5)  =============
# =====================================================================
# Fill each cell with the point AUROC (5 competition classes) from your
# eval_precisions.py / bootstrap output. Leave None where not yet measured;
# the Pareto / AUROC figures simply skip missing points.
# Confirmed anchors are pre-filled — VERIFY against your bootstrap medians.
AUROC5 = {
    #  variant          fp32     fp16     int8      (MACRO-5, from eval_precisions.py)
    "baseline":      {"fp32": 0.7953, "fp16": None,   "int8": None},    # TODO: fp16/int8 from loop
    "logitKD":       {"fp32": 0.8647, "fp16": 0.865,  "int8": None},    # TODO: int8 from loop
    "featureKD":     {"fp32": 0.8578, "fp16": 0.8581, "int8": 0.7859},  # measured
    "featureHardKD": {"fp32": 0.8548, "fp16": 0.8543, "int8": 0.7884},  # measured
}
TEACHER_AUROC5 = 0.8675   # ViT-B/16 ceiling (reference line on Pareto). None to hide.

# Params / GMACs for the complexity table (from your build_efficiency_table.py).
COMPLEXITY = {
    "ViT-B/16 (teacher)":       {"params_M": 85.81, "gmacs": 16.87},
    "EfficientNet-B0 (student)":{"params_M": 4.03,  "gmacs": 0.40},
}

VARIANT_ORDER = ["baseline", "logitKD", "featureKD", "featureHardKD"]
PREC_ORDER = ["fp32", "fp16", "int8"]
PREC_LABEL = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}
VARIANT_LABEL = {"baseline": "Baseline", "logitKD": "Logit-KD",
                 "featureKD": "Feature-KD", "featureHardKD": "Feature+Hard-KD"}


# ----------------------------- helpers -----------------------------
def parse_tag(tag):
    parts = tag.split("_")
    if parts[-1] in PREC_ORDER:
        return "_".join(parts[:-1]), parts[-1]
    return tag, None  # e.g. sweep '..._dyn'


def load_all(results_dir):
    """Returns (main[(variant,prec)] -> batch1 dict + engine meta), sweep(list)."""
    main, sweep = {}, []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        by_batch = d.get("by_batch", {})
        tag = d.get("tag", os.path.basename(path))
        variant, prec = parse_tag(tag)
        if len(by_batch) > 1 or prec is None:
            sweep.append(d)               # multi-batch or non-standard -> sweep
            continue
        b1 = by_batch.get("1")
        if b1 and "error" not in b1:
            main[(variant, prec)] = {"b1": b1, "engine": d}
    return main, sweep


def g(b1, *keys, default=None):
    cur = b1
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def _tex(s):
    """Make a cell LaTeX-safe (works with plain IEEEtran, no utf8 needed)."""
    s = "" if s is None else str(s)
    return (s.replace("×", r"$\times$").replace("−", r"$-$")
             .replace("Δ", r"$\Delta$").replace("≈", r"$\approx$")
             .replace("%", r"\%").replace("&", r"\&"))


def latex_table(path, header, rows, caption, label, colspec=None):
    ncol = len(header)
    colspec = colspec or ("l" * ncol)
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{{_tex(caption)}}}\n\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
        f.write(" & ".join(_tex(h) for h in header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(_tex(x) for x in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


# ----------------------------- T1: main benchmark -----------------------------
def build_main_table(main, outdir):
    header = ["Variant", "Prec.", "AUROC", "Lat.p50 (ms)", "Lat.p99 (ms)",
              "Thr. (img/s)", "Power (W)", "Energy (mJ)", "Perf/W (img/s/W)"]
    rows = []
    for v in VARIANT_ORDER:
        for p in PREC_ORDER:
            m = main.get((v, p))
            if not m:
                continue
            b1 = m["b1"]
            auroc = AUROC5.get(v, {}).get(p)
            power_w = (g(b1, "power_mw", "total_mean_mw") or 0) / 1000.0
            rows.append([
                VARIANT_LABEL[v], PREC_LABEL[p], fmt(auroc, 3),
                fmt(g(b1, "latency_ms", "median")), fmt(g(b1, "latency_ms", "p99")),
                fmt(g(b1, "throughput_img_per_s"), 1),
                fmt(power_w), fmt(g(b1, "energy_per_image_mj")),
                fmt(g(b1, "perf_per_watt_img_per_s_per_w")),
            ])
    write_csv(os.path.join(outdir, "table_main.csv"), header, rows)
    latex_table(os.path.join(outdir, "table_main.tex"), header, rows,
                "On-device benchmark on Jetson Orin Nano Super (batch=1, MAXN).",
                "tab:main", colspec="ll" + "r" * 7)
    return rows


# ----------------------------- T2: distillation efficacy -----------------------------
def build_distillation_table(bootstrap_csv, outdir, ref="baseline"):
    # ΔAUROC/CI/p for MACRO(5) vs the reference, from bootstrap_auroc_ci.py.
    deltas = {}
    if bootstrap_csv and os.path.exists(bootstrap_csv):
        with open(bootstrap_csv) as f:
            for row in csv.DictReader(f):
                comp = row["comparison"]
                if row["metric"].startswith("MACRO(5") and comp.endswith(f"-{ref}"):
                    deltas[comp[: -(len(ref) + 1)]] = row   # strip '-baseline'
    header = ["Model", "Params (M)", "AUROC(5)", "ΔAUROC vs base", "95% CI", "p"]
    rows = []
    student_params = COMPLEXITY["EfficientNet-B0 (student)"]["params_M"]
    for v in VARIANT_ORDER:
        auroc = AUROC5.get(v, {}).get("fp32") or AUROC5.get(v, {}).get("fp16")
        drow = deltas.get(v)
        if drow:
            d = f"{float(drow['delta']):+.3f}"
            ci = f"[{float(drow['ci_lo']):+.3f}, {float(drow['ci_hi']):+.3f}]"
            p = drow["p_value"]
        else:
            d = ci = p = ("baseline" if v == "baseline" else "")
        rows.append([VARIANT_LABEL[v], fmt(student_params, 2), fmt(auroc, 3), d, ci, p])
    if TEACHER_AUROC5:
        tp = COMPLEXITY["ViT-B/16 (teacher)"]["params_M"]
        rows.append(["ViT-B/16 (teacher)", fmt(tp, 2), fmt(TEACHER_AUROC5, 3),
                     "(ceiling)", "", ""])
    write_csv(os.path.join(outdir, "table_distillation.csv"), header, rows)
    latex_table(os.path.join(outdir, "table_distillation.tex"), header, rows,
                "Distillation efficacy: MACRO-5 AUROC and paired ΔAUROC vs baseline.",
                "tab:distill", colspec="lrrrcc")


# ----------------------------- T3: precision robustness -----------------------------
def build_precision_table(outdir):
    header = ["Variant", "AUROC FP32", "AUROC FP16", "AUROC INT8", "Δ(INT8−FP32)"]
    rows = []
    for v in VARIANT_ORDER:
        a = AUROC5.get(v, {})
        d = (a.get("int8") - a.get("fp32")) if (a.get("int8") is not None
                                                and a.get("fp32") is not None) else None
        rows.append([VARIANT_LABEL[v], fmt(a.get("fp32"), 3), fmt(a.get("fp16"), 3),
                     fmt(a.get("int8"), 3), (f"{d:+.3f}" if d is not None else "")])
    write_csv(os.path.join(outdir, "table_precision_robustness.csv"), header, rows)
    latex_table(os.path.join(outdir, "table_precision_robustness.tex"), header, rows,
                "AUROC (MACRO-5) across TensorRT precisions; INT8 degradation.",
                "tab:precision", colspec="lrrrr")


# ----------------------------- T4: complexity -----------------------------
def build_complexity_table(outdir):
    t = COMPLEXITY["ViT-B/16 (teacher)"]
    s = COMPLEXITY["EfficientNet-B0 (student)"]
    header = ["Model", "Params (M)", "GMACs", "Param ratio", "MAC ratio"]
    rows = [
        ["ViT-B/16 (teacher)", fmt(t["params_M"], 2), fmt(t["gmacs"], 2), "1.0×", "1.0×"],
        ["EfficientNet-B0 (student)", fmt(s["params_M"], 2), fmt(s["gmacs"], 2),
         f"{t['params_M']/s['params_M']:.1f}× smaller",
         f"{t['gmacs']/s['gmacs']:.1f}× fewer"],
    ]
    latex_table(os.path.join(outdir, "table_complexity.tex"), header, rows,
                "Teacher vs. student model complexity.",
                "tab:complexity", colspec="lrrrr")


# ----------------------------- T5 + F3: batch sweep -----------------------------
def build_sweep(sweep, outdir):
    if not sweep:
        print("[i] no sweep JSON found; skipping T5/F3")
        return
    d = max(sweep, key=lambda x: len(x.get("by_batch", {})))  # richest one
    batches = sorted((int(b) for b in d["by_batch"]), key=int)
    header = ["Batch", "Thr. (img/s)", "Lat.mean (ms)", "Lat.p99 (ms)",
              "Power (W)", "Perf/W"]
    rows, thr, lat, pw = [], [], [], []
    for B in batches:
        b = d["by_batch"][str(B)]
        if "error" in b:
            continue
        power_w = (g(b, "power_mw", "total_mean_mw") or 0) / 1000.0
        rows.append([B, fmt(g(b, "throughput_img_per_s"), 1),
                     fmt(g(b, "latency_ms", "mean")), fmt(g(b, "latency_ms", "p99")),
                     fmt(power_w), fmt(g(b, "perf_per_watt_img_per_s_per_w"))])
        thr.append(g(b, "throughput_img_per_s"))
        lat.append(g(b, "latency_ms", "mean"))
        pw.append(power_w)
    write_csv(os.path.join(outdir, "table_batch_sweep.csv"), header, rows)
    latex_table(os.path.join(outdir, "table_batch_sweep.tex"), header, rows,
                f"Batch-size sweep ({d.get('tag','')}) on Jetson Orin Nano Super.",
                "tab:sweep", colspec="rrrrrr")

    bx = [int(r[0]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(5.2, 3.4))
    ax1.plot(bx, thr, "o-", color="#1f77b4", label="Throughput")
    ax1.set_xlabel("Batch size"); ax1.set_ylabel("Throughput (img/s)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xscale("log", base=2); ax1.set_xticks(bx); ax1.set_xticklabels(bx)
    ax2 = ax1.twinx()
    ax2.plot(bx, lat, "s--", color="#d62728", label="Latency (mean)")
    ax2.set_ylabel("Latency (ms)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Throughput vs. latency across batch size")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_batch_sweep.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


# ----------------------------- T6: deployment characterization -----------------------------
def build_deployment_table(main, outdir):
    header = ["Variant", "Prec.", "Engine load (s)", "Peak RAM (MB)"]
    rows = []
    for v in VARIANT_ORDER:
        for p in PREC_ORDER:
            m = main.get((v, p))
            if not m:
                continue
            eng = m["engine"]
            rows.append([VARIANT_LABEL[v], PREC_LABEL[p],
                         fmt(eng.get("engine_load_s"), 3),
                         g(m["b1"], "ram_used_mb", "peak_mb", default="")])
    write_csv(os.path.join(outdir, "table_deployment.csv"), header, rows)
    latex_table(os.path.join(outdir, "table_deployment.tex"), header, rows,
                "Deployment characterization: engine load time and peak memory.",
                "tab:deploy", colspec="llrr")


# ----------------------------- F2: perf-per-watt bar -----------------------------
def build_ppw_bar(main, outdir):
    xs = np.arange(len(VARIANT_ORDER))
    width = 0.25
    colors = {"fp32": "#8c8c8c", "fp16": "#1f77b4", "int8": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for k, p in enumerate(PREC_ORDER):
        vals = [g(main.get((v, p), {}).get("b1", {}),
                  "perf_per_watt_img_per_s_per_w") for v in VARIANT_ORDER]
        vals = [np.nan if x is None else x for x in vals]
        ax.bar(xs + (k - 1) * width, vals, width, label=PREC_LABEL[p], color=colors[p])
    ax.set_xticks(xs); ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANT_ORDER],
                                          rotation=15, ha="right")
    ax.set_ylabel("Perf-per-watt (img/s/W)")
    ax.set_title("Energy efficiency by precision (batch=1)")
    ax.legend(title="Precision"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_ppw_bar.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


# ----------------------------- F1: Pareto (AUROC vs energy) -----------------------------
def build_pareto(main, outdir):
    vcolor = {"baseline": "#8c8c8c", "logitKD": "#1f77b4",
              "featureKD": "#2ca02c", "featureHardKD": "#d62728"}
    pmark = {"fp32": "o", "fp16": "s", "int8": "^"}
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    plotted = 0
    for v in VARIANT_ORDER:
        for p in PREC_ORDER:
            m = main.get((v, p))
            auroc = AUROC5.get(v, {}).get(p)
            if not m or auroc is None:
                continue
            e = g(m["b1"], "energy_per_image_mj")
            if e is None:
                continue
            ax.scatter(e, auroc, c=vcolor[v], marker=pmark[p], s=70,
                       edgecolors="k", linewidths=0.4, zorder=3)
            plotted += 1
    if TEACHER_AUROC5:
        ax.axhline(TEACHER_AUROC5, ls="--", color="k", lw=1)
        ax.text(0.98, TEACHER_AUROC5, " ViT-B/16 teacher (ceiling)", va="bottom",
                ha="right", transform=ax.get_yaxis_transform(), fontsize=8)
    # legends: variant (color) + precision (marker)
    from matplotlib.lines import Line2D
    vh = [Line2D([0], [0], marker="o", color="w", markerfacecolor=vcolor[v],
                 markeredgecolor="k", label=VARIANT_LABEL[v], markersize=8)
          for v in VARIANT_ORDER]
    ph = [Line2D([0], [0], marker=pmark[p], color="w", markerfacecolor="gray",
                 markeredgecolor="k", label=PREC_LABEL[p], markersize=8)
          for p in PREC_ORDER]
    leg1 = ax.legend(handles=vh, title="Variant", loc="lower right", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=ph, title="Precision", loc="lower left", fontsize=8)
    ax.set_xlabel("Energy per inference (mJ)")
    ax.set_ylabel("AUROC (MACRO-5)")
    ax.set_title("Accuracy–energy Pareto frontier (Jetson Orin Nano Super)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_pareto.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    if plotted == 0:
        print("[!] Pareto empty — fill AUROC5 to populate it.")


# ----------------------------- F5: AUROC vs precision -----------------------------
def build_auroc_vs_precision(outdir):
    vcolor = {"baseline": "#8c8c8c", "logitKD": "#1f77b4",
              "featureKD": "#2ca02c", "featureHardKD": "#d62728"}
    xs = np.arange(len(PREC_ORDER))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    any_pt = False
    for v in VARIANT_ORDER:
        ys = [AUROC5.get(v, {}).get(p) for p in PREC_ORDER]
        xf = [x for x, y in zip(xs, ys) if y is not None]
        yf = [y for y in ys if y is not None]
        if yf:
            any_pt = True
            ax.plot(xf, yf, "o-", color=vcolor[v], label=VARIANT_LABEL[v])
    if TEACHER_AUROC5:
        ax.axhline(TEACHER_AUROC5, ls="--", color="k", lw=1, label="Teacher ceiling")
    ax.set_xticks(xs); ax.set_xticklabels([PREC_LABEL[p] for p in PREC_ORDER])
    ax.set_ylabel("AUROC (MACRO-5)"); ax.set_title("Accuracy vs. precision")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_auroc_vs_precision.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    if not any_pt:
        print("[!] AUROC-vs-precision empty — fill AUROC5.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--bootstrap", default="bootstrap_diffs.csv")
    ap.add_argument("--out", default=None,
                    help="output dir (default: paper_assets/run_<timestamp> — never overwrites)")
    args = ap.parse_args()
    out = args.out or os.path.join(
        "paper_assets", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)

    main_data, sweep = load_all(args.results)
    print(f"[i] loaded {len(main_data)} static engines, {len(sweep)} sweep file(s)")

    build_main_table(main_data, out)
    build_distillation_table(args.bootstrap, out)
    build_precision_table(out)
    build_complexity_table(out)
    build_sweep(sweep, out)
    build_deployment_table(main_data, out)
    build_ppw_bar(main_data, out)
    build_pareto(main_data, out)
    build_auroc_vs_precision(out)
    print(f"[OK] wrote tables + figures to {out}/")


if __name__ == "__main__":
    main()
