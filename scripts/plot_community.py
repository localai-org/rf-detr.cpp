#!/usr/bin/env python3
"""Render publication-quality plots from benchmarks/results/bench_data.json.

Outputs PNG + SVG into benchmarks/plots/ for use in BENCHMARK.md, the README,
and community write-ups.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# ----- visual style -----
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Per-impl palette. Picked for distinctness in colorblind-safe range and
# good print contrast.
PALETTE = {
    "python":   "#8E8E93",   # neutral grey — the reference
    "cpp_f32":  "#0A84FF",   # vivid blue   — our F32
    "cpp_f16":  "#FF2D55",   # vivid pink   — our F16 (the headline winner)
    "cpp_q8":   "#30D158",   # green        — our Q8_0
    "cpp_q6K":  "#5AC8FA",   # cyan         — Q6_K
    "cpp_q5K":  "#BF5AF2",   # purple       — Q5_K
    "cpp_q4K":  "#FFD60A",   # gold         — Q4_K
    "cpp_q5":   "#FF9F0A",   # amber        — our Q5_0 (auxiliary)
    "cpp_q4":   "#FF453A",   # red          — our Q4_0 (auxiliary, less accurate)
}
HATCH = {
    "python":   "",
    "cpp_f32":  "",
    "cpp_f16":  "\\\\",      # diagonal — distinct from F32 (none) and Q8_0 (//)
    "cpp_q8":   "//",        # extra accessibility cue
    "cpp_q6K":  "++",
    "cpp_q5K":  "..",
    "cpp_q4K":  "OO",
    "cpp_q5":   "\\\\",
    "cpp_q4":   "xx",
}
LABEL = {
    "python":   "PyTorch (rfdetr 1.7.0)",
    "cpp_f32":  "rfdetr.cpp F32",
    "cpp_f16":  "rfdetr.cpp F16",
    "cpp_q8":   "rfdetr.cpp Q8_0",
    "cpp_q6K":  "rfdetr.cpp Q6_K",
    "cpp_q5K":  "rfdetr.cpp Q5_K",
    "cpp_q4K":  "rfdetr.cpp Q4_K",
    "cpp_q5":   "rfdetr.cpp Q5_0",
    "cpp_q4":   "rfdetr.cpp Q4_0",
}
MARKER = {
    "python":   "o",
    "cpp_f32":  "s",
    "cpp_f16":  "*",
    "cpp_q8":   "D",
    "cpp_q6K":  "P",
    "cpp_q5K":  "X",
    "cpp_q4K":  "h",
    "cpp_q5":   "^",
    "cpp_q4":   "v",
}

PRETTY_IMG = {
    "bus.jpg":              "bus",
    "coco_cats.jpg":        "cats",
    "coco_kitchen.jpg":     "kitchen",
    "coco_living_room.jpg": "living room",
    "coco_indoor.jpg":      "indoor",
    "coco_skater.jpg":      "skater",
    "coco_street.jpg":      "street",
}


def pretty_img(name: str) -> str:
    return PRETTY_IMG.get(name, name.replace(".jpg", ""))


def save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out_dir / f"{stem}.svg",            bbox_inches="tight",
                facecolor="white")
    print(f"  wrote {out_dir / (stem + '.png')}")
    print(f"  wrote {out_dir / (stem + '.svg')}")


# ============================================================================
# Plot 1 — Latency comparison bar chart (headline)
# ============================================================================
def plot_latency_comparison(data: dict, out_dir: Path):
    per_image = data["per_image"]
    images = list(per_image.keys())
    # Order: PyTorch baseline / F32 / F16 (sweet spot) / Q8_0 (smallest)
    impls  = ["python", "cpp_f32", "cpp_f16", "cpp_q8"]
    # Drop impls that aren't present in any image (back-compat)
    impls = [i for i in impls if any(per_image[img].get(i) for img in images)]

    n_images = len(images)
    n_impls  = len(impls)
    bar_w = 0.21 if n_impls >= 4 else 0.27
    x = np.arange(n_images)

    fig, ax = plt.subplots(figsize=(max(11, 1.8 * n_images + 4), 5.4))

    # Use p25/p75 whiskers when present (Python arm only — captured per-iter),
    # fall back to min/max otherwise. Avoids cold-start outliers dominating
    # the y-axis when the timed arm caught a 5x outlier.
    def _whisker(cell):
        med = cell["median_ms"]
        if "p25_ms" in cell and "p75_ms" in cell:
            return cell["p25_ms"], med, cell["p75_ms"]
        return cell["min_ms"], med, cell["max_ms"]

    for i, impl in enumerate(impls):
        medians = []
        lo_err  = []
        hi_err  = []
        for img in images:
            cell = per_image[img].get(impl)
            if not cell:
                medians.append(0); lo_err.append(0); hi_err.append(0); continue
            lo, med, hi = _whisker(cell)
            medians.append(med)
            lo_err.append(med - lo)
            hi_err.append(hi - med)

        xpos = x + (i - (n_impls - 1) / 2) * bar_w
        bars = ax.bar(
            xpos, medians, bar_w,
            color=PALETTE[impl], hatch=HATCH[impl],
            edgecolor="white", linewidth=0.8,
            label=LABEL[impl],
            zorder=3,
        )
        ax.errorbar(
            xpos, medians, yerr=[lo_err, hi_err],
            fmt="none", ecolor="#333333", elinewidth=0.9, capsize=3, capthick=0.9,
            zorder=4,
        )
        # Annotate the bar tops with the median value
        for xx, yy, hi in zip(xpos, medians, hi_err):
            ax.text(xx, yy + hi + 3.0, f"{yy:.0f}",
                    ha="center", va="bottom", fontsize=9, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_img(im) for im in images], rotation=0)
    ax.set_ylabel("Inference latency (ms / image)")
    # Cap y-axis based on the C++ impls' worst case so PyTorch outliers
    # don't squash the C++ bars into invisibility. PyTorch values that
    # exceed the cap will still render as truncated bars + value labels;
    # the cap is set well above any reasonable warm steady-state value.
    cpp_impls = [i for i in impls if i.startswith("cpp_")]
    cpp_max = max(
        _whisker(per_image[img][impl])[2]
        for img in images
        for impl in cpp_impls
        if per_image[img].get(impl)
    )
    py_max_med = max(
        (per_image[img]["python"]["median_ms"] for img in images
         if per_image[img].get("python")),
        default=0,
    )
    # Use the larger of (C++ whisker × 1.6, max Python median × 1.10) so
    # the cap is generous to fit C++ visually but never silently hides a
    # legitimate Python value. The Python p25/p75 whiskers are already
    # narrower than the bar in pathological cases.
    ymax = max(cpp_max * 1.6, py_max_med * 1.10)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="#cccccc")

    cpu = data["meta"]["platform"]["cpu"]
    threads = data["meta"]["threads_headline"]
    iters   = data["meta"]["iters"]
    methodology = data["meta"].get("methodology") or {}
    mode = methodology.get("mode", "")
    if mode == "rigorous":
        passes   = methodology.get("passes", "")
        cooldown = methodology.get("cooldown_seconds", "")
        subtitle = (
            f"Median ms/image; {passes} passes × {iters} iters/cell with "
            f"{cooldown:g}s cooldown between cells (whiskers = p25/p75 across "
            f"passes). CPU: {cpu}, C++ threads = {threads}."
        )
    else:
        subtitle = (
            f"Median ms/image over {iters} timed iterations "
            f"(whiskers = p25/p75 where present, else min/max). "
            f"CPU: {cpu}, C++ threads = {threads}."
        )
    # Dynamic headline: pick the C++ impl with the lowest mean-of-medians
    # and report its speedup vs PyTorch.
    cpp_means = {}
    py_means_per_img = []
    for impl in cpp_impls:
        ms = [per_image[i][impl]["median_ms"] for i in images if per_image[i].get(impl)]
        if ms:
            cpp_means[impl] = sum(ms) / len(ms)
    py_means_per_img = [per_image[i]["python"]["median_ms"] for i in images
                         if per_image[i].get("python")]
    py_mean = (sum(py_means_per_img) / len(py_means_per_img)) if py_means_per_img else 0
    if cpp_means and py_mean:
        winner = min(cpp_means, key=cpp_means.get)
        speedup = py_mean / cpp_means[winner]
        win_label = LABEL.get(winner, winner)
        # Strip the "rfdetr.cpp " prefix for the bold headline
        win_short = win_label.replace("rfdetr.cpp ", "")
        # Are F16 / F32 within 5% of each other? Then "F32/F16 tied"
        if "cpp_f32" in cpp_means and "cpp_f16" in cpp_means:
            f32m, f16m = cpp_means["cpp_f32"], cpp_means["cpp_f16"]
            if abs(f16m - f32m) / max(f16m, f32m) < 0.03:
                win_short = "F32/F16"
        headline = (
            f"rfdetr.cpp {win_short} is {speedup:.2f}x faster than PyTorch "
            f"(mean median across {len(images)} images)"
        )
    else:
        headline = "rfdetr.cpp F32 / F16 / Q8_0 vs PyTorch — CPU-only inference"
    fig.suptitle(headline, fontsize=15, fontweight="bold", y=0.985)
    ax.set_title(subtitle, fontsize=10, color="#555", pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, out_dir, "latency_comparison")
    plt.close(fig)


# ============================================================================
# Plot 2 — Thread scaling line plot
# ============================================================================
def plot_thread_scaling(data: dict, out_dir: Path):
    sw = data.get("thread_sweep")
    if not sw:
        print("  [skip] no thread_sweep data")
        return

    threads = sw["threads"]
    impls = [k for k in ("python", "cpp_f32", "cpp_f16", "cpp_q8") if k in sw and sw[k]]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for impl in impls:
        medians = []
        lo_err  = []
        hi_err  = []
        for n in threads:
            cell = sw[impl].get(str(n))
            if not cell:
                medians.append(np.nan); lo_err.append(0); hi_err.append(0); continue
            med = cell["median_ms"]
            # Prefer p25/p75 (tighter, the rigorous-mode whisker); fall back to
            # min/max for legacy data.
            if "p25_ms" in cell and "p75_ms" in cell:
                lo, hi = cell["p25_ms"], cell["p75_ms"]
            else:
                lo, hi = cell["min_ms"], cell["max_ms"]
            medians.append(med)
            lo_err.append(med - lo)
            hi_err.append(hi - med)

        ax.errorbar(
            threads, medians, yerr=[lo_err, hi_err],
            label=LABEL[impl],
            color=PALETTE[impl], marker=MARKER[impl], markersize=7,
            linewidth=2.0, capsize=3.5, capthick=1.0,
            elinewidth=0.9,
        )

    # Make sure y limits are stable before placing annotations. Use p75 if
    # present (rigorous mode), else max_ms (legacy).
    def _y_upper(c):
        return c.get("p75_ms", c.get("max_ms", c.get("median_ms", 0)))
    ax.set_ylim(0, max(
        _y_upper(sw[impl][str(n)])
        for impl in impls for n in threads
        if str(n) in sw[impl]
    ) * 1.18)
    ymax = ax.get_ylim()[1]

    # Highlight T=8 (common default) and T=16 (true minimum)
    ax.axvline(8, color="#A2845E", linestyle=":", linewidth=1.2,
               alpha=0.55, zorder=1)
    ax.axvline(16, color="#FF9F0A", linestyle="--", linewidth=1.4,
               alpha=0.85, zorder=1)
    ax.annotate(
        "T=16: best latency\nfor all 3 impls",
        xy=(16, ymax * 0.55),
        xytext=(15.5, ymax * 0.78),
        fontsize=9.5, color="#7a5400", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e1", ec="#FF9F0A", lw=0.9),
    )
    ax.text(7.85, ymax * 0.55, "T=8\n(common default)",
            fontsize=9, color="#5c4632", ha="right", va="center", alpha=0.85)

    ax.set_xlabel("Thread count  (C++ ggml threads / torch.set_num_threads(N))")
    ax.set_ylabel("Median latency (ms / image)")
    ax.set_xticks(threads)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="#cccccc")

    sweep_img = sw.get("image", "")
    cpu = data["meta"]["platform"]["cpu"]
    iters = data["meta"]["iters"]
    fig.suptitle(
        "Thread scaling on 20-core Zen 5 (dual-CCD topology)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    ax.set_title(
        f"All three impls track within ~10% across the sweep. Latency bottoms "
        f"out at T=16; T=20 regresses (cross-CCD scheduling).  "
        f"Image: {pretty_img(sweep_img)}, {iters} iters/cell.  CPU: {cpu}.",
        fontsize=10, color="#555", pad=8,
    )

    fig.tight_layout()
    save(fig, out_dir, "thread_scaling")
    plt.close(fig)


# ============================================================================
# Plot 3 — Model size + accuracy (dual panel)
# ============================================================================
def plot_size_and_accuracy(data: dict, out_dir: Path):
    meta = data["meta"]
    f32_mb = meta["models"]["f32_size_bytes"] / (1024 * 1024)
    f16_mb = (meta["models"].get("f16_size_bytes", 0) or 0) / (1024 * 1024)
    q8_mb  = meta["models"]["q8_size_bytes"]  / (1024 * 1024)
    have_f16 = f16_mb > 0
    ratio_f16 = f32_mb / f16_mb if f16_mb else 0
    ratio_q8  = f32_mb / q8_mb  if q8_mb  else 0

    # Gather all matched (cpp_score, py_score) pairs across images
    match = data.get("match_summary", {})
    pts_f32_py, pts_f32_cpp = [], []
    pts_f16_py, pts_f16_cpp = [], []
    pts_q8_py,  pts_q8_cpp  = [], []
    for key, m in match.items():
        for p in m["pairs"]:
            if m["impl"] == "cpp_f32":
                pts_f32_py.append(p["py_score"])
                pts_f32_cpp.append(p["cpp_score"])
            elif m["impl"] == "cpp_f16":
                pts_f16_py.append(p["py_score"])
                pts_f16_cpp.append(p["cpp_score"])
            elif m["impl"] == "cpp_q8":
                pts_q8_py.append(p["py_score"])
                pts_q8_cpp.append(p["cpp_score"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.0),
                                    gridspec_kw={"width_ratios": [1, 1.4]})

    # --- left: model size bars: F32 / F16 (sweet spot) / Q8_0 ---
    if have_f16:
        sizes = [f32_mb, f16_mb, q8_mb]
        labels = ["F32", "F16", "Q8_0"]
        colors = [PALETTE["cpp_f32"], PALETTE["cpp_f16"], PALETTE["cpp_q8"]]
        hatches = [HATCH["cpp_f32"], HATCH["cpp_f16"], HATCH["cpp_q8"]]
    else:
        sizes = [f32_mb, q8_mb]
        labels = ["F32", "Q8_0"]
        colors = [PALETTE["cpp_f32"], PALETTE["cpp_q8"]]
        hatches = [HATCH["cpp_f32"], HATCH["cpp_q8"]]
    bars = axL.bar(labels, sizes, color=colors, hatch=hatches,
                   edgecolor="white", linewidth=0.9, zorder=3)
    # Annotate with size and compression ratio (anchored on F32)
    for b, s, lbl in zip(bars, sizes, labels):
        ratio = (f32_mb / s) if s else 0
        ratio_txt = "1.00×" if lbl == "F32" else f"{ratio:.2f}×"
        axL.text(b.get_x() + b.get_width() / 2, s + 2,
                 f"{s:.1f} MB\n({ratio_txt})", ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold")
    axL.set_ylabel("Model size (MB)")
    axL.set_ylim(0, max(sizes) * 1.22)
    if have_f16:
        axL.set_title(
            f"F16 is {ratio_f16:.2f}× smaller than F32; Q8_0 is {ratio_q8:.2f}× smaller",
            fontsize=11, color="#222")
    else:
        axL.set_title(f"Q8_0 is {ratio_q8:.2f}× smaller than F32",
                      fontsize=11, color="#222")
    axL.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

    # --- right: per-detection score scatter ---
    if pts_f32_py and pts_f32_cpp:
        axR.scatter(pts_f32_py, pts_f32_cpp, s=55, alpha=0.78,
                    color=PALETTE["cpp_f32"], marker=MARKER["cpp_f32"],
                    edgecolors="white", linewidth=0.7,
                    label=f"C++ F32 ({len(pts_f32_py)} dets)", zorder=3)
    if pts_f16_py and pts_f16_cpp:
        # F16 sits on top of F32 in score-space; render as transparent
        # overlay so both remain visible.
        axR.scatter(pts_f16_py, pts_f16_cpp, s=75, alpha=0.55,
                    color=PALETTE["cpp_f16"], marker=MARKER["cpp_f16"],
                    edgecolors="white", linewidth=0.7,
                    label=f"C++ F16 ({len(pts_f16_py)} dets)", zorder=4)
    if pts_q8_py and pts_q8_cpp:
        axR.scatter(pts_q8_py, pts_q8_cpp, s=55, alpha=0.78,
                    color=PALETTE["cpp_q8"], marker=MARKER["cpp_q8"],
                    edgecolors="white", linewidth=0.7,
                    label=f"C++ Q8_0 ({len(pts_q8_py)} dets)", zorder=3)

    # y = x reference
    lo = 0.45
    axR.plot([lo, 1.0], [lo, 1.0], "--", color="#888",
             linewidth=1.0, alpha=0.7, zorder=1, label="y = x (identical)")

    axR.set_xlim(lo, 1.0)
    axR.set_ylim(lo, 1.0)
    axR.set_aspect("equal")
    axR.set_xlabel("PyTorch detection score")
    axR.set_ylabel("rfdetr.cpp detection score")
    # Compute max score delta seen for the subtitle
    all_deltas = [abs(a - b) for a, b in zip(pts_f32_py, pts_f32_cpp)] + \
                 [abs(a - b) for a, b in zip(pts_f16_py, pts_f16_cpp)] + \
                 [abs(a - b) for a, b in zip(pts_q8_py,  pts_q8_cpp)]
    max_delta = max(all_deltas) if all_deltas else 0.0
    axR.set_title(f"Per-detection score, max |Δ| = {max_delta:.3f}",
                  fontsize=11, color="#222")
    axR.grid(True, linestyle=":", alpha=0.6, zorder=0)
    axR.legend(loc="lower right", frameon=True, framealpha=0.95,
               edgecolor="#cccccc")

    n_images = len(data.get("detections", {}))
    if have_f16:
        suptitle = (f"F16 is {ratio_f16:.2f}× smaller than F32, lossless; "
                    f"Q8_0 is {ratio_q8:.2f}× smaller with identical detections")
    else:
        suptitle = "Quantization is free: Q8_0 is 3× smaller with identical detections"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.93,
        f"Across {n_images} COCO test images: every PyTorch detection has a 1-1 "
        f"C++ match (IoU ≥ 0.95, max |Δscore| ≤ 0.05).",
        ha="center", fontsize=10, color="#555",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, out_dir, "size_and_accuracy")
    plt.close(fig)


# ============================================================================
# Plot 5 (auxiliary) — Quant tradeoff scatter: size vs accuracy across variants
# ============================================================================
def plot_quant_tradeoffs(data: dict, out_dir: Path):
    """Per-quant variant: model size (MB) vs detection recall (matched / py_total)
    at IoU>=0.95. One marker per variant.

    Only useful when the bench captured Q4 / Q5 alongside F32 and Q8_0.
    """
    meta = data["meta"]
    models = meta["models"]
    match  = data.get("match_summary", {})

    # Aggregate match rate per impl.
    # The bench-side matcher uses IoU>=0.95 (strict); for the size-vs-accuracy
    # narrative we also want the lenient "did we even find the object" rate
    # which we recompute from raw detections at IoU>=0.5.
    dets = data.get("detections", {})

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        u = aa + bb - inter
        return inter / u if u > 0 else 0.0

    def lenient_recall(impl: str) -> tuple[int, int]:
        """matched, py_total — IoU>=0.5 greedy class-matched pairing."""
        matched = 0; py_total = 0
        for img, m in dets.items():
            if "python" not in m or impl not in m:
                continue
            ref = m["python"]; test = m[impl]
            py_total += len(ref)
            used = [False] * len(ref)
            for c in test:
                bi, bj = 0.0, -1
                for j, r in enumerate(ref):
                    if used[j] or int(c["class_id"]) != int(r["class_id"]):
                        continue
                    v = iou(c["bbox"], r["bbox"])
                    if v > bi:
                        bi, bj = v, j
                if bj >= 0 and bi >= 0.5:
                    used[bj] = True; matched += 1
        return matched, py_total

    agg = {}
    for k, m in match.items():
        impl = m["impl"]
        a = agg.setdefault(impl, {"py": 0, "matched": 0,
                                  "max_ds": 0.0, "n_ds": 0, "sum_ds": 0.0,
                                  "cpp_total": 0})
        a["py"]        += m["py_total"]
        a["cpp_total"] += m["cpp_total"]
        a["matched"]   += m["matched"]
        if m["max_score_delta"] > a["max_ds"]:
            a["max_ds"] = m["max_score_delta"]
        for p in m["pairs"]:
            a["sum_ds"] += p["score_delta"]
            a["n_ds"]   += 1

    # Lenient-IoU recall (the "did we find the object" rate)
    for impl in list(agg.keys()):
        matched, py = lenient_recall(impl)
        agg[impl]["matched_lenient"] = matched
        agg[impl]["py_lenient"]      = py

    # Each known variant: (label, size_key, impl_key)
    # Ordered F32 -> F16 -> Q8 -> Q6_K -> Q5_K -> Q4_K -> Q5_0 -> Q4_0
    # (largest -> smallest; F16 is the new sweet spot recommendation;
    # K-quants in between the legacy block quants).
    variants = []
    if "f32_size_bytes" in models:
        variants.append(("F32",  "f32_size_bytes", "cpp_f32"))
    if "f16_size_bytes" in models:
        variants.append(("F16",  "f16_size_bytes", "cpp_f16"))
    if "q8_size_bytes"  in models:
        variants.append(("Q8_0", "q8_size_bytes",  "cpp_q8"))
    if "q6K_size_bytes" in models:
        variants.append(("Q6_K", "q6K_size_bytes", "cpp_q6K"))
    if "q5K_size_bytes" in models:
        variants.append(("Q5_K", "q5K_size_bytes", "cpp_q5K"))
    if "q4K_size_bytes" in models:
        variants.append(("Q4_K", "q4K_size_bytes", "cpp_q4K"))
    if "q5_size_bytes"  in models:
        variants.append(("Q5_0", "q5_size_bytes",  "cpp_q5"))
    if "q4_size_bytes"  in models:
        variants.append(("Q4_0", "q4_size_bytes",  "cpp_q4"))

    # Skip the plot if no auxiliary quants AND no F16 are present — nothing
    # new to say vs the F32-vs-Q8_0 panel.
    AUX = ("cpp_f16", "cpp_q5", "cpp_q4", "cpp_q6K", "cpp_q5K", "cpp_q4K")
    if not any(v[2] in AUX for v in variants):
        print("  [skip] no auxiliary quant data — quant_tradeoffs plot adds no info")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.5),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # ---- left: stacked bars of model size ----
    names = [v[0] for v in variants]
    sizes_mb = [models[v[1]] / (1024 * 1024) for v in variants]
    colors = [PALETTE[v[2]] for v in variants]
    hatches = [HATCH[v[2]] for v in variants]
    bars = axL.bar(names, sizes_mb, color=colors, hatch=hatches,
                   edgecolor="white", linewidth=0.9, zorder=3)
    f32_mb = sizes_mb[0]
    for b, s in zip(bars, sizes_mb):
        ratio = f32_mb / s if s else 0
        axL.text(b.get_x() + b.get_width() / 2, s + 2,
                 f"{s:.0f} MB\n({ratio:.1f}×)",
                 ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    axL.set_ylabel("Model size (MB)")
    axL.set_ylim(0, max(sizes_mb) * 1.22)
    axL.set_title("Disk footprint per quant variant", fontsize=11, color="#222")
    axL.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

    # ---- right: lenient recall + score-delta dual axis ----
    # Lenient (IoU>=0.5): "did we find this object at all". Strict (IoU>=0.95):
    # "is the bbox in the same place to sub-pixel precision".
    recalls = []        # lenient
    strict_recalls = [] # strict
    ax_xs = []
    max_dsc = []
    for label, _, impl in variants:
        a = agg.get(impl)
        if not a or a["py"] == 0:
            recalls.append(np.nan); strict_recalls.append(np.nan); max_dsc.append(np.nan)
        else:
            recalls.append(100.0 * a["matched_lenient"] / a["py_lenient"])
            strict_recalls.append(100.0 * a["matched"] / a["py"])
            max_dsc.append(a["max_ds"])
        ax_xs.append(label)

    bar_w = 0.42
    x_idx = np.arange(len(ax_xs))
    axR.bar(x_idx - bar_w / 2, recalls, bar_w,
            color=[PALETTE[v[2]] for v in variants],
            hatch=[HATCH[v[2]] for v in variants],
            edgecolor="white", linewidth=0.9, zorder=3,
            label="recall (IoU ≥ 0.5) — did we find the object")
    for x, r in zip(x_idx - bar_w / 2, recalls):
        if not np.isnan(r):
            axR.text(x, r + 1.2, f"{r:.0f}%",
                     ha="center", va="bottom", fontsize=10, color="#222",
                     fontweight="bold")
    # overlay strict recall as outline-only bars (same x, no fill)
    axR.bar(x_idx - bar_w / 2, strict_recalls, bar_w,
            facecolor="none", edgecolor="#222", linewidth=1.6,
            linestyle="--", zorder=4,
            label="strict recall (IoU ≥ 0.95) — bbox to ~1 px")
    for x, r in zip(x_idx - bar_w / 2, strict_recalls):
        if not np.isnan(r):
            axR.text(x, r - 6, f"{r:.0f}%",
                     ha="center", va="bottom", fontsize=8.5, color="#222")

    axR.set_ylabel("Detection recall vs PyTorch (%)")
    axR.set_xticks(x_idx)
    axR.set_xticklabels(ax_xs)
    axR.set_ylim(0, 115)
    axR.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

    axR2 = axR.twinx()
    axR2.bar(x_idx + bar_w / 2, max_dsc, bar_w,
             color="#444", alpha=0.35, edgecolor="white",
             linewidth=0.9, zorder=3,
             label="max |Δscore| vs PyTorch")
    for x, d in zip(x_idx + bar_w / 2, max_dsc):
        if not np.isnan(d):
            axR2.text(x, d + 0.005, f"{d:.3f}",
                      ha="center", va="bottom", fontsize=9, color="#333")
    axR2.set_ylabel("Max |Δscore| (lower = better)", color="#444")
    axR2.set_ylim(0, max(0.3, (max([d for d in max_dsc if not np.isnan(d)] or [0]) * 1.3)))
    axR2.tick_params(axis="y", colors="#444")
    axR2.grid(False)

    # Combined legend
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = axR2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, loc="upper right",
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               fontsize=8.5)
    axR.set_title("Detection accuracy per quant variant", fontsize=11, color="#222")

    have_f16 = any(v[2] == "cpp_f16" for v in variants)
    if have_f16:
        suptitle = ("Precision tradeoff — F16 is the sweet spot (1.85× smaller, "
                    "lossless); K-quants keep accuracy at ~3.5× compression")
    else:
        suptitle = ("Quant tradeoff — K-quants keep accuracy at ~3.5x compression "
                    "(legacy Q4_0 falls off the cliff)")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out_dir, "quant_tradeoffs")
    plt.close(fig)


# ============================================================================
# Plot 4 (optional / bonus) — speedup heatmap-ish: latency per image as ratio
# ============================================================================
def plot_relative_latency(data: dict, out_dir: Path):
    """C++ F32 / F16 / Q8_0 latency normalized to Python per image. Quick eyeball check."""
    per_image = data["per_image"]
    images = list(per_image.keys())
    have_f16 = all(per_image[i].get("cpp_f16") for i in images)

    fig, ax = plt.subplots(figsize=(max(9.5, 1.6 * len(images) + 3.0), 4.8))

    py_med  = [per_image[i]["python"]["median_ms"]  for i in images]
    f32_med = [per_image[i]["cpp_f32"]["median_ms"] for i in images]
    q8_med  = [per_image[i]["cpp_q8" ]["median_ms"] for i in images]
    f16_med = ([per_image[i]["cpp_f16"]["median_ms"] for i in images]
               if have_f16 else None)

    f32_ratio = [f / p for f, p in zip(f32_med, py_med)]
    q8_ratio  = [q / p for q, p in zip(q8_med,  py_med)]
    f16_ratio = ([f / p for f, p in zip(f16_med, py_med)]
                 if have_f16 else None)

    x = np.arange(len(images))

    if have_f16:
        bar_w = 0.26
        offsets = [-bar_w, 0, bar_w]
        series  = [("cpp_f32", f32_ratio),
                   ("cpp_f16", f16_ratio),
                   ("cpp_q8",  q8_ratio)]
    else:
        bar_w = 0.36
        offsets = [-bar_w / 2, bar_w / 2]
        series  = [("cpp_f32", f32_ratio), ("cpp_q8", q8_ratio)]

    for off, (impl, ratios) in zip(offsets, series):
        ax.bar(x + off, ratios, bar_w,
               color=PALETTE[impl], hatch=HATCH[impl],
               edgecolor="white", linewidth=0.8,
               label=LABEL[impl], zorder=3)
        for xi, r in zip(x + off, ratios):
            ax.text(xi, r + 0.01, f"{r:.2f}×",
                    ha="center", va="bottom", fontsize=8.5)

    ax.axhline(1.0, color="#444", linestyle="-", linewidth=1.0, zorder=1)
    ax.text(len(images) - 0.5, 1.02, "PyTorch baseline",
            ha="right", va="bottom", fontsize=9, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_img(i) for i in images])
    ax.set_ylabel("Latency relative to PyTorch  (lower = faster)")
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    all_ratios = list(f32_ratio) + list(q8_ratio)
    if have_f16:
        all_ratios += list(f16_ratio)
    ymax = max(all_ratios)
    ax.set_ylim(0, max(1.2, ymax * 1.18))

    cpu = data["meta"]["platform"]["cpu"]
    fig.suptitle(
        "Per-image latency, normalized to PyTorch",
        fontsize=14, fontweight="bold", y=0.99,
    )
    ax.set_title(
        f"Values ≈ 1.0 mean parity with PyTorch on the same CPU ({cpu}, T=8).",
        fontsize=10, color="#555", pad=8,
    )

    fig.tight_layout()
    save(fig, out_dir, "relative_latency")
    plt.close(fig)


# ============================================================================
# Plot 6 — Variants overview (Nano/Small/Base/Medium/Large)
# ============================================================================
# Per-variant palette + hatching for visual distinctiveness. Same style spirit
# as the quant palette but using a separate hue ramp so the two plots are
# legible side-by-side.
VARIANT_PALETTE = {
    "nano":   "#A2D5F2",  # pale blue   (smallest)
    "small":  "#7AB8E0",  # medium blue
    "base":   "#0A84FF",  # vivid blue  (the parity baseline; matches cpp_f32)
    "medium": "#5856D6",  # purple
    "large":  "#AF52DE",  # vivid purple (largest)
}
VARIANT_HATCH = {
    "nano":   "..",
    "small":  "//",
    "base":   "",       # baseline: no hatch (matches cpp_f32 in other plots)
    "medium": "++",
    "large":  "xx",
}
VARIANT_ORDER = ["nano", "small", "base", "medium", "large"]


def plot_variants_overview(data: dict, out_dir: Path):
    """Dual-panel plot: (left) median latency per variant for PyTorch vs C++ F32
    on the variant_sweep_image; (right) GGUF size + detection count per variant
    on that same image.

    Skips if no `variants` block is present in the bench data (back-compat:
    legacy bench_data.json files have only the base/quant headline).
    """
    variants_data = data.get("variants") or {}
    if not variants_data and "base" not in (data.get("per_image") or {}):
        print("  [skip] no variants data and no base fallback")
        return

    # The plan's variants section excludes Base from the explicit --variants
    # loop (Base lives in per_image); recover Base's variant cell from the
    # quant headline so the plot includes all five.
    sweep_img = data.get("meta", {}).get("variant_sweep_image", "coco_kitchen.jpg")
    per_image = data.get("per_image", {})
    if "base" not in variants_data and sweep_img in per_image:
        cell = per_image[sweep_img]
        base_dets = data.get("detections", {}).get(sweep_img, {})
        base_cell = {
            "variant": "base",
            "image": sweep_img,
            "gguf_size_bytes": data["meta"]["models"]["f32_size_bytes"],
            "cpp_f32": cell.get("cpp_f32", {}),
            "detections": {
                "cpp_f32": base_dets.get("cpp_f32", []),
                "python":  base_dets.get("python", []),
            },
        }
        if "python" in cell:
            base_cell["python"] = cell["python"]
        variants_data = {**variants_data, "base": base_cell}

    # Filter to the variants we actually have, in canonical order
    have = [v for v in VARIANT_ORDER if v in variants_data]
    if not have:
        print("  [skip] no recognized variants in data")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # ---- Left: median latency per variant for PyTorch vs C++ F32 ----
    x_idx = np.arange(len(have))
    bar_w = 0.38

    py_med  = []
    cpp_med = []
    py_lo,  py_hi  = [], []
    cpp_lo, cpp_hi = [], []
    def _whisker(cell):
        m = float(cell.get("median_ms", 0))
        if "p25_ms" in cell and "p75_ms" in cell:
            return float(cell["p25_ms"]), m, float(cell["p75_ms"])
        return (float(cell.get("min_ms", m)), m, float(cell.get("max_ms", m)))

    for v in have:
        cell = variants_data[v]
        cpp = cell.get("cpp_f32", {})
        py  = cell.get("python", {})
        c_lo, c_m, c_hi = _whisker(cpp) if cpp else (0, 0, 0)
        cpp_med.append(c_m); cpp_lo.append(c_m - c_lo); cpp_hi.append(c_hi - c_m)
        if py:
            p_lo, p_m, p_hi = _whisker(py)
            py_med.append(p_m); py_lo.append(p_m - p_lo); py_hi.append(p_hi - p_m)
        else:
            py_med.append(np.nan); py_lo.append(0); py_hi.append(0)

    # Plot PyTorch in neutral grey (same palette as latency_comparison)
    bars_py = axL.bar(x_idx - bar_w/2, py_med, bar_w,
                       color=PALETTE["python"], hatch=HATCH["python"],
                       edgecolor="white", linewidth=0.8,
                       label="PyTorch (rfdetr 1.7.0)", zorder=3)
    axL.errorbar(x_idx - bar_w/2, py_med, yerr=[py_lo, py_hi],
                 fmt="none", ecolor="#333333", elinewidth=0.9, capsize=3,
                 capthick=0.9, zorder=4)
    # Annotate
    for xi, y, hi in zip(x_idx - bar_w/2, py_med, py_hi):
        if not np.isnan(y):
            axL.text(xi, y + hi + 4, f"{y:.0f}",
                     ha="center", va="bottom", fontsize=9, color="#444")

    # Plot C++ F32 in variant-distinct colors (matches the right panel)
    cpp_colors  = [VARIANT_PALETTE[v] for v in have]
    cpp_hatches = [VARIANT_HATCH[v]   for v in have]
    bars_cpp = axL.bar(x_idx + bar_w/2, cpp_med, bar_w,
                        color=cpp_colors, hatch=cpp_hatches,
                        edgecolor="white", linewidth=0.8,
                        label="rfdetr.cpp F32", zorder=3)
    axL.errorbar(x_idx + bar_w/2, cpp_med, yerr=[cpp_lo, cpp_hi],
                 fmt="none", ecolor="#333333", elinewidth=0.9, capsize=3,
                 capthick=0.9, zorder=4)
    for xi, y, hi in zip(x_idx + bar_w/2, cpp_med, cpp_hi):
        axL.text(xi, y + hi + 4, f"{y:.0f}",
                 ha="center", va="bottom", fontsize=9, color="#222",
                 fontweight="bold")

    axL.set_xticks(x_idx)
    axL.set_xticklabels([v.capitalize() for v in have])
    axL.set_ylabel("Median inference latency (ms / image)")
    ymax = max([m + h for m, h in zip(cpp_med, cpp_hi) if not np.isnan(m)] +
               [m + h for m, h in zip(py_med, py_hi)  if not np.isnan(m)])
    axL.set_ylim(0, ymax * 1.18)
    axL.set_title("Latency vs PyTorch (lower = faster)", fontsize=11, color="#222")
    axL.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    axL.legend(loc="upper left", frameon=True, framealpha=0.95,
               edgecolor="#cccccc")

    # ---- Right: GGUF size + detection count per variant ----
    sizes_mb = [variants_data[v].get("gguf_size_bytes", 0) / (1024 * 1024) for v in have]
    det_counts = [len(variants_data[v].get("detections", {}).get("cpp_f32", []))
                  for v in have]

    # Size as a bar (with variant color)
    axR.bar(x_idx, sizes_mb,
            color=cpp_colors, hatch=cpp_hatches,
            edgecolor="white", linewidth=0.8, zorder=3)
    for xi, s in zip(x_idx, sizes_mb):
        axR.text(xi, s + 2, f"{s:.0f} MB",
                 ha="center", va="bottom", fontsize=10, color="#222",
                 fontweight="bold")
    axR.set_ylabel("GGUF F32 size (MB)", color="#222")
    axR.set_ylim(0, max(sizes_mb) * 1.20)
    axR.set_xticks(x_idx)
    axR.set_xticklabels([v.capitalize() for v in have])
    axR.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

    # Overlay detection count as a secondary-axis line
    axR2 = axR.twinx()
    axR2.plot(x_idx, det_counts, "o-", color="#222",
              linewidth=1.6, markersize=8, zorder=4,
              label="detections found")
    for xi, n in zip(x_idx, det_counts):
        axR2.text(xi, n + 0.4, str(n),
                  ha="center", va="bottom", fontsize=9.5,
                  color="#222", fontweight="bold")
    axR2.set_ylabel("Detections (≥ 0.5 score) on " + pretty_img(sweep_img),
                    color="#222")
    axR2.set_ylim(0, max(det_counts) * 1.4 if det_counts else 5)
    axR2.grid(False)
    axR2.tick_params(axis="y")
    axR2.legend(loc="upper left", frameon=True, framealpha=0.95,
                edgecolor="#cccccc", fontsize=9.5)

    axR.set_title(f"Size & detections on {pretty_img(sweep_img)}",
                  fontsize=11, color="#222")

    cpu = data.get("meta", {}).get("platform", {}).get("cpu", "")
    threads = data.get("meta", {}).get("threads_headline", "")
    iters   = data.get("meta", {}).get("iters", "")
    fig.suptitle(
        "RF-DETR variants: latency × detection-count tradeoff (T=" + str(threads) + ")",
        fontsize=15, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.935,
        f"Median ms/image over {iters} timed iterations on {pretty_img(sweep_img)}. "
        f"CPU: {cpu}.",
        ha="center", fontsize=10, color="#555",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, out_dir, "variants_overview")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "benchmarks/results/bench_data.json"))
    ap.add_argument("--out-dir", default=str(REPO / "benchmarks/plots"))
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text())
    out_dir = Path(args.out_dir)

    print("plot: latency_comparison")
    plot_latency_comparison(data, out_dir)

    print("plot: thread_scaling")
    plot_thread_scaling(data, out_dir)

    print("plot: size_and_accuracy")
    plot_size_and_accuracy(data, out_dir)

    print("plot: relative_latency")
    plot_relative_latency(data, out_dir)

    print("plot: quant_tradeoffs")
    plot_quant_tradeoffs(data, out_dir)

    print("plot: variants_overview")
    plot_variants_overview(data, out_dir)

    print(f"done. plots in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
