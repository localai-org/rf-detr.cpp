#!/usr/bin/env python3
"""Render markdown accuracy tables from benchmarks/results/accuracy_sweep.json.

Reads the JSON produced by `sweep_accuracy.py` and emits two markdown tables
(one for detection variants, one for segmentation variants) to stdout. The
output is the section we drop into BENCHMARK.md.

Usage:
    .venv/bin/python scripts/accuracy_table.py \
        --input benchmarks/results/accuracy_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VARIANT_LABEL = {
    "nano":        "Nano",
    "small":       "Small",
    "base":        "Base",
    "medium":      "Medium",
    "large":       "Large",
    "seg-nano":    "Seg-Nano",
    "seg-small":   "Seg-Small",
    "seg-medium":  "Seg-Medium",
    "seg-large":   "Seg-Large",
    "seg-xlarge":  "Seg-XLarge",
    "seg-2xlarge": "Seg-2XLarge",
}

QUANT_LABEL = {
    "f32":  "F32",
    "f16":  "F16",
    "q8_0": "Q8_0",
    "q4_K": "Q4_K",
}

DET_VARIANTS = ["nano", "small", "base", "medium", "large"]
SEG_VARIANTS = ["seg-nano", "seg-small", "seg-medium",
                "seg-large", "seg-xlarge", "seg-2xlarge"]
VARIANT_ORDER = DET_VARIANTS + SEG_VARIANTS
QUANT_ORDER = ["f32", "f16", "q8_0", "q4_K"]


def cell_by_key(cells: list[dict]) -> dict[tuple[str, str], dict]:
    return {(c["variant"], c["quant"]): c for c in cells}


def render_detection(cells: list[dict]) -> str:
    by = cell_by_key(cells)
    lines = []
    lines.append("### Detection variants")
    lines.append("")
    lines.append("| Variant | Quant | Size (MB) | Recall@0.5 | Recall@0.95 | Max \\|Δscore\\| | Mean \\|Δscore\\| | Extra dets |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for v in DET_VARIANTS:
        for q in QUANT_ORDER:
            c = by.get((v, q))
            if c is None or "error" in c:
                err = c["error"] if c else "missing"
                lines.append(f"| {VARIANT_LABEL[v]} | {QUANT_LABEL[q]} | "
                             f"— | — | — | — | — | _{err}_ |")
                continue
            m = c.get("metrics", {})
            size = c.get("file_size_mb", 0.0)
            lines.append(
                f"| {VARIANT_LABEL[v]} | {QUANT_LABEL[q]} | {size:.1f} | "
                f"{m.get('recall_iou_0.5', 0):.3f} | "
                f"{m.get('recall_iou_0.95', 0):.3f} | "
                f"{m.get('max_abs_score_delta', 0):.4f} | "
                f"{m.get('mean_abs_score_delta', 0):.4f} | "
                f"{m.get('extra_cpp_detections', 0):.2f} |"
            )
    return "\n".join(lines)


def render_segmentation(cells: list[dict]) -> str:
    by = cell_by_key(cells)
    lines = []
    lines.append("### Segmentation variants")
    lines.append("")
    lines.append("| Variant | Quant | Size (MB) | Recall@0.5 | Recall@0.95 | Mean mask IoU | Pixel agreement | Mean \\|Δscore\\| |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for v in SEG_VARIANTS:
        for q in QUANT_ORDER:
            c = by.get((v, q))
            if c is None or "error" in c:
                err = c["error"] if c else "missing"
                lines.append(f"| {VARIANT_LABEL[v]} | {QUANT_LABEL[q]} | "
                             f"— | — | — | — | — | _{err}_ |")
                continue
            m = c.get("metrics", {})
            size = c.get("file_size_mb", 0.0)
            lines.append(
                f"| {VARIANT_LABEL[v]} | {QUANT_LABEL[q]} | {size:.1f} | "
                f"{m.get('recall_iou_0.5', 0):.3f} | "
                f"{m.get('recall_iou_0.95', 0):.3f} | "
                f"{m.get('mean_mask_iou', 0):.4f} | "
                f"{m.get('mean_pixel_agreement', 0):.4f} | "
                f"{m.get('mean_abs_score_delta', 0):.4f} |"
            )
    return "\n".join(lines)


def render_all(data: dict) -> str:
    meta = data.get("meta", {})
    cells = data.get("cells", [])
    n_images = meta.get("n_images", 0)
    thresh = meta.get("threshold", 0.5)
    date = meta.get("date", "")
    rfdetr_v = meta.get("rfdetr_version", "")
    torch_v = meta.get("pytorch_version", "")
    image_names = meta.get("image_names", [])

    out: list[str] = []
    out.append("## Accuracy across the full model matrix")
    out.append("")
    out.append(
        f"All C++ values measured vs PyTorch ground truth on "
        f"{n_images} COCO val images at threshold={thresh}. "
        f"Sweep recorded {date} with rfdetr {rfdetr_v} (torch {torch_v}). "
        f"Greedy 1-1 matching, score-desc, same class, IoU ≥ 0.5 "
        f"(for `Recall@0.5`) or IoU ≥ 0.95 (for `Recall@0.95`). "
        f"Score deltas are absolute, over matched pairs only. "
        f"`Extra dets` is the average number of unmatched C++ detections per image "
        f"(should be 0 in the ideal case). "
        f"For segmentation variants, `Mean mask IoU` and `Pixel agreement` are "
        f"computed over matched-pair binary masks (both at image resolution)."
    )
    out.append("")
    if image_names:
        out.append("Images: " + ", ".join(f"`{n}`" for n in image_names))
        out.append("")
    out.append(render_detection(cells))
    out.append("")
    out.append(render_segmentation(cells))
    out.append("")
    out.append("Raw per-cell + per-image data: "
               "[`benchmarks/results/accuracy_sweep.json`]"
               "(benchmarks/results/accuracy_sweep.json).")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=Path("benchmarks/results/accuracy_sweep.json"))
    args = ap.parse_args()
    with open(args.input, "r") as f:
        data = json.load(f)
    print(render_all(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
