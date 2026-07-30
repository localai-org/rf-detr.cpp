#!/usr/bin/env python3
"""Full accuracy + mask-quality sweep across the 32-model matrix.

Compares C++ (rfdetr-cli detect) output against the PyTorch rfdetr reference
for every (variant, quant) pair, over a fixed set of test images. Produces
per-cell metrics (recall@IoU≥{0.5,0.95}, score delta, extra-detection count;
plus mask IoU + pixel agreement for seg variants) and writes a single JSON
blob suitable for downstream report generation (scripts/accuracy_table.py).

Strategy:
- Group cells by variant. Load the PyTorch model once per variant, run all
  images, cache the resulting ground-truth detections (and masks for seg).
  Then iterate quants and run the C++ CLI for each (variant, quant, image).
- Greedy 1-1 matching: sort PyTorch dets by score desc, for each find the
  best-IoU C++ det of the same class that isn't already taken; assign at
  IoU ≥ threshold (0.5 / 0.95 reported).
- Mask metrics: PyTorch returns a `(N, H, W)` bool array at image size; C++
  writes per-detection 8-bit PNGs (`det_NNN_classCC_scoreSS.png`, 0/255)
  also at image size. We pair via the same greedy match used for bboxes
  (matched at IoU ≥ 0.5), then compute pixel IoU and exact-pixel agreement
  on the binary masks.

Errors per cell are caught and recorded; the sweep never crashes the whole
run because of one bad cell.

Usage:
    .venv/bin/python scripts/sweep_accuracy.py \
        --output benchmarks/results/accuracy_sweep.json \
        --images benchmarks/images/*.jpg \
        --threshold 0.5 \
        --models-dir models
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# Force CPU before importing torch / rfdetr.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[1]

# Variant -> (rfdetr Python class name, gguf file stem, is_seg)
VARIANTS: list[tuple[str, str, str, bool]] = [
    # variant_key,    rfdetr class,        gguf_stem,        is_seg
    ("nano",          "RFDETRNano",        "rfdetr-nano",        False),
    ("small",         "RFDETRSmall",       "rfdetr-small",       False),
    ("base",          "RFDETRBase",        "rfdetr-base",        False),
    ("medium",        "RFDETRMedium",      "rfdetr-medium",      False),
    ("large",         "RFDETRLarge",       "rfdetr-large",       False),
    ("seg-nano",      "RFDETRSegNano",     "rfdetr-seg-nano",    True),
    ("seg-small",     "RFDETRSegSmall",    "rfdetr-seg-small",   True),
    ("seg-medium",    "RFDETRSegMedium",   "rfdetr-seg-medium",  True),
    ("seg-large",     "RFDETRSegLarge",    "rfdetr-seg-large",   True),
    ("seg-xlarge",    "RFDETRSegXLarge",   "rfdetr-seg-xlarge",  True),
    ("seg-2xlarge",   "RFDETRSeg2XLarge",  "rfdetr-seg-2xlarge", True),
]

QUANTS = ["f32", "f16", "q8_0", "q4_K"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def greedy_match(py_dets: list[dict], cpp_dets: list[dict],
                 iou_thresh: float) -> list[tuple[int, int, float]]:
    """Greedy match PyTorch dets -> C++ dets at IoU >= iou_thresh, same class.

    Returns list of (py_idx, cpp_idx, iou_value). PyTorch dets are processed
    in score-desc order, claiming the highest-IoU unclaimed C++ det.
    """
    cpp_used = [False] * len(cpp_dets)
    pairs: list[tuple[int, int, float]] = []
    order = sorted(range(len(py_dets)), key=lambda i: -py_dets[i]["score"])
    for pi in order:
        p = py_dets[pi]
        best_j = -1
        best_v = 0.0
        for j, c in enumerate(cpp_dets):
            if cpp_used[j]:
                continue
            if int(c["class_id"]) != int(p["class_id"]):
                continue
            v = iou(p["bbox"], c["bbox"])
            if v > best_v:
                best_v = v
                best_j = j
        if best_j >= 0 and best_v >= iou_thresh:
            cpp_used[best_j] = True
            pairs.append((pi, best_j, best_v))
    return pairs


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def load_cpp_mask(mask_path: Path) -> "np.ndarray":
    """Load a C++ mask PNG (8-bit grayscale, 0/255) as a bool array (H, W)."""
    import numpy as np
    from PIL import Image
    im = Image.open(mask_path)
    if im.mode != "L":
        im = im.convert("L")
    return np.array(im) > 127  # (H, W) bool


def find_cpp_mask(masks_dir: Path, idx: int) -> Path | None:
    """Resolve the C++ mask PNG for detection index `idx`."""
    cands = sorted(masks_dir.glob(f"det_{idx:03d}_*.png"))
    return cands[0] if cands else None


def mask_iou(a, b) -> float:
    """IoU between two binary masks; resizes b to a's shape if mismatched."""
    import numpy as np
    from PIL import Image
    if a.shape != b.shape:
        # Resize b to a's shape using nearest-neighbor (mask domain).
        b_img = Image.fromarray((b.astype("uint8") * 255))
        b_img = b_img.resize((a.shape[1], a.shape[0]), Image.NEAREST)
        b = np.array(b_img) > 127
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def mask_pixel_agreement(a, b) -> float:
    """Fraction of pixels equal between two binary masks (after resize)."""
    import numpy as np
    from PIL import Image
    if a.shape != b.shape:
        b_img = Image.fromarray((b.astype("uint8") * 255))
        b_img = b_img.resize((a.shape[1], a.shape[0]), Image.NEAREST)
        b = np.array(b_img) > 127
    return float((a == b).sum()) / float(a.size)


# ---------------------------------------------------------------------------
# Running PyTorch reference
# ---------------------------------------------------------------------------

def run_pytorch(rfdetr_class_name: str, images: list[Path], threshold: float,
                is_seg: bool) -> dict[str, dict]:
    """Load one variant, predict on every image; return {image_name: result}.

    Result is {"detections": [...], "image_size": (W, H), "masks": np.ndarray
    or None}. detections are sorted score-desc.
    """
    import numpy as np
    from PIL import Image
    import rfdetr

    klass = getattr(rfdetr, rfdetr_class_name)
    print(f"[pytorch] loading {rfdetr_class_name} ...", flush=True)
    model = klass()

    results: dict[str, dict] = {}
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        det = model.predict(img, threshold=threshold)
        xyxy = det.xyxy if det.xyxy is not None else np.zeros((0, 4))
        conf = det.confidence if det.confidence is not None else np.zeros((0,))
        cls = det.class_id if det.class_id is not None else np.zeros((0,), dtype=int)
        dets = []
        for i in range(len(xyxy)):
            dets.append({
                "class_id": int(cls[i]),
                "score": float(conf[i]),
                "bbox": [float(x) for x in xyxy[i]],
            })
        # Sort by score desc for deterministic matching.
        order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
        dets_sorted = [dets[i] for i in order]
        masks_sorted = None
        if is_seg and getattr(det, "mask", None) is not None:
            masks = det.mask  # (N, H, W) bool
            masks_sorted = masks[order] if len(masks) else masks
        results[img_path.name] = {
            "detections": dets_sorted,
            "image_size": (W, H),
            "masks": masks_sorted,
        }
        print(f"[pytorch] {rfdetr_class_name} {img_path.name}: {len(dets_sorted)} dets",
              flush=True)

    # Free up the model before next variant.
    del model
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception:
        pass
    return results


# ---------------------------------------------------------------------------
# Running C++ CLI
# ---------------------------------------------------------------------------

def run_cpp(cli: Path, model: Path, image: Path, threshold: float,
            is_seg: bool, masks_dir: Path | None) -> dict:
    """Run rfdetr-cli detect; return parsed JSON output dict.

    Raises on non-zero exit; the caller catches and records the error.
    """
    with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as tf:
        out_json = Path(tf.name)
    try:
        cmd = [
            str(cli), "detect",
            "--model", str(model),
            "--input", str(image),
            "--threshold", f"{threshold}",
            "--output", str(out_json),
        ]
        if is_seg and masks_dir is not None:
            masks_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--masks", str(masks_dir)]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"rfdetr-cli detect exit={proc.returncode}\n"
                f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
            )
        with open(out_json, "r") as f:
            data = json.load(f)
        return data
    finally:
        try:
            out_json.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Per-cell comparison
# ---------------------------------------------------------------------------

def compare_cell(py: dict, cpp: dict, is_seg: bool,
                 masks_dir: Path | None) -> dict:
    """Compute per-image accuracy + mask metrics for one (variant, quant, image)."""
    import numpy as np

    py_dets: list[dict] = py["detections"]
    cpp_dets: list[dict] = cpp.get("detections", [])
    n_py = len(py_dets)
    n_cpp = len(cpp_dets)

    # Greedy match at both IoU thresholds.
    pairs50 = greedy_match(py_dets, cpp_dets, 0.5)
    pairs95 = greedy_match(py_dets, cpp_dets, 0.95)
    matched50 = len(pairs50)
    matched95 = len(pairs95)
    extra = n_cpp - matched50  # unmatched C++ detections (at 0.5)

    score_deltas = []
    for pi, ci, _ in pairs50:
        score_deltas.append(abs(py_dets[pi]["score"] - cpp_dets[ci]["score"]))

    out = {
        "py_total":           n_py,
        "cpp_total":          n_cpp,
        "matched_iou_0.5":    matched50,
        "matched_iou_0.95":   matched95,
        "recall_iou_0.5":     (matched50 / n_py) if n_py else 0.0,
        "recall_iou_0.95":    (matched95 / n_py) if n_py else 0.0,
        "max_abs_score_delta":  max(score_deltas) if score_deltas else 0.0,
        "mean_abs_score_delta": (sum(score_deltas) / len(score_deltas))
                                if score_deltas else 0.0,
        "extra_cpp_detections": extra,
    }

    if is_seg and py.get("masks") is not None and masks_dir is not None:
        # PyTorch masks come in the original detection order (after our sort
        # by score desc, so pi indexes into them directly). C++ writes masks
        # as det_{ci:03d}_*.png in detection-output order.
        py_masks = py["masks"]
        ious = []
        agreements = []
        for pi, ci, _ in pairs50:
            cpp_png = find_cpp_mask(masks_dir, ci)
            if cpp_png is None:
                continue
            try:
                cpp_mask = load_cpp_mask(cpp_png)
            except Exception:
                continue
            py_mask = py_masks[pi]
            ious.append(mask_iou(py_mask, cpp_mask))
            agreements.append(mask_pixel_agreement(py_mask, cpp_mask))
        out["n_matched_for_mask"] = len(ious)
        out["mean_mask_iou"] = (sum(ious) / len(ious)) if ious else 0.0
        out["min_mask_iou"]  = min(ious) if ious else 0.0
        out["mean_pixel_agreement"] = (sum(agreements) / len(agreements)) \
                                      if agreements else 0.0

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(per_image: list[dict], is_seg: bool) -> dict:
    """Mean of each numeric metric across images (skipping errored ones)."""
    rows = [r for r in per_image if "error" not in r]
    if not rows:
        return {}

    keys = ["recall_iou_0.5", "recall_iou_0.95",
            "max_abs_score_delta", "mean_abs_score_delta",
            "extra_cpp_detections"]
    if is_seg:
        keys += ["mean_mask_iou", "min_mask_iou", "mean_pixel_agreement"]

    agg = {}
    for k in keys:
        vals = [r["metrics"][k] for r in rows if k in r["metrics"]]
        agg[k] = (sum(vals) / len(vals)) if vals else 0.0
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path,
                    help="Output JSON path")
    ap.add_argument("--images", nargs="+", required=True, type=Path,
                    help="Test images (JPG/PNG)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--models-dir", type=Path,
                    default=REPO / "models")
    ap.add_argument("--cli", type=Path,
                    default=REPO / "build/bin/rfdetr-cli")
    ap.add_argument("--only-variant", action="append", default=None,
                    help="Restrict to this variant (repeatable). Useful for debug.")
    ap.add_argument("--only-quant", action="append", default=None,
                    help="Restrict to this quant (repeatable). Useful for debug.")
    args = ap.parse_args()

    images = [p.resolve() for p in args.images if p.exists()]
    if not images:
        print("[error] no images found", file=sys.stderr)
        return 1
    print(f"[sweep] images ({len(images)}):", flush=True)
    for p in images:
        print(f"    {p}", flush=True)

    variants = [v for v in VARIANTS
                if args.only_variant is None or v[0] in args.only_variant]
    quants = [q for q in QUANTS
              if args.only_quant is None or q in args.only_quant]

    pyver = ""
    try:
        import torch
        pyver = torch.__version__
    except Exception:
        pass

    meta = {
        "date": datetime.date.today().isoformat(),
        "n_images": len(images),
        "image_names": [p.name for p in images],
        "threshold": args.threshold,
        "pytorch_version": pyver,
        "rfdetr_version": importlib.metadata.version("rfdetr"),
        "cli": str(args.cli),
        "models_dir": str(args.models_dir),
    }

    all_cells: list[dict] = []
    masks_root = Path(tempfile.mkdtemp(prefix="sweep_masks_"))
    print(f"[sweep] mask scratch dir: {masks_root}", flush=True)

    for variant_key, class_name, gguf_stem, is_seg in variants:
        # Load Python ground truth once per variant.
        try:
            gt = run_pytorch(class_name, images, args.threshold, is_seg)
        except Exception as e:
            print(f"[error] PyTorch load failed for {variant_key}: {e}",
                  file=sys.stderr)
            traceback.print_exc()
            for q in quants:
                all_cells.append({
                    "variant": variant_key,
                    "quant": q,
                    "model_path": f"models/{gguf_stem}-{q}.gguf",
                    "error": f"pytorch_load_failed: {e}",
                })
            continue

        for q in quants:
            model_path = args.models_dir / f"{gguf_stem}-{q}.gguf"
            cell = {
                "variant": variant_key,
                "quant": q,
                "model_path": str(model_path.relative_to(REPO))
                if model_path.is_relative_to(REPO) else str(model_path),
                "is_seg": is_seg,
            }
            if not model_path.exists():
                cell["error"] = f"model not found: {model_path}"
                all_cells.append(cell)
                print(f"[skip] {variant_key} {q}: model not found", flush=True)
                continue
            cell["file_size_mb"] = round(
                model_path.stat().st_size / (1024 * 1024), 2)
            cell["n_images"] = len(images)

            per_image: list[dict] = []
            for img_path in images:
                row: dict = {"image": img_path.name}
                masks_dir = (masks_root / f"{variant_key}_{q}_{img_path.stem}"
                             if is_seg else None)
                try:
                    cpp = run_cpp(args.cli, model_path, img_path,
                                  args.threshold, is_seg, masks_dir)
                    py_record = gt[img_path.name]
                    metrics = compare_cell(py_record, cpp, is_seg, masks_dir)
                    row["metrics"] = metrics
                except Exception as e:
                    row["error"] = str(e)
                    print(f"[error] {variant_key} {q} {img_path.name}: {e}",
                          file=sys.stderr, flush=True)
                per_image.append(row)

            cell["per_image"] = per_image
            cell["metrics"] = aggregate(per_image, is_seg)
            ok = sum(1 for r in per_image if "error" not in r)
            print(f"[cell] {variant_key:<12} {q:<5} "
                  f"ok={ok}/{len(images)} "
                  f"R@0.5={cell['metrics'].get('recall_iou_0.5', 0):.3f} "
                  f"R@0.95={cell['metrics'].get('recall_iou_0.95', 0):.3f} "
                  f"meanΔ={cell['metrics'].get('mean_abs_score_delta', 0):.4f}"
                  + (f" maskIoU={cell['metrics'].get('mean_mask_iou', 0):.4f}"
                     if is_seg else ""),
                  flush=True)
            all_cells.append(cell)

    out = {"meta": meta, "cells": all_cells}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[sweep] wrote {args.output} ({len(all_cells)} cells)", flush=True)

    # Cleanup mask scratch.
    import shutil
    shutil.rmtree(masks_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
