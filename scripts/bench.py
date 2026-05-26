#!/usr/bin/env python3
"""Benchmark rfdetr.cpp (C++ F32/Q8_0) against upstream Python rfdetr.

Measures wall-clock inference time and cross-validates detection outputs on
the same set of test images. Both implementations run on CPU.

Usage:
    .venv/bin/python scripts/bench.py \\
        --cli build/bin/rfdetr-cli \\
        --image /tmp/coco_sample.jpg --image /tmp/coco_sample2.jpg \\
        --iters 5 --warmup 2

Outputs a markdown report to stdout (and to --out if provided).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Force CPU before importing torch / rfdetr.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[1]


def parse_bench_stdout(text: str) -> dict:
    """Parse the C++ rfdetr-cli `bench` stdout into a dict."""
    out: dict = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s+(.+)$", line.strip())
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k in ("load_ms", "min_ms", "median_ms", "mean_ms", "max_ms"):
            out[k] = float(v)
        elif k in ("warmup", "iters", "detections"):
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


def run_cpp_bench(cli: Path, model: Path, image: Path, iters: int, warmup: int,
                  threads: int | None = None) -> dict:
    cmd = [
        str(cli), "bench",
        "--model", str(model),
        "--input", str(image),
        "--iters", str(iters),
        "--warmup", str(warmup),
    ]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"C++ bench failed (rc={proc.returncode}):\n{proc.stderr}")
    return parse_bench_stdout(proc.stdout)


def run_cpp_detect(cli: Path, model: Path, image: Path,
                   threads: int | None = None) -> list[dict]:
    """One-shot detect; returns parsed detections list."""
    out_json = Path("/tmp") / f"_bench_detect_{os.getpid()}.json"
    cmd = [
        str(cli), "detect",
        "--model", str(model),
        "--input", str(image),
        "--output", str(out_json),
        "--threshold", "0.5",
    ]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"C++ detect failed (rc={proc.returncode}):\n{proc.stderr}")
    data = json.loads(out_json.read_text())
    out_json.unlink(missing_ok=True)
    return data["detections"]


def bench_python(images: list[Path], iters: int, warmup: int) -> dict:
    """Load the Python model once, time predict() per image."""
    # Lazy imports so we can fail fast if missing.
    import torch
    from rfdetr import RFDETRBase

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[python] torch device: {device}", file=sys.stderr)

    model = RFDETRBase()  # CPU because CUDA_VISIBLE_DEVICES="".

    results: dict = {"device": device, "per_image": {}}
    for img in images:
        # Warmup.
        for _ in range(warmup):
            _ = model.predict(str(img), threshold=0.5)
        # Timed.
        ms_list: list[float] = []
        last_det = None
        for _ in range(iters):
            t0 = time.perf_counter()
            det = model.predict(str(img), threshold=0.5)
            t1 = time.perf_counter()
            ms_list.append((t1 - t0) * 1000.0)
            last_det = det
        ms_sorted = sorted(ms_list)
        results["per_image"][str(img)] = {
            "min_ms":    ms_sorted[0],
            "median_ms": ms_sorted[len(ms_sorted) // 2],
            "mean_ms":   sum(ms_list) / len(ms_list),
            "max_ms":    ms_sorted[-1],
            "detections": [
                {
                    "class_id": int(last_det.class_id[i]),
                    "score":    float(last_det.confidence[i]),
                    "bbox":     [float(x) for x in last_det.xyxy[i].tolist()],
                }
                for i in range(len(last_det.class_id))
            ],
        }
    return results


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


def center_dist(a: list[float], b: list[float]) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def compare_detections(py_dets: list[dict], cpp_dets: list[dict],
                       iou_thresh: float = 0.95) -> dict:
    """Greedy 1-1 matching by (class match AND IoU >= thresh)."""
    py_used = [False] * len(py_dets)
    matched = 0
    score_deltas = []
    center_deltas = []
    iou_vals = []
    for c in cpp_dets:
        best_idx = -1
        best_iou = 0.0
        for j, p in enumerate(py_dets):
            if py_used[j]:
                continue
            if int(p["class_id"]) != int(c["class_id"]):
                continue
            v = iou(c["bbox"], p["bbox"])
            if v > best_iou:
                best_iou = v
                best_idx = j
        if best_idx >= 0 and best_iou >= iou_thresh:
            py_used[best_idx] = True
            matched += 1
            score_deltas.append(abs(c["score"] - py_dets[best_idx]["score"]))
            center_deltas.append(center_dist(c["bbox"], py_dets[best_idx]["bbox"]))
            iou_vals.append(best_iou)
    return {
        "matched":         matched,
        "cpp_total":       len(cpp_dets),
        "py_total":        len(py_dets),
        "mean_score_delta":  sum(score_deltas) / len(score_deltas) if score_deltas else 0.0,
        "max_score_delta":   max(score_deltas) if score_deltas else 0.0,
        "mean_center_dist":  sum(center_deltas) / len(center_deltas) if center_deltas else 0.0,
        "max_center_dist":   max(center_deltas) if center_deltas else 0.0,
        "mean_iou":          sum(iou_vals) / len(iou_vals) if iou_vals else 0.0,
        "min_iou":           min(iou_vals) if iou_vals else 0.0,
    }


def fmt_ms(x: float) -> str:
    return f"{x:8.1f}"


def render_report(rows: list[dict], py_results: dict) -> str:
    lines = []
    lines.append("# rfdetr.cpp vs upstream Python rfdetr — benchmark\n")
    lines.append(f"Torch device: `{py_results['device']}`  ")
    lines.append("(CPU forced via `CUDA_VISIBLE_DEVICES=\"\"`)\n")

    # Timing table.
    lines.append("## Inference time (ms per image, lower is better)\n")
    lines.append("| image | impl | min | median | mean | max | speedup vs Python |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        py = r["python"]
        for impl in ("python", "cpp_f32", "cpp_q8"):
            if impl not in r:
                continue
            d = r[impl]
            if impl == "python":
                speed = "1.00x"
            else:
                speed = f"{py['median_ms'] / d['median_ms']:.2f}x"
            lines.append(
                f"| {r['image_name']} | {impl} | {fmt_ms(d['min_ms'])} | "
                f"{fmt_ms(d['median_ms'])} | {fmt_ms(d['mean_ms'])} | "
                f"{fmt_ms(d['max_ms'])} | {speed} |"
            )
    lines.append("")

    # Detection comparison vs Python (reference).
    lines.append("## Detection cross-check (vs Python as reference, IoU >= 0.95)\n")
    lines.append("| image | impl | py_det | cpp_det | matched | mean IoU | mean |score Δ| | max |score Δ| | mean center px Δ | max center px Δ |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        for impl in ("cpp_f32", "cpp_q8"):
            if impl not in r:
                continue
            c = r[f"{impl}_cmp"]
            lines.append(
                f"| {r['image_name']} | {impl} | {c['py_total']} | {c['cpp_total']} | "
                f"{c['matched']} | {c['mean_iou']:.4f} | "
                f"{c['mean_score_delta']:.4f} | {c['max_score_delta']:.4f} | "
                f"{c['mean_center_dist']:.2f} | {c['max_center_dist']:.2f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli",     default=str(REPO / "build/bin/rfdetr-cli"))
    ap.add_argument("--f32",     default=str(REPO / "models/rfdetr-base-f32.gguf"))
    ap.add_argument("--q8",      default=str(REPO / "models/rfdetr-base-q8_0.gguf"))
    ap.add_argument("--image",   action="append", required=True,
                    help="Path to a test image; pass multiple times.")
    ap.add_argument("--iters",   type=int, default=5)
    ap.add_argument("--warmup",  type=int, default=2)
    ap.add_argument("--threads", type=int, default=None,
                    help="C++ ggml thread count (default: CLI auto = hardware_concurrency).")
    ap.add_argument("--skip-q8", action="store_true")
    ap.add_argument("--out",     default=None, help="Write markdown report to this file.")
    args = ap.parse_args()

    cli = Path(args.cli)
    f32 = Path(args.f32)
    q8  = Path(args.q8)
    images = [Path(p) for p in args.image]

    for p in (cli, f32):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 1
    if not args.skip_q8 and not q8.exists():
        print(f"missing q8 model: {q8} (pass --skip-q8 to skip)", file=sys.stderr)
        return 1
    for img in images:
        if not img.exists():
            print(f"missing image: {img}", file=sys.stderr)
            return 1

    # 1. Python (warmup+iters per image, model loaded once).
    print("=== Python rfdetr ===", file=sys.stderr)
    py_results = bench_python(images, args.iters, args.warmup)

    # 2. C++ F32 and Q8 per image.
    rows = []
    for img in images:
        print(f"=== C++ bench: {img.name} ===", file=sys.stderr)
        row = {"image": str(img), "image_name": img.name}
        # Python entry.
        py = py_results["per_image"][str(img)]
        row["python"] = py

        # C++ F32 timing + one detect for cross-check.
        f32_t = run_cpp_bench(cli, f32, img, args.iters, args.warmup, args.threads)
        f32_d = run_cpp_detect(cli, f32, img, args.threads)
        row["cpp_f32"]     = f32_t
        row["cpp_f32_dets"] = f32_d
        row["cpp_f32_cmp"] = compare_detections(py["detections"], f32_d)

        if not args.skip_q8:
            q8_t = run_cpp_bench(cli, q8, img, args.iters, args.warmup, args.threads)
            q8_d = run_cpp_detect(cli, q8, img, args.threads)
            row["cpp_q8"]     = q8_t
            row["cpp_q8_dets"] = q8_d
            row["cpp_q8_cmp"] = compare_detections(py["detections"], q8_d)

        rows.append(row)

    report = render_report(rows, py_results)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"[bench] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
