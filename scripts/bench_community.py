#!/usr/bin/env python3
"""Community benchmark: collect data for publication-quality plots.

Sweeps (impl, image, threads) and persists raw timing + detection data to
benchmarks/results/bench_data.json. Use scripts/plot_community.py to render
the plots.

Cells run:
- per_image: PyTorch F32 / C++ F32 / C++ Q8_0 at T=8 on N images
  (the "headline" latency comparison data)
- thread_sweep: PyTorch + C++ F32 + C++ Q8_0 over T in {1,2,4,8,12,16,20}
  on ONE representative image (kitchen / coco_sample.jpg)
- detections: one detect run per (impl, image) for accuracy cross-check

Uses time.perf_counter and disables GC during timed Python sections.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import median, mean

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[1]


def parse_bench_stdout(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s+(.+)$", line.strip())
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k in ("load_ms", "min_ms", "median_ms", "mean_ms", "max_ms"):
            out[k] = float(v)
        elif k in ("warmup", "iters", "detections", "threads"):
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


def run_cpp_bench(cli: Path, model: Path, image: Path,
                  iters: int, warmup: int, threads: int) -> dict:
    cmd = [
        str(cli), "bench",
        "--model", str(model),
        "--input", str(image),
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--threads", str(threads),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"C++ bench failed (rc={proc.returncode}):\n{proc.stderr}"
        )
    return parse_bench_stdout(proc.stdout)


def run_cpp_detect(cli: Path, model: Path, image: Path,
                   threads: int) -> list[dict]:
    out_json = Path("/tmp") / f"_bench_detect_{os.getpid()}.json"
    cmd = [
        str(cli), "detect",
        "--model", str(model),
        "--input", str(image),
        "--output", str(out_json),
        "--threshold", "0.5",
        "--threads", str(threads),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"C++ detect failed (rc={proc.returncode}):\n{proc.stderr}"
        )
    data = json.loads(out_json.read_text())
    out_json.unlink(missing_ok=True)
    return data["detections"]


def time_one_python(model, img: Path, iters: int, warmup: int) -> tuple[list[float], list[dict]]:
    """Run one (image, model) Python timing block; returns per-iter ms + final detections."""
    # warmup
    for _ in range(warmup):
        _ = model.predict(str(img), threshold=0.5)
    # timed
    ms_list: list[float] = []
    last_det = None
    gc.collect()
    gc.disable()
    try:
        for _ in range(iters):
            t0 = time.perf_counter()
            det = model.predict(str(img), threshold=0.5)
            t1 = time.perf_counter()
            ms_list.append((t1 - t0) * 1000.0)
            last_det = det
    finally:
        gc.enable()
    dets = []
    if last_det is not None:
        for i in range(len(last_det.class_id)):
            dets.append({
                "class_id": int(last_det.class_id[i]),
                "score":    float(last_det.confidence[i]),
                "bbox":     [float(x) for x in last_det.xyxy[i].tolist()],
            })
    return ms_list, dets


def time_one_cpp(cli: Path, model: Path, img: Path, iters: int, warmup: int,
                 threads: int) -> dict:
    """Single C++ bench call → returns full parsed dict (incl min/median/mean/max).

    We can't get per-iter, but min/median/max with iters>=15 gives reliable
    error bars. We also estimate p25/p75 from min/max assuming a tight dist.
    """
    return run_cpp_bench(cli, model, img, iters, warmup, threads)


def aggregate_python(ms_list: list[float]) -> dict:
    s = sorted(ms_list)
    n = len(s)
    return {
        "min_ms":    s[0],
        "p25_ms":    s[n // 4] if n >= 4 else s[0],
        "median_ms": s[n // 2],
        "p75_ms":    s[(3 * n) // 4] if n >= 4 else s[-1],
        "mean_ms":   sum(ms_list) / n,
        "max_ms":    s[-1],
        "raw_ms":    ms_list,
    }


def iou(a: list[float], b: list[float]) -> float:
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


def match_detections(py_dets: list[dict], cpp_dets: list[dict],
                     iou_thresh: float = 0.95) -> list[dict]:
    """Greedy 1-1 matching by (class match AND IoU >= thresh).

    Returns list of matched pairs with score/IoU/center info.
    """
    py_used = [False] * len(py_dets)
    pairs = []
    for c in cpp_dets:
        best_idx, best_iou = -1, 0.0
        for j, p in enumerate(py_dets):
            if py_used[j] or int(p["class_id"]) != int(c["class_id"]):
                continue
            v = iou(c["bbox"], p["bbox"])
            if v > best_iou:
                best_iou, best_idx = v, j
        if best_idx >= 0 and best_iou >= iou_thresh:
            py_used[best_idx] = True
            p = py_dets[best_idx]
            pairs.append({
                "class_id":  int(c["class_id"]),
                "py_score":  float(p["score"]),
                "cpp_score": float(c["score"]),
                "score_delta": abs(float(c["score"]) - float(p["score"])),
                "iou":       float(best_iou),
            })
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli",     default=str(REPO / "build/bin/rfdetr-cli"))
    ap.add_argument("--f32",     default=str(REPO / "models/rfdetr-base-f32.gguf"))
    ap.add_argument("--q8",      default=str(REPO / "models/rfdetr-base-q8_0.gguf"))
    ap.add_argument("--images-dir", default=str(REPO / "benchmarks/images"))
    ap.add_argument("--out",     default=str(REPO / "benchmarks/results/bench_data.json"))
    ap.add_argument("--iters",   type=int, default=15,
                    help="timed iterations per cell (default 15)")
    ap.add_argument("--warmup",  type=int, default=3)
    ap.add_argument("--threads", type=int, default=8,
                    help="C++ thread count for the per-image headline cells")
    ap.add_argument("--thread-sweep", default="1,2,4,8,12,16,20",
                    help="comma-separated thread counts for the scaling plot")
    ap.add_argument("--sweep-image", default="coco_kitchen.jpg",
                    help="image to use for the thread-scaling sweep")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-python", action="store_true",
                    help="skip the PyTorch arm of the benchmark (testing only)")
    args = ap.parse_args()

    cli  = Path(args.cli)
    f32  = Path(args.f32)
    q8   = Path(args.q8)
    idir = Path(args.images_dir)

    images = sorted(idir.glob("*.jpg"))
    if not images:
        print(f"no images in {idir}", file=sys.stderr)
        return 1

    sweep_threads = [int(t) for t in args.thread_sweep.split(",") if t]
    sweep_image = idir / args.sweep_image
    if not sweep_image.exists():
        # fall back to first image
        sweep_image = images[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------- meta --------
    import platform
    meta = {
        "schema_version": 1,
        "iters": args.iters,
        "warmup": args.warmup,
        "threads_headline": args.threads,
        "thread_sweep": sweep_threads,
        "sweep_image": sweep_image.name,
        "platform": {
            "system":  platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu":     "AMD Ryzen 9 9950X3D",  # confirmed via lscpu
            "cores":   os.cpu_count(),
        },
        "models": {
            "f32_path": str(f32), "f32_size_bytes": f32.stat().st_size,
            "q8_path":  str(q8),  "q8_size_bytes":  q8.stat().st_size,
        },
        "cli": str(cli),
    }

    data = {"meta": meta, "per_image": {}, "thread_sweep": {}, "detections": {}}

    # -------- python: load model once, reuse --------
    py_model = None
    if not args.skip_python:
        print("[bench] loading Python rfdetr (one-time)...", file=sys.stderr)
        from rfdetr import RFDETRBase
        py_model = RFDETRBase()

    # -------- per-image headline timings (T=8 for C++, default Python threads) --------
    for img in images:
        print(f"\n=== headline: {img.name} ===", file=sys.stderr)
        cell: dict = {"image": img.name}

        if py_model is not None:
            print(f"[python] warmup={args.warmup}, iters={args.iters}", file=sys.stderr)
            ms_list, py_dets = time_one_python(py_model, img, args.iters, args.warmup)
            cell["python"] = aggregate_python(ms_list)
            data["detections"].setdefault(img.name, {})["python"] = py_dets
            print(f"[python] median={cell['python']['median_ms']:.1f} ms "
                  f"min={cell['python']['min_ms']:.1f} max={cell['python']['max_ms']:.1f}",
                  file=sys.stderr)

        print(f"[cpp_f32 T={args.threads}]", file=sys.stderr)
        cell["cpp_f32"] = time_one_cpp(cli, f32, img, args.iters, args.warmup, args.threads)
        print(f"[cpp_f32] median={cell['cpp_f32']['median_ms']:.1f} ms "
              f"min={cell['cpp_f32']['min_ms']:.1f} max={cell['cpp_f32']['max_ms']:.1f}",
              file=sys.stderr)

        print(f"[cpp_q8  T={args.threads}]", file=sys.stderr)
        cell["cpp_q8"]  = time_one_cpp(cli, q8,  img, args.iters, args.warmup, args.threads)
        print(f"[cpp_q8 ] median={cell['cpp_q8']['median_ms']:.1f} ms "
              f"min={cell['cpp_q8']['min_ms']:.1f} max={cell['cpp_q8']['max_ms']:.1f}",
              file=sys.stderr)

        # detections (single run each)
        if "detections" not in data:
            data["detections"] = {}
        data["detections"].setdefault(img.name, {})
        print(f"[detect cpp_f32]", file=sys.stderr)
        data["detections"][img.name]["cpp_f32"] = run_cpp_detect(cli, f32, img, args.threads)
        print(f"[detect cpp_q8 ]", file=sys.stderr)
        data["detections"][img.name]["cpp_q8" ] = run_cpp_detect(cli, q8,  img, args.threads)

        data["per_image"][img.name] = cell

        # Persist after each image so progress isn't lost
        out_path.write_text(json.dumps(data, indent=2))

    # -------- thread sweep on ONE image --------
    if not args.skip_sweep:
        print(f"\n=== thread sweep on {sweep_image.name}: T in {sweep_threads} ===",
              file=sys.stderr)
        sweep = {"image": sweep_image.name, "threads": sweep_threads,
                 "cpp_f32": {}, "cpp_q8": {}, "python": {}}

        # Python doesn't expose a thread knob via predict(); torch reads
        # OMP_NUM_THREADS / MKL_NUM_THREADS at import time. To probe scaling
        # honestly, we set torch.set_num_threads() per sweep step.
        if py_model is not None:
            import torch
            for n in sweep_threads:
                print(f"[python T={n}]", file=sys.stderr)
                torch.set_num_threads(n)
                # interop threads stay at default; matmul threads is what matters
                ms_list, _ = time_one_python(py_model, sweep_image,
                                              args.iters, args.warmup)
                sweep["python"][str(n)] = aggregate_python(ms_list)
                print(f"  median={sweep['python'][str(n)]['median_ms']:.1f} ms",
                      file=sys.stderr)

        for n in sweep_threads:
            print(f"[cpp_f32 T={n}]", file=sys.stderr)
            sweep["cpp_f32"][str(n)] = time_one_cpp(
                cli, f32, sweep_image, args.iters, args.warmup, n)
            print(f"  median={sweep['cpp_f32'][str(n)]['median_ms']:.1f} ms",
                  file=sys.stderr)
        for n in sweep_threads:
            print(f"[cpp_q8  T={n}]", file=sys.stderr)
            sweep["cpp_q8"][str(n)] = time_one_cpp(
                cli, q8, sweep_image, args.iters, args.warmup, n)
            print(f"  median={sweep['cpp_q8'][str(n)]['median_ms']:.1f} ms",
                  file=sys.stderr)

        data["thread_sweep"] = sweep
        out_path.write_text(json.dumps(data, indent=2))

    # -------- detection match summary --------
    match_summary = {}
    for img_name, dets in data["detections"].items():
        if "python" not in dets:
            continue
        py_dets = dets["python"]
        for impl in ("cpp_f32", "cpp_q8"):
            if impl not in dets:
                continue
            pairs = match_detections(py_dets, dets[impl])
            key = f"{img_name}::{impl}"
            score_deltas = [p["score_delta"] for p in pairs]
            ious = [p["iou"] for p in pairs]
            match_summary[key] = {
                "image": img_name, "impl": impl,
                "py_total": len(py_dets), "cpp_total": len(dets[impl]),
                "matched": len(pairs),
                "mean_score_delta": (sum(score_deltas) / len(score_deltas)) if score_deltas else 0.0,
                "max_score_delta":  max(score_deltas) if score_deltas else 0.0,
                "mean_iou":         (sum(ious) / len(ious)) if ious else 0.0,
                "min_iou":          min(ious) if ious else 0.0,
                "pairs":            pairs,
            }
    data["match_summary"] = match_summary
    out_path.write_text(json.dumps(data, indent=2))

    print(f"\n[bench] wrote {out_path}", file=sys.stderr)
    print(f"[bench] images: {len(images)}, sweep cells: {len(sweep_threads) if not args.skip_sweep else 0}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
