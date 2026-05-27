#!/usr/bin/env python3
"""Community benchmark: collect data for publication-quality plots.

Sweeps (impl, image, threads) and persists raw timing + detection data to
benchmarks/results/bench_data.json. Use scripts/plot_community.py to render
the plots.

Cells run:
- per_image: PyTorch F32 / C++ F32 / C++ F16 / C++ Q8_0 / C++ Q5_0 / C++ Q4_0
  plus optional C++ Q4_K / Q5_K / Q6_K (K-quants from `rfdetr-cli quantize`)
  at T=8 on N images (the "headline" latency comparison data). F16 is the
  recommended sweet-spot: faster than F32 and Q8_0, half the F32 size,
  lossless accuracy on this model.
- thread_sweep: PyTorch + C++ F32 + C++ F16 + C++ Q8_0 over T in {1,2,4,8,12,16,20}
  on ONE representative image (kitchen / coco_sample.jpg). Q4_0 / Q5_0 /
  K-quants are only swept at the headline T=8 cell — their per-thread shape is
  similar to Q8_0 and they're a side story, not the headline.
- detections: one detect run per (impl, image) for accuracy cross-check

Uses time.perf_counter and disables GC during timed Python sections.

Two modes:
- Default (legacy): cycles through (impl × image) cells in a single linear
  sweep. Back-to-back C++ runs can heat the CPU, polluting measurements that
  follow (PyTorch tends to suffer because it runs first per image, see git
  history for the methodology hole this revealed).
- --rigorous: round-robin per (image, impl) with a cooldown sleep between
  cells, plus multiple full passes through the (impl × image) grid. The
  per-cell median is then taken across passes (a trimmed-median when
  passes >= 4) and the IQR (p25..p75) across passes is the reported error
  bar. Python warmup+iters is still run inside one cell, with the cooldown
  applied between cells. Use this for publication-quality numbers.
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
                  iters: int, warmup: int, threads: int,
                  taskset: str = "") -> dict:
    cmd: list[str] = []
    if taskset:
        cmd = ["taskset", "-c", taskset]
    cmd += [
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
                 threads: int, taskset: str = "") -> dict:
    """Single C++ bench call → returns full parsed dict (incl min/median/mean/max).

    We can't get per-iter, but min/median/max with iters>=15 gives reliable
    error bars. We also estimate p25/p75 from min/max assuming a tight dist.
    """
    return run_cpp_bench(cli, model, img, iters, warmup, threads, taskset)


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


# ----------------------------------------------------------------------------
# Rigorous-mode helpers
# ----------------------------------------------------------------------------
def trimmed_aggregate(ms_list: list[float], trim_pct: float = 0.10) -> dict:
    """Drop top+bottom `trim_pct` of `ms_list` then compute median + IQR.

    For per-iter Python data with 20+ samples this kills the bimodal cold-
    iter / scheduler-jitter outliers that dominate min/max-based whiskers.

    Returns the same fields as `aggregate_python` plus `trim_pct`, `n_trimmed`,
    and an `iqr_pct` health metric (lower = more stable).
    """
    n = len(ms_list)
    if n == 0:
        return {"min_ms": 0.0, "p25_ms": 0.0, "median_ms": 0.0,
                "p75_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0,
                "raw_ms": [], "trim_pct": trim_pct, "n_trimmed": 0,
                "iqr_pct": 0.0}
    s = sorted(ms_list)
    drop = int(n * trim_pct)
    trimmed = s[drop:n - drop] if drop > 0 and n - 2 * drop >= 1 else s
    t = len(trimmed)
    median = trimmed[t // 2]
    p25 = trimmed[t // 4] if t >= 4 else trimmed[0]
    p75 = trimmed[(3 * t) // 4] if t >= 4 else trimmed[-1]
    iqr_pct = ((p75 - p25) / median * 100.0) if median > 0 else 0.0
    return {
        "min_ms":    s[0],
        "p25_ms":    p25,
        "median_ms": median,
        "p75_ms":    p75,
        "mean_ms":   sum(trimmed) / t,
        "max_ms":    s[-1],
        "raw_ms":    ms_list,
        "trim_pct":  trim_pct,
        "n_trimmed": n - t,
        "iqr_pct":   iqr_pct,
    }


def aggregate_across_passes(per_pass_cells: list[dict]) -> dict:
    """Aggregate per-pass cell dicts into one canonical cell.

    Each `per_pass_cells[i]` has {min_ms, median_ms, mean_ms, max_ms, ...}
    from one bench call. We take the median across the per-pass medians
    (no trim with passes <= 3, hard-min/max trim with passes >= 4) and
    expose the per-pass set as `passes_medians`.

    p25/p75 come from the per-pass medians directly (so they reflect
    pass-to-pass thermal/scheduler stability, not within-call jitter).
    """
    if not per_pass_cells:
        return {"min_ms": 0.0, "p25_ms": 0.0, "median_ms": 0.0,
                "p75_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0,
                "passes_medians": [], "iqr_pct": 0.0}
    medians = sorted(float(c["median_ms"]) for c in per_pass_cells)
    n = len(medians)
    # With >=4 passes, drop one from each end before aggregating.
    if n >= 4:
        core = medians[1:-1]
    else:
        core = medians
    m = len(core)
    median = core[m // 2]
    p25 = core[m // 4] if m >= 4 else core[0]
    p75 = core[(3 * m) // 4] if m >= 4 else core[-1]
    iqr_pct = ((p75 - p25) / median * 100.0) if median > 0 else 0.0
    out = {
        "min_ms":         min(float(c["min_ms"])    for c in per_pass_cells),
        "p25_ms":         p25,
        "median_ms":      median,
        "p75_ms":         p75,
        "mean_ms":        sum(float(c["mean_ms"]) for c in per_pass_cells) / n,
        "max_ms":         max(float(c["max_ms"])    for c in per_pass_cells),
        "passes_medians": medians,
        "n_passes":       n,
        "iqr_pct":        iqr_pct,
    }
    # Echo non-numeric fields (model/image/threads/iters) from the first cell.
    for k in ("model", "image", "threads", "warmup", "iters",
             "detections", "load_ms"):
        if k in per_pass_cells[0]:
            out[k] = per_pass_cells[0][k]
    return out


def cooldown_sleep(seconds: float, label: str = "") -> None:
    """Sleep for thermal cooldown, with optional progress dot output."""
    if seconds <= 0:
        return
    msg = f"[cooldown {seconds:.1f}s]"
    if label:
        msg = f"[cooldown {seconds:.1f}s after {label}]"
    print(msg, file=sys.stderr, flush=True)
    time.sleep(seconds)


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
    ap.add_argument("--f16",     default=str(REPO / "models/rfdetr-base-f16.gguf"),
                    help="optional F16 model; skipped if file missing")
    ap.add_argument("--q8",      default=str(REPO / "models/rfdetr-base-q8_0.gguf"))
    ap.add_argument("--q5",      default=str(REPO / "models/rfdetr-base-q5_0.gguf"),
                    help="optional Q5_0 model; skipped if file missing")
    ap.add_argument("--q4",      default=str(REPO / "models/rfdetr-base-q4_0.gguf"),
                    help="optional Q4_0 model; skipped if file missing")
    ap.add_argument("--q4_K",    default=str(REPO / "models/rfdetr-base-q4_K.gguf"),
                    help="optional Q4_K model; skipped if file missing")
    ap.add_argument("--q5_K",    default=str(REPO / "models/rfdetr-base-q5_K.gguf"),
                    help="optional Q5_K model; skipped if file missing")
    ap.add_argument("--q6_K",    default=str(REPO / "models/rfdetr-base-q6_K.gguf"),
                    help="optional Q6_K model; skipped if file missing")
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
    ap.add_argument("--variants", default="",
                    help="Comma-separated detection variants to bench in "
                         "addition to the quant-on-base headline (e.g. "
                         "'nano,small,medium,large'). For each variant, "
                         "models/rfdetr-{variant}-f32.gguf must exist. The "
                         "results land under data['variants'][<variant>] "
                         "with C++ F32 + PyTorch RFDETR{Variant}() timings "
                         "on every image. Skips any variant whose GGUF is "
                         "missing.")
    ap.add_argument("--variant-sweep-image", default="coco_kitchen.jpg",
                    help="image to use for the per-variant timing sweep "
                         "(also used for detection counts in plots)")
    # ---- rigorous mode -----------------------------------------------------
    ap.add_argument("--rigorous", action="store_true",
                    help="Use the rigorous methodology: round-robin per "
                         "(image, impl), cooldown sleep between cells, and N "
                         "full passes through the (impl × image) grid. Final "
                         "per-cell numbers are aggregated across passes "
                         "(median-of-medians + IQR from per-pass medians). "
                         "Recommended for publication-quality data; defaults "
                         "(legacy linear sweep) are kept for back-compat.")
    ap.add_argument("--cooldown", type=float, default=8.0,
                    help="seconds to sleep between cells in rigorous mode "
                         "(default 8.0) — lets the CPU return toward baseline "
                         "temperature between impls and images")
    ap.add_argument("--passes", type=int, default=3,
                    help="number of full round-robin passes through the "
                         "(impl × image) grid in rigorous mode (default 3); "
                         "averaging across passes cancels monotonic thermal "
                         "drift and dual-CCD scheduler jitter")
    ap.add_argument("--taskset", default="",
                    help="optional taskset CPU mask/range for C++ runs in "
                         "rigorous mode (e.g. '0-15' to pin to one CCD on "
                         "9950X3D). Empty = no pinning. Python isn't pinned "
                         "(rfdetr's torch threads ignore the outer cpuset).")
    args = ap.parse_args()

    cli  = Path(args.cli)
    f32  = Path(args.f32)
    f16  = Path(args.f16)
    q8   = Path(args.q8)
    q5   = Path(args.q5)
    q4   = Path(args.q4)
    q4K  = Path(args.q4_K)
    q5K  = Path(args.q5_K)
    q6K  = Path(args.q6_K)
    idir = Path(args.images_dir)

    have_f16 = f16.exists()
    have_q5 = q5.exists()
    have_q4 = q4.exists()
    have_q4K = q4K.exists()
    have_q5K = q5K.exists()
    have_q6K = q6K.exists()
    if not have_f16:
        print(f"[bench] WARN: F16 model not found at {f16} — F16 cells will be skipped",
              file=sys.stderr)

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
        "methodology": (
            {
                "mode":     "rigorous",
                "passes":   args.passes,
                "cooldown_seconds": args.cooldown,
                "iters_per_cell":   args.iters,
                "warmup_per_cell":  args.warmup,
                "round_robin":      True,
                "trim_pct_python":  0.10,
                "passes_aggregate": "median-of-per-pass-medians; IQR from per-pass medians",
                "taskset":          args.taskset or None,
                "notes": (
                    "Round-robin per (image, impl); cooldown sleep between "
                    "every cell to keep the CPU near baseline temperature. "
                    "Per-pass C++ medians come from rfdetr-cli bench's "
                    "internal median over `iters` timed iterations after "
                    "`warmup`. Python per-iter ms are trimmed top+bottom "
                    "10% before computing the in-cell median, then per-pass "
                    "medians are aggregated as above. The IQR reported per "
                    "cell measures pass-to-pass stability, not within-call "
                    "jitter, so it's a direct thermal/scheduler-jitter probe."
                ),
            }
            if args.rigorous else
            {
                "mode":     "legacy-linear-sweep",
                "iters_per_cell":   args.iters,
                "warmup_per_cell":  args.warmup,
                "notes": (
                    "Single linear pass through (impl × image); no cooldown "
                    "between cells. Back-to-back C++ runs may thermally "
                    "pollute the PyTorch measurement that follows. Use "
                    "--rigorous for publication-quality numbers."
                ),
            }
        ),
        "platform": {
            "system":  platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu":     "AMD Ryzen 9 9950X3D",  # confirmed via lscpu
            "cores":   os.cpu_count(),
        },
        "models": {
            "f32_path": str(f32), "f32_size_bytes": f32.stat().st_size,
            **({"f16_path": str(f16), "f16_size_bytes": f16.stat().st_size} if have_f16 else {}),
            "q8_path":  str(q8),  "q8_size_bytes":  q8.stat().st_size,
            **({"q5_path": str(q5), "q5_size_bytes": q5.stat().st_size} if have_q5 else {}),
            **({"q4_path": str(q4), "q4_size_bytes": q4.stat().st_size} if have_q4 else {}),
            **({"q4K_path": str(q4K), "q4K_size_bytes": q4K.stat().st_size} if have_q4K else {}),
            **({"q5K_path": str(q5K), "q5K_size_bytes": q5K.stat().st_size} if have_q5K else {}),
            **({"q6K_path": str(q6K), "q6K_size_bytes": q6K.stat().st_size} if have_q6K else {}),
        },
        "cli": str(cli),
    }

    data = {"meta": meta, "per_image": {}, "thread_sweep": {},
            "detections": {}, "variants": {}}

    # Per-variant config. Always includes the "base" entry implicit in the
    # quant headline; --variants adds more.
    variant_specs = []
    if args.variants:
        for v in [s.strip().lower() for s in args.variants.split(",") if s.strip()]:
            if v == "base":
                # base is already exercised by the quant headline; record it
                # here too for plotting symmetry.
                pass
            gguf = REPO / "models" / f"rfdetr-{v}-f32.gguf"
            if not gguf.exists():
                print(f"[variants] SKIP {v} -- missing {gguf}", file=sys.stderr)
                continue
            variant_specs.append((v, gguf))
        meta["variants"] = [v for v, _ in variant_specs]
        meta["variant_sweep_image"] = args.variant_sweep_image

    # -------- python: load model once, reuse --------
    py_model = None
    if not args.skip_python:
        print("[bench] loading Python rfdetr (one-time)...", file=sys.stderr)
        from rfdetr import RFDETRBase
        py_model = RFDETRBase()

    # -------- per-image headline timings (T=8 for C++, default Python threads) --------
    # Build the (label, path, has-model) impl list used by both modes.
    cpp_impls: list[tuple[str, Path]] = [("cpp_f32", f32)]
    if have_f16: cpp_impls.append(("cpp_f16", f16))
    cpp_impls.append(("cpp_q8", q8))
    if have_q5:  cpp_impls.append(("cpp_q5", q5))
    if have_q4:  cpp_impls.append(("cpp_q4", q4))
    if have_q4K: cpp_impls.append(("cpp_q4K", q4K))
    if have_q5K: cpp_impls.append(("cpp_q5K", q5K))
    if have_q6K: cpp_impls.append(("cpp_q6K", q6K))

    if args.rigorous:
        # ---- Rigorous mode: round-robin per (image, impl), N passes,
        #      cooldown between every cell. Aggregate per-pass medians.
        print(f"\n=== RIGOROUS sweep: passes={args.passes} "
              f"cooldown={args.cooldown}s iters={args.iters} warmup={args.warmup} "
              f"taskset='{args.taskset or 'none'}' ===", file=sys.stderr)

        # Per-cell accumulators: per_cell["coco_kitchen.jpg"]["cpp_f32"] = [pass1, pass2, ...]
        per_cell: dict[str, dict[str, list[dict]]] = {
            img.name: {} for img in images
        }
        # Python per-iter timings accumulated across passes (one flat list per image).
        python_iter_ms: dict[str, list[float]] = {img.name: [] for img in images}
        python_last_dets: dict[str, list[dict]] = {}

        # Pin Python threads if rigorous (so torch doesn't oversubscribe; the
        # legacy mode left this to user env, which led to inconsistent T).
        if py_model is not None:
            import torch
            torch.set_num_threads(args.threads)
            print(f"[python] torch.set_num_threads({args.threads})", file=sys.stderr)

        for pass_idx in range(args.passes):
            print(f"\n--- pass {pass_idx + 1}/{args.passes} ---", file=sys.stderr)
            for img in images:
                # Python arm first (so we get the same cooldown discipline as C++).
                if py_model is not None:
                    print(f"[pass {pass_idx+1}] [python] {img.name} "
                          f"warmup={args.warmup} iters={args.iters}",
                          file=sys.stderr)
                    ms_list, py_dets = time_one_python(
                        py_model, img, args.iters, args.warmup)
                    python_iter_ms[img.name].extend(ms_list)
                    python_last_dets[img.name] = py_dets
                    s = sorted(ms_list)
                    print(f"  pass median={s[len(s)//2]:.1f} ms "
                          f"min={s[0]:.1f} max={s[-1]:.1f}", file=sys.stderr)
                    cooldown_sleep(args.cooldown, f"pass{pass_idx+1}/{img.name}/python")

                for label, path in cpp_impls:
                    print(f"[pass {pass_idx+1}] [{label} T={args.threads}] {img.name}",
                          file=sys.stderr)
                    cell_pass = time_one_cpp(
                        cli, path, img, args.iters, args.warmup,
                        args.threads, taskset=args.taskset)
                    per_cell[img.name].setdefault(label, []).append(cell_pass)
                    print(f"  pass median={cell_pass['median_ms']:.1f} ms "
                          f"min={cell_pass['min_ms']:.1f} "
                          f"max={cell_pass['max_ms']:.1f}", file=sys.stderr)
                    cooldown_sleep(args.cooldown,
                                   f"pass{pass_idx+1}/{img.name}/{label}")

            # Persist partial progress after every pass (so a crash mid-bench
            # doesn't lose the data we already have).
            partial = {}
            for img_name, impl_passes in per_cell.items():
                cell: dict = {"image": img_name}
                for impl, passes_list in impl_passes.items():
                    if passes_list:
                        cell[impl] = aggregate_across_passes(passes_list)
                if python_iter_ms.get(img_name):
                    cell["python"] = trimmed_aggregate(python_iter_ms[img_name])
                partial[img_name] = cell
            data["per_image"] = partial
            out_path.write_text(json.dumps(data, indent=2))

        # Final aggregate.
        for img in images:
            cell = {"image": img.name}
            for impl, passes_list in per_cell[img.name].items():
                cell[impl] = aggregate_across_passes(passes_list)
            if python_iter_ms.get(img.name):
                cell["python"] = trimmed_aggregate(python_iter_ms[img.name])
                data["detections"].setdefault(img.name, {})["python"] = \
                    python_last_dets.get(img.name, [])
            data["per_image"][img.name] = cell

            # Per-cell IQR sanity print
            for impl in ["python"] + [l for l, _ in cpp_impls]:
                if impl in cell and isinstance(cell[impl], dict) and "iqr_pct" in cell[impl]:
                    iqr = cell[impl]["iqr_pct"]
                    med = cell[impl]["median_ms"]
                    flag = "" if iqr < 10 else "  [HIGH-IQR]"
                    print(f"  {img.name:25s} {impl:8s} "
                          f"median={med:6.1f} ms  IQR={iqr:4.1f}%{flag}",
                          file=sys.stderr)

        # One-shot detection run for the cross-check (correctness data only —
        # one run per cell is fine for this).
        print("\n=== detections (single run per cell, for accuracy cross-check) ===",
              file=sys.stderr)
        for img in images:
            data["detections"].setdefault(img.name, {})
            print(f"[detect cpp_f32] {img.name}", file=sys.stderr)
            data["detections"][img.name]["cpp_f32"] = run_cpp_detect(cli, f32, img, args.threads)
            if have_f16:
                print(f"[detect cpp_f16] {img.name}", file=sys.stderr)
                data["detections"][img.name]["cpp_f16"] = run_cpp_detect(cli, f16, img, args.threads)
            print(f"[detect cpp_q8 ] {img.name}", file=sys.stderr)
            data["detections"][img.name]["cpp_q8"] = run_cpp_detect(cli, q8, img, args.threads)
            if have_q5:
                data["detections"][img.name]["cpp_q5"] = run_cpp_detect(cli, q5, img, args.threads)
            if have_q4:
                data["detections"][img.name]["cpp_q4"] = run_cpp_detect(cli, q4, img, args.threads)
            for label, path, have in [
                ("cpp_q4K", q4K, have_q4K),
                ("cpp_q5K", q5K, have_q5K),
                ("cpp_q6K", q6K, have_q6K),
            ]:
                if have:
                    data["detections"][img.name][label] = run_cpp_detect(cli, path, img, args.threads)

        out_path.write_text(json.dumps(data, indent=2))

    else:
        # ---- Legacy mode (the original linear sweep) ------------------------
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

            for label, path in cpp_impls:
                print(f"[{label} T={args.threads}]", file=sys.stderr)
                cell[label] = time_one_cpp(cli, path, img, args.iters,
                                            args.warmup, args.threads)
                print(f"[{label}] median={cell[label]['median_ms']:.1f} ms "
                      f"min={cell[label]['min_ms']:.1f} "
                      f"max={cell[label]['max_ms']:.1f}", file=sys.stderr)

            # detections (single run each)
            data["detections"].setdefault(img.name, {})
            print(f"[detect cpp_f32]", file=sys.stderr)
            data["detections"][img.name]["cpp_f32"] = run_cpp_detect(cli, f32, img, args.threads)
            if have_f16:
                print(f"[detect cpp_f16]", file=sys.stderr)
                data["detections"][img.name]["cpp_f16"] = run_cpp_detect(cli, f16, img, args.threads)
            print(f"[detect cpp_q8 ]", file=sys.stderr)
            data["detections"][img.name]["cpp_q8" ] = run_cpp_detect(cli, q8,  img, args.threads)
            if have_q5:
                data["detections"][img.name]["cpp_q5" ] = run_cpp_detect(cli, q5, img, args.threads)
            if have_q4:
                data["detections"][img.name]["cpp_q4" ] = run_cpp_detect(cli, q4, img, args.threads)
            for label, path, have in [
                ("cpp_q4K", q4K, have_q4K),
                ("cpp_q5K", q5K, have_q5K),
                ("cpp_q6K", q6K, have_q6K),
            ]:
                if have:
                    data["detections"][img.name][label] = run_cpp_detect(cli, path, img, args.threads)

            data["per_image"][img.name] = cell
            # Persist after each image so progress isn't lost
            out_path.write_text(json.dumps(data, indent=2))

    # -------- thread sweep on ONE image --------
    if not args.skip_sweep:
        print(f"\n=== thread sweep on {sweep_image.name}: T in {sweep_threads} ===",
              file=sys.stderr)
        sweep = {"image": sweep_image.name, "threads": sweep_threads,
                 "cpp_f32": {}, "cpp_q8": {}, "python": {}}
        if have_f16:
            sweep["cpp_f16"] = {}

        sweep_cooldown = args.cooldown if args.rigorous else 0.0
        sweep_taskset  = args.taskset if args.rigorous else ""

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
                sweep["python"][str(n)] = (
                    trimmed_aggregate(ms_list) if args.rigorous
                    else aggregate_python(ms_list)
                )
                print(f"  median={sweep['python'][str(n)]['median_ms']:.1f} ms",
                      file=sys.stderr)
                cooldown_sleep(sweep_cooldown, f"sweep python T={n}")

        for n in sweep_threads:
            print(f"[cpp_f32 T={n}]", file=sys.stderr)
            sweep["cpp_f32"][str(n)] = time_one_cpp(
                cli, f32, sweep_image, args.iters, args.warmup, n,
                taskset=sweep_taskset)
            print(f"  median={sweep['cpp_f32'][str(n)]['median_ms']:.1f} ms",
                  file=sys.stderr)
            cooldown_sleep(sweep_cooldown, f"sweep cpp_f32 T={n}")
        if have_f16:
            for n in sweep_threads:
                print(f"[cpp_f16 T={n}]", file=sys.stderr)
                sweep["cpp_f16"][str(n)] = time_one_cpp(
                    cli, f16, sweep_image, args.iters, args.warmup, n,
                    taskset=sweep_taskset)
                print(f"  median={sweep['cpp_f16'][str(n)]['median_ms']:.1f} ms",
                      file=sys.stderr)
                cooldown_sleep(sweep_cooldown, f"sweep cpp_f16 T={n}")
        for n in sweep_threads:
            print(f"[cpp_q8  T={n}]", file=sys.stderr)
            sweep["cpp_q8"][str(n)] = time_one_cpp(
                cli, q8, sweep_image, args.iters, args.warmup, n,
                taskset=sweep_taskset)
            print(f"  median={sweep['cpp_q8'][str(n)]['median_ms']:.1f} ms",
                  file=sys.stderr)
            cooldown_sleep(sweep_cooldown, f"sweep cpp_q8 T={n}")

        data["thread_sweep"] = sweep
        out_path.write_text(json.dumps(data, indent=2))

    # -------- per-variant headline (Nano/Small/Medium/Large; Base implied) --------
    # Each cell: C++ F32 + PyTorch RFDETR{Variant}() median latency on the
    # variant_sweep_image. We don't re-loop over all images here -- the
    # headline-per-image story is told by the quant per_image section.
    # Schema: data['variants'][variant] = {
    #     'cpp_f32': {min/median/mean/max...},
    #     'python':  {min/median/mean/max...},
    #     'detections': {'cpp_f32': [...], 'python': [...]},
    #     'gguf_size_bytes': N,
    # }
    if variant_specs:
        try:
            from rfdetr import (
                RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge,
            )
            _PY_VARIANT_CLASSES = {
                "nano":   RFDETRNano,
                "small":  RFDETRSmall,
                "base":   RFDETRBase,
                "medium": RFDETRMedium,
                "large":  RFDETRLarge,
            }
        except ImportError:
            _PY_VARIANT_CLASSES = {}

        sweep_img_path = idir / args.variant_sweep_image
        if not sweep_img_path.exists():
            sweep_img_path = images[0]

        for v, gguf in variant_specs:
            print(f"\n=== variant {v}: image={sweep_img_path.name} ===",
                  file=sys.stderr)
            cell: dict = {"variant": v,
                          "image": sweep_img_path.name,
                          "gguf_path": str(gguf),
                          "gguf_size_bytes": gguf.stat().st_size}

            # C++ F32
            print(f"[cpp_f32 T={args.threads}]", file=sys.stderr)
            cell["cpp_f32"] = time_one_cpp(cli, gguf, sweep_img_path,
                                            args.iters, args.warmup, args.threads)
            print(f"  median={cell['cpp_f32']['median_ms']:.1f} ms "
                  f"min={cell['cpp_f32']['min_ms']:.1f} max={cell['cpp_f32']['max_ms']:.1f}",
                  file=sys.stderr)

            # Detection count from C++ F32 (used by the plot)
            cell["detections"] = {}
            cell["detections"]["cpp_f32"] = run_cpp_detect(
                cli, gguf, sweep_img_path, args.threads)

            # PyTorch arm (load + bench once per variant)
            if not args.skip_python and _PY_VARIANT_CLASSES and v in _PY_VARIANT_CLASSES:
                print(f"[python warmup={args.warmup}, iters={args.iters}]",
                      file=sys.stderr)
                py_model_v = _PY_VARIANT_CLASSES[v]()
                ms_list, py_dets = time_one_python(py_model_v, sweep_img_path,
                                                    args.iters, args.warmup)
                cell["python"] = aggregate_python(ms_list)
                cell["detections"]["python"] = py_dets
                print(f"  median={cell['python']['median_ms']:.1f} ms "
                      f"min={cell['python']['min_ms']:.1f} max={cell['python']['max_ms']:.1f}",
                      file=sys.stderr)
                # Release the Python model immediately so the next variant
                # has memory headroom.
                del py_model_v
                gc.collect()

            data["variants"][v] = cell
            # Persist after each variant
            out_path.write_text(json.dumps(data, indent=2))

    # -------- detection match summary --------
    match_summary = {}
    for img_name, dets in data["detections"].items():
        if "python" not in dets:
            continue
        py_dets = dets["python"]
        for impl in ("cpp_f32", "cpp_f16", "cpp_q8", "cpp_q5", "cpp_q4",
                     "cpp_q4K", "cpp_q5K", "cpp_q6K"):
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
