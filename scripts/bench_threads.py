#!/usr/bin/env python3
"""Thread-scaling benchmark for rfdetr.cpp.

Runs the C++ `rfdetr-cli bench` over a sweep of `--threads` values (and both
F32 and Q8_0 model variants) on a set of test images. Emits a markdown report
of per-(image, dtype, threads) median/min/mean latencies and the implied
scaling factor against the single-thread baseline.

Python rfdetr is intentionally NOT re-run here — re-use the numbers from
BENCHMARK.md if you need a cross-impl comparison.

Usage:
    .venv/bin/python scripts/bench_threads.py \\
        --image /tmp/coco_sample.jpg \\
        --image /tmp/coco_sample2.jpg \\
        --image /tmp/bus.jpg \\
        --iters 5 --warmup 2 \\
        --threads 1 --threads 4 --threads 8 --threads 16 \\
        --out /tmp/bench_threads.md
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

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
        raise RuntimeError(f"C++ bench failed (rc={proc.returncode}):\n{proc.stderr}")
    return parse_bench_stdout(proc.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli",      default=str(REPO / "build/bin/rfdetr-cli"))
    ap.add_argument("--f32",      default=str(REPO / "models/rfdetr-base-f32.gguf"))
    ap.add_argument("--q8",       default=str(REPO / "models/rfdetr-base-q8_0.gguf"))
    ap.add_argument("--image",    action="append", required=True)
    ap.add_argument("--iters",    type=int, default=5)
    ap.add_argument("--warmup",   type=int, default=2)
    ap.add_argument("--threads",  action="append", type=int, default=None,
                    help="Thread count to test (repeatable). Default: 1,4,8,nproc.")
    ap.add_argument("--skip-q8",  action="store_true")
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()

    thread_list = args.threads
    if not thread_list:
        thread_list = [1, 4, 8, os.cpu_count() or 1]
    # Dedup and sort
    thread_list = sorted(set(thread_list))

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

    # results[image_name][dtype][threads] = bench_stdout dict
    results: dict = {}
    for img in images:
        results[img.name] = {"f32": {}, "q8": {}}
        for n in thread_list:
            print(f"=== {img.name} | f32 | threads={n} ===", file=sys.stderr)
            results[img.name]["f32"][n] = run_cpp_bench(
                cli, f32, img, args.iters, args.warmup, n)
            if not args.skip_q8:
                print(f"=== {img.name} | q8  | threads={n} ===", file=sys.stderr)
                results[img.name]["q8"][n] = run_cpp_bench(
                    cli, q8, img, args.iters, args.warmup, n)

    # Render markdown
    lines: list[str] = []
    lines.append("# rfdetr.cpp — thread-scaling benchmark\n")
    lines.append(f"CPU cores reported: `os.cpu_count() = {os.cpu_count()}`  ")
    lines.append(f"Iterations: {args.iters} timed + {args.warmup} warmup per cell  ")
    lines.append(f"Threads swept: {thread_list}\n")

    dtypes = [("f32", "F32")]
    if not args.skip_q8:
        dtypes.append(("q8", "Q8_0"))

    for dtype_key, dtype_label in dtypes:
        lines.append(f"## {dtype_label}\n")
        lines.append("| image | threads | min_ms | median_ms | mean_ms | max_ms | speedup vs T=1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for img in images:
            base = results[img.name][dtype_key].get(thread_list[0], None)
            base_median = base["median_ms"] if base else None
            for n in thread_list:
                d = results[img.name][dtype_key][n]
                speed = (f"{base_median / d['median_ms']:.2f}x"
                         if base_median is not None else "—")
                lines.append(
                    f"| {img.name} | {n} | {d['min_ms']:.1f} | "
                    f"{d['median_ms']:.1f} | {d['mean_ms']:.1f} | "
                    f"{d['max_ms']:.1f} | {speed} |"
                )
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"[bench-threads] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
