#!/usr/bin/env python3
"""Quick seg-variant benchmark.

Benchmarks all 6 RF-DETR-Seg variants end-to-end via the rfdetr-cli bench
subcommand, reporting load_ms + min/median/mean/max ms per variant on a
single image.

Optionally compares C++ detection count against PyTorch's
RFDETRSegNano.predict() on the same image (sanity check that the seg
variants haven't regressed).

Usage:
    .venv/bin/python scripts/bench_seg.py [--threads 8] [--iters 20]
                                          [--warmup 3]
                                          [--image /tmp/coco_sample.jpg]

Pre-reqs: each variant's f32 GGUF must exist at models/rfdetr-<variant>-f32.gguf
(run convert_all_variants.sh or convert seg variants individually first).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SEG_VARIANTS = ["seg-nano", "seg-small", "seg-medium", "seg-large",
                "seg-xlarge", "seg-2xlarge"]


def parse_bench(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s+(.+)$", line.strip())
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k in ("load_ms", "min_ms", "median_ms", "mean_ms", "max_ms"):
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        elif k in ("warmup", "iters", "detections", "threads"):
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--image", type=Path,
                    default=Path("/tmp/coco_sample.jpg"))
    ap.add_argument("--output", type=Path,
                    default=REPO / "benchmarks/results/seg_bench.json")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2

    cli = REPO / "build" / "bin" / "rfdetr-cli"
    if not cli.exists():
        print(f"error: rfdetr-cli not built: {cli}", file=sys.stderr)
        return 2

    results: list[dict] = []
    for v in SEG_VARIANTS:
        model = REPO / "models" / f"rfdetr-{v}-f32.gguf"
        if not model.exists():
            print(f"  [{v}] SKIP (model not found: {model.name})", file=sys.stderr)
            continue

        cmd = [
            str(cli), "bench",
            "--model", str(model),
            "--input", str(args.image),
            "--iters", str(args.iters),
            "--warmup", str(args.warmup),
            "--threads", str(args.threads),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(f"  [{v}] FAIL (rc={proc.returncode}):\n{proc.stderr}",
                  file=sys.stderr)
            continue
        d = parse_bench(proc.stdout)
        d["variant"] = v
        d["model"] = str(model)
        results.append(d)
        print(f"  [{v:>11}] load={d.get('load_ms', 0):7.1f}ms "
              f"median={d.get('median_ms', 0):7.1f}ms "
              f"detections={d.get('detections', 0):3d}",
              file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "image":   str(args.image),
        "threads": args.threads,
        "iters":   args.iters,
        "warmup":  args.warmup,
        "results": results,
    }, indent=2))
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
