#!/usr/bin/env python3
"""Compare actual CI detection output against committed reference.

Used by .github/workflows/ci.yml to catch regressions in detection
behavior across quantization levels. Exits 1 on any failure.

Tolerance (defaults):
  - exact detection count match
  - exact class ID match (sorted by score desc)
  - bbox delta <= 1.0 px (each side)
  - score delta <= 0.05
"""
import json
import sys
import argparse
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def match(actual_dets, ref_dets, score_tol, bbox_tol):
    """Pair detections sorted by score (desc) and verify each pair within tolerance."""
    failures = []
    if len(actual_dets) != len(ref_dets):
        return [
            f"detection count mismatch: actual={len(actual_dets)}, ref={len(ref_dets)}"
        ]
    # Sort both by score descending for stable pairing
    actual_sorted = sorted(actual_dets, key=lambda d: -d["score"])
    ref_sorted = sorted(ref_dets, key=lambda d: -d["score"])
    for i, (a, r) in enumerate(zip(actual_sorted, ref_sorted)):
        if a["class_id"] != r["class_id"]:
            failures.append(
                f"det[{i}] class mismatch: {a['class_id']} vs ref {r['class_id']}"
            )
            continue
        ds = abs(a["score"] - r["score"])
        if ds > score_tol:
            failures.append(
                f"det[{i}] (class={a['class_id']}) score delta={ds:.4f} > tol {score_tol}"
            )
        for j, (av, rv) in enumerate(zip(a["bbox"], r["bbox"])):
            if abs(av - rv) > bbox_tol:
                failures.append(
                    f"det[{i}] (class={a['class_id']}) bbox[{j}]: {av:.2f} vs ref {rv:.2f}, delta={abs(av-rv):.2f}"
                )
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actual", required=True, help="Actual JSON output from rfdetr-cli detect")
    p.add_argument("--reference", required=True, help="Committed reference JSON")
    p.add_argument("--score-tolerance", type=float, default=0.05)
    p.add_argument("--bbox-tolerance", type=float, default=1.0)
    p.add_argument("--label", default="", help="Label for log output")
    args = p.parse_args()

    actual = load(args.actual)
    ref = load(args.reference)
    failures = match(
        actual["detections"],
        ref["detections"],
        args.score_tolerance,
        args.bbox_tolerance,
    )
    label = args.label or args.actual
    if failures:
        print(f"FAIL {label}", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print(f"OK   {label} - {len(actual['detections'])} detections match")


if __name__ == "__main__":
    main()
