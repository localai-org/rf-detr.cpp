#!/usr/bin/env python3
"""Compare actual CI detection output against committed reference.

Used by .github/workflows/ci.yml to catch regressions in detection
behavior across quantization levels. Exits 1 on any failure.

Matching strategy:
  - For each reference detection (in descending-score order), find the
    actual detection of the same class with the highest IoU among those
    not already matched.
  - Pair is valid if IoU >= iou_threshold.
  - Pair must satisfy score-delta and bbox-delta tolerances.
  - Extra actual detections (no matching ref) are warnings, not failures
    — Q4_K and similar quants may produce 1 extra borderline detection
    that vanishes on a different machine; that's noise, not a regression.

Tolerance (defaults):
  - IoU threshold:    0.5  (same class + spatial overlap is a "match")
  - score delta:      0.05
  - bbox edge delta:  1.0 px
"""
import json
import sys
import argparse
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def iou(b1, b2):
    """Standard IoU on [x1, y1, x2, y2] boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def match(actual_dets, ref_dets, score_tol, bbox_tol, iou_threshold=0.5):
    """Match by class + IoU greedy. Each ref detection (descending score)
    tries to find its best-IoU actual detection of the same class that
    isn't already matched.

    Failures (these break CI):
      - Any unmatched ref detection (missing class or no IoU >= threshold)
      - Any matched pair where score delta exceeds tolerance
      - Any matched pair where any bbox edge exceeds bbox_tol

    Warnings (don't break CI):
      - Extra actual detections that don't match any ref
    """
    failures = []
    warnings = []

    # Sort refs by descending score (greedy assigns high-confidence refs first)
    ref_sorted = sorted(enumerate(ref_dets), key=lambda x: -x[1]["score"])
    matched_actual_idx = set()

    for ref_idx, r in ref_sorted:
        # Candidates: same class, not yet matched
        candidates = [
            (i, a)
            for i, a in enumerate(actual_dets)
            if i not in matched_actual_idx and a["class_id"] == r["class_id"]
        ]
        if not candidates:
            failures.append(
                f"ref det[{ref_idx}] class={r['class_id']} score={r['score']:.3f} "
                f"has no actual match (no remaining detection of that class)"
            )
            continue

        # Pick the candidate with the highest IoU
        best_iou = 0.0
        best_idx = -1
        for i, a in candidates:
            io = iou(a["bbox"], r["bbox"])
            if io > best_iou:
                best_iou = io
                best_idx = i

        if best_iou < iou_threshold:
            failures.append(
                f"ref det[{ref_idx}] class={r['class_id']} score={r['score']:.3f}: "
                f"best IoU {best_iou:.3f} < {iou_threshold}"
            )
            continue

        matched_actual_idx.add(best_idx)
        a = actual_dets[best_idx]
        ds = abs(a["score"] - r["score"])
        if ds > score_tol:
            failures.append(
                f"ref det[{ref_idx}] class={r['class_id']}: "
                f"score delta={ds:.4f} > tol {score_tol} "
                f"(actual={a['score']:.4f}, ref={r['score']:.4f})"
            )
        for j, (av, rv) in enumerate(zip(a["bbox"], r["bbox"])):
            if abs(av - rv) > bbox_tol:
                failures.append(
                    f"ref det[{ref_idx}] class={r['class_id']}: "
                    f"bbox[{j}]={av:.2f} vs ref {rv:.2f}, "
                    f"delta={abs(av - rv):.2f} > tol {bbox_tol}"
                )

    # Extra actual detections — warning only (tolerate borderline extras)
    extras = len(actual_dets) - len(matched_actual_idx)
    if extras > 0:
        extra_dets = [
            f"class={a['class_id']} score={a['score']:.3f}"
            for i, a in enumerate(actual_dets)
            if i not in matched_actual_idx
        ]
        warnings.append(
            f"{extras} extra actual detection(s) without a ref match: "
            + ", ".join(extra_dets)
        )

    return failures, warnings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actual", required=True, help="Actual JSON output from rfdetr-cli detect")
    p.add_argument("--reference", required=True, help="Committed reference JSON")
    p.add_argument("--score-tolerance", type=float, default=0.05)
    p.add_argument("--bbox-tolerance", type=float, default=1.0)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--label", default="", help="Label for log output")
    args = p.parse_args()

    actual = load(args.actual)
    ref = load(args.reference)
    failures, warnings = match(
        actual["detections"],
        ref["detections"],
        args.score_tolerance,
        args.bbox_tolerance,
        args.iou_threshold,
    )
    label = args.label or args.actual
    n_ref = len(ref["detections"])
    n_actual = len(actual["detections"])
    n_matched = n_ref - sum(1 for f in failures if "has no actual match" in f or "best IoU" in f)
    n_extra = max(0, n_actual - n_matched)

    # Warnings to stderr but never fail
    for w in warnings:
        print(f"WARN {label}: {w}", file=sys.stderr)

    if failures:
        print(f"FAIL {label}", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print(
        f"OK   {label} - {n_matched}/{n_ref} ref dets matched, "
        f"{n_extra} extras (warning)"
    )


if __name__ == "__main__":
    main()
