#!/usr/bin/env python3
"""Publish rfdetr.cpp GGUF models to HuggingFace Hub.

Creates one repo per variant under ``mudler/rfdetr-cpp-<variant>``, each
containing the F32 / F16 / Q8_0 / Q4_K quantizations plus a data-driven
model card backed by the Phase 2 accuracy sweep and the per-cell latency
microbench.

Usage:
    .venv/bin/python scripts/publish_hf.py [--dry-run]
                                           [--only nano,base,...]
                                           [--skip-uploads]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
ACCURACY_JSON = RESULTS_DIR / "accuracy_sweep.json"
LATENCY_JSON = RESULTS_DIR / "per_cell_latency.json"

HF_USER = "mudler"
QUANTS = ["f32", "f16", "q8_0", "q4_K"]

# All 8 variants, in publish order.
VARIANTS = [
    "nano",
    "small",
    "base",
    "medium",
    "large",
    "seg-nano",
    "seg-small",
    "seg-medium",
]

# Static per-variant architecture facts (resolution, decoder layers, queries,
# patch size). These come from the upstream rfdetr 1.7.0 model registry —
# they're not changing per quant so we keep them as a small lookup.
ARCH = {
    "nano":       {"resolution": 384, "patch": 14, "decoder_layers": 2, "queries": 300, "backbone": "DINOv2-small"},
    "small":      {"resolution": 512, "patch": 14, "decoder_layers": 3, "queries": 300, "backbone": "DINOv2-small"},
    "base":       {"resolution": 560, "patch": 14, "decoder_layers": 3, "queries": 300, "backbone": "DINOv2-small"},
    "medium":     {"resolution": 576, "patch": 14, "decoder_layers": 4, "queries": 300, "backbone": "DINOv2-small"},
    "large":      {"resolution": 704, "patch": 14, "decoder_layers": 4, "queries": 300, "backbone": "DINOv2-small"},
    "seg-nano":   {"resolution": 312, "patch": 12, "decoder_layers": 4, "queries": 100, "backbone": "DINOv2-small"},
    "seg-small":  {"resolution": 384, "patch": 12, "decoder_layers": 4, "queries": 100, "backbone": "DINOv2-small"},
    "seg-medium": {"resolution": 432, "patch": 12, "decoder_layers": 5, "queries": 200, "backbone": "DINOv2-small"},
}

# Pretty display names.
DISPLAY = {
    "nano": "Nano",
    "small": "Small",
    "base": "Base",
    "medium": "Medium",
    "large": "Large",
    "seg-nano": "Seg-Nano",
    "seg-small": "Seg-Small",
    "seg-medium": "Seg-Medium",
}

UPSTREAM = "https://github.com/roboflow/rf-detr"


@dataclass
class CellMetrics:
    variant: str
    quant: str
    file_size_mb: float
    recall_05: float
    recall_095: float
    max_score_delta: float
    mean_score_delta: float
    extra_cpp: float
    # mask metrics (None for detection-only variants)
    mean_mask_iou: Optional[float] = None
    mean_pixel_agreement: Optional[float] = None
    min_mask_iou: Optional[float] = None
    # latency (ms @ T=8)
    median_ms: Optional[float] = None
    min_ms: Optional[float] = None


def load_metrics() -> Dict[str, Dict[str, CellMetrics]]:
    """Returns metrics[variant][quant] = CellMetrics.

    Note: accuracy_sweep.json uses quant names matching our QUANTS list,
    but the variant/quant key in latency.json is "{variant}/{quant}".
    """
    with open(ACCURACY_JSON) as f:
        sweep = json.load(f)
    with open(LATENCY_JSON) as f:
        latency = json.load(f)

    out: Dict[str, Dict[str, CellMetrics]] = {}
    for cell in sweep["cells"]:
        v = cell["variant"]
        q = cell["quant"]
        m = cell["metrics"]
        key = f"{v}/{q}"
        lat = latency.get(key, {})
        cm = CellMetrics(
            variant=v,
            quant=q,
            file_size_mb=cell["file_size_mb"],
            recall_05=m["recall_iou_0.5"],
            recall_095=m["recall_iou_0.95"],
            max_score_delta=m["max_abs_score_delta"],
            mean_score_delta=m["mean_abs_score_delta"],
            extra_cpp=m["extra_cpp_detections"],
            mean_mask_iou=m.get("mean_mask_iou"),
            mean_pixel_agreement=m.get("mean_pixel_agreement"),
            min_mask_iou=m.get("min_mask_iou"),
            median_ms=lat.get("median_ms"),
            min_ms=lat.get("min_ms"),
        )
        out.setdefault(v, {})[q] = cm
    return out


def build_model_card(variant: str, cells: Dict[str, CellMetrics]) -> str:
    """Generate a Markdown model card for one variant repo."""
    is_seg = variant.startswith("seg-")
    pretty = DISPLAY[variant]
    arch = ARCH[variant]
    pipeline_tag = "image-segmentation" if is_seg else "object-detection"
    seg_tag = "image-segmentation" if is_seg else "object-detection"

    # YAML frontmatter
    tags = [
        "object-detection",
        "rfdetr",
        "gguf",
        "ggml",
        "cpp-inference",
    ]
    if is_seg:
        tags.append("image-segmentation")
        tags.append("instance-segmentation")

    yaml = ["---", "license: apache-2.0", "library_name: rfdetr.cpp"]
    yaml.append("tags:")
    for t in tags:
        yaml.append(f"  - {t}")
    yaml.append(f"pipeline_tag: {pipeline_tag}")
    yaml.append("base_model: roboflow/rfdetr")
    yaml.append("---")
    yaml.append("")

    lines: List[str] = []
    lines.extend(yaml)

    kind = "segmentation" if is_seg else "detection"
    lines.append(f"# RF-DETR {pretty} — GGUF for rfdetr.cpp")
    lines.append("")
    lines.append(
        f"GGUF-format weights of [Roboflow RF-DETR {pretty}]({UPSTREAM}) "
        f"({kind} variant) for use with [rfdetr.cpp](https://github.com/mudler/rf-detr.cpp), "
        f"a C++/ggml implementation that matches the upstream PyTorch model on CPU."
    )
    lines.append("")
    lines.append("This repo contains all four standard quantizations of this variant. "
                 "**F16 is the recommended default** — same accuracy as F32, "
                 "1.85× smaller, and typically the fastest on modern CPUs thanks to "
                 "ggml's F32×F16 matmul fast path.")
    lines.append("")

    # File table
    lines.append("## Available files")
    lines.append("")
    if is_seg:
        lines.append("| File | Quant | Size (MB) | Recall @ IoU 0.5 | Recall @ IoU 0.95 | Mean mask IoU | Pixel agreement | Latency (median ms, T=8) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| File | Quant | Size (MB) | Recall @ IoU 0.5 | Recall @ IoU 0.95 | Mean \\|Δscore\\| | Latency (median ms, T=8) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")

    quant_label = {
        "f32": "F32",
        "f16": "F16",
        "q8_0": "Q8_0",
        "q4_K": "Q4_K",
    }
    recommended = "f16"
    for q in QUANTS:
        if q not in cells:
            continue
        c = cells[q]
        fname = f"rfdetr-{variant}-{q}.gguf"
        rec_marker = " ← **recommended**" if q == recommended else ""
        if c.median_ms is not None:
            lat = f"{c.median_ms:.1f}"
        else:
            lat = "—"
        if is_seg:
            mask_iou = f"{c.mean_mask_iou:.4f}" if c.mean_mask_iou is not None else "—"
            pix = f"{c.mean_pixel_agreement:.4f}" if c.mean_pixel_agreement is not None else "—"
            lines.append(
                f"| `{fname}`{rec_marker} | {quant_label[q]} | {c.file_size_mb:.1f} "
                f"| {c.recall_05:.4f} | {c.recall_095:.4f} | {mask_iou} | {pix} | {lat} |"
            )
        else:
            lines.append(
                f"| `{fname}`{rec_marker} | {quant_label[q]} | {c.file_size_mb:.1f} "
                f"| {c.recall_05:.4f} | {c.recall_095:.4f} | {c.mean_score_delta:.4f} | {lat} |"
            )
    lines.append("")

    # Methodology note
    lines.append("All accuracy numbers are computed against the upstream PyTorch "
                 "reference (`rfdetr 1.7.0`) on 7 COCO val2017 images at threshold "
                 "0.5. Latency is measured with `rfdetr-cli bench` (8 iters + 3 "
                 "warmup) at T=8 threads on a single AMD Ryzen 9 9950X3D image "
                 "(`coco_kitchen.jpg`, 640x427).")
    lines.append("")

    # Architecture
    lines.append("## Architecture")
    lines.append("")
    lines.append(f"- Backbone: {arch['backbone']}")
    lines.append(f"- Input resolution: {arch['resolution']}×{arch['resolution']}")
    lines.append(f"- Patch size: {arch['patch']}")
    lines.append(f"- Decoder layers: {arch['decoder_layers']}")
    lines.append(f"- Object queries: {arch['queries']}")
    lines.append(f"- Task: {'instance segmentation (boxes + per-query masks)' if is_seg else 'object detection (boxes only)'}")
    if is_seg:
        lines.append(f"- Mask resolution: {arch['resolution'] // 4}×{arch['resolution'] // 4} per query (image_size / 4)")
    lines.append("")

    # Quant notes
    lines.append("## Quantization notes")
    lines.append("")
    lines.append("- **F32** — full-precision reference, ~120 MB. Bit-exact PyTorch parity.")
    lines.append("- **F16** — matmul-multiplicand weights only; LayerNorms, conv kernels, "
                 "embeddings, biases, and layer-scale gammas stay F32. Lossless on this "
                 "model and consistently the fastest variant on CPU.")
    lines.append("- **Q8_0** — best size/accuracy tradeoff under F16; ~3× smaller than F32 "
                 "with effectively identical detections.")
    lines.append("- **Q4_K** — smallest practical quant. Rows with `ne[0] % 256 != 0` (the "
                 "decoder's 128-dim MLP halves, 60 tensors) silently fall back to Q8_0 per "
                 "ggml's quantizer logic — net compression is still ~3.8× over F32. "
                 "Use only when the size budget is tight; expect a measurable Recall@0.95 "
                 "drop relative to F16/Q8_0 (see file table above).")
    lines.append("")

    # Usage
    lines.append("## Usage")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Clone + build rfdetr.cpp")
    lines.append("git clone https://github.com/mudler/rf-detr.cpp")
    lines.append("cd rt-detr.cpp")
    lines.append("cmake -B build -DRFDETR_BUILD_CLI=ON && cmake --build build -j")
    lines.append("")
    lines.append("# 2. Download a quant (F16 recommended)")
    lines.append(f"hf download {HF_USER}/rfdetr-cpp-{variant} rfdetr-{variant}-f16.gguf --local-dir models/")
    lines.append("")
    if is_seg:
        lines.append(f"# 3. Run segmentation (writes per-detection PNG masks to /tmp/seg_masks/)")
        lines.append("build/bin/rfdetr-cli detect \\")
        lines.append(f"    --model models/rfdetr-{variant}-f16.gguf \\")
        lines.append("    --input my_image.jpg \\")
        lines.append("    --threshold 0.5 --threads 8 \\")
        lines.append("    --masks /tmp/seg_masks \\")
        lines.append("    --output detections.json")
    else:
        lines.append("# 3. Run detection")
        lines.append("build/bin/rfdetr-cli detect \\")
        lines.append(f"    --model models/rfdetr-{variant}-f16.gguf \\")
        lines.append("    --input my_image.jpg \\")
        lines.append("    --threshold 0.5 --threads 8 \\")
        lines.append("    --output detections.json")
    lines.append("```")
    lines.append("")

    # Accuracy methodology
    lines.append("## Accuracy methodology")
    lines.append("")
    lines.append(
        "All accuracy metrics are computed against the upstream PyTorch reference "
        "(rfdetr 1.7.0) on 7 COCO val2017 images at threshold 0.5. Each detection "
        "match uses greedy Hungarian-style assignment by IoU (≥ 0.5 lenient, "
        "≥ 0.95 strict) with class equality required."
    )
    lines.append("")
    if is_seg:
        lines.append(
            "Mask metrics are pixel-wise IoU between binary masks at the **original** "
            "image resolution (not the network's working resolution), after sigmoid + "
            "bicubic upsample of the per-query mask logits. **Pixel agreement** is "
            "the fraction of pixels where the C++ and PyTorch binary masks match."
        )
        lines.append("")
    lines.append(
        "See [BENCHMARK.md](https://github.com/mudler/rf-detr.cpp/blob/main/BENCHMARK.md) "
        "and [`benchmarks/results/accuracy_sweep.json`](https://github.com/mudler/rf-detr.cpp/blob/main/benchmarks/results/accuracy_sweep.json) "
        "for the full sweep across all 32 (variant × quant) cells."
    )
    lines.append("")

    # License
    lines.append("## License")
    lines.append("")
    lines.append("Apache-2.0 — matches the upstream [rfdetr](https://github.com/roboflow/rf-detr) license.")
    lines.append("")

    return "\n".join(lines)


def list_local_files(variant: str) -> List[Path]:
    """Return the 4 GGUF files for this variant, in QUANTS order."""
    out: List[Path] = []
    for q in QUANTS:
        p = MODELS_DIR / f"rfdetr-{variant}-{q}.gguf"
        if not p.is_file():
            raise FileNotFoundError(f"missing model file: {p}")
        out.append(p)
    return out


def upload_with_retry(api: HfApi, local_path: Path, remote_name: str, repo_id: str, max_retries: int = 3) -> None:
    """Upload a single file with exponential backoff."""
    size_mb = local_path.stat().st_size / 1e6
    print(f"  -> {remote_name} ({size_mb:.1f} MB)... ", end="", flush=True)
    delay = 2.0
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="model",
            )
            dt = time.time() - t0
            mbps = size_mb / dt if dt > 0 else 0
            print(f"ok ({dt:.1f}s, {mbps:.1f} MB/s)")
            return
        except (HfHubHTTPError, OSError, ConnectionError) as e:
            last_err = e
            print(f"\n     attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"upload failed after {max_retries} attempts: {last_err}") from last_err


def publish_variant(api: HfApi, variant: str, cells: Dict[str, CellMetrics], *,
                    dry_run: bool, skip_existing: bool) -> Dict[str, object]:
    """Publish one variant. Returns a status dict for the summary table."""
    repo_id = f"{HF_USER}/rfdetr-cpp-{variant}"
    print(f"\n=== {repo_id} ===")
    local_files = list_local_files(variant)
    card = build_model_card(variant, cells)

    total_bytes = sum(p.stat().st_size for p in local_files)
    print(f"  4 GGUF files, total {total_bytes / 1e6:.1f} MB")

    if dry_run:
        print(f"  [dry-run] would create repo {repo_id} (public)")
        print(f"  [dry-run] would upload README.md ({len(card)} bytes)")
        for p in local_files:
            print(f"  [dry-run] would upload {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
        return {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "uploaded_files": 0,
            "bytes": 0,
            "dry_run": True,
        }

    # Create repo (idempotent)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    # List existing remote files for skip-existing
    existing: set[str] = set()
    if skip_existing:
        try:
            existing = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
            if existing:
                print(f"  existing files in repo: {sorted(existing)}")
        except HfHubHTTPError as e:
            print(f"  could not list existing files (will re-upload all): {e}", file=sys.stderr)
            existing = set()

    uploaded = 0
    bytes_up = 0

    # Upload README first (so the repo isn't empty in the UI mid-flight)
    readme_path = f"README.md"
    # Always overwrite README — it's small, and we may have new accuracy data.
    print(f"  uploading model card README.md ({len(card)} bytes)")
    delay = 2.0
    for attempt in range(3):
        try:
            api.upload_file(
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo=readme_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add/update model card for {variant}",
            )
            print(f"     ok")
            break
        except HfHubHTTPError as e:
            print(f"     attempt {attempt + 1}/3 failed: {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
            else:
                raise
    # README upload doesn't count in the 32 GGUF count, but we track bytes
    bytes_up += len(card)
    uploaded += 1

    # Upload GGUFs
    for p in local_files:
        if p.name in existing:
            print(f"  -> {p.name} already in repo, skipping")
            continue
        upload_with_retry(api, p, p.name, repo_id)
        uploaded += 1
        bytes_up += p.stat().st_size

    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "uploaded_files": uploaded,
        "bytes": bytes_up,
        "dry_run": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would happen without uploading")
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated subset of variants to publish")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="re-upload files even if already in the remote repo")
    args = parser.parse_args()

    if not ACCURACY_JSON.is_file():
        print(f"error: missing {ACCURACY_JSON}", file=sys.stderr)
        return 2
    if not LATENCY_JSON.is_file():
        print(f"error: missing {LATENCY_JSON} — run scripts/quick_bench.sh first", file=sys.stderr)
        return 2

    metrics = load_metrics()

    variants = VARIANTS
    if args.only:
        wanted = {v.strip() for v in args.only.split(",") if v.strip()}
        variants = [v for v in VARIANTS if v in wanted]
        if not variants:
            print(f"error: --only={args.only} matched no known variants", file=sys.stderr)
            return 2

    # Auth check
    api = HfApi()
    if not args.dry_run:
        try:
            me = api.whoami()
            print(f"authenticated as: {me['name']}")
        except Exception as e:
            print(f"error: HF auth failed: {e}", file=sys.stderr)
            return 2

    results: List[Dict[str, object]] = []
    t_start = time.time()
    for v in variants:
        cells = metrics.get(v)
        if not cells:
            print(f"warning: no metrics for variant {v}, skipping", file=sys.stderr)
            continue
        try:
            r = publish_variant(api, v, cells,
                                dry_run=args.dry_run,
                                skip_existing=not args.no_skip_existing)
            results.append(r)
        except Exception as e:
            print(f"\n!!! variant {v} failed: {type(e).__name__}: {e}", file=sys.stderr)
            results.append({"repo_id": f"{HF_USER}/rfdetr-cpp-{v}", "url": "", "uploaded_files": 0, "bytes": 0, "error": str(e)})

    elapsed = time.time() - t_start

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Repo':45s} {'Files':>5s} {'Bytes':>11s}")
    total_files = 0
    total_bytes = 0
    for r in results:
        files_ = r.get("uploaded_files", 0)
        bytes_ = r.get("bytes", 0)
        total_files += files_ if isinstance(files_, int) else 0
        total_bytes += bytes_ if isinstance(bytes_, int) else 0
        url = r.get("url", "")
        err = r.get("error")
        marker = "  ERROR" if err else ""
        print(f"{r['repo_id']:45s} {files_:>5} {bytes_:>11,}{marker}")
        if url:
            print(f"  {url}")
        if err:
            print(f"  {err}")
    print()
    print(f"Total uploaded: {total_files} files, {total_bytes / 1e6:.1f} MB")
    print(f"Elapsed: {elapsed:.1f}s")

    failed = sum(1 for r in results if r.get("error"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
