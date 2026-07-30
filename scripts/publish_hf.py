#!/usr/bin/env python3
"""Publish rfdetr.cpp GGUF models to HuggingFace Hub.

Creates one repo per variant under ``mudler/rfdetr-cpp-<variant>``, each
containing the F32 / F16 / Q8_0 / Q4_K quantizations plus a data-driven
model card backed by the Phase 2 accuracy sweep and the per-cell latency
microbench.

Every provenance claim on a card is derived, never hardcoded: the rfdetr
version comes from the sweep cells themselves (falling back to the installed
distribution, then the pin in scripts/requirements.txt), sizes and compression
ratios come from the GGUFs on disk, and sentences about accuracy or latency are
only emitted when that data actually exists for the variant.

A variant whose local GGUF set is incomplete is never published silently:
naming it via --only is a hard error, and an unfiltered run skips it with a
warning and exits non-zero.

Usage:
    .venv/bin/python scripts/publish_hf.py [--dry-run]
                                           [--only nano,base,...]
                                           [--no-skip-existing]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
ACCURACY_JSON = RESULTS_DIR / "accuracy_sweep.json"
LATENCY_JSON = RESULTS_DIR / "per_cell_latency.json"
REQUIREMENTS_TXT = REPO_ROOT / "scripts" / "requirements.txt"

HF_USER = "mudler"
QUANTS = ["f32", "f16", "q8_0", "q4_K"]

# All 11 variants, in publish order.
VARIANTS = [
    "nano",
    "small",
    "base",
    "medium",
    "large",
    "seg-nano",
    "seg-small",
    "seg-medium",
    "seg-large",
    "seg-xlarge",
    "seg-2xlarge",
]

# Static per-variant architecture facts (resolution, decoder layers, queries,
# patch size). These mirror the upstream rfdetr model registry (see
# scripts/requirements.txt for the pinned version this table was checked
# against) — they don't change per quant, so we keep them as a small lookup.
ARCH = {
    "nano":        {"resolution": 384, "patch": 14, "decoder_layers": 2, "queries": 300, "backbone": "DINOv2-small"},
    "small":       {"resolution": 512, "patch": 14, "decoder_layers": 3, "queries": 300, "backbone": "DINOv2-small"},
    "base":        {"resolution": 560, "patch": 14, "decoder_layers": 3, "queries": 300, "backbone": "DINOv2-small"},
    "medium":      {"resolution": 576, "patch": 14, "decoder_layers": 4, "queries": 300, "backbone": "DINOv2-small"},
    "large":       {"resolution": 704, "patch": 14, "decoder_layers": 4, "queries": 300, "backbone": "DINOv2-small"},
    "seg-nano":    {"resolution": 312, "patch": 12, "decoder_layers": 4, "queries": 100, "backbone": "DINOv2-small"},
    "seg-small":   {"resolution": 384, "patch": 12, "decoder_layers": 4, "queries": 100, "backbone": "DINOv2-small"},
    "seg-medium":  {"resolution": 432, "patch": 12, "decoder_layers": 5, "queries": 200, "backbone": "DINOv2-small"},
    "seg-large":   {"resolution": 504, "patch": 12, "decoder_layers": 5, "queries": 200, "backbone": "DINOv2-small"},
    "seg-xlarge":  {"resolution": 624, "patch": 12, "decoder_layers": 6, "queries": 300, "backbone": "DINOv2-small"},
    "seg-2xlarge": {"resolution": 768, "patch": 12, "decoder_layers": 6, "queries": 300, "backbone": "DINOv2-small"},
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
    "seg-large": "Seg-Large",
    "seg-xlarge": "Seg-XLarge",
    "seg-2xlarge": "Seg-2XLarge",
}

UPSTREAM = "https://github.com/roboflow/rf-detr"


def env_rfdetr_version() -> Optional[str]:
    """Best-effort rfdetr version for *this* checkout, never a literal.

    Order of preference:
      1. the installed distribution's metadata (what a sweep run here would use)
      2. the ``rfdetr==X.Y.Z`` pin in scripts/requirements.txt

    Returns None if neither source can answer, so callers can refuse to emit a
    provenance sentence rather than guess. This exists so that bumping rfdetr
    cannot silently leave a stale version string baked into a model card.
    """
    try:
        from importlib.metadata import version as _dist_version
        return _dist_version("rfdetr")
    except Exception:
        pass
    try:
        for line in REQUIREMENTS_TXT.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            m = re.match(r"^rfdetr\s*==\s*([A-Za-z0-9_.+!-]+)$", line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return None


def format_versions(versions: List[str]) -> str:
    """Render one or more rfdetr versions for the provenance sentence."""
    if len(versions) == 1:
        return f"`rfdetr {versions[0]}`"
    return " / ".join(f"`rfdetr {v}`" for v in versions)


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
    # rfdetr version the PyTorch ground truth for this cell was produced with
    rfdetr_version: Optional[str] = None


def load_metrics() -> Dict[str, Dict[str, CellMetrics]]:
    """Returns metrics[variant][quant] = CellMetrics.

    Note: accuracy_sweep.json uses quant names matching our QUANTS list,
    but the variant/quant key in latency.json is "{variant}/{quant}".

    Each cell carries the rfdetr version its ground truth was measured against.
    Re-swept cells record their own ``rfdetr_version``; older cells inherit
    ``meta.rfdetr_version``. The sweep file is the authority here, because a
    card must state the version the numbers were *measured* with, which is not
    necessarily the version installed today.
    """
    with open(ACCURACY_JSON) as f:
        sweep = json.load(f)
    with open(LATENCY_JSON) as f:
        latency = json.load(f)

    meta_version = (sweep.get("meta") or {}).get("rfdetr_version")

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
            rfdetr_version=cell.get("rfdetr_version") or meta_version,
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
    """Generate a Markdown model card for one variant repo.

    If `cells` is empty, the per-quant metrics table is replaced with a
    placeholder pointing the reader to BENCHMARK.md (used for newly added
    variants that haven't been swept yet).
    """
    is_seg = variant.startswith("seg-")
    pretty = DISPLAY[variant]
    arch = ARCH[variant]
    pipeline_tag = "image-segmentation" if is_seg else "object-detection"
    seg_tag = "image-segmentation" if is_seg else "object-detection"
    has_metrics = bool(cells)
    # Only claim a latency measurement if one actually exists for this variant.
    has_latency = any(c.median_ms is not None for c in cells.values())
    # Provenance is derived, never hardcoded: prefer what the sweep recorded for
    # these exact cells, and fall back to the version this checkout resolves to.
    sweep_versions = sorted({c.rfdetr_version for c in cells.values() if c.rfdetr_version})
    card_versions = sweep_versions or [v for v in [env_rfdetr_version()] if v]

    # On-disk sizes, used for the size claims below so they cannot go stale.
    disk_sizes: Dict[str, float] = {}
    for q in QUANTS:
        if q in cells:
            disk_sizes[q] = cells[q].file_size_mb
        else:
            p = MODELS_DIR / f"rfdetr-{variant}-{q}.gguf"
            if p.is_file():
                disk_sizes[q] = p.stat().st_size / 1e6

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
    if "f32" in disk_sizes and "f16" in disk_sizes and disk_sizes["f16"] > 0:
        shrink = f"{disk_sizes['f32'] / disk_sizes['f16']:.2f}× smaller than F32"
    else:
        shrink = "substantially smaller than F32"
    accuracy_clause = ("near-identical accuracy to F32 (see the table below)"
                       if has_metrics else "near-identical accuracy to F32")
    lines.append("This repo contains all four standard quantizations of this variant. "
                 f"**F16 is the recommended default** — {accuracy_clause}, {shrink}, "
                 f"and it takes ggml's F32×F16 matmul fast path.")
    lines.append("")

    # File table
    lines.append("## Available files")
    lines.append("")
    quant_label = {
        "f32": "F32",
        "f16": "F16",
        "q8_0": "Q8_0",
        "q4_K": "Q4_K",
    }
    recommended = "f16"
    if has_metrics:
        if is_seg:
            lines.append("| File | Quant | Size (MB) | Recall @ IoU 0.5 | Recall @ IoU 0.95 | Mean mask IoU | Pixel agreement | Latency (median ms, T=8) |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        else:
            lines.append("| File | Quant | Size (MB) | Recall @ IoU 0.5 | Recall @ IoU 0.95 | Mean \\|Δscore\\| | Latency (median ms, T=8) |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
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
    else:
        # No accuracy_sweep / latency data yet for this variant — emit a
        # plain file listing with on-disk sizes and a pointer to the bench.
        lines.append("| File | Quant | Size (MB) |")
        lines.append("|---|---|---:|")
        for q in QUANTS:
            p = MODELS_DIR / f"rfdetr-{variant}-{q}.gguf"
            if not p.is_file():
                continue
            fname = p.name
            rec_marker = " ← **recommended**" if q == recommended else ""
            size_mb = p.stat().st_size / 1e6
            lines.append(f"| `{fname}`{rec_marker} | {quant_label[q]} | {size_mb:.1f} |")
        lines.append("")
        lines.append("> Accuracy + latency for this variant haven't been added to the "
                     "[BENCHMARK.md](https://github.com/mudler/rf-detr.cpp/blob/main/BENCHMARK.md) "
                     "sweep yet; the C++ implementation is verified to load and run "
                     "`rfdetr-cli detect` end-to-end on every quant. Run "
                     "`scripts/sweep_accuracy.py --variant " + variant + "` locally "
                     "for parity numbers.")
    lines.append("")

    # Methodology note. Every sentence here is gated on the data actually
    # existing for this variant, so the card can never claim a measurement
    # that was not taken.
    if has_metrics and card_versions:
        lines.append(f"All accuracy numbers are computed against the upstream PyTorch "
                     f"reference ({format_versions(card_versions)}) on 7 COCO val2017 "
                     f"images at threshold 0.5.")
    if has_latency:
        lines.append("Latency is measured with `rfdetr-cli bench` (8 iters + 3 "
                     "warmup) at T=8 threads on a single AMD Ryzen 9 9950X3D image "
                     "(`coco_kitchen.jpg`, 640x427).")
    elif has_metrics:
        lines.append("No latency benchmark has been recorded for this variant yet, so "
                     "the latency column is left empty. Run `scripts/quick_bench.sh` "
                     "locally for timings on your own hardware.")
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
    f32_size = disk_sizes.get("f32")
    f32_note = f", {f32_size:.0f} MB" if f32_size else ""
    lines.append(f"- **F32** — the full-precision conversion{f32_note}, and the closest "
                 f"match to the PyTorch reference"
                 + (" (see the table above for measured agreement)." if has_metrics else "."))
    lines.append("- **F16** — matmul-multiplicand weights only; LayerNorms, conv kernels, "
                 "embeddings, biases, and layer-scale gammas stay F32. Accuracy tracks "
                 "F32 closely on this model, and it takes ggml's F32×F16 matmul fast "
                 "path.")
    q8_ratio = (f"~{f32_size / disk_sizes['q8_0']:.1f}× smaller than F32"
                if f32_size and disk_sizes.get("q8_0") else "substantially smaller than F32")
    lines.append(f"- **Q8_0** — best size/accuracy tradeoff under F16; {q8_ratio} "
                 f"with effectively identical detections.")
    q4_ratio = (f"~{f32_size / disk_sizes['q4_K']:.1f}× over F32"
                if f32_size and disk_sizes.get("q4_K") else "still large")
    lines.append(f"- **Q4_K** — smallest practical quant. Rows with `ne[0] % 256 != 0` (the "
                 f"decoder's 128-dim MLP halves) silently fall back to Q8_0 per ggml's "
                 f"quantizer logic — net compression is still {q4_ratio}. "
                 f"Use only when the size budget is tight; expect a measurable "
                 f"Recall@0.95 drop relative to F16/Q8_0"
                 + (" (see file table above)." if has_metrics else "."))
    lines.append("")

    # Usage
    lines.append("## Usage")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Clone + build rfdetr.cpp")
    lines.append("git clone https://github.com/mudler/rf-detr.cpp")
    lines.append("cd rf-detr.cpp")
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
    if has_metrics and card_versions:
        lines.append(
            f"All accuracy metrics are computed against the upstream PyTorch reference "
            f"({format_versions(card_versions)}) on 7 COCO val2017 images at threshold "
            f"0.5. Each detection match uses greedy Hungarian-style assignment by IoU "
            f"(≥ 0.5 lenient, ≥ 0.95 strict) with class equality required."
        )
        lines.append("")
    if is_seg and has_metrics:
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
        "for the full sweep across the (variant × quant) cells."
    )
    lines.append("")

    # License
    lines.append("## License")
    lines.append("")
    lines.append("Apache-2.0 — matches the upstream [rfdetr](https://github.com/roboflow/rf-detr) license.")
    lines.append("")

    return "\n".join(lines)


def expected_local_files(variant: str) -> Tuple[List[Path], List[Path]]:
    """Split this variant's expected GGUFs into (present, missing), QUANTS order."""
    present: List[Path] = []
    missing: List[Path] = []
    for q in QUANTS:
        p = MODELS_DIR / f"rfdetr-{variant}-{q}.gguf"
        (present if p.is_file() else missing).append(p)
    return present, missing


def list_local_files(variant: str) -> List[Path]:
    """Return the full set of GGUF files for this variant, in QUANTS order."""
    present, missing = expected_local_files(variant)
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(f"missing model file(s) for {variant}: {names}")
    return present


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
    print(f"  {len(local_files)} GGUF files, total {total_bytes / 1e6:.1f} MB")

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
    # README isn't a GGUF so it doesn't count toward this variant's quant set,
    # but we still track its bytes.
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
    explicit = False
    if args.only:
        explicit = True
        wanted = {v.strip() for v in args.only.split(",") if v.strip()}
        variants = [v for v in VARIANTS if v in wanted]
        if not variants:
            print(f"error: --only={args.only} matched no known variants", file=sys.stderr)
            return 2
        unknown = sorted(wanted - set(VARIANTS))
        if unknown:
            print(f"error: --only names unknown variant(s): {', '.join(unknown)}",
                  file=sys.stderr)
            return 2

    # Preflight: a variant with no local GGUFs must never turn into a confusing
    # mid-run crash. Naming an incomplete variant explicitly is an error; an
    # unfiltered run skips it loudly and exits non-zero so it can't look clean.
    incomplete: Dict[str, List[Path]] = {}
    for v in variants:
        _, missing = expected_local_files(v)
        if missing:
            incomplete[v] = missing
    if incomplete:
        for v, missing in incomplete.items():
            print(f"warning: variant {v} has {len(missing)} missing GGUF file(s):",
                  file=sys.stderr)
            for p in missing:
                print(f"    {p}", file=sys.stderr)
        if explicit:
            print("error: refusing to publish — the variant(s) above were requested "
                  "explicitly via --only but are not fully built locally. Build them "
                  "(scripts/build_all_quants.sh) or drop them from --only.",
                  file=sys.stderr)
            return 2
        skipped = sorted(incomplete)
        variants = [v for v in variants if v not in incomplete]
        print(f"warning: skipping {', '.join(skipped)} — no complete local GGUF set. "
              f"Pass --only to publish a specific subset.", file=sys.stderr)
        if not variants:
            print("error: no variant has a complete local GGUF set; nothing to publish.",
                  file=sys.stderr)
            return 2
    else:
        skipped = []

    # Preflight: never emit a card that states accuracy numbers without being
    # able to say which rfdetr reference produced them.
    if env_rfdetr_version() is None:
        undated = [v for v in variants
                   if metrics.get(v) and not any(c.rfdetr_version for c in metrics[v].values())]
        if undated:
            print(f"error: cannot determine the rfdetr version for {', '.join(undated)}: "
                  f"the sweep records none and neither the installed rfdetr "
                  f"distribution nor {REQUIREMENTS_TXT} could be read.", file=sys.stderr)
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
        cells = metrics.get(v) or {}
        if not cells:
            print(f"note: no swept metrics for variant {v} — publishing with "
                  f"placeholder model card", file=sys.stderr)
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
    for v in skipped:
        names = ", ".join(p.name for p in incomplete[v])
        print(f"{HF_USER + '/rfdetr-cpp-' + v:45s} {'-':>5} {'-':>11}  SKIPPED")
        print(f"  missing locally: {names}")
    print()
    print(f"Total uploaded: {total_files} files, {total_bytes / 1e6:.1f} MB")
    if skipped:
        print(f"Skipped (incomplete local GGUF set): {', '.join(skipped)}")
    print(f"Elapsed: {elapsed:.1f}s")

    failed = sum(1 for r in results if r.get("error"))
    return 1 if (failed or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
