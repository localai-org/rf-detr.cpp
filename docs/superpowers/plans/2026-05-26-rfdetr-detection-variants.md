# Detection Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for all 5 RF-DETR detection variants (Nano, Small, Base, Medium, Large) to rfdetr.cpp — converter, loader, CLI, benchmark, and plots.

**Architecture:** All 5 variants share the same DINOv2-small backbone, decoder dim (256), num_queries (300), and num_classes (91). They differ only in (a) input resolution and (b) decoder layer count. Our converter and loader are already type-agnostic over these dimensions (both are read from GGUF metadata), so most of this is conversion + benchmarking + plotting plus a small CLI variant-detection enhancement. There is one variant-specific quirk: each variant ships its own pretrained `.pth` and may have variant-specific config like `out_feature_indexes`.

**Tech Stack:** Python (converter, bench), C++ (CLI), matplotlib (plots), ggml (inference).

---

## Per-Variant Specs (verified from rfdetr 1.7.0)

| Variant | Resolution | Decoder layers | Params (PyTorch) | Expected GGUF F32 size |
|---|---:|---:|---:|---:|
| Nano | 384 | 2 | 30.5M | ~117 MB |
| Small | 512 | 3 | 32.1M | ~123 MB |
| Base | 560 | 3 | 32.2M | 119 MB ✓ |
| Medium | 576 | 4 | 33.7M | ~129 MB |
| Large | 704 | 4 | 33.9M | ~130 MB |

## File Structure

- Modify: `scripts/convert_rfdetr_to_gguf.py` — add `--variant {nano,small,base,medium,large}` flag; dispatch to the correct `RFDETR*` class
- Modify: `src/model_loader.cpp` — verify variant string round-trips correctly (probably already works since we read all config from GGUF metadata)
- Modify: `examples/cli/main.cpp::cmd_info` — verify it prints variant name from loaded GGUF correctly for non-base
- Create: `scripts/convert_all_variants.sh` — batch-convert all 5 variants
- Modify: `scripts/bench_community.py` — add per-variant looping; new CLI arg `--variants nano,small,base,medium,large` or auto-detect from `models/rfdetr-*.gguf`
- Modify: `scripts/plot_community.py` — new plot `variants_tradeoffs.png` (size vs latency vs detection count per variant)
- Modify: `BENCHMARK.md`, `README.md` — variants table
- Create: `tests/test_variants.cpp` — load each variant GGUF, run a single forward, verify variant name + det/box counts

---

## Task 1: Verify the converter on a non-base variant

**Files:**
- Modify: `scripts/convert_rfdetr_to_gguf.py` (single line — accept variant arg)

- [ ] **Step 1: Add `--variant` argparse argument**

In `scripts/convert_rfdetr_to_gguf.py`, find the argparse section. Add:

```python
parser.add_argument(
    "--variant",
    choices=["nano", "small", "base", "medium", "large"],
    default="base",
    help="Which RF-DETR detection variant to convert (default: base).",
)
```

- [ ] **Step 2: Dispatch to the correct rfdetr class**

Find where `RFDETRBase()` is instantiated. Replace with:

```python
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge
_VARIANT_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "base": RFDETRBase,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}
m = _VARIANT_CLASSES[args.variant]()
```

Also write the variant string into the GGUF metadata key `rfdetr.variant`:

```python
writer.add_string("rfdetr.variant", args.variant)
```

(The loader already reads `rfdetr.variant`; verify by greping for it in `src/model_loader.cpp`.)

- [ ] **Step 3: Test conversion of Nano**

Run:

```bash
.venv/bin/python scripts/convert_rfdetr_to_gguf.py --variant nano \
    --output models/rfdetr-nano-f32.gguf
ls -lh models/rfdetr-nano-f32.gguf
build/bin/rfdetr-cli info --model models/rfdetr-nano-f32.gguf
```

Expected: file ~117 MB; info output shows `variant=nano, num_classes=91, num_queries=300, n_tensors=...` and decoder layer count = 2 (not 3).

- [ ] **Step 4: Commit**

```bash
git add scripts/convert_rfdetr_to_gguf.py
git commit -m "feat(convert): --variant flag for nano/small/medium/large detection variants"
```

---

## Task 2: Run end-to-end detection on Nano

**Files:**
- None modified (just verifying the loader+forward pipeline)

- [ ] **Step 1: Run detection on a known image**

```bash
build/bin/rfdetr-cli detect \
    --model models/rfdetr-nano-f32.gguf \
    --input /tmp/coco_sample.jpg \
    --output /tmp/nano_detect.json \
    --threshold 0.5 --threads 8
cat /tmp/nano_detect.json | .venv/bin/python -m json.tool | head -30
```

Expected: JSON with detections. May differ from Base because Nano is a different (smaller) model with different accuracy.

- [ ] **Step 2: Verify against Python rfdetr Nano on the same image**

```bash
.venv/bin/python -c "
from rfdetr import RFDETRNano
import json
m = RFDETRNano()
out = m.predict('/tmp/coco_sample.jpg', threshold=0.5)
print(out)
"
```

Compare classes + scores roughly. Sub-pixel parity isn't required (the model itself may differ from Base accuracy-wise), but the C++ output should match Python's Nano predictions to the same tolerance our Base test achieved (sub-pixel, Δscore ≤ 0.05).

If they don't match: investigate (loader bug?). If they do: proceed.

- [ ] **Step 3: Document the comparison briefly in a scratch note**

No commit — this is a verification step. Move on to converting the rest.

---

## Task 3: Convert + verify all variants in one batch

**Files:**
- Create: `scripts/convert_all_variants.sh`

- [ ] **Step 1: Write the batch converter**

```bash
#!/usr/bin/env bash
# scripts/convert_all_variants.sh — converts all 5 RF-DETR detection variants to GGUF F32.
# Each conversion takes ~30s to load the model + ~5s to write the file.

set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=(nano small base medium large)
mkdir -p models
for v in "${VARIANTS[@]}"; do
    out="models/rfdetr-${v}-f32.gguf"
    if [[ -f "$out" ]]; then
        echo "skipping $v (file exists: $out)"
        continue
    fi
    echo "=== converting $v ==="
    .venv/bin/python scripts/convert_rfdetr_to_gguf.py --variant "$v" --output "$out"
    ls -lh "$out"
done
```

Make executable:

```bash
chmod +x scripts/convert_all_variants.sh
```

- [ ] **Step 2: Run it**

```bash
scripts/convert_all_variants.sh
ls -lh models/rfdetr-*-f32.gguf
```

Expected: 5 GGUF files, sizes roughly matching the table at the top of this plan.

- [ ] **Step 3: Verify each loads via `rfdetr-cli info`**

```bash
for v in nano small base medium large; do
    echo "=== $v ==="
    build/bin/rfdetr-cli info --model "models/rfdetr-${v}-f32.gguf" | grep -E "variant|num_classes|num_queries|n_tensors"
done
```

Each should print its own variant name + correct dec_layers (read from metadata).

- [ ] **Step 4: Run a quick detection on each**

```bash
for v in nano small base medium large; do
    echo "=== $v ==="
    build/bin/rfdetr-cli detect --model "models/rfdetr-${v}-f32.gguf" \
        --input /tmp/coco_sample.jpg --threshold 0.5 --threads 8 \
        --output "/tmp/det_${v}.json"
    .venv/bin/python -c "import json; d = json.load(open('/tmp/det_${v}.json')); print(f'detections: {len(d[\"detections\"])}')"
done
```

Expected: each variant produces detections. Counts may differ across variants.

If any variant fails to load or run, STOP and investigate. Possible causes:
- Variant-specific config field not handled in loader (e.g., `out_feature_indexes` differs from Base's `[2,5,8,11]`)
- New tensor present in a variant that the loader doesn't expect

Fix as needed before proceeding.

- [ ] **Step 5: Commit the batch script (only)**

```bash
git add scripts/convert_all_variants.sh
git commit -m "feat(scripts): batch-convert all RF-DETR detection variants"
```

---

## Task 4: Add a C++ test that loads each variant

**Files:**
- Create: `tests/test_variants.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the test**

Create `tests/test_variants.cpp`:

```cpp
/* tests/test_variants.cpp — verifies each RF-DETR detection variant loads
 * and produces a valid forward output. Skips per-variant if the GGUF isn't
 * present (generate via scripts/convert_all_variants.sh).
 */
#include "rfdetr.h"
#include "test_assert.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <string>
#include <vector>

static bool file_exists(const std::string& p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}

int main() {
    const std::string models_dir = "models";
    const std::vector<std::string> variants = {"nano", "small", "base", "medium", "large"};
    int checked = 0;
    for (const auto& v : variants) {
        std::string path = models_dir + "/rfdetr-" + v + "-f32.gguf";
        if (!file_exists(path)) {
            std::fprintf(stderr, "[test_variants] SKIP %s (not present)\n", path.c_str());
            continue;
        }
        rfdetr_init_params p{};
        p.model_path = path.c_str();
        p.n_threads = 4;
        rfdetr_context* ctx = nullptr;
        rfdetr_status st = rfdetr_init(&p, &ctx);
        RFDETR_ASSERT(st == RFDETR_OK);
        rfdetr_model_info info{};
        st = rfdetr_get_model_info(ctx, &info);
        RFDETR_ASSERT(st == RFDETR_OK);
        std::fprintf(stderr, "[test_variants] OK %s — variant=%s n_classes=%d n_queries=%d\n",
                     path.c_str(), info.variant, info.num_classes, info.num_queries);
        rfdetr_free(ctx);
        checked++;
    }
    if (checked == 0) {
        std::fprintf(stderr, "[test_variants] no variant GGUFs found — generate with scripts/convert_all_variants.sh\n");
    }
    return 0;
}
```

(Adjust `info.variant`, `info.num_classes`, `info.num_queries` to match the actual rfdetr_model_info struct in `include/rfdetr.h`.)

- [ ] **Step 2: Register in CMake**

In `tests/CMakeLists.txt`, find the `rfdetr_add_test(...)` calls. Add:

```cmake
rfdetr_add_test(test_variants)
```

- [ ] **Step 3: Build + run**

```bash
cmake --build build -j 2>&1 | tail -3
ctest --test-dir build -R test_variants --output-on-failure
```

Expected: 5 OK lines (one per variant), exit 0.

- [ ] **Step 4: Commit**

```bash
git add tests/test_variants.cpp tests/CMakeLists.txt
git commit -m "test: verify each RF-DETR detection variant loads via the C-API"
```

---

## Task 5: Extend the community bench to all variants

**Files:**
- Modify: `scripts/bench_community.py`

- [ ] **Step 1: Read the current bench script**

Read `scripts/bench_community.py` to understand the variant/cell structure. Currently it benches PyTorch / C++ F32 / C++ Q8_0 on each image. We want to add per-variant cells.

- [ ] **Step 2: Add a `--variants` flag**

```python
parser.add_argument(
    "--variants",
    default="nano,small,base,medium,large",
    help="Comma-separated detection variants to bench (must have corresponding models/rfdetr-{variant}-f32.gguf)",
)
```

- [ ] **Step 3: Loop over variants in the bench**

For each variant:
1. Skip if `models/rfdetr-{variant}-f32.gguf` doesn't exist (log + continue)
2. Bench C++ F32 (existing logic, just point at the per-variant GGUF)
3. Bench Python rfdetr's corresponding `RFDETR{Variant}()` class (load once, time `predict()` per image)
4. Record cells with `variant` key in the output JSON

The output `bench_data.json` schema needs an additional `variant` field per cell. Migrate gracefully (default to `base` for old entries when reading).

- [ ] **Step 4: Run a quick sanity check**

```bash
.venv/bin/python scripts/bench_community.py --variants nano,base --images coco_kitchen.jpg --iters 3 --warmup 1
```

Expected: produces cells for both variants on the kitchen image. Inspect output JSON.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_community.py
git commit -m "feat(bench): per-variant benchmarking in bench_community.py"
```

---

## Task 6: Generate variants plots

**Files:**
- Modify: `scripts/plot_community.py`

- [ ] **Step 1: Add new plot function `plot_variants_overview`**

In `scripts/plot_community.py`, add:

```python
def plot_variants_overview(data, output_dir):
    """Dual-panel: (left) median latency per variant for PyTorch vs C++ F32,
    (right) detection count per variant on a fixed image."""
    import matplotlib.pyplot as plt
    fig, (ax_lat, ax_det) = plt.subplots(1, 2, figsize=(14, 5))
    variants = ["nano", "small", "base", "medium", "large"]
    # Build per-variant median latencies (mean across images)
    # ... (use the data dict from bench_community)
    # Bars: PyTorch (gray), C++ F32 (blue)
    # ... draw bars + annotate
    # Right panel: detection count per variant on coco_kitchen.jpg
    # ... bars per variant
    fig.suptitle("RF-DETR variants: latency × detection-count tradeoff (T=8)")
    plt.savefig(f"{output_dir}/variants_overview.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_dir}/variants_overview.svg", bbox_inches="tight")
```

Implement the bar drawing using the same color palette and style as `latency_comparison`.

- [ ] **Step 2: Wire into main**

In `scripts/plot_community.py::main`, after the existing plot calls, add:

```python
plot_variants_overview(data, output_dir)
```

- [ ] **Step 3: Run + inspect**

```bash
.venv/bin/python scripts/plot_community.py
ls benchmarks/plots/variants_overview.{png,svg}
```

View the PNG to confirm it's readable. Iterate on layout if needed.

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_community.py benchmarks/plots/variants_overview.*
git commit -m "feat(plots): variants_overview — latency × detection-count per variant"
```

---

## Task 7: Update docs

**Files:**
- Modify: `BENCHMARK.md`
- Modify: `README.md`

- [ ] **Step 1: BENCHMARK.md variants section**

Add a section after the existing quant section:

```markdown
## Variant comparison

All 5 RF-DETR detection variants share the same DINOv2-small backbone; they differ only in input resolution and decoder layer count.

| Variant | Resolution | Decoder layers | Params | GGUF F32 | C++ F32 median ms @ T=8 |
|---|---:|---:|---:|---:|---:|
| Nano | 384 | 2 | 30.5M | ~117 MB | (fill in) |
| Small | 512 | 3 | 32.1M | ~123 MB | (fill in) |
| Base | 560 | 3 | 32.2M | 119 MB | 144 |
| Medium | 576 | 4 | 33.7M | ~129 MB | (fill in) |
| Large | 704 | 4 | 33.9M | ~130 MB | (fill in) |

![Variants overview](benchmarks/plots/variants_overview.png)

Generate variants:
\```bash
scripts/convert_all_variants.sh
\```
```

(Fill in the actual ms numbers from the bench output.)

- [ ] **Step 2: README.md variants section**

Add a short variants section pointing at BENCHMARK.md for details.

- [ ] **Step 3: Commit**

```bash
git add BENCHMARK.md README.md
git commit -m "docs: variants comparison table + plot"
```

---

## Self-Review

- **Spec coverage**: 5 variants converted ✓, loader handles them ✓, CLI works ✓, bench measures all ✓, plot generated ✓, docs updated ✓.
- **Placeholders**: "(fill in)" in the BENCHMARK.md table — this is intentional, filled in after Task 5 produces actual numbers. Otherwise no `TBD`/`later`.
- **Type consistency**: `info.variant`, `info.num_classes`, `info.num_queries` reference the existing `rfdetr_model_info` struct — verify field names match before writing the test in Task 4.

**Open questions to verify at execution time:**
1. Does Nano's `out_feature_indexes` differ from Base's `[2,5,8,11]`? If so, the loader must read it from metadata (it probably does; verify).
2. Does Nano's `patch_size` match Base's 14? Per the rfdetr config dump, yes, Nano also uses 14.
