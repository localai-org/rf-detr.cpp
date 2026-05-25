# rt-detr.cpp Parity Workflow

## Goal

Verify that the C++ forward pass produces the same intermediate tensors as a
reference implementation, layer by layer, with declared per-checkpoint
tolerances. Catch divergences at the layer where they first appear.

## Reference implementations

Plan 3 ships a **numpy reference** (`scripts/gen_numpy_baseline.py`). It uses
the same Plan 2 GGUF format and produces a baseline bundle GGUF containing
expected intermediate tensors at named checkpoints. No torch / no rfdetr; CI
runs it directly.

Plan 6 will add a **torch + rfdetr reference** that consumes the same input
and produces a baseline bundle in the same format. The C++ parity harness
is reference-agnostic — it only consumes baseline bundles.

## Baseline bundle format

A baseline bundle is a GGUF file with:

- All tensors named `parity.<checkpoint_name>` — e.g.
  `parity.preprocess.input`, `parity.backbone.patch_embed.output`,
  `parity.backbone.block.0.norm1.output`, `parity.backbone.block.0.output`.
- Metadata `parity.format.version = "1"`.
- Metadata `parity.reference = "numpy" | "torch"`.
- Metadata `parity.input_shape = int32[4]` describing the input the
  reference consumed (NCHW: 1 × 3 × H × W).

## Named checkpoints captured by Plan 3

- `preprocess.input` — `(1, H, W, 3)` float32, post normalization (mean/std)
- `backbone.patch_embed.output` — `(1, N_patches, dim)` float32
- `backbone.cls_pos_embed.output` — `(1, N_patches + 1, dim)` float32
- `backbone.block.0.norm1.output` — `(1, N_patches, dim)` float32
- `backbone.block.0.attn.output` — `(1, N_patches, dim)` float32
- `backbone.block.0.mlp.output` — `(1, N_patches, dim)` float32
- `backbone.block.0.output` — `(1, N_patches, dim)` float32 (full block output)

Plan 4 adds CLS+pos_embed, blocks 1..11, the final backbone LayerNorm, and
4 multi-scale taps (`backbone.multiscale.level{0..3}`) at the layer
indices configured by `rfdetr.backbone.multi_scale_layers` ([2, 5, 8, 11]
for the `base` variant). Plans 5-6 add projector levels, encoder/decoder
layers, and heads.

## C++ trace callback

`src/trace.{cpp,hpp}` exposes:

```cpp
namespace rfdetr {
using trace_cb = std::function<void(const std::string& name, ggml_tensor* t)>;
void set_trace_callback(trace_cb cb);
void publish(const std::string& name, ggml_tensor* t);  // no-op if no cb
}
```

The forward-pass code calls `rfdetr::publish("backbone.patch_embed.output", t)`
at each defined checkpoint. Production inference doesn't register a callback;
the publish call is a hash-map lookup + early return.

## Per-checkpoint tolerances

Configured in `tests/test_parity_backbone.cpp` via a small table. Defaults:

| Checkpoint                            | atol  | rtol  |
|---------------------------------------|-------|-------|
| `preprocess.input`                    | 1e-6  | 0     |
| `backbone.patch_embed.output`         | 1e-5  | 1e-4  |
| `backbone.cls_pos_embed.output`       | 1e-5  | 1e-4  |
| `backbone.block.0.norm1.output`       | 1e-5  | 1e-4  |
| `backbone.block.0.attn.output`        | 1e-5  | 1e-4  |
| `backbone.block.0.mlp.output`         | 1e-5  | 1e-4  |
| `backbone.block.0.output`             | 1e-5  | 1e-4  |
| `backbone.norm.output`                | 1e-5  | 1e-4  |
| `backbone.multiscale.level0`          | 1e-5  | 1e-4  |
| `backbone.multiscale.level1`          | 1e-5  | 1e-4  |
| `backbone.multiscale.level2`          | 1e-5  | 1e-4  |
| `backbone.multiscale.level3`          | 1e-5  | 1e-4  |
| `projector.level0.output`             | 1e-5  | 1e-4  |
| `projector.level1.output`             | 1e-5  | 1e-4  |
| `projector.level2.output`             | 1e-5  | 1e-4  |
| `projector.level3.output`             | 1e-5  | 1e-4  |
| `projector.concat.output`             | 1e-5  | 1e-4  |
| `encoder.layer0.norm1.output`         | 1e-5  | 1e-4  |
| `encoder.layer0.attn.output`          | 1e-5  | 1e-4  |
| `encoder.layer0.mlp.output`           | 1e-5  | 1e-4  |
| `encoder.layer0.output`               | 1e-5  | 1e-4  |
| `encoder.layer1.norm1.output`         | 1e-5  | 1e-4  |
| `encoder.layer1.attn.output`          | 1e-5  | 1e-4  |
| `encoder.layer1.mlp.output`           | 1e-5  | 1e-4  |
| `encoder.layer1.output`               | 1e-5  | 1e-4  |
| `encoder.layer2.norm1.output`         | 1e-5  | 1e-4  |
| `encoder.layer2.attn.output`          | 1e-5  | 1e-4  |
| `encoder.layer2.mlp.output`           | 1e-5  | 1e-4  |
| `encoder.layer2.output`               | 1e-5  | 1e-4  |
| `encoder.output`                      | 1e-5  | 1e-4  |
| `decoder.queries`                     | 1e-5  | 1e-4  |
| `decoder.layer0.self_attn.output`     | 1e-5  | 1e-4  |
| `decoder.layer0.cross_attn.output`    | 1e-5  | 1e-4  |
| `decoder.layer0.mlp.output`           | 1e-5  | 1e-4  |
| `decoder.layer0.output`               | 1e-5  | 1e-4  |

Plan 4 switched the fixture to F32 weights (the generator now defaults to
`--dtype f32`), eliminating the F16 quantization noise floor that
previously forced a 1e-3 ceiling on `patch_embed.output` and the residual
that carries it. All backbone checkpoints now ride at `1e-5` atol /
`1e-4` rtol — tight enough to catch real correctness bugs, loose enough
to absorb ggml's F32 vs numpy's float64 order-of-operations drift. Plan
7 will re-introduce F16/quantized weight handling with explicit noise
discipline at that time.

## Regeneration

```bash
python3 scripts/gen_numpy_baseline.py \
    --model tests/fixtures/model_base.gguf \
    --output tests/fixtures/baseline_backbone.gguf
```

CMake runs this as a custom_command at build time (declared in
`tests/CMakeLists.txt`). Bundle is regenerated whenever the script changes
or the source GGUF fixture changes.

## Diagnosing a parity failure

`test_parity_backbone` prints, for each failing checkpoint:

- Checkpoint name
- Tensor shape
- Max absolute error and its location (flat index)
- Mean absolute error
- Sample values: `cpp[i] = X, ref[i] = Y` at the worst location

A failing checkpoint earlier in the graph causes all later checkpoints to
fail. Always fix from the earliest divergence forward.

## Window vs global attention

Backbone blocks dispatch between two attention paths based on `is_global_block(cfg, i)`:
- **Global** (`i` ∈ `multi_scale_layers` = `[2, 5, 8, 11]` for base): standard MHA over all N+1 tokens (CLS + patches)
- **Windowed** (otherwise): CLS bypasses; patches are W×W-window-partitioned, attended per window, unpartitioned, then re-concatenated with CLS

Both paths share the same `backbone.block.{i}.attn.output` parity checkpoint;
the test verifies windowed blocks' values match the numpy reference at the
same 1e-5 tolerance as global blocks. The two paths produce numerically
distinct values (windowed blocks see only `window_size`² tokens per
attention; global blocks see all N+1), but on the fixture both ride at
~1e-8 max_abs against the numpy reference.

Convention: CLS bypasses windowed blocks (most common ViT-window
implementation). Real-rfdetr convention will be verified in Plan 7 (torch
baseline).
