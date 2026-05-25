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
- `backbone.block.0.norm1.output` — `(1, N_patches, dim)` float32
- `backbone.block.0.attn.output` — `(1, N_patches, dim)` float32
- `backbone.block.0.mlp.output` — `(1, N_patches, dim)` float32
- `backbone.block.0.output` — `(1, N_patches, dim)` float32 (full block output)

Plans 4-6 add more checkpoints (block 1..11, projector levels, encoder/decoder
layers, heads).

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

Configured in `tests/test_parity_block0.cpp` via a small table. Defaults:

| Checkpoint                            | atol  | rtol  |
|---------------------------------------|-------|-------|
| `preprocess.input`                    | 1e-6  | 0     |
| `backbone.patch_embed.output`         | 1e-3  | 1e-2  |
| `backbone.block.0.norm1.output`       | 1e-4  | 1e-3  |
| `backbone.block.0.attn.output`        | 1e-5  | 1e-4  |
| `backbone.block.0.mlp.output`         | 1e-5  | 1e-4  |
| `backbone.block.0.output`             | 1e-3  | 1e-2  |

Tolerances are tightened to match actual measured deltas. The two
F16-noise-bound checkpoints (`patch_embed.output` and the full
`block.0.output`, which carries patch_embed noise through the residual)
keep `1e-3 atol`: the test fixture stores weights as F16; numpy reads them
as F16→F32 then accumulates in float64 while ggml accumulates in F32 —
patch_embed alone does 588 mul-adds of ~0.02-scale F16 values, producing
observable drift in the 4th decimal. Norm1 / attn / mlp run on the
already-noisy patch_embed output but don't add further F16 quantization,
so their deltas stay in the 1e-5..1e-4 range and the tolerances are
correspondingly tight. Tolerances across the board will drop once the
production model is stored in F32 (planned in Plan 6) or when we add
per-layer F32 reference baselines.

## Regeneration

```bash
python3 scripts/gen_numpy_baseline.py \
    --model tests/fixtures/model_base.gguf \
    --output tests/fixtures/baseline_block0.gguf
```

CMake runs this as a custom_command at build time (declared in
`tests/CMakeLists.txt`). Bundle is regenerated whenever the script changes
or the source GGUF fixture changes.

## Diagnosing a parity failure

`test_parity_block0` prints, for each failing checkpoint:

- Checkpoint name
- Tensor shape
- Max absolute error and its location (flat index)
- Mean absolute error
- Sample values: `cpp[i] = X, ref[i] = Y` at the worst location

A failing checkpoint earlier in the graph causes all later checkpoints to
fail. Always fix from the earliest divergence forward.
