# AGENTS.md

Durable reference for humans and agents maintaining rf-detr.cpp.

## What this project is

rf-detr.cpp is a C++/ggml inference engine for Roboflow RF-DETR. It runs
detection and segmentation natively on CPU with parity to the PyTorch
reference, and is published on HuggingFace as a set of 44 quantized GGUF
models (5 detection variants x 4 quants + 6 segmentation variants x 4 quants).
The 16 published seg-medium / seg-large / seg-xlarge / seg-2xlarge files
predate the segmentation head block-count fix and need re-converting and
re-publishing; see [`models/MANIFEST.md`](models/MANIFEST.md).

The repo also exposes a flat C-API (`include/rfdetr_capi.h`) intended for
dlopen-based language bindings, and is integrated into LocalAI as a native
backend.

## Repository layout

```
src/                 C++ implementation
                     dinov2, projector, two_stage, decoder, heads,
                     segmentation, transformer_ops, postprocess,
                     model_loader, image_io, backend, trace,
                     rfdetr, rfdetr_model, rfdetr_capi
include/             public headers
                     rfdetr.h        (native C++/C API)
                     rfdetr_capi.h   (flat C-API for FFI / dlopen)
examples/cli/        rfdetr-cli with subcommands:
                     detect, bench, info, quantize
scripts/             converter, batch scripts, benchmark, plot, HF publish
tests/               ctest targets (parity, capi, CI smoke fixtures)
tests/ci/            compare_detections.py used by GitHub Actions smoke job
tests/fixtures/      baseline_torch*.gguf + small CI image and expected JSONs
benchmarks/          results JSON + matplotlib plots
third_party/         vendored ggml submodule, stb_image, patches
third_party/ggml-patches/  local ggml patches applied at configure time
models/              output dir for converted GGUFs (gitignored;
                     MANIFEST.md tracks the expected set)
docs/                conversion, finetuning, parity, variants references
.github/workflows/   ci.yml (build + smoke-test jobs)
```

## Build

```
cmake -B build -DRFDETR_BUILD_TESTS=ON -DRFDETR_BUILD_CLI=ON -DGGML_NATIVE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: 24/24 tests pass on a typical Linux dev box. Parity tests require
`tests/fixtures/baseline_torch*.gguf` to be present (committed to the repo).

### CMake options

| Option                | Default | Purpose                                       |
| --------------------- | ------- | --------------------------------------------- |
| `RFDETR_BUILD_TESTS`  | OFF     | Compile and register ctest targets            |
| `RFDETR_BUILD_CLI`    | ON      | Build the `rfdetr-cli` example binary         |
| `RFDETR_SHARED`       | OFF     | Build `librfdetr` as a shared library (dlopen)|
| `RFDETR_GGML_CUDA`    | OFF     | Forwarded to ggml (`GGML_CUDA`)               |
| `RFDETR_GGML_METAL`   | OFF     | Forwarded to ggml (`GGML_METAL`)              |
| `RFDETR_GGML_VULKAN`  | OFF     | Forwarded to ggml (`GGML_VULKAN`)             |
| `RFDETR_GGML_HIPBLAS` | OFF     | Forwarded to ggml (`GGML_HIPBLAS`)            |

Notes:
- GPU backends are wired through but not exercised in CI. CPU is the
  supported path today.
- For LocalAI integration build with `-DRFDETR_SHARED=ON` to get
  `librfdetr.so`.

## Converting a model

Set up a Python venv from the pinned requirements first (`rfdetr==1.9.0`).

```
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt

.venv/bin/python scripts/convert_rfdetr_to_gguf.py \
    --variant base --dtype f16 \
    --output models/rfdetr-base-f16.gguf
```

`--checkpoint` loads through RF-DETR 1.9's restricted loader. A legacy
checkpoint the loader refuses makes the converter exit 4; `--trust-checkpoint`
is the opt-out, and is only for checkpoints whose source is fully trusted. See
[`docs/finetuning.md`](docs/finetuning.md).

Supported `--variant`:
- Detection: `nano`, `small`, `base`, `medium`, `large`
- Segmentation: `seg-nano`, `seg-small`, `seg-medium`, `seg-large`,
  `seg-xlarge`, `seg-2xlarge`

Supported `--dtype` (handled by the Python writer):
`f32`, `f16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`

For K-quants (`Q4_K`, `Q5_K`, `Q6_K`) the Python `gguf` writer doesn't have
support, so quantize an F32 GGUF with the CLI:

```
build/bin/rfdetr-cli quantize \
    models/rfdetr-base-f32.gguf \
    models/rfdetr-base-q4_K.gguf \
    q4_K
```

Custom fine-tuned checkpoints: pass `--checkpoint path/to/local.pth` to
override the pretrained download.

## Running inference

```
build/bin/rfdetr-cli detect \
    --model models/rfdetr-base-f16.gguf \
    --input image.jpg \
    --threshold 0.5 \
    --output dets.json
```

For segmentation models, also pass `--masks /path/to/mask_output_dir/` to
write one PNG per detection.

Other CLI subcommands: `bench`, `info`, `quantize`.

## GGUF schema

See `docs/conversion.md` for the full schema. Quick summary:

- Format version metadata key: `"2"`.
- Tensor naming convention mirrors the PyTorch state dict, with `.` swapped
  for `.` and a small set of fixups (backbone, projector, decoder, heads,
  segmentation prototype mask head).
- Only 2D weight tensors with both dims `>= 32` and divisible by the block
  size of the target quant get quantized. Embeddings, biases, norms and
  small projection matrices stay F32.

## Test fixtures

CI smoke uses small committed fixtures under `tests/fixtures/ci/`:
- `test_image.jpg`: the test input
- `expected_<variant>-<dtype>.json`: expected detections at T=0.55

To regenerate after a deliberate behavior change:

```
for v in nano-f32 nano-f16 nano-q8_0 nano-q4_K base-f16; do
    ./build/bin/rfdetr-cli detect \
        --model models/rfdetr-${v}.gguf \
        --input tests/fixtures/ci/test_image.jpg \
        --threshold 0.55 \
        --output tests/fixtures/ci/expected_${v}.json \
        --threads 8
done
```

## Parity baselines

`tests/fixtures/baseline_torch.gguf` and `baseline_torch_seg.gguf` are torch
ground-truth bundles used by `test_parity_*`. Regenerate with:

```
.venv/bin/python scripts/gen_torch_baseline.py
```

These need to be regenerated when the architecture changes (any modification
to `src/dinov2.cpp`, `src/decoder.cpp`, `src/heads.cpp`,
`src/segmentation.cpp`, `src/projector.cpp`, `src/transformer_ops.cpp`).

## Benchmarking

The community benchmark used in `BENCHMARK.md`:

```
.venv/bin/python scripts/bench_community.py \
    --rigorous --iters 20 --warmup 5 --cooldown 8 --passes 3
.venv/bin/python scripts/plot_community.py
```

Other benchmark scripts:
- `scripts/bench.py`: quick single-model timing
- `scripts/bench_seg.py`: segmentation-specific timing
- `scripts/bench_threads.py`: thread-count sweep

## Publishing models to HuggingFace

```
.venv/bin/python scripts/publish_hf.py
```

This uploads everything in `models/` plus per-variant READMEs. Requires an
HF token at `~/.cache/huggingface/token` (`huggingface-cli login`).

Repos live under `mudler/rfdetr-cpp-{variant}`. Note the `-cpp` (no dot) in
the HF repo name; that's intentional and shouldn't be changed.

## CI workflow

`.github/workflows/ci.yml` has two jobs:

1. **build**: cmake configure + build + ctest with the committed parity
   baselines.
2. **smoke-test**: downloads `mudler/rfdetr-cpp-nano` quants and
   `mudler/rfdetr-cpp-base-f16` from HF, runs `rfdetr-cli detect` on
   `tests/fixtures/ci/test_image.jpg`, and compares the JSON output against
   the committed `expected_*.json` via `tests/ci/compare_detections.py`.

The comparison uses class + IoU greedy matching and tolerates score ties,
so small numeric drift on the last decimal won't break CI.

## ggml integration

ggml is vendored as a submodule at `third_party/ggml`. Local
performance/debug patches live in `third_party/ggml-patches/` and are
applied at CMake configure time by `scripts/apply_ggml_patches.sh`.

Current patches:
- `0001-ggml-cpu-fold-broadcast-iterations-in-llamafile_sgem.patch`
- `0002-ggml-cpu-per-op-profile-gated-on-GGML_PROFILE_OPS-1.patch`

To add a new patch:
1. Edit the submodule directly to develop the change.
2. `git -C third_party/ggml format-patch -1` to generate the patch file.
3. Copy the generated `.patch` to `third_party/ggml-patches/`.
4. Reset the submodule to its tracked SHA.
5. Re-run `scripts/apply_ggml_patches.sh` and the full test suite to verify
   the patch applies cleanly.

To bump ggml:
1. Update the submodule SHA.
2. Re-run `scripts/apply_ggml_patches.sh`. Resolve any rejected hunks.
3. Run `ctest --output-on-failure` to catch any API breakage.

## LocalAI integration

A native backend lives in the LocalAI repo at
`LocalAI/backend/go/rfdetr-cpp/`. It dlopens `librfdetr.so` (built with
`RFDETR_SHARED=ON`) and uses the flat C-API in `include/rfdetr_capi.h`.

Symbols the LocalAI side depends on:

```
rfdetr_capi_load
rfdetr_capi_unload
rfdetr_capi_detect_path
rfdetr_capi_detect_buffer
rfdetr_capi_free_string
rfdetr_capi_get_n_detections
rfdetr_capi_get_detection_class_id
rfdetr_capi_get_detection_box
rfdetr_capi_get_detection_score
rfdetr_capi_get_detection_class_name
rfdetr_capi_get_detection_mask_png
```

Don't remove or change the signature of any of these without bumping a
version field on the LocalAI side. Additions are fine.

## Common maintenance tasks

### Add a new RF-DETR variant

1. Add the variant config in `scripts/convert_rfdetr_to_gguf.py` (the
   `VARIANT_CFG` table near the top).
2. Add it to the `--variant` argparse choices.
3. Convert + quantize and update `models/MANIFEST.md`.

The C++ loader is metadata-driven, so no source changes are typically
needed.

### Update to a newer upstream rfdetr Python version

1. Bump the version in the converter's `pip install` instructions.
2. Regenerate the parity baselines via `scripts/gen_torch_baseline.py`.
3. Run the full test suite. Any parity drift will surface in the
   `test_parity_*` targets.

### Update to a newer ggml

1. Bump the submodule SHA.
2. Re-apply local patches via `scripts/apply_ggml_patches.sh`.
3. Run `ctest --output-on-failure`.

### Add a new quantization type

1. Extend `examples/cli/main.cpp::cmd_quantize` with the new type mapping.
2. If the heuristic in `should_quantize_tensor` (in
   `scripts/convert_rfdetr_to_gguf.py`) needs to skip more tensor shapes
   for the new quant, add a case there.
3. Regenerate the accuracy sweep with `scripts/sweep_accuracy.py` and
   update `BENCHMARK.md` if the numbers change materially.
