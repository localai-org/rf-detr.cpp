# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Status

**Projector + encoder (Plan 6a) complete.** The forward pipeline now runs:
DINOv2 backbone → multi-scale projector (4 levels, per-level linear + level
embeddings, concat) → 3-layer transformer encoder. 73 parity checkpoints
all green at 1e-5 absolute tolerance against the numpy reference. Ten tests
pass on a clean build.

`dinov2_forward` now returns a `BackboneOutput { final, multi_scale[4] }`
so downstream consumers access multi-scale features directly. `layer_norm`,
`mha`, and `mlp` moved out of `dinov2.cpp`'s anonymous namespace into a
shared `transformer_ops` module so the encoder, decoder (Plan 6b), and
heads (Plan 6c) can call them.

Plan 6b adds the decoder (300 learnable queries + 3 layers of self-attn +
cross-attn + FFN). Plan 6c wires the class+bbox heads and end-to-end
detect.

The Python conversion script body is still deferred (see Plan 2 Task 3).
The C++ side uses a synthesized F32 GGUF fixture for tests.

## Build

```
git clone --recursive <repo-url>
cd rt-detr.cpp
cmake -B build -DRFDETR_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## License

Apache-2.0. See `LICENSE`.
