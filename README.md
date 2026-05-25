# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Status

**Full backbone (Plan 4) complete.** The C++ runtime runs the entire
DINOv2 backbone — patch_embed, CLS token + positional embedding, all 12
transformer blocks (global attention), final layer norm, and 4 multi-scale
feature taps — and matches a pure-numpy reference at 1e-5 absolute
tolerance on all 55 parity checkpoints. Ten tests pass on a clean build.

Plan 5 adds window attention (the only DINOv2 feature still on the global
path). Plans 6+ wire the projector, encoder, decoder, heads, and end-to-end
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
