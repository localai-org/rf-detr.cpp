# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Status

**Forward-pass foundation (Plan 3) complete.** The C++ runtime now loads
real tensor data into a ggml backend buffer, exposes a thread-local
`trace_callback` for named intermediate tensors, implements DINOv2's
`patch_embed` and one transformer block, and verifies parity against a
pure-numpy reference within 1e-3 absolute tolerance. Ten tests pass on
a clean build. `detect` still returns `not_implemented` until Plans 4–5
land the remaining blocks, projector, encoder/decoder, and heads.

The parity workflow doc is at `docs/parity.md`. Plan 6 will swap the numpy
reference for a torch+rfdetr reference using the same baseline-bundle format.

Note: the Python conversion script body is still deferred — see Plan 2 Task 3.
The C++ side uses a synthesized GGUF fixture for tests, so the forward pass
is fully exercised in CI without the heavy PyTorch dependency.

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
