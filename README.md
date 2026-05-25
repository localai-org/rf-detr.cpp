# rt-detr.cpp

C++/ggml inference for [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

Project layout mirrors [vibevoice.cpp](https://github.com/mudler/vibevoice.cpp).
See `docs/superpowers/specs/2026-05-25-rfdetr-cpp-design.md` for the design.

## Status

**Loader (Plan 2) complete.** The repo can convert an upstream `rfdetr-base`
PyTorch checkpoint to GGUF, load it from C++ (variant detection, config
parsing, tensor inventory validation), and report it via `rfdetr-cli info`.
The flat dlopen ABI is wired through to a JSON envelope. Nine tests pass on
a clean build. Inference (`detect`) still returns `not_implemented` —
Plan 3 builds the forward graph and parity harness.

Note: the Python conversion script body is deferred — see Plan 2 Task 3.
The C++ side uses a synthesized GGUF fixture for tests, so the loader is
fully exercised in CI without the heavy PyTorch dependency.

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
