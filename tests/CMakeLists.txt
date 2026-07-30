# Each test_*.cpp builds into its own binary and is registered as a ctest test.

set(RFDETR_TEST_FIXTURES ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

function(rfdetr_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE rfdetr rfdetr_stb)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_definitions(${name} PRIVATE
        RFDETR_TEST_FIXTURES="${RFDETR_TEST_FIXTURES}")
    add_test(NAME ${name} COMMAND ${name})
endfunction()

rfdetr_add_test(test_common)

# Build the fixture generator
add_executable(gen_fixture fixtures/gen_fixture.cpp)
target_link_libraries(gen_fixture PRIVATE rfdetr_stb)

# Run it at build time to produce tests/fixtures/cats.png
add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/cats.png
    COMMAND gen_fixture ${RFDETR_TEST_FIXTURES}
    DEPENDS gen_fixture
    VERBATIM
)
add_custom_target(rfdetr_fixtures ALL DEPENDS ${RFDETR_TEST_FIXTURES}/cats.png)

rfdetr_add_test(test_image_io)
add_dependencies(test_image_io rfdetr_fixtures)

rfdetr_add_test(test_postprocess)

rfdetr_add_test(test_visualize)

rfdetr_add_test(test_cli_smoke)

if(TARGET rfdetr-cli)
    rfdetr_add_test(test_cli_integration)
    target_compile_definitions(test_cli_integration PRIVATE
        RFDETR_CLI_BINARY="$<TARGET_FILE:rfdetr-cli>")
    add_dependencies(test_cli_integration rfdetr-cli rfdetr_model_fixture)
endif()

# Synthesized model GGUF for tests (no PyTorch needed)
add_executable(gen_model_gguf fixtures/gen_model_gguf.cpp)
target_link_libraries(gen_model_gguf PRIVATE ggml)
add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/model_base.gguf
    COMMAND gen_model_gguf ${RFDETR_TEST_FIXTURES}/model_base.gguf
    DEPENDS gen_model_gguf
    VERBATIM
)
add_custom_target(rfdetr_model_fixture ALL DEPENDS ${RFDETR_TEST_FIXTURES}/model_base.gguf)

add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/model_base_missing.gguf
    COMMAND gen_model_gguf ${RFDETR_TEST_FIXTURES}/model_base_missing.gguf --missing backbone.norm.weight
    DEPENDS gen_model_gguf
    VERBATIM
)
add_custom_target(rfdetr_model_fixture_missing ALL DEPENDS ${RFDETR_TEST_FIXTURES}/model_base_missing.gguf)

add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf
    COMMAND gen_model_gguf ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf --seed 42
    DEPENDS gen_model_gguf
    VERBATIM
)
add_custom_target(rfdetr_model_fixture_seeded ALL
    DEPENDS ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf)

# rfdetr.preprocess.resize_mode variants. model_base.gguf above deliberately
# omits the key, so these cover the two explicit values: a valid one that must
# enable the 1.9 path, and a bogus one that must be rejected rather than
# silently falling back to legacy.
add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/model_base_resize_bilinear.gguf
    COMMAND gen_model_gguf ${RFDETR_TEST_FIXTURES}/model_base_resize_bilinear.gguf
            --resize-mode bilinear_no_antialias
    DEPENDS gen_model_gguf
    VERBATIM
)
add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/model_base_resize_bogus.gguf
    COMMAND gen_model_gguf ${RFDETR_TEST_FIXTURES}/model_base_resize_bogus.gguf
            --resize-mode lanczos_from_the_future
    DEPENDS gen_model_gguf
    VERBATIM
)
add_custom_target(rfdetr_model_fixture_resize_mode ALL
    DEPENDS ${RFDETR_TEST_FIXTURES}/model_base_resize_bilinear.gguf
            ${RFDETR_TEST_FIXTURES}/model_base_resize_bogus.gguf)

rfdetr_add_test(test_model_loader)
add_dependencies(test_model_loader rfdetr_model_fixture)
add_dependencies(test_model_loader rfdetr_model_fixture_missing)
add_dependencies(test_model_loader rfdetr_model_fixture_resize_mode)

rfdetr_add_test(test_capi)
add_dependencies(test_capi rfdetr_model_fixture rfdetr_fixtures)

rfdetr_add_test(test_capi_flat)
add_dependencies(test_capi_flat rfdetr_model_fixture rfdetr_fixtures)

# Exercises every flat-C-API accessor (n_detections, class_id, class_name,
# box, score, mask_png) on a real GGUF. Skips gracefully if no real model
# is present.
rfdetr_add_test(test_capi_accessors)
target_compile_definitions(test_capi_accessors PRIVATE
    RFDETR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Plan 13: torch parity baseline (regenerated on demand, NOT in ALL).
#
# To regenerate the baseline:
#   cmake --build build --target rfdetr_baseline_torch
#
# Slow: requires .venv/ with rfdetr 1.9.0 + torch 2.5.1 + gguf + numpy.
# Runs real rfdetr-base forward with hooks at named checkpoints. The output
# is consumed by test_parity_torch and (later, in Plans 8-12) by the per-
# module C++ vs torch comparison loops.
# ---------------------------------------------------------------------------
find_program(PYTHON3 python3 REQUIRED)
# Prefer .venv/bin/python if it exists (has rfdetr + torch installed); fall back
# to the system python3 (user is responsible for installing the deps).
set(_torch_baseline_python "${PYTHON3}")
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(_torch_baseline_python "${CMAKE_SOURCE_DIR}/.venv/bin/python")
endif()
add_custom_target(rfdetr_baseline_torch
    COMMAND ${_torch_baseline_python} ${CMAKE_SOURCE_DIR}/scripts/gen_torch_baseline.py
            --output ${RFDETR_TEST_FIXTURES}/baseline_torch.gguf
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating torch parity baseline (requires .venv with rfdetr 1.9.0)"
    VERBATIM)

# Bundle loader stub: verifies format + required checkpoint coverage.
# Skips gracefully if the baseline isn't generated (so a fresh clone passes
# ctest without needing the .venv). Plans 8-12 add the actual per-module
# C++ vs torch tensor comparison loops.
rfdetr_add_test(test_parity_torch)
target_link_libraries(test_parity_torch PRIVATE ggml)

# Plan 8: C++ windowed DinoV2 backbone vs torch parity baseline.
# Skips gracefully if either the real rfdetr-base GGUF model or the torch
# baseline isn't present.
rfdetr_add_test(test_parity_backbone)
target_link_libraries(test_parity_backbone PRIVATE ggml)

# Plan 9: C++ MultiScaleProjector vs torch parity baseline.
# Phase 1 feeds the torch baseline's backbone multiscale outputs into the C++
# projector (isolating the projector). Phase 2 runs backbone+projector end-
# to-end and reports drift. Skips gracefully if either input is missing.
rfdetr_add_test(test_parity_projector)
target_link_libraries(test_parity_projector PRIVATE ggml)

# Plan 10: C++ two-stage init module vs torch parity baseline.
# Module-isolated: feeds the baseline's parity.projector.output directly into
# C++ two_stage_forward. Skips gracefully if either the model or the baseline
# isn't present.
rfdetr_add_test(test_parity_two_stage)
target_link_libraries(test_parity_two_stage PRIVATE ggml)

# Plan 11: C++ LWDETR decoder vs torch parity baseline.
# Module-isolated: feeds baseline projector.output (as flattened memory) and
# decoder.input.{tgt,reference_points} into C++ decoder_forward. Diffs every
# sub-module checkpoint (self_attn, norm1, cross_attn, norm2, linear1,
# linear2, layer output) for each of the 3 layers + the final decoder.norm.
rfdetr_add_test(test_parity_decoder)
target_link_libraries(test_parity_decoder PRIVATE ggml)

# Plan 14: C++ shared class + bbox heads vs torch parity baseline.
# Module-isolated: feeds baseline decoder.norm.output into C++ heads forward,
# diffs against baseline heads.class_logits[..., -1] / heads.bbox_pred[..., -1]
# (the last-layer slice; what inference actually consumes).
rfdetr_add_test(test_parity_heads)
target_link_libraries(test_parity_heads PRIVATE ggml)

# Plan 14: end-to-end parity (full C++ rfdetr_model_forward vs torch baseline).
# Uses the baseline's preprocess.input as the model input; logs (does not
# enforce) cumulative drift against the baseline's final class_logits[-1].
rfdetr_add_test(test_parity_e2e)
target_link_libraries(test_parity_e2e PRIVATE ggml)

# Plan C: segmentation head end-to-end parity vs the seg torch baselines.
#
# Runs two variants: seg-nano (4 seg blocks) and seg-medium (5 blocks). The
# 5-block variant is the acceptance gate for "one segmentation block per
# decoder layer" — seg-nano cannot catch a dropped block. Each variant skips
# only if BOTH of its fixtures are absent; a fixture that is present but
# unusable is a failure, not a skip.
#
# To regenerate all four artifacts (all gitignored, ~640MB total, needs the
# .venv with rfdetr 1.9.0 + torch 2.5.1):
#   .venv/bin/python scripts/gen_torch_baseline.py --seg-variant nano \
#       --output tests/fixtures/baseline_torch_seg.gguf
#   .venv/bin/python scripts/gen_torch_baseline.py --seg-variant medium \
#       --output tests/fixtures/baseline_torch_seg_medium.gguf
#   .venv/bin/python scripts/convert_rfdetr_to_gguf.py --variant seg-nano \
#       --dtype f32 --output models/rfdetr-seg-nano-f32.gguf
#   .venv/bin/python scripts/convert_rfdetr_to_gguf.py --variant seg-medium \
#       --dtype f32 --output models/rfdetr-seg-medium-f32.gguf
rfdetr_add_test(test_parity_segmentation)
target_link_libraries(test_parity_segmentation PRIVATE ggml)

# Plan 8 (final): Q8_0 quantized model sanity check.
# Skips gracefully if models/rfdetr-base-q8_0.gguf isn't present.
rfdetr_add_test(test_q8_quant)
target_link_libraries(test_q8_quant PRIVATE ggml)
target_compile_definitions(test_q8_quant PRIVATE
    RFDETR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Q4_0 quantized model sanity check. Same structural test as Q8_0; Q4_0
# itself ships with caveats documented in BENCHMARK.md.
# Skips gracefully if models/rfdetr-base-q4_0.gguf isn't present.
rfdetr_add_test(test_q4_quant)
target_link_libraries(test_q4_quant PRIVATE ggml)
target_compile_definitions(test_q4_quant PRIVATE
    RFDETR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# K-quant (Q4_K / Q5_K / Q6_K) sanity checks. Generated via
#   build/bin/rfdetr-cli quantize ... q{4,5,6}_K
# Skips per-file gracefully if absent.
rfdetr_add_test(test_kquant)
target_link_libraries(test_kquant PRIVATE ggml)
target_compile_definitions(test_kquant PRIVATE
    RFDETR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# F16 model sanity + load test. Regression guard for a read-overrun bug
# where the loader assumed pos_embed was F32 and crashed on F16 GGUFs.
# Skips gracefully if models/rfdetr-base-f16.gguf isn't present. Generate:
#   .venv/bin/python scripts/convert_rfdetr_to_gguf.py --variant base \
#       --dtype f16 --output models/rfdetr-base-f16.gguf
rfdetr_add_test(test_f16_model)
target_link_libraries(test_f16_model PRIVATE ggml)
target_compile_definitions(test_f16_model PRIVATE
    RFDETR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Per-variant load smoke-test: loads each of the 5 detection variants and
# verifies the metadata round-trips. Skips per-variant gracefully if the
# matching GGUF isn't present, so a fresh clone passes ctest before running
# scripts/convert_all_variants.sh.
rfdetr_add_test(test_variants)
target_compile_definitions(test_variants PRIVATE
    RFDETR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Fine-tune check: load a custom-num_classes checkpoint and verify the
# C-API reports the right class count. Skips gracefully if /tmp/custom5.gguf
# isn't present (generate via scripts/build_custom_checkpoint.py +
# scripts/convert_rfdetr_to_gguf.py --checkpoint).
rfdetr_add_test(test_custom_classes)

# Plan GPU Task 6: backend GPU smoke test. Verifies the gpu/sched invariant
# (gpu != null implies sched != null; no gpu implies no sched) plus cpu != null
# and that free_backend_ctx works. Build-independent: passes on CPU-only
# builds (prints "no GPU backend") and on GPU builds with/without a device.
rfdetr_add_test(test_backend_gpu)

