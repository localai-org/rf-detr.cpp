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

rfdetr_add_test(test_model_loader)
add_dependencies(test_model_loader rfdetr_model_fixture)
add_dependencies(test_model_loader rfdetr_model_fixture_missing)

rfdetr_add_test(test_capi)
add_dependencies(test_capi rfdetr_model_fixture rfdetr_fixtures)

rfdetr_add_test(test_capi_flat)
add_dependencies(test_capi_flat rfdetr_model_fixture rfdetr_fixtures)

# ---------------------------------------------------------------------------
# Plan 13: torch parity baseline (regenerated on demand, NOT in ALL).
#
# To regenerate the baseline:
#   cmake --build build --target rfdetr_baseline_torch
#
# Slow: requires .venv/ with rfdetr 1.7.0 + torch 2.5.1 + gguf + numpy.
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
    COMMENT "Generating torch parity baseline (requires .venv with rfdetr 1.7.0)"
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

# ---------------------------------------------------------------------------
# DISABLED (Plan 7 transitional state):
#
# test_parity_full_forward exercises a full forward pass against a numpy
# reference that was built for the v1 GGUF schema. The schema was rewritten
# to v2 in Plan 7 (real rfdetr-base architecture). The numpy reference,
# the parity baseline generator (scripts/gen_numpy_baseline.py), and the
# C++ numerical modules (dinov2, encoder, decoder, projector, heads) still
# expect the v1 shapes and metadata keys.
#
# Plans 8-12 rewrite each numerical module against the v2 schema. Plan 13
# rewrites the parity reference using torch+rfdetr (matching real upstream).
# Re-enable once Plan 13 lands.
#
# The fixture generator (gen_model_gguf), conversion script, and C++ loader
# already understand v2; the in-between modules are the gap.
# ---------------------------------------------------------------------------
if(0)  # Plan 7 transitional — re-enable after Plan 13
    # Parity baseline bundle: numpy reference for the full DINOv2 backbone.
    # See docs/parity.md and scripts/gen_numpy_baseline.py.
    find_program(PYTHON3 python3 REQUIRED)
    add_custom_command(
        OUTPUT  ${RFDETR_TEST_FIXTURES}/baseline_backbone.gguf
        COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/scripts/gen_numpy_baseline.py
                --model  ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf
                --output ${RFDETR_TEST_FIXTURES}/baseline_backbone.gguf
        DEPENDS ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf
                ${CMAKE_SOURCE_DIR}/scripts/gen_numpy_baseline.py
        VERBATIM
    )
    add_custom_target(rfdetr_baseline_backbone ALL
        DEPENDS ${RFDETR_TEST_FIXTURES}/baseline_backbone.gguf)

    rfdetr_add_test(test_parity_full_forward)
    target_link_libraries(test_parity_full_forward PRIVATE ggml)
    add_dependencies(test_parity_full_forward
        rfdetr_model_fixture_seeded
        rfdetr_baseline_backbone)
endif()  # Plan 7 disable
