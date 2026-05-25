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

# Parity baseline bundle: numpy reference for patch_embed + backbone block 0.
# See docs/parity.md and scripts/gen_numpy_baseline.py.
find_program(PYTHON3 python3 REQUIRED)
add_custom_command(
    OUTPUT  ${RFDETR_TEST_FIXTURES}/baseline_block0.gguf
    COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/scripts/gen_numpy_baseline.py
            --model  ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf
            --output ${RFDETR_TEST_FIXTURES}/baseline_block0.gguf
    DEPENDS ${RFDETR_TEST_FIXTURES}/model_base_seeded.gguf
            ${CMAKE_SOURCE_DIR}/scripts/gen_numpy_baseline.py
    VERBATIM
)
add_custom_target(rfdetr_baseline_block0 ALL
    DEPENDS ${RFDETR_TEST_FIXTURES}/baseline_block0.gguf)

rfdetr_add_test(test_model_loader)
add_dependencies(test_model_loader rfdetr_model_fixture)
add_dependencies(test_model_loader rfdetr_model_fixture_missing)

rfdetr_add_test(test_capi)
add_dependencies(test_capi rfdetr_model_fixture rfdetr_fixtures)

rfdetr_add_test(test_capi_flat)
add_dependencies(test_capi_flat rfdetr_model_fixture rfdetr_fixtures)

rfdetr_add_test(test_parity_block0)
target_link_libraries(test_parity_block0 PRIVATE ggml)
add_dependencies(test_parity_block0
    rfdetr_model_fixture_seeded
    rfdetr_baseline_block0)
