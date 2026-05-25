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
