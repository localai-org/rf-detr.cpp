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
