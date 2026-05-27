add_executable(rfdetr-cli main.cpp)
# `quantize` subcommand uses ggml/gguf directly. The rfdetr lib already pulls
# in ggml transitively, but link it explicitly so the include paths resolve
# even if that ever changes.
target_link_libraries(rfdetr-cli PRIVATE rfdetr ggml)
target_include_directories(rfdetr-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
