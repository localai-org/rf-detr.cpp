cmake_minimum_required(VERSION 3.14)
project(rfdetr LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(RFDETR_BUILD_TESTS "Build unit tests"             OFF)
option(RFDETR_BUILD_CLI   "Build the rfdetr-cli binary"  ON)
option(RFDETR_SHARED      "Build librfdetr as a shared library" OFF)

# Backend forwarding to ggml (mirrors vibevoice.cpp pattern)
option(RFDETR_GGML_CUDA    "Enable CUDA backend in ggml"    OFF)
option(RFDETR_GGML_METAL   "Enable Metal backend in ggml"   OFF)
option(RFDETR_GGML_VULKAN  "Enable Vulkan backend in ggml"  OFF)
option(RFDETR_GGML_HIPBLAS "Enable HIPBLAS backend in ggml" OFF)

if(RFDETR_GGML_CUDA)
    set(GGML_CUDA    ON CACHE BOOL "" FORCE)
endif()
if(RFDETR_GGML_METAL)
    set(GGML_METAL   ON CACHE BOOL "" FORCE)
endif()
if(RFDETR_GGML_VULKAN)
    set(GGML_VULKAN  ON CACHE BOOL "" FORCE)
endif()
if(RFDETR_GGML_HIPBLAS)
    set(GGML_HIP     ON CACHE BOOL "" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ggml (built but not yet used in this plan — wires the integration early)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# stb (header-only interface target)
add_library(rfdetr_stb INTERFACE)
target_include_directories(rfdetr_stb INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

# librfdetr
set(RFDETR_SOURCES
    src/common.cpp
    src/image_io.cpp
    src/postprocess.cpp
    src/visualize.cpp
    src/cli.cpp
    src/model_loader.cpp
    src/rfdetr.cpp
)

if(RFDETR_SHARED)
    add_library(rfdetr SHARED ${RFDETR_SOURCES})
else()
    add_library(rfdetr STATIC ${RFDETR_SOURCES})
endif()

target_include_directories(rfdetr
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(rfdetr PRIVATE rfdetr_stb ggml)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rfdetr PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# CLI
if(RFDETR_BUILD_CLI AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli/CMakeLists.txt)
    add_subdirectory(examples/cli)
endif()

# Tests
if(RFDETR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
