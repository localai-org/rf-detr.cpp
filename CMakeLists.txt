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

# BLAS support — when ON, ggml's BLAS backend is built and linked, and rfdetr
# wires it through ggml_backend_sched so large F32 mul_mat ops dispatch to the
# host BLAS library (OpenBLAS / MKL / Accelerate). Default ON when a BLAS
# implementation is detectable; gracefully degrades to the tinyBLAS path
# otherwise.
option(RFDETR_GGML_BLAS    "Enable BLAS backend in ggml (OpenBLAS / MKL / Accelerate)" ON)
set(RFDETR_GGML_BLAS_VENDOR "OpenBLAS" CACHE STRING
    "BLAS vendor passed to ggml (OpenBLAS / Intel10_64lp / Apple / Generic ...)")

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

# BLAS detection. We probe with find_package BEFORE handing the option to ggml
# so a missing library degrades silently instead of erroring the configure step.
set(RFDETR_HAVE_BLAS OFF)
if(RFDETR_GGML_BLAS)
    if(RFDETR_GGML_BLAS_VENDOR STREQUAL "OpenBLAS")
        # Prefer pkg-config: more reliable than CMake's FindBLAS for openblas,
        # which sometimes reports the reference libblas instead.
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(OPENBLAS QUIET openblas)
        endif()
        if(OPENBLAS_FOUND OR EXISTS /usr/lib/x86_64-linux-gnu/libopenblas.so)
            set(RFDETR_HAVE_BLAS ON)
        endif()
    endif()
    if(NOT RFDETR_HAVE_BLAS)
        # Fall back to CMake's generic FindBLAS — works for MKL / Accelerate.
        set(BLA_VENDOR ${RFDETR_GGML_BLAS_VENDOR})
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            set(RFDETR_HAVE_BLAS ON)
        endif()
    endif()
endif()

if(RFDETR_HAVE_BLAS)
    message(STATUS "rfdetr: enabling ggml BLAS backend (vendor=${RFDETR_GGML_BLAS_VENDOR})")
    set(GGML_BLAS         ON CACHE BOOL "" FORCE)
    set(GGML_BLAS_VENDOR  ${RFDETR_GGML_BLAS_VENDOR} CACHE STRING "" FORCE)
else()
    if(RFDETR_GGML_BLAS)
        message(STATUS "rfdetr: BLAS not found — falling back to tinyBLAS only")
    endif()
    set(GGML_BLAS         OFF CACHE BOOL "" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ggml.
#
# CPU optimization flags applied by default (override on cmake command line if
# you need a portable / non-AVX-512 / minimal build):
#
#   GGML_NATIVE=ON      -> compiles with `-march=native`, so the build adapts
#                          to the host (AVX2 on Zen2/Skylake, AVX-512 on Zen4 /
#                          Ice Lake / 9950X3D, etc.). Safe on any x86_64 host
#                          because it never sets explicit feature flags the
#                          local CPU doesn't have.
#   GGML_LLAMAFILE=ON   -> enables ggml's tinyBLAS path (an AVX-512 / AVX2
#                          tiled SGEMM for F32 x F32 mul_mat). Big win for the
#                          F32 backbone hotspot, ~10-15% at T=8 and ~30% at
#                          T=1. tinyBLAS gracefully falls back to AVX2 / scalar
#                          if AVX-512 isn't available, so this is also safe on
#                          older hosts.
#   GGML_OPENMP=ON      -> ggml's default; relies on OpenMP for the thread pool
#                          which has lower wake-up overhead than the pthread
#                          fallback for this workload.
#
# Explicit GGML_AVX512* options are intentionally NOT set: -march=native already
# tells the compiler to emit AVX-512 intrinsics when the host supports them,
# and the explicit options would make the binary non-portable to non-AVX-512
# machines.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE      ON  CACHE BOOL "Enable tinyBLAS SGEMM kernels in ggml" FORCE)
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
    src/rfdetr_capi.cpp
    src/trace.cpp
    src/backend.cpp
    src/dinov2.cpp
    src/transformer_ops.cpp
    src/projector.cpp
    src/two_stage.cpp
    src/decoder.cpp
    src/heads.cpp
    src/rfdetr_model.cpp
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

if(RFDETR_HAVE_BLAS)
    target_compile_definitions(rfdetr PRIVATE RFDETR_HAVE_BLAS=1)
    # ggml's target_link_libraries already pulls ggml-blas into the umbrella
    # `ggml` interface, so we don't need to link ggml-blas here directly.
endif()

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
