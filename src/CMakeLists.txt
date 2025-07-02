include(cmake/Library.cmake)
include(cmake/BLAS.cmake)
find_package(guanaqo REQUIRED)
if (BATMAT_WITH_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()
if (BATMAT_WITH_GSI_HPC_SIMD)
    find_package(gsi-hpc-simd REQUIRED)
endif()

# Configuration options
# ------------------------------------------------------------------------------
add_library(config INTERFACE)
configure_file("batmat/include/batmat/config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/batmat/config.hpp" @ONLY)
target_sources(config INTERFACE FILE_SET headers TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/include"
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/batmat/config.hpp"
)
target_compile_definitions(config INTERFACE
    $<$<BOOL:${BATMAT_VERIFY_ASSUMPTIONS}>:BATMAT_VERIFY_ASSUMPTIONS>
    $<$<BOOL:${BATMAT_WITH_MKL}>:BATMAT_WITH_MKL>
    $<$<BOOL:${BATMAT_WITH_OPENMP}>:BATMAT_WITH_OPENMP>
    $<$<BOOL:${BATMAT_WITH_GSI_HPC_SIMD}>:BATMAT_WITH_GSI_HPC_SIMD>
)
batmat_configure_interface_library(config)

# Main library
# ------------------------------------------------------------------------------
add_library(batmat
    "batmat/src/batmat.cpp"
    "batmat/src/thread-pool.cpp"
)
target_sources(batmat PUBLIC FILE_SET headers TYPE HEADERS
    BASE_DIRS "batmat/include"
    FILES "batmat/include/batmat/assume.hpp"
          "batmat/include/batmat/kib.hpp"
          "batmat/include/batmat/loop.hpp"
          "batmat/include/batmat/lut.hpp"
          "batmat/include/batmat/openmp.h"
          "batmat/include/batmat/unroll.h"
          "batmat/include/batmat/simd.hpp"
          "batmat/include/batmat/thread-pool.hpp"
          "batmat/include/batmat/matrix/layout.hpp"
          "batmat/include/batmat/matrix/matrix.hpp"
          "batmat/include/batmat/matrix/storage.hpp"
          "batmat/include/batmat/matrix/view.hpp"
          "batmat/include/batmat/ops/cneg.hpp"
          "batmat/include/batmat/ops/gather.hpp"
          "batmat/include/batmat/ops/transpose.hpp"
          "batmat/include/batmat/ops/rotate.hpp"
          "batmat/include/batmat/ops/rsqrt.hpp"
          "batmat/include/batmat/platform/platform.hpp"
          "batmat/include/batmat/platform/avx-512.hpp"
          "batmat/include/batmat/platform/avx2.hpp"
          "batmat/include/batmat/platform/generic.hpp"
          "batmat/include/batmat/platform/neon.hpp"
)
target_link_libraries(batmat PUBLIC batmat::config guanaqo::guanaqo)
batmat_configure_library(batmat)
if (BATMAT_WITH_OPENMP)
    target_link_libraries(batmat PUBLIC OpenMP::OpenMP_CXX)
endif()
if (BATMAT_WITH_GSI_HPC_SIMD)
    target_link_libraries(batmat PUBLIC gsi-hpc-simd::simd)
endif()

# Linear algebra routines for compact batches of matrices
# ------------------------------------------------------------------------------
target_sources(batmat PUBLIC FILE_SET headers TYPE HEADERS
    BASE_DIRS "batmat/include"
    FILES "batmat/include/batmat/linalg/micro-kernels/gemm.hpp"
          "batmat/include/batmat/linalg/micro-kernels/trsm.hpp"
          "batmat/include/batmat/linalg/micro-kernels/potrf.hpp"
          "batmat/include/batmat/linalg/micro-kernels/gemm-diag.hpp"
          "batmat/include/batmat/linalg/micro-kernels/trtri.hpp"
          "batmat/include/batmat/linalg/shift.hpp"
          "batmat/include/batmat/linalg/flops.hpp"
          "batmat/include/batmat/linalg/structure.hpp"
          "batmat/include/batmat/linalg/trsm.hpp"
          "batmat/include/batmat/linalg/compress.hpp"
          "batmat/include/batmat/linalg/simdify.hpp"
          "batmat/include/batmat/linalg/potrf.hpp"
          "batmat/include/batmat/linalg/gemm-diag.hpp"
          "batmat/include/batmat/linalg/triangular.hpp"
          "batmat/include/batmat/linalg/trtri.hpp"
          "batmat/include/batmat/linalg/uview.hpp"
          "batmat/include/batmat/linalg/copy.hpp"
          "batmat/include/batmat/linalg/gemm.hpp"
)

include("batmat/src/linalg/micro-kernels/Codegen.cmake")
batmat_codegen_micro_kernels(batmat-linalg batmat)
target_link_libraries(batmat PRIVATE batmat-linalg-micro-kernels)

# ==============================================================================

# OCP solver
add_library(cyclocp
    "cyclocp/src/cyclocp.cpp"
    "cyclocp/src/cyclocp-storage.cpp"
)
target_sources(cyclocp PUBLIC FILE_SET headers TYPE HEADERS
    BASE_DIRS "cyclocp/include"
    FILES "cyclocp/include/cyclocp/cyclocp.hpp"
          "cyclocp/include/cyclocp/cyclocp-storage.hpp"
          "cyclocp/include/cyclocp/ocp.hpp"
          "cyclocp/include/cyclocp/implementation/data.tpp"
          "cyclocp/include/cyclocp/implementation/factor.tpp"
          "cyclocp/include/cyclocp/implementation/mat-vec.tpp"
          "cyclocp/include/cyclocp/implementation/indexing.tpp"
          "cyclocp/include/cyclocp/implementation/sparse.tpp"
          "cyclocp/include/cyclocp/implementation/update.tpp"
          "cyclocp/include/cyclocp/implementation/pcg.tpp"
          "cyclocp/include/cyclocp/implementation/solve.tpp")
target_link_libraries(cyclocp PUBLIC batmat)
find_package(guanaqo REQUIRED COMPONENTS BLAS)  # TODO
target_link_libraries(cyclocp PUBLIC guanaqo::blas)

# ==============================================================================

# Save the project version to a generated file
configure_file("cmake/batmat-version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/batmat-version.h" @ONLY)
target_sources(batmat PUBLIC FILE_SET headers TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/include"
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/batmat-version.h"
)

# Build time
if (BATMAT_WITH_ACCURATE_BUILD_TIME)
    set(BATMAT_BUILD_TIME_CPP "${CMAKE_CURRENT_BINARY_DIR}/batmat-build-time.cpp")
    add_custom_target(batmat-build-time-generate
        BYPRODUCTS ${BATMAT_BUILD_TIME_CPP}
        COMMAND ${CMAKE_COMMAND}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildTime.cmake"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    target_sources(batmat PRIVATE ${BATMAT_BUILD_TIME_CPP})
    add_dependencies(batmat batmat-build-time-generate)
else()
    set(BATMAT_BUILD_TIME_CPP "${CMAKE_CURRENT_BINARY_DIR}/batmat-build-time.cpp")
    if (NOT EXISTS ${BATMAT_BUILD_TIME_CPP})
        include(cmake/BuildTime.cmake)
    endif()
    target_sources(batmat PRIVATE ${BATMAT_BUILD_TIME_CPP})
endif()

# Installation
include(cmake/Install.cmake)
