include(cmake/Library.cmake)
include(cmake/BLAS.cmake)
find_package(guanaqo REQUIRED COMPONENTS BLAS)
if (BATMAT_WITH_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
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
)
batmat_configure_interface_library(config EXPORT_PREFIX batmat)

# Main library
# ------------------------------------------------------------------------------
add_library(batmat
    "batmat/src/batmat.cpp"
    "batmat/src/thread-pool.cpp"
)
target_sources(batmat PUBLIC FILE_SET headers TYPE HEADERS
    BASE_DIRS "batmat/include"
    FILES "batmat/include/batmat/assume.hpp"
          "batmat/include/batmat/lut.hpp"
          "batmat/include/batmat/loop.hpp"
          "batmat/include/batmat/openmp.h"
          "batmat/include/batmat/unroll.h"
          "batmat/include/batmat/ops/cneg.hpp"
          "batmat/include/batmat/ops/rotate.hpp"
          "batmat/include/batmat/ops/rsqrt.hpp"
)
target_link_libraries(batmat PUBLIC batmat::config guanaqo::guanaqo)
batmat_configure_library(batmat)
if (BATMAT_WITH_OPENMP)
    target_link_libraries(batmat PUBLIC OpenMP::OpenMP_CXX)
endif()

# Linear algebra routines for compact batches of matrices
# ------------------------------------------------------------------------------
add_library(linalg-headers INTERFACE)
target_sources(linalg-headers INTERFACE FILE_SET headers TYPE HEADERS
    BASE_DIRS "batmat/include"
    FILES "batmat/include/batmat/linalg/micro-kernels/gemm.hpp"
            "batmat/include/batmat/linalg/micro-kernels/gemm.tpp"
)
target_link_libraries(linalg-headers INTERFACE batmat::batmat)
target_compile_features(linalg-headers INTERFACE cxx_std_20)

include("batmat/src/linalg/micro-kernels/Codegen.cmake")
batmat_codegen_micro_kernels(linalg linalg-headers)

add_library(linalg "batmat/src/linalg/linalg.cpp")
target_link_libraries(linalg PRIVATE linalg-micro-kernels)
target_link_libraries(linalg PUBLIC linalg-headers)
batmat_configure_library(linalg EXPORT_PREFIX batmat)

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
