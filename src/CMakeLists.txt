include(cmake/Library.cmake)
include(cmake/BLAS.cmake)
find_package(guanaqo REQUIRED)
if (KOQKATOO_WITH_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# Configuration options
# ------------------------------------------------------------------------------
add_library(config INTERFACE)
configure_file("koqkatoo/include/koqkatoo/config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/koqkatoo/config.hpp" @ONLY)
target_sources(config INTERFACE FILE_SET headers TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/include"
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/koqkatoo/config.hpp"
)
target_compile_definitions(config INTERFACE
    $<$<BOOL:${KOQKATOO_VERIFY_ASSUMPTIONS}>:KOQKATOO_VERIFY_ASSUMPTIONS>
    $<$<BOOL:${KOQKATOO_WITH_MKL}>:KOQKATOO_WITH_MKL>
    $<$<BOOL:${KOQKATOO_WITH_OPENMP}>:KOQKATOO_WITH_OPENMP>
    $<$<BOOL:${KOQKATOO_WITH_LIBFORK}>:KOQKATOO_WITH_LIBFORK>
)
koqkatoo_configure_interface_library(config EXPORT_PREFIX koqkatoo)

# Main library
# ------------------------------------------------------------------------------
add_library(koqkatoo
    "koqkatoo/src/koqkatoo.cpp"
    "koqkatoo/src/thread-pool.cpp"
    "koqkatoo/src/timing.cpp"
)
target_sources(koqkatoo PUBLIC FILE_SET headers TYPE HEADERS
    BASE_DIRS "koqkatoo/include"
    FILES "koqkatoo/include/koqkatoo/assume.hpp"
          "koqkatoo/include/koqkatoo/kib.hpp"
          "koqkatoo/include/koqkatoo/koqkatoo.hpp"
          "koqkatoo/include/koqkatoo/cneg.hpp"
          "koqkatoo/include/koqkatoo/fork-pool.hpp"
          "koqkatoo/include/koqkatoo/lut.hpp"
          "koqkatoo/include/koqkatoo/matrix-view.hpp"
          "koqkatoo/include/koqkatoo/loop.hpp"
          "koqkatoo/include/koqkatoo/openmp.h"
          "koqkatoo/include/koqkatoo/thread-pool.hpp"
          "koqkatoo/include/koqkatoo/timing.hpp"
          "koqkatoo/include/koqkatoo/unroll.h"
)
target_link_libraries(koqkatoo PUBLIC koqkatoo::config guanaqo::guanaqo)
koqkatoo_configure_library(koqkatoo)
if (KOQKATOO_WITH_OPENMP)
    target_link_libraries(koqkatoo PUBLIC OpenMP::OpenMP_CXX)
endif()
if (KOQKATOO_WITH_ITT)
    target_link_libraries(koqkatoo PUBLIC ITT::ITT)
endif()
if (KOQKATOO_WITH_LIBFORK)
    find_package(libfork CONFIG REQUIRED)
    target_link_libraries(koqkatoo PUBLIC libfork::libfork)
    target_sources(koqkatoo PRIVATE "koqkatoo/src/fork-pool.cpp")
endif()

# Linear algebra routines and BLAS interface
# ------------------------------------------------------------------------------
add_library(linalg
    "linalg/src/blas-interface.cpp"
    "linalg/src/small-potrf.cpp"
)
target_sources(linalg PUBLIC FILE_SET headers TYPE HEADERS
    BASE_DIRS "linalg/include"
    FILES "linalg/include/koqkatoo/linalg/lapack.hpp"
          "linalg/include/koqkatoo/linalg/blas-interface.hpp"
          "linalg/include/koqkatoo/linalg/blas.hpp"
          "linalg/include/koqkatoo/linalg/small-potrf.hpp"
)
target_link_libraries(linalg PUBLIC koqkatoo::koqkatoo koqkatoo::blas-lapack-lib)
target_compile_features(linalg PRIVATE cxx_std_20)
koqkatoo_configure_library(linalg EXPORT_PREFIX koqkatoo)

# Cholesky factorization up/downdates
# ------------------------------------------------------------------------------
add_library(cholundate-headers INTERFACE)
target_sources(cholundate-headers INTERFACE FILE_SET headers TYPE HEADERS
    BASE_DIRS "cholundate/include"
    FILES "cholundate/include/koqkatoo/cholundate/micro-kernels/common.hpp"
          "cholundate/include/koqkatoo/cholundate/micro-kernels/householder-updowndate.hpp"
          "cholundate/include/koqkatoo/cholundate/updown.hpp"
          "cholundate/include/koqkatoo/cholundate/cholundate.hpp"
          "cholundate/include/koqkatoo/cholundate/householder-updowndate.hpp"
          "cholundate/include/koqkatoo/cholundate/householder-updowndate-static.tpp"
          "cholundate/include/koqkatoo/cholundate/householder-updowndate-naive.tpp"
          "cholundate/include/koqkatoo/cholundate/micro-kernels/householder-updowndate-diag.tpp"
          "cholundate/include/koqkatoo/cholundate/micro-kernels/householder-updowndate-tail.tpp"
          "cholundate/include/koqkatoo/cholundate/micro-kernels/householder-updowndate-full.tpp"
          "cholundate/include/koqkatoo/cholundate/householder-updowndate-libfork.tpp"
          "cholundate/include/koqkatoo/cholundate/householder-updowndate-serial.tpp"
          "cholundate/include/koqkatoo/cholundate/householder-updowndate-common.tpp"
)
target_link_libraries(cholundate-headers INTERFACE koqkatoo::koqkatoo)
target_compile_features(cholundate-headers INTERFACE cxx_std_20)

include("cholundate/src/micro-kernels/Codegen.cmake")
codegen_cholundate_microkernels(cholundate cholundate-headers)

add_library(cholundate "cholundate/src/cholundate.cpp")
target_link_libraries(cholundate PRIVATE cholundate-microkernels)
target_link_libraries(cholundate PUBLIC cholundate-headers)
koqkatoo_configure_library(cholundate EXPORT_PREFIX koqkatoo)

# Linear algebra routines for compact batches of matrices
# ------------------------------------------------------------------------------
add_library(linalg-compact-headers INTERFACE)
target_sources(linalg-compact-headers INTERFACE FILE_SET headers TYPE HEADERS
    BASE_DIRS "linalg-compact/include"
    FILES "linalg-compact/include/koqkatoo/linalg-compact/aligned-storage.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/preferred-backend.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/mkl.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/matrix-batch.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/util.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/common.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xshhud.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xtrmm.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xshh.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xpotrf.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xtrtri.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xgemm.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xtrsm.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xshhud-diag.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xpntrf.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/rsqrt.hpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/xtrsm.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/xtrtri.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/xshhud-diag.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/xsyrk.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xshhud.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xgemm-diag-mask.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xtrsm.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xgemm-diag.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xtrtri.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xshhud-diag.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xgemmt-diag-mask.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xtrmm.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xgemmt.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xshh.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xpntrf.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xgemm.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xpotrf.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/micro-kernels/xgemmt-diag.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/xshh.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/xpntrf.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/xgemm.tpp"
          "linalg-compact/include/koqkatoo/linalg-compact/compact/xpotrf.tpp"
)
target_link_libraries(linalg-compact-headers INTERFACE koqkatoo::linalg koqkatoo::cholundate)
target_compile_features(linalg-compact-headers INTERFACE cxx_std_20)

include("linalg-compact/src/micro-kernels/Codegen.cmake")
codegen_compact_microkernels(linalg-compact linalg-compact-headers)

add_library(linalg-compact "linalg-compact/src/compact.cpp")
target_link_libraries(linalg-compact PRIVATE linalg-compact-microkernels)
target_link_libraries(linalg-compact PUBLIC linalg-compact-headers)
koqkatoo_configure_library(linalg-compact EXPORT_PREFIX koqkatoo)

# Optimal control problem backend
# ------------------------------------------------------------------------------
add_library(ocp
    "ocp/src/conversion.cpp"
    "ocp/src/solve.cpp"
    "ocp/src/cyclic-solver.cpp"
)
target_sources(ocp PUBLIC FILE_SET headers TYPE HEADERS
    BASE_DIRS "ocp/include"
    FILES "ocp/include/koqkatoo/ocp/solver/solve.hpp"
          "ocp/include/koqkatoo/ocp/solver/solve-new.tpp"
          "ocp/include/koqkatoo/ocp/solver/storage.hpp"
          "ocp/include/koqkatoo/ocp/conversion.hpp"
          "ocp/include/koqkatoo/ocp/ocp.hpp"
          "ocp/include/koqkatoo/ocp/random-ocp.hpp"
          "ocp/include/koqkatoo/ocp/solver/storage.tpp"
          "ocp/include/koqkatoo/ocp/solver/mat-vec.tpp"
          "ocp/include/koqkatoo/ocp/solver/updowndate.tpp"
          "ocp/include/koqkatoo/ocp/cyclic-solver/cyclic-solver.hpp"
          "ocp/include/koqkatoo/ocp/cyclic-solver/mat-vec.tpp"
          "ocp/include/koqkatoo/ocp/cyclic-solver/packing.tpp"
          "ocp/include/koqkatoo/ocp/cyclic-solver/factor.tpp"
          "ocp/include/koqkatoo/ocp/cyclic-solver/cyclic-storage.ipp"
)
target_link_libraries(ocp PUBLIC koqkatoo::linalg-compact)
target_compile_features(ocp PRIVATE cxx_std_23)
koqkatoo_configure_library(ocp EXPORT_PREFIX koqkatoo)
if (KOQKATOO_WITH_TBB)
    find_package(TBB REQUIRED)
    target_link_libraries(ocp PUBLIC onetbb::onetbb)
endif()
if (KOQKATOO_WITH_LIBFORK)
    koqkatoo_coroutine_flags("ocp/src/solve.cpp")
endif()

# ==============================================================================

# Save the project version to a generated file
configure_file("cmake/koqkatoo-version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/koqkatoo-version.h" @ONLY)
target_sources(koqkatoo PUBLIC FILE_SET headers TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/include"
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/koqkatoo-version.h"
)

# Build time
if (KOQKATOO_WITH_ACCURATE_BUILD_TIME)
    set(KOQKATOO_BUILD_TIME_CPP "${CMAKE_CURRENT_BINARY_DIR}/koqkatoo-build-time.cpp")
    add_custom_target(koqkatoo-build-time-generate
        BYPRODUCTS ${KOQKATOO_BUILD_TIME_CPP}
        COMMAND ${CMAKE_COMMAND}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildTime.cmake"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    target_sources(koqkatoo PRIVATE ${KOQKATOO_BUILD_TIME_CPP})
    add_dependencies(koqkatoo koqkatoo-build-time-generate)
else()
    set(KOQKATOO_BUILD_TIME_CPP "${CMAKE_CURRENT_BINARY_DIR}/koqkatoo-build-time.cpp")
    if (NOT EXISTS ${KOQKATOO_BUILD_TIME_CPP})
        include(cmake/BuildTime.cmake)
    endif()
    target_sources(koqkatoo PRIVATE ${KOQKATOO_BUILD_TIME_CPP})
endif()

# Installation
include(cmake/Install.cmake)
