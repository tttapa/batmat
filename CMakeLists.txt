cmake_minimum_required(VERSION 3.24)
project(batmat
    VERSION 1.0.0
    DESCRIPTION "Batched linear algebra routines"
    HOMEPAGE_URL "https://github.com/tttapa/batmat"
    LANGUAGES CXX
)
include(CTest)

# Options
option(BATMAT_WITH_MKL
    "Use the Intel MKL" Off)
option(BATMAT_WITH_OPENBLAS
    "Use the OpenBLAS library" Off)
option(BATMAT_WITH_OPENMP
    "Enable parallelization using OpenMP" Off)
set(BATMAT_DENSE_REAL_TYPE "double" CACHE STRING
    "The main floating point type for representing real numbers")
set(BATMAT_DENSE_INDEX_TYPE "long long" CACHE STRING
    "The main integer type for indices and sizes")
option(BATMAT_WITH_GSI_HPC_SIMD
    "If enabled, use the GSI-HPC/simd library instead of GCC's <experimental/simd>" Off)
# Target options
option(BATMAT_HAVE_TWO_512_FMA_UNITS
    "Does the CPU have 0.5 CPI 512-bit FMA" Off)
# Enable/disable optional components
option(BATMAT_WITH_TESTS
    "Build the tests" ${BUILD_TESTING})
option(BATMAT_WITH_BENCHMARKS
    "Build the benchmarks" Off)
# Developer options
option(BATMAT_WARNINGS_AS_ERRORS
    "Enable -Werror or /WX" Off)
option(BATMAT_WITH_COVERAGE
    "Generate coverage information" Off)
set(BATMAT_DOXYFILE "Doxyfile" CACHE FILEPATH
    "The Doxyfile to use for the docs target")
option(BATMAT_WITH_ACCURATE_BUILD_TIME
    "Update the build time on every build" On)
option(BATMAT_VERIFY_ASSUMPTIONS
    "Check assumptions at run time instead of blindly assuming" Off)
option(BATMAT_DEVELOPER_MODE
    "Enable developer options such as sccache and colored compiler output" Off)

# Installation paths
include(GNUInstallDirs)
set(BATMAT_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "Installation directory for archives and libraries")
set(BATMAT_INSTALL_CMAKEDIR "${BATMAT_INSTALL_LIBDIR}/cmake/batmat"
    CACHE PATH "Installation directory for CMake configuration files")
set(BATMAT_INSTALL_BINDIR "${CMAKE_INSTALL_BINDIR}"
    CACHE PATH "Installation directory for binaries and DLLs")
set(BATMAT_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}"
    CACHE PATH "Installation directory for headers")
option(BATMAT_STANDALONE
    "Install with relative RPATH to locate its own shared libraries" On)

# Development
if (PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS On)
endif()
if (BATMAT_DEVELOPER_MODE)
    include(cmake/Develop.cmake)
endif()

# Compiler warnings
include(cmake/Warnings.cmake)
add_warnings_target(warnings ${BATMAT_WARNINGS_AS_ERRORS})
add_library(batmat::warnings ALIAS warnings)

# Compiler options
if (MSVC)
    add_compile_options("/bigobj")
endif()
set(CMAKE_RELEASE_POSTFIX "")
set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd")
set(CMAKE_MINSIZEREL_POSTFIX "_rs")

# Coverage
if (BATMAT_WITH_COVERAGE)
    add_custom_target(coverage
        ${CMAKE_CURRENT_LIST_DIR}/scripts/coverage.sh
        ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        USES_TERMINAL)
    add_compile_options("--coverage")
    add_link_options("--coverage")
    add_dependencies(coverage batmat::tests)
endif()

# Documentation
find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_custom_target(docs Doxygen::doxygen ${BATMAT_DOXYFILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/doxygen)
endif()

# Libraries
add_subdirectory(src)

# Tests
if (BATMAT_WITH_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Benchmarks
if (BATMAT_WITH_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
