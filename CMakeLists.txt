cmake_minimum_required(VERSION 3.24)
project(koqkatoo
    VERSION 1.0.0
    DESCRIPTION "Linear solvers and adapters for solving KKT systems"
    HOMEPAGE_URL "https://github.com/tttapa/koqkatoo"
    LANGUAGES CXX
)
include(CTest)

# Options
include(CMakeDependentOption)
option(KOQKATOO_WITH_MKL
    "Use the Intel MKL" Off)
option(KOQKATOO_WITH_OPENMP
    "Enable parallelization using OpenMP" Off)
option(KOQKATOO_WITH_LIBFORK
    "Use libfork for parallelization" Off)
set(KOQKATOO_DENSE_REAL_TYPE "double" CACHE STRING
    "The main floating point type for representing real numbers")
set(KOQKATOO_DENSE_INDEX_TYPE "long long" CACHE STRING
    "The main integer type for indices and sizes")
# Enable/disable optional components
option(KOQKATOO_WITH_TESTS
    "Build the tests" ${BUILD_TESTING})
option(KOQKATOO_WITH_BENCHMARKS
    "Build the benchmarks" Off)
# Enable/disable interfaces to other programming languages
option(KOQKATOO_WITH_PYTHON
    "Build the Python bindings" Off)
# Developer options
option(KOQKATOO_EXTENSIVE_TESTS Off
    "Longer tests with more combinations of sizes and parameters")
option(KOQKATOO_WARNINGS_AS_ERRORS
    "Enable -Werror or /WX" Off)
option(KOQKATOO_WITH_COVERAGE
    "Generate coverage information" Off)
set(KOQKATOO_DOXYFILE "Doxyfile" CACHE FILEPATH
    "The Doxyfile to use for the docs target")
option(KOQKATOO_WITH_ACCURATE_BUILD_TIME
    "Update the build time on every build" On)
option(KOQKATOO_DEVELOPER_MODE
    "Enable developer options such as ccache and colored compiler output" Off)

# Installation paths
include(GNUInstallDirs)
set(KOQKATOO_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "Installation directory for archives and libraries")
set(KOQKATOO_INSTALL_CMAKEDIR "${KOQKATOO_INSTALL_LIBDIR}/cmake/koqkatoo"
    CACHE PATH "Installation directory for CMake configuration files")
set(KOQKATOO_INSTALL_BINDIR "${CMAKE_INSTALL_BINDIR}"
    CACHE PATH "Installation directory for binaries and DLLs")
set(KOQKATOO_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}"
    CACHE PATH "Installation directory for headers")
option(KOQKATOO_STANDALONE
    "Install with relative RPATH to locate its own shared libraries" On)

# Development
if (PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS On)
endif()
if (KOQKATOO_DEVELOPER_MODE)
    include(cmake/Develop.cmake)
endif()

# Compiler warnings
include(cmake/Warnings.cmake)
add_warnings_target(warnings ${KOQKATOO_WARNINGS_AS_ERRORS})
add_library(koqkatoo::warnings ALIAS warnings)

# Compiler options
if (MSVC)
    add_compile_options("/bigobj")
endif()
set(CMAKE_RELEASE_POSTFIX "")
set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd")
set(CMAKE_MINSIZEREL_POSTFIX "_rs")

# Coverage
if (KOQKATOO_WITH_COVERAGE)
    add_custom_target(coverage
        ${CMAKE_CURRENT_LIST_DIR}/scripts/coverage.sh
        ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    add_compile_options("--coverage")
    add_link_options("--coverage")
    add_dependencies(coverage koqkatoo::tests)
endif()

# Documentation
find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_custom_target(docs Doxygen::doxygen ${KOQKATOO_DOXYFILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/doxygen)
endif()

# Libraries
add_subdirectory(src)

# Tests
if (KOQKATOO_WITH_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Benchmarks
if (KOQKATOO_WITH_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if (false) # TODO
    # Interfaces to other languages
    add_subdirectory(interfaces)
endif()
