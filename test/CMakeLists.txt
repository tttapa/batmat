cmake_minimum_required(VERSION 3.21)
project(batmat-test CXX)
enable_testing()
include(GoogleTest)

# Dependencies
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(guanaqo REQUIRED COMPONENTS BLAS)
if (NOT TARGET batmat::batmat)
    find_package(batmat REQUIRED COMPONENTS Extra) # TODO
endif()

add_executable(tests
    "ops/test-rotate.cpp"
    "matrix/test-view.cpp"
    "linalg/test-compress.cpp"
    "linalg/test-copy.cpp"
    "linalg/test-gemm.cpp"
)
target_compile_definitions(tests PRIVATE
    $<$<BOOL:${BATMAT_EXTENSIVE_TESTS}>:BATMAT_EXTENSIVE_TESTS>)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(tests PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(tests PRIVATE
    batmat::batmat batmat::linalg batmat::warnings
    guanaqo::blas
    GTest::gtest_main GTest::gmock
    Eigen3::Eigen)
if (MSVC)
    target_compile_definitions(tests PRIVATE __PRETTY_FUNCTION__=__FUNCSIG__)
endif()

option(BATMAT_FORCE_TEST_DISCOVERY
    "Query the test executable even when cross-compiling" Off)
if (NOT CMAKE_CROSSCOMPILING OR BATMAT_FORCE_TEST_DISCOVERY)
    gtest_discover_tests(tests DISCOVERY_TIMEOUT 60)
endif()
add_executable(batmat::tests ALIAS tests)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.21)
    # Copy DLLs for test executable
    add_custom_command(TARGET tests PRE_LINK COMMAND
        COMMAND ${CMAKE_COMMAND} -E
            $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:tests>>,copy,true>
            $<TARGET_RUNTIME_DLLS:tests> $<TARGET_FILE_DIR:tests>
        COMMAND_EXPAND_LISTS)
endif()
