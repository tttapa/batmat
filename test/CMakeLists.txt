cmake_minimum_required(VERSION 3.21)
project(koqkatoo-test CXX)
enable_testing()
include(GoogleTest)

# Dependencies
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spectra REQUIRED)
if (NOT TARGET koqkatoo::koqkatoo)
    find_package(koqkatoo REQUIRED COMPONENTS Extra) # TODO
endif()

add_executable(tests
    "ocp/test-reconstruct.cpp"
    "ocp/test-solve.cpp"
    "ocp/test-solve-cyclic.cpp"
    # "ocp/solve-cyclic.cpp"
    "ocp/test-cyclic-util.cpp"
    "ocp/reference-solution.cpp"
    # "linalg-compact/test-trmv.cpp"
    "linalg-compact/test-rotate.cpp"
    "linalg-compact/test-gemm.cpp"
    "linalg-compact/test-trsm.cpp"
    "linalg-compact/test-syrk.cpp"
    "linalg-compact/test-potrf.cpp"
    "linalg-compact/test-shh.cpp"
    "linalg-compact/test-shhud-diag.cpp"
    "linalg-compact/test-pntrf.cpp"
    "linalg-compact/test-update.cpp"
    "linalg-compact/test-trtri.cpp"
    "cholundate/test-shh-downdate.cpp"
    "cholundate/test-shh-updowndate.cpp"
    "ocp/test-cyclic-schedule.cpp"
)
target_compile_definitions(tests PRIVATE
    $<$<BOOL:${KOQKATOO_EXTENSIVE_TESTS}>:KOQKATOO_EXTENSIVE_TESTS>)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(tests PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(tests PRIVATE
    koqkatoo::koqkatoo koqkatoo::linalg-compact koqkatoo::warnings
    koqkatoo::ocp koqkatoo::cholundate
    GTest::gtest_main GTest::gmock
    Eigen3::Eigen
    Spectra::Spectra)
if (MSVC)
    target_compile_definitions(tests PRIVATE __PRETTY_FUNCTION__=__FUNCSIG__)
endif()

option(KOQKATOO_FORCE_TEST_DISCOVERY
    "Query the test executable even when cross-compiling" Off)
if (NOT CMAKE_CROSSCOMPILING OR KOQKATOO_FORCE_TEST_DISCOVERY)
    gtest_discover_tests(tests DISCOVERY_TIMEOUT 60)
endif()
add_executable(koqkatoo::tests ALIAS tests)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.21)
    # Copy DLLs for test executable
    add_custom_command(TARGET tests PRE_LINK COMMAND
        COMMAND ${CMAKE_COMMAND} -E
            $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:tests>>,copy,true>
            $<TARGET_RUNTIME_DLLS:tests> $<TARGET_FILE_DIR:tests>
        COMMAND_EXPAND_LISTS)
endif()
