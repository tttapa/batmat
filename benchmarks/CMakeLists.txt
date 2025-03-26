find_package(benchmark REQUIRED)
find_package(Eigen3)

add_compile_definitions(KQT_BENCHMARK_DEPTH=64)

add_executable(benchmark-gemm gemm.cpp benchmark-main.cpp)
target_link_libraries(benchmark-gemm PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)

add_executable(benchmark-trsm trsm.cpp benchmark-main.cpp)
target_link_libraries(benchmark-trsm PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)

add_executable(benchmark-syrk syrk.cpp benchmark-main.cpp)
target_link_libraries(benchmark-syrk PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)

add_executable(benchmark-potrf potrf.cpp benchmark-main.cpp)
target_link_libraries(benchmark-potrf PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)

add_executable(benchmark-trtri trtri.cpp benchmark-main.cpp)
target_link_libraries(benchmark-trtri PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)

if (TARGET Eigen3::Eigen)
    add_executable(benchmark-shh shh.cpp)
    target_link_libraries(benchmark-shh PRIVATE
        Eigen3::Eigen benchmark::benchmark koqkatoo::linalg koqkatoo::cholundate
        koqkatoo::warnings)
endif()

option(KOQKATOO_WITH_BLASFEO
    "Enable the benchmarks that depend on BLASFEO" Off)
if (KOQKATOO_WITH_BLASFEO)
    find_package(blasfeo REQUIRED)
    add_executable(benchmark-gemm-blasfeo blasfeo/blasfeo-gemm.cpp benchmark-main.cpp)
    target_link_libraries(benchmark-gemm-blasfeo PRIVATE
        benchmark::benchmark blasfeo koqkatoo::koqkatoo koqkatoo::warnings)

    add_executable(benchmark-potrf-blasfeo blasfeo/blasfeo-potrf.cpp benchmark-main.cpp)
    target_link_libraries(benchmark-potrf-blasfeo PRIVATE
        benchmark::benchmark blasfeo koqkatoo::koqkatoo koqkatoo::warnings)

    add_executable(benchmark-trsm-blasfeo blasfeo/blasfeo-trsm.cpp benchmark-main.cpp)
    target_link_libraries(benchmark-trsm-blasfeo PRIVATE
        benchmark::benchmark blasfeo koqkatoo::koqkatoo koqkatoo::warnings)

    add_executable(benchmark-syrk-blasfeo blasfeo/blasfeo-syrk.cpp benchmark-main.cpp)
    target_link_libraries(benchmark-syrk-blasfeo PRIVATE
        benchmark::benchmark blasfeo koqkatoo::koqkatoo koqkatoo::warnings)
endif()

add_executable(benchmark-solve solve.cpp benchmark-main.cpp)
target_link_libraries(benchmark-solve PRIVATE
    benchmark::benchmark koqkatoo::ocp koqkatoo::warnings Eigen3::Eigen)

add_executable(benchmark-custom-potrf custom-potrf.cpp custom-potrf-microkernel.cpp benchmark-main.cpp)
target_link_libraries(benchmark-custom-potrf PRIVATE
    benchmark::benchmark Eigen3::Eigen koqkatoo::linalg-compact koqkatoo::warnings)

add_subdirectory(ocp-solve)
