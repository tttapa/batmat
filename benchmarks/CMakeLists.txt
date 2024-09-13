find_package(benchmark REQUIRED)
find_package(Eigen3)

add_executable(benchmark-gemm gemm.cpp benchmark-main.cpp)
target_link_libraries(benchmark-gemm PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)

add_executable(benchmark-trsm trsm.cpp benchmark-main.cpp)
target_link_libraries(benchmark-trsm PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)

add_executable(benchmark-potrf potrf.cpp benchmark-main.cpp)
target_link_libraries(benchmark-potrf PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)

if (TARGET Eigen3::Eigen)
    add_executable(benchmark-shh shh.cpp)
    target_link_libraries(benchmark-shh PRIVATE
        Eigen3::Eigen benchmark::benchmark koqkatoo::linalg koqkatoo::cholundate
        koqkatoo::warnings)
endif()

set(KOQKATOO_BLASFEO_TARGET "X64_INTEL_HASWELL" CACHE STRING
    "Target architecture for the BLASFEO library")
find_package(blasfeo
    PATHS "${PROJECT_SOURCE_DIR}/third-party/staging/blasfeo-${KOQKATOO_BLASFEO_TARGET}")
if (TARGET blasfeo)
    add_executable(benchmark-gemm-blasfeo blasfeo/blasfeo-gemm.cpp benchmark-main.cpp)
    target_link_libraries(benchmark-gemm-blasfeo PRIVATE
        benchmark::benchmark blasfeo koqkatoo::koqkatoo koqkatoo::warnings)

    add_executable(benchmark-potrf-blasfeo blasfeo/blasfeo-potrf.cpp benchmark-main.cpp)
    target_link_libraries(benchmark-potrf-blasfeo PRIVATE
        benchmark::benchmark blasfeo koqkatoo::koqkatoo koqkatoo::warnings)
endif()
