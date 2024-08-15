find_package(benchmark REQUIRED)

add_executable(benchmark-gemm gemm.cpp)
target_link_libraries(benchmark-gemm PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)

add_executable(benchmark-trsm trsm.cpp)
target_link_libraries(benchmark-trsm PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)

    add_executable(benchmark-potrf potrf.cpp)
target_link_libraries(benchmark-potrf PRIVATE
    benchmark::benchmark koqkatoo::linalg-compact koqkatoo::warnings)
