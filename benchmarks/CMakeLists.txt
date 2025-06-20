find_package(benchmark REQUIRED)

# TODO: port benchmarks
add_executable(benchmark-gemm "gemm.cpp" "benchmark-main.cpp")
target_link_libraries(benchmark-gemm PRIVATE batmat::linalg batmat::warnings benchmark::benchmark -static-libstdc++ -static-libgcc)
