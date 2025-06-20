find_package(benchmark REQUIRED)

# TODO: port benchmarks
add_executable(benchmarks "benchmark-main.cpp")
target_link_libraries(benchmarks PRIVATE batmat::batmat batmat::warnings benchmark::benchmark)
