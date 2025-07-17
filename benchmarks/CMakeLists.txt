find_package(benchmark REQUIRED)

# TODO: port benchmarks
add_library(benchmark-main "benchmark-main.cpp")
target_link_libraries(benchmark-main PUBLIC batmat::batmat benchmark::benchmark
                      -static-libstdc++ -static-libgcc)
add_executable(benchmark-gemm "gemm.cpp")
target_link_libraries(benchmark-gemm PRIVATE benchmark-main batmat::warnings)
add_executable(benchmark-trtri "trtri.cpp")
target_link_libraries(benchmark-trtri PRIVATE benchmark-main batmat::warnings)
add_executable(benchmark-potrf "potrf.cpp")
target_link_libraries(benchmark-potrf PRIVATE benchmark-main batmat::warnings)

add_subdirectory(ocp-solve)
