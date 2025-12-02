cmake_minimum_required(VERSION 3.25)
project(batmat_benchmarks_mkl)

find_package(MKL CONFIG REQUIRED)
find_package(benchmark REQUIRED)

add_executable(benchmark-potrf-mkl potrf-mkl.cpp)
target_link_libraries(benchmark-potrf-mkl PRIVATE MKL::MKL benchmark::benchmark)
