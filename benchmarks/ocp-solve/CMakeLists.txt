find_package(benchmark REQUIRED)
find_package(hyhound REQUIRED COMPONENTS OCP)

add_executable(benchmark-ocp-solve ocp-solve.cpp)
target_link_libraries(benchmark-ocp-solve PRIVATE
    benchmark::benchmark
    koqkatoo::ocp hyhound::ocp
    koqkatoo::warnings)
