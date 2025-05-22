find_package(benchmark REQUIRED)
find_package(hyhound REQUIRED COMPONENTS OCP)

add_executable(benchmark-ocp-solve ocp-solve.cpp)
target_link_libraries(benchmark-ocp-solve PRIVATE
    benchmark::benchmark
    koqkatoo::ocp hyhound::ocp
    koqkatoo::warnings)

if (KOQKATOO_WITH_BLASFEO)
    find_package(blasfeo REQUIRED)
    target_sources(benchmark-ocp-solve PRIVATE "riccati-blasfeo-factor.cpp")
    target_link_libraries(benchmark-ocp-solve PRIVATE blasfeo)
    target_compile_definitions(benchmark-ocp-solve PRIVATE WITH_BLASFEO)
endif()
target_sources(benchmark-ocp-solve PRIVATE "${PROJECT_SOURCE_DIR}/test/ocp/test-cyclic-schedule.cpp")
target_compile_definitions(benchmark-ocp-solve PRIVATE WITHOUT_TESTS)
