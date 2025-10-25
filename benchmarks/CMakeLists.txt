find_package(benchmark REQUIRED)
find_package(hyhound REQUIRED)
find_package(guanaqo REQUIRED COMPONENTS BLAS)
find_package(Python3 COMPONENTS Interpreter)
if (BATMAT_WITH_BLASFEO)
    find_package(blasfeo REQUIRED)
endif()

set(BATMAT_BENCHMARK_DEPTH 8 CACHE STRING "Batch depth used in benchmarks")

add_compile_definitions(BATMAT_BENCHMARK_DEPTH=${BATMAT_BENCHMARK_DEPTH})
add_library(benchmark-main "benchmark-main.cpp")
target_link_libraries(benchmark-main PUBLIC batmat::batmat benchmark::benchmark)
add_custom_target(run-benchmarks)

function(setup_benchmark_target name)
    set(target_name benchmark-${name})
    add_executable(${target_name} "${name}.cpp")
    target_link_libraries(${target_name} PRIVATE benchmark-main guanaqo::blas batmat::warnings)
    if (BATMAT_WITH_BLASFEO)
        target_compile_definitions(${target_name} PRIVATE BATMAT_WITH_BLASFEO)
        target_link_libraries(${target_name} PRIVATE blasfeo)
        target_include_directories(${target_name} PRIVATE include)
    endif()
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/results/${target_name}.json
        COMMAND ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            taskset -c 0
            $<TARGET_FILE:${target_name}>
            --benchmark_min_time=0.1s
            --benchmark_min_warmup_time=0.01
            --benchmark_repetitions=3
            --benchmark_enable_random_interleaving
            --benchmark_out=${CMAKE_CURRENT_BINARY_DIR}/results/${target_name}.json
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${target_name}
        USES_TERMINAL)
    add_custom_target(run-${target_name}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/results/${target_name}.json)
    add_dependencies(run-benchmarks run-${target_name})
    if (Python3_EXECUTABLE)
        add_custom_target(viz-${target_name}
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/visualize.py
            ${CMAKE_CURRENT_BINARY_DIR}/results/${target_name}.json
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/results/${target_name}.json
            USES_TERMINAL)
    endif()
endfunction()

setup_benchmark_target(gemm)
setup_benchmark_target(trmm)
setup_benchmark_target(trsm)
setup_benchmark_target(syrk)
setup_benchmark_target(trtri)
setup_benchmark_target(potrf)
setup_benchmark_target(syrk-potrf)
setup_benchmark_target(hyh)

target_link_libraries(benchmark-hyh PRIVATE hyhound::hyhound)
