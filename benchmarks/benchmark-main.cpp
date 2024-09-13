#include <benchmark/benchmark.h>
#include <koqkatoo-version.h>
#include <cstdlib>

#include <koqkatoo/openmp.h>

int main(int argc, char **argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
#if KOQKATOO_WITH_OPENMP
    benchmark::AddCustomContext("OMP_NUM_THREADS",
                                std::to_string(omp_get_max_threads()));
#endif
    benchmark::AddCustomContext("koqkatoo_build_time", koqkatoo_build_time);
    benchmark::AddCustomContext("koqkatoo_commit_hash", koqkatoo_commit_hash);
#if defined(__AVX512F__)
    benchmark::AddCustomContext("arch", "avx512f");
#elif defined(__AVX2__)
    benchmark::AddCustomContext("arch", "avx2");
#elif defined(__AVX__)
    benchmark::AddCustomContext("arch", "avx");
#elif defined(__SSE3__)
    benchmark::AddCustomContext("arch", "sse3");
#endif
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
}
