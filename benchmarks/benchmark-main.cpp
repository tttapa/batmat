#include <benchmark/benchmark.h>
#include <batmat-version.h>
#include <string>

#include <batmat/openmp.h>

int main(int argc, char **argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
#if BATMAT_WITH_OPENMP
    benchmark::AddCustomContext("OMP_NUM_THREADS", std::to_string(omp_get_max_threads()));
#endif
    benchmark::AddCustomContext("batmat_build_time", batmat_build_time);
    benchmark::AddCustomContext("batmat_commit_hash", batmat_commit_hash);
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
