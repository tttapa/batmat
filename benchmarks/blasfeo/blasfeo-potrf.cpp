#include <koqkatoo/openmp.h>
#include <benchmark/benchmark.h>
#include <blasfeo.h>
#include <blasfeo_d_blasfeo_api.h>
#include <random>
#include <ranges>

using real_t  = double;
using index_t = int;

void dpotrf_blasfeo(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    auto n         = static_cast<index_t>(state.range(0));
    index_t depth  = 64;
    index_t n_elem = depth * n * n;
    std::vector<real_t> A(n_elem);
    std::ranges::generate(A, [&] { return nrml(rng); });

    std::vector<blasfeo_dmat> batch_A(depth), batch_B(depth);
    std::vector<blasfeo_dmat> batch_D(depth); // result
    for (auto &&[i, mat] : std::views::enumerate(batch_A)) {
        blasfeo_allocate_dmat(n, n, &mat);
        blasfeo_pack_dmat(n, n, &A[i * n * n], n, &mat, 0, 0);
    }
    for (auto &&[i, mat] : std::views::enumerate(batch_B)) {
        blasfeo_allocate_dmat(n, n, &mat);
    }
    for (auto &&[i, mat] : std::views::enumerate(batch_D)) {
        blasfeo_allocate_dmat(n, n, &mat);
    }
    for (index_t i = 0; i < depth; ++i)
        blasfeo_dgemm_tn(n, n, n, 1.0, &batch_A[i], 0, 0, &batch_A[i], 0, 0,
                         0.0, &batch_A[i], 0, 0, &batch_B[i], 0, 0);
    auto batch_dpotrf_blasfeo = [&] {
        KOQKATOO_OMP(parallel for)
        for (index_t i = 0; i < depth; ++i)
            blasfeo_dpotrf_l(n, &batch_B[i], 0, 0, &batch_D[i], 0, 0);
    };
    for (auto _ : state) {
        batch_dpotrf_blasfeo();
    }
    auto flop_cnt = 64e-9 * std::pow(static_cast<double>(n), 3) / 6;
    state.counters["GFLOP count"] = {flop_cnt};
    state.counters["GFLOPS"]      = {flop_cnt,
                                     benchmark::Counter::kIsIterationInvariantRate};
}

#define BM_RANGES()                                                            \
    DenseRange(1, 63, 1)                                                       \
        ->DenseRange(64, 255, 4)                                               \
        ->DenseRange(256, 511, 8)                                              \
        ->MeasureProcessCPUTime()
BENCHMARK(dpotrf_blasfeo)->BM_RANGES();
