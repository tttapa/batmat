#include <koqkatoo/openmp.h>
#include <benchmark/benchmark.h>
#include <blasfeo.h>
#include <random>
#include <ranges>

using real_t  = double;
using index_t = int;

void dgemm_blasfeo(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    auto n         = static_cast<index_t>(state.range(0));
    index_t depth  = 64;
    index_t n_elem = depth * n * n;
    std::vector<real_t> A(n_elem), B(n_elem), C(n_elem);
    std::ranges::generate(A, [&] { return nrml(rng); });
    std::ranges::generate(B, [&] { return nrml(rng); });
    std::ranges::generate(C, [&] { return nrml(rng); });

    std::vector<blasfeo_dmat> batch_A(depth), batch_B(depth), batch_C(depth);
    std::vector<blasfeo_dmat> batch_D(depth); // result
    for (auto &&[i, mat] : std::views::enumerate(batch_A)) {
        blasfeo_allocate_dmat(n, n, &mat);
        blasfeo_pack_dmat(n, n, &A[i * n * n], n, &mat, 0, 0);
    }
    for (auto &&[i, mat] : std::views::enumerate(batch_B)) {
        blasfeo_allocate_dmat(n, n, &mat);
        blasfeo_pack_dmat(n, n, &B[i * n * n], n, &mat, 0, 0);
    }
    for (auto &&[i, mat] : std::views::enumerate(batch_C)) {
        blasfeo_allocate_dmat(n, n, &mat);
        blasfeo_pack_dmat(n, n, &C[i * n * n], n, &mat, 0, 0);
    }
    for (auto &&[i, mat] : std::views::enumerate(batch_D)) {
        blasfeo_allocate_dmat(n, n, &mat);
    }
    auto batch_dgemm_blasfeo = [&] {
        KOQKATOO_OMP(parallel for)
        for (index_t i = 0; i < depth; ++i)
            blasfeo_dgemm_nn(n, n, n, 1.0, &batch_A[i], 0, 0, &batch_B[i], 0, 0,
                             1.0, &batch_C[i], 0, 0, &batch_D[i], 0, 0);
    };
    for (auto _ : state) {
        batch_dgemm_blasfeo();
    }
    auto flop_cnt                 = 64e-9 * std::pow(static_cast<double>(n), 3);
    state.counters["GFLOP count"] = {flop_cnt};
    state.counters["GFLOPS"]      = {flop_cnt,
                                     benchmark::Counter::kIsIterationInvariantRate};
}

#define BM_RANGES() DenseRange(1, 512)->MeasureProcessCPUTime()
BENCHMARK(dgemm_blasfeo)->BM_RANGES();
