#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/potrf.hpp>
#include <benchmark/benchmark.h>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <random>

using namespace batmat::linalg;
using batmat::index_t;
using batmat::real_t;

template <class Abi, StorageOrder OA, StorageOrder OC = OA>
void syrk_potrf(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    matrix<real_t, Abi, OA> A{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OC> C{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OC> D{{.depth = d, .rows = n, .cols = n}};
    std::ranges::generate(A, [&] { return uni(rng); });
    std::ranges::generate(C, [&] { return uni(rng); });
    std::ranges::generate(D, [&] { return uni(rng); });
    C.view().add_to_diagonal(10 * static_cast<real_t>(n));
    for (auto _ : state)
        for (index_t l = 0; l < C.num_batches(); ++l)
            if constexpr (decltype(C)::batch_size_type::value == 1) {
                state.PauseTiming();
                copy(tril(C.batch(l)), tril(D.batch(l)));
                state.ResumeTiming();
                if constexpr (OA == StorageOrder::ColMajor)
                    guanaqo::blas::xsyrk_LN<real_t>(1, A(l), 1, D(l));
                else
                    guanaqo::blas::xsyrk_LT<real_t>(1, A(l).transposed(), 1, D(l));
                guanaqo::blas::xpotrf_L(D(l));
            } else {
                syrk_add_potrf(A.batch(l), tril(C.batch(l)), tril(D.batch(l)));
            }
    const auto nd = static_cast<double>(n), dd = static_cast<double>(d);
    auto flop_cnt                 = 2 * dd * std::pow(nd, 3) / 3;
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {dd};
}

using batmat::datapar::deduced_abi;
using scalar_abi = batmat::datapar::scalar_abi<real_t>;

using enum StorageOrder;
#define BM_RANGES()                                                                                \
    DenseRange(1, 63, 1)                                                                           \
        ->DenseRange(64, 127, 4)                                                                   \
        ->DenseRange(128, 255, 16)                                                                 \
        ->DenseRange(256, 511, 32)                                                                 \
        ->DenseRange(512, 1024, 128)                                                               \
        ->MeasureProcessCPUTime()                                                                  \
        ->UseRealTime()
#ifdef __AVX512F__
using default_abi = deduced_abi<real_t, 8>;
#else
using default_abi = deduced_abi<real_t, 4>;
#endif

BENCHMARK(syrk_potrf<default_abi, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(syrk_potrf<default_abi, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(syrk_potrf<default_abi, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(syrk_potrf<default_abi, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(syrk_potrf<scalar_abi, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(syrk_potrf<scalar_abi, RowMajor, ColMajor>)->BM_RANGES();
