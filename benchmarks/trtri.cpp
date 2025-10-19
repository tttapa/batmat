#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/trtri.hpp>
#include <benchmark/benchmark.h>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <guanaqo/mat-view.hpp>
#include <random>

using namespace batmat::linalg;
using batmat::index_t;
using batmat::real_t;

template <class Abi, StorageOrder OA = StorageOrder::ColMajor, StorageOrder OB = OA>
void trtri(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    matrix<real_t, Abi, OA> A{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OB> D{{.depth = d, .rows = n, .cols = n}};
    std::ranges::generate(A, [&] { return uni(rng); });
    std::ranges::generate(D, [&] { return uni(rng); });
    A.view().add_to_diagonal(10 * static_cast<real_t>(n));
    for (auto _ : state)
        for (index_t l = 0; l < A.num_batches(); ++l)
            if constexpr (decltype(A)::batch_size_type::value == 1) {
                state.PauseTiming();
                copy(tril(A.batch(l)), tril(D.batch(l)));
                state.ResumeTiming();
                guanaqo::blas::xtrtri_LN(D(l));
            } else {
                trtri(tril(A.batch(l)), tril(D.batch(l)));
            }
    auto flop_cnt                 = static_cast<double>(d * total(flops::trtri(A.rows())));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
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

BENCHMARK(trtri<default_abi, RowMajor>)->BM_RANGES();
BENCHMARK(trtri<default_abi, ColMajor>)->BM_RANGES();
BENCHMARK(trtri<scalar_abi, ColMajor>)->BM_RANGES();
