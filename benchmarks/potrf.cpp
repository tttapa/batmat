#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/potrf.hpp>
#include <benchmark/benchmark.h>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <random>

using batmat::index_t;
using batmat::real_t;
using batmat::linalg::StorageOrder;
namespace flops = batmat::linalg::flops;

template <class Abi, StorageOrder OA>
constexpr auto potrf = [](benchmark::State &state) {
    using namespace batmat::linalg;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    matrix<real_t, Abi, OA> A{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OA> B{{.depth = d, .rows = n, .cols = n}};
    std::ranges::generate(A, [&] { return uni(rng); });
    std::ranges::generate(B, [&] { return uni(rng); });
    A.view().add_to_diagonal(10 * static_cast<real_t>(n));
    for (auto _ : state)
        for (index_t l = 0; l < A.num_batches(); ++l)
            if constexpr (decltype(A)::batch_size_type::value == 1) {
                state.PauseTiming();
                copy(tril(A.batch(l)), tril(B.batch(l)));
                state.ResumeTiming();
                guanaqo::blas::xpotrf_L(B(l));
            } else {
                batmat::linalg::potrf(tril(A.batch(l)), tril(B.batch(l)));
            }
    auto flop_cnt = static_cast<double>(d * total(flops::potrf(A.rows(), A.cols())));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
};

#ifdef BATMAT_WITH_BLASFEO
#include <blasfeo.hpp>

template <StorageOrder OA>
constexpr auto potrf<struct blasfeo, OA> = [](benchmark::State &state) {
    static_assert(OA == StorageOrder::ColMajor);
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    auto A          = batmat::blasfeo::dmat::random_batch_pos_def(d, n, n, rng);
    auto B          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    for (auto _ : state)
        for (index_t l = 0; l < d; ++l) {
            auto pA = A[l].get(), pB = B[l].get();
            blasfeo_dpotrf_l(pA->m, pA, 0, 0, pB, 0, 0);
        }
    auto flop_cnt                 = static_cast<double>(d * total(flops::potrf(n, n)));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
};
#endif

using enum StorageOrder;
#define BM_RANGES()                                                                                \
    DenseRange(1, 127, 1)                                                                          \
        ->DenseRange(128, 255, 16)                                                                 \
        ->DenseRange(256, 511, 32)                                                                 \
        ->DenseRange(512, 1024, 128)                                                               \
        ->MeasureProcessCPUTime()                                                                  \
        ->UseRealTime()

using scalar = batmat::datapar::scalar_abi<real_t>;
using simd8  = batmat::datapar::deduced_abi<real_t, 8>;
using simd4  = batmat::datapar::deduced_abi<real_t, 4>;

#ifdef __AVX512F__
BENCHMARK(potrf<simd8, ColMajor>)->BM_RANGES();
BENCHMARK(potrf<simd8, RowMajor>)->BM_RANGES();
#endif
BENCHMARK(potrf<simd4, ColMajor>)->BM_RANGES();
BENCHMARK(potrf<simd4, RowMajor>)->BM_RANGES();
BENCHMARK(potrf<scalar, ColMajor>)->BM_RANGES();
#ifdef BATMAT_WITH_BLASFEO
BENCHMARK(potrf<blasfeo, ColMajor>)->BM_RANGES();
#endif
