#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/gemm.hpp>
#include <benchmark/benchmark.h>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <random>

using batmat::index_t;
using batmat::real_t;
using batmat::linalg::StorageOrder;
namespace flops = batmat::linalg::flops;

template <class Abi, StorageOrder OA, StorageOrder OC = OA>
constexpr auto syrk = [](benchmark::State &state) {
    using namespace batmat::linalg;
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
            } else {
                syrk_add(A.batch(l), tril(C.batch(l)), tril(D.batch(l)));
            }
    auto flop_cnt                 = static_cast<double>(d * total(flops::syrk(A.rows(), A.cols())));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
};

#ifdef BATMAT_WITH_BLASFEO
#include <blasfeo.hpp>

template <StorageOrder OA, StorageOrder OC>
constexpr auto syrk<struct blasfeo, OA, OC> = [](benchmark::State &state) {
    static_assert(OC == StorageOrder::ColMajor);
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    auto A          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    auto C          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    auto D          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    for (auto _ : state)
        for (index_t l = 0; l < d; ++l) {
            auto pA = A[l].get(), pC = C[l].get(), pD = D[l].get();
            if constexpr (OA == StorageOrder::ColMajor)
                blasfeo_dsyrk_ln(pA->m, pA->n, 1, pA, 0, 0, pA, 0, 0, 1, pC, 0, 0, pD, 0, 0);
            else
                blasfeo_dsyrk_lt(pA->n, pA->m, 1, pA, 0, 0, pA, 0, 0, 1, pC, 0, 0, pD, 0, 0);
        }
    auto flop_cnt                 = static_cast<double>(d * total(flops::syrk(n, n)));
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
BENCHMARK(syrk<simd8, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(syrk<simd8, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(syrk<simd8, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(syrk<simd8, RowMajor, RowMajor>)->BM_RANGES();
#endif
BENCHMARK(syrk<simd4, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(syrk<simd4, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(syrk<simd4, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(syrk<simd4, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(syrk<scalar, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(syrk<scalar, RowMajor, ColMajor>)->BM_RANGES();
#ifdef BATMAT_WITH_BLASFEO
BENCHMARK(syrk<blasfeo, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(syrk<blasfeo, RowMajor, ColMajor>)->BM_RANGES();
#endif
