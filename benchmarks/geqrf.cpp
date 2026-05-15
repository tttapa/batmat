#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/geqrf.hpp>
#include <benchmark/benchmark.h>
#include <guanaqo/blas/blas.hpp>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <random>

using batmat::index_t;
using batmat::real_t;
using batmat::linalg::StorageOrder;
namespace flops = batmat::linalg::flops;

#ifndef LAPACK_dgeqrf
#define LAPACK_dgeqrf dgeqrf
#endif

template <class Abi, StorageOrder OA>
constexpr auto geqrf = [](benchmark::State &state) {
    using namespace batmat::linalg;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    const auto ni   = static_cast<guanaqo::blas::blas_index_t>(n);
    matrix<real_t, Abi, OA> A{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OA> B{{.depth = d, .rows = n, .cols = n}};
    auto [rw, cw] = geqrf_size_W(A.batch(0));
    matrix<real_t, Abi> W{{.depth = d, .rows = rw, .cols = cw}};
    std::ranges::generate(A, [&] { return uni(rng); });
    std::ranges::generate(B, [&] { return uni(rng); });
    // Allocate LAPACK workspace
    std::vector<real_t> work(1);
    if constexpr (decltype(A)::batch_size_type::value == 1) {
        const guanaqo::blas::blas_index_t neg_one = -1;
        guanaqo::blas::blas_index_t info;
        LAPACK_dgeqrf(&ni, &ni, nullptr, &ni, nullptr, work.data(), &neg_one, &info);
        BATMAT_ASSERT(info == 0);
        work.resize(static_cast<size_t>(work[0]));
        BATMAT_ASSERT(W.size() >= n);
    }
    for (auto _ : state)
        for (index_t l = 0; l < A.num_batches(); ++l)
            if constexpr (decltype(A)::batch_size_type::value == 1) {
                state.PauseTiming();
                copy(A.batch(l), B.batch(l));
                const auto lwork = static_cast<guanaqo::blas::blas_index_t>(work.size());
                guanaqo::blas::blas_index_t info;
                state.ResumeTiming();
                // guanaqo::blas::xgeqrf(B(l)); // TODO
                LAPACK_dgeqrf(&ni, &ni, B.batch(l).data(), &ni, W.batch(l).data(), work.data(),
                              &lwork, &info);
            } else {
                batmat::linalg::geqrf(A.batch(l), B.batch(l), W.batch(l));
            }
    auto flop_cnt = static_cast<double>(d * total(flops::geqrf(A.rows(), A.cols())));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
};

#ifdef BATMAT_WITH_BLASFEO
#include <blasfeo.hpp>

template <StorageOrder OA>
constexpr auto geqrf<struct blasfeo, OA> = [](benchmark::State &state) {
    static_assert(OA == StorageOrder::ColMajor);
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    const auto ni   = static_cast<int>(n);
    auto A          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    auto B          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    struct Del64 {
        void operator()(void *p) const noexcept { ::operator delete[](p, std::align_val_t{64}); }
    };
    const auto ws = blasfeo_dgeqrf_worksize(ni, ni);
    std::unique_ptr<void, Del64> w{new (std::align_val_t{64}) std::byte[ws]};
    for (auto _ : state)
        for (index_t l = 0; l < d; ++l) {
            auto pA = A[l].get(), pB = B[l].get();
            blasfeo_dgeqrf(pA->m, pA->n, pA, 0, 0, pB, 0, 0, w.get());
        }
    auto flop_cnt                 = static_cast<double>(d * total(flops::geqrf(n, n)));
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
BENCHMARK(geqrf<simd8, ColMajor>)->BM_RANGES();
BENCHMARK(geqrf<simd8, RowMajor>)->BM_RANGES();
#endif
BENCHMARK(geqrf<simd4, ColMajor>)->BM_RANGES();
BENCHMARK(geqrf<simd4, RowMajor>)->BM_RANGES();
BENCHMARK(geqrf<scalar, ColMajor>)->BM_RANGES();
#ifdef BATMAT_WITH_BLASFEO
BENCHMARK(geqrf<blasfeo, ColMajor>)->BM_RANGES();
#endif
