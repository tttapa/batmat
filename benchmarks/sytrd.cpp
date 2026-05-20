#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/sytrd.hpp>
#include <benchmark/benchmark.h>
#include <guanaqo/blas/blas.hpp>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <random>

using batmat::index_t;
using batmat::real_t;
using batmat::linalg::StorageOrder;
namespace flops = batmat::linalg::flops;

#ifndef LAPACK_dsytrd
#define LAPACK_dsytrd dsytrd
#endif

template <class Abi, StorageOrder OA>
constexpr auto sytrd = [](benchmark::State &state) {
    using namespace batmat::linalg;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    const auto ni   = static_cast<guanaqo::blas::blas_index_t>(n);
    matrix<real_t, Abi, OA> A{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OA> B{{.depth = d, .rows = n, .cols = n}};
    auto [rw, cw] = sytrd_size_W(A.batch(0));
    matrix<real_t, Abi> W{{.depth = d, .rows = rw, .cols = cw}};
    auto [ry, cy] = sytrd_size_Y(A.batch(0));
    matrix<real_t, Abi> Y{{.depth = d, .rows = ry, .cols = cy}};
    std::ranges::generate(A, [&] { return uni(rng); });
    std::ranges::generate(B, [&] { return uni(rng); });
    // Allocate LAPACK workspace
    std::vector<real_t> diag(n), subdiag(n - 1), tau(n - 1);
    std::vector<real_t> work(1);
    if constexpr (decltype(A)::batch_size_type::value == 1) {
        const guanaqo::blas::blas_index_t neg_one = -1;
        guanaqo::blas::blas_index_t info;
        LAPACK_dsytrd("L", &ni, nullptr, &ni, nullptr, nullptr, nullptr, work.data(), &neg_one,
                      &info);
        BATMAT_ASSERT(info == 0);
        work.resize(static_cast<size_t>(work[0]));
        BATMAT_ASSERT(W.size() >= n - 1);
    }
    for (auto _ : state)
        for (index_t l = 0; l < A.num_batches(); ++l)
            if constexpr (decltype(A)::batch_size_type::value == 1) {
                state.PauseTiming();
                copy(A.batch(l), B.batch(l));
                const auto lwork = static_cast<guanaqo::blas::blas_index_t>(work.size());
                guanaqo::blas::blas_index_t info;
                state.ResumeTiming();
                // guanaqo::blas::xsytrd(B(l)); // TODO
                LAPACK_dsytrd("L", &ni, B.batch(l).data(), &ni, diag.data(), subdiag.data(),
                              W.batch(l).data(), work.data(), &lwork, &info);
            } else {
                state.PauseTiming();
                copy(A.batch(l), B.batch(l));
                state.ResumeTiming();
                batmat::linalg::sytrd(tril(B.batch(l)), W.batch(l), Y.batch(l));
            }
    auto flop_cnt                 = static_cast<double>(d * total(flops::sytrd(A.rows())));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
};

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
BENCHMARK(sytrd<simd8, ColMajor>)->BM_RANGES();
BENCHMARK(sytrd<simd8, RowMajor>)->BM_RANGES();
#endif
BENCHMARK(sytrd<simd4, ColMajor>)->BM_RANGES();
BENCHMARK(sytrd<simd4, RowMajor>)->BM_RANGES();
BENCHMARK(sytrd<scalar, ColMajor>)->BM_RANGES();
