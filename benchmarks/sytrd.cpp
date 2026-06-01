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
using guanaqo::blas::blas_index_t;
namespace flops = batmat::linalg::flops;

// TODO: move to guanaqo::blas
#ifndef LAPACK_dsytrd
#define LAPACK_dsytrd dsytrd
#endif
#ifndef LAPACK_ssytrd
#define LAPACK_ssytrd ssytrd
#endif

template <class T, class I>
void xsytrd(const char *uplo, const I *n, std::type_identity_t<T *> a, const I *lda,
            std::type_identity_t<T *> d, std::type_identity_t<T *> e, std::type_identity_t<T *> tau,
            T *work, const I *lwork, I *info);
template <>
void xsytrd<float, blas_index_t>(const char *uplo, const blas_index_t *n,
                                 std::type_identity_t<float *> a, const blas_index_t *lda,
                                 std::type_identity_t<float *> d, std::type_identity_t<float *> e,
                                 std::type_identity_t<float *> tau, float *work,
                                 const blas_index_t *lwork, blas_index_t *info) {
    LAPACK_ssytrd(uplo, n, a, lda, d, e, tau, work, lwork, info);
}
template <>
void xsytrd<double, blas_index_t>(const char *uplo, const blas_index_t *n,
                                  std::type_identity_t<double *> a, const blas_index_t *lda,
                                  std::type_identity_t<double *> d,
                                  std::type_identity_t<double *> e,
                                  std::type_identity_t<double *> tau, double *work,
                                  const blas_index_t *lwork, blas_index_t *info) {
    LAPACK_dsytrd(uplo, n, a, lda, d, e, tau, work, lwork, info);
}

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
        xsytrd("L", &ni, nullptr, &ni, nullptr, nullptr, nullptr, work.data(), &neg_one, &info);
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
                xsytrd("L", &ni, B.batch(l).data(), &ni, diag.data(), subdiag.data(),
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

#ifdef BATMAT_WITH_EIGEN
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

template <StorageOrder OA>
constexpr auto sytrd<struct eigen, OA> = [](benchmark::State &state) {
    constexpr auto Order = OA == StorageOrder::ColMajor ? Eigen::ColMajor : Eigen::RowMajor;
    using EMat           = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Order>;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    std::vector<EMat> A;
    std::vector<Eigen::Tridiagonalization<EMat>> T;
    for (index_t l = 0; l < d; ++l) {
        auto &Al = A.emplace_back(n, n);
        std::ranges::generate(Al.reshaped(), [&] { return uni(rng); });
        T.emplace_back(n);
    }
    for (auto _ : state)
        for (index_t l = 0; l < d; ++l) {
            T[l].compute(A[l]);
        }
    auto flop_cnt                 = static_cast<double>(d * total(flops::sytrd(n)));
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
BENCHMARK(sytrd<simd8, ColMajor>)->BM_RANGES();
BENCHMARK(sytrd<simd8, RowMajor>)->BM_RANGES();
#endif
BENCHMARK(sytrd<simd4, ColMajor>)->BM_RANGES();
BENCHMARK(sytrd<simd4, RowMajor>)->BM_RANGES();
BENCHMARK(sytrd<scalar, ColMajor>)->BM_RANGES();
#ifdef BATMAT_WITH_EIGEN
BENCHMARK(sytrd<eigen, ColMajor>)->BM_RANGES();
BENCHMARK(sytrd<eigen, RowMajor>)->BM_RANGES();
#endif
