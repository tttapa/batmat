#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/gemm.hpp>
#include <benchmark/benchmark.h>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <random>

using batmat::index_t;
using batmat::real_t;
using batmat::linalg::PackingSelector;
using batmat::linalg::StorageOrder;
namespace flops = batmat::linalg::flops;

template <class Abi, StorageOrder OA, StorageOrder OB,
          PackingSelector PA = PackingSelector::Transpose,
          PackingSelector PB = PackingSelector::Transpose, bool Tiling = true>
constexpr auto gemm = [](benchmark::State &state) {
    using namespace batmat::linalg;
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    matrix<real_t, Abi, OA> A{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OB> B{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi> C{{.depth = d, .rows = n, .cols = n}};
    std::ranges::generate(A, [&] { return nrml(rng); });
    std::ranges::generate(B, [&] { return nrml(rng); });
    std::ranges::generate(C, [&] { return nrml(rng); });
    for (auto _ : state)
        for (index_t l = 0; l < A.num_batches(); ++l)
            if constexpr (decltype(A)::batch_size_type::value == 1) {
                namespace blas = guanaqo::blas;
                if constexpr (OA == StorageOrder::ColMajor && OB == StorageOrder::ColMajor)
                    blas::xgemm_NN<real_t>(1, A(l), B(l), 0, C(l));
                else if constexpr (OA == StorageOrder::ColMajor && OB == StorageOrder::RowMajor)
                    blas::xgemm_NT<real_t>(1, A(l), B(l).transposed(), 0, C(l));
                else if constexpr (OA == StorageOrder::RowMajor && OB == StorageOrder::ColMajor)
                    blas::xgemm_TN<real_t>(1, A(l).transposed(), B(l), 0, C(l));
                else // if constexpr (OA == StorageOrder::RowMajor && OB == StorageOrder::RowMajor)
                    blas::xgemm_TT<real_t>(1, A(l).transposed(), B(l).transposed(), 0, C(l));
            } else
                batmat::linalg::gemm(A.batch(l), B.batch(l), C.batch(l), {!Tiling, PA, PB});
    auto flop_cnt = static_cast<double>(d * total(flops::gemm(A.rows(), B.cols(), A.cols())));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
};

#ifdef BATMAT_WITH_BLASFEO
#include <blasfeo.hpp>

template <StorageOrder OA, StorageOrder OB>
constexpr auto gemm<struct blasfeo, OA, OB> = [](benchmark::State &state) {
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    auto A          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    auto B          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    auto D          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    for (auto _ : state)
        for (index_t l = 0; l < d; ++l) {
            auto pA = A[l].get(), pB = B[l].get(), pC = D[l].get(), pD = D[l].get();
            if constexpr (OA == StorageOrder::ColMajor && OB == StorageOrder::ColMajor)
                blasfeo_dgemm_nn(pA->m, pB->n, pA->n, 1, pA, 0, 0, pB, 0, 0, 0, pC, 0, 0, pD, 0, 0);
            else if constexpr (OA == StorageOrder::ColMajor && OB == StorageOrder::RowMajor)
                blasfeo_dgemm_nt(pA->m, pB->m, pA->n, 1, pA, 0, 0, pB, 0, 0, 0, pC, 0, 0, pD, 0, 0);
            else if constexpr (OA == StorageOrder::RowMajor && OB == StorageOrder::ColMajor)
                blasfeo_dgemm_tn(pA->n, pB->n, pA->m, 1, pA, 0, 0, pB, 0, 0, 0, pC, 0, 0, pD, 0, 0);
            else // if constexpr (OA == StorageOrder::RowMajor && OB == StorageOrder::RowMajor)
                blasfeo_dgemm_tt(pA->n, pB->m, pA->m, 1, pA, 0, 0, pB, 0, 0, 0, pC, 0, 0, pD, 0, 0);
        }
    auto flop_cnt                 = static_cast<double>(d * total(flops::gemm(n, n, n)));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
};
#endif

using enum StorageOrder;
using enum PackingSelector;
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
BENCHMARK(gemm<simd8, RowMajor, ColMajor, Transpose, Always, true>)->BM_RANGES();
BENCHMARK(gemm<simd8, RowMajor, RowMajor, Transpose, Always, true>)->BM_RANGES();
BENCHMARK(gemm<simd8, ColMajor, ColMajor, Transpose, Always, true>)->BM_RANGES();
BENCHMARK(gemm<simd8, ColMajor, RowMajor, Transpose, Always, true>)->BM_RANGES();

BENCHMARK(gemm<simd8, RowMajor, ColMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(gemm<simd8, RowMajor, RowMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(gemm<simd8, ColMajor, ColMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(gemm<simd8, ColMajor, RowMajor, Never, Never, true>)->BM_RANGES();

BENCHMARK(gemm<simd8, RowMajor, ColMajor, Transpose, Always, false>)->BM_RANGES();
BENCHMARK(gemm<simd8, RowMajor, RowMajor, Transpose, Always, false>)->BM_RANGES();
BENCHMARK(gemm<simd8, ColMajor, ColMajor, Transpose, Always, false>)->BM_RANGES();
BENCHMARK(gemm<simd8, ColMajor, RowMajor, Transpose, Always, false>)->BM_RANGES();
#endif
BENCHMARK(gemm<simd4, RowMajor, ColMajor, Transpose, Always, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, RowMajor, RowMajor, Transpose, Always, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, ColMajor, ColMajor, Transpose, Always, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, ColMajor, RowMajor, Transpose, Always, true>)->BM_RANGES();

BENCHMARK(gemm<simd4, RowMajor, ColMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, RowMajor, RowMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, ColMajor, ColMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, ColMajor, RowMajor, Never, Never, true>)->BM_RANGES();

BENCHMARK(gemm<simd4, RowMajor, ColMajor, Transpose, Always, false>)->BM_RANGES();
BENCHMARK(gemm<simd4, RowMajor, RowMajor, Transpose, Always, false>)->BM_RANGES();
BENCHMARK(gemm<simd4, ColMajor, ColMajor, Transpose, Always, false>)->BM_RANGES();
BENCHMARK(gemm<simd4, ColMajor, RowMajor, Transpose, Always, false>)->BM_RANGES();

BENCHMARK(gemm<scalar, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(gemm<scalar, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(gemm<scalar, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(gemm<scalar, ColMajor, RowMajor>)->BM_RANGES();

#ifdef BATMAT_WITH_BLASFEO
BENCHMARK(gemm<blasfeo, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(gemm<blasfeo, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(gemm<blasfeo, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(gemm<blasfeo, ColMajor, RowMajor>)->BM_RANGES();
#endif

#if 0
BENCHMARK(gemm<simd4, RowMajor, ColMajor, Transpose, Transpose, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, RowMajor, RowMajor, Transpose, Transpose, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, ColMajor, ColMajor, Transpose, Transpose, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, ColMajor, RowMajor, Transpose, Transpose, true>)->BM_RANGES();

BENCHMARK(gemm<simd4, RowMajor, ColMajor, Always, Always, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, RowMajor, RowMajor, Always, Always, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, ColMajor, ColMajor, Always, Always, true>)->BM_RANGES();
BENCHMARK(gemm<simd4, ColMajor, RowMajor, Always, Always, true>)->BM_RANGES();
#endif
