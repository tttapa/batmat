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

enum Side { Left, Right };

template <class Abi, Side S, StorageOrder OA, StorageOrder OB = OA>
constexpr auto trmm = [](benchmark::State &state) {
    using namespace batmat::linalg;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    matrix<real_t, Abi, OA> A{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OB> B{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, S == Left ? OB : OA> C{{.depth = d, .rows = n, .cols = n}};
    std::ranges::generate(A, [&] { return uni(rng); });
    std::ranges::generate(B, [&] { return uni(rng); });
    std::ranges::generate(C, [&] { return uni(rng); });
    for (auto _ : state)
        for (index_t l = 0; l < A.num_batches(); ++l)
            if constexpr (decltype(A)::batch_size_type::value == 1) {
                if constexpr (S == Side::Left) {
                    state.PauseTiming();
                    batmat::linalg::copy(B.batch(l), C.batch(l));
                    state.ResumeTiming();
                    if constexpr (OB == StorageOrder::ColMajor) {
                        // B ← tril(A) B
                        if constexpr (OA == StorageOrder::ColMajor)
                            guanaqo::blas::xtrmm_LLNN<real_t>(1, A(l), C(l));
                        else
                            guanaqo::blas::xtrmm_LLTN<real_t>(1, A(l).transposed(), C(l));
                    } else {
                        // Bᵀ ← Bᵀ tril(A)ᵀ
                        if constexpr (OA == StorageOrder::ColMajor)
                            guanaqo::blas::xtrmm_RLTN<real_t>(1, A(l), C(l).transposed());
                        else
                            guanaqo::blas::xtrmm_RLNN<real_t>(1, A(l).transposed(),
                                                              C(l).transposed());
                    }
                } else {
                    state.PauseTiming();
                    batmat::linalg::copy(A.batch(l), C.batch(l));
                    state.ResumeTiming();
                    if constexpr (OA == StorageOrder::ColMajor) {
                        // A ← A tril(B)
                        if constexpr (OB == StorageOrder::ColMajor)
                            guanaqo::blas::xtrmm_RLNN<real_t>(1, B(l), C(l));
                        else
                            guanaqo::blas::xtrmm_RLTN<real_t>(1, B(l).transposed(), C(l));
                    } else {
                        // Aᵀ ← tril(B)ᵀ Aᵀ
                        if constexpr (OB == StorageOrder::ColMajor)
                            guanaqo::blas::xtrmm_LLTN<real_t>(1, B(l), C(l).transposed());
                        else
                            guanaqo::blas::xtrmm_LLNN<real_t>(1, B(l).transposed(),
                                                              C(l).transposed());
                    }
                }
            } else {
                if constexpr (S == Side::Left)
                    batmat::linalg::trmm(tril(A.batch(l)), B.batch(l), C.batch(l));
                else
                    batmat::linalg::trmm(A.batch(l), tril(B.batch(l)), C.batch(l));
            }
    auto flop_cnt = static_cast<double>(
        d * total(flops::trmm(
                A.rows(), B.cols(), A.cols(),
                S == Side::Left ? MatrixStructure::LowerTriangular : MatrixStructure::General,
                S == Side::Left ? MatrixStructure::General : MatrixStructure::LowerTriangular,
                MatrixStructure::General)));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
};

#ifdef BATMAT_WITH_BLASFEO
#include <blasfeo.hpp>

template <Side S, StorageOrder OA, StorageOrder OB>
constexpr auto trmm<struct blasfeo, S, OA, OB> = [](benchmark::State &state) {
    using namespace batmat::linalg;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    auto A          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    auto B          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    auto D          = batmat::blasfeo::dmat::random_batch(d, n, n, rng);
    for (auto _ : state)
        for (index_t l = 0; l < d; ++l) {
            auto pA = A[l].get(), pB = B[l].get(), pD = D[l].get();
            if constexpr (S == Side::Left) {
                if constexpr (OB == StorageOrder::ColMajor) {
                    // B ← tril(A) B
                    if constexpr (OA == StorageOrder::ColMajor)
                        blasfeo_dtrmm_llnn(pA->m, pB->n, 1, pA, 0, 0, pB, 0, 0, pD, 0, 0);
                    else
                        blasfeo_dtrmm_lltn(pA->n, pB->n, 1, pA, 0, 0, pB, 0, 0, pD, 0, 0);
                } else {
                    // Bᵀ ← Bᵀ tril(A)ᵀ
                    if constexpr (OA == StorageOrder::ColMajor)
                        blasfeo_dtrmm_rltn(pA->n, pB->m, 1, pA, 0, 0, pB, 0, 0, pD, 0, 0);
                    else
                        blasfeo_dtrmm_rlnn(pA->m, pB->m, 1, pA, 0, 0, pB, 0, 0, pD, 0, 0);
                }
            } else {
                if constexpr (OA == StorageOrder::ColMajor) {
                    // A ← A tril(B)
                    if constexpr (OB == StorageOrder::ColMajor)
                        blasfeo_dtrmm_rlnn(pB->m, pA->n, 1, pB, 0, 0, pA, 0, 0, pD, 0, 0);
                    else
                        blasfeo_dtrmm_rltn(pB->n, pA->n, 1, pB, 0, 0, pA, 0, 0, pD, 0, 0);
                } else {
                    // Aᵀ ← tril(B)ᵀ Aᵀ
                    if constexpr (OB == StorageOrder::ColMajor)
                        blasfeo_dtrmm_lltn(pB->n, pA->m, 1, pB, 0, 0, pA, 0, 0, pD, 0, 0);
                    else
                        blasfeo_dtrmm_llnn(pB->m, pA->m, 1, pB, 0, 0, pA, 0, 0, pD, 0, 0);
                }
            }
        }
    auto flop_cnt = static_cast<double>(
        d * total(flops::trmm(
                n, n, n, //
                S == Side::Left ? MatrixStructure::LowerTriangular : MatrixStructure::General,
                S == Side::Left ? MatrixStructure::General : MatrixStructure::LowerTriangular,
                MatrixStructure::General)));
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
BENCHMARK(trmm<simd8, Left, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<simd8, Left, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<simd8, Left, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<simd8, Left, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<simd8, Right, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<simd8, Right, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<simd8, Right, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<simd8, Right, RowMajor, RowMajor>)->BM_RANGES();
#endif
BENCHMARK(trmm<simd4, Left, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<simd4, Left, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<simd4, Left, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<simd4, Left, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<simd4, Right, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<simd4, Right, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<simd4, Right, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<simd4, Right, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<scalar, Left, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<scalar, Left, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<scalar, Left, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<scalar, Left, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<scalar, Right, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<scalar, Right, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<scalar, Right, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<scalar, Right, RowMajor, RowMajor>)->BM_RANGES();

#ifdef BATMAT_WITH_BLASFEO
BENCHMARK(trmm<blasfeo, Left, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<blasfeo, Left, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<blasfeo, Left, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<blasfeo, Left, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<blasfeo, Right, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<blasfeo, Right, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trmm<blasfeo, Right, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trmm<blasfeo, Right, RowMajor, RowMajor>)->BM_RANGES();
#endif
