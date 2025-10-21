#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/trsm.hpp>
#include <benchmark/benchmark.h>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <random>

using namespace batmat::linalg;
using batmat::index_t;
using batmat::real_t;

enum Side { Left, Right };

template <class Abi, Side S, StorageOrder OA, StorageOrder OB = OA>
void trsm(benchmark::State &state) {
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
    A.view().add_to_diagonal(10 * static_cast<real_t>(n));
    B.view().add_to_diagonal(10 * static_cast<real_t>(n));
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
                            guanaqo::blas::xtrsm_LLNN<real_t>(1, A(l), C(l));
                        else
                            guanaqo::blas::xtrsm_LLTN<real_t>(1, A(l).transposed(), C(l));
                    } else {
                        // Bᵀ ← Bᵀ tril(A)ᵀ
                        if constexpr (OA == StorageOrder::ColMajor)
                            guanaqo::blas::xtrsm_RLTN<real_t>(1, A(l), C(l).transposed());
                        else
                            guanaqo::blas::xtrsm_RLNN<real_t>(1, A(l).transposed(),
                                                              C(l).transposed());
                    }
                } else {
                    state.PauseTiming();
                    batmat::linalg::copy(A.batch(l), C.batch(l));
                    state.ResumeTiming();
                    if constexpr (OA == StorageOrder::ColMajor) {
                        // A ← A tril(B)
                        if constexpr (OB == StorageOrder::ColMajor)
                            guanaqo::blas::xtrsm_RLNN<real_t>(1, B(l), C(l));
                        else
                            guanaqo::blas::xtrsm_RLTN<real_t>(1, B(l).transposed(), C(l));
                    } else {
                        // Aᵀ ← tril(B)ᵀ Aᵀ
                        if constexpr (OB == StorageOrder::ColMajor)
                            guanaqo::blas::xtrsm_LLTN<real_t>(1, B(l), C(l).transposed());
                        else
                            guanaqo::blas::xtrsm_LLNN<real_t>(1, B(l).transposed(),
                                                              C(l).transposed());
                    }
                }
            } else {
                if constexpr (S == Side::Left)
                    trsm(tril(A.batch(l)), B.batch(l), C.batch(l));
                else
                    trsm(A.batch(l), tril(B.batch(l)), C.batch(l));
            }
    auto flop_cnt =
        static_cast<double>(d * total(flops::trsm(S == Side::Left ? B.rows() : A.cols(),
                                                  S == Side::Left ? B.cols() : A.rows())));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
}

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
BENCHMARK(trsm<simd8, Left, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<simd8, Left, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trsm<simd8, Left, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<simd8, Left, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trsm<simd8, Right, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<simd8, Right, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trsm<simd8, Right, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<simd8, Right, RowMajor, RowMajor>)->BM_RANGES();
#endif
BENCHMARK(trsm<simd4, Left, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<simd4, Left, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trsm<simd4, Left, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<simd4, Left, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trsm<simd4, Right, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<simd4, Right, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trsm<simd4, Right, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<simd4, Right, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trsm<scalar, Left, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<scalar, Left, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trsm<scalar, Left, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<scalar, Left, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trsm<scalar, Right, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<scalar, Right, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(trsm<scalar, Right, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(trsm<scalar, Right, RowMajor, RowMajor>)->BM_RANGES();
