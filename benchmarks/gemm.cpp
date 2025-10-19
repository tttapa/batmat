#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/gemm.hpp>
#include <benchmark/benchmark.h>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <random>

using namespace batmat::linalg;
using batmat::index_t;
using batmat::real_t;

template <class Abi, StorageOrder OA, StorageOrder OB,
          PackingSelector PA = PackingSelector::Transpose,
          PackingSelector PB = PackingSelector::Transpose, bool Tiling = true>
void gemm(benchmark::State &state) {
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
                gemm(A.batch(l), B.batch(l), C.batch(l), {!Tiling, PA, PB});
    auto flop_cnt = static_cast<double>(d * total(flops::gemm(A.rows(), B.cols(), A.cols())));
    state.counters["GFLOP count"] = {1e-9 * flop_cnt};
    state.counters["GFLOPS"] = {1e-9 * flop_cnt, benchmark::Counter::kIsIterationInvariantRate};
    state.counters["depth"]  = {static_cast<double>(d)};
}

using batmat::datapar::deduced_abi;
using scalar_abi = batmat::datapar::scalar_abi<real_t>;

using enum StorageOrder;
using enum PackingSelector;
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

BENCHMARK(gemm<default_abi, RowMajor, ColMajor, Transpose, Always, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, RowMajor, RowMajor, Transpose, Always, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, ColMajor, ColMajor, Transpose, Always, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, ColMajor, RowMajor, Transpose, Always, true>)->BM_RANGES();

BENCHMARK(gemm<default_abi, RowMajor, ColMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, RowMajor, RowMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, ColMajor, ColMajor, Never, Never, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, ColMajor, RowMajor, Never, Never, true>)->BM_RANGES();

BENCHMARK(gemm<default_abi, RowMajor, ColMajor, Transpose, Always, false>)->BM_RANGES();
BENCHMARK(gemm<default_abi, RowMajor, RowMajor, Transpose, Always, false>)->BM_RANGES();
BENCHMARK(gemm<default_abi, ColMajor, ColMajor, Transpose, Always, false>)->BM_RANGES();
BENCHMARK(gemm<default_abi, ColMajor, RowMajor, Transpose, Always, false>)->BM_RANGES();

BENCHMARK(gemm<scalar_abi, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(gemm<scalar_abi, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(gemm<scalar_abi, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(gemm<scalar_abi, ColMajor, RowMajor>)->BM_RANGES();

#if 0
BENCHMARK(gemm<default_abi, RowMajor, ColMajor, Transpose, Transpose, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, RowMajor, RowMajor, Transpose, Transpose, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, ColMajor, ColMajor, Transpose, Transpose, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, ColMajor, RowMajor, Transpose, Transpose, true>)->BM_RANGES();

BENCHMARK(gemm<default_abi, RowMajor, ColMajor, Always, Always, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, RowMajor, RowMajor, Always, Always, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, ColMajor, ColMajor, Always, Always, true>)->BM_RANGES();
BENCHMARK(gemm<default_abi, ColMajor, RowMajor, Always, Always, true>)->BM_RANGES();
#endif
