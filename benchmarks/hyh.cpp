#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/hyhound.hpp>
#include <benchmark/benchmark.h>
#include <hyhound/householder-updowndate.hpp>
#include <hyhound/updown.hpp>
#include <random>

using namespace batmat::linalg;
using batmat::index_t;
using batmat::real_t;

template <class Abi, StorageOrder OL, StorageOrder OA>
void hyh(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution brnl{0.75};

    const index_t d = BATMAT_BENCHMARK_DEPTH;
    const auto n    = static_cast<index_t>(state.range(0));
    const auto m    = n - 3 * n / 4;
    matrix<real_t, Abi> S{{.depth = d, .rows = m, .cols = 1}};
    matrix<real_t, Abi, OL> L{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OA> A{{.depth = d, .rows = n, .cols = m}};
    matrix<real_t, Abi, OL> L̃{{.depth = d, .rows = n, .cols = n}};
    matrix<real_t, Abi, OA> Ã{{.depth = d, .rows = n, .cols = m}};
    std::ranges::generate(L, [&] { return nrml(rng); });
    std::ranges::generate(A, [&] { return nrml(rng); });
    std::ranges::generate(S, [&] { return brnl(rng) ? +real_t{0} : -real_t{0}; });
    L.view().add_to_diagonal(10 * static_cast<real_t>(n));
    for (auto _ : state)
        for (index_t l = 0; l < L.num_batches(); ++l) {
            state.PauseTiming();
            batmat::linalg::copy(L.batch(l), L̃.batch(l));
            batmat::linalg::copy(A.batch(l), Ã.batch(l));
            state.ResumeTiming();
            if constexpr (decltype(L)::batch_size_type::value == 1) {
                hyhound::UpDowndate ud{std::span{S(l).data, static_cast<size_t>(m)}};
                hyhound::update_cholesky(L̃(l), Ã(l), ud);
            } else {
                hyhound_sign(tril(L̃.batch(l)), Ã.batch(l), S.batch(l));
            }
        }
    auto flop_cnt = static_cast<double>(d * total(flops::hyh(L.rows(), L.cols(), A.cols())));
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

BENCHMARK(hyh<default_abi, RowMajor, ColMajor>)->BM_RANGES();
BENCHMARK(hyh<default_abi, RowMajor, RowMajor>)->BM_RANGES();
BENCHMARK(hyh<default_abi, ColMajor, ColMajor>)->BM_RANGES();
BENCHMARK(hyh<default_abi, ColMajor, RowMajor>)->BM_RANGES();
BENCHMARK(hyh<scalar_abi, ColMajor, ColMajor>)->BM_RANGES();
