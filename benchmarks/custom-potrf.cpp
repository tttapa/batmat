#include <koqkatoo/config.hpp>

#include <koqkatoo/unroll.h>
#include <Eigen/Core>
#include <iostream>

#include <benchmark/benchmark.h>

using real_t  = koqkatoo::real_t;
using index_t = koqkatoo::index_t;

template <index_t NS>
void potrf_trsm_kernel(real_t *L, index_t ldL, index_t rows);

template <index_t NS>
void benchmark_potrf(benchmark::State &state) {
    using EMat      = Eigen::MatrixX<real_t>;
    const index_t n = 8;
    const index_t m = 8 * (NS + 1);
    EMat L          = 1e3 * EMat::Identity(n, n);
    if (n >= 8)
        L.topLeftCorner(8, 8) << 11, 0, 0, 0, 0, 0, 0, 0, //
            21, 22, 0, 0, 0, 0, 0, 0,                     //
            31, 32, 33, 0, 0, 0, 0, 0,                    //
            41, 42, 43, 44, 0, 0, 0, 0,                   //
            51, 52, 53, 54, 55, 0, 0, 0,                  //
            61, 62, 63, 64, 65, 66, 0, 0,                 //
            71, 72, 73, 74, 75, 76, 77, 0,                //
            81, 82, 83, 84, 85, 86, 87, 88;               //
    else if (n >= 4)
        L.topLeftCorner(4, 4) << 11, 0, 0, 0, //
            21, 22, 0, 0,                     //
            31, 32, 33, 0,                    //
            41, 42, 43, 44;                   //
    EMat A = EMat::Zero(n + m, n);
    A.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().rankUpdate(L);
    A.block(n, 0, n, n) = L.transpose();
    // std::cout << A << "\n\n" << L << "\n" << std::endl;
    for (auto _ : state) {
        benchmark::DoNotOptimize(A.data());
        potrf_trsm_kernel<NS>(A.data(), A.outerStride(), A.rows());
        benchmark::ClobberMemory();
    }
    // std::cout << A << "\n" << std::endl;
    const auto flop_cnt = (n + 1) * n * (n - 1) / 6 + n * (n - 1) / 2 + 2 * n +
                          n * (n + 1) * (m - n) / 2;
    state.counters["GFLOP count"] = {1e-9 * static_cast<double>(flop_cnt)};
    state.counters["GFLOPS"]      = {1e-9 * static_cast<double>(flop_cnt),
                                     benchmark::Counter::kIsIterationInvariantRate};
}

BENCHMARK(benchmark_potrf<1>);
BENCHMARK(benchmark_potrf<2>);
BENCHMARK(benchmark_potrf<3>);
BENCHMARK(benchmark_potrf<4>);
