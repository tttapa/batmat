#include <koqkatoo/cholundate/householder-downdate.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <guanaqo/eigen/view.hpp>
#include <koqkatoo-version.h>
#include <algorithm>
#include <cstdlib>
#include <format>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using koqkatoo::index_t;
using koqkatoo::real_t;
static constexpr auto use_index_t = guanaqo::with_index_type<index_t>;

#if KOQKATOO_WITH_OPENMP
#include <omp.h>
#include <thread>
#endif

struct ProblemMatrices {
    Eigen::MatrixXd K̃, K, L, A;
};
using cache_t = std::map<std::pair<index_t, index_t>, ProblemMatrices>;
std::mutex cache_mtx;
cache_t cache;

struct CholeskyFixture : benchmark::Fixture {
    index_t m, n;
    Eigen::MatrixXd L̃;
    cache_t::const_iterator matrices;

    static cache_t::const_iterator generate_problem(index_t m, index_t n) {
        std::lock_guard lck{cache_mtx};
        auto [it, inserted] =
            cache.try_emplace(std::pair{m, n}, ProblemMatrices{});
        auto &mat = it->second;
        if (!inserted)
            return it;

#if KOQKATOO_WITH_OPENMP
        int old_num_threads = omp_get_max_threads();
        omp_set_num_threads(std::thread::hardware_concurrency());
#endif

        std::mt19937 rng{12345};
        std::uniform_real_distribution<> dist(0.0, 1.0);
        mat.K̃.resize(n, n), mat.K.resize(n, n), mat.L.resize(n, n);
        mat.A.resize(n, m);
        std::ranges::generate(mat.K.reshaped(), [&] { return dist(rng); });
        std::ranges::generate(mat.A.reshaped(), [&] { return dist(rng); });
        const auto ldK = static_cast<index_t>(mat.K.outerStride()),
                   ldA = static_cast<index_t>(mat.A.outerStride());
        koqkatoo::linalg::xsyrk<real_t, index_t>(
            CblasColMajor, CblasLower, CblasTrans, n, n, 1, mat.K.data(), ldK,
            0, mat.K̃.data(), ldK);
        mat.K = mat.K̃;
        koqkatoo::linalg::xsyrk<real_t, index_t>(
            CblasColMajor, CblasLower, CblasNoTrans, n, m, 1, mat.A.data(), ldA,
            1, mat.K.data(), ldK);
        mat.L          = mat.K;
        const auto ldL = static_cast<index_t>(mat.L.outerStride());
        index_t info   = 0;
        koqkatoo::linalg::xpotrf<real_t, index_t>("L", &n, mat.L.data(), &ldL,
                                                  &info);
        mat.L.triangularView<Eigen::StrictlyUpper>().setZero();
        mat.K̃.triangularView<Eigen::StrictlyUpper>() =
            mat.K̃.triangularView<Eigen::StrictlyLower>().transpose();
        mat.K.triangularView<Eigen::StrictlyUpper>() =
            mat.K.triangularView<Eigen::StrictlyLower>().transpose();

#if KOQKATOO_WITH_OPENMP
        omp_set_num_threads(old_num_threads);
#endif
        using namespace std::chrono_literals;
        // Prevent down-clocking after heavy multi-threaded code
        std::this_thread::sleep_for(1000ms);

        return it;
    }

    void SetUp(benchmark::State &state) final {
        m        = static_cast<index_t>(state.range(0));
        n        = static_cast<index_t>(state.range(1));
        matrices = generate_problem(m, n);
        L̃.resize(n, n);
    }

    void TearDown(benchmark::State &state) final {
        Eigen::MatrixXd E = matrices->second.K̃;
        const auto n      = static_cast<index_t>(L̃.rows()),
                   ldL̃    = static_cast<index_t>(L̃.outerStride()),
                   ldE    = static_cast<index_t>(E.outerStride());
#if KOQKATOO_WITH_OPENMP
        int old_num_threads = omp_get_max_threads();
        omp_set_num_threads(std::thread::hardware_concurrency());
#endif
        koqkatoo::linalg::xsyrk<real_t, index_t>(
            CblasColMajor, CblasLower, CblasNoTrans, n, n, -1, L̃.data(), ldL̃, 1,
            E.data(), ldE);
#if KOQKATOO_WITH_OPENMP
        omp_set_num_threads(old_num_threads);
#endif
        E.triangularView<Eigen::StrictlyUpper>().setZero();
        real_t r          = E.lpNorm<Eigen::Infinity>();
        const char *color = r < 1e-9 ? "" : "\x1b[0;31m";
        const char *reset = r < 1e-9 ? "" : "\x1b[0m";
        state.SetLabel(std::format("{}resid={:7e}{}", color, r, reset));
    }

    template <auto Func>
    void customRun(benchmark::State &state) {
        Eigen::MatrixXd Ã(m, n);
        for (auto _ : state) {
            state.PauseTiming();
            Ã = matrices->second.A;
            L̃ = matrices->second.L;
            state.ResumeTiming();
            benchmark::DoNotOptimize(Ã.data());
            benchmark::DoNotOptimize(L̃.data());
            Func(as_view(L̃, use_index_t), as_view(Ã, use_index_t));
            benchmark::ClobberMemory();
        }
        state.SetComplexityN(m); // (m * n * n + n * n + m * n)
    }
};

template <index_t BsL, index_t BsA, auto...>
struct BlockedFixture : CholeskyFixture {};

BENCHMARK_DEFINE_F(CholeskyFixture, blas)(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();
        L̃ = matrices->second.K.triangularView<Eigen::Lower>();
        state.ResumeTiming();
        benchmark::DoNotOptimize(matrices->second.A.data());
        benchmark::DoNotOptimize(L̃.data());
        const auto ldA = static_cast<index_t>(matrices->second.A.outerStride());
        const auto ldL = static_cast<index_t>(L̃.outerStride());
        koqkatoo::linalg::xsyrk<real_t, index_t>(
            CblasColMajor, CblasLower, CblasNoTrans, n, m, -1,
            matrices->second.A.data(), ldA, 1, L̃.data(), ldL);
        index_t info = 0;
        koqkatoo::linalg::xpotrf<real_t, index_t>("L", &n, L̃.data(), &ldL,
                                                  &info);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(m); // (n * n * n / 6 - n / 6 + m * n * n / 2)
}

#define BM_BLK_IMPL_NAME(name, BsL, BsA, ...) name##_L##BsL##_A##BsA
#define BM_BLK_NAME(name, BsL, BsA, ...) #name "<" #BsL ", " #BsA ">"
#define BENCHMARK_BLOCKED(name, func, ...)                                     \
    BENCHMARK_TEMPLATE_DEFINE_F(                                               \
        BlockedFixture, BM_BLK_IMPL_NAME(name, __VA_ARGS__, 0), __VA_ARGS__)   \
    (benchmark::State & state) {                                               \
        this->customRun<func<{__VA_ARGS__}>>(state);                           \
    }                                                                          \
    BENCHMARK_REGISTER_F(BlockedFixture,                                       \
                         BM_BLK_IMPL_NAME(name, __VA_ARGS__, 0))               \
        ->Name(BM_BLK_NAME(name, __VA_ARGS__, 0))

#if 0
#define N 512
auto m_range = [] {
    std::vector<int64_t> v;
    for (int64_t i = 1; i < 16; ++i)
        v.push_back(i);
    for (int64_t i = 16; i <= N; i *= 2)
        v.push_back(i);
    return v;
}();
std::vector<int64_t> n_range{N};
#else
auto n_range = [] {
    std::vector<int64_t> v;
    for (int64_t i = 4; i <= 8192; i *= 2)
        v.push_back(i);
    return v;
}();
std::vector<int64_t> m_range{128};
#endif

#define RNGS() ArgNames({"m", "n"})->ArgsProduct({m_range, n_range})

// clang-format off
BENCHMARK_REGISTER_F(CholeskyFixture, blas)->Name("blas")->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 1, 32)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 4, 4)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 4, 8)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 4, 12)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 4, 16)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 4, 24)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 4, 32)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 8, 8)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 8, 12)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 8, 16)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 8, 24)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 8, 32)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 12, 4)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 12, 8)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 12, 12)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 16, 8)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 16, 12)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 16, 16)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 16, 24)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 16, 32)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 32, 8)->RNGS();
BENCHMARK_BLOCKED(shh, koqkatoo::cholundate::householder::downdate_blocked, 32, 32)->RNGS();
#if KOQKATOO_WITH_LIBFORK
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 1, 32)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 4, 4)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 4, 8)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 4, 12)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 4, 16)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 4, 24)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 4, 32)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 8, 8)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 8, 16)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 8, 24)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 8, 32)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 12, 12)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 16, 16)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 16, 32)->RNGS()->MeasureProcessCPUTime();
BENCHMARK_BLOCKED(shh_fork, koqkatoo::cholundate::householder::parallel::downdate_blocked, 32, 32)->RNGS()->MeasureProcessCPUTime();
#endif
// clang-format on
