#include <koqkatoo/openmp.h>
#include <benchmark/benchmark.h>
#include <blasfeo.h>
#include <blasfeo_d_blasfeo_api.h>
#include <random>
#include <ranges>

using real_t  = double;
using index_t = int;

struct raii_blasfeo_dmat {
    blasfeo_dmat mat{};
    raii_blasfeo_dmat()                                     = default;
    raii_blasfeo_dmat(const raii_blasfeo_dmat &)            = delete;
    raii_blasfeo_dmat &operator=(const raii_blasfeo_dmat &) = delete;
    raii_blasfeo_dmat(raii_blasfeo_dmat &&o) noexcept
        : mat{std::exchange(o.mat, {})} {}
    raii_blasfeo_dmat &operator=(raii_blasfeo_dmat &&o) noexcept {
        using std::swap;
        swap(o.mat, mat);
        return *this;
    }
    ~raii_blasfeo_dmat() {
        if (mat.mem)
            blasfeo_free_dmat(&mat);
    }
    [[nodiscard]] blasfeo_dmat *get() { return &mat; }
    [[nodiscard]] const blasfeo_dmat *get() const { return &mat; }
};

void dpotrf_blasfeo(benchmark::State &state) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    auto n         = static_cast<index_t>(state.range(0));
    index_t depth  = 64;
    index_t n_elem = depth * n * n;
    std::vector<real_t> A(n_elem);
    std::ranges::generate(A, [&] { return nrml(rng); });

    std::vector<raii_blasfeo_dmat> batch_A(depth), batch_B(depth);
    for (auto &&[i, mat] : std::views::enumerate(batch_A)) {
        blasfeo_allocate_dmat(n, n, mat.get());
        blasfeo_pack_dmat(n, n, &A[i * n * n], n, mat.get(), 0, 0);
    }
    for (auto &&[i, mat] : std::views::enumerate(batch_B)) {
        blasfeo_allocate_dmat(n, n, mat.get());
    }
    for (index_t i = 0; i < depth; ++i)
        blasfeo_dgemm_tn(n, n, n, 1.0, batch_A[i].get(), 0, 0, batch_A[i].get(),
                         0, 0, 0.0, batch_A[i].get(), 0, 0, batch_B[i].get(), 0,
                         0);
    auto batch_dpotrf_blasfeo = [&] {
#if KOQKATOO_WITH_OPENMP
        if (omp_get_max_threads() == 1) {
            for (index_t i = 0; i < depth; ++i)
                blasfeo_dpotrf_l(n, batch_A[i].get(), 0, 0, batch_A[i].get(), 0,
                                 0);
            return;
        }
#endif
        KOQKATOO_OMP(parallel for)
        for (index_t i = 0; i < depth; ++i)
            blasfeo_dpotrf_l(n, batch_A[i].get(), 0, 0, batch_A[i].get(), 0, 0);
    };
    for (auto _ : state) {
        state.PauseTiming();
        for (index_t i = 0; i < depth; ++i)
            blasfeo_dgecp(n, n, batch_B[i].get(), 0, 0, batch_A[i].get(), 0, 0);
        state.ResumeTiming();
        batch_dpotrf_blasfeo();
    }
    auto flop_cnt = 64e-9 * std::pow(static_cast<double>(n), 3) / 6;
    state.counters["GFLOP count"] = {flop_cnt};
    state.counters["GFLOPS"]      = {flop_cnt,
                                     benchmark::Counter::kIsIterationInvariantRate};
}

#define BM_RANGES()                                                            \
    DenseRange(1, 63, 1)                                                       \
        ->DenseRange(64, 127, 4)                                               \
        ->DenseRange(128, 255, 16)                                             \
        ->DenseRange(256, 512, 32)                                             \
        ->MeasureProcessCPUTime()                                              \
        ->UseRealTime()
BENCHMARK(dpotrf_blasfeo)->BM_RANGES();
