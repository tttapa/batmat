#include <koqkatoo/assume.hpp>
#include <koqkatoo/ocp/cyclic-solver/cyclic-solver.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/openmp.h>
#include <benchmark/benchmark.h>
#include <guanaqo/eigen/span.hpp>
#include <guanaqo/openmp.h>
#include <hyhound-version.h>
#include <koqkatoo-version.h>

#include <hyhound/ocp/riccati.hpp>
#include <hyhound/ocp/schur.hpp>

#include <experimental/simd>
#include <utility>

using namespace hyhound;
using namespace hyhound::ocp;
namespace stdx = std::experimental;
using guanaqo::as_span;

auto generate_ocp(benchmark::State &state) {
    using std::exp2;
    std::mt19937 rng{321};
    std::normal_distribution<real_t> nrml{0, 10};

    OCPDataRiccati ocp{.N  = static_cast<index_t>(state.range(0)),
                       .nx = static_cast<index_t>(state.range(1)),
                       .nu = static_cast<index_t>(state.range(2)),
                       .ny = static_cast<index_t>(state.range(3))};
    ocp.init_random(123);

    mat Σ = mat::Zero(ocp.ny, ocp.N + 1);
    std::ranges::generate(Σ.reshaped(), [&] { return exp2(nrml(rng)); });
    return std::pair{std::move(ocp), std::move(Σ)};
}

auto build_cyclic_ocp_solver(const OCPDataRiccati &ocp_ric) {
    using abi             = stdx::simd_abi::native<real_t>;
    using CyclicOCPSolver = koqkatoo::ocp::CyclicOCPSolver<abi>;
    koqkatoo::ocp::LinearOCPStorage ocp{.dim{.N_horiz = ocp_ric.N,
                                             .nx      = ocp_ric.nx,
                                             .nu      = ocp_ric.nu,
                                             .ny      = ocp_ric.ny,
                                             .ny_N    = ocp_ric.ny}};
    const auto N = ocp.dim.N_horiz;
    for (index_t i = 0; i < N; ++i) {
        as_eigen(ocp.A(i))       = ocp_ric.A(i);
        as_eigen(ocp.B(i))       = ocp_ric.B(i);
        as_eigen(ocp.C(i))       = ocp_ric.C(i);
        as_eigen(ocp.D(i))       = ocp_ric.D(i);
        as_eigen(ocp.Q(i))       = ocp_ric.Q(i);
        as_eigen(ocp.S(i))       = ocp_ric.S(i).transpose();
        as_eigen(ocp.S_trans(i)) = ocp_ric.S(i);
        as_eigen(ocp.R(i))       = ocp_ric.R(i);
    }
    as_eigen(ocp.C(N)) = ocp_ric.C(N);
    as_eigen(ocp.Q(N)) = ocp_ric.Q(N);
    CyclicOCPSolver solver{ocp};
    return solver;
}

void bm_factor_riccati(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    RiccatiFactor fac{.ocp = ocp};
    for (auto _ : state)
        factor(fac, Σ);
}

void bm_solve_riccati(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    RiccatiFactor fac{.ocp = ocp};
    factor(fac, Σ);
    for (auto _ : state)
        solve(fac);
}

void bm_update_riccati(benchmark::State &state) {
    using std::exp2;
    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 10};
    std::bernoulli_distribution bern{0.25};

    auto [ocp, Σ] = generate_ocp(state);
    RiccatiFactor fac{.ocp = ocp};
    mat ΔΣ = Σ;
    for (auto _ : state) {
        state.PauseTiming();
        factor(fac, Σ);
        std::ranges::generate(ΔΣ.reshaped(), [&] {
            return bern(rng) ? exp2(nrml(rng)) : real_t{0};
        });
        state.ResumeTiming();
        update(fac, ΔΣ);
    }
}

void bm_factor_schur(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    auto ocp_sch  = OCPDataSchur::from_riccati(ocp);
    SchurFactor factor_sch{.ocp = ocp_sch};
    for (auto _ : state)
        factor(factor_sch, Σ);
}

void bm_solve_schur(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    auto ocp_sch  = OCPDataSchur::from_riccati(ocp);
    SchurFactor factor_sch{.ocp = ocp_sch};
    factor(factor_sch, Σ);
    for (auto _ : state)
        solve(factor_sch);
}

void bm_update_schur(benchmark::State &state) {
    using std::exp2;
    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 10};
    std::bernoulli_distribution bern{0.25};

    auto [ocp, Σ] = generate_ocp(state);
    auto ocp_sch  = OCPDataSchur::from_riccati(ocp);
    SchurFactor factor_sch{.ocp = ocp_sch};
    mat ΔΣ = Σ;

    for (auto _ : state) {
        state.PauseTiming();
        factor(factor_sch, Σ);
        std::ranges::generate(ΔΣ.reshaped(), [&] {
            return bern(rng) ? exp2(nrml(rng)) : real_t{0};
        });
        state.ResumeTiming();
        update(factor_sch, ΔΣ);
    }
}

void bm_factor_schur_kqt(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    auto solver   = build_cyclic_ocp_solver(ocp);
    auto Σb       = solver.pack_constr(as_span(Σ.reshaped()));
    auto Jb       = solver.pack_constr(std::span<bool>{});
    Jb.set_constant(true);
    KOQKATOO_OMP(parallel) {
        for (auto _ : state) {
            solver.compute_Ψ(1, Σb, Jb);
            solver.factor_Ψ();
        }
    }
}

void bm_solve_schur_kqt(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    auto solver   = build_cyclic_ocp_solver(ocp);
    auto Σb       = solver.pack_constr(as_span(Σ.reshaped()));
    auto Jb       = solver.pack_constr(std::span<bool>{});
    auto gradb = solver.pack_var(), Mᵀλb = solver.pack_var(),
         Aᵀŷb = solver.pack_var(), db = solver.pack_var(),
         MᵀΔλb = solver.pack_var();
    auto Δλb   = solver.pack_dyn();
    Jb.set_constant(true);
    KOQKATOO_OMP(parallel) {
        solver.compute_Ψ(1, Σb, Jb);
        solver.factor_Ψ();
        for (auto _ : state) {
            solver.solve_H_fwd(gradb, Mᵀλb, Aᵀŷb, db, Δλb);
            solver.solve_Ψ(Δλb);
            solver.solve_H_rev(db, Δλb, MᵀΔλb);
        }
    }
}

std::vector<benchmark::internal::Benchmark *> benchmarks;
void register_benchmark(benchmark::internal::Benchmark *bm) {
    benchmarks.push_back(bm);
}
#define OCP_BENCHMARK(...) BENCHMARK(__VA_ARGS__)->Apply(register_benchmark)

OCP_BENCHMARK(bm_factor_riccati);
OCP_BENCHMARK(bm_solve_riccati);
OCP_BENCHMARK(bm_update_riccati);
OCP_BENCHMARK(bm_factor_schur);
OCP_BENCHMARK(bm_solve_schur);
OCP_BENCHMARK(bm_update_schur);
OCP_BENCHMARK(bm_factor_schur_kqt);
OCP_BENCHMARK(bm_solve_schur_kqt);

enum class BenchmarkType { vary_N, vary_nu, vary_ny, vary_nx_frac };

int main(int argc, char **argv) {
    if (argc < 1)
        return 1;
    // Parse command-line arguments
    std::vector<char *> argvv{argv, argv + argc};
    int64_t N = 32, nx = 16, nu = 8, ny = 8, step = 1;
    BenchmarkType type = BenchmarkType::vary_N;
    for (auto it = argvv.begin(); it != argvv.end();) {
        std::string_view arg = *it;
        if (std::string_view flag = "--N="; arg.starts_with(flag))
            N = std::stoi(std::string(arg.substr(flag.size())));
        else if (std::string_view flag = "--nx="; arg.starts_with(flag))
            nx = std::stoi(std::string(arg.substr(flag.size())));
        else if (std::string_view flag = "--nu="; arg.starts_with(flag))
            nu = std::stoi(std::string(arg.substr(flag.size())));
        else if (std::string_view flag = "--ny="; arg.starts_with(flag))
            ny = std::stoi(std::string(arg.substr(flag.size())));
        else if (std::string_view flag = "--step="; arg.starts_with(flag))
            step = std::stoi(std::string(arg.substr(flag.size())));
        else if (arg == "--vary-N")
            type = BenchmarkType::vary_N;
        else if (arg == "--vary-nu")
            type = BenchmarkType::vary_nu;
        else if (arg == "--vary-ny")
            type = BenchmarkType::vary_ny;
        else if (arg == "--vary-nx-frac")
            type = BenchmarkType::vary_nx_frac;
        else {
            ++it;
            continue;
        }
        it = argvv.erase(it);
    }

    for (auto *bm : benchmarks) {
        bm->MeasureProcessCPUTime()->UseRealTime();
        bm->ArgNames({"N", "nx", "nu", "ny"});
        switch (type) {
            case BenchmarkType::vary_N:
                for (int64_t i = step; i <= N; i += step)
                    bm->Args({i, nx, nu, ny});
                break;
            case BenchmarkType::vary_nu:
                for (int64_t i = step; i <= nu; i += step)
                    bm->Args({N, nx, i, ny});
                break;
            case BenchmarkType::vary_ny:
                for (int64_t i = step; i <= ny; i += step)
                    bm->Args({N, nx, nu, i});
                break;
            case BenchmarkType::vary_nx_frac:
                for (int64_t i = step; i <= nx; i += step)
                    bm->Args({N, i, (i * nu + (nx - 1) / 2) / nx, ny});
                break;
            default: KOQKATOO_ASSUME(false);
        }
    }

    argc = static_cast<int>(argvv.size());
    ::benchmark::Initialize(&argc, argvv.data());
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
#if GUANAQO_WITH_OPENMP
    benchmark::AddCustomContext("OMP_NUM_THREADS",
                                std::to_string(omp_get_max_threads()));
#endif
    benchmark::AddCustomContext("koqkatoo_build_time", koqkatoo_build_time);
    benchmark::AddCustomContext("koqkatoo_commit_hash", koqkatoo_commit_hash);
    benchmark::AddCustomContext("hyhound_build_time", hyhound_build_time);
    benchmark::AddCustomContext("hyhound_commit_hash", hyhound_commit_hash);
#if defined(__AVX512F__)
    benchmark::AddCustomContext("arch", "avx512f");
#elif defined(__AVX2__)
    benchmark::AddCustomContext("arch", "avx2");
#elif defined(__AVX__)
    benchmark::AddCustomContext("arch", "avx");
#elif defined(__SSE3__)
    benchmark::AddCustomContext("arch", "sse3");
#endif
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
}
