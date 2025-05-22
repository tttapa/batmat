#include <koqkatoo/assume.hpp>
#include <koqkatoo/linalg-compact/mkl.hpp>
#include <koqkatoo/ocp/cyclic-solver/cyclic-solver.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/openmp.h>
#include <benchmark/benchmark.h>
#include <guanaqo/eigen/span.hpp>
#include <guanaqo/openmp.h>
#include <hyhound-version.h>
#include <koqkatoo-version.h>
#include <omp.h>

#include <guanaqo/trace.hpp>
#include <hyhound/ocp/riccati.hpp>
#include <hyhound/ocp/schur.hpp>

#include <experimental/simd>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>

#if WITH_BLASFEO
#include "riccati-blasfeo.hpp"
#include <blasfeo.h>
#endif

using namespace hyhound;
using namespace hyhound::ocp;
namespace stdx = std::experimental;
using guanaqo::as_span;

namespace koqkatoo::ocp::test {
template <koqkatoo::index_t>
struct CyclicOCPSolver;
}

std::shared_ptr<koqkatoo::ocp::test::CyclicOCPSolver<4>>
build_new_cyclic_solver(const koqkatoo::ocp::LinearOCPStorage &ocp,
                        koqkatoo::index_t lP);
void run_new_cyclic_solver(koqkatoo::ocp::test::CyclicOCPSolver<4> &solver);

#if GUANAQO_WITH_TRACING
std::map<std::tuple<std::string, std::string>, std::filesystem::path> traces;
void trace(auto &&fun, const auto &name, const auto &params) {
    std::string filename = std::format("{}.csv", name);
    std::filesystem::path out_dir{"traces"};
    out_dir /= *koqkatoo_commit_hash ? koqkatoo_commit_hash : "unknown";
    out_dir /= KOQKATOO_MKL_IF_ELSE("mkl", "openblas");
    out_dir /= params;
    std::filesystem::path out_file = out_dir / filename;
    if (auto [_, ins] = traces.insert({{name, params}, out_file}); !ins)
        return;
    guanaqo::trace_logger.reset();
    fun();
    std::filesystem::create_directories(out_dir);
    std::ofstream csv{out_file};
    guanaqo::TraceLogger::write_column_headings(csv) << '\n';
    for (const auto &log : guanaqo::trace_logger.get_logs())
        csv << log << '\n';
}
void print_traces(std::ostream &os) {
    for (const auto &[_, pth] : traces)
        os << pth << '\n';
}
#else
void trace(auto &&, const auto &, const auto &) {}
void print_traces(std::ostream &) {}
#endif

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
    using abi             = stdx::simd_abi::deduce_t<real_t, 4>;
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

auto build_cyclic_ocp_solver_new(const OCPDataRiccati &ocp_ric, index_t lP) {
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
    return build_new_cyclic_solver(ocp, lP);
}

void bm_factor_riccati(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    RiccatiFactor fac{.ocp = ocp};
    for (auto _ : state)
        factor(fac, Σ);
    auto params = std::format("nx={}-nu={}-ny={}-N={}-thr={}", ocp.nx, ocp.nu,
                              ocp.ny, ocp.N, 1);
    trace([&] { factor(fac, Σ); }, "factor_riccati", params);
}

void bm_solve_riccati(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    RiccatiFactor fac{.ocp = ocp};
    factor(fac, Σ);
    for (auto _ : state)
        solve(fac);
    auto params = std::format("nx={}-nu={}-ny={}-N={}-thr={}", ocp.nx, ocp.nu,
                              ocp.ny, ocp.N, 1);
    trace([&] { solve(fac); }, "solve_riccati", params);
}

void bm_update_riccati(benchmark::State &state) {
    using std::exp2;
    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 10};
    std::bernoulli_distribution bern{0.25};

    auto [ocp, Σ] = generate_ocp(state);
    RiccatiFactor fac{.ocp = ocp};
    mat ΔΣ          = Σ;
    const auto prep = [&] {
        factor(fac, Σ);
        std::ranges::generate(ΔΣ.reshaped(), [&] {
            return bern(rng) ? exp2(nrml(rng)) : real_t{0};
        });
    };
    for (auto _ : state) {
        state.PauseTiming();
        prep();
        state.ResumeTiming();
        update(fac, ΔΣ);
    }
    auto params = std::format("nx={}-nu={}-ny={}-N={}-thr={}-upd={}", ocp.nx,
                              ocp.nu, ocp.ny, ocp.N, 1, bern.p());
    prep();
    trace([&] { update(fac, ΔΣ); }, "update_riccati", params);
}

void bm_factor_schur(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    auto ocp_sch  = OCPDataSchur::from_riccati(ocp);
    SchurFactor factor_sch{.ocp = ocp_sch};
    for (auto _ : state)
        factor(factor_sch, Σ);
    auto params = std::format("nx={}-nu={}-ny={}-N={}-thr={}", ocp.nx, ocp.nu,
                              ocp.ny, ocp.N, 1);
    trace([&] { factor(factor_sch, Σ); }, "factor_schur", params);
}

void bm_solve_schur(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    auto ocp_sch  = OCPDataSchur::from_riccati(ocp);
    SchurFactor factor_sch{.ocp = ocp_sch};
    factor(factor_sch, Σ);
    for (auto _ : state)
        solve(factor_sch);
    auto params = std::format("nx={}-nu={}-ny={}-N={}-thr={}", ocp.nx, ocp.nu,
                              ocp.ny, ocp.N, 1);
    trace([&] { solve(factor_sch); }, "solve_schur", params);
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
    const auto S  = std::numeric_limits<real_t>::infinity();
    Jb.set_constant(true);
    const auto do_factor = [&] {
        KOQKATOO_OMP(parallel) {
            solver.compute_Ψ(S, Σb, Jb);
            solver.factor_Ψ();
        }
    };
    for (auto _ : state)
        do_factor();
    auto params =
        std::format("nx={}-nu={}-ny={}-N={}-thr={}", ocp.nx, ocp.nu, ocp.ny,
                    ocp.N, KOQKATOO_OMP_IF_ELSE(omp_get_max_threads(), 1));
    trace([&] { do_factor(); }, "factor_schur_kqt", params);
}

void bm_factor_schur_kqt_1(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    auto solver   = build_cyclic_ocp_solver(ocp);
    auto Σb       = solver.pack_constr(as_span(Σ.reshaped()));
    auto Jb       = solver.pack_constr(std::span<bool>{});
    const auto S  = std::numeric_limits<real_t>::infinity();
    Jb.set_constant(true);
    for (auto _ : state) {
        solver.compute_Ψ(S, Σb, Jb);
        solver.factor_Ψ();
    }
}

void bm_solve_schur_kqt(benchmark::State &state) {
    auto [ocp, Σ] = generate_ocp(state);
    auto solver   = build_cyclic_ocp_solver(ocp);
    auto Σb       = solver.pack_constr(as_span(Σ.reshaped()));
    auto Jb       = solver.pack_constr(std::span<bool>{});
    auto gradb = solver.pack_var(), Mᵀλb = solver.pack_var(),
         Aᵀŷb = solver.pack_var(), db = solver.pack_var(),
         MᵀΔλb   = solver.pack_var();
    auto Δλb     = solver.pack_dyn();
    const auto S = std::numeric_limits<real_t>::infinity();
    Jb.set_constant(true);
    KOQKATOO_OMP(parallel) {
        solver.compute_Ψ(S, Σb, Jb);
        solver.factor_Ψ();
    }
    const auto do_solve = [&] {
        KOQKATOO_OMP(parallel) {
            solver.solve_H_fwd(gradb, Mᵀλb, Aᵀŷb, db, Δλb);
            solver.solve_Ψ(Δλb);
            solver.solve_H_rev(db, Δλb, MᵀΔλb);
        }
    };
    for (auto _ : state)
        do_solve();
    auto params =
        std::format("nx={}-nu={}-ny={}-N={}-thr={}", ocp.nx, ocp.nu, ocp.ny,
                    ocp.N, KOQKATOO_OMP_IF_ELSE(omp_get_max_threads(), 1));
    trace([&] { do_solve(); }, "solve_schur_kqt", params);
}

void bm_factor_new_kqt(benchmark::State &state) {
    auto [ocp, Σ]    = generate_ocp(state);
    const index_t lP = state.range(4);
#if !KOQKATOO_WITH_OPENMP
#error ""
#endif
    omp_set_num_threads(1 << lP);
    auto solver          = build_cyclic_ocp_solver_new(ocp, lP);
    const auto do_factor = [&] { run_new_cyclic_solver(*solver); };
    for (auto _ : state)
        do_factor();
    auto params =
        std::format("nx={}-nu={}-ny={}-N={}-thr={}", ocp.nx, ocp.nu, ocp.ny,
                    ocp.N, KOQKATOO_OMP_IF_ELSE(omp_get_max_threads(), 1));
    trace([&] { do_factor(); }, "factor_schur_kqt", params);
}

#if WITH_BLASFEO

void bm_factor_riccati_blasfeo(benchmark::State &state) {
    auto [ocp, Σ]    = generate_ocp(state);
    auto ocp_blasfeo = koqkatoo::ocp::OCPDataRiccati::from_riccati(ocp);
    koqkatoo::ocp::RiccatiFactor fac{.ocp = ocp_blasfeo};
    for (auto _ : state)
        factor(fac, Σ);
    auto params = std::format("nx={}-nu={}-ny={}-N={}-thr={}", ocp.nx, ocp.nu,
                              ocp.ny, ocp.N, 1);
    trace([&] { factor(fac, Σ); }, "factor_riccati_blasfeo", params);
}

#endif

std::vector<benchmark::internal::Benchmark *> benchmarks;
void register_benchmark(benchmark::internal::Benchmark *bm) {
    benchmarks.push_back(bm);
}
#define OCP_BENCHMARK(...) BENCHMARK(__VA_ARGS__)->Apply(register_benchmark)

OCP_BENCHMARK(bm_factor_riccati);
// OCP_BENCHMARK(bm_solve_riccati);
// OCP_BENCHMARK(bm_update_riccati);
OCP_BENCHMARK(bm_factor_schur);
// OCP_BENCHMARK(bm_solve_schur);
// OCP_BENCHMARK(bm_update_schur);
#if WITH_BLASFEO
OCP_BENCHMARK(bm_factor_riccati_blasfeo);
#endif
OCP_BENCHMARK(bm_factor_schur_kqt);
OCP_BENCHMARK(bm_factor_schur_kqt_1);
OCP_BENCHMARK(bm_factor_new_kqt);
// OCP_BENCHMARK(bm_solve_schur_kqt);

enum class BenchmarkType {
    None,
    vary_N,
    vary_N_pow_2,
    vary_nu,
    vary_ny,
    vary_nx_frac
};

int main(int argc, char **argv) {
    if (argc < 1)
        return 1;
    // Parse command-line arguments
    std::vector<char *> argvv{argv, argv + argc};
    int64_t N = 31, nx = 16, nu = 8, ny = 8, step = 1,
            lP = KOQKATOO_OMP_IF_ELSE(
                std::bit_width(static_cast<unsigned>(omp_get_num_threads())) -
                    1,
                0);
    BenchmarkType type = BenchmarkType::None;
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
        else if (std::string_view flag = "--lP="; arg.starts_with(flag))
            lP = std::stoi(std::string(arg.substr(flag.size())));
        else if (arg == "--vary-N")
            type = BenchmarkType::vary_N;
        else if (arg == "--vary-N-pow-2")
            type = BenchmarkType::vary_N_pow_2;
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
        bm->ArgNames({"N", "nx", "nu", "ny", "lP"});
        switch (type) {
            case BenchmarkType::None: bm->Args({N, nx, nu, ny}); break;
            case BenchmarkType::vary_N:
                for (int64_t i = step; i <= N; i += step)
                    bm->Args({i, nx, nu, ny, lP});
                break;
            case BenchmarkType::vary_N_pow_2:
                for (int64_t i = 8; i <= N; i <<= step)
                    for (int64_t lPi = 0; 1 << (lPi + 2) <= i; lPi += 1)
                        bm->Args({i, nx, nu, ny, lPi});
                break;
            case BenchmarkType::vary_nu:
                for (int64_t i = step; i <= nu; i += step)
                    bm->Args({N, nx, i, ny, lP});
                break;
            case BenchmarkType::vary_ny:
                for (int64_t i = step; i <= ny; i += step)
                    bm->Args({N, nx, nu, i, lP});
                break;
            case BenchmarkType::vary_nx_frac:
                for (int64_t i = step; i <= nx; i += step)
                    bm->Args({N, i, (i * nu + (nx - 1) / 2) / nx, ny, lP});
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
    print_traces(std::cout);
}
