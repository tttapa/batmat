#include <gtest/gtest.h>

#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/linalg-compact/mkl.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/ocp/cyclocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/thread-pool.hpp>
#include <guanaqo/print.hpp>
#include <guanaqo/trace.hpp>
#include <koqkatoo-version.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

using koqkatoo::index_t;
using koqkatoo::real_t;

TEST(CyclOCP, factor) {
    using namespace koqkatoo::ocp;

    const int log_n_threads = 2;

    KOQKATOO_OMP_IF(omp_set_num_threads(1 << log_n_threads));
    koqkatoo::pool_set_num_threads(1 << log_n_threads);
    koqkatoo::fork_set_num_threads(1 << log_n_threads);
    GUANAQO_IF_ITT(koqkatoo::foreach_thread([](index_t i, index_t) {
        __itt_thread_set_name(std::format("OMP({})", i).c_str());
    }));

    using Solver     = cyclocp::CyclicOCPSolver<4>;
    const index_t lP = log_n_threads + Solver::lvl;
    OCPDim dim{.N_horiz = 96, .nx = 40, .nu = 30, .ny = 50, .ny_N = 50};
    auto ocp = generate_random_ocp(dim);
    Solver solver{.dim = dim, .lP = lP};
    solver.initialize(ocp);
    const index_t ny_ny_N = std::max(dim.ny, dim.ny_N);
    std::vector<real_t> Σ_lin(dim.N_horiz * dim.ny + dim.ny_N);
    Solver::matrix λ{{.depth = dim.N_horiz, .rows = dim.nx, .cols = 1}},
        ux{{.depth = dim.N_horiz, .rows = dim.nu + dim.nx, .cols = 1}},
        Mᵀλ{{.depth = dim.N_horiz, .rows = dim.nu + dim.nx, .cols = 1}},
        DCux{{.depth = dim.N_horiz, .rows = ny_ny_N, .cols = 1}},
        DCᵀΣDCux{{.depth = dim.N_horiz, .rows = dim.nu + dim.nx, .cols = 1}},
        grad{{.depth = dim.N_horiz, .rows = dim.nu + dim.nx, .cols = 1}},
        Mxb{{.depth = dim.N_horiz, .rows = dim.nx, .cols = 1}},
        Σ{{.depth = dim.N_horiz, .rows = dim.ny, .cols = 1}},
        Σ2{{.depth = dim.N_horiz, .rows = dim.ny, .cols = 1}},
        ΔΣ{{.depth = dim.N_horiz, .rows = dim.ny, .cols = 1}};
    std::mt19937 rng(102030405);
    std::uniform_real_distribution<real_t> uni(-1, 1);
    std::bernoulli_distribution bern(0.01);
    std::ranges::generate(λ, [&] { return uni(rng); });
    std::ranges::generate(ux, [&] { return uni(rng); });
    std::ranges::generate(Σ_lin, [&] { return std::exp2(uni(rng)); });
    std::vector<real_t> Σ_lin2 = Σ_lin;
    for (auto &Σ2i : Σ_lin2)
        if (bern(rng))
            Σ2i = std::exp2(uni(rng));
    solver.initialize_Σ(Σ_lin, Σ);
    solver.initialize_Σ(Σ_lin2, Σ2);
    std::ranges::transform(Σ2, Σ, std::ranges::begin(ΔΣ), std::minus<>{});

    if (std::ofstream f("rhs.csv"); f) {
        auto b = solver.build_rhs(ux, λ);
        for (auto x : b)
            f << guanaqo::float_to_str(x) << '\n';
    }

    const bool alt        = true;
    const auto ux_initial = ux, λ_initial = λ;
    for (int i = 0; i < 500; ++i) {
        solver.factor(Σ, alt);
        solver.update(ΔΣ);
        solver.solve(ux, λ);
        solver.residual_dynamics_constr(ux, λ_initial, Mxb);
        solver.transposed_dynamics_constr(λ, Mᵀλ);
        solver.cost_gradient(ux, -1, ux_initial, 0, grad);
        solver.general_constr(ux, DCux);
        Solver::compact_blas::xhadamard(Σ2, DCux);
        solver.transposed_general_constr(DCux, DCᵀΣDCux);
        ux.view = ux_initial.view;
        λ.view  = λ_initial.view;
    }
#if GUANAQO_WITH_TRACING
    guanaqo::trace_logger.reset();
#endif
    solver.factor(Σ, alt);
    solver.update(ΔΣ);
    solver.solve(ux, λ);
    solver.residual_dynamics_constr(ux, λ_initial, Mxb);
    solver.transposed_dynamics_constr(λ, Mᵀλ);
    solver.cost_gradient(ux, -1, ux_initial, 0, grad);
    solver.general_constr(ux, DCux);
    Solver::compact_blas::xhadamard(Σ2, DCux);
    solver.transposed_general_constr(DCux, DCᵀΣDCux);

    using std::pow;
    const auto ε = pow(std::numeric_limits<real_t>::epsilon(), 0.6);
    EXPECT_LE(Solver::compact_blas::xnrminf(Mxb), ε);
    Solver::compact_blas::xadd_copy(grad, grad, DCᵀΣDCux, Mᵀλ);
    EXPECT_LE(Solver::compact_blas::xnrminf(grad), ε);

#if GUANAQO_WITH_TRACING
    {
        koqkatoo::foreach_thread(
            [](index_t i, index_t) { GUANAQO_TRACE("thread_id", i); });
        const auto N     = solver.dim.N_horiz;
        const auto VL    = solver.vl;
        std::string name = std::format("factor_cyclic_new.csv");
        std::filesystem::path out_dir{"traces"};
#if USE_JACOBI_PREC
        const std::string_view pcg = "jacobi";
#else
        const std::string_view pcg = "stair";
#endif
        out_dir /= *koqkatoo_commit_hash ? koqkatoo_commit_hash : "unknown";
        out_dir /= KOQKATOO_MKL_IF_ELSE("mkl", "openblas");
        out_dir /= std::format("nx={}-nu={}-ny={}-N={}-thr={}-vl={}-pcg={}{}",
                               solver.dim.nx, solver.dim.nu, solver.dim.ny, N,
                               1 << log_n_threads, VL, pcg, alt ? "-alt" : "");
        std::filesystem::create_directories(out_dir);
        std::ofstream csv{out_dir / name};
        guanaqo::TraceLogger::write_column_headings(csv) << '\n';
        for (const auto &log : guanaqo::trace_logger.get_logs())
            csv << log << '\n';
        std::cout << out_dir << std::endl;
    }
#endif

    if (std::ofstream f("sparse.csv"); f) {
        auto sp = solver.build_sparse(ocp, Σ_lin2);
        for (auto [r, c, x] : sp)
            f << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
    if (std::ofstream f("sparse_factor.csv"); f) {
        auto sp = solver.build_sparse_factor();
        for (auto [r, c, x] : sp)
            f << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
    if (std::ofstream f("sparse_diag.csv"); f) {
        auto sp = solver.build_sparse_diag();
        for (auto [r, c, x] : sp)
            f << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
    if (std::ofstream f("sol.csv"); f) {
        auto b = solver.build_rhs(ux, λ);
        for (auto x : b)
            f << guanaqo::float_to_str(x) << '\n';
    }

    solver.factor(Σ2, alt);
    if (std::ofstream f("sparse_refactor.csv"); f) {
        auto sp = solver.build_sparse_factor();
        for (auto [r, c, x] : sp)
            f << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
}
