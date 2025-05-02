#include <gtest/gtest.h>

#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/linalg-compact/mkl.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/matrix-view.hpp>
#include <koqkatoo/ocp/conversion.hpp>
#include <koqkatoo/ocp/cyclic-solver/cyclic-solver.hpp>
#include <koqkatoo/ocp/cyclic-solver/packing.tpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/thread-pool.hpp>
#include <guanaqo/trace.hpp>
#include <koqkatoo-version.h>

#include <guanaqo/eigen/span.hpp>
#include <guanaqo/eigen/view.hpp>
#include <guanaqo/linalg/eigen/sparse.hpp>
#include <guanaqo/linalg/sparsity-conversions.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>

#include <Eigen/Eigen>
#include <Eigen/SparseLU>

#include "eigen-matchers.hpp"
#include "reference-solution.hpp"

namespace ko   = koqkatoo::ocp;
namespace stdx = std::experimental;
using guanaqo::as_eigen;
using guanaqo::as_span;
using koqkatoo::index_t;
using koqkatoo::real_t;
using koqkatoo::RealMatrixView;
namespace sp = guanaqo::linalg::sparsity;

const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.9));

struct OCPCyclic
    : testing::TestWithParam<std::tuple<index_t, index_t, index_t, bool>> {};

template <index_t VL = 4>
void solve_cyclic(const koqkatoo::ocp::LinearOCPStorage &ocp, real_t S,
                  std::span<const real_t> Σ, std::span<const bool> J,
                  std::span<const real_t> x, std::span<const real_t> grad,
                  std::span<const real_t> λ, std::span<const real_t> b,
                  std::span<const real_t> ŷ, std::span<real_t> Mxb,
                  std::span<real_t> Mᵀλ, std::span<real_t> Aᵀŷ,
                  std::span<real_t> d, std::span<real_t> Δλ,
                  std::span<real_t> MᵀΔλ, bool use_pcg) {
    using Solver = ko::CyclicOCPSolver<stdx::simd_abi::deduce_t<real_t, VL>>;

    Solver solver{ocp, {.use_pcg = use_pcg}};
    auto Σb = solver.pack_constr(Σ), ŷb = solver.pack_constr(ŷ),
         xb = solver.pack_var(x), gradb = solver.pack_var(grad),
         bb = solver.pack_dyn(b), λb = solver.pack_dyn(λ);
    auto Jb   = solver.pack_constr(J);
    auto Mᵀλb = solver.pack_var(), Aᵀŷb = solver.pack_var(),
         db = solver.pack_var(), Δλb = solver.pack_dyn(),
         MᵀΔλb = solver.pack_var();
    for (int i = 0; i < 10; ++i) {
        KOQKATOO_OMP(parallel) {
            solver.mat_vec_dyn_tp(λb, Mᵀλb);
            solver.mat_vec_dyn(xb, bb, Δλb);
            KOQKATOO_OMP(single)
            solver.template unpack_dyn<real_t>(Δλb, Mxb);
            solver.mat_vec_constr_tp(ŷb, Aᵀŷb);
#if GUANAQO_WITH_TRACING
            KOQKATOO_OMP(single)
            guanaqo::trace_logger.reset();
#endif
            solver.compute_Ψ(S, Σb, Jb);
            solver.factor_Ψ();
            solver.solve_H_fwd(gradb, Mᵀλb, Aᵀŷb, db, Δλb);
            solver.solve_Ψ(Δλb);
            solver.solve_H_rev(db, Δλb, MᵀΔλb);
        }
    }
    solver.template unpack_var<real_t>(Mᵀλb, Mᵀλ);
    solver.template unpack_var<real_t>(MᵀΔλb, MᵀΔλ);
    solver.template unpack_var<real_t>(Aᵀŷb, Aᵀŷ);
    solver.template unpack_var<real_t>(db, d);
    solver.template unpack_dyn<real_t>(Δλb, Δλ);
}

const int n_threads = 8;
TEST_P(OCPCyclic, solve) {
    KOQKATOO_OMP_IF(omp_set_num_threads(n_threads));
    koqkatoo::pool_set_num_threads(n_threads);
    koqkatoo::fork_set_num_threads(n_threads);
    GUANAQO_IF_ITT(koqkatoo::foreach_thread([](index_t i, index_t) {
        __itt_thread_set_name(std::format("OMP({})", i).c_str());
    }));

    auto [N_horiz, nx, vl, use_pcg] = GetParam();
    if (N_horiz / vl <= 2)
        GTEST_SKIP() << "Horizon too short (N=" << N_horiz << ", vl=" << vl
                     << ")";

    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp({.N_horiz = N_horiz, //
                                        .nx      = nx,
                                        .nu      = 30 * nx / 40,
                                        .ny      = 0 * nx / 40,
                                        .ny_N    = 0 * nx / 40},
                                       12345);

    // Instantiate the OCP KKT solver.
    index_t n_var = ocp.num_variables(), n_constr = ocp.num_constraints(),
            n_dyn_constr = ocp.num_dynamics_constraints();

    auto solve_cyclic = vl == 2   ? ::solve_cyclic<2>
                        : vl == 4 ? ::solve_cyclic<4>
                        : vl == 8 ? ::solve_cyclic<8>
                                  : nullptr;
    if (!solve_cyclic)
        FAIL() << "Invalid vector length " << vl;

    // Generate some random optimization solver data.
    using VectorXreal = Eigen::VectorX<real_t>;
    Eigen::VectorX<bool> J(n_constr),  // Active set.
        J0(n_constr), J1(n_constr);    // Active set for initialization.
    VectorXreal Σ(n_constr),           // ALM penalty factors
        ŷ(n_constr);                   //  & corresponding Lagrange multipliers.
    VectorXreal x(n_var), grad(n_var); // Current iterate and cost gradient.
    VectorXreal b(n_dyn_constr),       // Dynamics constraints right-hand side
        λ(n_dyn_constr);               //  & corresponding Lagrange multipliers.

    real_t S = std::exp2(nrml(rng)) * 1e-2; // primal regularization
    std::ranges::generate(J, [&] { return bernoulli(rng); });
    std::ranges::generate(J0, [&] { return bernoulli(rng); });
    std::ranges::generate(J1, [&] { return bernoulli(rng); });
    std::ranges::generate(Σ, [&] { return std::exp2(nrml(rng)); });
    std::ranges::generate(ŷ, [&] { return nrml(rng); });
    std::ranges::generate(x, [&] { return nrml(rng); });
    std::ranges::generate(grad, [&] { return nrml(rng); });
    std::ranges::generate(b, [&] { return nrml(rng); });
    std::ranges::generate(λ, [&] { return nrml(rng); });

    auto [Q, G, M, K] = ko::testing::reference_qp(ocp, S, Σ, J);

    VectorXreal Mxb(n_dyn_constr), Mᵀλ(n_var), d(n_var), Δλ(n_dyn_constr),
        MᵀΔλ(n_var), Aᵀŷ(n_var);
    for (index_t i = 0; i < 10; ++i)
        solve_cyclic(ocp, S, as_span(Σ), as_span(J), as_span(x), as_span(grad),
                     as_span(λ), as_span(b), as_span(ŷ), as_span(Mxb),
                     as_span(Mᵀλ), as_span(Aᵀŷ), as_span(d), as_span(Δλ),
                     as_span(MᵀΔλ), use_pcg);
#if GUANAQO_WITH_TRACING
    guanaqo::trace_logger.reset();
#endif
    solve_cyclic(ocp, S, as_span(Σ), as_span(J), as_span(x), as_span(grad),
                 as_span(λ), as_span(b), as_span(ŷ), as_span(Mxb), as_span(Mᵀλ),
                 as_span(Aᵀŷ), as_span(d), as_span(Δλ), as_span(MᵀΔλ), use_pcg);
#if GUANAQO_WITH_TRACING
    {
        const auto [N, nx, nu, ny, ny_N] = ocp.dim;
        std::string name                 = std::format("factor_cyclic.csv");
        std::filesystem::path out_dir{"traces"};
        out_dir /= *koqkatoo_commit_hash ? koqkatoo_commit_hash : "unknown";
        out_dir /= KOQKATOO_MKL_IF_ELSE("mkl", "openblas");
        out_dir /= std::format("nx={}-nu={}-ny={}-N={}-thr={}-vl={}{}", nx, nu,
                               ny, N, n_threads, vl, use_pcg ? "-pcg" : "");
        std::filesystem::create_directories(out_dir);
        std::ofstream csv{out_dir / name};
        guanaqo::TraceLogger::write_column_headings(csv) << '\n';
        for (const auto &log : guanaqo::trace_logger.get_logs())
            csv << log << '\n';
        std::cout << out_dir << std::endl;
    }
#endif

    // Solve the full KKT system using Eigen (LU because indefinite).
    VectorXreal kkt_rhs_ref = VectorXreal::Zero(K.rows()),
                kkt_sol_ref(K.rows());
    VectorXreal Gᵀŷ_ref                  = G.transpose() * ŷ;
    VectorXreal Mᵀλ_ref                  = M.transpose() * λ;
    VectorXreal Mxb_ref                  = M * x - b;
    kkt_rhs_ref.topRows(n_var)           = -grad - Mᵀλ_ref - Gᵀŷ_ref;
    kkt_rhs_ref.bottomRows(n_dyn_constr) = -Mxb_ref;
    EXPECT_THAT(Mxb, EigenAlmostEqual(Mxb_ref, 10 * ε));
    EXPECT_THAT(Mᵀλ, EigenAlmostEqual(Mᵀλ_ref, 10 * ε));
    EXPECT_THAT(Aᵀŷ, EigenAlmostEqual(Gᵀŷ_ref, 10 * ε));

    Eigen::SparseLU<decltype(K)> luK(K);
    ASSERT_TRUE(luK.info() == Eigen::Success);
    kkt_sol_ref          = luK.solve(kkt_rhs_ref);
    auto d_ref           = kkt_sol_ref.topRows(n_var);
    auto Δλ_ref          = kkt_sol_ref.bottomRows(n_dyn_constr);
    VectorXreal MᵀΔλ_ref = M.transpose() * Δλ_ref;

    real_t κ = ko::testing::cond_sparse_sym(K, &luK);
    std::cout << "κ(K) = " << guanaqo::float_to_str(κ) << std::endl;

    // Compare the koqkatoo OCP solution to the Eigen reference solution.
    EXPECT_THAT(Δλ, EigenAlmostEqual(Δλ_ref, ε * κ));
    EXPECT_THAT(d, EigenAlmostEqual(d_ref, ε * κ));
    EXPECT_THAT(MᵀΔλ, EigenAlmostEqual(MᵀΔλ_ref, ε * κ));
    // TODO: compute condition number instead of hard-coded tolernaces
}

INSTANTIATE_TEST_SUITE_P(OCPCyclic, OCPCyclic,
                         testing::Combine( //
                             testing::Values<index_t>(31),
                             testing::Values<index_t>(40),
                             testing::Values<index_t>(4),
                             testing::Bool()));
