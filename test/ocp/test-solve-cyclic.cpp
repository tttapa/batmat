#include <gtest/gtest.h>

#include <koqkatoo/matrix-view.hpp>
#include <koqkatoo/ocp/conversion.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/openmp.h>

#include <guanaqo/eigen/span.hpp>
#include <guanaqo/eigen/view.hpp>
#include <guanaqo/linalg/eigen/sparse.hpp>
#include <guanaqo/linalg/sparsity-conversions.hpp>

#include <algorithm>
#include <limits>
#include <random>

#include <Eigen/Eigen>
#include <Eigen/LU>

#include "eigen-matchers.hpp"

namespace ko   = koqkatoo::ocp;
namespace stdx = std::experimental;
using guanaqo::as_eigen;
using guanaqo::as_span;
using koqkatoo::index_t;
using koqkatoo::real_t;
using koqkatoo::RealMatrixView;
namespace sp = guanaqo::linalg::sparsity;

const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.9));

struct OCPCyclic : testing::TestWithParam<std::tuple<index_t, index_t>> {};

void solve_cyclic(const koqkatoo::ocp::LinearOCPStorage &ocp, real_t S,
                  std::span<const real_t> Σ, std::span<const bool> J,
                  std::span<const real_t> grad, std::span<const real_t> x,
                  std::span<const real_t> λ, std::span<const real_t> b,
                  std::span<real_t> Mxb, std::span<real_t> Mᵀλ,
                  std::span<real_t> d, std::span<real_t> Δλ,
                  std::span<real_t> MᵀΔλ);

TEST_P(OCPCyclic, solve) {
    auto [N_horiz, nx] = GetParam();
    using VectorXreal  = Eigen::VectorX<real_t>;
    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp({.N_horiz = N_horiz, //
                                        .nx      = nx,
                                        .nu      = 1,
                                        .ny      = 13,
                                        .ny_N    = 3},
                                       12345);

    // Instantiate the OCP KKT solver.
    index_t n_var = ocp.num_variables(), n_constr = ocp.num_constraints(),
            n_dyn_constr = ocp.num_dynamics_constraints();

    // Generate some random optimization solver data.
    Eigen::VectorX<bool> J(n_constr),  // Active set.
        J0(n_constr), J1(n_constr);    // Active set for initialization.
    VectorXreal Σ(n_constr),           // ALM penalty factors
        ŷ(n_constr);                   //  & corresponding Lagrange multipliers.
    VectorXreal x(n_var), grad(n_var); // Current iterate and cost gradient.
    VectorXreal b(n_dyn_constr),       // Dynamics constraints right-hand side
        λ(n_dyn_constr);               //  & corresponding Lagrange multipliers.

    real_t S = std::exp2(nrml(rng)); // primal regularization
    std::ranges::generate(J, [&] { return bernoulli(rng); });
    std::ranges::generate(J0, [&] { return bernoulli(rng); });
    std::ranges::generate(J1, [&] { return bernoulli(rng); });
    std::ranges::generate(Σ, [&] { return std::exp2(nrml(rng)); });
    std::ranges::generate(ŷ, [&] { return nrml(rng); });
    std::ranges::generate(x, [&] { return nrml(rng); });
    std::ranges::generate(grad, [&] { return nrml(rng); });
    std::ranges::generate(b, [&] { return nrml(rng); });
    std::ranges::generate(λ, [&] { return nrml(rng); });

    ŷ.setZero(); // TODO
    λ.setZero(); // TODO

    // Build quadratic program and standard KKT system for the OCP.
    auto qp  = ko::LinearOCPSparseQP::build(ocp);
    auto kkt = qp.build_kkt(S, as_span(Σ), as_span(J));
    // Convert to dense matrices to compare using Eigen.
    sp::SparsityConverter<sp::Sparsity, sp::Dense> conv_Q{qp.Q_sparsity}; //   Q
    auto qp_Q_values = conv_Q.convert_values_copy(std::span{qp.Q_values});
    RealMatrixView qp_Q{{
        .data = qp_Q_values.data(),
        .rows = static_cast<index_t>(conv_Q.get_sparsity().rows),
        .cols = static_cast<index_t>(conv_Q.get_sparsity().cols),
    }};
    sp::SparsityConverter<sp::Sparsity, sp::Dense> conv_A{qp.A_sparsity}; //   A
    auto qp_A_values = conv_A.convert_values_copy(std::span{qp.A_values});
    RealMatrixView qp_A{{
        .data = qp_A_values.data(),
        .rows = static_cast<index_t>(conv_A.get_sparsity().rows),
        .cols = static_cast<index_t>(conv_A.get_sparsity().cols),
    }};
    auto Q = as_eigen(qp_Q);
    auto G = as_eigen(qp_A.bottom_rows(n_constr));
    auto M = as_eigen(qp_A.top_rows(n_dyn_constr));
    sp::SparsityConverter<sp::Sparsity, sp::Dense> conv_K{kkt.sparsity}; //    K
    auto kkt_values = conv_K.convert_values_copy(std::span{kkt.values});
    RealMatrixView K{{
        .data = kkt_values.data(),
        .rows = static_cast<index_t>(conv_K.get_sparsity().rows),
        .cols = static_cast<index_t>(conv_K.get_sparsity().cols),
    }};

    {
        auto [N, nx, nu, ny, ny_N] = ocp.dim;

        Eigen::MatrixX<real_t> H = Q, GsqrtΣ(N * ny + ny_N, N * (nx + nu) + nx),
                               Ψ((N + 1) * nx, (N + 1) * nx);
        GsqrtΣ.noalias() = J.select(Σ.cwiseSqrt(), 0).asDiagonal() * G;
        H.selfadjointView<Eigen::Lower>().rankUpdate(GsqrtΣ.transpose());
        H.diagonal().array() += 1 / S;
        auto LH                     = H.selfadjointView<Eigen::Lower>().llt();
        Eigen::MatrixX<real_t> MLHT = M;
        LH.matrixL().solveInPlace(MLHT.transpose());
        Ψ.setZero();
        Ψ.selfadjointView<Eigen::Lower>().rankUpdate(MLHT);
        guanaqo::print_python(std::cout << "Eigen 1\n",
                              guanaqo::as_view(Ψ.block(nx, nx, nx, nx)));
    }

    VectorXreal Mxb(n_dyn_constr), Mᵀλ(n_var), d(n_var), Δλ(n_dyn_constr),
        MᵀΔλ(n_var);
    solve_cyclic(ocp, S, as_span(Σ), as_span(J), as_span(x), as_span(grad),
                 as_span(λ), as_span(b), as_span(Mxb), as_span(Mᵀλ), as_span(d),
                 as_span(Δλ), as_span(MᵀΔλ));

    // Solve the full KKT system using Eigen (LU because indefinite).
    VectorXreal kkt_rhs_ref    = VectorXreal::Zero(K.rows), kkt_sol_ref(K.rows);
    VectorXreal Gᵀŷ_ref        = G.transpose() * ŷ;
    VectorXreal Mᵀλ_ref        = M.transpose() * λ;
    VectorXreal Mxb_ref        = M * x - b;
    kkt_rhs_ref.topRows(n_var) = -grad - Mᵀλ_ref - Gᵀŷ_ref;
    kkt_rhs_ref.bottomRows(n_dyn_constr) = -Mxb_ref;
    auto luK                             = as_eigen(K).fullPivLu();
    ASSERT_TRUE(luK.isInvertible());
    kkt_sol_ref          = luK.solve(kkt_rhs_ref);
    auto d_ref           = kkt_sol_ref.topRows(n_var);
    auto Δλ_ref          = kkt_sol_ref.bottomRows(n_dyn_constr);
    VectorXreal MᵀΔλ_ref = M.transpose() * Δλ_ref;

    // Compare the koqkatoo OCP solution to the Eigen reference solution.
    std::cout << "ε κ(K) = " << guanaqo::float_to_str(ε / luK.rcond())
              << std::endl;
    EXPECT_THAT(Mxb, EigenAlmostEqual(Mxb_ref, ε / luK.rcond()));
    EXPECT_THAT(Δλ, EigenAlmostEqual(Δλ_ref, ε / luK.rcond()));
    SUCCEED(); // TODO
    return;
    EXPECT_THAT(Mᵀλ, EigenAlmostEqual(Mᵀλ_ref, ε / luK.rcond()));
    EXPECT_THAT(d, EigenAlmostEqual(d_ref, ε / luK.rcond()));
    EXPECT_THAT(MᵀΔλ, EigenAlmostEqual(MᵀΔλ_ref, ε / luK.rcond()));
}

INSTANTIATE_TEST_SUITE_P(OCPCyclic, OCPCyclic,
                         testing::Combine( //
                             testing::Values<index_t>(7, 15, 31, 63),
                             testing::Values<index_t>(3)));
