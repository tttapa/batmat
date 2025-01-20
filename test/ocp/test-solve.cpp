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

struct OCP : testing::TestWithParam<index_t> {};

TEST_P(OCP, solve) {
    KOQKATOO_OMP_IF(omp_set_num_threads(1));
    using VectorXreal = Eigen::VectorX<real_t>;
    using simd_abi    = stdx::simd_abi::scalar;
    // using simd_abi    = stdx::simd_abi::deduce_t<real_t, 4>;
    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp({.N_horiz = GetParam(), //
                                        .nx      = 1,
                                        .nu      = 1,
                                        .ny      = 5,
                                        .ny_N    = 4},
                                       12345);

    // Instantiate the OCP KKT solver.
    ko::Solver<simd_abi> s = ocp;
    s.settings.preferred_backend =
        koqkatoo::linalg::compact::PreferredBackend::Reference; // TODO
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

    // Convert this data into the compact format used by the OCP solver.
    // (Normally, you would only do this once, and re-use the format in the
    //  optimization solver, only converting back at the very end to return a
    //  solution to the user.)
    auto J_strided    = s.storage.initialize_active_set(as_span(J));
    auto J0_strided   = s.storage.initialize_active_set(as_span(J0));
    auto J1_strided   = s.storage.initialize_active_set(as_span(J1));
    auto Σ_strided    = s.storage.initialize_constraints(as_span(Σ));
    auto ŷ_strided    = s.storage.initialize_constraints(as_span(ŷ));
    auto x_strided    = s.storage.initialize_variables(as_span(x));
    auto grad_strided = s.storage.initialize_variables(as_span(grad));
    auto b_strided    = s.storage.initialize_dynamics_constraints(as_span(b));
    auto λ_strided    = s.storage.initialize_dynamics_constraints(as_span(λ));

    // Prepare storage for some intermediate quantities.
    auto Gᵀŷ_strided  = s.storage.initialize_variables();
    auto Mᵀλ_strided  = s.storage.initialize_variables();
    auto MᵀΔλ_strided = s.storage.initialize_variables();
    auto d_strided    = s.storage.initialize_variables();
    auto Δλ_strided   = s.storage.initialize_dynamics_constraints();
    auto Mxb_strided  = s.storage.initialize_dynamics_constraints();

    // Actually solve the KKT system:
    // ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

    // Compute the necessary matrix-vector products.
    s.mat_vec_transpose_constr(ŷ_strided, Gᵀŷ_strided);
    s.mat_vec_transpose_dynamics_constr(λ_strided, Mᵀλ_strided);
    s.residual_dynamics_constr(x_strided, b_strided, Mxb_strided);
    // Perform the factorization of the KKT system (with wrong active set).
    // s.factor_new(S, Σ_strided, J0_strided);
    // // Update factorization to correct active set.
    // s.updowndate_new(Σ_strided, J0_strided, J1_strided);
    // s.updowndate_new(Σ_strided, J1_strided, J_strided);
    // Solve the KKT system.
    s.factor_new(S, Σ_strided, J_strided);                          // TODO
    s.factor_rev(S, Σ_strided, J_strided);                          // TODO
    s.solve_rev(grad_strided, Mᵀλ_strided, Gᵀŷ_strided, Mxb_strided, //
                d_strided, Δλ_strided, MᵀΔλ_strided);
    // d:    Newton step for x
    // Δλ:   negative Newton step for λ
    // MᵀΔλ: by-product that can be re-used to compute Mᵀ(λ - Δλ)

    // ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    // Restore the solution to normal vectors to verify them.
    VectorXreal d(n_var), Δλ(n_dyn_constr), MᵀΔλ(n_var), Gᵀŷ(n_var);
    s.storage.restore_variables(d_strided, as_span(d));
    s.storage.restore_dynamics_constraints(Δλ_strided, as_span(Δλ));
    s.storage.restore_variables(MᵀΔλ_strided, as_span(MᵀΔλ));
    s.storage.restore_variables(Gᵀŷ_strided, as_span(Gᵀŷ));

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
    auto G = as_eigen(qp_A.bottom_rows(n_constr));
    auto M = as_eigen(qp_A.top_rows(n_dyn_constr));
    sp::SparsityConverter<sp::Sparsity, sp::Dense> conv_K{kkt.sparsity}; //    K
    auto kkt_values = conv_K.convert_values_copy(std::span{kkt.values});
    RealMatrixView K{{
        .data = kkt_values.data(),
        .rows = static_cast<index_t>(conv_K.get_sparsity().rows),
        .cols = static_cast<index_t>(conv_K.get_sparsity().cols),
    }};

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
    EXPECT_THAT(d, EigenAlmostEqual(d_ref, ε / luK.rcond()));
    EXPECT_THAT(Δλ, EigenAlmostEqual(Δλ_ref, ε / luK.rcond()));
    EXPECT_THAT(Gᵀŷ, EigenAlmostEqual(Gᵀŷ_ref, ε / luK.rcond()));
    EXPECT_THAT(MᵀΔλ, EigenAlmostEqual(MᵀΔλ_ref, ε / luK.rcond()));
    decltype(s)::compact_blas::xadd_copy(x_strided, x_strided, d_strided);
    s.residual_dynamics_constr(x_strided, b_strided, Mxb_strided);
    const auto nrm_Mxb = decltype(s)::compact_blas::xnrminf(Mxb_strided);
    std::cout << "‖Mx-b‖ = " << guanaqo::float_to_str(nrm_Mxb) << std::endl;
    EXPECT_LE(nrm_Mxb, ε / luK.rcond());
}

TEST_P(OCP, recompute) {
    using VectorXreal = Eigen::VectorX<real_t>;
    using simd_abi    = stdx::simd_abi::deduce_t<real_t, 4>;
    std::mt19937 rng{654321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp({.N_horiz = GetParam(), //
                                        .nx      = 5,
                                        .nu      = 3,
                                        .ny      = 7,
                                        .ny_N    = 4},
                                       12345);

    // Instantiate the OCP KKT solver.
    ko::Solver<simd_abi> s = ocp;
    s.settings.preferred_backend =
        koqkatoo::linalg::compact::PreferredBackend::MKLScalarBatched;
    index_t n_var = ocp.num_variables(), n_constr = ocp.num_constraints(),
            n_dyn_constr = ocp.num_dynamics_constraints();

    // Generate some random optimization solver data.
    VectorXreal ŷ(n_constr), Ax(n_constr);
    VectorXreal x0(n_var), x(n_var), q(n_var);
    VectorXreal b(n_dyn_constr), // Dynamics constraints right-hand side
        λ(n_dyn_constr),         //  & corresponding Lagrange multipliers.
        Mx(n_dyn_constr);

    real_t S = std::exp2(nrml(rng)); // primal regularization
    std::ranges::generate(ŷ, [&] { return nrml(rng); });
    std::ranges::generate(x, [&] { return nrml(rng); });
    std::ranges::generate(x0, [&] { return nrml(rng); });
    std::ranges::generate(q, [&] { return nrml(rng); });
    std::ranges::generate(b, [&] { return nrml(rng); });
    std::ranges::generate(λ, [&] { return nrml(rng); });

    // Convert this data into the compact format used by the OCP solver.
    // (Normally, you would only do this once, and re-use the format in the
    //  optimization solver, only converting back at the very end to return a
    //  solution to the user.)
    auto ŷ_strided  = s.storage.initialize_constraints(as_span(ŷ));
    auto x_strided  = s.storage.initialize_variables(as_span(x));
    auto x0_strided = s.storage.initialize_variables(as_span(x0));
    auto q_strided  = s.storage.initialize_variables(as_span(q));
    auto b_strided  = s.storage.initialize_dynamics_constraints(as_span(b));
    auto λ_strided  = s.storage.initialize_dynamics_constraints(as_span(λ));

    // Prepare storage for some intermediate quantities.
    auto grad_inner_strided = s.storage.initialize_variables();
    auto Mᵀλ_inner_strided  = s.storage.initialize_variables();
    auto grad_outer_strided = s.storage.initialize_variables();
    auto Aᵀŷ_outer_strided  = s.storage.initialize_variables();
    auto Mᵀλ_outer_strided  = s.storage.initialize_variables();
    auto Mxb_strided        = s.storage.initialize_dynamics_constraints();
    auto Ax_inner_strided   = s.storage.initialize_constraints();
    auto Ax_outer_strided   = s.storage.initialize_constraints();

    // Actually compute the matrix-vector products:
    // ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

    // Compute the necessary matrix-vector products.
    s.recompute_inner(S, x0_strided, x_strided, λ_strided, q_strided,
                      grad_inner_strided, Ax_inner_strided, Mᵀλ_inner_strided);
    real_t inf_nrm_al_grad = s.recompute_outer(
        x_strided, ŷ_strided, λ_strided, q_strided, grad_outer_strided,
        Ax_outer_strided, Aᵀŷ_outer_strided, Mᵀλ_outer_strided);
    s.residual_dynamics_constr(x_strided, b_strided, Mxb_strided);

    // ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    // Restore the solution to normal vectors to verify them.
    VectorXreal grad_inner(n_var), Mᵀλ_inner(n_var), Ax_inner(n_constr),
        grad_outer(n_var), Mᵀλ_outer(n_var), Aᵀŷ_outer(n_var),
        Mxb(n_dyn_constr);
    s.storage.restore_variables(grad_inner_strided, as_span(grad_inner));
    s.storage.restore_variables(Mᵀλ_inner_strided, as_span(Mᵀλ_inner));
    s.storage.restore_variables(grad_outer_strided, as_span(grad_outer));
    s.storage.restore_variables(Mᵀλ_outer_strided, as_span(Mᵀλ_outer));
    s.storage.restore_variables(Aᵀŷ_outer_strided, as_span(Aᵀŷ_outer));
    s.storage.restore_dynamics_constraints(Mxb_strided, as_span(Mxb));

    // Build quadratic program and standard KKT system for the OCP.
    auto qp = ko::LinearOCPSparseQP::build(ocp);
    // Convert to dense matrices to compare using Eigen.
    ASSERT_EQ(qp.Q_sparsity.symmetry, sp::Symmetry::Lower);
    auto Q = as_eigen(qp.Q_sparsity, std::span{qp.Q_values})
                 .selfadjointView<Eigen::Lower>();
    auto A = as_eigen(qp.A_sparsity, std::span{qp.A_values});
    auto G = A.bottomRows(n_constr);
    auto M = A.topRows(n_dyn_constr);

    // Compute matrix-vector products using Eigen
    VectorXreal grad_inner_ref = Q * x + (x - x0) / S + q;
    VectorXreal Gᵀŷ_ref        = G.transpose() * ŷ;
    VectorXreal Mᵀλ_ref        = M.transpose() * λ;
    VectorXreal Mxb_ref        = M * x - b;
    VectorXreal grad_outer_ref = Q * x + q;
    real_t inf_nrm_al_grad_ref =
        (grad_outer_ref + Gᵀŷ_ref + Mᵀλ_ref).lpNorm<Eigen::Infinity>();

    EXPECT_THAT(grad_inner, EigenAlmostEqual(grad_inner_ref, 10 * ε));
    EXPECT_THAT(Mᵀλ_inner, EigenAlmostEqual(Mᵀλ_ref, 10 * ε));
    EXPECT_THAT(grad_outer, EigenAlmostEqual(grad_outer_ref, 10 * ε));
    EXPECT_THAT(Aᵀŷ_outer, EigenAlmostEqual(Gᵀŷ_ref, 10 * ε));
    EXPECT_THAT(Mᵀλ_outer, EigenAlmostEqual(Mᵀλ_ref, 10 * ε));
    EXPECT_THAT(Mxb, EigenAlmostEqual(Mxb_ref, 10 * ε));
    EXPECT_NEAR(inf_nrm_al_grad, inf_nrm_al_grad_ref, 10 * ε);
}

INSTANTIATE_TEST_SUITE_P(OCP, OCP, testing::Range<index_t>(1, 21));
