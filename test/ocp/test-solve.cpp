#include <gtest/gtest.h>

#include <koqkatoo/matrix-view.hpp>
#include <koqkatoo/ocp/conversion.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>

#include <guanaqo/eigen/span.hpp>
#include <guanaqo/eigen/view.hpp>
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

TEST(OCP, solve) {
    using VectorXreal = Eigen::VectorX<real_t>;
    using simd_abi    = stdx::simd_abi::deduce_t<real_t, 4>;
    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp({.N_horiz = 11, //
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
    s.mat_vec_transpose_dynamics_constr(λ_strided, Mᵀλ_strided);
    s.mat_vec_transpose_constr(ŷ_strided, Gᵀŷ_strided);
    s.residual_dynamics_constr(x_strided, b_strided, Mxb_strided);
    // Perform the factorization of the KKT system (with wrong active set).
    s.factor(S, Σ_strided, J0_strided);
    // Update factorization to correct active set.
    s.updowndate(Σ_strided, J0_strided, J1_strided, nullptr);
    s.updowndate(Σ_strided, J1_strided, J_strided, nullptr);
    // Solve the KKT system.
    s.solve(grad_strided, Mᵀλ_strided, Gᵀŷ_strided, Mxb_strided, //
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
    kkt_rhs_ref.topRows(n_var) = -(grad + Gᵀŷ_ref + Mᵀλ_ref);
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
}
