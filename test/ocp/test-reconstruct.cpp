#include <gtest/gtest.h>

#include <koqkatoo/matrix-view.hpp>
#include <koqkatoo/ocp/conversion.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>

#include <guanaqo/eigen/span.hpp>
#include <guanaqo/eigen/view.hpp>
#include <guanaqo/linalg/sparsity-conversions.hpp>
#include <guanaqo/print.hpp>
#include <guanaqo/timed-cpu.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <ostream>
#include <print>
#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Eigen>

using namespace koqkatoo::ocp;
using guanaqo::as_eigen;
using guanaqo::as_span;
using guanaqo::as_view;
using koqkatoo::index_t;
using koqkatoo::real_t;
using koqkatoo::RealMatrixView;
namespace sp = guanaqo::linalg::sparsity;
using namespace std::chrono_literals;

#if !(defined(__clang__) || defined(__aarch64__))
namespace std {
static ostream &operator<<(ostream &os, float128_t f) {
    return os << static_cast<long double>(f);
}
} // namespace std
[[maybe_unused]] static void PrintTo(const std::float128_t f,
                                     std::ostream *os) {
    *os << f;
}
#endif

TEST(OCP, reconstructRandom) {
    // Generate some random OCP matrices
    auto ocp = generate_random_ocp({.N_horiz = 13, //
                                    .nx      = 5,
                                    .nu      = 3,
                                    .ny      = 7,
                                    .ny_N    = 5},
                                   12345);

    using simd_abi             = stdx::simd_abi::deduce_t<real_t, 8>;
    Solver<simd_abi> s         = ocp;
    auto [N, nx, nu, ny, ny_N] = ocp.dim;
    index_t n_var = N * (nx + nu) + nx, n_constr = N * ny + ny_N;
    index_t n_dyn_constr = (N + 1) * nx;
    Eigen::VectorX<real_t> Σ(n_constr);
    Eigen::VectorX<bool> J(n_constr);
    Eigen::VectorX<real_t> b(n_var), r(n_dyn_constr), λ;

    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};
    std::ranges::generate(Σ, [&] { return std::exp2(nrml(rng)); });
    std::ranges::generate(J, [&] { return bernoulli(rng); });
    std::ranges::generate(b, [&] { return nrml(rng); });
    std::ranges::generate(r, [&] { return nrml(rng); });
    auto Σ_strided = s.storage.initialize_constraints(as_span(Σ));
    auto J_strided = s.storage.initialize_active_set(as_span(J));
    auto b_strided = s.storage.initialize_variables(as_span(b));
    decltype(b_strided) x_strided;
    using guanaqo::TimingsCPU;
    TimingsCPU t_schur, t_chol_H, t_solve_x, t_prep_Ψ, t_chol_Ψ, t_solve_λ;
    for (index_t i = 0; i < 25; ++i) {
        guanaqo::timed(t_schur,
                       [&] { s.schur_complement_H(Σ_strided, J_strided); });
        std::this_thread::sleep_for(10ms);
        guanaqo::timed(t_chol_H, [&] { s.cholesky_H(); });
        std::this_thread::sleep_for(10ms);
        x_strided = b_strided;
        guanaqo::timed(t_solve_x, [&] { s.solve_H(x_strided); });
        std::this_thread::sleep_for(10ms);
        guanaqo::timed(t_prep_Ψ, [&] { s.prepare_Ψ(); });
        std::this_thread::sleep_for(10ms);
        guanaqo::timed(t_chol_Ψ, // TODO
                       [&] { s.factor(1e100, Σ_strided, J_strided); });
        std::this_thread::sleep_for(10ms);
        λ = r;
        guanaqo::timed(t_solve_λ, [&] { s.solve_Ψ_scalar(as_span(λ)); });
        std::this_thread::sleep_for(10ms);
    }
    Eigen::VectorX<real_t> x(n_var);
    s.storage.restore_variables(x_strided, as_span(x));

    // Print timings
    using millis_f64 = std::chrono::duration<double, std::milli>;
    std::cout << "Schur:      " << millis_f64(t_schur.wall_time).count()
              << " ms (wall) ─ " << millis_f64(t_schur.cpu_time).count()
              << " ms (CPU) ─ "
              << 100 * millis_f64(t_schur.cpu_time).count() /
                     millis_f64(t_schur.wall_time).count()
              << "%\n";
    std::cout << "Cholesky H: " << millis_f64(t_chol_H.wall_time).count()
              << " ms (wall) ─ " << millis_f64(t_chol_H.cpu_time).count()
              << " ms (CPU) ─ "
              << 100 * millis_f64(t_chol_H.cpu_time).count() /
                     millis_f64(t_chol_H.wall_time).count()
              << "%\n";
    std::cout << "Solve x:    " << millis_f64(t_solve_x.wall_time).count()
              << " ms (wall) ─ " << millis_f64(t_solve_x.cpu_time).count()
              << " ms (CPU) ─ "
              << 100 * millis_f64(t_solve_x.cpu_time).count() /
                     millis_f64(t_solve_x.wall_time).count()
              << "%\n";
    std::cout << "Prepare Ψ:  " << millis_f64(t_prep_Ψ.wall_time).count()
              << " ms (wall) ─ " << millis_f64(t_prep_Ψ.cpu_time).count()
              << " ms (CPU) ─ "
              << 100 * millis_f64(t_prep_Ψ.cpu_time).count() /
                     millis_f64(t_prep_Ψ.wall_time).count()
              << "%\n";
    std::cout << "Cholesky Ψ: " << millis_f64(t_chol_Ψ.wall_time).count()
              << " ms (wall) ─ " << millis_f64(t_chol_Ψ.cpu_time).count()
              << " ms (CPU) ─ "
              << 100 * millis_f64(t_chol_Ψ.cpu_time).count() /
                     millis_f64(t_chol_Ψ.wall_time).count()
              << "%\n";
    std::cout << "Solve λ:    " << millis_f64(t_solve_λ.wall_time).count()
              << " ms (wall) ─ " << millis_f64(t_solve_λ.cpu_time).count()
              << " ms (CPU) ─ "
              << 100 * millis_f64(t_solve_λ.cpu_time).count() /
                     millis_f64(t_solve_λ.wall_time).count()
              << "%\n";

    // Build quadratic program
    auto qp = LinearOCPSparseQP::build(ocp);
    // Convert to dense matrices to compare with Eigen
    sp::SparsityConverter<sp::Sparsity, sp::Dense> conv_Q{qp.Q_sparsity};
    auto qp_Q_values = conv_Q.convert_values_copy(std::span{qp.Q_values});
    RealMatrixView qp_Q{{
        .data = qp_Q_values.data(),
        .rows = static_cast<index_t>(conv_Q.get_sparsity().rows),
        .cols = static_cast<index_t>(conv_Q.get_sparsity().cols),
    }};
    sp::SparsityConverter<sp::Sparsity, sp::Dense> conv_A{qp.A_sparsity};
    auto qp_A_values = conv_A.convert_values_copy(std::span{qp.A_values});
    RealMatrixView qp_A{{
        .data = qp_A_values.data(),
        .rows = static_cast<index_t>(conv_A.get_sparsity().rows),
        .cols = static_cast<index_t>(conv_A.get_sparsity().cols),
    }};
    auto A_ineq = qp_A.bottom_rows(n_constr);
    auto A_eq   = qp_A.top_rows(n_dyn_constr);

    // Compare Eigen solution
    using mat   = Eigen::MatrixX<real_t>;
    using vec   = Eigen::VectorX<real_t>;
    mat H_schur = as_eigen(qp_Q) + as_eigen(A_ineq).transpose() *
                                       J.select(Σ, 0).asDiagonal() *
                                       as_eigen(A_ineq);
    auto Hllt      = H_schur.selfadjointView<Eigen::Lower>().llt();
    real_t rcond_H = H_schur.selfadjointView<Eigen::Lower>().ldlt().rcond();
    vec x_eigen    = Hllt.solve(b);

    const auto eps = 50 * std::numeric_limits<real_t>::epsilon();
    std::cout << "cond H: " << 1 / rcond_H << "\n";
    EXPECT_LE((x - x_eigen).lpNorm<Eigen::Infinity>(), eps / rcond_H);

    // Build matrix ψ using Eigen
    mat Ψ_half = Hllt.matrixL().solve(as_eigen(A_eq).transpose());
    mat Ψ      = mat::Zero(n_dyn_constr, n_dyn_constr);
    Ψ.selfadjointView<Eigen::Lower>().rankUpdate(Ψ_half.transpose());

    auto Ψllt      = Ψ.selfadjointView<Eigen::Lower>().llt();
    real_t rcond_Ψ = Ψ.selfadjointView<Eigen::Lower>().ldlt().rcond();
    vec λ_eigen    = Ψllt.solve(r);

    std::cout << "cond Ψ: " << 1 / rcond_Ψ << "\n";
    EXPECT_LE((λ - λ_eigen).lpNorm<Eigen::Infinity>(), eps / rcond_Ψ);

    // Factor matrix ψ using Eigen
    mat LΨ = Ψllt.matrixL();

    // Extract matrix H̃
    s.schur_complement_H(Σ_strided, J_strided);
    mat H_rec = mat::Zero(n_var, n_var);
    for (index_t i = 0; i < N; ++i)
        for (index_t c = 0; c < nx + nu; ++c)
            for (index_t r = c; r < nx + nu; ++r)
                H_rec(i * (nx + nu) + r, i * (nx + nu) + c) = s.LH()(i, r, c);
    for (index_t c = 0; c < nx; ++c)
        for (index_t r = c; r < nx; ++r)
            H_rec(N * (nx + nu) + r, N * (nx + nu) + c) = s.LH()(N, r, c);

    H_rec.triangularView<Eigen::Upper>() =
        H_rec.triangularView<Eigen::Lower>().transpose();
    mat err_H = H_schur - H_rec;
    EXPECT_LE((H_rec - H_schur).lpNorm<Eigen::Infinity>() /
                  H_schur.lpNorm<Eigen::Infinity>(),
              eps);

    // Extract matrix chol(H)
    s.cholesky_H();
    mat LH_rec = mat::Zero(n_var, n_var);
    for (index_t i = 0; i < N; ++i)
        for (index_t c = 0; c < nx + nu; ++c)
            for (index_t r = c; r < nx + nu; ++r)
                LH_rec(i * (nx + nu) + r, i * (nx + nu) + c) = s.LH()(i, r, c);
    for (index_t c = 0; c < nx; ++c)
        for (index_t r = c; r < nx; ++r)
            LH_rec(N * (nx + nu) + r, N * (nx + nu) + c) = s.LH()(N, r, c);

    mat LH_schur = Hllt.matrixL();
    mat err_LH   = LH_schur - LH_rec;
    EXPECT_LE((LH_rec - LH_schur).lpNorm<Eigen::Infinity>() /
                  LH_schur.lpNorm<Eigen::Infinity>(),
              eps);

    // Extract matrix WᵀW
    s.prepare_Ψ();
    mat Pᵀ = mat::Zero(nx + nu, nx);
    Pᵀ.leftCols(nx).setIdentity();
    mat W_eig = LH_schur.block(0, 0, nx + nu, nx + nu)
                    .triangularView<Eigen::Lower>()
                    .solve(Pᵀ);
    mat WᵀW = W_eig.transpose() * W_eig;
    mat WᵀW_rec(nx, nx);
    for (index_t c = 0; c < nx; ++c)
        for (index_t r = c; r < nx; ++r)
            WᵀW_rec(r, c) = s.WWᵀ()(0, r, c);
    WᵀW_rec.triangularView<Eigen::Upper>() =
        WᵀW_rec.triangularView<Eigen::Lower>().transpose();

    mat err_WᵀW = WᵀW - WᵀW_rec;
    EXPECT_LE((WᵀW_rec - WᵀW).lpNorm<Eigen::Infinity>() /
                  WᵀW.lpNorm<Eigen::Infinity>(),
              eps / rcond_H);

    // Extract matrix ψ
    mat Ψ_rec = mat::Zero(n_dyn_constr, n_dyn_constr);
    for (index_t i = 0; i < N + 1; ++i) {
        for (index_t c = 0; c < nx; ++c) {
            for (index_t r = c; r < nx; ++r) {
                Ψ_rec(i * nx + r, i * nx + c) = s.WWᵀ()(i, r, c);
                if (i > 0)
                    Ψ_rec(i * nx + r, i * nx + c) += s.VVᵀ()(i - 1, r, c);
            }
            for (index_t r = 0; r < nx; ++r) {
                if (i < N)
                    Ψ_rec(i * nx + r + nx, i * nx + c) = s.VWᵀ()(i, r, c);
            }
        }
    }

    mat err_Ψ = Ψ - Ψ_rec;
    EXPECT_LE((Ψ_rec - Ψ).lpNorm<Eigen::Infinity>() /
                  Ψ.lpNorm<Eigen::Infinity>(),
              eps / rcond_H);

    // Compare matrix-vector products
    vec dyn_b(n_dyn_constr);
    std::ranges::generate(dyn_b, [&] { return nrml(rng); });
    auto dyn_b_strided =
        s.storage.initialize_dynamics_constraints(as_span(dyn_b));
    auto Mxb_strided = dyn_b_strided;
    s.residual_dynamics_constr(x_strided, dyn_b_strided, Mxb_strided);
    vec Mxb(n_dyn_constr);
    s.storage.restore_dynamics_constraints(Mxb_strided, as_span(Mxb));

    vec Mxb_ref = as_eigen(A_eq) * x - dyn_b;
    EXPECT_LE((Mxb - Mxb_ref).lpNorm<Eigen::Infinity>() /
                  Mxb_ref.lpNorm<Eigen::Infinity>(),
              eps);

    vec Mᵀb_ref = as_eigen(A_eq).transpose() * dyn_b;
    vec Mᵀb(n_var);
    auto Mᵀb_strided = x_strided;
    s.mat_vec_transpose_dynamics_constr(dyn_b_strided, Mᵀb_strided);
    s.storage.restore_variables(Mᵀb_strided, as_span(Mᵀb));

    EXPECT_LE((Mᵀb - Mᵀb_ref).lpNorm<Eigen::Infinity>() /
                  Mᵀb_ref.lpNorm<Eigen::Infinity>(),
              eps);
}
