#include <benchmark/benchmark.h>

#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/matrix-view.hpp>
#include <koqkatoo/ocp/conversion.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/thread-pool.hpp>
#include <koqkatoo/trace.hpp>

#include <guanaqo/eigen/span.hpp>
#include <guanaqo/eigen/view.hpp>
#include <guanaqo/print.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>

#include <Eigen/Eigen>

#if 0
constexpr int n_threads = 8;
constexpr int n_horiz   = 3;
constexpr int n_states  = 2;
constexpr int n_inputs  = 2;
#else
constexpr int n_threads = 4;
constexpr int n_horiz   = 63;
constexpr int n_states  = 40;
constexpr int n_inputs  = 30;
#endif

namespace ko   = koqkatoo::ocp;
namespace stdx = std::experimental;
using guanaqo::as_eigen;
using guanaqo::as_span;
using koqkatoo::index_t;
using koqkatoo::real_t;
using koqkatoo::RealMatrixView;
namespace sp = guanaqo::linalg::sparsity;

#define USE_GEMMT_INEQ_SCHUR 1

static bool init_done = false;

void solve_riccati(koqkatoo::ocp::LinearOCPStorage &ocp, real_t S, auto Σ,
                   auto J, auto q, auto b, auto xu) {
    auto [N, nx, nu, ny, ny_N] = ocp.dim;
    using koqkatoo::linalg::compact::BatchedMatrix;
    using scal_blas =
        koqkatoo::linalg::compact::CompactBLAS<stdx::simd_abi::scalar>;
    // TODO: probably shouldn't be static, but I'm lazy
    static BatchedMatrix<real_t, index_t, scal_blas::scalar_simd_stride_t> LP{
        {.depth = N + 1, .rows = nx + nu, .cols = nx + nu}},
        LF{{.rows = nx, .cols = nx + nu}},
        BA{{.depth = N, .rows = nx, .cols = nx + nu}},
        DC{{.depth = N, .rows = ny, .cols = nx + nu}},
#if USE_GEMMT_INEQ_SCHUR
        CN{{.rows = ny_N, .cols = nx}}, ΣCN{{.rows = ny_N, .cols = nx}},
        ΣDC{{.depth = N, .rows = ny, .cols = nx + nu}},
#endif
        RSQ{{.depth = N, .rows = nx + nu, .cols = nx + nu}};
    static bool initialized = false;
    if (!std::exchange(initialized, true)) {
        for (index_t k = 0; k < N; ++k) {
            BA(k).left_cols(nu)         = ocp.B(k);
            BA(k).right_cols(nx)        = ocp.A(k);
            RSQ(k).top_left(nu, nu)     = ocp.R(k);
            RSQ(k).bottom_left(nx, nu)  = ocp.S_trans(k);
            RSQ(k).bottom_right(nx, nx) = ocp.Q(k);
        }
    }
    {
        KOQKATOO_TRACE("riccati factor", N);
        auto PN = LP.batch(N).bottom_right(nx, nx);
#if USE_GEMMT_INEQ_SCHUR
        auto ocpCN = ocp.C(N);
        index_t nJ = 0;
        for (index_t c = 0; c < nx; ++c) {
            nJ = 0;
            for (index_t r = 0; r < ny_N; ++r)
                if (J(N, r, 0)) {
                    ΣCN(0, nJ, c) = Σ(N, r, 0) * (CN(0, nJ, c) = ocpCN(r, c));
                    ++nJ;
                }
        }
        PN(0) = ocp.Q(N);
        koqkatoo::linalg::xgemmt(
            CblasColMajor, CblasLower, CblasTrans, CblasNoTrans, PN.rows(), nJ,
            real_t{1}, CN.data(), CN.outer_stride(), ΣCN.data(),
            ΣCN.outer_stride(), real_t{1}, PN.data, PN.outer_stride());
#else
        scal_blas::mut_batch_view QN{{.data         = ocp.Q(N).data,
                                      .rows         = ocp.Q(N).rows,
                                      .cols         = ocp.Q(N).cols,
                                      .outer_stride = ocp.Q(N).outer_stride}};
        scal_blas::mut_batch_view CN{{.data         = ocp.C(N).data,
                                      .rows         = ocp.C(N).rows,
                                      .cols         = ocp.C(N).cols,
                                      .outer_stride = ocp.C(N).outer_stride}};
        scal_blas::xsyrk_T_schur_copy(CN.batch(0), Σ.batch(N).top_rows(ny_N),
                                      J.batch(N).top_rows(ny_N), QN.batch(0),
                                      PN.batch(0));
#endif
        PN.add_to_diagonal(1 / S);
        index_t info;
        koqkatoo::linalg::xpotrf("L", PN.rows(), PN.data, PN.outer_stride(),
                                 &info);
        if (info != 0)
            throw std::runtime_error(
                std::format("cholesky fail {} in stage {}", info, N));
    }
    for (index_t k = N; k-- > 0;) {
        KOQKATOO_TRACE("riccati factor", k);
        LF(0)    = BA(k);
        auto Lxx = LP(k + 1).bottom_right(nx, nx);
        koqkatoo::linalg::xtrmm(
            CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
            LF(0).rows, LF(0).cols, real_t(1), Lxx.data, Lxx.outer_stride,
            LF(0).data, LF(0).outer_stride);
        auto ocpCDk = ocp.CD(k);
#if USE_GEMMT_INEQ_SCHUR
        index_t nJ = 0;
        for (index_t c = 0; c < nx; ++c) {
            nJ = 0;
            for (index_t r = 0; r < ny; ++r)
                if (J(k, r, 0)) {
                    ΣDC(k, nJ, nu + c) =
                        Σ(k, r, 0) * (DC(k, nJ, nu + c) = ocpCDk(r, c));
                    ++nJ;
                }
        }
        for (index_t c = 0; c < nu; ++c) {
            nJ = 0;
            for (index_t r = 0; r < ny; ++r)
                if (J(k, r, 0)) {
                    ΣDC(k, nJ, c) =
                        Σ(k, r, 0) * (DC(k, nJ, c) = ocpCDk(r, nx + c));
                    ++nJ;
                }
        }
        LP(k) = RSQ(k);
        koqkatoo::linalg::xgemmt(
            CblasColMajor, CblasLower, CblasTrans, CblasNoTrans, LP(k).rows, nJ,
            real_t{1}, DC(k).data, DC(k).outer_stride, ΣDC(k).data,
            ΣDC(k).outer_stride, real_t{1}, LP(k).data, LP(k).outer_stride);
#else
        DC(k).left_cols(nu)  = ocpCDk.right_cols(nu);
        DC(k).right_cols(nx) = ocpCDk.left_cols(nx);
        scal_blas::xsyrk_T_schur_copy(DC.batch(k), Σ.batch(k), J.batch(k),
                                      RSQ.batch(k), LP.batch(k));
#endif
        LP(k).add_to_diagonal(1 / S);
        koqkatoo::linalg::xsyrk(CblasColMajor, CblasLower, CblasTrans,
                                LP.rows(), LF.rows(), real_t(1), LF.data(),
                                LF.outer_stride(), real_t(1), LP(k).data,
                                LP.outer_stride());
        index_t info;
        koqkatoo::linalg::xpotrf("L", LP.rows(), LP(k).data, LP(k).outer_stride,
                                 &info);
        if (info != 0)
            throw std::runtime_error(
                std::format("cholesky fail {} in stage {}", info, k));
    }
    static BatchedMatrix<real_t, index_t> p{{.depth = N + 1, .rows = nx}},
        Pb{{.depth = N + 1, .rows = nx}};
    p(N) = q(N).top_rows(nx);
    for (index_t k = N; k-- > 0;) {
        KOQKATOO_TRACE("riccati solve bwd", k);
        auto B = BA(k).left_cols(nu), A = BA(k).right_cols(nx);
        auto Lxx = LP(k + 1).bottom_right(nx, nx), Luu = LP(k).top_left(nu, nu),
             Lxu = LP(k).bottom_left(nx, nu);
        auto u   = xu(k).bottom_rows(nu);

        // lₖ = Luu⁻¹ₖ (rₖ + Bₖᵀ (Pₖ₊₁ bₖ₊₁ + pₖ₊₁))
        // ---------------------------------------
        Pb(k + 1) = b(k + 1);
        // Lxxₖ₊₁ᵀ bₖ₊₁
        koqkatoo::linalg::xtrmv(CblasColMajor, CblasLower, CblasTrans,
                                CblasNonUnit, Lxx.rows, Lxx.data,
                                Lxx.outer_stride, Pb(k + 1).data, index_t{1});
        // Lxxₖ₊₁ (Lxxₖ₊₁ᵀ bₖ₊₁)
        koqkatoo::linalg::xtrmv(CblasColMajor, CblasLower, CblasNoTrans,
                                CblasNonUnit, Lxx.rows, Lxx.data,
                                Lxx.outer_stride, Pb(k + 1).data, index_t{1});
        // Lxxₖ₊₁ (Lxxₖ₊₁ᵀ bₖ₊₁) + pₖ₊₁
        Pb(k + 1) += p(k + 1);
        // rₖ + Bₖᵀ (Lxxₖ₊₁ (Lxxₖ₊₁ᵀ bₖ₊₁) + pₖ₊₁)
        u = q(k).bottom_rows(nu);
        koqkatoo::linalg::xgemv(CblasColMajor, CblasTrans, B.rows, B.cols,
                                real_t{1}, B.data, B.outer_stride,
                                Pb(k + 1).data, index_t{1}, real_t{1}, u.data,
                                index_t{1});
        // Luu⁻¹ₖ (rₖ + Bₖᵀ (Lxxₖ₊₁ (Lxxₖ₊₁ᵀ bₖ₊₁) + pₖ₊₁))
        koqkatoo::linalg::xtrsv(CblasColMajor, CblasLower, CblasNoTrans,
                                CblasNonUnit, Luu.rows, Luu.data,
                                Luu.outer_stride, u.data, index_t{1});

        // pₖ = qₖ + Aₖᵀ (Pₖ₊₁ bₖ₊₁ + pₖ₊₁) - Lxuᵀ lₖ
        p(k) = q(k).top_rows(nx);
        // qₖ + Aₖᵀ (Pₖ₊₁ bₖ₊₁ + pₖ₊₁)
        koqkatoo::linalg::xgemv(CblasColMajor, CblasTrans, A.rows, A.cols,
                                real_t{1}, A.data, A.outer_stride,
                                Pb(k + 1).data, index_t{1}, real_t{1},
                                p(k).data, index_t{1});
        // qₖ + Aₖᵀ (Pₖ₊₁ bₖ₊₁ + pₖ₊₁) - Lxuᵀ lₖ
        koqkatoo::linalg::xgemv(CblasColMajor, CblasNoTrans, Lxu.rows, Lxu.cols,
                                real_t{-1}, Lxu.data, Lxu.outer_stride, u.data,
                                index_t{1}, real_t{1}, p(k).data, index_t{1});
    }
    xu(0).top_rows(nx) = b(0);
    for (index_t k = 0; k < N; ++k) {
        KOQKATOO_TRACE("riccati solve fwd", k);
        auto Luu = LP(k).top_left(nu, nu), Lxu = LP(k).bottom_left(nx, nu);
        auto u = xu(k).bottom_rows(nu), x = xu(k).top_rows(nx),
             x_next = xu(k + 1).top_rows(nx);
        koqkatoo::linalg::xgemv(CblasColMajor, CblasTrans, Lxu.rows, Lxu.cols,
                                real_t{-1}, Lxu.data, Lxu.outer_stride, x.data,
                                index_t{1}, real_t{-1}, u.data, index_t{1});
        koqkatoo::linalg::xtrsv(CblasColMajor, CblasLower, CblasTrans,
                                CblasNonUnit, Luu.rows, Luu.data,
                                Luu.outer_stride, u.data, index_t{1});
        x_next  = b(k + 1);
        auto AB = ocp.AB(k);
        koqkatoo::linalg::xgemv(CblasColMajor, CblasNoTrans, AB.rows, AB.cols,
                                real_t{1}, AB.data, AB.outer_stride, xu(k).data,
                                index_t{1}, real_t{1}, x_next.data, index_t{1});
    }
}

void benchmark_riccati(benchmark::State &state) {
    using koqkatoo::linalg::compact::BatchedMatrixView;
    using VectorXreal = Eigen::VectorX<real_t>;
    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp({.N_horiz = n_horiz, //
                                        .nx      = n_states,
                                        .nu      = n_inputs,
                                        .ny      = n_inputs,
                                        .ny_N    = n_inputs},
                                       12345);

    auto [N, nx, nu, ny, ny_N] = ocp.dim;

    ko::Solver<stdx::simd_abi::scalar> s = ocp;
    s.settings.preferred_backend =
        koqkatoo::linalg::compact::PreferredBackend::MKLScalarBatched;
    index_t n_var = ocp.num_variables(), n_constr = ocp.num_constraints(),
            n_dyn_constr = ocp.num_dynamics_constraints();

    // Generate some random optimization solver data.
    Eigen::VectorX<bool> J(n_constr);  // Active set.
    VectorXreal Σ(n_constr),           // ALM penalty factors
        ŷ(n_constr);                   //  & corresponding Lagrange multipliers.
    VectorXreal x(n_var), grad(n_var); // Current iterate and cost gradient.
    VectorXreal b(n_dyn_constr),       // Dynamics constraints right-hand side
        λ(n_dyn_constr);               //  & corresponding Lagrange multipliers.

    real_t S = std::exp2(nrml(rng)) * 1e-1; // primal regularization
    std::ranges::generate(J, [&] { return bernoulli(rng); });
    std::ranges::generate(Σ, [&] { return std::exp2(nrml(rng)); });
    std::ranges::generate(ŷ, [&] { return nrml(rng); });
    std::ranges::generate(x, [&] { return nrml(rng); });
    std::ranges::generate(grad, [&] { return nrml(rng); });
    std::ranges::generate(b, [&] { return nrml(rng); });
    std::ranges::generate(λ, [&] { return nrml(rng); });
    Eigen::VectorX<bool> J2 = J;
    std::uniform_int_distribution<index_t> uconstr(0, n_constr - 1);
    for (index_t i = 0; i < 10; ++i) {
        auto j = uconstr(rng);
        J2(j)  = !J2(j);
    }

    // Convert this data into the compact format used by the OCP solver.
    // (Normally, you would only do this once, and re-use the format in the
    //  optimization solver, only converting back at the very end to return a
    //  solution to the user.)
    auto J_strided    = s.storage.initialize_active_set(as_span(J));
    auto J2_strided   = s.storage.initialize_active_set(as_span(J2));
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

    using scal_blas =
        koqkatoo::linalg::compact::CompactBLAS<stdx::simd_abi::scalar>;
    std::vector<real_t> xu(n_var + nx);
    scal_blas::mut_batch_view xu_batched{
        {.data = xu.data(), .depth = N + 1, .rows = nx + nu}};
    scal_blas::mut_batch_view xu_mat{
        {.data = xu.data(), .rows = nx + nu, .cols = N + 1}};

    grad_strided.view += Mᵀλ_strided.view;
    grad_strided.view += Gᵀŷ_strided.view;
    scal_blas::xneg(Mxb_strided);

    for (auto _ : state) {
#if KOQKATOO_WITH_TRACING
        koqkatoo::trace_logger.reset();
#endif
        solve_riccati(ocp, S, Σ_strided.view, J_strided.view, grad_strided.view,
                      Mxb_strided.view, xu_batched);
    }

    static bool printed = false;
    if (!std::exchange(printed, true) && n_var < 100)
        guanaqo::print_python(std::cout, xu_mat(0));

#if KOQKATOO_WITH_TRACING
    std::cout << "\n" << __PRETTY_FUNCTION__ << "\n";
    std::cout << "[";
    for (const auto &log : koqkatoo::trace_logger.get_logs())
        std::cout << log << ", ";
    std::cout << "]" << std::endl;
#endif
}

template <class simd_abi>
void benchmark_solve(benchmark::State &state) {
    if (!std::exchange(init_done, true)) {
        KOQKATOO_OMP_IF(omp_set_num_threads(n_threads));
        koqkatoo::pool_set_num_threads(n_threads);
        koqkatoo::fork_set_num_threads(n_threads);
    }
    using VectorXreal = Eigen::VectorX<real_t>;
    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp({.N_horiz = n_horiz, //
                                        .nx      = n_states,
                                        .nu      = n_inputs,
                                        .ny      = n_inputs,
                                        .ny_N    = n_inputs},
                                       12345);

    // Instantiate the OCP KKT solver.
    ko::Solver<simd_abi> s = ocp;
    s.settings.preferred_backend =
        koqkatoo::linalg::compact::PreferredBackend::MKLScalarBatched;
    index_t n_var = ocp.num_variables(), n_constr = ocp.num_constraints(),
            n_dyn_constr = ocp.num_dynamics_constraints();

    // Generate some random optimization solver data.
    Eigen::VectorX<bool> J(n_constr);  // Active set.
    VectorXreal Σ(n_constr),           // ALM penalty factors
        ŷ(n_constr);                   //  & corresponding Lagrange multipliers.
    VectorXreal x(n_var), grad(n_var); // Current iterate and cost gradient.
    VectorXreal b(n_dyn_constr),       // Dynamics constraints right-hand side
        λ(n_dyn_constr);               //  & corresponding Lagrange multipliers.

    real_t S = std::exp2(nrml(rng)) * 1e-1; // primal regularization
    std::ranges::generate(J, [&] { return bernoulli(rng); });
    std::ranges::generate(Σ, [&] { return std::exp2(nrml(rng)); });
    std::ranges::generate(ŷ, [&] { return nrml(rng); });
    std::ranges::generate(x, [&] { return nrml(rng); });
    std::ranges::generate(grad, [&] { return nrml(rng); });
    std::ranges::generate(b, [&] { return nrml(rng); });
    std::ranges::generate(λ, [&] { return nrml(rng); });
    Eigen::VectorX<bool> J2 = J;
    std::uniform_int_distribution<index_t> uconstr(0, n_constr - 1);
    for (index_t i = 0; i < 10; ++i) {
        auto j = uconstr(rng);
        J2(j)  = !J2(j);
    }

    // Convert this data into the compact format used by the OCP solver.
    // (Normally, you would only do this once, and re-use the format in the
    //  optimization solver, only converting back at the very end to return a
    //  solution to the user.)
    auto J_strided    = s.storage.initialize_active_set(as_span(J));
    auto J2_strided   = s.storage.initialize_active_set(as_span(J2));
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

    for (auto _ : state) {
#if KOQKATOO_WITH_TRACING
        koqkatoo::trace_logger.reset();
#endif
        s.factor_fork(S, Σ_strided, J_strided);
        s.solve_fork(grad_strided, Mᵀλ_strided, Gᵀŷ_strided, Mxb_strided,
                     d_strided, Δλ_strided, MᵀΔλ_strided);
        // s.updowndate_fork(Σ_strided, J_strided, J2_strided, nullptr);
    }

    auto [N, nx, nu, ny, ny_N] = ocp.dim;

    VectorXreal d = VectorXreal::Zero(n_var + ocp.dim.nx);
    s.storage.restore_variables(d_strided, as_span(d).first(n_var));

    koqkatoo::MutableRealMatrixView xu_mat{
        {.data = d.data(), .rows = nx + nu, .cols = N + 1}};

    static bool printed = false;
    if (!std::exchange(printed, true) && n_var < 100)
        guanaqo::print_python(std::cout, xu_mat);

#if KOQKATOO_WITH_TRACING
    std::cout << "\n" << __PRETTY_FUNCTION__ << "\n";
    std::cout << "[";
    for (const auto &log : koqkatoo::trace_logger.get_logs())
        std::cout << log << ", ";
    std::cout << "]" << std::endl;
#endif
}

BENCHMARK(benchmark_solve<stdx::simd_abi::deduce_t<real_t, 8>>);
BENCHMARK(benchmark_solve<stdx::simd_abi::deduce_t<real_t, 4>>);
BENCHMARK(benchmark_solve<stdx::simd_abi::deduce_t<real_t, 2>>);
BENCHMARK(benchmark_solve<stdx::simd_abi::scalar>);
BENCHMARK(benchmark_riccati);
