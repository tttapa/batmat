#include <gtest/gtest.h>

#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/matrix-view.hpp>
#include <koqkatoo/ocp/conversion.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/thread-pool.hpp>
#include <koqkatoo/trace.hpp>
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
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include "eigen-matchers.hpp"
#include "koqkatoo/linalg-compact/mkl.hpp"

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
                  std::span<real_t> MᵀΔλ, bool use_pcg);

auto reference_qp(const ko::LinearOCPStorage &ocp, real_t S,
                  Eigen::Ref<const Eigen::VectorX<real_t>> Σ,
                  Eigen::Ref<const Eigen::VectorX<bool>> J) {
    using SpMat      = Eigen::SparseMatrix<real_t, 0, index_t>;
    index_t n_constr = ocp.num_constraints(),
            n_dyn    = ocp.num_dynamics_constraints();
    // Build quadratic program and standard KKT system for the OCP.
    auto qp   = ko::LinearOCPSparseQP::build(ocp);
    auto kkt  = qp.build_kkt(S, as_span(Σ), as_span(J));
    auto &qpA = qp.A_sparsity, &qpQ = qp.Q_sparsity, &qpK = kkt.sparsity;
    SpMat Q = Eigen::Map<const SpMat>(
        qpQ.rows, qpQ.cols, qpQ.nnz(), qpQ.outer_ptr.data(),
        qpQ.inner_idx.data(), qp.Q_values.data(), nullptr);
    SpMat G(n_constr, qpA.cols), M(n_dyn, qpA.cols), K(qpK.rows, qpK.cols);
    std::vector<Eigen::Triplet<real_t>> triplets_G, triplets_M, triplets_K;
    for (index_t c = 0; c < qpA.cols; ++c)
        for (index_t i = qpA.outer_ptr[c]; i < qpA.outer_ptr[c + 1]; ++i)
            if (index_t r = qpA.inner_idx[i]; r < n_dyn) // top rows
                triplets_M.emplace_back(r, c, qp.A_values[i]);
            else // bottom rows
                triplets_G.emplace_back(r - n_dyn, c, qp.A_values[i]);
    for (index_t c = 0; c < qpK.cols; ++c)
        for (index_t i = qpK.outer_ptr[c]; i < qpK.outer_ptr[c + 1]; ++i) {
            index_t r = qpK.inner_idx[i];
            if (r >= c)
                triplets_K.emplace_back(r, c, kkt.values[i]);
            if (r > c)
                triplets_K.emplace_back(c, r, kkt.values[i]);
        }
    G.setFromTriplets(triplets_G.begin(), triplets_G.end());
    M.setFromTriplets(triplets_M.begin(), triplets_M.end());
    K.setFromTriplets(triplets_K.begin(), triplets_K.end());
    return std::tuple{std::move(Q), std::move(G), std::move(M), std::move(K)};
}

real_t cond_sparse_sym(
    Eigen::Ref<const Eigen::SparseMatrix<real_t, 0, index_t>> K,
    const Eigen::SparseLU<Eigen::SparseMatrix<real_t, 0, index_t>> *luK =
        nullptr) {
    Spectra::SparseSymMatProd<real_t, Eigen::Lower, 0, index_t> op(K);
    Spectra::SymEigsSolver eigs{op, 1, 10};
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn);
    if (eigs.info() != Spectra::CompInfo::Successful)
        throw std::runtime_error("Largest eigenvalue failed to converge");
    real_t λ_min, λ_max = eigs.eigenvalues()[0];
    if (luK) {
        struct InvKOp {
            std::remove_pointer_t<decltype(luK)> &lu;
            using Scalar = real_t;
            index_t rows() const { return lu.rows(); }
            index_t cols() const { return lu.cols(); }
            void perform_op(const Scalar *x_in, Scalar *x_out) const {
                Eigen::Map<const Eigen::VectorX<real_t>> xi{x_in, cols()};
                Eigen::Map<Eigen::VectorX<real_t>> xo{x_out, rows()};
                xo = lu.solve(xi);
            }
        };
        InvKOp op{*luK};
        Spectra::SymEigsSolver eigs{op, 1, 10};
        eigs.init();
        eigs.compute(Spectra::SortRule::LargestMagn);
        if (eigs.info() != Spectra::CompInfo::Successful)
            throw std::runtime_error("Smallest eigenvalue failed to converge");
        λ_min = 1 / eigs.eigenvalues()[0];
    } else {
        eigs.init();
        eigs.compute(Spectra::SortRule::SmallestMagn);
        if (eigs.info() != Spectra::CompInfo::Successful)
            throw std::runtime_error("Smallest eigenvalue failed to converge");
        λ_min = eigs.eigenvalues()[0];
    }
    using std::abs;
    return abs(λ_max) / abs(λ_min);
}

const int n_threads = 8;
TEST_P(OCPCyclic, solve) {
    KOQKATOO_OMP_IF(omp_set_num_threads(n_threads));
    koqkatoo::pool_set_num_threads(n_threads);
    koqkatoo::fork_set_num_threads(n_threads);
    KOQKATOO_IF_ITT(koqkatoo::foreach_thread([](index_t i, index_t) {
        __itt_thread_set_name(std::format("OMP({})", i).c_str());
    }));

    auto [N_horiz, nx, vl, use_pcg] = GetParam();
    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp({.N_horiz = N_horiz, //
                                        .nx      = nx,
                                        .nu      = 30 * nx / 40,
                                        .ny      = 10 * nx / 40,
                                        .ny_N    = 10 * nx / 40},
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

    auto [Q, G, M, K] = reference_qp(ocp, S, Σ, J);

    VectorXreal Mxb(n_dyn_constr), Mᵀλ(n_var), d(n_var), Δλ(n_dyn_constr),
        MᵀΔλ(n_var), Aᵀŷ(n_var);
    for (index_t i = 0; i < 100; ++i)
        solve_cyclic(ocp, S, as_span(Σ), as_span(J), as_span(x), as_span(grad),
                     as_span(λ), as_span(b), as_span(ŷ), as_span(Mxb),
                     as_span(Mᵀλ), as_span(Aᵀŷ), as_span(d), as_span(Δλ),
                     as_span(MᵀΔλ), use_pcg);
#if KOQKATOO_WITH_TRACING
    koqkatoo::trace_logger.reset();
#endif
    solve_cyclic(ocp, S, as_span(Σ), as_span(J), as_span(x), as_span(grad),
                 as_span(λ), as_span(b), as_span(ŷ), as_span(Mxb), as_span(Mᵀλ),
                 as_span(Aᵀŷ), as_span(d), as_span(Δλ), as_span(MᵀΔλ), use_pcg);
#if KOQKATOO_WITH_TRACING
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
        koqkatoo::TraceLogger::write_column_headings(csv) << '\n';
        for (const auto &log : koqkatoo::trace_logger.get_logs())
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

    real_t κ = cond_sparse_sym(K, &luK);
    std::cout << "κ(K) = " << guanaqo::float_to_str(κ) << std::endl;

    // Compare the koqkatoo OCP solution to the Eigen reference solution.
    EXPECT_THAT(Δλ, EigenAlmostEqual(Δλ_ref, ε * κ));
    EXPECT_THAT(d, EigenAlmostEqual(d_ref, ε * κ));
    EXPECT_THAT(MᵀΔλ, EigenAlmostEqual(MᵀΔλ_ref, ε * κ));
    // TODO: compute condition number instead of hard-coded tolernaces
}

INSTANTIATE_TEST_SUITE_P(OCPCyclic, OCPCyclic,
                         testing::Combine( //
                             testing::Values<index_t>(7, 15, 31, 63, 127, 255),
                             testing::Values<index_t>(40),
                             testing::Values<index_t>(2, 4, 8),
                             testing::Bool()));
