#include <benchmark/benchmark.h>

#include <koqkatoo/matrix-view.hpp>
#include <koqkatoo/ocp/conversion.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/trace.hpp>

#include <guanaqo/eigen/span.hpp>
#include <guanaqo/eigen/view.hpp>

#include <algorithm>
#include <iostream>
#include <random>

#include <Eigen/Eigen>

namespace ko   = koqkatoo::ocp;
namespace stdx = std::experimental;
using guanaqo::as_eigen;
using guanaqo::as_span;
using koqkatoo::index_t;
using koqkatoo::real_t;
using koqkatoo::RealMatrixView;
namespace sp = guanaqo::linalg::sparsity;

template <class simd_abi>
void benchmark_solve(benchmark::State &state) {
    using VectorXreal = Eigen::VectorX<real_t>;
    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp({.N_horiz = 63, //
                                        .nx      = 40,
                                        .nu      = 40,
                                        .ny      = 30,
                                        .ny_N    = 10},
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

    // Convert this data into the compact format used by the OCP solver.
    // (Normally, you would only do this once, and re-use the format in the
    //  optimization solver, only converting back at the very end to return a
    //  solution to the user.)
    auto J_strided    = s.storage.initialize_active_set(as_span(J));
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
    }

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
