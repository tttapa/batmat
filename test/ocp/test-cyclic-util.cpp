#include <gtest/gtest.h>

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>

#include <experimental/simd>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <print>
#include <type_traits>

namespace stdx = std::experimental;

#define PRINTLN(...) std::println(__VA_ARGS__)

namespace koqkatoo::ocp {

template <index_t VL = 4>
struct CyclicOCPSolver {
    using simd_abi      = stdx::simd_abi::deduce_t<real_t, VL>;
    using compact_blas  = linalg::compact::CompactBLAS<simd_abi>;
    using scalar_blas   = linalg::compact::CompactBLAS<stdx::simd_abi::scalar>;
    using real_matrix   = compact_blas::matrix;
    using mut_real_view = compact_blas::mut_batch_view;
    using bool_matrix   = compact_blas::bool_matrix;

    [[nodiscard]] static constexpr index_t get_depth(index_t n) {
        assert(n > 0);
        auto un = static_cast<std::make_unsigned_t<index_t>>(n);
        return static_cast<index_t>(std::bit_width(un - 1));
    }

    [[nodiscard]] static constexpr index_t get_level(index_t i) {
        assert(i > 0);
        auto ui = static_cast<std::make_unsigned_t<index_t>>(i);
        return static_cast<index_t>(std::countr_zero(ui));
    }

    static constexpr index_t vl  = VL;
    static constexpr index_t lvl = get_depth(vl);

    const OCPDim dim;
    const index_t lN      = get_depth(dim.N_horiz + 1);
    const index_t ln      = lN - lvl;
    const index_t n       = index_t{1} << ln;
    const index_t vstride = index_t{1} << ln;

    [[nodiscard]] static constexpr index_t get_index_in_level(index_t i) {
        auto l = get_level(i);
        return i >> (l + 1);
    }

    [[nodiscard]] constexpr index_t get_level_width(index_t l) const {
        auto d = get_depth(n);
        assert(l < d);
        return index_t{1} << (d - l - 1);
    }

    [[nodiscard]] index_t get_batch_index(index_t i) {
        assert(i < n);
        if (i == 0)
            return n - 1;
        auto l  = get_level(i);
        auto il = get_index_in_level(i);
        return il + (1 << ln) - (1 << (ln - l));
    }

    void initialize(const LinearOCPStorage &ocp) {
        auto [N, nx, nu, ny, ny_N] = dim;
        for (index_t i = 0; i < n; ++i) {
            auto bi = get_batch_index(i);
            for (index_t vi = 0; vi < VL; ++vi) {
                auto k = i + vi * vstride;
                PRINTLN("  {} -> {}({})", k, bi, vi);
                if (k < dim.N_horiz) {
                    H.batch(bi)(vi)  = ocp.H(k);
                    CD.batch(bi)(vi) = ocp.CD(k);
                    AB.batch(bi)(vi) = ocp.AB(k);
                } else if (k < dim.N_horiz) {
                    H.batch(bi)(vi).bottom_left(nu, nx).set_constant(0);
                    H.batch(bi)(vi).right_cols(nu).set_constant(0);
                    H.batch(bi)(vi).bottom_right(nu, nu).set_diagonal(1);
                    H.batch(bi)(vi).top_left(nx, nx) = ocp.Q(k);
                    CD.batch(bi)(vi).right_cols(nu).set_constant(0);
                    CD.batch(bi)(vi).bottom_left(ny - ny_N, nx).set_constant(0);
                    CD.batch(bi)(vi).top_left(ny_N, nx) = ocp.C(k);
                    AB.batch(bi)(vi).set_constant(0);
                } else {
                    H.batch(bi)(vi).set_constant(0);
                    H.batch(bi)(vi).set_diagonal(1);
                    CD.batch(bi)(vi).set_constant(0);
                    AB.batch(bi)(vi).set_constant(0);
                }
            }
        }
    }

#include "cyclic-storage.ipp"

  public:
    CyclicOCPSolver(OCPDim dim) : dim{dim} {}
    CyclicOCPSolver(const CyclicOCPSolver &)            = delete;
    CyclicOCPSolver &operator=(const CyclicOCPSolver &) = delete;
};

} // namespace koqkatoo::ocp

#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/thread-pool.hpp>
#include <koqkatoo/trace.hpp>

#include <Eigen/Eigen>

#include <random>

namespace ko = koqkatoo::ocp;
using koqkatoo::index_t;
using koqkatoo::real_t;
using EVec = Eigen::VectorX<real_t>;

const int n_threads = 8;
TEST(CyclicUtil, heapIndex) {
    using namespace koqkatoo::ocp;

    KOQKATOO_OMP_IF(omp_set_num_threads(n_threads));
    koqkatoo::pool_set_num_threads(n_threads);
    koqkatoo::fork_set_num_threads(n_threads);
    KOQKATOO_IF_ITT(koqkatoo::foreach_thread([](index_t i, index_t) {
        __itt_thread_set_name(std::format("OMP({})", i).c_str());
    }));

    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    OCPDim dim{.N_horiz = 31, .nx = 2, .nu = 1, .ny = 0, .ny_N = 0};

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp(dim, 12345);
    CyclicOCPSolver<> solver{ocp.dim};

    // Instantiate the OCP KKT solver.
    index_t n_var = ocp.num_variables(), n_constr = ocp.num_constraints(),
            n_dyn_constr = ocp.num_dynamics_constraints();

    // Generate some random optimization solver data.
    Eigen::VectorX<bool> J(n_constr), // Active set.
        J0(n_constr), J1(n_constr);   // Active set for initialization.
    EVec Σ(n_constr),                 // ALM penalty factors
        ŷ(n_constr);                  //  & corresponding Lagrange multipliers.
    EVec x(n_var),
        grad(n_var);      // Current iterate and cost gradient.
    EVec b(n_dyn_constr), // Dynamics constraints right-hand side
        λ(n_dyn_constr);  //  & corresponding Lagrange multipliers.

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

    solver.initialize(ocp);
}
