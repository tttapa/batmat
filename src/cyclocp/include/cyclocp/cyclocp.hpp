#pragma once

#include <batmat/assume.hpp>
#include <batmat/config.hpp>
#include <batmat/matrix/layout.hpp>
#include <batmat/matrix/matrix.hpp>
#include <batmat/openmp.h>
#include <batmat/simd.hpp>
#include <batmat/timing.hpp>
#include <cyclocp/cyclocp-storage.hpp>
#include <guanaqo/trace.hpp>

#include "compact.hpp" // TODO

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>
#if !BATMAT_WITH_OPENMP
#include <barrier>
#endif

namespace cyclocp::ocp::cyclocp {

using batmat::index_t;
using batmat::real_t;
using batmat::matrix::StorageOrder;

[[nodiscard]] constexpr index_t get_depth(index_t n) {
    BATMAT_ASSUME(n > 0);
    auto un = static_cast<std::make_unsigned_t<index_t>>(n);
    return static_cast<index_t>(std::bit_width(un - 1));
}

[[nodiscard]] constexpr index_t get_level(index_t i) {
    BATMAT_ASSUME(i > 0);
    auto ui = static_cast<std::make_unsigned_t<index_t>>(i);
    return static_cast<index_t>(std::countr_zero(ui));
}

[[nodiscard]] constexpr index_t get_index_in_level(index_t i) {
    if (i == 0)
        return 0;
    auto l = get_level(i);
    return i >> (l + 1);
}

template <index_t VL = 4, class T = real_t>
struct CyclicOCPSolver {
    using value_type             = T;
    using vl_t                   = std::integral_constant<index_t, VL>;
    using align_t                = std::integral_constant<index_t, VL * alignof(T)>;
    static constexpr index_t vl  = VL;
    static constexpr index_t lvl = get_depth(vl);

    const index_t N_horiz;
    const index_t nx, nu, ny, ny_0, ny_N;

    /// log2(P), logarithm of the number of parallel execution units
    /// (number of processors × vector length)
    const index_t lP = lvl + 3;

    const index_t ceil_N = ((N_horiz + (1 << lP) - 1) / (1 << lP)) * (1 << lP);

    [[nodiscard]] index_t add_wrap_N(index_t a, index_t b) const;
    [[nodiscard]] index_t sub_wrap_N(index_t a, index_t b) const;
    [[nodiscard]] index_t sub_wrap_PmV(index_t a, index_t b) const;
    [[nodiscard]] index_t add_wrap_PmV(index_t a, index_t b) const;
    [[nodiscard]] index_t sub_wrap_P(index_t a, index_t b) const;
    [[nodiscard]] index_t get_linear_batch_offset(index_t biA) const;

    template <StorageOrder O = StorageOrder::ColMajor>
    using matrix = batmat::matrix::Matrix<value_type, index_t, vl_t, index_t, O, align_t>;
    template <StorageOrder O = StorageOrder::ColMajor>
    using mask_matrix = batmat::matrix::Matrix<bool, index_t, vl_t, index_t, O, align_t>;
    template <StorageOrder O = StorageOrder::ColMajor>
    using view = batmat::matrix::View<const value_type, index_t, vl_t, index_t, index_t, O>;
    template <StorageOrder O = StorageOrder::ColMajor>
    using mut_view     = batmat::matrix::View<value_type, index_t, vl_t, index_t, index_t, O>;
    using layer_stride = batmat::matrix::DefaultStride;
    template <StorageOrder O = StorageOrder::ColMajor>
    using batch_view = batmat::matrix::View<const value_type, index_t, vl_t, vl_t, layer_stride, O>;
    template <StorageOrder O = StorageOrder::ColMajor>
    using mut_batch_view = batmat::matrix::View<value_type, index_t, vl_t, vl_t, layer_stride, O>;

    using compact_blas =
        batmat::linalg::compact::CompactBLAS<T, batmat::datapar::deduced_abi<T, VL>>; // TODO

    bool alt                      = true;
    bool use_stair_preconditioner = true;
    index_t pcg_max_iter          = 100;
    value_type pcg_tolerance      = std::numeric_limits<value_type>::epsilon() / 10;
    bool pcg_print_resid          = false;

    matrix<StorageOrder::ColMajor> coupling_D = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lP,
            .rows  = nx,
            .cols  = nx,
        }};
    }();
    matrix<StorageOrder::ColMajor> coupling_U = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lP,
            .rows  = nx,
            .cols  = nx,
        }};
    }();
    matrix<StorageOrder::ColMajor> coupling_Y = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lP,
            .rows  = nx,
            .cols  = nx,
        }};
    }();
    matrix<StorageOrder::ColMajor> work_update = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 4 << lvl,
            .rows  = nx,
            .cols  = (ceil_N >> lvl) * ny,
        }};
    }(); // TODO: merge with riccati_ΥΓ?
    matrix<StorageOrder::ColMajor> work_update_Σ = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lvl,
            .rows  = (ceil_N >> lvl) * ny,
            .cols  = 1,
        }};
    }();
    matrix<StorageOrder::ColMajor> riccati_ÂB̂ = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lP,
            .rows  = nx,
            .cols  = (ceil_N >> lP) * (nu + nx),
        }};
    }();
    matrix<StorageOrder::ColMajor> riccati_BAᵀ = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lP,
            .rows  = nu + nx,
            .cols  = ((ceil_N >> lP) - 1) * nx,
        }};
    }();
    matrix<StorageOrder::ColMajor> riccati_R̂ŜQ̂ = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lP,
            .rows  = nu + nx,
            .cols  = (ceil_N >> lP) * (nu + nx),
        }};
    }();
    matrix<StorageOrder::ColMajor> riccati_ΥΓ1 = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lP,
            .rows  = nu + nx + nx,
            .cols  = (ceil_N >> lP) * std::max(ny, ny_0 + ny_N),
        }};
    }();
    matrix<StorageOrder::ColMajor> riccati_ΥΓ2 = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lP,
            .rows  = nu + nx + nx,
            .cols  = (ceil_N >> lP) * std::max(ny, ny_0 + ny_N),
        }};
    }();
    matrix<StorageOrder::ColMajor> data_BA = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = ceil_N,
            .rows  = nx,
            .cols  = nu + nx,
        }};
    }();
    matrix<StorageOrder::ColMajor> data_DCᵀ = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = ceil_N,
            .rows  = nu + nx,
            .cols  = std::max(ny, ny_0 + ny_N),
        }};
    }();
    matrix<StorageOrder::ColMajor> data_rhs_constr = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = ceil_N,
            .rows  = nx,
            .cols  = 1,
        }};
    }();
    matrix<StorageOrder::ColMajor> work_Σ = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lP,
            .rows  = (ceil_N >> lP) * std::max(ny, ny_0 + ny_N),
            .cols  = 1,
        }};
    }();
    matrix<StorageOrder::ColMajor> data_RSQ = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = ceil_N,
            .rows  = nu + nx,
            .cols  = nu + nx,
        }};
    }();
    matrix<StorageOrder::ColMajor> riccati_work = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = 1 << lP,
            .rows  = nx,
            .cols  = 1,
        }};
    }();
    matrix<StorageOrder::ColMajor> work_pcg = [this] {
        return matrix<StorageOrder::ColMajor>{{
            .depth = vl,
            .rows  = nx,
            .cols  = 4,
        }};
    }();
    std::vector<index_t> nJs = std::vector<index_t>(1 << (lP - lvl));

    struct Timings {
        using type = batmat::DefaultTimings;
        type todo;
    };

    /// Constraints on u(0) and x(N) should be independent.
    ///
    ///                  nx  nu
    ///    ocp.CD(0) = [ 0 | D ] ny₀
    ///                [ 0 | 0 ] ny - ny₀
    ///
    /// Since ocp.D(0) and ocp.C(N) will be merged, the top ny₀ rows of ocp.C(N)
    /// should be zero.
    static CyclicOCPSolver build(const CyclicOCPStorage &ocp, index_t lP);
    void initialize_rhs(const CyclicOCPStorage &ocp, mut_view<> rhs) const;
    void initialize_gradient(const CyclicOCPStorage &ocp, mut_view<> grad) const;
    void initialize_bounds(const CyclicOCPStorage &ocp, mut_view<> b_min, mut_view<> b_max) const;
    void pack_variables(std::span<const value_type> ux_lin, mut_view<> ux) const;
    void unpack_variables(view<> ux, std::span<value_type> ux_lin) const;
    void pack_dynamics(std::span<const value_type> λ_lin, mut_view<> λ) const;
    void unpack_dynamics(view<> λ, std::span<value_type> λ_lin) const;
    void pack_constraints(std::span<const value_type> y_lin, mut_view<> y,
                          value_type fill = 0) const;
    void unpack_constraints(view<> y, std::span<value_type> y_lin) const;

    index_t num_variables() const { return N_horiz * (nu + nx); }
    index_t num_dynamics_constraints() const { return N_horiz * nx; }
    index_t num_general_constraints() const { return (N_horiz - 1) * ny + ny_0 + ny_N; }

    matrix<> initialize_variables() const {
        return matrix<>{{.depth = ceil_N, .rows = nu + nx, .cols = 1}};
    }
    matrix<> initialize_dynamics_constraints() const {
        return matrix<>{{.depth = ceil_N, .rows = nx, .cols = 1}};
    }
    matrix<> initialize_general_constraints() const {
        return matrix<>{{.depth = ceil_N, .rows = std::max(ny, ny_0 + ny_N), .cols = 1}};
    }
    mask_matrix<> initialize_active_set() const {
        return mask_matrix<>{{.depth = ceil_N, .rows = std::max(ny, ny_0 + ny_N), .cols = 1}};
    }

    // For lgp = 5, lgv = 2, N = 3 << lgp
    //
    // | Stage k | Thread t | Index i | Data di | λ(A) | λ(I) | bλ(A) | bλ(I) |
    // |:-------:|:--------:|:-------:|:-------:|-----:|-----:|------:|------:|
    // | 0/96    | 0        | 0       | 0       | 0    | 93   | 0     | 7*    |
    // | 95      | 0        | 1       | 1       |      |      |       |       |
    // | 94      | 0        | 2       | 2       |      |      |       |       |
    // |         |          |         |         |      |      |       |       |
    // | 3       | 1        | 0       | 3       | 3    | 0    | 1     | 0     |
    // | 2       | 1        | 1       | 4       |      |      |       |       |
    // | 1       | 1        | 2       | 5       |      |      |       |       |
    // |         |          |         |         |      |      |       |       |
    // | 6       | 2        | 0       | 6       | 6    | 3    | 2     | 1     |
    // | 5       | 2        | 1       | 7       |      |      |       |       |
    // | 4       | 2        | 2       | 8       |      |      |       |       |
    // |         |          |         |         |      |      |       |       |
    // | 9       | 3        | 0       | 9       | 9    | 6    | 3     | 2     |
    // | 8       | 3        | 1       | 10      |      |      |       |       |
    // | 7       | 3        | 2       | 11      |      |      |       |       |
    // |         |          |         |         |      |      |       |       |
    // | 12      | 4        | 0       | 12      | 12   | 9    | 4     | 3     |
    // | 11      | 4        | 1       | 13      |      |      |       |       |
    // | 10      | 4        | 2       | 14      |      |      |       |       |
    // |         |          |         |         |      |      |       |       |
    // | 15      | 5        | 0       | 15      | 15   | 12   | 5     | 4     |
    // | 14      | 5        | 1       | 16      |      |      |       |       |
    // | 13      | 5        | 2       | 17      |      |      |       |       |
    // |         |          |         |         |      |      |       |       |
    // | 18      | 6        | 0       | 18      | 18   | 15   | 6     | 5     |
    // | 17      | 6        | 1       | 19      |      |      |       |       |
    // | 16      | 6        | 2       | 20      |      |      |       |       |
    // |         |          |         |         |      |      |       |       |
    // | 21      | 7        | 0       | 21      | 21   | 18   | 7     | 6     |
    // | 20      | 7        | 1       | 22      |      |      |       |       |
    // | 19      | 7        | 2       | 23      |      |      |       |       |

    [[nodiscard]] bool is_active(index_t l, index_t bi) const;
    [[nodiscard]] bool is_U_below_Y(index_t l, index_t bi) const;

#if !BATMAT_WITH_OPENMP
    std::unique_ptr<std::barrier<>> std_barrier = std::make_unique<std::barrier<>>(1 << (lP - lvl));
#endif

    void barrier() const {
        BATMAT_OMP(barrier);
#if !BATMAT_WITH_OPENMP
        std_barrier->arrive_and_wait();
#endif
    }

    void residual_dynamics_constr(view<> x, view<> b, mut_view<> Mxb) const;
    void transposed_dynamics_constr(view<> λ, mut_view<> Mᵀλ) const;
    void general_constr(view<> ux, mut_view<> DCux) const;
    void transposed_general_constr(view<> y, mut_view<> DCᵀy) const;
    /// grad_f ← Q ux + a q + b grad_f
    void cost_gradient(view<> ux, value_type a, view<> q, value_type b, mut_view<> grad_f) const;
    void cost_gradient_regularized(value_type S, view<> ux, view<> ux0, view<> q,
                                   mut_view<> grad_f) const;
    void cost_gradient_remove_regularization(value_type S, view<> x, view<> x0,
                                             mut_view<> grad_f) const;

    void factor_schur_U(index_t l, index_t biU);
    void factor_schur_Y(index_t l, index_t biY);
    void factor_l0(index_t ti);
    void factor_riccati(index_t ti, bool alt, value_type S, view<> Σ);
    void factor(value_type S, view<> Σ, bool alt = false);

    void solve_active(index_t l, index_t biY, mut_view<> λ) const;
    void solve_active_secondary(index_t l, index_t biU, mut_view<> λ) const;
    void solve_riccati_forward(index_t ti, mut_view<> ux, mut_view<> λ) const;
    /// Preserves b in λ (except for coupling equations solved using CR)
    void solve_riccati_forward_alt(index_t ti, mut_view<> ux, mut_view<> λ, mut_view<> work) const;
    void solve_forward(mut_view<> ux, mut_view<> λ, mut_view<> work) const;

    value_type mul_A(batch_view<> p, mut_batch_view<> Ap, batch_view<> L, batch_view<> B) const;
    value_type mul_precond(batch_view<> r, mut_batch_view<> z, mut_batch_view<> w, batch_view<> L,
                           batch_view<> B) const;
    void solve_pcg(mut_batch_view<> λ, mut_batch_view<> work_pcg) const;
    void solve_pcg(mut_batch_view<> λ) { solve_pcg(λ, work_pcg.batch(0)); }

    void solve_reverse_active(index_t l, index_t bi, mut_view<> λ) const;
    void solve_riccati_reverse(index_t ti, mut_view<> ux, mut_view<> λ, mut_view<> work) const;
    void solve_riccati_reverse_alt(index_t ti, mut_view<> ux, mut_view<> λ, mut_view<> work) const;
    void solve_reverse(mut_view<> ux, mut_view<> λ, mut_view<> work) const;
    void solve(mut_view<> ux, mut_view<> λ, mut_batch_view<> work_pcg,
               mut_view<> work_riccati) const;
    void solve(mut_view<> ux, mut_view<> λ) { solve(ux, λ, work_pcg.batch(0), riccati_work); }

    void update_level(index_t l, index_t biY);
    void update(view<> ΔΣ);
    void update_riccati(index_t ti, view<> Σ);

    std::vector<std::tuple<index_t, index_t, value_type>>
    build_sparse(const CyclicOCPStorage &ocp, std::span<const value_type> Σ) const;
    std::vector<value_type> build_rhs(view<> ux, view<> λ) const;
    std::vector<std::tuple<index_t, index_t, value_type>> build_sparse_factor() const;
    std::vector<std::tuple<index_t, index_t, value_type>> build_sparse_diag() const;
};

namespace detail {
template <class T1, class I1, class S1, class T2, class I2, class S2>
void copy_T(guanaqo::MatrixView<T1, I1, S1> src, guanaqo::MatrixView<T2, I2, S2> dst) {
    assert(src.rows == dst.cols);
    assert(src.cols == dst.rows);
    for (index_t r = 0; r < src.rows; ++r) // TODO: optimize
        for (index_t c = 0; c < src.cols; ++c)
            dst(c, r) = src(r, c);
}
} // namespace detail

} // namespace cyclocp::ocp::cyclocp
