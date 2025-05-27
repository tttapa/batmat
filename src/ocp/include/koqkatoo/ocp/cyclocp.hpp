#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/linalg-compact/preferred-backend.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/timing.hpp>
#include <guanaqo/trace.hpp>

#include <experimental/simd>
#include <algorithm>
#include <bit>
#include <cassert>
#if !KOQKATOO_WITH_OPENMP
#include <barrier>
#endif

namespace koqkatoo::ocp::cyclocp {

namespace stdx = std::experimental;

[[nodiscard]] constexpr index_t get_depth(index_t n) {
    KOQKATOO_ASSERT(n > 0);
    auto un = static_cast<std::make_unsigned_t<index_t>>(n);
    return static_cast<index_t>(std::bit_width(un - 1));
}

[[nodiscard]] constexpr index_t get_level(index_t i) {
    KOQKATOO_ASSERT(i > 0);
    auto ui = static_cast<std::make_unsigned_t<index_t>>(i);
    return static_cast<index_t>(std::countr_zero(ui));
}

[[nodiscard]] constexpr index_t get_index_in_level(index_t i) {
    if (i == 0)
        return 0;
    auto l = get_level(i);
    return i >> (l + 1);
}

///                ₙ₋₁
///     minimize    ∑ [½ uᵢᵀ Rᵢ uᵢ + uᵢᵀ S xᵢ + ½ xᵢᵀ Qᵢ xᵢ + rᵢᵀuᵢ + qᵢᵀxᵢ]
///                ⁱ⁼⁰
///                + ½ xₙᵀ Qₙ xₙ + qₙᵀ xₙ
///     s.t.        x₀   = xᵢₙᵢₜ
///                 xᵢ₊₁ = Aᵢ xᵢ + Bᵢ uᵢ + cᵢ
///                 lᵢ   ≤ Cᵢ xᵢ + Dᵢ uᵢ ≤ uᵢ
///                 lₙ   ≤ Cₙ xₙ ≤ uₙ
///
///                ₙ₋₁
///     minimize    ∑ [½ uᵢᵀ Rᵢ uᵢ + uᵢᵀ Sᵢ xᵢ + ½ xᵢᵀ Qᵢ xᵢ + rᵢᵀuᵢ + qᵢᵀxᵢ]
///                ⁱ⁼¹
///                + ½ u₀ᵀ R₀ u₀ + (r₀ + S₀ x₀)ᵀ u₀
///                + ½ xₙᵀ Qₙ xₙ + qₙᵀ xₙ
///     s.t.        x₀   = xᵢₙᵢₜ
///                 xᵢ₊₁ = Aᵢ xᵢ + Bᵢ uᵢ + cᵢ
///                 lᵢ   ≤ Cᵢ xᵢ + Dᵢ uᵢ ≤ uᵢ
///                 l₀ - C₀ x₀ ≤ D₀ U₀ ≤ u₀ - C₀ x₀
///                 lₙ   ≤ Cₙ xₙ ≤ uₙ
struct CyclicOCPStorage {
    index_t N_horiz;
    index_t nx, nu, ny, ny_0, ny_N;
    /// Storage layout:         size                 offset
    ///     N × [ R  S ] = H    (nu+nx)²             0
    ///         [ Sᵀ Q ]
    ///     N × [ B  A ]        nx(nu+nx)            N (nu+nx)²
    /// (N-1) × [ D  C ]        ny(nu+nx)            N (nu+nx)² + N nx(nu+nx)
    ///     1 × [ D  C ]        (ny_0+ny_N)(nu+nx)   N (nu+nx)² + N nx(nu+nx) + (N-1)ny(nu+nx)
    using matrix  = linalg::compact::BatchedMatrix<real_t, index_t>;
    matrix data_H = [this] {
        return matrix{{.depth = N_horiz, .rows = nu + nx, .cols = nu + nx}};
    }();
    matrix data_F = [this] {
        return matrix{{.depth = N_horiz, .rows = nx, .cols = nu + nx}};
    }();
    matrix data_G = [this] {
        return matrix{{.depth = N_horiz - 1, .rows = ny, .cols = nu + nx}};
    }();
    matrix data_G0N = [this] {
        return matrix{{.depth = 1, .rows = ny_0 + ny_N, .cols = nu + nx}};
    }();
    matrix data_rq = [this] {
        return matrix{{.depth = N_horiz, .rows = nu + nx, .cols = 1}};
    }();
    matrix data_c = [this] {
        return matrix{{.depth = N_horiz, .rows = nx, .cols = 1}};
    }();
    matrix data_lb = [this] {
        return matrix{{.depth = N_horiz - 1, .rows = ny, .cols = 1}};
    }();
    matrix data_lb0N = [this] {
        return matrix{{.depth = 1, .rows = ny_0 + ny_N, .cols = 1}};
    }();
    matrix data_ub = [this] {
        return matrix{{.depth = N_horiz - 1, .rows = ny, .cols = 1}};
    }();
    matrix data_ub0N = [this] {
        return matrix{{.depth = 1, .rows = ny_0 + ny_N, .cols = 1}};
    }();

    static CyclicOCPStorage build(const LinearOCPStorage &ocp,
                                  std::span<const real_t> qr,
                                  std::span<const real_t> b_eq,
                                  std::span<const real_t> b_lb,
                                  std::span<const real_t> b_ub);
};

template <index_t VL = 4>
struct CyclicOCPSolver {
    static constexpr index_t vl  = VL;
    static constexpr index_t lvl = get_depth(vl);

    const index_t N_horiz;
    const index_t nx, nu, ny, ny_0, ny_N;

    /// log2(P), logarithm of the number of parallel execution units
    /// (number of processors × vector length)
    const index_t lP = lvl + 3;

    linalg::compact::PreferredBackend backend =
        linalg::compact::PreferredBackend::MKLScalarBatched;

    [[nodiscard]] index_t add_wrap_N(index_t a, index_t b) const;
    [[nodiscard]] index_t sub_wrap_N(index_t a, index_t b) const;
    [[nodiscard]] index_t sub_wrap_PmV(index_t a, index_t b) const;
    [[nodiscard]] index_t add_wrap_PmV(index_t a, index_t b) const;
    [[nodiscard]] index_t sub_wrap_P(index_t a, index_t b) const;
    [[nodiscard]] index_t get_linear_batch_offset(index_t biA) const;

    using simd_abi              = stdx::simd_abi::deduce_t<real_t, VL>;
    using compact_blas          = linalg::compact::CompactBLAS<simd_abi>;
    using matrix                = compact_blas::matrix;
    using mask_matrix           = compact_blas::bool_matrix;
    using mut_matrix_view       = compact_blas::mut_batch_view;
    using matrix_view           = compact_blas::batch_view;
    using matrix_view_batch     = compact_blas::single_batch_view;
    using mut_matrix_view_batch = compact_blas::mut_single_batch_view;

    bool alt                      = true;
    bool use_stair_preconditioner = true;
    index_t max_pcg_iter          = 100;

    matrix coupling_D = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = nx,
            .cols  = nx,
        }};
    }();
    matrix coupling_U = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = nx,
            .cols  = nx,
        }};
    }();
    matrix coupling_Y = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = nx,
            .cols  = nx,
        }};
    }();
    matrix work_update = [this] {
        return matrix{{
            .depth = 4 << lvl,
            .rows  = nx,
            .cols  = (N_horiz >> lvl) * ny,
        }};
    }(); // TODO: merge with riccati_ΥΓ?
    matrix work_update_Σ = [this] {
        return matrix{{
            .depth = 1 << lvl,
            .rows  = (N_horiz >> lvl) * ny,
            .cols  = 1,
        }};
    }();
    matrix riccati_ÂB̂ = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = nx,
            .cols  = (N_horiz >> lP) * (nu + nx),
        }};
    }();
    matrix riccati_BAᵀ = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = nu + nx,
            .cols  = ((N_horiz >> lP) - 1) * nx,
        }};
    }();
    matrix riccati_R̂ŜQ̂ = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = nu + nx,
            .cols  = (N_horiz >> lP) * (nu + nx),
        }};
    }();
    matrix riccati_ΥΓ1 = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = nu + nx + nx,
            .cols  = (N_horiz >> lP) * std::max(ny, ny_0 + ny_N),
        }};
    }();
    matrix riccati_ΥΓ2 = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = nu + nx + nx,
            .cols  = (N_horiz >> lP) * std::max(ny, ny_0 + ny_N),
        }};
    }();
    matrix data_BA = [this] {
        return matrix{{
            .depth = N_horiz,
            .rows  = nx,
            .cols  = nu + nx,
        }};
    }();
    matrix data_DCᵀ = [this] {
        return matrix{{
            .depth = N_horiz,
            .rows  = nu + nx,
            .cols  = std::max(ny, ny_0 + ny_N),
        }};
    }();
    matrix data_rhs_constr = [this] {
        return matrix{{
            .depth = N_horiz,
            .rows  = nx,
            .cols  = 1,
        }};
    }();
    matrix work_Σ = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = (N_horiz >> lP) * std::max(ny, ny_0 + ny_N),
            .cols  = 1,
        }};
    }();
    matrix data_RSQ = [this] {
        return matrix{{
            .depth = N_horiz,
            .rows  = nu + nx,
            .cols  = nu + nx,
        }};
    }();
    matrix riccati_work = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = nx,
            .cols  = 1,
        }};
    }();
    matrix work_pcg = [this] {
        return matrix{{
            .depth = vl,
            .rows  = nx,
            .cols  = 4,
        }};
    }();
    std::vector<index_t> nJs = std::vector<index_t>(1 << (lP - lvl));

    struct Timings {
        using type = koqkatoo::DefaultTimings;
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
    void initialize_rhs(const CyclicOCPStorage &ocp, mut_matrix_view rhs) const;
    void pack_variables(std::span<const real_t> ux_lin,
                        mut_matrix_view ux) const;
    void unpack_variables(matrix_view ux, std::span<real_t> ux_lin) const;
    void pack_dynamics(std::span<const real_t> λ_lin, mut_matrix_view λ) const;
    void unpack_dynamics(matrix_view λ, std::span<real_t> λ_lin) const;
    void pack_constraints(std::span<const real_t> y_lin, mut_matrix_view y,
                          real_t fill = 0) const;
    void unpack_constraints(matrix_view y, std::span<real_t> y_lin) const;

    index_t num_variables() const { return N_horiz * (nu + nx); }
    index_t num_dynamics_constraints() const { return N_horiz * nx; }
    index_t num_general_constraints() const {
        return N_horiz * ny + ny_0 + ny_N;
    }

    matrix initialize_variables() const {
        return matrix{{.depth = N_horiz, .rows = nu + nx, .cols = 1}};
    }
    matrix initialize_dynamics_constraints() const {
        return matrix{{.depth = N_horiz, .rows = nx, .cols = 1}};
    }
    matrix initialize_general_constraints() const {
        return matrix{
            {.depth = N_horiz, .rows = std::max(ny, ny_0 + ny_N), .cols = 1}};
    }
    mask_matrix initialize_active_set() const {
        return mask_matrix{
            {.depth = N_horiz, .rows = std::max(ny, ny_0 + ny_N), .cols = 1}};
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

#if !KOQKATOO_WITH_OPENMP
    mutable std::barrier<> std_barrier{1 << (lP - lvl)};
#endif

    void barrier() const {
        KOQKATOO_OMP(barrier);
#if !KOQKATOO_WITH_OPENMP
        std_barrier.arrive_and_wait();
#endif
    }

    void residual_dynamics_constr(matrix_view x, matrix_view b,
                                  mut_matrix_view Mxb) const;
    void transposed_dynamics_constr(matrix_view λ, mut_matrix_view Mᵀλ) const;
    void general_constr(matrix_view ux, mut_matrix_view DCux) const;
    void transposed_general_constr(matrix_view y, mut_matrix_view DCᵀy) const;
    /// grad_f ← Q ux + a q + b grad_f
    void cost_gradient(matrix_view ux, real_t a, matrix_view q, real_t b,
                       mut_matrix_view grad_f) const;
    void cost_gradient_regularized(real_t S, matrix_view ux, matrix_view ux0,
                                   matrix_view q, mut_matrix_view grad_f) const;
    void cost_gradient_remove_regularization(real_t S, matrix_view x,
                                             matrix_view x0,
                                             mut_matrix_view grad_f) const;

    void factor_schur_U(index_t l, index_t biU);
    void factor_schur_Y(index_t l, index_t biY);
    void factor_l0(index_t ti);
    void factor_riccati(index_t ti, bool alt, matrix_view Σ);
    void factor(matrix_view Σ, bool alt = false);

    void solve_active(index_t l, index_t biY, mut_matrix_view λ) const;
    void solve_active_secondary(index_t l, index_t biU,
                                mut_matrix_view λ) const;
    void solve_riccati_forward(index_t ti, mut_matrix_view ux,
                               mut_matrix_view λ) const;
    /// Preserves b in λ (except for coupling equations solved using CR)
    void solve_riccati_forward_alt(index_t ti, mut_matrix_view ux,
                                   mut_matrix_view λ,
                                   mut_matrix_view work) const;
    void solve_forward(mut_matrix_view ux, mut_matrix_view λ,
                       mut_matrix_view work) const;

    real_t mul_A(matrix_view_batch p, mut_matrix_view_batch Ap,
                 matrix_view_batch L, matrix_view_batch B) const;
    real_t mul_precond(matrix_view_batch r, mut_matrix_view_batch z,
                       mut_matrix_view_batch w, matrix_view_batch L,
                       matrix_view_batch B) const;
    void solve_pcg(mut_matrix_view_batch λ,
                   mut_matrix_view_batch work_pcg) const;
    void solve_pcg(mut_matrix_view_batch λ) { solve_pcg(λ, work_pcg.batch(0)); }

    void solve_reverse_active(index_t l, index_t bi, mut_matrix_view λ) const;
    void solve_riccati_reverse(index_t ti, mut_matrix_view ux,
                               mut_matrix_view λ, mut_matrix_view work) const;
    void solve_riccati_reverse_alt(index_t ti, mut_matrix_view ux,
                                   mut_matrix_view λ,
                                   mut_matrix_view work) const;
    void solve_reverse(mut_matrix_view ux, mut_matrix_view λ,
                       mut_matrix_view work) const;
    void solve(mut_matrix_view ux, mut_matrix_view λ,
               mut_matrix_view_batch work_pcg,
               mut_matrix_view work_riccati) const;
    void solve(mut_matrix_view ux, mut_matrix_view λ) {
        solve(ux, λ, work_pcg.batch(0), riccati_work);
    }

    void update_level(index_t l, index_t biY);
    void update(matrix_view ΔΣ);
    void update_riccati(index_t ti, matrix_view Σ);

    std::vector<std::tuple<index_t, index_t, real_t>>
    build_sparse(const CyclicOCPStorage &ocp, std::span<const real_t> Σ) const;
    std::vector<real_t> build_rhs(matrix_view ux, matrix_view λ) const;
    std::vector<std::tuple<index_t, index_t, real_t>>
    build_sparse_factor() const;
    std::vector<std::tuple<index_t, index_t, real_t>> build_sparse_diag() const;
};

namespace detail {
template <class T1, class I1, class S1, class T2, class I2, class S2>
void copy_T(guanaqo::MatrixView<T1, I1, S1> src,
            guanaqo::MatrixView<T2, I2, S2> dst) {
    assert(src.rows == dst.cols);
    assert(src.cols == dst.rows);
    for (index_t r = 0; r < src.rows; ++r) // TODO: optimize
        for (index_t c = 0; c < src.cols; ++c)
            dst(c, r) = src(r, c);
}
} // namespace detail

} // namespace koqkatoo::ocp::cyclocp
