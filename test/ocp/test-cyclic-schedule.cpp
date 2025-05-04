#include <Eigen/SparseCholesky>
#include <gtest/gtest.h>

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/linalg-compact/preferred-backend.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/openmp.h>
#include <guanaqo/trace.hpp>

#include <experimental/simd>
#include <guanaqo/print.hpp>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <print>
#include <stdexcept>
#include <type_traits>
#if !KOQKATOO_WITH_OPENMP
#include <barrier>
#endif

namespace stdx = std::experimental;

#define DO_PRINT 0
#if DO_PRINT
#define PRINTLN(...) std::println(__VA_ARGS__)
#else
#define PRINTLN(...)
#endif

struct VecReg {
    koqkatoo::index_t n, k0, stride, N;
};

// ANSI color codes
constexpr std::string_view colors[] = {
    "\033[34m", // Blue
    "\033[32m", // Green
    "\033[33m", // Yellow
    "\033[31m"  // Red
};
constexpr std::string_view reset = "\033[0m";

// Formatter specialization
template <>
struct std::formatter<VecReg> {
    constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }
    auto format(const VecReg &value, std::format_context &ctx) const {
        auto out = ctx.out();
        for (koqkatoo::index_t i = 0; i < value.n; ++i)
            out =
                std::format_to(out, "{}{:>3}{}", colors[i % 4],
                               (value.k0 + i * value.stride) % value.N, reset);
        return out;
    }
};

namespace koqkatoo::ocp::test {

[[nodiscard]] constexpr index_t get_depth(index_t n) {
    assert(n > 0);
    auto un = static_cast<std::make_unsigned_t<index_t>>(n);
    return static_cast<index_t>(std::bit_width(un - 1));
}

[[nodiscard]] constexpr index_t get_level(index_t i) {
    assert(i > 0);
    auto ui = static_cast<std::make_unsigned_t<index_t>>(i);
    return static_cast<index_t>(std::countr_zero(ui));
}

#if KQT_CYCLIC_TEMPLATE
template <index_t VL = 4>
#else
constexpr index_t VL = 4;
#endif
struct CyclicOCPSolver {
    static constexpr index_t vl  = VL;
    static constexpr index_t lvl = get_depth(vl);

    const OCPDim dim;
    /// log2(P), logarithm of the number of parallel execution units
    /// (number of processors × vector length)
    const index_t lP = lvl + 3;

    linalg::compact::PreferredBackend backend =
        linalg::compact::PreferredBackend::MKLScalarBatched;

    [[nodiscard]] index_t add_wrap_N(index_t a, index_t b) const {
        const index_t N = dim.N_horiz;
        KOQKATOO_ASSUME(a >= 0);
        KOQKATOO_ASSUME(b >= 0);
        KOQKATOO_ASSUME(a < N);
        a += b;
        return a >= N ? a - N : a;
    }
    [[nodiscard]] index_t sub_wrap_N(index_t a, index_t b) const {
        const index_t N = dim.N_horiz;
        KOQKATOO_ASSUME(a >= 0);
        KOQKATOO_ASSUME(b >= 0);
        KOQKATOO_ASSUME(a < N);
        a -= b;
        return a < 0 ? a + N : a;
    }
    [[nodiscard]] index_t sub_wrap_PmV(index_t a, index_t b) const {
        KOQKATOO_ASSUME(a >= 0);
        KOQKATOO_ASSUME(b >= 0);
        KOQKATOO_ASSUME(a < (1 << (lP - lvl)));
        a -= b;
        return a < 0 ? a + (1 << (lP - lvl)) : a;
    }
    [[nodiscard]] index_t add_wrap_PmV(index_t a, index_t b) const {
        KOQKATOO_ASSUME(a >= 0);
        KOQKATOO_ASSUME(b >= 0);
        KOQKATOO_ASSUME(a < (1 << (lP - lvl)));
        a += b;
        return a >= (1 << (lP - lvl)) ? a - (1 << (lP - lvl)) : a;
    }
    [[nodiscard]] index_t sub_wrap_P(index_t a, index_t b) const {
        KOQKATOO_ASSUME(a >= 0);
        KOQKATOO_ASSUME(b >= 0);
        KOQKATOO_ASSUME(a < (1 << lP));
        a -= b;
        return a < 0 ? a + (1 << lP) : a;
    }

    [[nodiscard]] static constexpr index_t get_index_in_level(index_t i) {
        if (i == 0)
            return 0;
        auto l = get_level(i);
        return i >> (l + 1);
    }

    struct join_counter_t {
        alignas(64) std::atomic<int32_t> value{};
        void wait(index_t i) const {
            do {
                int32_t old = value.load();
                if (old >= static_cast<int32_t>(i))
                    break;
                value.wait(old, std::memory_order_relaxed);
            } while (true);
        }
        void wait_and(index_t i) const {
            do {
                int32_t old = value.load();
                if ((old & static_cast<int32_t>(i)) == static_cast<int32_t>(i))
                    break;
                value.wait(old, std::memory_order_relaxed);
            } while (true);
        }
        void notify(index_t i) {
            value.store(static_cast<int32_t>(i));
            value.notify_all();
        }
        void notify_or(index_t i) {
            value.fetch_or(static_cast<int32_t>(i));
            value.notify_all();
        }
    };

    [[nodiscard]] constexpr index_t thread2batch(index_t l, index_t ti) const {
        return (ti + (1 << l) - 1) & ((1 << (lP - lvl)) - 1);
    }
    [[nodiscard]] constexpr index_t batch2thread(index_t l, index_t k) const {
        return (k - (1 << l) + 1) & ((1 << (lP - lvl)) - 1);
    }

    using simd_abi     = stdx::simd_abi::deduce_t<real_t, VL>;
    using compact_blas = linalg::compact::CompactBLAS<simd_abi>;
    using matrix       = compact_blas::matrix;

    matrix coupling_D = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = dim.nx,
            .cols  = dim.nx,
        }};
    }();
    matrix coupling_U = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = dim.nx,
            .cols  = dim.nx,
        }};
    }();
    matrix coupling_Y = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = dim.nx,
            .cols  = dim.nx,
        }};
    }();
    matrix riccati_ÂB̂ = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = dim.nx,
            .cols  = (dim.N_horiz >> lP) * (dim.nu + dim.nx),
        }};
    }();
    matrix riccati_BAᵀ = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = dim.nu + dim.nx,
            .cols  = ((dim.N_horiz >> lP) - 1) * dim.nx,
        }};
    }();
    matrix riccati_R̂ŜQ̂ = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = dim.nu + dim.nx,
            .cols  = (dim.N_horiz >> lP) * (dim.nu + dim.nx),
        }};
    }();
    matrix data_BA = [this] {
        return matrix{{
            .depth = dim.N_horiz,
            .rows  = dim.nx,
            .cols  = dim.nu + dim.nx,
        }};
    }();
    matrix data_RSQ = [this] {
        return matrix{{
            .depth = dim.N_horiz,
            .rows  = dim.nu + dim.nx,
            .cols  = dim.nu + dim.nx,
        }};
    }();
    std::vector<join_counter_t> counters =
        std::vector<join_counter_t>(1 << (lP - lvl));
    std::vector<join_counter_t> counters_UY =
        std::vector<join_counter_t>(1 << (lP - lvl));

    void initialize(const LinearOCPStorage &ocp) {
        KOQKATOO_ASSERT(ocp.dim == dim);
        auto [N, nx, nu, ny, ny_N] = dim;
        const auto vstride         = N >> lvl;
        const index_t num_stages   = N >> lP; // number of stages per thread
        for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
            const index_t k0  = ti * num_stages;
            const index_t bi0 = ti * num_stages;
            for (index_t i = 0; i < num_stages; ++i) {
                index_t bi = bi0 + i;
                for (index_t vi = 0; vi < vl; ++vi) {
                    auto k = sub_wrap_N(k0 + vi * vstride, i);
                    std::println(
                        "ti={:<2}, i={:<2}, k={:<2}, bi={:<2}, vi={:<2}", ti, i,
                        k, bi, vi);
                    if (k < dim.N_horiz) {
                        data_BA.batch(bi)(vi).left_cols(nu)     = ocp.B(k);
                        data_RSQ.batch(bi)(vi).top_left(nu, nu) = ocp.R(k);
                        if (k == 0) {
                            data_BA.batch(bi)(vi).right_cols(nx).set_constant(
                                0);
                            data_RSQ.batch(bi)(vi)
                                .bottom_left(nx, nu)
                                .set_constant(0);
                            data_RSQ.batch(bi)(vi).bottom_right(nx, nx) =
                                ocp.Q(N);
                        } else {
                            data_BA.batch(bi)(vi).right_cols(nx) = ocp.A(k);
                            data_RSQ.batch(bi)(vi).bottom_left(nx, nu) =
                                ocp.S_trans(k); // TODO
                            data_RSQ.batch(bi)(vi).bottom_right(nx, nx) =
                                ocp.Q(k);
                        }
                    } else {
                        // data_RSQ.batch(bi)(vi).set_constant(0);
                        // data_RSQ.batch(bi)(vi).set_diagonal(1);
                        // data_BA.batch(bi)(vi).set_constant(0);
                    }
                }
            }
        }
    }

    // For lgp = 5, lgv = 2, N = 3 << lgp
    //
    // | Stage k | Thread t | Index i | Batch b | λ(A) | λ(I) | bλ(A) | bλ(I) |
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

    // Algorithm
    //
    // factor_riccati (all)
    //
    // factor l0
    //     compute U, Y
    //     L⁻ᵀL⁻¹ -> D
    //     syrk AB -> D
    //     factor D (updating U and Y)   -- only odd
    //     compute next U/Y              -- only odd
    //
    // factor coupling
    //     YYᵀ -> D
    //     UUᵀ -> D
    //     factor D (updating U and Y)
    //     compute next U/Y

    // Counters D
    //     1 = L⁻ᵀL⁻¹ has been added to D
    //     2 = (BA)(BA)ᵀ has been added to D,
    //          and if odd, DUY has been factored
    //     2l + 1 = YYᵀ has been added to D
    //     2l + 2 = UUᵀ has been added to D,
    //          and if odd, DUY have been factored

    // Counters UY
    //     1 = Y is available (bit field)
    //     2 = U is available (bit field)

    [[nodiscard]] bool is_active(index_t l, index_t bi) const {
        const index_t lbi   = bi > 0 ? get_level(bi) : lP - lvl;
        const bool inactive = lbi < l;
        return ((bi >> l) & 1) == 1 && !inactive;
    }
    [[nodiscard]] bool is_U_below_Y(index_t l, index_t bi) const {
        return ((bi >> l) & 3) == 1 && l + 1 != lP - lvl;
    }

#if !KOQKATOO_WITH_OPENMP
    std::barrier<> std_barrier{1 << (lP - lvl)};
#endif

    void barrier() {
        KOQKATOO_OMP(barrier);
#if !KOQKATOO_WITH_OPENMP
        std_barrier.arrive_and_wait();
#endif
#if DO_PRINT
        KOQKATOO_OMP(single)
        std::println("---");
#endif
    }

    void process_active(index_t l, index_t biY) {
        const index_t offset = 1 << l;
        // Compute Y[bi]
        {
            PRINTLN("trsm D{} Y{}", biY, biY);
            GUANAQO_TRACE("Trsm Y", biY);
            compact_blas::xtrsm_RLTN(coupling_D.batch(biY),
                                     coupling_Y.batch(biY), backend);
        }
        // Wait for U[bi] from process_active_secondary
        barrier();
        for (index_t c = 0; c < coupling_U.cols(); c += 1)
            for (index_t r = 0; r < coupling_U.rows(); r += 16)
                __builtin_prefetch(&coupling_U.batch(biY)(0, r, c), 0, 3);
        // Compute UYᵀ or YUᵀ
        if (is_U_below_Y(l, biY)) {
            const index_t bi_next =
                add_wrap_PmV(biY, offset); // TODO: need mod?
            PRINTLN("gemm U{} Y{} -> U{}", biY, biY, bi_next);
            GUANAQO_TRACE("Compute U", bi_next);
            compact_blas::xgemm_NT_neg(coupling_U.batch(biY),
                                       coupling_Y.batch(biY),
                                       coupling_U.batch(bi_next), backend);
        } else {
            const index_t bi_prev =
                sub_wrap_PmV(biY, offset); // TODO: need mod?
            PRINTLN("gemm Y{} U{} -> Y{}", biY, biY, bi_prev);
            GUANAQO_TRACE("Compute Y", bi_prev);
            compact_blas::xgemm_NT_neg(coupling_Y.batch(biY),
                                       coupling_U.batch(biY),
                                       coupling_Y.batch(bi_prev), backend);
        }
    }

    void process_active_secondary(index_t l, index_t biU) {
        const index_t offset = 1 << l;
        const index_t biD    = sub_wrap_PmV(biU, offset);
        const index_t biY    = sub_wrap_PmV(biD, offset);
        for (index_t c = 0; c < coupling_D.cols(); c += 1)
            for (index_t r = 0; r < coupling_D.rows(); r += 16)
                __builtin_prefetch(&coupling_D.batch(biD)(0, r, c), 0, 3);
        // Compute U[bi]
        {
            PRINTLN("trsm D{} U{}", biU, biU);
            GUANAQO_TRACE("Trsm U", biU);
            compact_blas::xtrsm_RLTN(coupling_D.batch(biU),
                                     coupling_U.batch(biU), backend);
        }
        // Wait for Y[bi] from process_active
        barrier();
        for (index_t c = 0; c < coupling_Y.cols(); c += 1)
            for (index_t r = 0; r < coupling_Y.rows(); r += 16)
                __builtin_prefetch(&coupling_Y.batch(biY)(0, r, c), 0, 3);
        // D -= UUᵀ
        {
            PRINTLN("syrk U{} -> D{}", biU, biD);
            GUANAQO_TRACE("Subtract UUᵀ", biD);
            compact_blas::xsyrk_sub(coupling_U.batch(biU),
                                    coupling_D.batch(biD), backend);
        }
        // D -= YYᵀ
        {
            PRINTLN("syrk Y{} -> D{}", biY, biD);
            GUANAQO_TRACE("Subtract YYᵀ", biD);
            biD == 0 ? compact_blas::xsyrk_sub_shift(coupling_Y.batch(biY),
                                                     coupling_D.batch(biD))
                     : compact_blas::xsyrk_sub(coupling_Y.batch(biY),
                                               coupling_D.batch(biD), backend);
        }
        // chol(D)
        if (is_active(l + 1, biD)) {
            PRINTLN("chol D{}", biD);
            GUANAQO_TRACE("Factor D", biD);
            compact_blas::xpotrf(coupling_D.batch(biD), backend);
        }
    }

    void factor_l0(const index_t ti) {
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t biI        = sub_wrap_PmV(ti, 1);
        const index_t biA        = ti;
        const index_t k          = ti * num_stages;
        const index_t kI         = biI * num_stages;
        const index_t kA         = biA * num_stages;
        const auto be            = backend;
        const bool x_lanes       = ti == 0; // first stage wraps around
        // Coupling equation to previous stage is eliminated after coupling
        // equation to next stage for odd threads, vice versa for even threads.
        const bool I_below_A = (ti & 1) == 1;
        // Update the subdiagonal blocks U and Y of the coupling equations
        [[maybe_unused]] VecReg vec_curr{vl, k, N >> lvl, N},
            vecA{vl, kA, N >> lvl, N}, vecI{vl, kI, N >> lvl, N};
        auto DiI = coupling_D.batch(biI);
        auto DiA = coupling_D.batch(biA);
        auto Âi  = riccati_ÂB̂.batch(ti).middle_cols(nx * (num_stages - 1), nx);
        auto ÂB̂i = riccati_ÂB̂.batch(ti).right_cols(nx + nu * num_stages);
        auto Q̂i  = riccati_R̂ŜQ̂.batch(ti).bottom_right(nx, nx);
        {
            GUANAQO_TRACE("Invert Q", biI);
            compact_blas::xtrtri_T_copy_ref(Q̂i, DiI);
        }
        if (I_below_A) {
            // Top block is A → column index is row index of A (biA)
            // Target block in cyclic part is U in column λ(kA)
            GUANAQO_TRACE("Compute first U", biA);
            compact_blas::xtrmm_LUNN_T_neg_ref(DiI, Âi, coupling_U.batch(biA));
        } else {
            // Top block is I → column index is row index of I (biI)
            // Target block in cyclic part is Y in column λ(kI)
            GUANAQO_TRACE("Compute first Y", biI);
            x_lanes ? compact_blas::xtrmm_RUTN_neg_shift(Âi, DiI,
                                                         coupling_Y.batch(biI))
                    : compact_blas::xtrmm_RUTN_neg_ref(Âi, DiI,
                                                       coupling_Y.batch(biI));
        }
        // Each column of the cyclic part with coupling equations is updated by
        // two threads: one for the forward, and one for the backward coupling.
        // Update the diagonal blocks of the coupling equations,
        // first forward in time ...
        {
            GUANAQO_TRACE("Compute L⁻ᵀL⁻¹", biI);
            x_lanes ? compact_blas::xtrtrsyrk_UL_shift(DiI, DiI)
                    : compact_blas::xtrtrsyrk_UL(DiI, DiI);
        }
        // Then synchronize to make sure there are no two threads updating the
        // same diagonal block.
        barrier();
        // And finally backward in time, optionally merged with factorization.
        const bool ready_to_factor = (ti & 1) == 1;
        {
            GUANAQO_TRACE("Compute (BA)(BA)ᵀ", biA);
            compact_blas::xsyrk_add(ÂB̂i, DiA, be);
        }
        if (ready_to_factor) {
            GUANAQO_TRACE("Factor D", biA);
            compact_blas::xpotrf(coupling_D.batch(biA), backend);
        }
    }

    // Performs Riccati recursion and then factors level l=0 of
    // coupling equations + propagates the subdiagonal blocks to level l=1.
    void factor_riccati(index_t ti) {
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t bi0        = ti * num_stages;
        const index_t k0         = ti * num_stages;
        const index_t nux        = nu + nx;
        const auto be            = backend;
        PRINTLN("\nThread #{}", ti);
        auto R̂ŜQ̂ = riccati_R̂ŜQ̂.batch(ti);
        auto B̂   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
        auto Â   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
        auto BAᵀ = riccati_BAᵀ.batch(ti);
        // Copy B and A from the last stage
        {
            GUANAQO_TRACE("Riccati init", k0);
            compact_blas::xcopy(data_BA.batch(bi0).left_cols(nu),
                                B̂.left_cols(nu));
            compact_blas::xcopy(data_BA.batch(bi0).right_cols(nx),
                                Â.left_cols(nx));
            compact_blas::xcopy_L(data_RSQ.batch(bi0), R̂ŜQ̂.left_cols(nux));
        }
        for (index_t i = 0; i < num_stages; ++i) {
            index_t k = sub_wrap_N(k0, i);
            PRINTLN("  Riccati factor QRS{}", VecReg{vl, k, N >> lvl, N});
            auto R̂ŜQ̂i = R̂ŜQ̂.middle_cols(i * nux, nux);
            auto R̂Ŝi  = R̂ŜQ̂i.left_cols(nu);
            auto R̂i   = R̂Ŝi.top_rows(nu);
            auto Ŝi   = R̂Ŝi.bottom_rows(nx);
            auto Q̂i   = R̂ŜQ̂i.bottom_right(nx, nx);
            auto B̂i   = B̂.middle_cols(i * nu, nu);
            auto Âi   = Â.middle_cols(i * nx, nx);
            {
                GUANAQO_TRACE("Riccati QRS", k);
                // Factor R̂, update Ŝ, and compute LB̂ = B̂ LR̂⁻ᵀ
                compact_blas::xpotrf(R̂Ŝi, be);        // ┐
                compact_blas::xtrsm_RLTN(R̂i, B̂i, be); // ┘
                // Update Â = Ã - LB̂ LŜᵀ
                compact_blas::xgemm_NT_sub(B̂i, Ŝi, Âi, be);
                // Update and factor Q̂ = Q̃ - LŜ LŜᵀ
                compact_blas::xsyrk_sub(Ŝi, Q̂i, be); // ┐
                compact_blas::xpotrf(Q̂i, be);        // ┘
            }
            if (i + 1 < num_stages) {
                // Copy next B and A
                [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
                GUANAQO_TRACE("Riccati update AB", k_next);
                PRINTLN("  Riccati update AB{}",
                        VecReg{vl, k_next, N >> lvl, N});
                const auto bi_next = bi0 + i + 1;
                auto BAᵀi          = BAᵀ.middle_cols(i * nx, nx);
                auto BAi           = data_BA.batch(bi_next);
                auto Bi = BAi.left_cols(nu), Ai = BAi.right_cols(nx);
                // Compute next B̂ and Â
                auto B̂_next = B̂.middle_cols((i + 1) * nu, nu);
                auto Â_next = Â.middle_cols((i + 1) * nx, nx);
                compact_blas::xgemm(Âi, Bi, B̂_next, be);
                compact_blas::xgemm(Âi, Ai, Â_next, be);
                // Riccati update
                auto R̂ŜQ̂_next = R̂ŜQ̂.middle_cols((i + 1) * nux, nux);
                compact_blas::xcopy_T(data_BA.batch(bi_next), BAᵀi);
                compact_blas::xtrmm_RLNN(BAᵀi, Q̂i, BAᵀi, be);
                compact_blas::xcopy_L(data_RSQ.batch(bi_next), R̂ŜQ̂_next); // ┐
                compact_blas::xsyrk_add(BAᵀi, R̂ŜQ̂_next, be);              // ┘
            } else {
                // Compute LÂ = Ã LQ⁻ᵀ
                GUANAQO_TRACE("Riccati last", k);
                compact_blas::xtrsm_RLTN(Q̂i, Âi, be);
            }
        }
    }

    void run() {
        for (auto &c : counters)
            c.value.store(0, std::memory_order_relaxed);
        for (auto &c : counters_UY)
            c.value.store(0, std::memory_order_relaxed);
        coupling_D.set_constant(0); // TODO
        const index_t N = dim.N_horiz;
        KOQKATOO_ASSERT(((N >> lP) << lP) == N);
        koqkatoo::foreach_thread([this](index_t ti, index_t P) {
            if (P < (1 << (lP - lvl)))
                throw std::logic_error("Incorrect number of threads");
            if (ti >= (1 << (lP - lvl)))
                return;
            factor_riccati(ti);
            factor_l0(ti);
            for (index_t l = 0; l < lP - lvl; ++l) {
                barrier();
                const index_t offset = 1 << l;
                const auto bi        = sub_wrap_PmV(ti, offset);
                const auto biU       = ti;
                if (is_active(l, bi))
                    process_active(l, bi);
                else if (is_active(l, biU))
                    process_active_secondary(l, biU);
                else
                    barrier();
            }
        });
    }

    index_t get_linear_batch_offset(index_t biA) {
        const auto levA = biA > 0 ? get_level(biA) : lP;
        const auto levP = lP - lvl;
        if (levA >= levP)
            return (((1 << levP) - 1) << (lP - levP)) + (biA >> levP);
        return (((1 << levA) - 1) << (lP - levA)) + get_index_in_level(biA);
    }

    auto build_sparse(const koqkatoo::ocp::LinearOCPStorage &ocp) {
        std::vector<std::tuple<index_t, index_t, real_t>> tuples;

        auto [N, nx, nu, ny, ny_N] = ocp.dim;
        const index_t nux = nu + nx, nuxx = nux + nx;
        const index_t vstride    = N >> lvl;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t num_proc   = 1 << (lP - lvl);
        const index_t sλ         = N * nuxx - (nx << lP);
        for (index_t vi = 0; vi < vl; ++vi) {
            const index_t sv = vi * num_proc * (nuxx * num_stages - nx);
            for (index_t ti = 0; ti < num_proc; ++ti) {
                const index_t k0 = ti * num_stages + vi * vstride;
                const auto biA   = ti + vi * num_proc;
                const auto biI   = sub_wrap_P(biA, 1);
                const auto sλA   = sλ + nx * get_linear_batch_offset(biA);
                const auto sλI   = sλ + nx * get_linear_batch_offset(biI);
                std::println(
                    "k={:<2}  biA={:<2}  oA={:<2}  biI={:<2}  oI={:<2}", k0,
                    biA, get_linear_batch_offset(biA), biI,
                    get_linear_batch_offset(biI));
                // TODO: handle case if lev > or >= lP - lvl
                for (index_t i = 0; i < num_stages; ++i) {
                    const index_t k = sub_wrap_N(k0, i);
                    index_t s = sv + ti * (nuxx * num_stages - nx) + nuxx * i;
                    for (index_t c = 0; c < nu; ++c) {
                        for (index_t r = c; r < nu; ++r)
                            tuples.emplace_back(s + r, s + c, ocp.R(k)(r, c));
                        if (k > 0)
                            for (index_t r = 0; r < nx; ++r)
                                tuples.emplace_back(s + r + nu, s + c,
                                                    ocp.S_trans(k)(r, c));
                        if (i == 0) {
                            for (index_t r = 0; r < nx; ++r)
                                tuples.emplace_back(sλA + r, s + c,
                                                    ocp.B(k0)(r, c));
                        }
                    }
                    for (index_t c = 0; c < nx; ++c) {
                        for (index_t r = c; r < nx; ++r)
                            tuples.emplace_back(s + r + nu, s + c + nu,
                                                ocp.Q(k == 0 ? N : k)(r, c));
                        if (i + 1 < num_stages)
                            tuples.emplace_back(s + c + nux, s + c + nu, -1);
                        else
                            tuples.emplace_back(sλI + c, s + c + nu, -1);
                        if (i == 0 && k > 0) {
                            for (index_t r = 0; r < nx; ++r)
                                tuples.emplace_back(sλA + r, s + c + nu,
                                                    ocp.A(k0)(r, c));
                        }
                    }
                    if (i > 0) {
                        for (index_t c = 0; c < nx; ++c) {
                            for (index_t r = 0; r < nu; ++r)
                                tuples.emplace_back(s + r, s - nx + c,
                                                    ocp.B(k)(c, r));
                            for (index_t r = 0; r < nx; ++r)
                                tuples.emplace_back(s + nu + r, s - nx + c,
                                                    ocp.A(k)(c, r));
                        }
                    }
                }
            }
        }
        return tuples;
    }

    auto build_sparse_factor() {
        std::vector<std::tuple<index_t, index_t, real_t>> tuples;
        auto [N, nx, nu, ny, ny_N] = dim;
        const index_t nux = nu + nx, nuxx = nux + nx;
        const index_t vstride    = N >> lvl;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t num_proc   = 1 << (lP - lvl);
        const index_t sλ         = N * nuxx - (nx << lP);
        matrix invQᵀ{{
            .depth = 1 << lP,
            .rows  = dim.nx,
            .cols  = (dim.N_horiz >> lP) * dim.nx,
        }};
        matrix AinvQᵀ{{
            .depth = 1 << lP,
            .rows  = dim.nx,
            .cols  = (dim.N_horiz >> lP) * dim.nx,
        }};
        for (index_t ti = 0; ti < num_proc; ++ti) {
            for (index_t i = 0; i < num_stages; ++i) {
                auto RSQ   = riccati_R̂ŜQ̂.batch(ti);
                auto RSQi  = RSQ.middle_cols(i * nux, nux);
                auto Qi    = RSQi.bottom_right(nx, nx);
                auto Â     = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
                auto Âi    = Â.middle_cols(i * nx, nx);
                auto iQᵀ   = invQᵀ.batch(ti);
                auto iQiᵀ  = iQᵀ.middle_cols(i * nx, nx);
                auto AiQᵀ  = AinvQᵀ.batch(ti);
                auto AiQiᵀ = AiQᵀ.middle_cols(i * nx, nx);
                compact_blas::xtrtri_T_copy_ref(Qi, iQiᵀ);
                compact_blas::xcopy(Âi, AiQiᵀ);
                if (i + 1 < num_stages) // Final block is already Â LQ⁻ᵀ
                    compact_blas::xtrsm_RLTN(Qi, AiQiᵀ, backend);
            }
        }
        for (index_t vi = 0; vi < vl; ++vi) {
            const index_t sv = vi * num_proc * (nuxx * num_stages - nx);
            for (index_t ti = 0; ti < num_proc; ++ti) {
                const index_t k0 = ti * num_stages + vi * vstride;
                const auto biA   = ti + vi * num_proc;
                const auto biI   = sub_wrap_P(biA, 1);
                const auto sλA   = sλ + nx * get_linear_batch_offset(biA);
                const auto sλI   = sλ + nx * get_linear_batch_offset(biI);
                auto B̂   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
                auto R̂ŜQ̂ = riccati_R̂ŜQ̂.batch(ti);
                // TODO: handle case if lev > or >= lP - lvl
                for (index_t i = 0; i < num_stages; ++i) {
                    [[maybe_unused]] const index_t k = sub_wrap_N(k0, i);
                    index_t s  = sv + ti * (nuxx * num_stages - nx) + nuxx * i;
                    auto B̂i    = B̂.middle_cols(i * nu, nu);
                    auto R̂ŜQ̂i  = R̂ŜQ̂.middle_cols(i * nux, nux);
                    auto RSi   = R̂ŜQ̂i(vi).left_cols(nu);
                    auto Qi    = R̂ŜQ̂i(vi).bottom_right(nx, nx);
                    auto iQᵀ   = invQᵀ.batch(ti);
                    auto iQiᵀ  = iQᵀ.middle_cols(i * nx, nx);
                    auto AiQᵀ  = AinvQᵀ.batch(ti);
                    auto AiQiᵀ = AiQᵀ.middle_cols(i * nx, nx);
                    auto BAᵀ   = riccati_BAᵀ.batch(ti);
                    if (i > 0) {
                        auto BAᵀi     = BAᵀ.middle_cols((i - 1) * nx, nx);
                        auto iQprevᵀ  = iQᵀ.middle_cols((i - 1) * nx, nx);
                        auto AiQprevᵀ = AiQᵀ.middle_cols((i - 1) * nx, nx);
                        for (index_t c = 0; c < nx; ++c) {
                            for (index_t r = 0; r <= c; ++r)
                                tuples.emplace_back(s - nx + r, s - nx + c,
                                                    -iQprevᵀ(vi)(r, c));
                            for (index_t r = 0; r < nux; ++r)
                                tuples.emplace_back(s + r, s - nx + c,
                                                    BAᵀi(vi)(r, c));
                            for (index_t r = 0; r < nx; ++r)
                                tuples.emplace_back(sλA + r, s - nx + c,
                                                    AiQprevᵀ(vi)(r, c));
                        }
                    }
                    for (index_t c = 0; c < nu; ++c) {
                        for (index_t r = c; r < nux; ++r)
                            tuples.emplace_back(s + r, s + c, RSi(r, c));
                        for (index_t r = 0; r < nx; ++r)
                            tuples.emplace_back(sλA + r, s + c, B̂i(vi)(r, c));
                    }
                    for (index_t c = 0; c < nx; ++c) {
                        for (index_t r = c; r < nx; ++r) {
                            tuples.emplace_back(s + r + nu, s + c + nu,
                                                Qi(r, c));
                        }
                        if (i + 1 < num_stages)
                            for (index_t r = 0; r <= c; ++r)
                                tuples.emplace_back(s + r + nux, s + c + nu,
                                                    -iQiᵀ(vi)(r, c));
                        else
                            for (index_t r = 0; r <= c; ++r)
                                tuples.emplace_back(sλI + r, s + c + nu,
                                                    -iQiᵀ(vi)(r, c));
                        for (index_t r = 0; r < nx; ++r)
                            tuples.emplace_back(sλA + r, s + c + nu,
                                                AiQiᵀ(vi)(r, c));
                    }
                }
            }
        }
        index_t s               = sλ;
        const auto cyclic_block = [&](index_t i, index_t offset) {
            const index_t sY = sλ + nx * get_linear_batch_offset(i + offset);
            const index_t sU = sλ + nx * get_linear_batch_offset(i - offset);
            std::println("{:>2}: {:<2} {:<2} {:<2}", i, s, sY, sU);
            const index_t bi = i % (1 << (lP - lvl));
            const index_t vi = i / (1 << (lP - lvl));
            for (index_t c = 0; c < nx; ++c) {
                for (index_t r = c; r < nx; ++r)
                    tuples.emplace_back(s + r, s + c,
                                        coupling_D.batch(bi)(vi)(r, c));
                if (i + offset < (1 << lP))
                    for (index_t r = 0; r < nx; ++r)
                        tuples.emplace_back(sY + r, s + c,
                                            coupling_Y.batch(bi)(vi)(r, c));
                for (index_t r = 0; r < nx; ++r)
                    tuples.emplace_back(sU + r, s + c,
                                        coupling_U.batch(bi)(vi)(r, c));
            }
            s += nx;
        };
        const auto cyclic_block_final = [&](index_t i, index_t offset) {
            const index_t sY = sλ + nx * get_linear_batch_offset(i + offset);
            std::println("{:>2}: {:<2} {:<2}", i, s, sY);
            const index_t bi = i % (1 << (lP - lvl));
            const index_t vi = i / (1 << (lP - lvl));
            for (index_t c = 0; c < nx; ++c) {
                for (index_t r = c; r < nx; ++r)
                    tuples.emplace_back(s + r, s + c,
                                        coupling_D.batch(bi)(vi)(r, c));
                if (i + offset < (1 << lP))
                    for (index_t r = 0; r < nx; ++r)
                        tuples.emplace_back(sY + r, s + c,
                                            coupling_Y.batch(bi)(vi)(r, c));
            }
            s += nx;
        };
        if (lP != lvl) {
            for (index_t i = 0; i < (1 << (lP - 1)); ++i)
                cyclic_block(2 * i + 1, 1);
            for (index_t l = 1; l < lP - lvl; ++l) {
                index_t offset = 1 << l;
                index_t stride = offset << 1;
                for (index_t i = offset; i < (1 << lP); i += stride)
                    cyclic_block(i, offset);
            }
        }
        for (index_t i = 0; i < (1 << lP); i += (1 << (lP - lvl))) {
            cyclic_block_final(i, 1 << (lP - lvl));
        }
        return tuples;
    }

    auto build_sparse_diag() {
        std::vector<std::tuple<index_t, index_t, real_t>> tuples;
        auto [N, nx, nu, ny, ny_N] = dim;
        const index_t nux = nu + nx, nuxx = nux + nx;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t num_proc   = 1 << (lP - lvl);
        const index_t sλ         = N * nuxx - (nx << lP);
        for (index_t vi = 0; vi < vl; ++vi) {
            const index_t sv = vi * num_proc * (nuxx * num_stages - nx);
            for (index_t ti = 0; ti < num_proc; ++ti) {
                for (index_t i = 0; i < num_stages; ++i) {
                    index_t s = sv + ti * (nuxx * num_stages - nx) + nuxx * i;
                    if (i > 0)
                        for (index_t c = 0; c < nx; ++c)
                            tuples.emplace_back(s - nx + c, s - nx + c, -1);
                    for (index_t c = 0; c < nu; ++c)
                        tuples.emplace_back(s + c, s + c, 1);
                    for (index_t c = 0; c < nx; ++c)
                        tuples.emplace_back(s + c + nu, s + c + nu, 1);
                }
            }
        }
        for (index_t i = 0; i < 1 << lP; ++i)
            for (index_t r = 0; r < nx; ++r)
                tuples.emplace_back(sλ + nx * i + r, sλ + nx * i + r, -1);
        return tuples;
    }
};

} // namespace koqkatoo::ocp::test

#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/linalg-compact/mkl.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/thread-pool.hpp>
#include <guanaqo/eigen/span.hpp>
#include <guanaqo/trace.hpp>
#include <koqkatoo-version.h>

#include <Eigen/Eigen>

#include <chrono>
#include <filesystem>
#include <fstream>

using koqkatoo::index_t;
using koqkatoo::real_t;

const int log_n_threads = 3; // TODO
TEST(NewCyclic, scheduling) {
    using namespace koqkatoo::ocp;

    KOQKATOO_OMP_IF(omp_set_num_threads(1 << log_n_threads));
    koqkatoo::pool_set_num_threads(1 << log_n_threads);
    koqkatoo::fork_set_num_threads(1 << log_n_threads);
    GUANAQO_IF_ITT(koqkatoo::foreach_thread([](index_t i, index_t) {
        __itt_thread_set_name(std::format("OMP({})", i).c_str());
    }));

    const index_t lP = log_n_threads + test::CyclicOCPSolver::lvl;
    OCPDim dim{.N_horiz = 1 << lP, .nx = 40, .nu = 30, .ny = 0, .ny_N = 0};
    auto ocp = generate_random_ocp(dim);
    test::CyclicOCPSolver solver{.dim = dim, .lP = lP};
    solver.initialize(ocp);
    for (int i = 0; i < 100; ++i)
        solver.run();
#if GUANAQO_WITH_TRACING
    guanaqo::trace_logger.reset();
#endif
    solver.run();

#if GUANAQO_WITH_TRACING
    {
        const auto N     = solver.dim.N_horiz;
        const auto VL    = solver.vl;
        std::string name = std::format("factor_cyclic_new.csv");
        std::filesystem::path out_dir{"traces"};
        out_dir /= *koqkatoo_commit_hash ? koqkatoo_commit_hash : "unknown";
        out_dir /= KOQKATOO_MKL_IF_ELSE("mkl", "openblas");
        out_dir /= std::format("nx={}-nu={}-ny={}-N={}-thr={}-vl={}",
                               solver.dim.nx, solver.dim.nu, solver.dim.ny, N,
                               1 << log_n_threads, VL);
        std::filesystem::create_directories(out_dir);
        std::ofstream csv{out_dir / name};
        guanaqo::TraceLogger::write_column_headings(csv) << '\n';
        for (const auto &log : guanaqo::trace_logger.get_logs())
            csv << log << '\n';
        std::cout << out_dir << std::endl;
    }
#endif

    std::ofstream f1("sparse.csv");
    if (f1) {
        auto sp = solver.build_sparse(ocp);
        for (auto [r, c, x] : sp)
            f1 << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
    std::ofstream f2("sparse_factor.csv");
    if (f2) {
        auto sp = solver.build_sparse_factor();
        for (auto [r, c, x] : sp)
            f2 << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
    std::ofstream f3("sparse_diag.csv");
    if (f3) {
        auto sp = solver.build_sparse_diag();
        for (auto [r, c, x] : sp)
            f3 << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
}
