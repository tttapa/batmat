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

namespace stdx = std::experimental;

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
    const index_t lP = 5; ///< log2(P), logarithm of the number of processors
    // const index_t N  = dim.N_horiz;
    const index_t ln = lP - lvl;
    const index_t n  = index_t{1} << ln;

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

    [[nodiscard]] static constexpr index_t get_index_in_level(index_t i) {
        auto l = get_level(i);
        return i >> (l + 1);
    }

    [[nodiscard]] constexpr index_t get_level_width(index_t l) const {
        assert(l < ln);
        return index_t{1} << (ln - l - 1);
    }

    [[nodiscard]] constexpr index_t get_batch_index(index_t i) const {
        assert(i < n);
        if (i == 0)
            return n - 1;
        auto l  = get_level(i);
        auto il = get_index_in_level(i);
        return il + (1 << ln) - (1 << (ln - l));
    }

    struct join_counter_t {
        alignas(64) std::atomic<uint32_t> value{};
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

    // Counters
    //     1 = L⁻ᵀL⁻¹ has been added to D
    //     2 = (BA)(BA)ᵀ has been added to D,
    //          and if odd, DUY has been factored, and next UY is available
    //     2l + 1 = YYᵀ has been added to D
    //     2l + 2 = UUᵀ has been added to D,
    //          and if odd, DUY have been factored, and next UY is available

    void factor_coupling(index_t l, index_t ti) {
        const index_t bi = thread2batch(l, ti);
        bool last        = bi != 0 && get_level(bi) < l;
        std::println("Thread #{}: {} ({})", ti, bi, last);
        if (last)
            return;
        prop_diag_fwd(l, bi);
        prop_diag_rev(l, bi);
        if ((bi >> l) & 1) {
            std::println("odd");
            prop_subdiag(l, bi);
        }
    }
    void prop_diag_fwd(const index_t l, const index_t bi) {
        KOQKATOO_ASSERT(l < lP - lvl);
        const index_t N = dim.N_horiz;
        // Updates to batch 0 wrap around
        const bool cross_lanes = bi == 0;
        const index_t bi_prev = cross_lanes ? (1 << (lP - lvl)) - (1 << (l - 1))
                                            : bi - (1 << (l - 1));
        // Implicit synchronization because Y was computed on the same thread
        KOQKATOO_ASSERT(batch2thread(l - 1, bi_prev) == batch2thread(l, bi));
        index_t k = bi * N >> lP, k_prev = bi_prev * N >> lP;
        // Wait for Y(bi_prev) to be computed.
        std::println("  wait   ({}) D[{}]{}", 2 * l, bi_prev,
                     VecReg{vl, k_prev, N >> lvl, N});
        // Wait for the previous update to D to complete (different thread)
        std::println("  wait   ({}) D[{}]{}", 2 * l, bi,
                     VecReg{vl, k, N >> lvl, N});
        std::println("  YYᵀ[{}]{}  ->  D[{}]{}{}", bi_prev,
                     VecReg{vl, k_prev, N >> lvl, N}, bi,
                     VecReg{vl, k, N >> lvl, N}, cross_lanes ? "  (×)" : "");
        cross_lanes ? compact_blas::xsyrk_sub_shift(coupling_Y.batch(bi_prev),
                                                    coupling_D.batch(bi))
                    : compact_blas::xsyrk_sub(coupling_Y.batch(bi_prev),
                                              coupling_D.batch(bi), backend);
        std::println("  notify   ({}) D[{}]{}", 2 * l + 1, bi,
                     VecReg{vl, k, N >> lvl, N});
    }
    void prop_diag_rev(index_t l, index_t bi) {
        const index_t N       = dim.N_horiz;
        const index_t bi_next = bi + (1 << (l - 1));
        index_t k = bi * N >> lP, k_next = bi_next * N >> lP;
        // Wait for U(bi_next) to be computed.
        std::println("  wait   ({}) D[{}]{}", 2 * l, bi_next,
                     VecReg{vl, k_next, N >> lvl, N});
        std::println("  wait   ({}) D[{}]{}", 2 * l + 1, bi,
                     VecReg{vl, k, N >> lvl, N});
        std::println("  UUᵀ[{}]{}  ->  D[{}]{}", bi_next,
                     VecReg{vl, k_next, N >> lvl, N}, bi,
                     VecReg{vl, k, N >> lvl, N});
        compact_blas::xsyrk_sub(coupling_U.batch(bi_next), coupling_D.batch(bi),
                                backend);
    }
    void prop_subdiag(index_t l, index_t bi) {
        const index_t N       = dim.N_horiz;
        const bool U_below_Y  = ((bi >> l) & 3) == 1 && l + 1 != lP - lvl;
        const index_t bi_next = bi + (1 << l), bi_prev = bi - (1 << l);
        index_t k     = bi * N >> lP;
        auto Di       = coupling_D.batch(bi);
        const auto be = backend;
        std::println("  factor D[{}]{}", bi, VecReg{vl, k, N >> lvl, N});
        compact_blas::xpotrf(Di, be);                           // ┐
        compact_blas::xtrsm_RLTN(Di, coupling_U.batch(bi), be); // │
        compact_blas::xtrsm_RLTN(Di, coupling_Y.batch(bi), be); // ┘
        if (U_below_Y) {
            index_t k_next = bi_next * N >> lP;
            std::println("  UYᵀ[{}]{}  ->  U[{}]{}", bi,
                         VecReg{vl, k, N >> lvl, N}, bi_next,
                         VecReg{vl, k_next, N >> lvl, N});
            compact_blas::xgemm_NT_neg(coupling_U.batch(bi),
                                       coupling_Y.batch(bi),
                                       coupling_U.batch(bi_next), be);
        } else {
            index_t k_prev = bi_prev * N >> lP;
            std::println("  YUᵀ[{}]{}  ->  Y[{}]{}", bi,
                         VecReg{vl, k, N >> lvl, N}, bi_prev,
                         VecReg{vl, k_prev, N >> lvl, N});
            compact_blas::xgemm_NT_neg(coupling_Y.batch(bi),
                                       coupling_U.batch(bi),
                                       coupling_Y.batch(bi_prev), be);
        }
        std::println("  notify ({})  D[{}]{}", 2 * l + 2, bi,
                     VecReg{vl, k, N >> lvl, N});
    }

    void factor_l0(const index_t ti) {
        GUANAQO_TRACE("factor_l0", ti);
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t biI        = sub_wrap_PmV(ti, 1);
        const index_t biA        = ti;
        const index_t k          = ti * num_stages;
        const index_t kI         = biI * num_stages;
        const index_t kA         = biA * num_stages;
        const auto be            = backend;
        const bool cross_lanes   = ti == 0; // first stage wraps around
        // Coupling equation to previous stage is eliminated after coupling
        // equation to next stage for odd threads, vice versa for even threads.
        const bool I_below_A = (ti & 1) == 1;
        // Update the subdiagonal blocks U and Y of the coupling equations
        VecReg vec_curr{vl, k, N >> lvl, N}, vecA{vl, kA, N >> lvl, N},
            vecI{vl, kI, N >> lvl, N};
        auto DiI = coupling_D.batch(biI);
        auto DiA = coupling_D.batch(biA);
        auto Âi  = riccati_ÂB̂.batch(ti).middle_cols(nx * (num_stages - 1), nx);
        auto ÂB̂i = riccati_ÂB̂.batch(ti).right_cols(nx + nu * num_stages);
        auto Q̂i  = riccati_R̂ŜQ̂.batch(ti).bottom_right(nx, nx);
        compact_blas::xtrtri_T_copy_ref(Q̂i, DiI);
        if (I_below_A) {
            // Top block is A → column index is row index of A (biA)
            // Target block in cyclic part is U in column λ(kA)
            std::println("  L⁻ᵀÂᵀ [{}]{}  ->  U[{}]{}", ti, vec_curr, biA,
                         vecA);
            // TODO: trmm_LUNN_T
            compact_blas::xgemm_NT(DiI, Âi, coupling_U.batch(biA), be);
        } else {
            // Top block is I → column index is row index of I (biI)
            // Target block in cyclic part is Y in column λ(kI)
            std::println("  ÂL⁻¹  [{}]{}  ->  Y[{}]{}{}", ti, vec_curr, biI,
                         vecI, cross_lanes ? "  (×)" : "");
            // TODO: trmm_N_RUTN
            cross_lanes
                ? compact_blas::xgemm_NT_shift(Âi, DiI, coupling_Y.batch(biI))
                : compact_blas::xgemm_NT(Âi, DiI, coupling_Y.batch(biI), be);
        }
        // Each column of the cyclic part with coupling equations is updated by
        // two threads: one for the forward, and one for the backward coupling.
        // Update the diagonal blocks of the coupling equations,
        // first forward in time ...
        std::println("  L⁻ᵀL⁻¹[{}]{}  ->  D[{}]{}{}", ti, vec_curr, biI, vecI,
                     cross_lanes ? "  (×)" : "");
        cross_lanes ? compact_blas::xtrtrsyrk_UL_shift(DiI, DiI)
                    : compact_blas::xtrtrsyrk_UL(DiI, DiI);
        // Then synchronize to make sure there are no two threads updating the
        // same diagonal block.
        std::println("  notify (1)  D[{}]", biI);
        std::println("  wait   (1)  D[{}]", biA);
        // And finally backward in time, optionally merged with factorization.
        const bool ready_to_factor = (ti & 1) == 1;
        compact_blas::xsyrk_add(ÂB̂i, DiA, be);
        std::println("  {}  D[{}]{}", ready_to_factor ? "factor" : "update",
                     biA, vecA);
        if (ready_to_factor) {
            std::println("---");
            prop_subdiag(0, ti);
        } else {
            std::println("  notify (2)  D[{}]{}", biA, vecA);
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
        std::println("\nThread #{}", ti);
        auto R̂ŜQ̂ = riccati_R̂ŜQ̂.batch(ti);
        auto B̂   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
        auto Â   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
        auto BAᵀ = riccati_BAᵀ.batch(ti);
        // Copy B and A from the last stage
        compact_blas::xcopy(data_BA.batch(bi0).left_cols(nu), B̂.left_cols(nu));
        compact_blas::xcopy(data_BA.batch(bi0).right_cols(nx), Â.left_cols(nx));
        for (index_t i = 0; i < num_stages; ++i) {
            index_t k = sub_wrap_N(k0, i);
            std::println("  Riccati factor QRS{}", VecReg{vl, k, N >> lvl, N});
            auto R̂ŜQ̂i = R̂ŜQ̂.middle_cols(i * nux, nux);
            auto R̂Ŝi  = R̂ŜQ̂i.left_cols(nu);
            auto R̂i   = R̂Ŝi.top_rows(nu);
            auto Ŝi   = R̂Ŝi.bottom_rows(nx);
            auto Q̂i   = R̂ŜQ̂i.bottom_right(nx, nx);
            auto B̂i   = B̂.middle_cols(i * nu, nu);
            auto Âi   = Â.middle_cols(i * nx, nx);
            // Factor R̂, update Ŝ, and compute LB̂ = B̂ LR̂⁻ᵀ
            compact_blas::xcopy_L(data_RSQ.batch(bi0 + i), R̂ŜQ̂i); // ┐
            compact_blas::xpotrf(R̂i, be);                         // │
            compact_blas::xtrsm_RLTN(R̂i, B̂i, be);                 // ┘
            // Update Â = Ã - LB̂ LŜᵀ
            compact_blas::xgemm_NT_sub(B̂i, Ŝi, Âi, be);
            // Update and factor Q̂ = Q̃ - LŜ LŜᵀ
            compact_blas::xsyrk_sub(Ŝi, Q̂i, be); // ┐
            compact_blas::xpotrf(Q̂i, be);        // ┘
            if (i + 1 < num_stages) {
                // Copy next B and A
                const auto k_next = sub_wrap_N(k, 1);
                std::println("  Riccati update AB{}",
                             VecReg{vl, k_next, N >> lvl, N});
                const auto bi_next = bi0 + i + 1;
                auto BAᵀi          = BAᵀ.middle_cols(i * nx, nx);
                auto Bᵀi = BAᵀi.top_rows(nu), Aᵀi = BAᵀi.bottom_rows(nx);
                compact_blas::xcopy_T(data_BA.batch(bi_next), BAᵀi);
                // Compute next B̂ and Â
                auto B̂_next = B̂.middle_cols((i + 1) * nu, nu);
                auto Â_next = Â.middle_cols((i + 1) * nx, nx);
                compact_blas::xgemm_NT(Âi, Bᵀi, B̂_next, be);
                compact_blas::xgemm_NT(Âi, Aᵀi, Â_next, be);
                // Riccati update
                auto R̂ŜQ̂_next = R̂ŜQ̂.middle_cols((i + 1) * nux, nux);
                compact_blas::xtrmm_RLNN(BAᵀi, Q̂i, BAᵀi, be);
                compact_blas::xsyrk_add(BAᵀi, R̂ŜQ̂_next, be);
            } else {
                // Compute LÂ = Ã LQ⁻ᵀ
                compact_blas::xtrsm_RLTN(Q̂i, Âi, be);
            }
        }
    }

    void run() {
        const index_t N = dim.N_horiz;
        KOQKATOO_ASSERT(((N >> lP) << lP) == N);
        // koqkatoo::foreach_thread([this](index_t ti) {
        // // TODO
        // });
        for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
            factor_riccati(ti);
            factor_l0(ti);
        }
        for (index_t l = 1; l < lP - lvl; ++l) {
            std::println();
            for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
                // std::println("\nThread {}, Level {}", ti, l);
                factor_coupling(l, ti);
            }
        }
    }
};

} // namespace koqkatoo::ocp::test

#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/linalg-compact/mkl.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/thread-pool.hpp>
#include <guanaqo/eigen/span.hpp>
#include <guanaqo/trace.hpp>
#include <koqkatoo-version.h>

#include <Eigen/Eigen>

#include <chrono>
#include <filesystem>
#include <fstream>

const int n_threads = 8; // TODO
TEST(NewCyclic, scheduling) {
    using namespace koqkatoo::ocp;

    KOQKATOO_OMP_IF(omp_set_num_threads(n_threads));
    koqkatoo::pool_set_num_threads(n_threads);
    koqkatoo::fork_set_num_threads(n_threads);
    GUANAQO_IF_ITT(koqkatoo::foreach_thread([](index_t i, index_t) {
        __itt_thread_set_name(std::format("OMP({})", i).c_str());
    }));

#if GUANAQO_WITH_TRACING
    koqkatoo::trace_logger.reset();
#endif

    OCPDim dim{.N_horiz = 96, .nx = 40, .nu = 30, .ny = 0, .ny_N = 0};
    test::CyclicOCPSolver solver{.dim = dim};
    solver.run();

#if GUANAQO_WITH_TRACING
    {
        std::string name = std::format("factor_cyclic.csv");
        std::filesystem::path out_dir{"traces"};
        out_dir /= *koqkatoo_commit_hash ? koqkatoo_commit_hash : "unknown";
        out_dir /= KOQKATOO_MKL_IF_ELSE("mkl", "openblas");
        out_dir /= std::format("N={}-thr={}-vl={}{}", N, n_threads, VL);
        std::filesystem::create_directories(out_dir);
        std::ofstream csv{out_dir / name};
        koqkatoo::TraceLogger::write_column_headings(csv) << '\n';
        for (const auto &log : koqkatoo::trace_logger.get_logs())
            csv << log << '\n';
        std::cout << out_dir << std::endl;
    }
#endif
}
