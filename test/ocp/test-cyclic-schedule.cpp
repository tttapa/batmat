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
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <guanaqo/trace.hpp>

#include <experimental/simd>
#include <guanaqo/print.hpp>
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <print>
#include <random>
#include <stdexcept>
#include <string_view>
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

#define USE_JACOBI_PREC 0

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

template <index_t VL = 4>
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

    using simd_abi              = stdx::simd_abi::deduce_t<real_t, VL>;
    using compact_blas          = linalg::compact::CompactBLAS<simd_abi>;
    using matrix                = compact_blas::matrix;
    using mut_matrix_view       = compact_blas::mut_batch_view;
    using matrix_view           = compact_blas::batch_view;
    using matrix_view_batch     = compact_blas::single_batch_view;
    using mut_matrix_view_batch = compact_blas::mut_single_batch_view;

    bool alt = false;

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
    matrix work_update = [this] {
        return matrix{{
            .depth = 4 << lvl,
            .rows  = dim.nx,
            .cols  = (dim.N_horiz >> lvl) * dim.ny,
        }};
    }(); // TODO: merge with riccati_ΥΓ?
    matrix work_update_Σ = [this] {
        return matrix{{
            .depth = 1 << lvl,
            .rows  = (dim.N_horiz >> lvl) * dim.ny,
            .cols  = 1,
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
    matrix riccati_ΥΓ1 = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = dim.nu + dim.nx + dim.nx,
            .cols  = (dim.N_horiz >> lP) * std::max(dim.ny, dim.ny_N),
        }};
    }();
    matrix riccati_ΥΓ2 = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = dim.nu + dim.nx + dim.nx,
            .cols  = (dim.N_horiz >> lP) * std::max(dim.ny, dim.ny_N),
        }};
    }();
    matrix data_BA = [this] {
        return matrix{{
            .depth = dim.N_horiz,
            .rows  = dim.nx,
            .cols  = dim.nu + dim.nx,
        }};
    }();
    matrix data_DCᵀ = [this] {
        return matrix{{
            .depth = dim.N_horiz,
            .rows  = dim.nu + dim.nx,
            .cols  = std::max(dim.ny, dim.ny_N),
        }};
    }();
    matrix work_Σ = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = (dim.N_horiz >> lP) * std::max(dim.ny, dim.ny_N),
            .cols  = 1,
        }};
    }();
    matrix data_RSQ = [this] {
        return matrix{{
            .depth = dim.N_horiz,
            .rows  = dim.nu + dim.nx,
            .cols  = dim.nu + dim.nx,
        }};
    }();
    matrix riccati_work = [this] {
        return matrix{{
            .depth = 1 << lP,
            .rows  = dim.nx,
            .cols  = 1,
        }};
    }();
    std::vector<index_t> nJs = std::vector<index_t>(1 << (lP - lvl));
    std::vector<join_counter_t> counters =
        std::vector<join_counter_t>(1 << (lP - lvl));
    std::vector<join_counter_t> counters_UY =
        std::vector<join_counter_t>(1 << (lP - lvl));
    matrix work_pcg = [this] {
        return matrix{{
            .depth = vl,
            .rows  = dim.nx,
            .cols  = 4,
        }};
    }();

    template <class T1, class I1, class S1, class T2, class I2, class S2>
    static void copy_T(guanaqo::MatrixView<T1, I1, S1> src,
                       guanaqo::MatrixView<T2, I2, S2> dst) {
        assert(src.rows == dst.cols);
        assert(src.cols == dst.rows);
        for (index_t r = 0; r < src.rows; ++r) // TODO: optimize
            for (index_t c = 0; c < src.cols; ++c)
                dst(c, r) = src(r, c);
    }

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
                    PRINTLN("ti={:<2}, i={:<2}, k={:<2}, bi={:<2}, vi={:<2}",
                            ti, i, k, bi, vi);
                    copy_T(ocp.D(k),
                           data_DCᵀ.batch(bi)(vi).top_left(nu, dim.ny));
                    if (k < dim.N_horiz) {
                        data_BA.batch(bi)(vi).left_cols(nu) = ocp.B(k);
                        if (k == 0) {
                            auto data_DCᵀi = data_DCᵀ.batch(bi)(vi);
                            copy_T(ocp.C(N),
                                   data_DCᵀi.bottom_right(nx, dim.ny_N));
                            data_DCᵀi // TODO: check user input
                                .top_right(nu, dim.ny_N)
                                .set_constant(0);
                            data_DCᵀi
                                .bottom_left(nx, data_DCᵀ.cols() - dim.ny_N)
                                .set_constant(0);
                        } else {
                            auto data_DCᵀi = data_DCᵀ.batch(bi)(vi);
                            copy_T(ocp.C(k), data_DCᵀi.bottom_left(nx, dim.ny));
                            data_DCᵀi.right_cols(data_DCᵀ.cols() - dim.ny)
                                .set_constant(0);
                        }
                        data_RSQ.batch(bi)(vi).top_left(nu, nu) = ocp.R(k);
                        if (k == 0) {
                            data_BA.batch(bi)(vi)
                                .right_cols(nx) // A
                                .set_constant(0);
                            data_RSQ.batch(bi)(vi)
                                .bottom_left(nx, nu) // S
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

    void initialize_Σ(std::span<const real_t> Σ_lin, mut_matrix_view Σ) {
        auto [N, nx, nu, ny, ny_N] = dim;

        KOQKATOO_ASSERT(static_cast<index_t>(Σ_lin.size()) == N * ny + ny_N);
        KOQKATOO_ASSERT(Σ.depth() == N);
        KOQKATOO_ASSERT(Σ.rows() == std::max(ny, ny_N));
        KOQKATOO_ASSERT(Σ.cols() == 1);

        const auto vstride       = N >> lvl;
        const index_t num_stages = N >> lP; // number of stages per thread
        for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
            const index_t k0  = ti * num_stages;
            const index_t bi0 = ti * num_stages;
            for (index_t i = 0; i < num_stages; ++i) {
                index_t bi = bi0 + i;
                for (index_t vi = 0; vi < vl; ++vi) {
                    auto k       = sub_wrap_N(k0 + vi * vstride, i);
                    using crview = guanaqo::MatrixView<const real_t, index_t>;
                    if (k < N) {
                        Σ.batch(bi)(vi).top_rows(dim.ny) =
                            crview::as_column(Σ_lin.subspan(k * ny, ny));
                        Σ.batch(bi)(vi)
                            .bottom_rows(Σ.rows() - ny)
                            .set_constant(0);
                        if (k == 0) {
                            Σ.batch(bi)(vi).bottom_rows(ny_N) =
                                crview::as_column(Σ_lin.subspan(N * ny, ny_N));
                        }
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
    mutable std::barrier<> std_barrier{1 << (lP - lvl)};
#endif

    void barrier() const {
        KOQKATOO_OMP(barrier);
#if !KOQKATOO_WITH_OPENMP
        std_barrier.arrive_and_wait();
#endif
#if DO_PRINT
        KOQKATOO_OMP(single)
        std::println("---");
#endif
    }

    void process_Y(index_t l, index_t biY) {
        const index_t offset = 1 << l;
        // Compute Y[bi]
        {
            PRINTLN("trsm D{} Y{}", biY, biY);
            GUANAQO_TRACE("Trsm Y", biY);
            compact_blas::xtrsm_RLTN(coupling_D.batch(biY),
                                     coupling_Y.batch(biY), backend);
        }
        // Wait for U[bi] from process_U
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

    void solve_active([[maybe_unused]] index_t l, [[maybe_unused]] index_t biY,
                      [[maybe_unused]] mut_matrix_view λ) const {
        // TODO: nothing?
    }

    void process_U(index_t l, index_t biU) {
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
        // Wait for Y[bi] from process_Y
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
        if (is_active(l + 1, biD) || (l + 1 == lP - lvl && biD == 0)) {
            PRINTLN("chol D{}", biD);
            GUANAQO_TRACE("Factor D", biD);
            compact_blas::xpotrf(coupling_D.batch(biD), backend);
        }
    }

    void solve_active_secondary(index_t l, index_t biU,
                                mut_matrix_view λ) const {
        const index_t num_stages = dim.N_horiz >> lP;
        const index_t offset     = 1 << l;
        const index_t biD        = sub_wrap_PmV(biU, offset);
        const index_t biY        = sub_wrap_PmV(biD, offset);
        const index_t diU        = biU * num_stages;
        const index_t diD        = biD * num_stages;
        const index_t diY        = biY * num_stages;
        // b[diD] -= U[biU] b[diU]
        {
            PRINTLN("gemv U{} b{} -> b{}", biU, diU, diD);
            GUANAQO_TRACE("Subtract Ub", biD);
            compact_blas::xgemv_sub(coupling_U.batch(biU), λ.batch(diU),
                                    λ.batch(diD), backend);
        }
        // D -= YYᵀ
        {
            PRINTLN("gemv Y{} b{} -> b{}", biY, diY, diD);
            GUANAQO_TRACE("Subtract Yb", biD);
            biD == 0
                ? compact_blas::xgemv_sub_shift(coupling_Y.batch(biY),
                                                λ.batch(diY), λ.batch(diD))
                : compact_blas::xgemv_sub(coupling_Y.batch(biY), λ.batch(diY),
                                          λ.batch(diD), backend);
        }
        // chol(D)
        if (is_active(l + 1, biD)) {
            PRINTLN("trsm D{}", biD);
            GUANAQO_TRACE("Solve b", biD);
            compact_blas::xtrsv_LNN(coupling_D.batch(biD), λ.batch(diD),
                                    backend);
        }
    }

    void factor_l0(const index_t ti) {
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t biI        = sub_wrap_PmV(ti, 1);
        const index_t biA        = ti;
        const index_t kI         = biI * num_stages;
        const index_t kA         = biA * num_stages;
        const index_t k          = kA;
        const auto be            = backend;
        const auto biR           = biA;
        const bool x_lanes       = biA == 0; // first stage wraps around
        // Coupling equation to previous stage is eliminated after coupling
        // equation to next stage for odd threads, vice versa for even threads.
        const bool I_below_A = (biA & 1) == 1;
        // Update the subdiagonal blocks U and Y of the coupling equations
        [[maybe_unused]] VecReg vec_curr{vl, k, N >> lvl, N},
            vecA{vl, kA, N >> lvl, N}, vecI{vl, kI, N >> lvl, N};
        auto DiI = coupling_D.batch(biI);
        auto DiA = coupling_D.batch(biA);
        auto Âi  = riccati_ÂB̂.batch(biR).middle_cols(nx * (num_stages - 1), nx);
        auto ÂB̂i = riccati_ÂB̂.batch(biR).right_cols(nx + nu * num_stages);
        auto Q̂i  = riccati_R̂ŜQ̂.batch(biR).bottom_right(nx, nx);
        // Upper triangular matrix, one row up from LQ itself
        assert(nu >= 1);
        auto Q̂i_inv =
            riccati_R̂ŜQ̂.batch(biR).right_cols(nx).middle_rows(nu - 1, nx);
        {
            GUANAQO_TRACE("Invert Q", biI);
            compact_blas::xtrtri_T_copy_ref(Q̂i, Q̂i_inv);
        }
        if (I_below_A) {
            // Top block is A → column index is row index of A (biA)
            // Target block in cyclic part is U in column λ(kA)
            GUANAQO_TRACE("Compute first U", biA);
            compact_blas::xtrmm_LUNN_T_neg_ref(Q̂i_inv, Âi,
                                               coupling_U.batch(biA));
        } else {
            // Top block is I → column index is row index of I (biI)
            // Target block in cyclic part is Y in column λ(kI)
            GUANAQO_TRACE("Compute first Y", biI);
            x_lanes ? compact_blas::xtrmm_RUTN_neg_shift(Âi, Q̂i_inv,
                                                         coupling_Y.batch(biI))
                    : compact_blas::xtrmm_RUTN_neg_ref(Âi, Q̂i_inv,
                                                       coupling_Y.batch(biI));
        }
        // Each column of the cyclic part with coupling equations is updated by
        // two threads: one for the forward, and one for the backward coupling.
        // Update the diagonal blocks of the coupling equations,
        // first forward in time ...
        {
            GUANAQO_TRACE("Compute L⁻ᵀL⁻¹", biI);
            x_lanes ? compact_blas::xtrtrsyrk_UL_shift(Q̂i_inv, DiI)
                    : compact_blas::xtrtrsyrk_UL(Q̂i_inv, DiI);
        }
        // Then synchronize to make sure there are no two threads updating the
        // same diagonal block.
        barrier();
        // And finally backward in time, optionally merged with factorization.
        const bool do_factor = (biA & 1) == 1 || (lP - lvl == 0 && biA == 0);
        {
            GUANAQO_TRACE("Compute (BA)(BA)ᵀ", biA);
            compact_blas::xsyrk_add(ÂB̂i, DiA, be);
        }
        if (do_factor) {
            GUANAQO_TRACE("Factor D", biA);
            compact_blas::xpotrf(coupling_D.batch(biA), backend);
        }
    }

    // Performs Riccati recursion and then factors level l=0 of
    // coupling equations + propagates the subdiagonal blocks to level l=1.
    void factor_riccati(index_t ti, bool alt, matrix_view Σ) {
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t k0         = ti * num_stages; // stage index
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
            compact_blas::xcopy(data_BA.batch(di0).left_cols(nu),
                                B̂.left_cols(nu));
            compact_blas::xcopy(data_BA.batch(di0).right_cols(nx),
                                Â.left_cols(nx));
            compact_blas::xsyrk_schur_copy(data_DCᵀ.batch(di0), Σ.batch(di0),
                                           data_RSQ.batch(di0),
                                           R̂ŜQ̂.left_cols(nux));
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
                const auto di_next = di0 + i + 1;
                auto BAᵀi          = BAᵀ.middle_cols(i * nx, nx);
                auto BAi           = data_BA.batch(di_next);
                auto Bi = BAi.left_cols(nu), Ai = BAi.right_cols(nx);
                // Compute next B̂ and Â
                auto B̂_next = B̂.middle_cols((i + 1) * nu, nu);
                auto Â_next = Â.middle_cols((i + 1) * nx, nx);
                compact_blas::xgemm(Âi, Bi, B̂_next, be);
                compact_blas::xgemm(Âi, Ai, Â_next, be);
                if (alt)
                    compact_blas::xtrsm_RLTN(Q̂i, Âi, be);
                // Riccati update
                auto R̂ŜQ̂_next = R̂ŜQ̂.middle_cols((i + 1) * nux, nux);
                compact_blas::xcopy_T(data_BA.batch(di_next), BAᵀi);
                compact_blas::xtrmm_RLNN(BAᵀi, Q̂i, BAᵀi, be);
                compact_blas::xsyrk_schur_copy(data_DCᵀ.batch(di_next),
                                               Σ.batch(di_next),
                                               data_RSQ.batch(di_next),
                                               R̂ŜQ̂_next);    // ┐
                compact_blas::xsyrk_add(BAᵀi, R̂ŜQ̂_next, be); // ┘
            } else {
                // Compute LÂ = Ã LQ⁻ᵀ
                GUANAQO_TRACE("Riccati last", k);
                compact_blas::xtrsm_RLTN(Q̂i, Âi, be);
            }
        }
    }

    void solve_riccati_forward(index_t ti, mut_matrix_view ux,
                               mut_matrix_view λ) const {
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t biI        = sub_wrap_PmV(ti, 1);
        const index_t diI        = biI * num_stages;
        const index_t di_last    = di0 + num_stages - 1;
        const index_t k0         = ti * num_stages; // stage index
        const index_t nux        = nu + nx;
        const auto be            = backend;
        PRINTLN("\nThread #{}", ti);
        auto R̂ŜQ̂ = riccati_R̂ŜQ̂.batch(ti);
        auto B̂   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
        auto Â   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
        auto BAᵀ = riccati_BAᵀ.batch(ti);
        for (index_t i = 0; i < num_stages; ++i) {
            index_t k  = sub_wrap_N(k0, i);
            index_t di = di0 + i;
            PRINTLN("  Riccati factor QRS{}", VecReg{vl, k, N >> lvl, N});
            auto R̂ŜQ̂i = R̂ŜQ̂.middle_cols(i * nux, nux);
            auto Q̂i   = R̂ŜQ̂i.bottom_right(nx, nx);
            auto B̂i   = B̂.middle_cols(i * nu, nu);
            auto Âi   = Â.middle_cols(i * nx, nx);
            {
                GUANAQO_TRACE("Riccati solve QRS", k);
                // l = LR⁻¹ r, q = LQ⁻¹(q - LS l)
                compact_blas::xtrsv_LNN(R̂ŜQ̂i, ux.batch(di), be);
                // λ0 -= LB̂ l
                compact_blas::xgemv_sub(B̂i, ux.batch(di).top_rows(nu),
                                        λ.batch(di0), be);
            }
            if (i + 1 < num_stages) {
                [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
                const auto di_next                 = di + 1;
                PRINTLN("k={:>2}  di={:>2}  k_next={:>2}  di_next={:>2}", k, di,
                        k_next, di_next);
                auto BAᵀi = BAᵀ.middle_cols(i * nx, nx);
                GUANAQO_TRACE("Riccati solve b", k_next);
                // λ0 += Â λ
                compact_blas::xgemv_add(Âi, λ.batch(di_next), λ.batch(di0), be);
                // b = LQᵀb + q
                compact_blas::xtrmv_T(Q̂i, λ.batch(di_next), be);
                compact_blas::xadd_copy(λ.batch(di_next), λ.batch(di_next),
                                        ux.batch(di).bottom_rows(nx));
                // l += LB λ, q += LA λ
                compact_blas::xgemv_add(BAᵀi, λ.batch(di_next),
                                        ux.batch(di_next), be);
            } else {
                GUANAQO_TRACE("Riccati last", k);
                // λ0 -= Â λ
                compact_blas::xgemv_sub(Âi, ux.batch(di).bottom_rows(nx),
                                        λ.batch(di0), be);
            }
        }
        barrier();
        GUANAQO_TRACE("Riccati coupling I", k0);
        // b = LQ⁻ᵀ x + b
        const bool x_lanes = ti == 0; // first stage wraps around
        auto x_last        = ux.batch(di_last).bottom_rows(nx);
        auto λI            = λ.batch(diI);
        compact_blas::xtrsv_LTN(R̂ŜQ̂.right_cols(nx).bottom_rows(nx), x_last, be);
        x_lanes ? compact_blas::template xadd_copy<-1>(λI, x_last, λI)
                : compact_blas::xadd_copy(λI, x_last, λI);
        compact_blas::xneg(λI); // TODO: merge
        // TODO: remove after testing
        // compact_blas::xtrmv_T(R̂ŜQ̂.right_cols(nx).bottom_rows(nx), x_last, be);
        if (is_active(0, biI))
            compact_blas::xtrsv_LNN(coupling_D.batch(biI), λI, be);
    }

    /// Preserves b in λ (except for coupling equations solved using CR)
    void solve_riccati_forward_alt(index_t ti, mut_matrix_view ux,
                                   mut_matrix_view λ,
                                   mut_matrix_view work) const {
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t biI        = sub_wrap_PmV(ti, 1);
        const index_t diI        = biI * num_stages;
        const index_t di_last    = di0 + num_stages - 1;
        const index_t k0         = ti * num_stages; // stage index
        const index_t nux        = nu + nx;
        const auto be            = backend;
        PRINTLN("\nThread #{}", ti);
        auto R̂ŜQ̂ = riccati_R̂ŜQ̂.batch(ti);
        auto B̂   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
        auto Â   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
        auto w   = work.batch(ti);
        for (index_t i = 0; i < num_stages; ++i) {
            index_t k  = sub_wrap_N(k0, i);
            index_t di = di0 + i;
            PRINTLN("  Riccati solve QRS{}", VecReg{vl, k, N >> lvl, N});
            auto R̂ŜQ̂i = R̂ŜQ̂.middle_cols(i * nux, nux);
            auto R̂i   = R̂ŜQ̂i.top_left(nu, nu);
            auto Ŝi   = R̂ŜQ̂i.bottom_left(nx, nu);
            auto Q̂i   = R̂ŜQ̂i.bottom_right(nx, nx);
            auto B̂i   = B̂.middle_cols(i * nu, nu);
            auto Âi   = Â.middle_cols(i * nx, nx);
            {
                GUANAQO_TRACE("Riccati solve ux", k);
                // l = LR⁻¹ r
                compact_blas::xtrsv_LNN(R̂i, ux.batch(di).top_rows(nu), be);
                // p = q - LS l
                compact_blas::xgemv_sub(Ŝi, ux.batch(di).top_rows(nu),
                                        ux.batch(di).bottom_rows(nx), be);
                // λ0 -= LB̂ l
                compact_blas::xgemv_sub(B̂i, ux.batch(di).top_rows(nu),
                                        λ.batch(di0), be);
            }
            if (i + 1 < num_stages) {
                [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
                const auto di_next                 = di + 1;
                PRINTLN("k={:>2}  di={:>2}  k_next={:>2}  di_next={:>2}", k, di,
                        k_next, di_next);
                GUANAQO_TRACE("Riccati solve b", k_next);
                // b' = LQᵀb
                compact_blas::xcopy(λ.batch(di_next), w);
                compact_blas::xtrmv_T(Q̂i, w, be);
                // λ0 += LÂ LQᵀb
                compact_blas::xgemv_add(Âi, w, λ.batch(di0), be);
                // d' = LQ b'
                compact_blas::xtrmv(Q̂i, w, be);
                // d = LQ LQᵀ b + p
                compact_blas::xadd_copy(w, w, ux.batch(di).bottom_rows(nx));
                // l += Bᵀd, q += Aᵀd
                compact_blas::xgemv_T_add(data_BA.batch(di_next), w,
                                          ux.batch(di_next), be);
            } else {
                GUANAQO_TRACE("Riccati solve last", k);
                // q = LQ⁻¹ p
                compact_blas::xtrsv_LNN(Q̂i, ux.batch(di).bottom_rows(nx), be);
                // λ0 -= LÂ q
                compact_blas::xgemv_sub(Âi, ux.batch(di).bottom_rows(nx),
                                        λ.batch(di0), be);
            }
        }
        barrier();
        GUANAQO_TRACE("Riccati coupling I", k0);
        // b = LQ⁻ᵀ x + b
        const bool x_lanes = ti == 0; // first stage wraps around
        auto x_last        = ux.batch(di_last).bottom_rows(nx);
        auto λI            = λ.batch(diI);
        compact_blas::xtrsv_LTN(R̂ŜQ̂.right_cols(nx).bottom_rows(nx), x_last, be);
        // Note: this leaves LQ⁻ᵀ q in x_last,
        //       which is reused during the backward solve
        x_lanes ? compact_blas::template xadd_copy<-1>(λI, x_last, λI)
                : compact_blas::xadd_copy(λI, x_last, λI);
        compact_blas::xneg(λI); // TODO: merge
        if (is_active(0, biI))
            compact_blas::xtrsv_LNN(coupling_D.batch(biI), λI, be);
    }

    void solve_forward(mut_matrix_view ux, mut_matrix_view λ,
                       mut_matrix_view work) const {
        const index_t N = dim.N_horiz;
        KOQKATOO_ASSERT(((N >> lP) << lP) == N);
        koqkatoo::foreach_thread([this, &ux, &λ, &work](index_t ti, index_t P) {
            if (P < (1 << (lP - lvl)))
                throw std::logic_error("Incorrect number of threads");
            if (ti >= (1 << (lP - lvl)))
                return;
            alt ? solve_riccati_forward_alt(ti, ux, λ, work)
                : solve_riccati_forward(ti, ux, λ);
            for (index_t l = 0; l < lP - lvl; ++l) {
                barrier();
                const index_t offset = 1 << l;
                const auto biY       = sub_wrap_PmV(ti, offset);
                const auto biU       = ti;
                if (is_active(l, biY))
                    solve_active(l, biY, λ);
                else if (is_active(l, biU))
                    solve_active_secondary(l, biU, λ);
            }
        });
    }

    real_t mul_A(matrix_view_batch p, mut_matrix_view_batch Ap,
                 matrix_view_batch L, matrix_view_batch B) const {
        compact_blas::xcopy(p, Ap);
        compact_blas::xtrmv_T(L, Ap, backend);
        compact_blas::xtrmv(L, Ap, backend);
        compact_blas::xsyomv(B, p, Ap);
        return compact_blas::xdot(p, Ap);
    }

    real_t mul_precond(matrix_view_batch r, mut_matrix_view_batch z,
                       mut_matrix_view_batch w, matrix_view_batch L,
                       matrix_view_batch B) const {
        compact_blas::xcopy(r, z);
#if USE_JACOBI_PREC
        std::ignore = w;
        std::ignore = B;
#else
        compact_blas::xcopy(r, w);
        compact_blas::xtrsv_LNN(L, w, backend);
        compact_blas::xtrsv_LTN(L, w, backend);
        compact_blas::xsyomv_neg(B, w.as_const(), z);
#endif
        compact_blas::xtrsv_LNN(L, z, backend);
        compact_blas::xtrsv_LTN(L, z, backend);
        return compact_blas::xdot(r, z);
    }

    void solve_pcg(mut_matrix_view_batch λ,
                   mut_matrix_view_batch work_pcg) const {
        auto r = work_pcg.middle_cols(0, 1), z = work_pcg.middle_cols(1, 1),
             p = work_pcg.middle_cols(2, 1), Ap = work_pcg.middle_cols(3, 1);
        auto A = coupling_D.batch(0), B = coupling_Y.batch(0);
        real_t rᵀz = [&] {
            GUANAQO_TRACE("solve Ψ pcg", 0);
            compact_blas::xcopy(λ, r);
            compact_blas::xfill(0, λ);
            real_t rᵀz = mul_precond(r, z, Ap, A, B);
            compact_blas::xcopy(z, p);
            return rᵀz;
        }();
        for (index_t it = 0; it < 100; ++it) { // TODO
            GUANAQO_TRACE("solve Ψ pcg", it + 1);
            real_t pᵀAp = mul_A(p, Ap, A, B);
            real_t α    = rᵀz / pᵀAp;
            compact_blas::xaxpy(+α, p, λ);
            compact_blas::xaxpy(-α, Ap, r);
            real_t r2          = compact_blas::xdot(r, r);
            constexpr real_t ε = std::numeric_limits<real_t>::epsilon();
            if (r2 < ε * ε)
                break;
            real_t rᵀz_new = mul_precond(r, z, Ap, A, B);
            real_t β       = rᵀz_new / rᵀz;
            compact_blas::xaxpby(1, z, β, p);
            rᵀz = rᵀz_new;
        }
    }

    void solve_pcg(mut_matrix_view_batch λ) { solve_pcg(λ, work_pcg.batch(0)); }

    void solve_reverse_active(index_t l, index_t bi, mut_matrix_view λ) const {
        const index_t offset     = 1 << l;
        const index_t num_stages = dim.N_horiz >> lP;
        const index_t biY        = add_wrap_PmV(bi, offset);
        const index_t biU        = sub_wrap_PmV(bi, offset);
        const index_t di         = bi * num_stages;
        const index_t diY        = biY * num_stages;
        const index_t diU        = biU * num_stages;
        const bool x_lanes       = diY == 0;
        GUANAQO_TRACE("Solve coupling reverse", bi);
        x_lanes ? compact_blas::xgemv_T_sub_shift(coupling_Y.batch(bi),
                                                  λ.batch(diY), λ.batch(di))
                : compact_blas::xgemv_T_sub(coupling_Y.batch(bi), λ.batch(diY),
                                            λ.batch(di), backend);
        compact_blas::xgemv_T_sub(coupling_U.batch(bi), λ.batch(diU),
                                  λ.batch(di), backend);
        compact_blas::xtrsv_LTN(coupling_D.batch(bi), λ.batch(di), backend);
    }

    void solve_riccati_reverse(index_t ti, mut_matrix_view ux,
                               mut_matrix_view λ, mut_matrix_view work) const {
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t biI        = sub_wrap_PmV(ti, 1);
        const index_t diI        = biI * num_stages;
        const index_t k0         = ti * num_stages; // stage index
        const index_t nux        = nu + nx;
        const auto be            = backend;
        PRINTLN("\nThread #{}", ti);
        auto R̂ŜQ̂ = riccati_R̂ŜQ̂.batch(ti);
        auto B̂   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
        auto Â   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
        auto BAᵀ = riccati_BAᵀ.batch(ti);

        for (index_t i = num_stages; i-- > 0;) {
            index_t k  = sub_wrap_N(k0, i);
            index_t di = di0 + i;
            PRINTLN("  Riccati factor QRS{}", VecReg{vl, k, N >> lvl, N});
            auto R̂ŜQ̂i = R̂ŜQ̂.middle_cols(i * nux, nux);
            auto Q̂i   = R̂ŜQ̂i.bottom_right(nx, nx);
            auto R̂i   = R̂ŜQ̂i.top_left(nu, nu);
            auto Ŝi   = R̂ŜQ̂i.bottom_left(nx, nu);
            auto B̂i   = B̂.middle_cols(i * nu, nu);
            auto Âi   = Â.middle_cols(i * nx, nx);
            if (i + 1 < num_stages) {
                [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
                const auto di_next                 = di + 1;
                GUANAQO_TRACE("Riccati solve b", k_next);
                auto BAᵀi = BAᵀ.middle_cols(i * nx, nx);
                // b -= LBᵀ u + LAᵀ x
                compact_blas::xgemv_T_sub(BAᵀi, ux.batch(di_next),
                                          λ.batch(di_next), be);
                compact_blas::xneg(λ.batch(di_next)); // TODO
                // q -= b
                compact_blas::xadd_copy(ux.batch(di).bottom_rows(nx),
                                        ux.batch(di).bottom_rows(nx),
                                        λ.batch(di_next));
                compact_blas::xtrmv(Q̂i, λ.batch(di_next), backend);
                compact_blas::xgemv_T_add(Âi, λ.batch(di0), λ.batch(di_next),
                                          backend);
            } else {
                // x_last = LQ⁻ᵀ(q_last + LQ⁻¹ λ - LÂᵀ λ)
                GUANAQO_TRACE("Riccati last", k);
                // λ0 -= Â λ
                const auto x_last  = ux.batch(di).bottom_rows(nx);
                const bool x_lanes = ti == 0;
                const auto w       = work.batch(ti);
                x_lanes ? compact_blas::template xadd_copy<1>(w, λ.batch(diI))
                              : compact_blas::template xadd_copy<0>(w, λ.batch(diI));
                // LQ⁻¹ λ
                compact_blas::xtrsv_LNN(Q̂i, w, backend);
                // LQ⁻¹ λ - LÂᵀ λ
                compact_blas::xgemv_T_sub(Âi, λ.batch(di0), w, be);
                // x_last = LQ⁻ᵀ(LQ⁻¹ λ - LÂᵀ λ)
                compact_blas::xtrsv_LTN(Q̂i, w, backend);
                compact_blas::xadd_copy(x_last, x_last, w);
            }
            if (i + 1 < num_stages) {
                GUANAQO_TRACE("Riccati solve QRS", k);
                // l -= LB̂ᵀ λ0
                compact_blas::xgemv_T_sub(B̂i, λ.batch(di0),
                                          ux.batch(di).top_rows(nu), be);
                // x = LQ⁻ᵀ q, u = LR⁻ᵀ (l - LSᵀ x)
                compact_blas::xtrsv_LTN(R̂ŜQ̂i, ux.batch(di), be);
            } else {
                GUANAQO_TRACE("Riccati solve QRS", k);
                // l -= LB̂ᵀ λ0 + LSᵀ q
                compact_blas::xgemv_T_sub(B̂i, λ.batch(di0),
                                          ux.batch(di).top_rows(nu), be);
                compact_blas::xgemv_T_sub(Ŝi, ux.batch(di).bottom_rows(nx),
                                          ux.batch(di).top_rows(nu), be);
                // x = LQ⁻ᵀ q, u = LR⁻ᵀ (l - LSᵀ x)
                compact_blas::xtrsv_LTN(R̂i, ux.batch(di).top_rows(nu), be);
            }
        }
    }

    void solve_riccati_reverse_alt(index_t ti, mut_matrix_view ux,
                                   mut_matrix_view λ,
                                   mut_matrix_view work) const {
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t biI        = sub_wrap_PmV(ti, 1);
        const index_t diI        = biI * num_stages;
        const index_t k0         = ti * num_stages; // stage index
        const index_t nux        = nu + nx;
        const auto be            = backend;
        PRINTLN("\nThread #{}", ti);
        auto R̂ŜQ̂     = riccati_R̂ŜQ̂.batch(ti);
        auto B̂       = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
        auto Â       = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
        const auto w = work.batch(ti);

        for (index_t i = num_stages; i-- > 0;) {
            index_t k  = sub_wrap_N(k0, i);
            index_t di = di0 + i;
            PRINTLN("  Riccati factor QRS{}", VecReg{vl, k, N >> lvl, N});
            auto R̂ŜQ̂i = R̂ŜQ̂.middle_cols(i * nux, nux);
            auto Q̂i   = R̂ŜQ̂i.bottom_right(nx, nx);
            auto R̂i   = R̂ŜQ̂i.top_left(nu, nu);
            auto Ŝi   = R̂ŜQ̂i.bottom_left(nx, nu);
            auto B̂i   = B̂.middle_cols(i * nu, nu);
            auto Âi   = Â.middle_cols(i * nx, nx);
            if (i + 1 < num_stages) {
                [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
                const auto di_next                 = di + 1;
                GUANAQO_TRACE("Riccati solve rev", k_next);
                auto BAi = data_BA.batch(di_next);

                // w = p
                compact_blas::xcopy(ux.batch(di).bottom_rows(nx), w);
                // x = A x(next) + B u(next) - b(next)
                compact_blas::xadd_neg_copy(ux.batch(di).bottom_rows(nx),
                                            λ.batch(di_next));
                compact_blas::xgemv_add(BAi, ux.batch(di_next),
                                        ux.batch(di).bottom_rows(nx), be);
                // u = LR⁻ᵀ(l - LSᵀ x - LB̂ᵀ λ(last))
                compact_blas::xgemv_T_sub(B̂i, λ.batch(di0),
                                          ux.batch(di).top_rows(nu), be);
                compact_blas::xgemv_T_sub(Ŝi, ux.batch(di).bottom_rows(nx),
                                          ux.batch(di).top_rows(nu), be);
                compact_blas::xtrsv_LTN(R̂i, ux.batch(di).top_rows(nu), be);

                // λ(next) = LQ (LQᵀ x + LÂᵀ λ(last)) - p
                compact_blas::xcopy(ux.batch(di).bottom_rows(nx),
                                    λ.batch(di_next));
                compact_blas::xtrmv_T(Q̂i, λ.batch(di_next), be);
                compact_blas::xgemv_T_add(Âi, λ.batch(di0), λ.batch(di_next),
                                          be);
                compact_blas::xtrmv(Q̂i, λ.batch(di_next), be);
                compact_blas::xsub_copy(λ.batch(di_next), λ.batch(di_next), w);
            } else {
                // x_last = LQ⁻ᵀ(q_last + LQ⁻¹ λ - LÂᵀ λ)
                GUANAQO_TRACE("Riccati solve rev", k);
                // λ0 -= Â λ
                const auto x_last  = ux.batch(di).bottom_rows(nx);
                const bool x_lanes = ti == 0;
                x_lanes ? compact_blas::template xadd_copy<1>(w, λ.batch(diI))
                        : compact_blas::template xadd_copy<0>(w, λ.batch(diI));
                // LQ⁻¹ λ
                compact_blas::xtrsv_LNN(Q̂i, w, backend);
                // LQ⁻¹ λ - LÂᵀ λ
                compact_blas::xgemv_T_sub(Âi, λ.batch(di0), w, be);
                // x_last = LQ⁻ᵀ(LQ⁻¹ λ - LÂᵀ λ)
                compact_blas::xtrsv_LTN(Q̂i, w, backend);
                compact_blas::xadd_copy(x_last, x_last, w);

                // u -= LB̂ᵀ λ0 + LSᵀ q
                compact_blas::xgemv_T_sub(B̂i, λ.batch(di0),
                                          ux.batch(di).top_rows(nu), be);
                compact_blas::xgemv_T_sub(Ŝi, ux.batch(di).bottom_rows(nx),
                                          ux.batch(di).top_rows(nu), be);
                // x = LQ⁻ᵀ q, u = LR⁻ᵀ (u - LSᵀ x)
                compact_blas::xtrsv_LTN(R̂i, ux.batch(di).top_rows(nu), be);
            }
        }
    }

    void solve_reverse(mut_matrix_view ux, mut_matrix_view λ,
                       mut_matrix_view work) const {
        const index_t N = dim.N_horiz;
        KOQKATOO_ASSERT(((N >> lP) << lP) == N);
        koqkatoo::foreach_thread([this, &ux, &λ, &work](index_t ti, index_t P) {
            if (P < (1 << (lP - lvl)))
                throw std::logic_error("Incorrect number of threads");
            if (ti >= (1 << (lP - lvl)))
                return;
            for (index_t l = lP - lvl; l-- > 0;) {
                const index_t offset = 1 << l;
                const auto bi        = sub_wrap_PmV(ti, offset);
                if (is_active(l, bi))
                    solve_reverse_active(l, bi, λ);
                barrier();
            }
            alt ? solve_riccati_reverse_alt(ti, ux, λ, work)
                : solve_riccati_reverse(ti, ux, λ, work);
        });
    }

    void solve(mut_matrix_view ux, mut_matrix_view λ,
               mut_matrix_view_batch work_pcg,
               mut_matrix_view work_riccati) const {
        solve_forward(ux, λ, work_riccati);
        solve_pcg(λ.batch(0), work_pcg);
        solve_reverse(ux, λ, work_riccati);
    }

    void solve(mut_matrix_view ux, mut_matrix_view λ) {
        solve(ux, λ, work_pcg.batch(0), riccati_work);
    }

    void run(matrix_view Σ, bool alt = false) {
        this->alt = alt;
        for (auto &c : counters)
            c.value.store(0, std::memory_order_relaxed);
        for (auto &c : counters_UY)
            c.value.store(0, std::memory_order_relaxed);
        coupling_D.set_constant(0); // TODO
        const index_t N = dim.N_horiz;
        KOQKATOO_ASSERT(((N >> lP) << lP) == N);
        koqkatoo::foreach_thread([this, alt, Σ](index_t ti, index_t P) {
            if (P < (1 << (lP - lvl)))
                throw std::logic_error("Incorrect number of threads");
            if (ti >= (1 << (lP - lvl)))
                return;
            factor_riccati(ti, alt, Σ);
            factor_l0(ti);
            for (index_t l = 0; l < lP - lvl; ++l) {
                barrier();
                const index_t offset = 1 << l;
                const auto biY       = sub_wrap_PmV(ti, offset);
                const auto biU       = ti;
                if (is_active(l, biY))
                    process_Y(l, biY);
                else if (is_active(l, biU))
                    process_U(l, biU);
                else
                    barrier();
            }
        });
    }

    void update_level(index_t l, index_t biY) {
        const index_t offset = 1 << l;
        const index_t i      = biY >> (l + 1);
        const index_t j0     = biY == offset ? 0 : nJs[biY - 1 - offset],
                      j1 = nJs[biY - 1 + offset], nj = j1 - j0,
                      jsplit = nJs[biY - 1] - j0;
        constexpr index_t w3_out_lut[]{1, 0, 0, 1};
        const index_t w3_out = w3_out_lut[i & 3];
        std::println("biY={:>2},  i={:>2}  w3={:>2}  {:>2}:{}:{:<2}", biY, i,
                     w3_out, j0, j0 + jsplit, j1);
        if (i & 1) {
            compact_blas::xshhud_diag_cyclic(
                coupling_D.batch(biY),
                work_update.batch(l & 3).middle_cols(j0, nj),
                coupling_Y.batch(biY),
                work_update.batch((l + 2) % 4).middle_cols(j0, nj),
                work_update.batch((l + 2 + w3_out) % 4).middle_cols(j0, nj),
                coupling_U.batch(biY),
                work_update.batch((l + 1) % 4).middle_cols(j0, nj),
                work_update.batch((l + 1) % 4).middle_cols(j0, nj),
                work_update_Σ.batch(0).middle_rows(j0, nj), jsplit, 0);
            // if (x_lanes) { // TODO
            //     compact_blas::template xadd_copy<1>(
            //         work_update_Σ.batch(0).middle_rows(j0, nj),
            //         work_update_Σ.batch(0).middle_rows(j0, nj));
            // }
        } else {
            compact_blas::xshhud_diag_cyclic(
                coupling_D.batch(biY),
                work_update.batch(l & 3).middle_cols(j0, nj),
                coupling_Y.batch(biY),
                work_update.batch((l + 1) % 4).middle_cols(j0, nj),
                work_update.batch((l + 1) % 4).middle_cols(j0, nj),
                coupling_U.batch(biY),
                work_update.batch((l + 2) % 4).middle_cols(j0, nj),
                work_update.batch((l + 2 + w3_out) % 4).middle_cols(j0, nj),
                work_update_Σ.batch(0).middle_rows(j0, nj), jsplit, 0);
        }
    }

    void update(matrix_view ΔΣ) {
        const index_t N = dim.N_horiz;
        KOQKATOO_ASSERT(((N >> lP) << lP) == N);
        koqkatoo::foreach_thread([this, ΔΣ](index_t ti, index_t P) {
            if (P < (1 << (lP - lvl)))
                throw std::logic_error("Incorrect number of threads");
            if (ti >= (1 << (lP - lvl)))
                return;
            update_riccati(ti, alt, ΔΣ);
            for (index_t l = 0; l < lP - lvl; ++l) {
                barrier();
                if (ti == 0)
                    std::println("=====");
                barrier();

                const index_t offset = 1 << l;
                const auto biY       = sub_wrap_PmV(ti, offset);
                if (is_active(l, biY)) {
                    const index_t offset = 1 << l;
                    const index_t i      = biY >> (l + 1);
                    const index_t j0 =
                                      biY == offset ? 0 : nJs[biY - 1 - offset],
                                  j1 = nJs[biY - 1 + offset];
                    std::println("ti={:>2}, biY={:>2},  i={:>2}  {:>2}:{:<2}",
                                 ti, biY, i, j0, j1);
                    update_level(l, biY);
                    if (l + 1 == lP - lvl) {
                        const index_t j0 = 0, j1 = nJs.back(), nj = j1 - j0;
                        compact_blas::xshhud_diag_ref(
                            coupling_D.batch(0),
                            work_update.batch((l + 3) & 3).middle_cols(j0, nj),
                            work_update_Σ.batch(0).middle_rows(j0, nj));
                    }
                }
            }
        });
        this->alt = true;
    }

    // Performs Riccati recursion and then factors level l=0 of
    // coupling equations + propagates the subdiagonal blocks to level l=1.
    void update_riccati(index_t ti, bool alt, matrix_view Σ) {
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t nyM                = std::max(ny, ny_N);
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t k0         = ti * num_stages; // stage index
        const index_t nux        = nu + nx;
        const auto be            = backend;
        PRINTLN("\nThread #{}", ti);
        auto R̂ŜQ̂ = riccati_R̂ŜQ̂.batch(ti);
        auto B̂   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
        auto Â   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
        auto ΥΓ1 = riccati_ΥΓ1.batch(ti);
        auto ΥΓ2 = riccati_ΥΓ2.batch(ti);
        auto wΣ  = work_Σ.batch(ti);

        // index_t nJ_tot = 0;
        // for (index_t i = 0; i < num_stages; ++i) {
        //     const auto di = di0 + i;
        //     nJ_tot += compact_blas::compress_masks_count(Σ.batch(di));
        // }
        // std::println("nJ_tot={}", nJ_tot);

        index_t nJ;
        {
            GUANAQO_TRACE("Riccati update compress", k0);
            auto DC0 = ΥΓ2.top_left(nu + nx, nyM);
            nJ = compact_blas::compress_masks(data_DCᵀ.batch(di0), Σ.batch(di0),
                                              DC0, wΣ.top_rows(nyM));
            ΥΓ2.bottom_left(nx, nJ).set_constant(0);
        }

        for (index_t i = 0; i < num_stages; ++i) {
            index_t k = sub_wrap_N(k0, i);
            PRINTLN("  Riccati factor QRS{}", VecReg{vl, k, N >> lvl, N});
            auto R̂ŜQ̂i = R̂ŜQ̂.middle_cols(i * nux, nux);
            auto R̂Ŝi  = R̂ŜQ̂i.left_cols(nu);
            auto Q̂i   = R̂ŜQ̂i.bottom_right(nx, nx);
            auto B̂i   = B̂.middle_cols(i * nu, nu);
            auto Âi   = Â.middle_cols(i * nx, nx);

            index_t nJi = nJ;
            auto ΥΓi    = ((i & 1) ? ΥΓ1 : ΥΓ2).left_cols(nJi);
            if (nJi > 0) {
                GUANAQO_TRACE("Riccati update R", k);
                compact_blas::xshhud_diag_2_ref(R̂Ŝi, ΥΓi.top_rows(nu + nx), B̂i,
                                                ΥΓi.bottom_rows(nx),
                                                wΣ.top_rows(nJi));
            }
            if (i + 1 < num_stages) {
                [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
                const auto di_next                 = di0 + i + 1;
                auto ΥΓ_next = ((i & 1) ? ΥΓ2 : ΥΓ1).left_cols(nJi + nyM);
                if (nJi > 0) {
                    GUANAQO_TRACE("Riccati update prop", k_next);
                    compact_blas::xgemm_TN(data_BA.batch(di_next),
                                           ΥΓi.middle_rows(nu, nx),
                                           ΥΓ_next.top_left(nu + nx, nJi), be);
                    compact_blas::xcopy(ΥΓi.bottom_rows(nx),
                                        ΥΓ_next.bottom_left(nx, nJi));
                }
                if (!alt) {
                    GUANAQO_TRACE("Riccati convert alt", k);
                    compact_blas::xtrsm_RLTN(Q̂i, Âi, be);
                }
                {
                    GUANAQO_TRACE("Riccati update compress", k_next);
                    auto DC_next = ΥΓ_next.block(0, nJi, nu + nx, nyM);
                    nJ += compact_blas::compress_masks(
                        data_DCᵀ.batch(di_next), Σ.batch(di_next), DC_next,
                        wΣ.middle_rows(nJi, nyM));
                    ΥΓ_next.block(nu + nx, nJi, nx, nJ - nJi).set_constant(0);
                }
                if (nJi > 0) {
                    GUANAQO_TRACE("Riccati update Q", k);
                    compact_blas::xshhud_diag_2_ref(Q̂i, ΥΓi.middle_rows(nu, nx),
                                                    Âi, ΥΓi.bottom_rows(nx),
                                                    wΣ.top_rows(nJi));
                }
            } else {
                const auto bi_upd = sub_wrap_PmV(ti, 1);
                nJs[bi_upd]       = nJi;
                barrier();
                if (ti == 0)
                    std::inclusive_scan(begin(nJs), end(nJs), begin(nJs));
                barrier();
                const index_t j0 = bi_upd == 0 ? 0 : nJs[bi_upd - 1],
                              j1 = nJs[bi_upd];
                assert(nJi == j1 - j0);
                constexpr index_t wiA_table[]{0, 1, 0, 2};
                constexpr index_t wiI_table[]{2, 0, 1, 0};
                const index_t wiA = wiA_table[bi_upd & 3];
                const index_t wiI = wiI_table[bi_upd & 3];
                std::println("ti={:>2}, bi_upd={:>2}, {:>2}:{:<2} [{}][{}]", ti,
                             bi_upd, j0, j1, wiA, wiI);
                if (nJi > 0) {
                    GUANAQO_TRACE("Riccati update Q", k);
                    auto Q̂i_inv = R̂ŜQ̂i.block(nu - 1, nu, nx, nx);
                    compact_blas::xshhud_diag_riccati(
                        Q̂i, ΥΓi.middle_rows(nu, nx), Âi, ΥΓi.bottom_rows(nx),
                        work_update.batch(wiA).middle_cols(j0, nJi), Q̂i_inv,
                        work_update.batch(wiI).middle_cols(j0, nJi),
                        wΣ.top_rows(nJi), ti == 0); // TODO
                    compact_blas::xneg(
                        work_update.batch(wiI).middle_cols(j0, nJi)); // TODO
                    ti == 0 ? compact_blas::template xadd_neg_copy<-1>(
                                  work_update_Σ.batch(0).middle_rows(j0, nJi),
                                  wΣ.top_rows(nJi))
                            : compact_blas::xadd_neg_copy(
                                  work_update_Σ.batch(0).middle_rows(j0, nJi),
                                  wΣ.top_rows(nJi));
                }
            }
        }
    }

    index_t get_linear_batch_offset(index_t biA) {
        const auto levA = biA > 0 ? get_level(biA) : lP;
        const auto levP = lP - lvl;
        if (levA >= levP)
            return (((1 << levP) - 1) << (lP - levP)) + (biA >> levP);
        return (((1 << levA) - 1) << (lP - levA)) + get_index_in_level(biA);
    }

    auto build_sparse(const koqkatoo::ocp::LinearOCPStorage &ocp,
                      std::span<const real_t> Σ) {
        using std::sqrt;
        std::vector<std::tuple<index_t, index_t, real_t>> tuples;

        auto [N, nx, nu, ny, ny_N] = ocp.dim;
        const index_t nux = nu + nx, nuxx = nux + nx;
        const index_t vstride    = N >> lvl;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t num_proc   = 1 << (lP - lvl);
        const index_t sλ         = N * nuxx - (nx << lP);

        linalg::compact::BatchedMatrix<real_t, index_t> RSQ_DC{{
            .depth = N,
            .rows  = nux,
            .cols  = nux,
        }};
        linalg::compact::BatchedMatrix<real_t, index_t> DC{{
            .depth = N,
            .rows  = std::max(ny, ny_N),
            .cols  = nux,
        }};
        auto R  = RSQ_DC.top_left(nu, nu);
        auto Q  = RSQ_DC.bottom_right(nx, nx);
        auto Sᵀ = RSQ_DC.bottom_left(nx, nu);
        for (index_t k = 0; k < N; ++k) {
            R(k) = ocp.R(k);
            Q(k) = ocp.Q(k == 0 ? N : k);
            if (k > 0) {
                Sᵀ(k)                   = ocp.S_trans(k);
                DC.top_left(ny, nu)(k)  = ocp.D(k);
                DC.top_right(ny, nx)(k) = ocp.C(k);
                for (index_t j = 0; j < ny; ++j)
                    for (index_t i = 0; i < nux; ++i)
                        DC(k, j, i) *= sqrt(Σ[k * ny + j]);
                guanaqo::blas::xsyrk_LT(real_t{1}, DC(k), real_t{1}, RSQ_DC(k));
            } else {
                auto D0 = DC.top_left(ny, nu)(k);
                auto CN = DC.bottom_right(ny_N, nx)(k);
                D0      = ocp.D(0);
                CN      = ocp.C(N);
                DC.bottom_left(ny_N, nu)(k).set_constant(0); // TODO
                for (index_t j = 0; j < ny; ++j)
                    for (index_t i = 0; i < nu; ++i)
                        D0(j, i) *= sqrt(Σ[k * ny + j]);
                for (index_t j = 0; j < ny_N; ++j)
                    for (index_t i = 0; i < nx; ++i)
                        CN(j, i) *= sqrt(Σ[N * ny + j]);
                guanaqo::blas::xsyrk_LT(real_t{1}, DC(k), real_t{1}, RSQ_DC(k));
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
                PRINTLN("k={:<2}  biA={:<2}  oA={:<2}  biI={:<2}  oI={:<2}", k0,
                        biA, get_linear_batch_offset(biA), biI,
                        get_linear_batch_offset(biI));
                // TODO: handle case if lev > or >= lP - lvl
                for (index_t i = 0; i < num_stages; ++i) {
                    const index_t k = sub_wrap_N(k0, i);
                    index_t s = sv + ti * (nuxx * num_stages - nx) + nuxx * i;
                    for (index_t c = 0; c < nu; ++c) {
                        for (index_t r = c; r < nu; ++r)
                            tuples.emplace_back(s + r, s + c, R(k)(r, c));
                        if (k > 0)
                            for (index_t r = 0; r < nx; ++r)
                                tuples.emplace_back(s + r + nu, s + c,
                                                    Sᵀ(k)(r, c));
                        if (i == 0) {
                            for (index_t r = 0; r < nx; ++r)
                                tuples.emplace_back(sλA + r, s + c,
                                                    ocp.B(k0)(r, c));
                        }
                    }
                    for (index_t c = 0; c < nx; ++c) {
                        for (index_t r = c; r < nx; ++r)
                            tuples.emplace_back(s + r + nu, s + c + nu,
                                                Q(k)(r, c));
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

    auto build_rhs(matrix_view ux, matrix_view λ) {
        auto [N, nx, nu, ny, ny_N] = dim;
        const index_t nux = nu + nx, nuxx = nux + nx;
        std::vector<real_t> tuples(nuxx * N);
        std::ranges::fill(tuples, std::numeric_limits<real_t>::quiet_NaN());
        // const index_t vstride    = N >> lvl;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t num_proc   = 1 << (lP - lvl);
        const index_t sλ         = N * nuxx - (nx << lP);

        for (index_t vi = 0; vi < vl; ++vi) {
            const index_t sv = vi * num_proc * (nuxx * num_stages - nx);
            for (index_t ti = 0; ti < num_proc; ++ti) {
                const index_t di0 = ti * num_stages;
                for (index_t i = 0; i < num_stages; ++i) {
                    const index_t di = di0 + i;
                    index_t s = sv + ti * (nuxx * num_stages - nx) + nuxx * i;
                    PRINTLN("vi={:>2}  ti={:>2}  i={:>2}  di={:>2}  s={:>2}",
                            vi, ti, i, di, s);
                    if (i > 0)
                        for (index_t c = 0; c < nx; ++c)
                            tuples[s - nx + c] = λ.batch(di)(vi)(c, 0);
                    for (index_t c = 0; c < nux; ++c)
                        tuples[s + c] = ux.batch(di)(vi)(c, 0);
                }
            }
        }
        index_t s               = sλ;
        const auto cyclic_block = [&](index_t i) {
            const index_t bi = i % (1 << (lP - lvl));
            const index_t vi = i / (1 << (lP - lvl));
            const index_t di = bi * num_stages;
            for (index_t c = 0; c < nx; ++c)
                tuples[s + c] = λ.batch(di)(vi)(c, 0);
            s += nx;
        };
        if (lP != lvl) {
            for (index_t i = 0; i < (1 << (lP - 1)); ++i)
                cyclic_block(2 * i + 1);
            for (index_t l = 1; l < lP - lvl; ++l) {
                index_t offset = 1 << l;
                index_t stride = offset << 1;
                for (index_t i = offset; i < (1 << lP); i += stride)
                    cyclic_block(i);
            }
        }
        for (index_t i = 0; i < (1 << lP); i += (1 << (lP - lvl))) {
            cyclic_block(i);
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
        matrix AinvQᵀ{{
            .depth = 1 << lP,
            .rows  = dim.nx,
            .cols  = (dim.N_horiz >> lP) * dim.nx,
        }};
        matrix LBA{{
            .depth = 1 << lP,
            .rows  = dim.nu + dim.nx,
            .cols  = ((dim.N_horiz >> lP) - 1) * dim.nx,
        }};
        for (index_t ti = 0; ti < num_proc; ++ti) {
            const index_t di0 = ti * num_stages; // data batch index
            for (index_t i = 0; i < num_stages; ++i) {
                const auto di = di0 + i;
                auto RSQ      = riccati_R̂ŜQ̂.batch(ti);
                auto RSQi     = RSQ.middle_cols(i * nux, nux);
                auto Qi       = RSQi.bottom_right(nx, nx);
                auto Qi_inv   = RSQi.block(nu - 1, nu, nx, nx);
                auto Â        = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
                auto Âi       = Â.middle_cols(i * nx, nx);
                auto AiQᵀ     = AinvQᵀ.batch(ti);
                auto AiQiᵀ    = AiQᵀ.middle_cols(i * nx, nx);
                auto BAᵀ      = riccati_BAᵀ.batch(ti);
                auto LBAt     = LBA.batch(ti);

                compact_blas::xcopy(Âi, AiQiᵀ);
                if (i + 1 < num_stages) // Final block already inverted
                    compact_blas::xtrtri_T_copy_ref(Qi, Qi_inv);
                if (i + 1 < num_stages && !alt) // Final block is already Â LQ⁻ᵀ
                    compact_blas::xtrsm_RLTN(Qi, AiQiᵀ, backend);
                if (i > 0) {
                    auto LBAi = LBAt.middle_cols((i - 1) * nx, nx);
                    if (alt) {
                        auto RSQ_prev = RSQ.middle_cols((i - 1) * nux, nux);
                        auto Q_prev   = RSQ_prev.bottom_right(nx, nx);
                        auto BA       = data_BA.batch(di);
                        compact_blas::xcopy_T(BA, LBAi);
                        compact_blas::xtrmm_RLNN(LBAi, Q_prev, LBAi, backend);
                    } else {
                        auto BAᵀi = BAᵀ.middle_cols((i - 1) * nx, nx);
                        compact_blas::xcopy(BAᵀi, LBAi);
                    }
                }
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
                auto B̂    = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
                auto R̂ŜQ̂  = riccati_R̂ŜQ̂.batch(ti);
                auto LBAt = LBA.batch(ti);
                // TODO: handle case if lev > or >= lP - lvl
                for (index_t i = 0; i < num_stages; ++i) {
                    [[maybe_unused]] const index_t k = sub_wrap_N(k0, i);
                    index_t s  = sv + ti * (nuxx * num_stages - nx) + nuxx * i;
                    auto B̂i    = B̂.middle_cols(i * nu, nu);
                    auto R̂ŜQ̂i  = R̂ŜQ̂.middle_cols(i * nux, nux);
                    auto RSi   = R̂ŜQ̂i(vi).left_cols(nu);
                    auto Qi    = R̂ŜQ̂i(vi).bottom_right(nx, nx);
                    auto iQiᵀ  = R̂ŜQ̂i(vi).block(nu - 1, nu, nx, nx);
                    auto AiQᵀ  = AinvQᵀ.batch(ti);
                    auto AiQiᵀ = AiQᵀ.middle_cols(i * nx, nx);
                    if (i > 0) {
                        auto LBAi      = LBAt.middle_cols((i - 1) * nx, nx);
                        auto R̂ŜQ̂i_prev = R̂ŜQ̂.middle_cols((i - 1) * nux, nux);
                        auto iQᵀprev  = R̂ŜQ̂i_prev(vi).block(nu - 1, nu, nx, nx);
                        auto AiQprevᵀ = AiQᵀ.middle_cols((i - 1) * nx, nx);
                        for (index_t c = 0; c < nx; ++c) {
                            for (index_t r = 0; r <= c; ++r)
                                tuples.emplace_back(s - nx + r, s - nx + c,
                                                    -iQᵀprev(r, c));
                            for (index_t r = 0; r < nux; ++r)
                                tuples.emplace_back(s + r, s - nx + c,
                                                    LBAi(vi)(r, c));
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
                                                    -iQiᵀ(r, c));
                        else
                            for (index_t r = 0; r <= c; ++r)
                                tuples.emplace_back(sλI + r, s + c + nu,
                                                    -iQiᵀ(r, c));
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
            PRINTLN("{:>2}: {:<2} {:<2} {:<2}", i, s, sY, sU);
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
            PRINTLN("{:>2}: {:<2} {:<2}", i, s, sY);
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

const int log_n_threads = 5; // TODO
TEST(NewCyclic, scheduling) {
    using namespace koqkatoo::ocp;

    KOQKATOO_OMP_IF(omp_set_num_threads(1 << log_n_threads));
    koqkatoo::pool_set_num_threads(1 << log_n_threads);
    koqkatoo::fork_set_num_threads(1 << log_n_threads);
    GUANAQO_IF_ITT(koqkatoo::foreach_thread([](index_t i, index_t) {
        __itt_thread_set_name(std::format("OMP({})", i).c_str());
    }));

    using Solver     = test::CyclicOCPSolver<16>;
    const index_t lP = log_n_threads + Solver::lvl;
    OCPDim dim{.N_horiz = 2 << lP, .nx = 5, .nu = 3, .ny = 10, .ny_N = 10};
    auto ocp = generate_random_ocp(dim);
    Solver solver{.dim = dim, .lP = lP};
    solver.initialize(ocp);
    std::vector<real_t> Σ_lin(dim.N_horiz * dim.ny + dim.ny_N);
    Solver::matrix λ{{.depth = dim.N_horiz, .rows = dim.nx, .cols = 1}},
        ux{{.depth = dim.N_horiz, .rows = dim.nu + dim.nx, .cols = 1}},
        Σ{{.depth = dim.N_horiz, .rows = dim.ny, .cols = 1}},
        Σ2{{.depth = dim.N_horiz, .rows = dim.ny, .cols = 1}},
        ΔΣ{{.depth = dim.N_horiz, .rows = dim.ny, .cols = 1}};
    std::mt19937 rng(102030405);
    std::uniform_real_distribution<real_t> uni(-1, 1);
    std::bernoulli_distribution bern(0.125);
    std::ranges::generate(λ, [&] { return uni(rng); });
    std::ranges::generate(ux, [&] { return uni(rng); });
    std::ranges::generate(Σ_lin, [&] { return std::exp2(uni(rng)); });
    std::vector<real_t> Σ_lin2 = Σ_lin;
    for (auto &Σ2i : Σ_lin2)
        if (bern(rng)) // TODO
            Σ2i = std::exp2(uni(rng));
    // std::ranges::transform(Σ_lin, Σ_lin2.begin(),
    //                        [](auto x) { return x + 1; }); // TODO

    solver.initialize_Σ(Σ_lin, Σ);
    solver.initialize_Σ(Σ_lin2, Σ2);
    std::ranges::transform(Σ2, Σ, std::ranges::begin(ΔΣ), std::minus<>{});
    for (index_t i = 0; i < dim.N_horiz && 0; ++i) {
        guanaqo::print_python(std::cout << "Σ1(" << i << "): ", Σ(i));
        guanaqo::print_python(std::cout << "Σ2(" << i << "): ", Σ2(i));
        guanaqo::print_python(std::cout << "ΔΣ(" << i << "): ", ΔΣ(i));
    }

    if (std::ofstream f("rhs.csv"); f) {
        auto b = solver.build_rhs(ux, λ);
        for (auto x : b)
            f << guanaqo::float_to_str(x) << '\n';
    }

    const bool alt = true;
    for (int i = 0; i < 500; ++i)
        solver.run(Σ, alt);
#if GUANAQO_WITH_TRACING
    guanaqo::trace_logger.reset();
#endif
    solver.run(Σ, alt);
    solver.update(ΔΣ);
    solver.solve(ux, λ);

#if GUANAQO_WITH_TRACING
    {
        koqkatoo::foreach_thread(
            [](index_t i, index_t) { GUANAQO_TRACE("thread_id", i); });
        const auto N     = solver.dim.N_horiz;
        const auto VL    = solver.vl;
        std::string name = std::format("factor_cyclic_new.csv");
        std::filesystem::path out_dir{"traces"};
#if USE_JACOBI_PREC
        const std::string_view pcg = "jacobi";
#else
        const std::string_view pcg = "stair";
#endif
        out_dir /= *koqkatoo_commit_hash ? koqkatoo_commit_hash : "unknown";
        out_dir /= KOQKATOO_MKL_IF_ELSE("mkl", "openblas");
        out_dir /= std::format("nx={}-nu={}-ny={}-N={}-thr={}-vl={}-pcg={}",
                               solver.dim.nx, solver.dim.nu, solver.dim.ny, N,
                               1 << log_n_threads, VL, pcg);
        std::filesystem::create_directories(out_dir);
        std::ofstream csv{out_dir / name};
        guanaqo::TraceLogger::write_column_headings(csv) << '\n';
        for (const auto &log : guanaqo::trace_logger.get_logs())
            csv << log << '\n';
        std::cout << out_dir << std::endl;
    }
#endif

    if (std::ofstream f("sparse.csv"); f) {
        auto sp = solver.build_sparse(ocp, Σ_lin2);
        for (auto [r, c, x] : sp)
            f << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
    if (std::ofstream f("sparse_factor.csv"); f) {
        auto sp = solver.build_sparse_factor();
        for (auto [r, c, x] : sp)
            f << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
    if (std::ofstream f("sparse_diag.csv"); f) {
        auto sp = solver.build_sparse_diag();
        for (auto [r, c, x] : sp)
            f << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
    if (std::ofstream f("sol.csv"); f) {
        auto b = solver.build_rhs(ux, λ);
        for (auto x : b)
            f << guanaqo::float_to_str(x) << '\n';
    }

    solver.run(Σ2, alt);
    if (std::ofstream f("sparse_refactor.csv"); f) {
        auto sp = solver.build_sparse_factor();
        for (auto [r, c, x] : sp)
            f << r << ',' << c << ',' << guanaqo::float_to_str(x) << '\n';
    }
}
