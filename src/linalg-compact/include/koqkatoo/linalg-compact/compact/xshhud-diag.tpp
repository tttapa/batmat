#pragma once

#include <koqkatoo/cholundate/householder-updowndate.hpp>
#include <koqkatoo/kib.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/lapack.hpp>
#include <koqkatoo/loop.hpp>

#include <cmath>
#include <concepts>
#include <print>

#include <koqkatoo/linalg-compact/compact/micro-kernels/xshhud-diag.hpp>
#include "util.hpp"

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xshhud_diag_ref(mut_single_batch_view L,
                                       mut_single_batch_view A,
                                       single_batch_view D) {
    using namespace micro_kernels::shhud_diag;
    static constexpr index_constant<SizeR> R;
    static constexpr index_constant<SizeS> S;
    const index_t C = A.cols();
    auto n1 = L.cols(), n2 = L.rows() - n1;
    auto flop_count_diag_11 = (C + 1) * n1 * n1 + 2 * C * n1;
    auto flop_count_tail_11 = 2 * (C + 1) * n2 * n1 + C * n2;
    [[maybe_unused]] index_t flop_count =
        flop_count_diag_11 + flop_count_tail_11;
    GUANAQO_TRACE("xshhud_diag", 0, flop_count * L.depth());
    if (C == 0)
        return;

    using W_t = triangular_accessor<Abi, real_t, R>;
    alignas(W_t::alignment()) real_t W[W_t::size()];

    // Process all diagonal blocks (in multiples of R, except the last).
    if (L.rows() == L.cols()) {
        foreach_chunked(
            0, L.cols(), R,
            [&](index_t k) {
                // Part of A corresponding to this diagonal block
                // TODO: packing
                auto Ad = A.middle_rows(k, R);
                auto Ld = L.block(k, k, R, R);
                // Process the diagonal block itself
                xshhud_diag_diag_microkernel<Abi, R>(A.cols(), W, Ld, Ad, D);
                // Process all rows below the diagonal block (in multiples of S).
                foreach_chunked_merged(
                    k + R, L.rows(), S,
                    [&](index_t i, auto rem_i) {
                        auto As = A.middle_rows(i, rem_i);
                        auto Ls = L.block(i, k, rem_i, R);
                        microkernel_tail_lut<Abi>[rem_i - 1](
                            0, A.cols(), A.cols(), W, Ls, As, As, Ad, D,
                            Structure::General, 0);
                    },
                    LoopDir::Backward); // TODO: decide on order
            },
            [&](index_t k, index_t rem_k) {
                auto Ad = A.middle_rows(k, rem_k);
                auto Ld = L.block(k, k, rem_k, rem_k);
                microkernel_full_lut<Abi>[rem_k - 1](A.cols(), Ld, Ad, D);
            });
    } else {
        foreach_chunked_merged(0, L.cols(), R, [&](index_t k, auto rem_k) {
            // Part of A corresponding to this diagonal block
            // TODO: packing
            auto Ad = A.middle_rows(k, rem_k);
            auto Ld = L.block(k, k, rem_k, rem_k);
            // Process the diagonal block itself
            microkernel_diag_lut<Abi>[rem_k - 1](A.cols(), W, Ld, Ad, D);
            // Process all rows below the diagonal block (in multiples of S).
            foreach_chunked_merged(
                k + rem_k, L.rows(), S,
                [&](index_t i, auto rem_i) {
                    auto As = A.middle_rows(i, rem_i);
                    auto Ls = L.block(i, k, rem_i, rem_k);
                    microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                        0, A.cols(), A.cols(), W, Ls, As, As, Ad, D,
                        Structure::General, 0);
                },
                LoopDir::Backward); // TODO: decide on order
        });
    }
}

template <class Abi>
void CompactBLAS<Abi>::xshhud_diag_2_ref(mut_single_batch_view L11,
                                         mut_single_batch_view A1,
                                         mut_single_batch_view L21,
                                         mut_single_batch_view A2,
                                         single_batch_view D) {
    assert(L11.rows() >= L11.cols());
    assert(L11.rows() == A1.rows());
    assert(A1.cols() == D.rows());
    assert(A2.cols() == A1.cols());
    assert(L21.cols() == L11.cols());
    assert(L21.rows() == A2.rows());
    using namespace micro_kernels::shhud_diag;
    static constexpr index_constant<SizeR> R;
    static constexpr index_constant<SizeS> S;
    const index_t C = A1.cols();
    auto n1 = L11.cols(), n2 = L11.rows() - n1;
    auto flop_count_diag_11 = (C + 1) * n1 * n1 + 2 * C * n1;
    auto flop_count_tail_11 = 2 * (C + 1) * n2 * n1 + C * n2;
    auto flop_count_tail_21 = 2 * (C + 1) * L21.rows() * n1 + C * L21.rows();
    [[maybe_unused]] index_t flop_count =
        flop_count_diag_11 + flop_count_tail_11 + flop_count_tail_21;
    GUANAQO_TRACE("xshhud_diag_2", 0, flop_count * L11.depth());
    if (C == 0)
        return;

    using W_t = triangular_accessor<Abi, real_t, R>;
    alignas(W_t::alignment()) real_t W[W_t::size()];

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L11.cols(), R, [&](index_t k, auto rem_k) {
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A1.middle_rows(k, rem_k);
        auto Ld = L11.block(k, k, rem_k, rem_k);
        // Process the diagonal block itself
        microkernel_diag_lut<Abi>[rem_k - 1](A1.cols(), W, Ld, Ad, D);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            k + rem_k, L11.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As = A1.middle_rows(i, rem_i);
                auto Ls = L11.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                    0, A1.cols(), A1.cols(), W, Ls, As, As, Ad, D,
                    Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L21.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As = A2.middle_rows(i, rem_i);
                auto Ls = L21.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                    0, A1.cols(), A1.cols(), W, Ls, As, As, Ad, D,
                    Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

template <class Abi>
void CompactBLAS<Abi>::xshhud_diag_2T_ref(mut_single_batch_view L11,
                                          mut_single_batch_view A1,
                                          mut_single_batch_view L21ᵀ,
                                          mut_single_batch_view A2,
                                          single_batch_view D) {
    assert(L11.rows() >= L11.cols());
    assert(L11.rows() == A1.rows());
    assert(A1.cols() == D.rows());
    assert(A2.cols() == A1.cols());
    assert(L21ᵀ.rows() == L11.cols());
    assert(L21ᵀ.cols() == A2.rows());
    using namespace micro_kernels::shhud_diag;
    static constexpr index_constant<SizeR> R;
    static constexpr index_constant<SizeS> S;
    const index_t C = A1.cols();
    auto n1 = L11.cols(), n2 = L11.rows() - n1;
    auto flop_count_diag_11 = (C + 1) * n1 * n1 + 2 * C * n1;
    auto flop_count_tail_11 = 2 * (C + 1) * n2 * n1 + C * n2;
    auto flop_count_tail_21 = 2 * (C + 1) * L21ᵀ.cols() * n1 + C * L21ᵀ.cols();
    [[maybe_unused]] index_t flop_count =
        flop_count_diag_11 + flop_count_tail_11 + flop_count_tail_21;
    GUANAQO_TRACE("xshhud_diag_2T", 0, flop_count * L11.depth());
    if (C == 0)
        return;

    using W_t = triangular_accessor<Abi, real_t, R>;
    alignas(W_t::alignment()) real_t W[W_t::size()];

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L11.cols(), R, [&](index_t k, auto rem_k) {
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A1.middle_rows(k, rem_k);
        auto Ld = L11.block(k, k, rem_k, rem_k);
        // Process the diagonal block itself
        microkernel_diag_lut<Abi>[rem_k - 1](A1.cols(), W, Ld, Ad, D);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            k + rem_k, L11.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As = A1.middle_rows(i, rem_i);
                auto Ls = L11.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                    0, A1.cols(), A1.cols(), W, Ls, As, As, Ad, D,
                    Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L21ᵀ.cols(), S,
            [&](index_t i, auto rem_i) {
                auto As = A2.middle_rows(i, rem_i);
                auto Ls = L21ᵀ.block(k, i, rem_k, rem_i);
                microkernel_tail_lut_2<Abi, true>[rem_k - 1][rem_i - 1](
                    0, A1.cols(), A1.cols(), W, Ls, As, As, Ad, D,
                    Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

/**
 * Performs a factorization update of the following matrix:
 *
 *     [ A11 A12 | L11 ]     [  0   0  | L̃11 ]
 *     [  0  A22 | L21 ] Q = [ Ã21 Ã22 | L̃21 ]
 *     [ A31  0  | L31 ]     [ Ã31 Ã32 | L̃31 ]
 *           ↑ split_A
 */
template <class Abi>
void CompactBLAS<Abi>::xshhud_diag_cyclic(
    mut_single_batch_view L11, mut_single_batch_view A1,
    mut_single_batch_view L21, single_batch_view A2,
    mut_single_batch_view A2_out, mut_single_batch_view L31,
    single_batch_view A3, mut_single_batch_view A3_out, single_batch_view D,
    index_t split_A, int rot_A2) {
    assert(L11.rows() >= L11.cols());
    assert(L11.rows() == A1.rows());
    assert(L21.rows() == A2.rows());
    assert(L31.rows() == A3.rows());
    assert(A1.cols() == D.rows());
    assert(A2.cols() == A1.cols());
    assert(A3.cols() == A1.cols());
    assert(L21.cols() == L11.cols());
    assert(L31.cols() == L11.cols());
    using namespace micro_kernels::shhud_diag;
    static constexpr index_constant<SizeR> R;
    static constexpr index_constant<SizeS> S;
    const index_t C = A1.cols();
    auto n1 = L11.cols(), n2 = L11.rows() - n1;
    auto flop_count_diag_11 = (C + 1) * n1 * n1 + 2 * C * n1;
    auto flop_count_tail_11 = 2 * (C + 1) * n2 * n1 + C * n2;
    auto flop_count_tail_21 = 2 * (C + 1) * L21.rows() * n1 + C * L21.rows();
    auto flop_count_tail_31 = 2 * (C + 1) * L31.rows() * n1 + C * L31.rows();
    // Note: initial zero values of A for simplicity (for large matrices this
    //       does not matter)
    [[maybe_unused]] index_t flop_count =
        flop_count_diag_11 + flop_count_tail_11 + flop_count_tail_21 +
        flop_count_tail_31;
    GUANAQO_TRACE("xshhud_diag_cyclic", 0, flop_count * L11.depth());
    if (C == 0)
        return;

    using W_t = triangular_accessor<Abi, real_t, R>;
    alignas(W_t::alignment()) real_t W[W_t::size()];

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L11.cols(), R, [&](index_t k, auto rem_k) {
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A1.middle_rows(k, rem_k);
        auto Ld = L11.block(k, k, rem_k, rem_k);
        // Process the diagonal block itself
        microkernel_diag_lut<Abi>[rem_k - 1](C, W, Ld, Ad, D);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            k + rem_k, L11.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As = A1.middle_rows(i, rem_i);
                auto Ls = L11.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                    0, C, C, W, Ls, As, As, Ad, D, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        const auto rot = k + rem_k == L11.cols() ? rot_A2 : 0;
        foreach_chunked_merged(
            0, L21.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As_out = A2_out.middle_rows(i, rem_i);
                auto As     = k == 0 ? A2.middle_rows(i, rem_i) : As_out;
                auto Ls     = L21.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                    k == 0 ? split_A : 0, C, C, W, Ls, As, As_out, Ad, D,
                    Structure::General, rot);
                // First half of A2 is implicitly zero in first pass
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, L31.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As_out = A3_out.middle_rows(i, rem_i);
                auto As     = k == 0 ? A3.middle_rows(i, rem_i) : As_out;
                auto Ls     = L31.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                    0, k == 0 ? split_A : C, C, W, Ls, As, As_out, Ad, D,
                    Structure::General, 0);
                // Second half of A3 is implicitly zero in first pass
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

/**
 * Performs a factorization update of the following matrix:
 *
 *     [ A1 | L11 ]     [  0 | L̃11 ]
 *     [ A2 | L21 ] Q = [ Ã2 | L̃21 ]
 *     [  0 | Lu1 ]     [ Ãu | L̃u1 ]
 * where Lu1 and L̃u1 are upper triangular
 */
template <class Abi>
template <bool TransL21>
void CompactBLAS<Abi>::xshhud_diag_riccati(
    mut_single_batch_view L11, mut_single_batch_view A1,
    mut_single_batch_view L21, single_batch_view A2,
    mut_single_batch_view A2_out, mut_single_batch_view Lu1,
    mut_single_batch_view Au_out, single_batch_view D, bool shift_A_out) {
    assert(L11.rows() >= L11.cols());
    assert(L11.rows() == A1.rows());
    assert(A2_out.rows() == A2.rows());
    assert(A2_out.cols() == A2.cols());
    assert(Lu1.rows() == Au_out.rows());
    assert(A1.cols() == D.rows());
    assert(A2.cols() == A1.cols());
    const auto r_L21                  = TransL21 ? L21.cols() : L21.rows();
    [[maybe_unused]] const auto c_L21 = TransL21 ? L21.rows() : L21.cols();
    assert(L21.cols() == L11.cols());
    assert(r_L21 == A2.rows());
    assert(c_L21 == L11.cols());
    assert(Lu1.cols() == L11.cols());
    using namespace micro_kernels::shhud_diag;
    static constexpr index_constant<SizeR> R;
    static constexpr index_constant<SizeS> S;
    static_assert(R == S);
    const index_t C = A1.cols();
    auto n1 = L11.cols(), n2 = L11.rows() - n1;
    auto flop_count_diag_11 = (C + 1) * n1 * n1 + 2 * C * n1;
    auto flop_count_tail_11 = 2 * (C + 1) * n2 * n1 + C * n2;
    auto flop_count_tail_21 = 2 * (C + 1) * r_L21 * n1 + C * r_L21;
    auto flop_count_tail_u1 = 2 * (C + 1) * Lu1.rows() * n1 + C * Lu1.rows();
    // Note: ignoring upper trapezoidal shape and initial zero value of Au for
    //       simplicity (for large matrices this does not matter)
    [[maybe_unused]] index_t flop_count =
        flop_count_diag_11 + flop_count_tail_11 + flop_count_tail_21 +
        flop_count_tail_u1;
    GUANAQO_TRACE("xshhud_diag_riccati", 0, flop_count * L11.depth());
    if (C == 0)
        return;

    using W_t = triangular_accessor<Abi, real_t, R>;
    alignas(W_t::alignment()) real_t W[W_t::size()];

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, L11.cols(), R, [&](index_t k, auto rem_k) {
        const bool do_shift = shift_A_out && k + rem_k == L11.cols();
        // Part of A corresponding to this diagonal block
        // TODO: packing
        auto Ad = A1.middle_rows(k, rem_k);
        auto Ld = L11.block(k, k, rem_k, rem_k);
        // Process the diagonal block itself
        microkernel_diag_lut<Abi>[rem_k - 1](C, W, Ld, Ad, D);
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked_merged(
            k + rem_k, L11.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As = A1.middle_rows(i, rem_i);
                auto Ls = L11.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                    0, C, C, W, Ls, As, As, Ad, D, Structure::General, 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, r_L21, S,
            [&](index_t i, auto rem_i) {
                auto As_out = A2_out.middle_rows(i, rem_i);
                auto As     = k == 0 ? A2.middle_rows(i, rem_i) : As_out;
                auto Ls     = TransL21 ? L21.block(k, i, rem_k, rem_i)
                                       : L21.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<Abi, TransL21>[rem_k - 1][rem_i - 1](
                    0, C, C, W, Ls, As, As_out, Ad, D, Structure::General,
                    do_shift ? -1 : 0);
            },
            LoopDir::Backward); // TODO: decide on order
        foreach_chunked_merged(
            0, Lu1.rows(), S,
            [&](index_t i, auto rem_i) {
                auto As_out = Au_out.middle_rows(i, rem_i);
                auto As     = As_out;
                auto Ls     = Lu1.block(i, k, rem_i, rem_k);
                microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                    0, k == 0 ? 0 : C, C, W, Ls, As, As_out, Ad, D,
                    i == k  ? Structure::Upper
                    : i < k ? Structure::General
                            : Structure::Zero,
                    do_shift ? -1 : 0);
                // TODO: rotate +1 or -1
                // Au is implicitly zero in first pass
            },
            LoopDir::Backward); // TODO: decide on order
    });
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xshhud_diag(mut_batch_view L, mut_batch_view A,
                                   batch_view D, PreferredBackend b) {
    assert(L.rows() >= L.cols());
    assert(L.rows() == A.rows());
    assert(L.depth() == A.depth());
    assert(D.depth() == A.depth());
    assert(D.rows() == A.cols());
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < L.num_batches(); ++i)
            xshhud_diag(L.batch(i), A.batch(i), D.batch(i), b);
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xshhud_diag(L.batch(i), A.batch(i), D.batch(i), b);
}

template <class Abi>
void CompactBLAS<Abi>::xshhud_diag(mut_single_batch_view L,
                                   mut_single_batch_view A, single_batch_view D,
                                   PreferredBackend b) {
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b) && L.rows() == L.cols()) {
            // TODO: implement non-square case efficiently
            std::span D_span{D.data, static_cast<size_t>(D.rows())};
            cholundate::householder::updowndate_blocked<{}>(
                L(0), A(0), cholundate::DiagonalUpDowndate(D_span));
            return;
        }
    }
    xshhud_diag_ref(L, A, D);
}

} // namespace koqkatoo::linalg::compact
