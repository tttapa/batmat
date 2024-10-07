#pragma once

#include <koqkatoo/cholundate/householder-updowndate.hpp>
#include <koqkatoo/kib.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/lapack.hpp>
#include <koqkatoo/loop.hpp>

#include <cmath>
#include <concepts>

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
                        microkernel_tail_lut<Abi>[rem_i - 1](A.cols(), W, Ls,
                                                             As, Ad, D);
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
                    auto Ls = L.block(i, k, rem_i, R);
                    microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                        A.cols(), W, Ls, As, Ad, D);
                },
                LoopDir::Backward); // TODO: decide on order
        });
    }
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
        if (use_blas_scalar(b)) {
            std::span D_span{D.data, static_cast<size_t>(D.rows())};
            cholundate::householder::updowndate_blocked<{}>(
                L(0), A(0), cholundate::DiagonalUpDowndate(D_span));
            return;
        }
    }
    xshhud_diag_ref(L, A, D);
}

} // namespace koqkatoo::linalg::compact
