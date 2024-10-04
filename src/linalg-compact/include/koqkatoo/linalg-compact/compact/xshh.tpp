#pragma once

#include <koqkatoo/cholundate/householder-downdate.hpp>
#include <koqkatoo/kib.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/lapack.hpp>
#include <koqkatoo/loop.hpp>

#include <cmath>
#include <concepts>

#include <koqkatoo/linalg-compact/compact/micro-kernels/xshh.hpp>
#include "util.hpp"

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xshh_ref(mut_single_batch_view L,
                                mut_single_batch_view A) {
    using namespace micro_kernels;
    static constexpr index_constant<shh::SizeR> R;
    static constexpr index_constant<shh::SizeS> S;

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked(
        0, L.rows(), R,
        [&](index_t k) {
            // Part of A corresponding to this diagonal block
            // TODO: packing
            auto Ad = A.middle_rows(k, R);
            auto Ld = L.block(k, k, R, R);
            // Process the diagonal block itself
            using W_t = shh::triangular_accessor<Abi, real_t, R>;
            alignas(W_t::alignment()) real_t W[W_t::size()];
            shh::xshh_diag_microkernel<Abi, R>(A.cols(), W, Ld, Ad);
            // Process all rows below the diagonal block (in multiples of S).
            foreach_chunked_merged(
                k + R, L.rows(), S,
                [&](index_t i, auto rem_i) {
                    auto As = A.middle_rows(i, rem_i);
                    auto Ls = L.block(i, k, rem_i, R);
                    shh::microkernel_tail_lut<Abi>[rem_i - 1](A.cols(), W, Ls,
                                                              As, Ad);
                },
                LoopDir::Backward); // TODO: decide on order
        },
        [&](index_t k, index_t rem_k) {
            auto Ad = A.middle_rows(k, rem_k);
            auto Ld = L.block(k, k, rem_k, rem_k);
            shh::microkernel_full_lut<Abi>[rem_k - 1](A.cols(), Ld, Ad);
        });
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xshh(mut_batch_view L, mut_batch_view A,
                            PreferredBackend b) {
    assert(L.rows() == L.cols());
    assert(L.rows() == A.rows());
    assert(L.depth() == A.depth());
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < L.num_batches(); ++i)
            xshh(L.batch(i), A.batch(i), b);
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xshh(L.batch(i), A.batch(i), b);
}

template <class Abi>
void CompactBLAS<Abi>::xshh(mut_single_batch_view L, mut_single_batch_view A,
                            PreferredBackend b) {
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            cholundate::householder::downdate_blocked<{}>(L(0), A(0));
            return;
        }
    }
    xshh_ref(L, A);
}

} // namespace koqkatoo::linalg::compact
