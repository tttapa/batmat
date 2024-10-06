#pragma once

#include <koqkatoo/cholundate/householder-downdate-common.tpp>
#include <koqkatoo/cholundate/householder-downdate.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/lut.hpp>

namespace koqkatoo::cholundate::householder::inline serial {

template <Config Conf>
void updowndate_blocked(MutableRealMatrixView L, MutableRealMatrixView A,
                        RealMatrixView signs) {
    assert(signs.rows == A.cols);
    assert(L.rows == L.cols);
    assert(L.rows == A.rows);
    using namespace micro_kernels;
    static constexpr index_constant<Conf.block_size_r> R;
    static constexpr index_constant<Conf.block_size_s> S;
    constexpr micro_kernels::householder::Config uConf{
        .block_size_r        = R,
        .block_size_s        = S,
        .prefetch_dist_col_a = Conf.prefetch_dist_col_a};
    static_assert(Conf.num_blocks_r == 1, "NYI");
    static_assert(Conf.num_blocks_s == 1, "NYI");
    static_assert(!Conf.enable_packing, "NYI");

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked(
        0, L.rows, R,
        [&](index_t k) {
            // Part of A corresponding to this diagonal block
            // TODO: packing
            auto Ad = A.middle_rows(k, R);
            auto Ld = L.block(k, k, R, R);
            // Process the diagonal block itself
            micro_kernels::householder::matrix_W_storage<R> W;
            updowndate_diag<R>(A.cols, W, Ld, Ad, signs);
            // Process all rows below the diagonal block (in multiples of S).
            foreach_chunked_merged(
                k + R, L.rows, S,
                [&](index_t i, auto rem_i) {
                    auto As = A.middle_rows(i, rem_i);
                    auto Ls = L.block(i, k, rem_i, R);
                    updowndate_tile_tail<uConf>(rem_i, As.cols, W, Ls, Ad, As,
                                                signs);
                },
                LoopDir::Backward); // TODO: decide on order
        },
        [&](index_t k, index_t rem_k) {
            auto updowndate_full_lut =
                make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
                    return updowndate_full<NR + 1>;
                });
            auto Ad = A.middle_rows(k, rem_k);
            auto Ld = L.block(k, k, rem_k, rem_k);
            updowndate_full_lut[rem_k - 1](A.cols, Ld, Ad, signs);
        });
}

} // namespace koqkatoo::cholundate::householder::inline serial
