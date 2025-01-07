#pragma once

#include <koqkatoo/cholundate/householder-updowndate-common.tpp>
#include <koqkatoo/cholundate/householder-updowndate.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/lut.hpp>

namespace koqkatoo::cholundate::householder::naive {

template <Config Conf, class UpDown>
void updowndate_blocked(MutableRealMatrixView L, MutableRealMatrixView A,
                        UpDown signs) {
    if constexpr (requires { signs.size(); })
        assert(A.cols == signs.size());
    assert(L.rows >= L.cols);
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

    auto updowndate_diag_lut =
        make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
            return updowndate_diag<NR + 1, UpDown>;
        });
    auto updowndate_tile_tail_lut =
        make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
            constexpr micro_kernels::householder::Config uConfNR{
                .block_size_r        = NR + 1,
                .block_size_s        = S,
                .prefetch_dist_col_a = Conf.prefetch_dist_col_a};
            return updowndate_tile_tail<uConfNR, UpDown>;
        });

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked(
        0, L.cols, R,
        [&](index_t k) {
            // Part of A corresponding to this diagonal block
            // TODO: packing
            auto Ad = A.middle_rows(k, R);
            auto Ld = L.block(k, k, R, R);
            // Process the diagonal block itself
            micro_kernels::householder::matrix_W_storage<> W;
            updowndate_diag<R, UpDown>(A.cols, W, Ld, Ad, signs);
            // Process all rows below the diagonal block (in multiples of S).
            foreach_chunked_merged(
                k + R, L.rows, S,
                [&](index_t i, auto rem_i) {
                    auto As = A.middle_rows(i, rem_i);
                    auto Ls = L.block(i, k, rem_i, R);
                    updowndate_tile_tail<uConf, UpDown>(rem_i, 0, As.cols, W,
                                                        Ls, Ad, As, signs);
                },
                LoopDir::Backward); // TODO: decide on order
        },
        [&](index_t k, index_t rem_k) {
            if (L.rows == L.cols) [[likely]] {
                auto updowndate_full_lut =
                    make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
                        return updowndate_full<NR + 1, UpDown>;
                    });
                auto Ad = A.middle_rows(k, rem_k);
                auto Ld = L.block(k, k, rem_k, rem_k);
                updowndate_full_lut[rem_k - 1](A.cols, Ld, Ad, signs);
            } else {
                // Part of A corresponding to this diagonal block
                // TODO: packing
                auto Ad = A.middle_rows(k, rem_k);
                auto Ld = L.block(k, k, rem_k, rem_k);
                // Process the diagonal block itself
                micro_kernels::householder::matrix_W_storage<> W;
                updowndate_diag_lut[rem_k - 1](A.cols, W, Ld, Ad, signs);
                // Process all rows below the diagonal block (in multiples of S).
                foreach_chunked_merged(
                    k + rem_k, L.rows, S,
                    [&](index_t i, auto rem_i) {
                        auto As = A.middle_rows(i, rem_i);
                        auto Ls = L.block(i, k, rem_i, rem_k);
                        updowndate_tile_tail_lut[rem_k - 1](
                            rem_i, 0, As.cols, W, Ls, Ad, As, signs);
                    },
                    LoopDir::Backward); // TODO: decide on order
            }
        });
}

} // namespace koqkatoo::cholundate::householder::naive
