#pragma once

#include <koqkatoo/cholundate/householder-downdate-common.tpp>
#include <koqkatoo/cholundate/householder-downdate.hpp>
#include <koqkatoo/lut.hpp>

namespace koqkatoo::cholundate::householder::inline serial {

template <Config Conf>
void downdate_blocked(MutableRealMatrixView L, MutableRealMatrixView A) {
    constexpr index_t R = Conf.block_size_r, S = Conf.block_size_s;
    constexpr micro_kernels::householder::Config uConf{.block_size_r = R,
                                                       .block_size_s = S};
    static_assert(Conf.num_blocks_r == 1 && Conf.num_blocks_s == 1, "NYI");
    assert(L.rows == L.cols);
    assert(L.rows == A.rows);
    constinit static auto full_microkernel_lut = make_1d_lut<R>(
        []<index_t N>(index_constant<N>) { return downdate_full<N + 1>; });

    // Leaner accessors (without unnecessary dimensions and strides).
    micro_kernels::mut_matrix_accessor L_{L}, A_{A};
    // Workspace storage for W (upper triangular Householder representation)
    micro_kernels::householder::matrix_W_storage<R> W;

    // Optional packing of one block row of A.
    auto A_pack = [&] {
        if constexpr (Conf.enable_packing) {
            index_t num_pack = R * A.cols;
            return std::vector<real_t>(num_pack);
        } else {
            struct Empty {};
            return Empty{};
        }
    }();
    auto pack_Ad = [&](index_t k) -> micro_kernels::mut_matrix_accessor {
        if constexpr (Conf.enable_packing) {
            MutableRealMatrixView Ad{
                {.data = A_pack.data(), .rows = R, .cols = A.cols}};
            Ad = A.middle_rows(k, R);
            return Ad;
        }
        return A.middle_rows(k, R);
    };

    // Process all diagonal blocks in multiples of R (except the last).
    index_t k;
    for (k = 0; k + R < L.rows; k += R) {
        auto Ad = pack_Ad(k);
        auto Ld = L_.block(k, k);
        downdate_diag<R>(A.cols, W, Ld, Ad);
#if 1
        // First process the sub-diagonal blocks in multiples of S.
        index_t i;
        for (i = k + R; i <= L.rows - S; i += S) {
            auto Ls = L_.block(i, k);
            auto As = A_.middle_rows(i);
            downdate_tail<uConf>(A.cols, W, Ls, Ad, As);
        }
        // Then process any remainder smaller than S.
        index_t rem_i = L.rows - i;
        assert(rem_i < S);
        if (rem_i > 0) {
            auto Ls = L_.block(i, k);
            auto As = A_.middle_rows(i);
            tile_tail<uConf>(rem_i, A.cols, W, Ls, Ad, As);
        }
#else
        for (index_t i = k + R; i < L.rows; i += S) {
            index_t ni = std::min<index_t>(L.rows, i + S) - i;
            auto Ls    = L_.block(i, k);
            auto As    = A_.middle_rows(i);
            tile_tail<uConf>(ni, A.cols, W, Ls, Ad, As);
        }
#endif
        // TODO: Could it be beneficial to traverse all even columns from top to
        //       bottom, and the odd columns from bottom to top?
    }
    index_t rem_k = L.rows - k;
    assert(rem_k <= R);
    auto Ad = A_.middle_rows(k); // TODO: pack?
    auto Ld = L_.block(k, k);
    full_microkernel_lut[rem_k - 1](A.cols, Ld, Ad);
}

} // namespace koqkatoo::cholundate::householder::inline serial
