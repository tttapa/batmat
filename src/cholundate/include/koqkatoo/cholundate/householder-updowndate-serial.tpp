#pragma once

#include <koqkatoo/cholundate/householder-updowndate-common.tpp>
#include <koqkatoo/cholundate/householder-updowndate.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/lut.hpp>
#include <type_traits>

namespace koqkatoo::cholundate::householder::inline serial {

#if 0
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
            downdate_tile_tail<uConf>(rem_i, A.cols, W, Ls, Ad, As);
        }
#else
        for (index_t i = k + R; i < L.rows; i += S) {
            index_t ni = std::min<index_t>(L.rows, i + S) - i;
            auto Ls    = L_.block(i, k);
            auto As    = A_.middle_rows(i);
            downdate_tile_tail<uConf>(ni, A.cols, W, Ls, Ad, As);
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
#elif 0
template <Config Conf>
void downdate_blocked(MutableRealMatrixView L, MutableRealMatrixView A) {
    constexpr index_t R = Conf.block_size_r, S = Conf.block_size_s;
    constexpr index_t N = Conf.num_blocks_r;
    constexpr micro_kernels::householder::Config uConf{.block_size_r = R,
                                                       .block_size_s = S};
    constexpr micro_kernels::householder::Config uConfR{.block_size_r = R,
                                                        .block_size_s = R};
    static_assert(Conf.num_blocks_s == 1, "NYI");
    assert(L.rows == L.cols);
    assert(L.rows == A.rows);
    constinit static auto full_microkernel_lut = make_1d_lut<R>(
        []<index_t N>(index_constant<N>) { return downdate_full<N + 1>; });

    // Leaner accessors (without unnecessary dimensions and strides).
    micro_kernels::mut_matrix_accessor L_{L}, A_{A};
    // Workspace storage for W (upper triangular Householder representation)
    micro_kernels::householder::matrix_W_storage<R> W[N];

    // Optional packing of one block row of A.
    auto A_pack_storage = [&] {
        if constexpr (Conf.enable_packing) {
            index_t num_pack = R * A.cols * N;
            return std::vector<real_t>(num_pack);
        } else {
            struct Empty {};
            return Empty{};
        }
    }();
    real_t *A_pack[N];
    if constexpr (Conf.enable_packing)
        for (index_t i = 0; i < N; ++i)
            A_pack[i] = &A_pack_storage[R * A.cols * i];
    auto pack_Ad = [&](index_t k) -> micro_kernels::mut_matrix_accessor {
        if constexpr (Conf.enable_packing) {
            MutableRealMatrixView Ad{
                {.data = A_pack[(k / R) % N], .rows = R, .cols = A.cols}};
            Ad = A.middle_rows(k, R);
            return Ad;
        }
        return A.middle_rows(k, R);
    };

    // Process all diagonal blocks (in multiples of NR, except the last).
    index_t k;
    for (k = 0; k + R * N <= L.rows; k += R * N) {
        micro_kernels::mut_matrix_accessor Adk[N];
        // Process all rows in the diagonal block (in multiples of R)
        for (index_t kk = 0; kk < R * N; kk += R) {
            // Pack the part of A corresponding to this diagonal block
            auto &Ad = Adk[kk / R] = pack_Ad(k + kk);
            // Process blocks left of the diagonal
            for (index_t cc = 0; cc < kk; cc += R) {
                auto Ls = L_.block(k + kk, k + cc);
                downdate_tail<uConfR>(A.cols, W[cc / R], Ls, Adk[cc / R], Ad);
            }
            auto Ld = L_.block(k + kk, k + kk);
            // Process the diagonal block itself
            downdate_diag<R>(A.cols, W[kk / R], Ld, Ad);
        }
        // Process all rows below the diagonal block (first in multiples of S).
        index_t i;
        for (i = k + R * N; i + S <= L.rows; i += S) {
            auto As = A_.middle_rows(i);
            // Process columns
            for (index_t cc = 0; cc < R * N; cc += R) {
                auto Ls = L_.block(i, k + cc);

                downdate_tail<uConf>(A.cols, W[cc / R], Ls, Adk[cc / R], As);
            }
        }
        // Then process any remainder smaller than S.
        index_t rem_i = L.rows - i;
        assert(rem_i < S);
        if (rem_i > 0) {
            auto As = A_.middle_rows(i);
            // Process columns
            for (index_t cc = 0; cc < R * N; cc += R) {
                auto Ls = L_.block(i, k + cc);
                for (index_t c = 0; c < R; ++c)
                    _mm_prefetch(&Ls(0, c), _MM_HINT_NTA);
                downdate_tile_tail<uConf>(rem_i, A.cols, W[cc / R], Ls,
                                          Adk[cc / R], As);
            }
        }
    }
    index_t rem_k = L.rows - k;
    assert(rem_k < R);
    if (rem_k > 0) {
        if (N != 1)
            throw std::logic_error("Not yet implemented");
        auto Ad = A_.middle_rows(k); // TODO: pack?
        auto Ld = L_.block(k, k);
        full_microkernel_lut[rem_k - 1](A.cols, Ld, Ad);
    }
}
#else
template <Config Conf, class UpDown>
void updowndate_blocked(MutableRealMatrixView L, MutableRealMatrixView A,
                        UpDown signs) {
    constexpr index_t R = Conf.block_size_r, S = Conf.block_size_s;
    constexpr index_t N       = Conf.num_blocks_r;
    constexpr bool do_packing = Conf.enable_packing;
    constexpr micro_kernels::householder::Config uConf{
        .block_size_r        = R,
        .block_size_s        = S,
        .prefetch_dist_col_a = Conf.prefetch_dist_col_a};
    constexpr micro_kernels::householder::Config uConfR{
        .block_size_r        = R,
        .block_size_s        = R,
        .prefetch_dist_col_a = Conf.prefetch_dist_col_a};
    static_assert(Conf.num_blocks_s == 1, "NYI");
    assert(L.rows == L.cols);
    assert(L.rows == A.rows);
    constinit static auto full_microkernel_lut =
        make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
            return updowndate_full<NR + 1, UpDown>;
        });

    // Leaner accessors (without unnecessary dimensions and strides).
    micro_kernels::mut_matrix_accessor L_{L}, A_{A};
    // Workspace storage for W (upper triangular Householder representation)
    micro_kernels::householder::matrix_W_storage<> W[N];

    // Optional packing of one block row of A.
    auto A_pack_storage = [&] {
        if constexpr (do_packing) {
            index_t num_pack = R * A.cols * N;
            return std::vector<real_t>(num_pack);
        } else {
            struct Empty {};
            return Empty{};
        }
    }();
    real_t *A_pack[N];
    if constexpr (do_packing)
        for (index_t i = 0; i < N; ++i)
            A_pack[i] = &A_pack_storage[R * A.cols * i];
    auto pack_Ad = [&](index_t k) -> micro_kernels::mut_matrix_accessor {
        if constexpr (do_packing) {
            MutableRealMatrixView Ad{
                {.data = A_pack[(k / R) % N], .rows = R, .cols = A.cols}};
            Ad = A.middle_rows(k, R);
            return Ad;
        }
        return A.middle_rows(k, R);
    };

    // Process all diagonal blocks (in multiples of NR, except the last).
    index_t k;
    for (k = 0; k + R * N <= L.rows; k += R * N) {
        micro_kernels::mut_matrix_accessor Adk[N];
        // Process all rows in the diagonal block (in multiples of R)
        for (index_t kk = 0; kk < R * N; kk += R) {
            // Pack the part of A corresponding to this diagonal block
            auto &Ad = Adk[kk / R] = pack_Ad(k + kk);
            // Process blocks left of the diagonal
            for (index_t cc = 0; cc < kk; cc += R) {
                auto Ls = L_.block(k + kk, k + cc);
                updowndate_tail<uConfR, UpDown>(0, A.cols, W[cc / R], Ls,
                                                Adk[cc / R], Ad, signs);
            }
            auto Ld = L_.block(k + kk, k + kk);
            // Process the diagonal block itself
            updowndate_diag<R, UpDown>(A.cols, W[kk / R], Ld, Ad, signs);
        }
        // Process all rows below the diagonal block (in multiples of S).
        foreach_chunked(
            k + R * N, L.rows, std::integral_constant<index_t, S>(),
            [&](index_t i) {
                auto As = A_.middle_rows(i);
                // Process columns
                for (index_t cc = 0; cc < R * N; cc += R) {
                    auto Ls = L_.block(i, k + cc);
                    for (index_t c = 0; c < R; ++c)
                        _mm_prefetch(&Ls(0, c), _MM_HINT_NTA);
                    updowndate_tail<uConf, UpDown>(0, A.cols, W[cc / R], Ls,
                                                   Adk[cc / R], As, signs);
                }
            },
            [&](index_t i, index_t rem_i) {
                auto As = A_.middle_rows(i);
                // Process columns
                for (index_t cc = 0; cc < R * N; cc += R) {
                    auto Ls = L_.block(i, k + cc);
                    for (index_t c = 0; c < R; ++c)
                        _mm_prefetch(&Ls(0, c), _MM_HINT_NTA);
                    updowndate_tile_tail<uConf, UpDown>(rem_i, 0, A.cols,
                                                        W[cc / R], Ls,
                                                        Adk[cc / R], As, signs);
                }
            },
            LoopDir::Forward);
    }
    index_t rem_k = L.rows - k;
    assert(rem_k < R);
    if (rem_k > 0) {
        if (N != 1)
            throw std::logic_error("Not yet implemented");
        auto Ad = A_.middle_rows(k); // TODO: pack?
        auto Ld = L_.block(k, k);
        full_microkernel_lut[rem_k - 1](A.cols, Ld, Ad, signs);
    }
}
#endif

} // namespace koqkatoo::cholundate::householder::inline serial
