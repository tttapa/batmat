#pragma once

#include <koqkatoo/loop.hpp>
#include <koqkatoo/lut.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/openmp.h>

#include <algorithm>

#include <koqkatoo/cholundate/householder-updowndate-common.tpp>
#include <koqkatoo/cholundate/householder-updowndate.hpp>
#include <koqkatoo/cholundate/micro-kernels/householder-updowndate.hpp>

namespace koqkatoo::ocp {

template <simd_abi_tag Abi>
void Solver<Abi>::updowndate_ψ() {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    auto &ranks                = storage.stagewise_update_counts;

    // Each block of V generates fill-in in A in the block row below it, and
    // W adds new columns to A, but we only ever need two block rows of A at any
    // given time, thanks to the block-tridiagonal structure of Ψ, so we use a
    // sliding window of depth 2, and use the stage index modulo 2.
    index_t colsA = 0;
    // TODO: move to storage
    scalar_real_matrix A{{.depth = 2, .rows = nx, .cols = (N + 1) * ny}};
    scalar_real_matrix D{{.depth = 1, .rows = (N + 1) * ny, .cols = 1}};
    using namespace cholundate;
    static constexpr index_constant<householder::DefaultSizeR> R;
    static constexpr index_constant<householder::DefaultSizeS> S;
    using W_t = micro_kernels::householder::matrix_W_storage<>;

    constinit static const auto microkernel_diag_lut =
        make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
            return householder::updowndate_diag<NR + 1, DownUpdate>;
        });
    constinit static const auto microkernel_full_lut =
        make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
            return householder::updowndate_full<NR + 1, DownUpdate>;
        });
    constinit static const auto microkernel_tail_lut =
        make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
            constexpr micro_kernels::householder::Config uConf{
                .block_size_r = NR + 1, .block_size_s = S};
            return householder::updowndate_tile_tail<uConf, DownUpdate>;
        });

    auto process_diag_block = [&](index_t k, auto rem_k, auto Ad, auto Dd,
                                  auto Ld, W_t &W) {
        auto Add = Ad.middle_rows(k, rem_k);
        auto Ldd = Ld.block(k, k, rem_k, rem_k);
        microkernel_diag_lut[rem_k - 1](colsA, W, Ldd, Add, DownUpdate{Dd});
    };
    auto process_subdiag_block_W = [&](index_t k, auto Ad, auto Dd, auto Ld,
                                       W_t &W) {
        auto Add = Ad.middle_rows(k, R);
        foreach_chunked_merged(k + R, Ld.rows, S, [&](index_t i, auto rem_i) {
            auto Ads = Ad.middle_rows(i, rem_i);
            auto Lds = Ld.block(i, k, rem_i, R);
            microkernel_tail_lut[R - 1](rem_i, colsA, W, Lds, Add, Ads,
                                        DownUpdate{Dd});
        });
    };
    auto process_subdiag_block_V = [&](index_t k, index_t rem_k, auto As,
                                       auto Ls, auto Ad, auto Dd, W_t &W) {
        auto Add = Ad.middle_rows(k, rem_k);
        foreach_chunked_merged(0, Ls.rows, S, [&](index_t i, auto rem_i) {
            auto Ass = As.middle_rows(i, rem_i);
            auto Lss = Ls.block(i, k, rem_i, rem_k);
            microkernel_tail_lut[rem_k - 1](rem_i, colsA, W, Lss, Add, Ass,
                                            DownUpdate{Dd});
        });
    };

    for (index_t k = 0; k < N; ++k) {
        index_t rk = ranks(k, 0, 0);
        colsA += rk;
        if (colsA == 0)
            continue;
        auto Ad = A(k % 2).left_cols(colsA), Ld = LΨd_scalar()(k);
        auto As = A((k + 1) % 2).left_cols(colsA), Ls = LΨs_scalar()(k);
        Ad.right_cols(rk) = storage.Wupd(k).left_cols(rk);
        As.right_cols(rk) = storage.Vupd(k).left_cols(rk);
        W_t W;
        auto Dd            = D(0).top_rows(colsA);
        Dd.bottom_rows(rk) = storage.Σ_sgn(k).top_rows(rk);
        std::span<real_t> Dd_spn{Dd.data, static_cast<size_t>(Dd.rows)};

        // Process all diagonal blocks (in multiples of R, except the last).
        foreach_chunked(
            0, Ld.cols, R,
            [&](index_t k) {
                process_diag_block(k, R, Ad, Dd_spn, Ld, W);
                process_subdiag_block_W(k, Ad, Dd_spn, Ld, W);
                process_subdiag_block_V(k, R, As, Ls, Ad, Dd_spn, W);
            },
            [&](index_t k, index_t rem_k) {
                process_diag_block(k, rem_k, Ad, Dd_spn, Ld, W);
                process_subdiag_block_V(k, rem_k, As, Ls, Ad, Dd_spn, W);
            });
        if (k < N - 1)
            Ad.set_constant(0);
    }
    // Final stage has no subdiagonal block (V)
    index_t rN = ranks(N, 0, 0);
    colsA += rN;
    if (colsA > 0) {
        auto Ad = A(N % 2).left_cols(colsA), Ld = LΨd_scalar()(N);
        Ad.right_cols(rN) = storage.Wupd(N).left_cols(rN);
        W_t W;
        auto Dd            = D(0).top_rows(colsA);
        Dd.bottom_rows(rN) = storage.Σ_sgn(N).top_rows(rN);
        std::span<real_t> Dd_spn{Dd.data, static_cast<size_t>(Dd.rows)};

        // Process all diagonal blocks (in multiples of R, except the last).
        foreach_chunked(
            0, Ld.cols, R,
            [&](index_t k) {
                process_diag_block(k, R, Ad, Dd_spn, Ld, W);
                process_subdiag_block_W(k, Ad, Dd_spn, Ld, W);
            },
            [&](index_t k, index_t rem_k) {
                auto Add = Ad.middle_rows(k, rem_k);
                auto Ldd = Ld.block(k, k, rem_k, rem_k);
                microkernel_full_lut[rem_k - 1](colsA, Ldd, Add,
                                                DownUpdate{Dd_spn});
            });
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::updowndate(real_view Σ, bool_view J_old, bool_view J_new) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    auto &ranks                = storage.stagewise_update_counts;
    index_t rj_max             = 0;
    for (index_t k = 0; k <= N; ++k) {
        index_t rj  = 0;
        index_t nyk = k == N ? ny_N : ny;
        for (index_t r = 0; r < nyk; ++r) {
            bool up   = J_new(k, r, 0) && !J_old(k, r, 0);
            bool down = !J_new(k, r, 0) && J_old(k, r, 0);
            if (up || down) {
                storage.Σ_sgn(k, rj, 0) = up ? +real_t(0) : -real_t(0);
                storage.Σ_ud(k, rj, 0)  = Σ(k, r, 0);
                index_t nxuk            = k == N ? nx : nx + nu;
                for (index_t c = 0; c < nxuk; ++c) // TODO: pre-transpose CD?
                    storage.Z(k, c, rj) = CD()(k, r, c);
                ++rj;
            }
        }
        ranks(k, 0, 0) = rj;
        rj_max         = std::max(rj_max, rj);
    }
    if (rj_max == 0)
        return;

    // TODO: OpenMP
    foreach_chunked_merged(0, N + 1, simd_stride, [&](index_t k, auto nk) {
        index_t batch_idx = k / simd_stride;
        auto ranksj       = ranks.batch(batch_idx);
        auto rjmm         = std::minmax_element(ranksj.data, ranksj.data + nk);
        index_t rj_min = *rjmm.first, rj_batch = *rjmm.second;
        if (rj_batch == 0)
            return;
        auto Σj    = storage.Σ_ud.batch(batch_idx).top_rows(rj_batch);
        auto Sj    = storage.Σ_sgn.batch(batch_idx).top_rows(rj_batch);
        auto Zj    = storage.Z.batch(batch_idx).left_cols(rj_batch);
        auto Z1j   = storage.Z1.batch(batch_idx).left_cols(rj_batch);
        auto Luj   = storage.Lupd.batch(batch_idx).top_left(rj_batch, rj_batch);
        auto Wuj   = storage.Wupd.batch(batch_idx).left_cols(rj_batch);
        using simd = typename types::simd;
#ifdef _GLIBCXX_EXPERIMENTAL_SIMD // TODO: need general way to convert masks
        using index_simd = typename types::index_simd;
        index_simd rjks{ranksj.data, stdx::vector_aligned};
        for (index_t rj = rj_min; rj < rj_batch; ++rj) {
            index_simd rjs{rj};
            auto z0_mask = rjs > rjks;
            simd zero{0};
            constexpr auto vec_align = stdx::vector_aligned;
            for (index_t r = 0; r < nx + nu; ++r)
                where(z0_mask.__cvt(), zero).copy_to(&Zj(0, r, rj), vec_align);
            where(z0_mask.__cvt(), zero).copy_to(&Σj(0, rj, 0), vec_align);
        }
#else
#error "Fallback not yet implemented"
#endif
        const auto be = settings.preferred_backend;
        // Copy Z
        compact_blas::xcopy(Zj, Z1j.top_rows(nx + nu));
        compact_blas::xfill(real_t(0), Z1j.bottom_rows(nx));
        // Z ← L⁻¹ Z
        compact_blas::xtrsm_LLNN(LH().batch(batch_idx), Zj, be);
        // Lu ← ZᵀZ ± Σ⁻¹
        compact_blas::xsyrk_T(Zj, Luj, be);
        for (index_t r = 0; r < rj_batch; ++r) {
            simd Lujrr{&Luj(0, r, r), stdx::vector_aligned};
            simd Σjr{&Σj(0, r, 0), stdx::vector_aligned};
            simd Sjr{&Sj(0, r, 0), stdx::vector_aligned};
            Lujrr += cneg(1 / Σjr, Sjr);
            Lujrr.copy_to(&Luj(0, r, r), stdx::vector_aligned);
        }
        // Lu ← chol(ZᵀZ ± Σ⁻¹)
        compact_blas::xpntrf(Luj, Sj);
        // Z ← Z Lu⁻ᵀ
        compact_blas::xtrsm_RLTN(Luj, Zj, be);
        // Wu ← W Z
        compact_blas::xgemm_TN(Wᵀ().batch(batch_idx), Zj, Wuj, be);
        if (batch_idx < AB().num_batches()) {
            // Vu ← -V Z
            auto Vuj = storage.Vupd.batch(batch_idx).left_cols(rj_batch);
            compact_blas::xgemm_neg(V().batch(batch_idx), Zj, Vuj, be);
        }
        for (index_t r = 0; r < rj_batch; ++r) {
            simd s{&Sj(0, r, 0), stdx::vector_aligned};
            simd σ{&Σj(0, r, 0), stdx::vector_aligned};
            cneg(σ, s).copy_to(&Σj(0, r, 0), stdx::vector_aligned);
        }
        // Update LH and V
        compact_blas::xshhud_diag(LHV().batch(batch_idx), Z1j, Σj, be);
        auto LHi = LH().batch(batch_idx), Wi = Wᵀ().batch(batch_idx);
        compact_blas::xcopy(LHi.top_left(nx + nu, nx), Wi);
        compact_blas::xtrtri(Wi, be);
        compact_blas::xtrsm_LLNN(LHi.bottom_right(nu, nu), Wi.bottom_rows(nu),
                                 be);
        // TODO: should we always eagerly update V and W?
    });
    updowndate_ψ();
}

} // namespace koqkatoo::ocp
