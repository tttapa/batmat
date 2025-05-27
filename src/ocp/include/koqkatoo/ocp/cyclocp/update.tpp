#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/assume.hpp>
#include <koqkatoo/loop.hpp>

#include <numeric>

namespace koqkatoo::ocp::cyclocp {

template <index_t VL>
void CyclicOCPSolver<VL>::update_level(index_t l, index_t biY) {
    GUANAQO_TRACE("update_level", biY);
    const index_t offset = 1 << l;
    const index_t i      = biY >> (l + 1);
    const index_t j0     = biY == offset ? 0 : nJs[biY - 1 - offset],
                  j1 = nJs[biY - 1 + offset], nj = j1 - j0,
                  jsplit = nJs[biY - 1] - j0;
    constexpr index_t w3_out_lut[]{1, 0, 0, 1};
    const index_t w3_out = w3_out_lut[i & 3];
    if (i & 1) {
        compact_blas::xshhud_diag_cyclic(
            coupling_D.batch(biY), work_update.batch(l & 3).middle_cols(j0, nj),
            coupling_Y.batch(biY),
            work_update.batch((l + 2) % 4).middle_cols(j0, nj),
            work_update.batch((l + 2 + w3_out) % 4).middle_cols(j0, nj),
            coupling_U.batch(biY),
            work_update.batch((l + 1) % 4).middle_cols(j0, nj),
            work_update.batch((l + 1) % 4).middle_cols(j0, nj),
            work_update_Σ.batch(0).middle_rows(j0, nj), jsplit, 0);
    } else {
        const bool x_lanes = l + 1 == lP - lvl && 0; // TODO
        compact_blas::xshhud_diag_cyclic(
            coupling_D.batch(biY), work_update.batch(l & 3).middle_cols(j0, nj),
            coupling_Y.batch(biY),
            work_update.batch((l + 1) % 4).middle_cols(j0, nj),
            work_update.batch((l + 1) % 4).middle_cols(j0, nj),
            coupling_U.batch(biY),
            work_update.batch((l + 2) % 4).middle_cols(j0, nj),
            work_update.batch((l + 2 + w3_out) % 4).middle_cols(j0, nj),
            work_update_Σ.batch(0).middle_rows(j0, nj), jsplit,
            x_lanes ? 1 : 0);
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::update(matrix_view ΔΣ) {
    const index_t N = dim.N_horiz;
    KOQKATOO_ASSERT(((N >> lP) << lP) == N);
    koqkatoo::foreach_thread([this, ΔΣ](index_t ti, index_t P) {
        if (P < (1 << (lP - lvl)))
            throw std::logic_error("Incorrect number of threads");
        if (ti >= (1 << (lP - lvl)))
            return;
        update_riccati(ti, ΔΣ);
        for (index_t l = 0; l < lP - lvl; ++l) {
            barrier();

            const index_t offset = 1 << l;
            const auto biY       = sub_wrap_PmV(ti, offset);
            if (is_active(l, biY))
                update_level(l, biY);
        }
        barrier(); // TODO: remove and simply execute on the last thread
        const index_t l      = lP - lvl;
        const index_t offset = 1 << l;
        const auto biY       = sub_wrap_PmV(ti, offset);
        if (biY == 0) {
            GUANAQO_TRACE("update_level last", biY);
            const index_t j0 = 0, j1 = nJs.back(), nj = j1 - j0;
            compact_blas::xgemm_NT_add_diag_ref(
                work_update.batch(l & 3).middle_cols(j0, nj),
                work_update.batch((l + 2) & 3).middle_cols(j0, nj),
                coupling_Y.batch(0),
                work_update_Σ.batch(0).middle_rows(j0, nj));
            compact_blas::xshhud_diag_ref(
                coupling_D.batch(biY),
                work_update.batch((l + 2) & 3).middle_cols(j0, nj),
                work_update_Σ.batch(0).middle_rows(j0, nj));
            compact_blas::template xadd_copy<1>(
                work_update_Σ.batch(0).middle_rows(j0, nj),
                work_update_Σ.batch(0).middle_rows(j0, nj));
            compact_blas::template xadd_copy<1>( // TODO
                work_update.batch(l & 3).middle_cols(j0, nj),
                work_update.batch(l & 3).middle_cols(j0, nj));
            compact_blas::xshhud_diag_ref(
                coupling_D.batch(biY),
                work_update.batch(l & 3).middle_cols(j0, nj),
                work_update_Σ.batch(0).middle_rows(j0, nj));
            // TODO: we should actually merge these two xshhud calls to
            //       make sure that the intermediate matrix does not become
            //       indefinite (although this shouldn't be an issue for
            //       QPALM)
        }
    });
    this->alt = true;
}

template <index_t VL>
void CyclicOCPSolver<VL>::update_riccati(index_t ti, matrix_view Σ) {
    const auto [N, nx, nu, ny, ny_N] = dim;
    const index_t nyM                = std::max(ny, ny_N);
    const index_t num_stages         = N >> lP; // number of stages per thread
    const index_t di0                = ti * num_stages; // data batch index
    const index_t k0                 = ti * num_stages; // stage index
    const index_t nux                = nu + nx;
    const auto be                    = backend;
    auto R̂ŜQ̂                         = riccati_R̂ŜQ̂.batch(ti);
    auto B̂   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
    auto Â   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
    auto ΥΓ1 = riccati_ΥΓ1.batch(ti);
    auto ΥΓ2 = riccati_ΥΓ2.batch(ti);
    auto wΣ  = work_Σ.batch(ti);

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
            {
                GUANAQO_TRACE("Riccati update compress", k_next);
                auto DC_next = ΥΓ_next.block(0, nJi, nu + nx, nyM);
                nJ += compact_blas::compress_masks(data_DCᵀ.batch(di_next),
                                                   Σ.batch(di_next), DC_next,
                                                   wΣ.middle_rows(nJi, nyM));
                ΥΓ_next.block(nu + nx, nJi, nx, nJ - nJi).set_constant(0);
            }
            if (nJi > 0) {
                GUANAQO_TRACE("Riccati update Q", k);
                compact_blas::xgemm_NT_add_diag_ref(ΥΓi.bottom_rows(nx),
                                                    ΥΓi.middle_rows(nu, nx), Âi,
                                                    wΣ.top_rows(nJi));
                compact_blas::xshhud_diag_ref(Q̂i, ΥΓi.middle_rows(nu, nx),
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
            if (nJi > 0) {
                GUANAQO_TRACE("Riccati update Q", k);
                auto Q̂i_inv = R̂ŜQ̂i.block(nu - 1, nu, nx, nx);
                compact_blas::xshhud_diag_riccati(
                    Q̂i, ΥΓi.middle_rows(nu, nx), Âi, ΥΓi.bottom_rows(nx),
                    work_update.batch(wiA).middle_cols(j0, nJi), Q̂i_inv,
                    work_update.batch(wiI).middle_cols(j0, nJi),
                    wΣ.top_rows(nJi), ti == 0); // TODO: optimize
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

} // namespace koqkatoo::ocp::cyclocp
