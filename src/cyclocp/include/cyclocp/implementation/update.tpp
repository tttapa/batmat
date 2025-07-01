#include <cyclocp/cyclocp.hpp>

#include <batmat/assume.hpp>
#include <batmat/linalg/compress.hpp>
#include <batmat/linalg/gemm-diag.hpp>
#include <batmat/linalg/gemm.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/loop.hpp>

#include <numeric>

namespace cyclocp::ocp::cyclocp {
using namespace batmat::linalg;

template <index_t VL, class T>
void CyclicOCPSolver<VL, T>::update_level(index_t l, index_t biY) {
    GUANAQO_TRACE("update_level", biY);
    const index_t offset = 1 << l;
    const index_t i      = biY >> (l + 1);
    const index_t j0 = biY == offset ? 0 : nJs[biY - 1 - offset], j1 = nJs[biY - 1 + offset],
                  nj = j1 - j0, jsplit = nJs[biY - 1] - j0;
    constexpr index_t w3_out_lut[]{1, 0, 0, 1};
    const index_t w3_out = w3_out_lut[i & 3];
    if (i & 1) {
        compact_blas::xshhud_diag_cyclic(
            simdify(coupling_D.batch(biY)), simdify(work_update.batch(l & 3).middle_cols(j0, nj)),
            simdify(coupling_Y.batch(biY)),
            simdify(work_update.batch((l + 2) % 4).middle_cols(j0, nj)),
            simdify(work_update.batch((l + 2 + w3_out) % 4).middle_cols(j0, nj)),
            simdify(coupling_U.batch(biY)),
            simdify(work_update.batch((l + 1) % 4).middle_cols(j0, nj)),
            simdify(work_update.batch((l + 1) % 4).middle_cols(j0, nj)),
            simdify(work_update_Σ.batch(0).middle_rows(j0, nj)), jsplit, 0);
    } else {
        const bool x_lanes = l + 1 == lP - lvl && 0; // TODO
        compact_blas::xshhud_diag_cyclic(
            simdify(coupling_D.batch(biY)), simdify(work_update.batch(l & 3).middle_cols(j0, nj)),
            simdify(coupling_Y.batch(biY)),
            simdify(work_update.batch((l + 1) % 4).middle_cols(j0, nj)),
            simdify(work_update.batch((l + 1) % 4).middle_cols(j0, nj)),
            simdify(coupling_U.batch(biY)),
            simdify(work_update.batch((l + 2) % 4).middle_cols(j0, nj)),
            simdify(work_update.batch((l + 2 + w3_out) % 4).middle_cols(j0, nj)),
            simdify(work_update_Σ.batch(0).middle_rows(j0, nj)), jsplit, x_lanes ? 1 : 0);
    }
}

template <index_t VL, class T>
void CyclicOCPSolver<VL, T>::update(view<> ΔΣ) {
    const index_t P = 1 << (lP - lvl);
    batmat::foreach_thread(P, [this, ΔΣ](index_t ti, index_t) {
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
            gemm_diag_add(work_update.batch(l & 3).middle_cols(j0, nj),
                          work_update.batch((l + 2) & 3).middle_cols(j0, nj).transposed(),
                          coupling_Y.batch(0), work_update_Σ.batch(0).middle_rows(j0, nj));
            compact_blas::xshhud_diag_ref(
                simdify(coupling_D.batch(biY)),
                simdify(work_update.batch((l + 2) & 3).middle_cols(j0, nj)),
                simdify(work_update_Σ.batch(0).middle_rows(j0, nj)));
            compact_blas::template xadd_copy<1>(
                simdify(work_update_Σ.batch(0).middle_rows(j0, nj)),
                simdify(work_update_Σ.batch(0).middle_rows(j0, nj)));
            compact_blas::template xadd_copy<1>( // TODO
                simdify(work_update.batch(l & 3).middle_cols(j0, nj)),
                simdify(work_update.batch(l & 3).middle_cols(j0, nj)));
            compact_blas::xshhud_diag_ref(simdify(coupling_D.batch(biY)),
                                          simdify(work_update.batch(l & 3).middle_cols(j0, nj)),
                                          simdify(work_update_Σ.batch(0).middle_rows(j0, nj)));
            // TODO: we should actually merge these two xshhud calls to
            //       make sure that the intermediate matrix does not become
            //       indefinite (although this shouldn't be an issue for
            //       QPALM)
        }
    });
    this->alt = true;
}

template <index_t VL, class T>
void CyclicOCPSolver<VL, T>::update_riccati(index_t ti, view<> Σ) {
    using abi                = simdified_abi_t<decltype(Σ.batch(0))>;
    const index_t nyM        = std::max(ny, ny_0 + ny_N);
    const index_t num_stages = ceil_N >> lP;    // number of stages per thread
    const index_t di0        = ti * num_stages; // data batch index
    const index_t k0         = ti * num_stages; // stage index
    const index_t nux        = nu + nx;
    auto R̂ŜQ̂                 = riccati_R̂ŜQ̂.batch(ti);
    auto B̂                   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
    auto Â                   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
    auto ΥΓ1                 = riccati_ΥΓ1.batch(ti);
    auto ΥΓ2                 = riccati_ΥΓ2.batch(ti);
    auto wΣ                  = work_Σ.batch(ti);

    index_t nJ;
    {
        GUANAQO_TRACE("Riccati update compress", k0);
        auto DC0 = ΥΓ2.top_left(nu + nx, nyM);
        nJ = compress_masks<abi>(simdify(data_DCᵀ.batch(di0)), simdify(Σ.batch(di0)), simdify(DC0),
                                 simdify(wΣ.top_rows(nyM)));
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
            compact_blas::xshhud_diag_2_ref(simdify(R̂Ŝi), simdify(ΥΓi.top_rows(nu + nx)),
                                            simdify(B̂i), simdify(ΥΓi.bottom_rows(nx)),
                                            simdify(wΣ.top_rows(nJi)));
        }
        if (i + 1 < num_stages) {
            [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
            const auto di_next                 = di0 + i + 1;
            auto ΥΓ_next                       = ((i & 1) ? ΥΓ2 : ΥΓ1).left_cols(nJi + nyM);
            if (nJi > 0) {
                GUANAQO_TRACE("Riccati update prop", k_next);
                gemm(data_BA.batch(di_next).transposed(), ΥΓi.middle_rows(nu, nx),
                     ΥΓ_next.top_left(nu + nx, nJi));
                copy(ΥΓi.bottom_rows(nx), ΥΓ_next.bottom_left(nx, nJi));
            }
            {
                GUANAQO_TRACE("Riccati update compress", k_next);
                auto DC_next = ΥΓ_next.block(0, nJi, nu + nx, nyM);
                nJ +=
                    compress_masks<abi>(simdify(data_DCᵀ.batch(di_next)), simdify(Σ.batch(di_next)),
                                        simdify(DC_next), simdify(wΣ.middle_rows(nJi, nyM)));
                ΥΓ_next.block(nu + nx, nJi, nx, nJ - nJi).set_constant(0);
            }
            if (nJi > 0) {
                GUANAQO_TRACE("Riccati update Q", k);
                gemm_diag_add(ΥΓi.bottom_rows(nx), ΥΓi.middle_rows(nu, nx).transposed(), Âi,
                              wΣ.top_rows(nJi));
                compact_blas::xshhud_diag_ref(simdify(Q̂i), simdify(ΥΓi.middle_rows(nu, nx)),
                                              simdify(wΣ.top_rows(nJi)));
            }
        } else {
            const auto bi_upd = sub_wrap_PmV(ti, 1);
            nJs[bi_upd]       = nJi;
            barrier();
            if (ti == 0)
                std::inclusive_scan(begin(nJs), end(nJs), begin(nJs));
            barrier();
            const index_t j0 = bi_upd == 0 ? 0 : nJs[bi_upd - 1], j1 = nJs[bi_upd];
            assert(nJi == j1 - j0);
            constexpr index_t wiA_table[]{0, 1, 0, 2};
            constexpr index_t wiI_table[]{2, 0, 1, 0};
            const index_t wiA = wiA_table[bi_upd & 3];
            const index_t wiI = wiI_table[bi_upd & 3];
            if (nJi > 0) {
                GUANAQO_TRACE("Riccati update Q", k);
                auto Q̂i_inv = R̂ŜQ̂i.block(nu - 1, nu, nx, nx);
                compact_blas::xshhud_diag_riccati(
                    simdify(Q̂i), simdify(ΥΓi.middle_rows(nu, nx)), simdify(Âi),
                    simdify(ΥΓi.bottom_rows(nx)),
                    simdify(work_update.batch(wiA).middle_cols(j0, nJi)), simdify(Q̂i_inv),
                    simdify(work_update.batch(wiI).middle_cols(j0, nJi)), simdify(wΣ.top_rows(nJi)),
                    ti == 0); // TODO: optimize
                compact_blas::xneg(simdify(work_update.batch(wiI).middle_cols(j0, nJi))); // TODO
                ti == 0 ? compact_blas::template xadd_neg_copy<-1>(
                              simdify(work_update_Σ.batch(0).middle_rows(j0, nJi)),
                              simdify(wΣ.top_rows(nJi)))
                        : compact_blas::xadd_neg_copy(
                              simdify(work_update_Σ.batch(0).middle_rows(j0, nJi)),
                              simdify(wΣ.top_rows(nJi)));
            }
        }
    }
}

} // namespace cyclocp::ocp::cyclocp
