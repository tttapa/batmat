#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/assume.hpp>
#include <koqkatoo/loop.hpp>

namespace koqkatoo::ocp::cyclocp {

template <index_t VL>
void CyclicOCPSolver<VL>::factor_schur_Y(index_t l, index_t biY) {
    const index_t offset = 1 << l;
    { // Compute Y[bi]
        GUANAQO_TRACE("Trsm Y", biY);
        compact_blas::xtrsm_RLTN(coupling_D.batch(biY), coupling_Y.batch(biY),
                                 backend);
    }
    // Wait for U[bi] from factor_schur_U
    barrier();
    for (index_t c = 0; c < coupling_U.cols(); c += 1)
        for (index_t r = 0; r < coupling_U.rows(); r += 16)
            __builtin_prefetch(&coupling_U.batch(biY)(0, r, c), 0, 3);
    // Compute UYᵀ or YUᵀ
    if (is_U_below_Y(l, biY)) {
        const index_t bi_next = add_wrap_PmV(biY, offset); // TODO: need mod?
        GUANAQO_TRACE("Compute U", bi_next);
        compact_blas::xgemm_NT_neg(coupling_U.batch(biY), coupling_Y.batch(biY),
                                   coupling_U.batch(bi_next), backend);
    } else {
        const index_t bi_prev = sub_wrap_PmV(biY, offset); // TODO: need mod?
        GUANAQO_TRACE("Compute Y", bi_prev);
        compact_blas::xgemm_NT_neg(coupling_Y.batch(biY), coupling_U.batch(biY),
                                   coupling_Y.batch(bi_prev), backend);
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::factor_schur_U(index_t l, index_t biU) {
    const index_t offset = 1 << l;
    const index_t biD    = sub_wrap_PmV(biU, offset);
    const index_t biY    = sub_wrap_PmV(biD, offset);
    for (index_t c = 0; c < coupling_D.cols(); c += 1)
        for (index_t r = 0; r < coupling_D.rows(); r += 16)
            __builtin_prefetch(&coupling_D.batch(biD)(0, r, c), 0, 3);
    { // Compute U[bi]
        GUANAQO_TRACE("Trsm U", biU);
        compact_blas::xtrsm_RLTN(coupling_D.batch(biU), coupling_U.batch(biU),
                                 backend);
    }
    // Wait for Y[bi] from factor_schur_Y
    barrier();
    for (index_t c = 0; c < coupling_Y.cols(); c += 1)
        for (index_t r = 0; r < coupling_Y.rows(); r += 16)
            __builtin_prefetch(&coupling_Y.batch(biY)(0, r, c), 0, 3);
    { // D -= UUᵀ
        GUANAQO_TRACE("Subtract UUᵀ", biD);
        compact_blas::xsyrk_sub(coupling_U.batch(biU), coupling_D.batch(biD),
                                backend);
    }
    { // D -= YYᵀ
        GUANAQO_TRACE("Subtract YYᵀ", biD);
        biD == 0 ? compact_blas::xsyrk_sub_shift(coupling_Y.batch(biY),
                                                 coupling_D.batch(biD))
                 : compact_blas::xsyrk_sub(coupling_Y.batch(biY),
                                           coupling_D.batch(biD), backend);
    }
    // chol(D)
    if (is_active(l + 1, biD) || (l + 1 == lP - lvl && biD == 0)) {
        GUANAQO_TRACE("Factor D", biD);
        compact_blas::xpotrf(coupling_D.batch(biD), backend);
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::factor_l0(const index_t ti) {
    const index_t num_stages = N_horiz >> lP; // number of stages per thread
    const index_t biI        = sub_wrap_PmV(ti, 1);
    const index_t biA        = ti;
    const auto be            = backend;
    const auto biR           = biA;
    const bool x_lanes       = biA == 0; // first stage wraps around
    // Coupling equation to previous stage is eliminated after coupling
    // equation to next stage for odd threads, vice versa for even threads.
    const bool I_below_A = (biA & 1) == 1;
    // Update the subdiagonal blocks U and Y of the coupling equations
    auto DiI = coupling_D.batch(biI);
    auto DiA = coupling_D.batch(biA);
    auto Âi  = riccati_ÂB̂.batch(biR).middle_cols(nx * (num_stages - 1), nx);
    auto ÂB̂i = riccati_ÂB̂.batch(biR).right_cols(nx + nu * num_stages);
    auto R̂ŜQ̂ = riccati_R̂ŜQ̂.batch(biR);
    auto Q̂i  = R̂ŜQ̂.bottom_right(nx, nx);
    // LQ⁻ᵀ is upper triangular, stored one row up from LQ itself
    assert(nu >= 1);
    auto Q̂i_inv = R̂ŜQ̂.right_cols(nx).middle_rows(nu - 1, nx);
    {
        GUANAQO_TRACE("Invert Q", biI);
        compact_blas::xtrtri_T_copy_ref(Q̂i, Q̂i_inv);
    }
    if (I_below_A) {
        // Top block is A → column index is row index of A (biA)
        // Target block in cyclic part is U in column λ(kA)
        GUANAQO_TRACE("Compute first U", biA);
        compact_blas::xtrmm_LUNN_T_neg_ref(Q̂i_inv, Âi, coupling_U.batch(biA));
    } else {
        // Top block is I → column index is row index of I (biI)
        // Target block in cyclic part is Y in column λ(kI)
        GUANAQO_TRACE("Compute first Y", biI);
        x_lanes ? compact_blas::xtrmm_RUTN_neg_shift(Âi, Q̂i_inv,
                                                     coupling_Y.batch(biI))
                : compact_blas::xtrmm_RUTN_neg_ref(Âi, Q̂i_inv,
                                                   coupling_Y.batch(biI));
    }
    // Each column of the cyclic part with coupling equations is updated by
    // two threads: one for the forward, and one for the backward coupling.
    // Update the diagonal blocks of the coupling equations,
    // first forward in time ...
    {
        GUANAQO_TRACE("Compute L⁻ᵀL⁻¹", biI);
        x_lanes ? compact_blas::xtrtrsyrk_UL_shift(Q̂i_inv, DiI)
                : compact_blas::xtrtrsyrk_UL(Q̂i_inv, DiI);
    }
    // Then synchronize to make sure there are no two threads updating the
    // same diagonal block.
    barrier();
    // And finally backward in time, optionally merged with factorization.
    const bool do_factor = (biA & 1) == 1 || (lP - lvl == 0 && biA == 0);
    {
        GUANAQO_TRACE("Compute (BA)(BA)ᵀ", biA);
        compact_blas::xsyrk_add(ÂB̂i, DiA, be);
    }
    if (do_factor) {
        GUANAQO_TRACE("Factor D", biA);
        compact_blas::xpotrf(coupling_D.batch(biA), backend);
    }
}

// Performs Riccati recursion and then factors level l=0 of
// coupling equations + propagates the subdiagonal blocks to level l=1.
template <index_t VL>
void CyclicOCPSolver<VL>::factor_riccati(index_t ti, bool alt, real_t S,
                                         matrix_view Σ) {
    const index_t num_stages = N_horiz >> lP;   // number of stages per thread
    const index_t di0        = ti * num_stages; // data batch index
    const index_t k0         = ti * num_stages; // stage index
    const index_t nux        = nu + nx;
    const auto be            = backend;
    auto R̂ŜQ̂                 = riccati_R̂ŜQ̂.batch(ti);
    auto B̂                   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
    auto Â                   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
    auto BAᵀ                 = riccati_BAᵀ.batch(ti);
    auto A0                  = data_BA.batch(di0).right_cols(nx);
    // Copy B and A from the last stage
    {
        GUANAQO_TRACE("Riccati init", k0);
        compact_blas::xcopy(data_BA.batch(di0).left_cols(nu), B̂.left_cols(nu));
        compact_blas::xsyrk_schur_copy(data_DCᵀ.batch(di0), Σ.batch(di0),
                                       data_RSQ.batch(di0), R̂ŜQ̂.left_cols(nux));
    }
    for (index_t i = 0; i < num_stages; ++i) {
        index_t k = sub_wrap_N(k0, i);
        auto R̂ŜQ̂i = R̂ŜQ̂.middle_cols(i * nux, nux);
        auto R̂Ŝi  = R̂ŜQ̂i.left_cols(nu);
        auto R̂i   = R̂Ŝi.top_rows(nu);
        auto Ŝi   = R̂Ŝi.bottom_rows(nx);
        auto Q̂i   = R̂ŜQ̂i.bottom_right(nx, nx);
        auto B̂i   = B̂.middle_cols(i * nu, nu);
        auto Âi   = Â.middle_cols(i * nx, nx);
        {
            GUANAQO_TRACE("Riccati QRS", k);
            using std::isfinite;
            if (isfinite(S))
                R̂i.add_to_diagonal(1 / S);
            // Factor R̂, update Ŝ, and compute LB̂ = B̂ LR̂⁻ᵀ
            compact_blas::xpotrf(R̂Ŝi, be);        // ┐
            compact_blas::xtrsm_RLTN(R̂i, B̂i, be); // ┘
            // Update Â = Ã - LB̂ LŜᵀ
            i == 0 ? compact_blas::xgemm_NT_sub_copy_ref(B̂i, Ŝi, A0, Âi)
                   : compact_blas::xgemm_NT_sub(B̂i, Ŝi, Âi, be);
            if (isfinite(S))
                Q̂i.add_to_diagonal(1 / S);
            // Update and factor Q̂ = Q̃ - LŜ LŜᵀ
            compact_blas::xsyrk_sub(Ŝi, Q̂i, be); // ┐
            compact_blas::xpotrf(Q̂i, be);        // ┘
        }
        if (i + 1 < num_stages) {
            // Copy next B and A
            [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
            GUANAQO_TRACE("Riccati update AB", k_next);
            const auto di_next = di0 + i + 1;
            auto BAᵀi          = BAᵀ.middle_cols(alt ? 0 : i * nx, nx);
            auto BAi           = data_BA.batch(di_next);
            auto Bi = BAi.left_cols(nu), Ai = BAi.right_cols(nx);
            // Compute next B̂ and Â
            auto B̂_next = B̂.middle_cols((i + 1) * nu, nu);
            auto Â_next = Â.middle_cols((i + 1) * nx, nx);
            compact_blas::xgemm(Âi, Bi, B̂_next, be);
            compact_blas::xgemm(Âi, Ai, Â_next, be);
            // Riccati update
            auto R̂ŜQ̂_next = R̂ŜQ̂.middle_cols((i + 1) * nux, nux);
            compact_blas::xtrmm_RLNN_T_ref(data_BA.batch(di_next), Q̂i, BAᵀi);
#if 1
            compact_blas::xsyrk_add_copy(BAᵀi, data_RSQ.batch(di_next),
                                         R̂ŜQ̂_next);
            compact_blas::xsyrk_schur_copy(data_DCᵀ.batch(di_next),
                                           Σ.batch(di_next), R̂ŜQ̂_next,
                                           R̂ŜQ̂_next); // TODO: non-copy variant
#else
            compact_blas::xsyrk_schur_copy(data_DCᵀ.batch(di_next),
                                           Σ.batch(di_next),
                                           data_RSQ.batch(di_next),
                                           R̂ŜQ̂_next);    // ┐
            compact_blas::xsyrk_add(BAᵀi, R̂ŜQ̂_next, be); // ┘
#endif
        } else {
            // Compute LÂ = Ã LQ⁻ᵀ
            GUANAQO_TRACE("Riccati last", k);
            compact_blas::xtrsm_RLTN(Q̂i, Âi, be);
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::factor(real_t S, matrix_view Σ, bool alt) {
    this->alt = alt;
    KOQKATOO_ASSERT(((N_horiz >> lP) << lP) == N_horiz);
    const index_t P = 1 << (lP - lvl);
    koqkatoo::foreach_thread(P, [this, alt, S, Σ](index_t ti, index_t) {
        factor_riccati(ti, alt, S, Σ);
        factor_l0(ti);
        for (index_t l = 0; l < lP - lvl; ++l) {
            barrier();
            const index_t offset = 1 << l;
            const auto biY       = sub_wrap_PmV(ti, offset);
            const auto biU       = ti;
            if (is_active(l, biY))
                factor_schur_Y(l, biY);
            else if (is_active(l, biU))
                factor_schur_U(l, biU);
            else
                barrier();
        }
    });
}

} // namespace koqkatoo::ocp::cyclocp
