#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/assume.hpp>
#include <koqkatoo/loop.hpp>

namespace koqkatoo::ocp::cyclocp {

template <index_t VL>
void CyclicOCPSolver<VL>::solve_active(
    [[maybe_unused]] index_t l, [[maybe_unused]] index_t biY,
    [[maybe_unused]] mut_matrix_view λ) const {
    // TODO: nothing?
}

template <index_t VL>
void CyclicOCPSolver<VL>::solve_active_secondary(index_t l, index_t biU,
                                                 mut_matrix_view λ) const {
    const index_t num_stages = N_horiz >> lP;
    const index_t offset     = 1 << l;
    const index_t biD        = sub_wrap_PmV(biU, offset);
    const index_t biY        = sub_wrap_PmV(biD, offset);
    const index_t diU        = biU * num_stages;
    const index_t diD        = biD * num_stages;
    const index_t diY        = biY * num_stages;
    { // b[diD] -= U[biU] b[diU]
        GUANAQO_TRACE("Subtract Ub", biD);
        compact_blas::xgemv_sub(coupling_U.batch(biU), λ.batch(diU),
                                λ.batch(diD), backend);
    }
    { // b[diD] -= Y[biY] b[diY]
        GUANAQO_TRACE("Subtract Yb", biD);
        biD == 0 ? compact_blas::xgemv_sub_shift(coupling_Y.batch(biY),
                                                 λ.batch(diY), λ.batch(diD))
                 : compact_blas::xgemv_sub(coupling_Y.batch(biY), λ.batch(diY),
                                           λ.batch(diD), backend);
    }
    // solve D⁻¹[diD] d[diD]
    if (is_active(l + 1, biD)) {
        GUANAQO_TRACE("Solve b", biD);
        compact_blas::xtrsv_LNN(coupling_D.batch(biD), λ.batch(diD), backend);
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::solve_riccati_forward(index_t ti, mut_matrix_view ux,
                                                mut_matrix_view λ) const {
    const index_t num_stages = N_horiz >> lP;   // number of stages per thread
    const index_t di0        = ti * num_stages; // data batch index
    const index_t biI        = sub_wrap_PmV(ti, 1);
    const index_t diI        = biI * num_stages;
    const index_t di_last    = di0 + num_stages - 1;
    const index_t k0         = ti * num_stages; // stage index
    const index_t nux        = nu + nx;
    const auto be            = backend;
    auto R̂ŜQ̂                 = riccati_R̂ŜQ̂.batch(ti);
    auto B̂                   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
    auto Â                   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
    auto BAᵀ                 = riccati_BAᵀ.batch(ti);
    for (index_t i = 0; i < num_stages; ++i) {
        index_t k  = sub_wrap_N(k0, i);
        index_t di = di0 + i;
        auto R̂ŜQ̂i  = R̂ŜQ̂.middle_cols(i * nux, nux);
        auto Q̂i    = R̂ŜQ̂i.bottom_right(nx, nx);
        auto B̂i    = B̂.middle_cols(i * nu, nu);
        auto Âi    = Â.middle_cols(i * nx, nx);
        {
            GUANAQO_TRACE("Riccati solve QRS", k);
            // l = LR⁻¹ r, q = LQ⁻¹(q - LS l)
            compact_blas::xtrsv_LNN(R̂ŜQ̂i, ux.batch(di), be);
            // λ0 -= LB̂ l
            compact_blas::xgemv_sub(B̂i, ux.batch(di).top_rows(nu), λ.batch(di0),
                                    be);
        }
        if (i + 1 < num_stages) {
            [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
            const auto di_next                 = di + 1;
            auto BAᵀi                          = BAᵀ.middle_cols(i * nx, nx);
            GUANAQO_TRACE("Riccati solve b", k_next);
            // λ0 += Â λ
            compact_blas::xgemv_add(Âi, λ.batch(di_next), λ.batch(di0), be);
            // b = LQᵀb + q
            compact_blas::xtrmv_T(Q̂i, λ.batch(di_next), be);
            compact_blas::xadd_copy(λ.batch(di_next), λ.batch(di_next),
                                    ux.batch(di).bottom_rows(nx));
            // l += LB λ, q += LA λ
            compact_blas::xgemv_add(BAᵀi, λ.batch(di_next), ux.batch(di_next),
                                    be);
        } else {
            GUANAQO_TRACE("Riccati last", k);
            // λ0 -= Â λ
            compact_blas::xgemv_sub(Âi, ux.batch(di).bottom_rows(nx),
                                    λ.batch(di0), be);
        }
    }
    barrier();
    GUANAQO_TRACE("Riccati coupling I", k0);
    // b = LQ⁻ᵀ x + b
    const bool x_lanes = ti == 0; // first stage wraps around
    auto x_last        = ux.batch(di_last).bottom_rows(nx);
    auto λI            = λ.batch(diI);
    compact_blas::xtrsv_LTN(R̂ŜQ̂.right_cols(nx).bottom_rows(nx), x_last, be);
    x_lanes ? compact_blas::template xadd_copy<-1>(λI, x_last, λI)
            : compact_blas::xadd_copy(λI, x_last, λI);
    compact_blas::xneg(λI); // TODO: merge
    if (is_active(0, biI))
        compact_blas::xtrsv_LNN(coupling_D.batch(biI), λI, be);
}

template <index_t VL>
void CyclicOCPSolver<VL>::solve_riccati_forward_alt(
    index_t ti, mut_matrix_view ux, mut_matrix_view λ,
    mut_matrix_view work) const {
    const index_t num_stages = N_horiz >> lP;   // number of stages per thread
    const index_t di0        = ti * num_stages; // data batch index
    const index_t biI        = sub_wrap_PmV(ti, 1);
    const index_t diI        = biI * num_stages;
    const index_t di_last    = di0 + num_stages - 1;
    const index_t k0         = ti * num_stages; // stage index
    const index_t nux        = nu + nx;
    const auto be            = backend;
    auto R̂ŜQ̂                 = riccati_R̂ŜQ̂.batch(ti);
    auto B̂                   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
    auto Â                   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
    auto w                   = work.batch(ti);
    for (index_t i = 0; i < num_stages; ++i) {
        index_t k  = sub_wrap_N(k0, i);
        index_t di = di0 + i;
        auto R̂ŜQ̂i  = R̂ŜQ̂.middle_cols(i * nux, nux);
        auto R̂i    = R̂ŜQ̂i.top_left(nu, nu);
        auto Ŝi    = R̂ŜQ̂i.bottom_left(nx, nu);
        auto Q̂i    = R̂ŜQ̂i.bottom_right(nx, nx);
        auto B̂i    = B̂.middle_cols(i * nu, nu);
        auto Âi    = Â.middle_cols(i * nx, nx);
        {
            GUANAQO_TRACE("Riccati solve ux", k);
            // l = LR⁻¹ r
            compact_blas::xtrsv_LNN(R̂i, ux.batch(di).top_rows(nu), be);
            // p = q - LS l
            compact_blas::xgemv_sub(Ŝi, ux.batch(di).top_rows(nu),
                                    ux.batch(di).bottom_rows(nx), be);
            // λ0 -= LB̂ l
            compact_blas::xgemv_sub(B̂i, ux.batch(di).top_rows(nu), λ.batch(di0),
                                    be);
        }
        if (i + 1 < num_stages) {
            [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
            const auto di_next                 = di + 1;
            GUANAQO_TRACE("Riccati solve b", k_next);
            // λ0 += Â b
            compact_blas::xgemv_add(Âi, λ.batch(di_next), λ.batch(di0), be);
            // b' = LQᵀb
            compact_blas::xcopy(λ.batch(di_next), w);
            compact_blas::xtrmv_T(Q̂i, w, be);
            // d' = LQ b'
            compact_blas::xtrmv(Q̂i, w, be);
            // d = LQ LQᵀ b + p
            compact_blas::xadd_copy(w, w, ux.batch(di).bottom_rows(nx));
            // l += Bᵀd, q += Aᵀd
            compact_blas::xgemv_T_add(data_BA.batch(di_next), w,
                                      ux.batch(di_next), be);
        } else {
            GUANAQO_TRACE("Riccati solve last", k);
            // q = LQ⁻¹ p
            compact_blas::xtrsv_LNN(Q̂i, ux.batch(di).bottom_rows(nx), be);
            // λ0 -= LÂ q
            compact_blas::xgemv_sub(Âi, ux.batch(di).bottom_rows(nx),
                                    λ.batch(di0), be);
        }
    }
    barrier();
    GUANAQO_TRACE("Riccati coupling I", k0);
    // b = LQ⁻ᵀ x + b
    const bool x_lanes = ti == 0; // first stage wraps around
    auto x_last        = ux.batch(di_last).bottom_rows(nx);
    auto λI            = λ.batch(diI);
    compact_blas::xtrsv_LTN(R̂ŜQ̂.right_cols(nx).bottom_rows(nx), x_last, be);
    // Note: this leaves LQ⁻ᵀ q in x_last,
    //       which is reused during the backward solve
    x_lanes ? compact_blas::template xadd_copy<-1>(λI, x_last, λI)
            : compact_blas::xadd_copy(λI, x_last, λI);
    compact_blas::xneg(λI); // TODO: merge
    if (is_active(0, biI))
        compact_blas::xtrsv_LNN(coupling_D.batch(biI), λI, be);
}

template <index_t VL>
void CyclicOCPSolver<VL>::solve_forward(mut_matrix_view ux, mut_matrix_view λ,
                                        mut_matrix_view work) const {
    KOQKATOO_ASSERT(((N_horiz >> lP) << lP) == N_horiz);
    const index_t P = 1 << (lP - lvl);
    koqkatoo::foreach_thread(P, [this, &ux, &λ, &work](index_t ti, index_t) {
        alt ? solve_riccati_forward_alt(ti, ux, λ, work)
            : solve_riccati_forward(ti, ux, λ);
        for (index_t l = 0; l < lP - lvl; ++l) {
            barrier();
            const index_t offset = 1 << l;
            const auto biY       = sub_wrap_PmV(ti, offset);
            const auto biU       = ti;
            if (is_active(l, biY))
                solve_active(l, biY, λ);
            else if (is_active(l, biU))
                solve_active_secondary(l, biU, λ);
        }
    });
}

template <index_t VL>
void CyclicOCPSolver<VL>::solve_reverse_active(index_t l, index_t bi,
                                               mut_matrix_view λ) const {
    const index_t offset     = 1 << l;
    const index_t num_stages = N_horiz >> lP;
    const index_t biY        = add_wrap_PmV(bi, offset);
    const index_t biU        = sub_wrap_PmV(bi, offset);
    const index_t di         = bi * num_stages;
    const index_t diY        = biY * num_stages;
    const index_t diU        = biU * num_stages;
    const bool x_lanes       = diY == 0;
    GUANAQO_TRACE("Solve coupling reverse", bi);
    x_lanes ? compact_blas::xgemv_T_sub_shift(coupling_Y.batch(bi),
                                              λ.batch(diY), λ.batch(di))
            : compact_blas::xgemv_T_sub(coupling_Y.batch(bi), λ.batch(diY),
                                        λ.batch(di), backend);
    compact_blas::xgemv_T_sub(coupling_U.batch(bi), λ.batch(diU), λ.batch(di),
                              backend);
    compact_blas::xtrsv_LTN(coupling_D.batch(bi), λ.batch(di), backend);
}

template <index_t VL>
void CyclicOCPSolver<VL>::solve_riccati_reverse(index_t ti, mut_matrix_view ux,
                                                mut_matrix_view λ,
                                                mut_matrix_view work) const {
    const index_t num_stages = N_horiz >> lP;   // number of stages per thread
    const index_t di0        = ti * num_stages; // data batch index
    const index_t biI        = sub_wrap_PmV(ti, 1);
    const index_t diI        = biI * num_stages;
    const index_t k0         = ti * num_stages; // stage index
    const index_t nux        = nu + nx;
    const auto be            = backend;
    auto R̂ŜQ̂                 = riccati_R̂ŜQ̂.batch(ti);
    auto B̂                   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
    auto Â                   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
    auto BAᵀ                 = riccati_BAᵀ.batch(ti);

    for (index_t i = num_stages; i-- > 0;) {
        index_t k  = sub_wrap_N(k0, i);
        index_t di = di0 + i;
        auto R̂ŜQ̂i  = R̂ŜQ̂.middle_cols(i * nux, nux);
        auto Q̂i    = R̂ŜQ̂i.bottom_right(nx, nx);
        auto R̂i    = R̂ŜQ̂i.top_left(nu, nu);
        auto Ŝi    = R̂ŜQ̂i.bottom_left(nx, nu);
        auto B̂i    = B̂.middle_cols(i * nu, nu);
        auto Âi    = Â.middle_cols(i * nx, nx);
        if (i + 1 < num_stages) {
            [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
            const auto di_next                 = di + 1;
            GUANAQO_TRACE("Riccati solve b", k_next);
            auto BAᵀi = BAᵀ.middle_cols(i * nx, nx);
            // b -= LBᵀ u + LAᵀ x
            compact_blas::xgemv_T_sub(BAᵀi, ux.batch(di_next), λ.batch(di_next),
                                      be);
            compact_blas::xneg(λ.batch(di_next)); // TODO
            // q -= b
            compact_blas::xadd_copy(ux.batch(di).bottom_rows(nx),
                                    ux.batch(di).bottom_rows(nx),
                                    λ.batch(di_next));
            compact_blas::xtrmv(Q̂i, λ.batch(di_next), backend);
            compact_blas::xgemv_T_add(Âi, λ.batch(di0), λ.batch(di_next),
                                      backend);
        } else {
            // x_last = LQ⁻ᵀ(q_last + LQ⁻¹ λ - LÂᵀ λ)
            GUANAQO_TRACE("Riccati last", k);
            // λ0 -= Â λ
            const auto x_last  = ux.batch(di).bottom_rows(nx);
            const bool x_lanes = ti == 0;
            const auto w       = work.batch(ti);
            x_lanes ? compact_blas::template xadd_copy<1>(w, λ.batch(diI))
                          : compact_blas::template xadd_copy<0>(w, λ.batch(diI));
            // LQ⁻¹ λ
            compact_blas::xtrsv_LNN(Q̂i, w, backend);
            // LQ⁻¹ λ - LÂᵀ λ
            compact_blas::xgemv_T_sub(Âi, λ.batch(di0), w, be);
            // x_last = LQ⁻ᵀ(LQ⁻¹ λ - LÂᵀ λ)
            compact_blas::xtrsv_LTN(Q̂i, w, backend);
            compact_blas::xadd_copy(x_last, x_last, w);
        }
        if (i + 1 < num_stages) {
            GUANAQO_TRACE("Riccati solve QRS", k);
            // l -= LB̂ᵀ λ0
            compact_blas::xgemv_T_sub(B̂i, λ.batch(di0),
                                      ux.batch(di).top_rows(nu), be);
            // x = LQ⁻ᵀ q, u = LR⁻ᵀ (l - LSᵀ x)
            compact_blas::xtrsv_LTN(R̂ŜQ̂i, ux.batch(di), be);
        } else {
            GUANAQO_TRACE("Riccati solve QRS", k);
            // l -= LB̂ᵀ λ0 + LSᵀ q
            compact_blas::xgemv_T_sub(B̂i, λ.batch(di0),
                                      ux.batch(di).top_rows(nu), be);
            compact_blas::xgemv_T_sub(Ŝi, ux.batch(di).bottom_rows(nx),
                                      ux.batch(di).top_rows(nu), be);
            // x = LQ⁻ᵀ q, u = LR⁻ᵀ (l - LSᵀ x)
            compact_blas::xtrsv_LTN(R̂i, ux.batch(di).top_rows(nu), be);
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::solve_riccati_reverse_alt(
    index_t ti, mut_matrix_view ux, mut_matrix_view λ,
    mut_matrix_view work) const {
    const index_t num_stages = N_horiz >> lP;   // number of stages per thread
    const index_t di0        = ti * num_stages; // data batch index
    const index_t biI        = sub_wrap_PmV(ti, 1);
    const index_t diI        = biI * num_stages;
    const index_t k0         = ti * num_stages; // stage index
    const index_t nux        = nu + nx;
    const auto be            = backend;
    auto R̂ŜQ̂                 = riccati_R̂ŜQ̂.batch(ti);
    auto B̂                   = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
    auto Â                   = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
    const auto w             = work.batch(ti);

    for (index_t i = num_stages; i-- > 0;) {
        index_t k  = sub_wrap_N(k0, i);
        index_t di = di0 + i;
        auto R̂ŜQ̂i  = R̂ŜQ̂.middle_cols(i * nux, nux);
        auto Q̂i    = R̂ŜQ̂i.bottom_right(nx, nx);
        auto R̂i    = R̂ŜQ̂i.top_left(nu, nu);
        auto Ŝi    = R̂ŜQ̂i.bottom_left(nx, nu);
        auto B̂i    = B̂.middle_cols(i * nu, nu);
        auto Âi    = Â.middle_cols(i * nx, nx);
        if (i + 1 < num_stages) {
            [[maybe_unused]] const auto k_next = sub_wrap_N(k, 1);
            const auto di_next                 = di + 1;
            GUANAQO_TRACE("Riccati solve rev", k_next);
            auto BAi = data_BA.batch(di_next);

            // w = p
            compact_blas::xcopy(ux.batch(di).bottom_rows(nx), w);
            // x = A x(next) + B u(next) - b(next)
            compact_blas::xadd_neg_copy(ux.batch(di).bottom_rows(nx),
                                        λ.batch(di_next));
            compact_blas::xgemv_add(BAi, ux.batch(di_next),
                                    ux.batch(di).bottom_rows(nx), be);
            // u = LR⁻ᵀ(l - LSᵀ x - LB̂ᵀ λ(last))
            compact_blas::xgemv_T_sub(B̂i, λ.batch(di0),
                                      ux.batch(di).top_rows(nu), be);
            compact_blas::xgemv_T_sub(Ŝi, ux.batch(di).bottom_rows(nx),
                                      ux.batch(di).top_rows(nu), be);
            compact_blas::xtrsv_LTN(R̂i, ux.batch(di).top_rows(nu), be);

            // λ(next) = LQ LQᵀ x + Âᵀ λ(last) - p
            compact_blas::xcopy(ux.batch(di).bottom_rows(nx), λ.batch(di_next));
            compact_blas::xtrmv_T(Q̂i, λ.batch(di_next), be);
            compact_blas::xtrmv(Q̂i, λ.batch(di_next), be);
            compact_blas::xgemv_T_add(Âi, λ.batch(di0), λ.batch(di_next), be);
            compact_blas::xsub_copy(λ.batch(di_next), λ.batch(di_next), w);
        } else {
            // x_last = LQ⁻ᵀ(q_last + LQ⁻¹ λ - LÂᵀ λ)
            GUANAQO_TRACE("Riccati solve rev", k);
            // λ0 -= Â λ
            const auto x_last  = ux.batch(di).bottom_rows(nx);
            const bool x_lanes = ti == 0;
            x_lanes ? compact_blas::template xadd_copy<1>(w, λ.batch(diI))
                    : compact_blas::template xadd_copy<0>(w, λ.batch(diI));
            // LQ⁻¹ λ
            compact_blas::xtrsv_LNN(Q̂i, w, backend);
            // LQ⁻¹ λ - LÂᵀ λ
            compact_blas::xgemv_T_sub(Âi, λ.batch(di0), w, be);
            // x_last = LQ⁻ᵀ(LQ⁻¹ λ - LÂᵀ λ)
            compact_blas::xtrsv_LTN(Q̂i, w, backend);
            compact_blas::xadd_copy(x_last, x_last, w);

            // u -= LB̂ᵀ λ0 + LSᵀ q
            compact_blas::xgemv_T_sub(B̂i, λ.batch(di0),
                                      ux.batch(di).top_rows(nu), be);
            compact_blas::xgemv_T_sub(Ŝi, ux.batch(di).bottom_rows(nx),
                                      ux.batch(di).top_rows(nu), be);
            // x = LQ⁻ᵀ q, u = LR⁻ᵀ (u - LSᵀ x)
            compact_blas::xtrsv_LTN(R̂i, ux.batch(di).top_rows(nu), be);
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::solve_reverse(mut_matrix_view ux, mut_matrix_view λ,
                                        mut_matrix_view work) const {
    KOQKATOO_ASSERT(((N_horiz >> lP) << lP) == N_horiz);
    const index_t P = 1 << (lP - lvl);
    koqkatoo::foreach_thread(P, [this, &ux, &λ, &work](index_t ti, index_t) {
        for (index_t l = lP - lvl; l-- > 0;) {
            const index_t offset = 1 << l;
            const auto bi        = sub_wrap_PmV(ti, offset);
            if (is_active(l, bi))
                solve_reverse_active(l, bi, λ);
            barrier();
        }
        alt ? solve_riccati_reverse_alt(ti, ux, λ, work)
            : solve_riccati_reverse(ti, ux, λ, work);
    });
}

template <index_t VL>
void CyclicOCPSolver<VL>::solve(mut_matrix_view ux, mut_matrix_view λ,
                                mut_matrix_view_batch work_pcg,
                                mut_matrix_view work_riccati) const {
    solve_forward(ux, λ, work_riccati);
    solve_pcg(λ.batch(0), work_pcg);
    solve_reverse(ux, λ, work_riccati);
}

} // namespace koqkatoo::ocp::cyclocp
