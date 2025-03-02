#pragma once

#include <koqkatoo/ocp/cyclic-solver/cyclic-solver.hpp>

namespace koqkatoo::ocp {

template <class Abi>
void CyclicOCPSolver<Abi>::compute_Ψ(real_t S, real_view Σb, bool_view Jb) {
    using std::isfinite;
    auto [N, nx, nu, ny, ny_N] = dim;
    KOQKATOO_OMP(for schedule(static, 1))
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("factor prep", i);
        const auto hi = get_batch_index(i);
        compact_blas::xcopy(AB.batch(hi), V.batch(hi));
        // Compute H = Hℓ + GᵀΣJ G + Γ⁻¹
        compact_blas::xsyrk_T_schur_copy(CD.batch(hi), Σb.batch(hi),
                                         Jb.batch(hi), H.batch(hi),
                                         LH.batch(hi));
        if (isfinite(S))
            LH.batch(hi).add_to_diagonal(1 / S);
        // Factorize H (and solve V)
        compact_blas::xpotrf(LHV.batch(hi), be);
        // Solve W = LH⁻¹ [I 0]ᵀ
        compact_blas::xtrtri_copy(LH.batch(hi).top_left(nx + nu, nx),
                                  Wᵀ.batch(hi), be);
        compact_blas::xtrsm_LLNN(LH.batch(hi).bottom_right(nu, nu),
                                 Wᵀ.batch(hi).bottom_rows(nu), be);
        // Compute -VWᵀ
        // TODO: fix case where N+1 = 2 vl
        if (n <= 2)
            throw std::logic_error("Horizon too short, not yet implemented");
        if (i & 1) {
            // Odd indices are not transposed. If i ≡ 1 (mod 4), it ends up
            // in the top subdiagonal block, if i ≡ 3 (mod 4), it ends up in
            // the bottom subdiagonal block.
            auto U = (i & 3) == 1 ? Ut : Ub;
            compact_blas::xtrmm_RLNN_neg(V.batch(hi), Wᵀ.batch(hi), U.batch(hi),
                                         be);
        } else if (i + 1 < n) {
            const auto his = get_batch_index(i + 1);
            // Even indices are transposed. If i ≡ 0 (mod 4), it ends up in
            // the bottom subdiagonal block, if i ≡ 2 (mod 4), it ends up
            // in the top subdiagonal block.
            auto U = (i & 3) == 0 ? Ub : Ut;
            compact_blas::xgemm_TT_neg(Wᵀ.batch(hi), V.batch(hi), U.batch(his),
                                       be);
            // TODO: exploit trapezoidal shape of Wᵀ
        }
        // Compute WWᵀ
        // TODO: exploit trapezoidal shape of Wᵀ
        compact_blas::xsyrk_T(Wᵀ.batch(hi), LΨd.batch(hi), be);
    }

    // Subtract VVᵀ from diagonal
    KOQKATOO_OMP(for schedule(static, 1))
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("build Ψ", i);
        const auto hi = get_batch_index(i);
        if (i + 1 < n) {
            const auto hi_next = get_batch_index(i + 1);
            // Compute VVᵀ and add to Ψd
            compact_blas::xsyrk_add(V.batch(hi), LΨd.batch(hi_next), be);
        } else {
            // Last batch is special since we cross batch boundaries
            const auto hi_next = get_batch_index(i - vstride + 1);
            compact_blas::xsyrk_add_shift(V.batch(hi), LΨd.batch(hi_next));
        }
    }
}

template <class Abi>
void CyclicOCPSolver<Abi>::factor_Ψ() {
    for (index_t l = 0; l < ln; ++l) {
        const auto offset     = index_t{1} << l; // first stage in level
        const bool last_level = l + 1 == ln;
        KOQKATOO_OMP(for schedule(static, 1))
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("factor Ψ", i);
            // Parity within the current level
            const auto even    = (get_index_in_level(i) & 1) == 0;
            const auto mod4    = get_index_in_level(i) & 3;
            const bool mod4_12 = mod4 == 1 || mod4 == 2;
            // Batch index of the current column
            const auto hi = get_batch_index(i);
            // Batch indices of top and bottom blocks of generated fill-in
            const auto i_next = i + offset, i_prev = i - offset;
            const auto it  = last_level ? i_prev : even ? i_next : i_prev;
            const auto ib  = even ? i_prev : i_next;
            const auto hit = get_batch_index(it);
            // Factor
            compact_blas::xpotrf(LΨU.batch(hi), be);
            // Update next diagonal block using our top subdiagonal block
            if (even)
                compact_blas::xsyrk_sub(Ut.batch(hi), LΨd.batch(hit), be);
            // Update next diagonal block using our bottom subdiagonal block
            else if (i + offset < n) {
                const auto hib = get_batch_index(ib);
                compact_blas::xsyrk_sub(Ub.batch(hi), LΨd.batch(hib), be);
            } else {
                const auto hib = n - 1;
                compact_blas::xsyrk_sub_shift(Ub.batch(hi), LΨd.batch(hib));
            }
            // Compute subdiagonal fill-in in next level
            auto Ufi = l + 2 >= ln ? (even ? Ut : Ub) : (mod4_12 ? Ut : Ub);
            compact_blas::xgemm_NT_neg(Ub.batch(hi), Ut.batch(hi),
                                       Ufi.batch(hit), be);
        }
        KOQKATOO_OMP(for schedule(static, 1))
        for (index_t i = offset; i < n; i += 2 * offset) {
            {
                KOQKATOO_TRACE("factor Ψ YY", i);
                // Parity within the current level
                const auto odd = (get_index_in_level(i) & 1) != 0;
                // Batch index of the current column
                const auto hi = get_batch_index(i);
                // Batch indices of top and bottom blocks of generated fill-in
                const auto it = odd ? i - offset : i + offset;
                const auto ib = odd ? i + offset : i - offset;
                // Update next diagonal block using our top subdiagonal block
                if (odd) {
                    const auto hit = get_batch_index(it);
                    compact_blas::xsyrk_sub(Ut.batch(hi), LΨd.batch(hit), be);
                }
                // Update next diagonal block using our bottom subdiagonal block
                else if (!last_level) {
                    const auto hib = get_batch_index(ib);
                    compact_blas::xsyrk_sub(Ub.batch(hi), LΨd.batch(hib), be);
                } else {
                    const auto hib = n - 1;
                    compact_blas::xsyrk_sub_shift(Ub.batch(hi), LΨd.batch(hib));
                }
            }
            if (last_level && i == offset) {
                KOQKATOO_TRACE("factor Ψ", 0);
                compact_blas::xpotrf(LΨd.batch(n - 1), be);
            }
        }
    }
    if (ln == 0) {
        KOQKATOO_OMP(single) {
            KOQKATOO_TRACE("factor Ψ", 0);
            compact_blas::xpotrf(LΨd.batch(n - 1), be);
        }
    }
}

template <class Abi>
void CyclicOCPSolver<Abi>::solve_H_fwd(real_view grad, real_view Mᵀλ,
                                       real_view Aᵀŷ, mut_real_view d,
                                       mut_real_view Δλ) const {
    // Solve Hv = -g
    KOQKATOO_OMP(for schedule(static, 1))
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("solve Hv=g", i);
        auto hi = get_batch_index(i);
        compact_blas::xadd_neg_copy(d.batch(hi), grad.batch(hi), Mᵀλ.batch(hi),
                                    Aᵀŷ.batch(hi));
        // Solve Lᴴ vʹ = g                                          (d ← vʹ)
        compact_blas::xtrsv_LNN(LH.batch(hi), d.batch(hi), be);
        // Solve Lᴴ⁻ᵀ v = vʹ                                        (λ ← Ev)
        compact_blas::xgemv_T_add(Wᵀ.batch(hi), d.batch(hi), Δλ.batch(hi), be);
    }
    KOQKATOO_OMP(for schedule(static, 1))
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("eval rhs Ψ", i);
        auto hi = get_batch_index(i);
        if (i + 1 < n) {
            auto hi_next = get_batch_index(i + 1);
            compact_blas::xgemv_sub(V.batch(hi), d.batch(hi), Δλ.batch(hi_next),
                                    be);
        } else {
            // Last batch is special since we cross batch boundaries
            auto hi_next = get_batch_index(i - vstride + 1);
            compact_blas::xgemv_sub_shift(V.batch(hi), d.batch(hi),
                                          Δλ.batch(hi_next));
        }
    }
}

template <class Abi>
void CyclicOCPSolver<Abi>::solve_H_rev(mut_real_view d, real_view Δλ,
                                       mut_real_view MᵀΔλ) const {
    const auto [N, nx, nu, ny, ny_N] = dim;
    KOQKATOO_OMP(for schedule(static, 1))
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("solve Hd=-g-MᵀΔλ", i);
        const auto hi = get_batch_index(i);
        compact_blas::xgemv_sub(Wᵀ.batch(hi), Δλ.batch(hi), d.batch(hi), be);
        if (i + 1 < n) {
            const auto hi_next = get_batch_index(i + 1);
            compact_blas::xgemv_T_add(V.batch(hi), Δλ.batch(hi_next),
                                      d.batch(hi), be);
        } else {
            const auto hi_next = get_batch_index(i - vstride + 1);
            compact_blas::xgemv_T_add_shift(V.batch(hi), Δλ.batch(hi_next),
                                            d.batch(hi));
        }
        compact_blas::xtrsv_LTN(LH.batch(hi), d.batch(hi), be);
        MᵀΔλ.batch(hi).top_rows(nx) = Δλ.batch(hi);
        MᵀΔλ.batch(hi).bottom_rows(nu).set_constant(0);
        if (i + 1 < n) {
            const auto hi_next = get_batch_index(i + 1);
            compact_blas::xgemv_T_sub(AB.batch(hi), Δλ.batch(hi_next),
                                      MᵀΔλ.batch(hi), be);
        } else {
            const auto hi_next = get_batch_index(i - vstride + 1);
            compact_blas::xgemv_T_sub_shift(AB.batch(hi), Δλ.batch(hi_next),
                                            MᵀΔλ.batch(hi));
        }
    }
}

template <class Abi>
void CyclicOCPSolver<Abi>::solve_Ψ_work(mut_real_view Δλ,
                                        mut_real_view work_pcg) const {
    // Forward pass
    for (index_t l = 0; l < ln; ++l) {
        const auto offset     = index_t{1} << l; // first stage in level
        const bool last_level = l + 1 == ln;
        KOQKATOO_OMP(for schedule(static, 1))
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ fwd", i);
            // Parity within the current level
            const auto even = (get_index_in_level(i) & 1) == 0;
            // Batch index of the current column
            const auto hi = get_batch_index(i);
            // Batch indices of top and bottom blocks of generated fill-in
            const auto i_next = i + offset, i_prev = i - offset;
            const auto it = last_level ? i_prev : even ? i_next : i_prev;
            const auto ib = even ? i_prev : i_next;
            // Solve
            compact_blas::xtrsv_LNN(LΨd.batch(hi), Δλ.batch(hi), be);
            // Update next diagonal block using our top subdiagonal block
            if (even) {
                const auto hit = get_batch_index(it);
                compact_blas::xgemv_sub(Ut.batch(hi), Δλ.batch(hi),
                                        Δλ.batch(hit), be);
            }
            // Update next diagonal block using our bottom subdiagonal block
            else if (i + offset < n) {
                const auto hib = get_batch_index(ib);
                compact_blas::xgemv_sub(Ub.batch(hi), Δλ.batch(hi),
                                        Δλ.batch(hib), be);
            } else {
                const auto hib = n - 1;
                compact_blas::xgemv_sub_shift(Ub.batch(hi), Δλ.batch(hi),
                                              Δλ.batch(hib));
            }
        }
        KOQKATOO_OMP(for schedule(static, 1))
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ fwd 2", i);
            // Parity within the current level
            const auto odd = (get_index_in_level(i) & 1) != 0;
            // Batch index of the current column
            const auto hi = get_batch_index(i);
            // Batch indices of top and bottom blocks of generated fill-in
            const auto i_next = i + offset, i_prev = i - offset;
            const auto it = odd ? i_prev : i_next;
            const auto ib = odd ? i_next : i_prev;
            // Update next diagonal block using our top subdiagonal block
            if (odd) {
                const auto hit = get_batch_index(it);
                compact_blas::xgemv_sub(Ut.batch(hi), Δλ.batch(hi),
                                        Δλ.batch(hit), be);
            }
            // Update next diagonal block using our bottom subdiagonal block
            else if (!last_level) {
                const auto hib = get_batch_index(ib);
                compact_blas::xgemv_sub(Ub.batch(hi), Δλ.batch(hi),
                                        Δλ.batch(hib), be);
            } else {
                const auto hib = n - 1;
                compact_blas::xgemv_sub_shift(Ub.batch(hi), Δλ.batch(hi),
                                              Δλ.batch(hib));
            }
        }
    }

    // Solve the remaining small tridiagonal system using PCG
    KOQKATOO_OMP(single) {
        auto r  = work_pcg.batch(0).middle_cols(0, 1),
             z  = work_pcg.batch(0).middle_cols(1, 1),
             p  = work_pcg.batch(0).middle_cols(2, 1),
             Ap = work_pcg.batch(0).middle_cols(3, 1);
        auto x  = Δλ.batch(n - 1);
        auto A = LΨd.batch(n - 1), B = Ut.batch(n - 1);
        real_t rᵀz = [&] {
            KOQKATOO_TRACE("solve Ψ pcg", 0);
            compact_blas::xcopy(x, r);
            compact_blas::xfill(0, x);
            real_t rᵀz = mul_precond(r, z, Ap, A, B);
            compact_blas::xcopy(z, p);
            return rᵀz;
        }();
        for (index_t it = 0; it < 20; ++it) {
            KOQKATOO_TRACE("solve Ψ pcg", it + 1);
            real_t pᵀAp = mul_A(p, Ap, A, B);
            real_t α    = rᵀz / pᵀAp;
            compact_blas::xaxpy(+α, p, x);
            compact_blas::xaxpy(-α, Ap, r);
            real_t r2          = compact_blas::xdot(r, r);
            constexpr real_t ε = std::numeric_limits<real_t>::epsilon();
            if (r2 < ε * ε)
                break;
            real_t rᵀz_new = mul_precond(r, z, Ap, A, B);
            real_t β       = rᵀz_new / rᵀz;
            compact_blas::xaxpby(1, z, β, p);
            rᵀz = rᵀz_new;
        }
    }

    // Backward pass
    for (index_t l = ln; l-- > 0;) {
        const auto offset     = index_t{1} << l; // first stage in level
        const bool last_level = l + 1 == ln;
        KOQKATOO_OMP(for schedule(static, 1))
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ rev 2", 0);
            // Parity within the current level
            const auto odd = (get_index_in_level(i) & 1) != 0;
            // Batch index of the current column
            const auto hi = get_batch_index(i);
            // Batch indices of top and bottom blocks of generated fill-in
            const auto i_next = i + offset, i_prev = i - offset;
            const auto it = odd ? i_prev : i_next;
            const auto ib = odd ? i_next : i_prev;
            // Update next diagonal block using our top subdiagonal block
            if (odd) {
                const auto hit = get_batch_index(it);
                compact_blas::xgemv_T_sub(Ut.batch(hi), Δλ.batch(hit),
                                          Δλ.batch(hi), be);
            }
            // Update next diagonal block using our bottom subdiagonal block
            else if (!last_level) {
                const auto hib = get_batch_index(ib);
                compact_blas::xgemv_T_sub(Ub.batch(hi), Δλ.batch(hib),
                                          Δλ.batch(hi), be);
            } else {
                const auto hib = n - 1;
                compact_blas::xgemv_T_sub_shift(Ub.batch(hi), Δλ.batch(hib),
                                                Δλ.batch(hi));
            }
        }
        KOQKATOO_OMP(for schedule(static, 1))
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ rev", i);
            // Parity within the current level
            const auto even = (get_index_in_level(i) & 1) == 0;
            // Batch index of the current column
            const auto hi = get_batch_index(i);
            // Batch indices of top and bottom blocks of generated fill-in
            const auto i_next = i + offset, i_prev = i - offset;
            const auto it = last_level ? i_prev : even ? i_next : i_prev;
            const auto ib = even ? i_prev : i_next;
            // Update next diagonal block using our top subdiagonal block
            if (even) {
                const auto hit = get_batch_index(it);
                compact_blas::xgemv_T_sub(Ut.batch(hi), Δλ.batch(hit),
                                          Δλ.batch(hi), be);
            }
            // Update next diagonal block using our bottom subdiagonal block
            else if (i + offset < n) {
                const auto hib = get_batch_index(ib);
                compact_blas::xgemv_T_sub(Ub.batch(hi), Δλ.batch(hib),
                                          Δλ.batch(hi), be);
            } else {
                const auto hib = n - 1;
                compact_blas::xgemv_T_sub_shift(Ub.batch(hi), Δλ.batch(hib),
                                                Δλ.batch(hi));
            }
            // Solve
            compact_blas::xtrsv_LTN(LΨd.batch(hi), Δλ.batch(hi), be);
        }
    }
}

} // namespace koqkatoo::ocp
