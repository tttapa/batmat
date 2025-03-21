#include <Eigen/SparseCholesky>
#include <gtest/gtest.h>

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/xgemm.hpp> // TODO
#include <koqkatoo/linalg-compact/preferred-backend.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/openmp.h>
#include <guanaqo/trace.hpp>

#include <experimental/simd>
#include <guanaqo/print.hpp>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <print>
#include <stdexcept>
#include <type_traits>

namespace stdx = std::experimental;

#if DO_PRINT
#define PRINTLN(...) std::println(__VA_ARGS__)
#else
#define PRINTLN(...)
#endif

namespace koqkatoo::ocp {

[[nodiscard]] constexpr index_t get_depth(index_t n) {
    assert(n > 0);
    auto un = static_cast<std::make_unsigned_t<index_t>>(n);
    return static_cast<index_t>(std::bit_width(un - 1));
}

[[nodiscard]] constexpr index_t get_level(index_t i) {
    assert(i > 0);
    auto ui = static_cast<std::make_unsigned_t<index_t>>(i);
    return static_cast<index_t>(std::countr_zero(ui));
}

#if KQT_CYCLIC_TEMPLATE
template <index_t VL = 4>
#else
constexpr index_t VL = 4;
#endif
struct CyclicOCPSolver {
    using simd_abi      = stdx::simd_abi::deduce_t<real_t, VL>;
    using compact_blas  = linalg::compact::CompactBLAS<simd_abi>;
    using scalar_blas   = linalg::compact::CompactBLAS<stdx::simd_abi::scalar>;
    using real_matrix   = compact_blas::matrix;
    using real_view     = compact_blas::batch_view;
    using mut_real_view = compact_blas::mut_batch_view;
    using bool_view     = compact_blas::bool_batch_view;
    using bool_matrix   = compact_blas::bool_matrix;
    using real_view_single     = compact_blas::single_batch_view;
    using mut_real_view_single = compact_blas::mut_single_batch_view;

    static constexpr index_t vl  = VL;
    static constexpr index_t lvl = get_depth(vl);

    const OCPDim dim;
    const index_t lN      = get_depth(dim.N_horiz + 1);
    const index_t ln      = lN - lvl;
    const index_t n       = index_t{1} << ln;
    const index_t vstride = index_t{1} << ln;

    linalg::compact::PreferredBackend be =
        linalg::compact::PreferredBackend::MKLScalarBatched;

    [[nodiscard]] static constexpr index_t get_index_in_level(index_t i) {
        auto l = get_level(i);
        return i >> (l + 1);
    }

    [[nodiscard]] constexpr index_t get_level_width(index_t l) const {
        assert(l < ln);
        return index_t{1} << (ln - l - 1);
    }

    [[nodiscard]] constexpr index_t get_batch_index(index_t i) const {
        assert(i < n);
        if (i == 0)
            return n - 1;
        auto l  = get_level(i);
        auto il = get_index_in_level(i);
        return il + (1 << ln) - (1 << (ln - l));
    }

    void initialize(const LinearOCPStorage &ocp) {
        auto [N, nx, nu, ny, ny_N] = dim;
        for (index_t i = 0; i < n; ++i) {
            auto bi = get_batch_index(i);
            for (index_t vi = 0; vi < VL; ++vi) {
                auto k = i + vi * vstride;
                PRINTLN("  {} -> {}({})", k, bi, vi);
                if (k < dim.N_horiz) {
                    H.batch(bi)(vi)  = ocp.H(k);
                    AB.batch(bi)(vi) = ocp.AB(k);
                    CD.batch(bi)(vi) = ocp.CD(k);
                } else if (k < dim.N_horiz) {
                    H.batch(bi)(vi).bottom_left(nu, nx).set_constant(0);
                    H.batch(bi)(vi).right_cols(nu).set_constant(0);
                    H.batch(bi)(vi).bottom_right(nu, nu).set_diagonal(1);
                    H.batch(bi)(vi).top_left(nx, nx) = ocp.Q(k);
                    AB.batch(bi)(vi).set_constant(0);
                    CD.batch(bi)(vi).right_cols(nu).set_constant(0);
                    CD.batch(bi)(vi).bottom_left(ny - ny_N, nx).set_constant(0);
                    CD.batch(bi)(vi).top_left(ny_N, nx) = ocp.C(k);
                } else {
                    H.batch(bi)(vi).set_constant(0);
                    H.batch(bi)(vi).set_diagonal(1);
                    AB.batch(bi)(vi).set_constant(0);
                    CD.batch(bi)(vi).set_constant(0);
                }
            }
        }
    }

    template <class Tin, class Tout>
    void pack_vectors(std::span<Tin> in,
                      typename compact_blas::template batch_view_t<Tout> out,
                      index_t rows, index_t rows_N) {
        auto [N, nx, nu, ny, ny_N] = dim;
        using view = typename compact_blas::template batch_view_scalar_t<Tin>;
        view vw{{.data = in.data(), .depth = N + 1, .rows = rows, .cols = 1}};
        assert(in.size() == static_cast<size_t>(N * rows + rows_N));
        for (index_t i = 0; i < n; ++i) {
            auto bi = get_batch_index(i);
            for (index_t vi = 0; vi < VL; ++vi) {
                auto k = i + vi * vstride;
                if (k < N) {
                    out.batch(bi)(vi) = vw(k);
                } else if (k == N) {
                    out.batch(bi)(vi).top_rows(rows_N) = vw(k).top_rows(rows_N);
                    out.batch(bi)(vi)
                        .bottom_rows(rows - rows_N)
                        .set_constant(0);
                } else {
                    out.batch(bi)(vi).set_constant(0);
                }
            }
        }
    }

    template <class Tin, class Tout>
    void unpack_vectors(typename compact_blas::template batch_view_t<Tin> in,
                        std::span<Tout> out, index_t rows, index_t rows_N) {
        auto [N, nx, nu, ny, ny_N] = dim;
        using view = typename compact_blas::template batch_view_scalar_t<Tout>;
        view vw{{.data = out.data(), .depth = N + 1, .rows = rows, .cols = 1}};
        assert(out.size() == static_cast<size_t>(N * rows + rows_N));
        for (index_t i = 0; i < n; ++i) {
            auto bi = get_batch_index(i);
            for (index_t vi = 0; vi < VL; ++vi) {
                auto k = i + vi * vstride;
                if (k < N) {
                    vw(k) = in.batch(bi)(vi);
                } else if (k == N) {
                    vw(k).top_rows(rows_N) = in.batch(bi)(vi).top_rows(rows_N);
                }
            }
        }
    }

    template <class Tin, class Tout>
    void pack_dyn(std::span<Tin> in,
                  typename compact_blas::template batch_view_t<Tout> out) {
        pack_vectors(in, out, dim.nx, dim.nx);
    }

    template <class Tin, class Tout>
    void unpack_dyn(typename compact_blas::template batch_view_t<Tin> in,
                    std::span<Tout> out) {
        unpack_vectors(in, out, dim.nx, dim.nx);
    }

    template <class Tin, class Tout>
    void pack_constr(std::span<Tin> in,
                     typename compact_blas::template batch_view_t<Tout> out) {
        pack_vectors(in, out, dim.ny, dim.ny_N);
    }

    template <class Tin, class Tout>
    void unpack_constr(typename compact_blas::template batch_view_t<Tin> in,
                       std::span<Tout> out) {
        unpack_vectors(in, out, dim.ny, dim.ny_N);
    }

    template <class Tin, class Tout>
    void pack_var(std::span<Tin> in,
                  typename compact_blas::template batch_view_t<Tout> out) {
        pack_vectors(in, out, dim.nx + dim.nu, dim.nx);
    }

    template <class Tin, class Tout>
    void unpack_var(typename compact_blas::template batch_view_t<Tin> in,
                    std::span<Tout> out) {
        unpack_vectors(in, out, dim.nx + dim.nu, dim.nx);
    }

    real_t mul_A(real_view_single p, // NOLINT(*-nodiscard)
                 mut_real_view_single Ap, real_view_single L,
                 real_view_single B) const {
        compact_blas::xcopy(p, Ap);
        compact_blas::xtrmv_T(L, Ap, be);
        compact_blas::xtrmv(L, Ap, be);
        using linalg::compact::micro_kernels::gemm::xsyomv_register; // TODO
        xsyomv_register<simd_abi, false>(B, p, Ap);
        return compact_blas::xdot(p, Ap);
    }

    real_t mul_precond(real_view_single r, // NOLINT(*-nodiscard)
                       mut_real_view_single z, mut_real_view_single w,
                       real_view_single L, real_view_single B) const {
        compact_blas::xcopy(r, z);
#if USE_JACOBI_PREC
        std::ignore = w;
        std::ignore = B;
#else
        compact_blas::xcopy(r, w);
        compact_blas::xtrsv_LNN(L, w, be);
        compact_blas::xtrsv_LTN(L, w, be);
        using linalg::compact::micro_kernels::gemm::xsyomv_register; // TODO
        xsyomv_register<simd_abi, true>(B, w.as_const(), z);
#endif
        compact_blas::xtrsv_LNN(L, z, be);
        compact_blas::xtrsv_LTN(L, z, be);
        return compact_blas::xdot(r, z);
    }

    void compute_Ψ(real_t S, real_view Σb, bool_view Jb);
    void factor_Ψ();
    void solve_Ψ_work(mut_real_view Δλ, mut_real_view work_pcg) const;
    void solve_Ψ(mut_real_view Δλ) { solve_Ψ_work(Δλ, work_pcg); }

#include "cyclic-storage.ipp"

    const mut_real_view H   = HAB.top_rows(dim.nx + dim.nu),
                        AB  = HAB.bottom_rows(dim.nx);
    const mut_real_view LH  = LHV.top_rows(dim.nx + dim.nu),
                        V   = LHV.bottom_rows(dim.nx);
    const mut_real_view LΨd = LΨU.top_rows(dim.nx),
                        Ut  = LΨU.middle_rows(dim.nx, dim.nx),
                        Ub  = LΨU.bottom_rows(dim.nx);

    real_matrix work_pcg{{.depth = vl, .rows = dim.nx, .cols = 4}};

  public:
    CyclicOCPSolver(OCPDim dim) : dim{dim} {}
    CyclicOCPSolver(const CyclicOCPSolver &)            = delete;
    CyclicOCPSolver &operator=(const CyclicOCPSolver &) = delete;
};

#if KQT_CYCLIC_TEMPLATE
template <index_t VL>
void CyclicOCPSolver<VL>
#else
void CyclicOCPSolver
#endif
    ::compute_Ψ(real_t S, real_view Σb, bool_view Jb) {
    using std::isfinite;
    auto [N, nx, nu, ny, ny_N] = dim;
    KOQKATOO_OMP(parallel) {
        KOQKATOO_OMP(for schedule(static, 1))
        for (index_t i = 0; i < n; ++i) {
            GUANAQO_TRACE("factor prep", i);
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
                throw std::logic_error(
                    "Horizon too short, not yet implemented");
            if (i & 1) {
                // Odd indices are not transposed. If i ≡ 1 (mod 4), it ends up
                // in the top subdiagonal block, if i ≡ 3 (mod 4), it ends up in
                // the bottom subdiagonal block.
                auto U = (i & 3) == 1 ? Ut : Ub;
                compact_blas::xtrmm_RLNN_neg(V.batch(hi), Wᵀ.batch(hi),
                                             U.batch(hi), be);
            } else if (i + 1 < n) {
                const auto his = get_batch_index(i + 1);
                // Even indices are transposed. If i ≡ 0 (mod 4), it ends up in
                // the bottom subdiagonal block, if i ≡ 2 (mod 4), it ends up
                // in the top subdiagonal block.
                auto U = (i & 3) == 0 ? Ub : Ut;
                compact_blas::xgemm_TT_neg(Wᵀ.batch(hi), V.batch(hi),
                                           U.batch(his), be);
                // TODO: exploit trapezoidal shape of Wᵀ
            }
            // Compute WWᵀ
            // TODO: exploit trapezoidal shape of Wᵀ
            compact_blas::xsyrk_T(Wᵀ.batch(hi), LΨd.batch(hi), be);
        }

        // Subtract VVᵀ from diagonal
        KOQKATOO_OMP(for schedule(static, 1))
        for (index_t i = 0; i < n; ++i) {
            GUANAQO_TRACE("build Ψ", i);
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
    } // end parallel
}

#if KQT_CYCLIC_TEMPLATE
template <index_t VL>
void CyclicOCPSolver<VL>
#else
void CyclicOCPSolver
#endif
    ::factor_Ψ() {
    KOQKATOO_OMP(parallel) {
        for (index_t l = 0; l < ln; ++l) {
            const auto offset     = index_t{1} << l; // first stage in level
            const bool last_level = l + 1 == ln;
            PRINTLN("Level {}\n  offset={:<2}, width={}", l, offset, width);
            KOQKATOO_OMP(for schedule(static, 1))
            for (index_t i = offset; i < n; i += 2 * offset) {
                GUANAQO_TRACE("factor Ψ", i);
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
                PRINTLN("  {:>2}:*:{:<2}({},{}) {}", hi, i, it, ib,
                        even ? "even" : "odd");
                // Factor
                compact_blas::xpotrf(LΨU.batch(hi), be);
                // Update next diagonal block using our top subdiagonal block
                if (even) {
                    PRINTLN("    diag block    {:>2}[1] -> {:>2}", hi, hit);
                    compact_blas::xsyrk_sub(Ut.batch(hi), LΨd.batch(hit), be);
                }
                // Update next diagonal block using our bottom subdiagonal block
                else if (i + offset < n) {
                    const auto hib = get_batch_index(ib);
                    PRINTLN("    diag block    {:>2}[2] -> {:>2}", hi, hib);
                    compact_blas::xsyrk_sub(Ub.batch(hi), LΨd.batch(hib), be);
                } else {
                    const auto hib = n - 1;
                    PRINTLN("    diag block*   {:>2}[2] -> {:>2}", hi, hib);
                    compact_blas::xsyrk_sub_shift(Ub.batch(hi), LΨd.batch(hib));
                }
                // Compute subdiagonal fill-in in next level
                auto Ufi = l + 2 >= ln ? (even ? Ut : Ub) : (mod4_12 ? Ut : Ub);
                compact_blas::xgemm_NT_neg(Ub.batch(hi), Ut.batch(hi),
                                           Ufi.batch(hit), be);
                PRINTLN("    subdiag block {:>2} -> {:>2}[{}]", hi, hit,
                        Ufi.data == Ut.data ? 1 : 2);
            }
            KOQKATOO_OMP(single)
            PRINTLN("\n---\n");
            KOQKATOO_OMP(for schedule(static, 1))
            for (index_t i = offset; i < n; i += 2 * offset) {
                {
                    GUANAQO_TRACE("factor Ψ YY", i);
                    // Parity within the current level
                    const auto odd = (get_index_in_level(i) & 1) != 0;
                    // Batch index of the current column
                    const auto hi = get_batch_index(i);
                    // Batch indices of top and bottom blocks of generated fill-in
                    const auto it = odd ? i - offset : i + offset;
                    const auto ib = odd ? i + offset : i - offset;
                    PRINTLN("  {:>2}:*:{:<2}({},{}) {}", hi, i, it, ib,
                            !odd ? "even" : "odd");
                    // Update next diagonal block using our top subdiagonal block
                    if (odd) {
                        const auto hit = get_batch_index(it);
                        PRINTLN("    diag block    {:>2}[1] -> {:>2}", hi, hit);
                        compact_blas::xsyrk_sub(Ut.batch(hi), LΨd.batch(hit),
                                                be);
                    }
                    // Update next diagonal block using our bottom subdiagonal block
                    else if (!last_level) {
                        const auto hib = get_batch_index(ib);
                        PRINTLN("    diag block    {:>2}[2] -> {:>2}", hi, hib);
                        compact_blas::xsyrk_sub(Ub.batch(hi), LΨd.batch(hib),
                                                be);
                    } else {
                        const auto hib = n - 1;
                        PRINTLN("    diag block*   {:>2}[2] -> {:>2}", hi, hib);
                        compact_blas::xsyrk_sub_shift(Ub.batch(hi),
                                                      LΨd.batch(hib));
                    }
                }
                if (last_level && i == offset) {
                    PRINTLN("Level {}\n\n===\n", ln);
                    GUANAQO_TRACE("factor Ψ", 0);
                    compact_blas::xpotrf(LΨd.batch(n - 1), be);
                }
            }
            KOQKATOO_OMP(single)
            PRINTLN("\n===\n");
        }
        if (ln == 0) {
            KOQKATOO_OMP(single) {
                PRINTLN("Level {}\n\n===\n", ln);
                GUANAQO_TRACE("factor Ψ", 0);
                compact_blas::xpotrf(LΨd.batch(n - 1), be);
            }
        }
    }
}

#if KQT_CYCLIC_TEMPLATE
template <index_t VL>
void CyclicOCPSolver<VL>
#else
void CyclicOCPSolver
#endif
    ::solve_Ψ_work(mut_real_view Δλ, mut_real_view work_pcg) const {
    KOQKATOO_OMP(parallel) {
        // Forward pass
        for (index_t l = 0; l < ln; ++l) {
            const auto offset     = index_t{1} << l; // first stage in level
            const bool last_level = l + 1 == ln;
            PRINTLN("Level {}\n  offset={:<2}, width={}", l, offset, width);
            KOQKATOO_OMP(for schedule(static, 1))
            for (index_t i = offset; i < n; i += 2 * offset) {
                GUANAQO_TRACE("solve ψ fwd", i);
                // Parity within the current level
                const auto even = (get_index_in_level(i) & 1) == 0;
                // Batch index of the current column
                const auto hi = get_batch_index(i);
                // Batch indices of top and bottom blocks of generated fill-in
                const auto i_next = i + offset, i_prev = i - offset;
                const auto it = last_level ? i_prev : even ? i_next : i_prev;
                const auto ib = even ? i_prev : i_next;
                PRINTLN("  {:>2}:*:{:<2}({},{}) {}", hi, i, it, ib,
                        even ? "even" : "odd");
                // Solve
                compact_blas::xtrsv_LNN(LΨd.batch(hi), Δλ.batch(hi), be);
                // Update next diagonal block using our top subdiagonal block
                if (even) {
                    const auto hit = get_batch_index(it);
                    PRINTLN("    rhs update    {:>2}[1] -> {:>2}", hi, hit);
                    compact_blas::xgemv_sub(Ut.batch(hi), Δλ.batch(hi),
                                            Δλ.batch(hit), be);
                }
                // Update next diagonal block using our bottom subdiagonal block
                else if (i + offset < n) {
                    const auto hib = get_batch_index(ib);
                    PRINTLN("    rhs update    {:>2}[2] -> {:>2}", hi, hib);
                    compact_blas::xgemv_sub(Ub.batch(hi), Δλ.batch(hi),
                                            Δλ.batch(hib), be);
                } else {
                    const auto hib = n - 1;
                    PRINTLN("    rhs update*   {:>2}[2] -> {:>2}", hi, hib);
                    compact_blas::xgemv_sub_shift(Ub.batch(hi), Δλ.batch(hi),
                                                  Δλ.batch(hib));
                }
            }
            KOQKATOO_OMP(single)
            PRINTLN("\n---\n");
            KOQKATOO_OMP(for schedule(static, 1))
            for (index_t i = offset; i < n; i += 2 * offset) {
                GUANAQO_TRACE("solve ψ fwd 2", i);
                // Parity within the current level
                const auto odd = (get_index_in_level(i) & 1) != 0;
                // Batch index of the current column
                const auto hi = get_batch_index(i);
                // Batch indices of top and bottom blocks of generated fill-in
                const auto i_next = i + offset, i_prev = i - offset;
                const auto it = odd ? i_prev : i_next;
                const auto ib = odd ? i_next : i_prev;
                PRINTLN("  {:>2}:*:{:<2}({},{}) {}", hi, i, it, ib,
                        !odd ? "even" : "odd");
                // Update next diagonal block using our top subdiagonal block
                if (odd) {
                    const auto hit = get_batch_index(it);
                    PRINTLN("    rhs update    {:>2}[1] -> {:>2}", hi, hit);
                    compact_blas::xgemv_sub(Ut.batch(hi), Δλ.batch(hi),
                                            Δλ.batch(hit), be);
                }
                // Update next diagonal block using our bottom subdiagonal block
                else if (!last_level) {
                    const auto hib = get_batch_index(ib);
                    PRINTLN("    rhs update    {:>2}[2] -> {:>2}", hi, hib);
                    compact_blas::xgemv_sub(Ub.batch(hi), Δλ.batch(hi),
                                            Δλ.batch(hib), be);
                } else {
                    const auto hib = n - 1;
                    PRINTLN("    rhs update*   {:>2}[2] -> {:>2}", hi, hib);
                    compact_blas::xgemv_sub_shift(Ub.batch(hi), Δλ.batch(hi),
                                                  Δλ.batch(hib));
                }
            }
            KOQKATOO_OMP(single)
            PRINTLN("\n===\n");
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
                GUANAQO_TRACE("solve Ψ pcg", 0);
                compact_blas::xcopy(x, r);
                compact_blas::xfill(0, x);
                real_t rᵀz = mul_precond(r, z, Ap, A, B);
                compact_blas::xcopy(z, p);
                return rᵀz;
            }();

#if DO_PRINT
            std::cout << "prec_rhs = [\n";
            for (index_t j = 0; j < VL; ++j) {
                guanaqo::print_python(std::cout, z(j), ",\n", false);
            }
            std::cout << "]\n";
#endif

            for (index_t it = 0; it < 20; ++it) {
                GUANAQO_TRACE("solve Ψ pcg", it + 1);
                real_t pᵀAp = mul_A(p, Ap, A, B);
                real_t α    = rᵀz / pᵀAp;
                compact_blas::xaxpy(+α, p, x);
                compact_blas::xaxpy(-α, Ap, r);
                real_t r2 = compact_blas::xdot(r, r);
                PRINTLN("# {}: {}", it, std::sqrt(r2));
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
            PRINTLN("Level {}\n  offset={:<2}, width={}", l, offset, width);
            KOQKATOO_OMP(for schedule(static, 1))
            for (index_t i = offset; i < n; i += 2 * offset) {
                GUANAQO_TRACE("solve ψ rev 2", 0);
                // Parity within the current level
                const auto odd = (get_index_in_level(i) & 1) != 0;
                // Batch index of the current column
                const auto hi = get_batch_index(i);
                // Batch indices of top and bottom blocks of generated fill-in
                const auto i_next = i + offset, i_prev = i - offset;
                const auto it = odd ? i_prev : i_next;
                const auto ib = odd ? i_next : i_prev;
                PRINTLN("  {:>2}:*:{:<2}({},{}) {}", hi, i, it, ib,
                        !odd ? "even" : "odd");
                // Update next diagonal block using our top subdiagonal block
                if (odd) {
                    const auto hit = get_batch_index(it);
                    PRINTLN("    sol update    {:>2}[1] -> {:>2}", hi, hit);
                    compact_blas::xgemv_T_sub(Ut.batch(hi), Δλ.batch(hit),
                                              Δλ.batch(hi), be);
                }
                // Update next diagonal block using our bottom subdiagonal block
                else if (!last_level) {
                    const auto hib = get_batch_index(ib);
                    PRINTLN("    sol update    {:>2}[2] -> {:>2}", hi, hib);
                    compact_blas::xgemv_T_sub(Ub.batch(hi), Δλ.batch(hib),
                                              Δλ.batch(hi), be);
                } else {
                    const auto hib = n - 1;
                    PRINTLN("    sol update*   {:>2}[2] -> {:>2}", hi, hib);
                    compact_blas::xgemv_T_sub_shift(Ub.batch(hi), Δλ.batch(hib),
                                                    Δλ.batch(hi));
                }
            }
            KOQKATOO_OMP(single)
            PRINTLN("\n---\n");
            KOQKATOO_OMP(for schedule(static, 1))
            for (index_t i = offset; i < n; i += 2 * offset) {
                GUANAQO_TRACE("solve ψ rev", i);
                // Parity within the current level
                const auto even = (get_index_in_level(i) & 1) == 0;
                // Batch index of the current column
                const auto hi = get_batch_index(i);
                // Batch indices of top and bottom blocks of generated fill-in
                const auto i_next = i + offset, i_prev = i - offset;
                const auto it = last_level ? i_prev : even ? i_next : i_prev;
                const auto ib = even ? i_prev : i_next;
                PRINTLN("  {:>2}:*:{:<2}({},{}) {}", hi, i, it, ib,
                        even ? "even" : "odd");
                // Update next diagonal block using our top subdiagonal block
                if (even) {
                    const auto hit = get_batch_index(it);
                    PRINTLN("    sol update    {:>2}[1] -> {:>2}", hi, hit);
                    compact_blas::xgemv_T_sub(Ut.batch(hi), Δλ.batch(hit),
                                              Δλ.batch(hi), be);
                }
                // Update next diagonal block using our bottom subdiagonal block
                else if (i + offset < n) {
                    const auto hib = get_batch_index(ib);
                    PRINTLN("    sol update    {:>2}[2] -> {:>2}", hi, hib);
                    compact_blas::xgemv_T_sub(Ub.batch(hi), Δλ.batch(hib),
                                              Δλ.batch(hi), be);
                } else {
                    const auto hib = n - 1;
                    PRINTLN("    sol update*   {:>2}[2] -> {:>2}", hi, hib);
                    compact_blas::xgemv_T_sub_shift(Ub.batch(hi), Δλ.batch(hib),
                                                    Δλ.batch(hi));
                }
                // Solve
                compact_blas::xtrsv_LTN(LΨd.batch(hi), Δλ.batch(hi), be);
            }
            KOQKATOO_OMP(single)
            PRINTLN("\n===\n");
        }
    }
}

} // namespace koqkatoo::ocp

#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/linalg-compact/mkl.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/thread-pool.hpp>
#include <guanaqo/trace.hpp>
#include <guanaqo/eigen/span.hpp>
#include <koqkatoo-version.h>

#include <Eigen/Eigen>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>

#include "eigen-matchers.hpp"

namespace ko = koqkatoo::ocp;
using koqkatoo::index_t;
using koqkatoo::real_t;
using EVec   = Eigen::VectorX<real_t>;
const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.9));

// #include <koqkatoo/ocp/conversion.hpp>
//
// auto build_reference_qp(const ko::LinearOCPStorage &ocp, real_t S,
//                         Eigen::Ref<const Eigen::VectorX<real_t>> Σ,
//                         Eigen::Ref<const Eigen::VectorX<bool>> J) {
//     using guanaqo::as_span;
//     using SpMat      = Eigen::SparseMatrix<real_t, 0, index_t>;
//     index_t n_constr = ocp.num_constraints(),
//             n_dyn    = ocp.num_dynamics_constraints();
//     // Build quadratic program and standard KKT system for the OCP.
//     auto qp   = ko::LinearOCPSparseQP::build(ocp);
//     auto kkt  = qp.build_kkt(S, as_span(Σ), as_span(J));
//     auto &qpA = qp.A_sparsity, &qpQ = qp.Q_sparsity, &qpK = kkt.sparsity;
//     SpMat Q = Eigen::Map<const SpMat>(
//         qpQ.rows, qpQ.cols, qpQ.nnz(), qpQ.outer_ptr.data(),
//         qpQ.inner_idx.data(), qp.Q_values.data(), nullptr);
//     SpMat G(n_constr, qpA.cols), M(n_dyn, qpA.cols), K(qpK.rows, qpK.cols);
//     std::vector<Eigen::Triplet<real_t>> triplets_G, triplets_M, triplets_K;
//     for (index_t c = 0; c < qpA.cols; ++c)
//         for (index_t i = qpA.outer_ptr[c]; i < qpA.outer_ptr[c + 1]; ++i)
//             if (index_t r = qpA.inner_idx[i]; r < n_dyn) // top rows
//                 triplets_M.emplace_back(r, c, qp.A_values[i]);
//             else // bottom rows
//                 triplets_G.emplace_back(r - n_dyn, c, qp.A_values[i]);
//     for (index_t c = 0; c < qpK.cols; ++c)
//         for (index_t i = qpK.outer_ptr[c]; i < qpK.outer_ptr[c + 1]; ++i) {
//             index_t r = qpK.inner_idx[i];
//             if (r >= c)
//                 triplets_K.emplace_back(r, c, kkt.values[i]);
//             if (r > c)
//                 triplets_K.emplace_back(c, r, kkt.values[i]);
//         }
//     G.setFromTriplets(triplets_G.begin(), triplets_G.end());
//     M.setFromTriplets(triplets_M.begin(), triplets_M.end());
//     K.setFromTriplets(triplets_K.begin(), triplets_K.end());
//     return std::tuple{std::move(Q), std::move(G), std::move(M), std::move(K)};
// }

const int n_threads = 8; // TODO
TEST(CyclicUtil, heapIndex) {
    using namespace koqkatoo::ocp;

    KOQKATOO_OMP_IF(omp_set_num_threads(n_threads));
    koqkatoo::pool_set_num_threads(n_threads);
    koqkatoo::fork_set_num_threads(n_threads);
    GUANAQO_IF_ITT(koqkatoo::foreach_thread([](index_t i, index_t) {
        __itt_thread_set_name(std::format("OMP({})", i).c_str());
    }));

    std::mt19937 rng{54321};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    OCPDim dim{.N_horiz = 31, .nx = 40, .nu = 30, .ny = 10, .ny_N = 10};
    auto [N, nx, nu, ny, ny_N] = dim;

    // Generate some random OCP matrices
    auto ocp = ko::generate_random_ocp(dim, 12345);
    CyclicOCPSolver solver{ocp.dim};

    // Instantiate the OCP KKT solver.
    index_t n_var = ocp.num_variables(), n_constr = ocp.num_constraints(),
            n_dyn_constr = ocp.num_dynamics_constraints();

    // Generate some random optimization solver data.
    Eigen::VectorX<bool> J(n_constr), // Active set.
        J0(n_constr), J1(n_constr);   // Active set for initialization.
    EVec Σ(n_constr),                 // ALM penalty factors
        ŷ(n_constr);                  //  & corresponding Lagrange multipliers.
    EVec x(n_var),
        grad(n_var);      // Current iterate and cost gradient.
    EVec b(n_dyn_constr), // Dynamics constraints right-hand side
        λ(n_dyn_constr);  //  & corresponding Lagrange multipliers.

    real_t S = std::exp2(nrml(rng)) * 1e-2; // primal regularization
    std::ranges::generate(J, [&] { return bernoulli(rng); });
    std::ranges::generate(J0, [&] { return bernoulli(rng); });
    std::ranges::generate(J1, [&] { return bernoulli(rng); });
    std::ranges::generate(Σ, [&] { return std::exp2(nrml(rng)); });
    std::ranges::generate(ŷ, [&] { return nrml(rng); });
    std::ranges::generate(x, [&] { return nrml(rng); });
    std::ranges::generate(grad, [&] { return nrml(rng); });
    std::ranges::generate(b, [&] { return nrml(rng); });
    std::ranges::generate(λ, [&] { return nrml(rng); });

    decltype(solver)::real_matrix Σb{{.depth = N + 1, .rows = ny, .cols = 1}};
    decltype(solver)::bool_matrix Jb{{.depth = N + 1, .rows = ny, .cols = 1}};
    solver.pack_constr<const real_t, real_t>(guanaqo::as_span(Σ), Σb);
    solver.pack_constr<const bool, bool>(guanaqo::as_span(J), Jb);

    solver.initialize(ocp);
    solver.compute_Ψ(S, Σb, Jb);

    std::vector<Eigen::Triplet<real_t>> triplets_Ψ;
    for (index_t j = 0; j < solver.vl; ++j) {
        for (index_t i = 0; i < solver.n; ++i) {
            const index_t bc = i + j * solver.vstride;
            const index_t d  = bc * solver.dim.nx;
            auto Ψd          = solver.LΨd.batch(solver.get_batch_index(i));
            bool transpose   = (i & 1) == 0;
            auto Ψs =
                solver.n == 1  ? solver.Ub.batch(0)
                : (i & 3) == 0 ? solver.Ub.batch(solver.get_batch_index(i + 1))
                : (i & 3) == 1 ? solver.Ut.batch(solver.get_batch_index(i))
                : (i & 3) == 2 ? solver.Ut.batch(solver.get_batch_index(i + 1))
                               : solver.Ub.batch(solver.get_batch_index(i));
            for (index_t c = 0; c < solver.dim.nx; ++c) {
                for (index_t r = c; r < solver.dim.nx; ++r)
                    triplets_Ψ.emplace_back(d + r, d + c, Ψd(j, r, c));
                if (bc < solver.dim.N_horiz && solver.n > 1) {
                    for (index_t r = 0; r < solver.dim.nx; ++r)
                        triplets_Ψ.emplace_back(d + r + solver.dim.nx, d + c,
                                                transpose ? Ψs(j, c, r)
                                                          : Ψs(j, r, c));
                }
            }
        }
    }
    using SpMat = Eigen::SparseMatrix<real_t, 0, index_t>;
    SpMat Ψ((N + 1) * nx, (N + 1) * nx);
    Ψ.setFromTriplets(triplets_Ψ.begin(), triplets_Ψ.end());
    using LLT = Eigen::SimplicialLLT<SpMat, Eigen::Lower,
                                     Eigen::NaturalOrdering<index_t>>;
    LLT cholΨ(Ψ);
    EVec Δλ     = λ;
    EVec Δλ_ref = cholΨ.solve(Δλ);

    if (Ψ.rows() < 1024) {
        auto Ψ_dense = Ψ.toDense();
#if DO_PRINT
        guanaqo::print_python(std::cout << "Ψ = \\\n",
                              guanaqo::as_view(Ψ_dense));
#endif
        std::cout << "κ(Ψ) = " << (1 / Ψ_dense.llt().rcond()) << std::endl;
    }

    decltype(solver)::real_matrix Δλb{{.depth = N + 1, .rows = nx, .cols = 1}};
    for (index_t i = 0; i < 1000; ++i) {
        solver.pack_dyn<const real_t, real_t>(guanaqo::as_span(Δλ), Δλb);
        solver.compute_Ψ(S, Σb, Jb);
        solver.factor_Ψ();
        solver.solve_Ψ(Δλb);
    }

#if GUANAQO_WITH_TRACING
    guanaqo::trace_logger.reset();
#endif

#if DO_PRINT
    std::cout << "A = [\n";
    for (index_t k = 0; k < N + 1; ++k) {
        const auto [j, i] = std::div(k, solver.vstride);
        index_t hi        = solver.get_batch_index(i);
        guanaqo::print_python(std::cout, solver.LΨd.batch(hi)(j), ",\n", false);
    }
    std::cout << "]\n";
    std::cout << "B = [\n";
    for (index_t k = 0; k < N + 1; ++k) {
        const auto [j, i] = std::div(k, solver.vstride);
        index_t hi        = solver.get_batch_index(i);
        guanaqo::print_python(std::cout,
                              solver.LΨU.bottom_rows(2 * nx).batch(hi)(j),
                              ",\n", false);
    }
    std::cout << "]\n";
#endif
    auto t0 = std::chrono::steady_clock::now();
    solver.compute_Ψ(S, Σb, Jb);
    solver.factor_Ψ();
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "factor: " << std::chrono::duration<double>(t1 - t0).count()
              << std::endl;

#if DO_PRINT
    std::cout << "A_fac = [\n";
    for (index_t k = 0; k < N + 1; ++k) {
        const auto [j, i] = std::div(k, solver.vstride);
        index_t hi        = solver.get_batch_index(i);
        guanaqo::print_python(std::cout, solver.LΨd.batch(hi)(j), ",\n", false);
    }
    std::cout << "]\n";
    std::cout << "B_fac = [\n";
    for (index_t k = 0; k < N + 1; ++k) {
        const auto [j, i] = std::div(k, solver.vstride);
        index_t hi        = solver.get_batch_index(i);
        guanaqo::print_python(std::cout,
                              solver.LΨU.bottom_rows(2 * nx).batch(hi)(j),
                              ",\n", false);
    }
    std::cout << "]\n";
#endif

    solver.pack_dyn<const real_t, real_t>(guanaqo::as_span(Δλ), Δλb);
    auto t2 = std::chrono::steady_clock::now();
    solver.solve_Ψ(Δλb);
    auto t3 = std::chrono::steady_clock::now();
    std::cout << "solve: " << std::chrono::duration<double>(t3 - t2).count()
              << std::endl;
    solver.unpack_dyn<const real_t, real_t>(Δλb, guanaqo::as_span(Δλ));
    EXPECT_THAT(Δλ, EigenAlmostEqual(Δλ_ref, ε * 1e7));
    std::cout << (Δλ - Δλ_ref).template lpNorm<Eigen::Infinity>() << std::endl;

#if GUANAQO_WITH_TRACING
    {
        const auto [N, nx, nu, ny, ny_N] = ocp.dim;
        std::string name                 = std::format("factor_cyclic.csv");
        std::filesystem::path out_dir{"traces"};
        out_dir /= *koqkatoo_commit_hash ? koqkatoo_commit_hash : "unknown";
        out_dir /= KOQKATOO_MKL_IF_ELSE("mkl", "openblas");
        out_dir /= std::format("nx={}-nu={}-ny={}-N={}-thr={}-vl={}{}", nx, nu,
                               ny, N, n_threads, VL, "-pcg");
        std::filesystem::create_directories(out_dir);
        std::ofstream csv{out_dir / name};
        koqkatoo::TraceLogger::write_column_headings(csv) << '\n';
        for (const auto &log : guanaqo::trace_logger.get_logs())
            csv << log << '\n';
        std::cout << out_dir << std::endl;
    }
#endif
}
