// TODO: move to src

#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/rotate.hpp>
#include <koqkatoo/linalg/small-potrf.hpp>
#include <koqkatoo/matrix-view.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/trace.hpp>
#include <guanaqo/eigen/span.hpp>
#include <guanaqo/eigen/view.hpp>
#include <guanaqo/print.hpp>

#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <print>
#include <type_traits>

#define PRINTLN(...)
#define USE_PCG 1
#define USE_JACOBI_PREC 1
#define DO_PRINT 0

namespace ko   = koqkatoo::ocp;
namespace stdx = std::experimental;
using koqkatoo::index_t;
using koqkatoo::real_t;
using koqkatoo::RealMatrixView;

index_t get_depth(index_t n) {
    assert(n > 0);
    auto un = static_cast<std::make_unsigned_t<index_t>>(n);
    return static_cast<index_t>(std::bit_width(un - 1));
}

index_t get_layer(index_t i) {
    assert(i > 0);
    auto ui = static_cast<std::make_unsigned_t<index_t>>(i);
    return static_cast<index_t>(std::countr_zero(ui));
}

index_t get_index_in_layer(index_t i) {
    auto l = get_layer(i);
    return i >> (l + 1);
}

index_t get_heap_index(index_t i, index_t n) {
    assert(i < n);
    if (i == 0)
        return n - 1;
    auto l  = get_layer(i);
    auto il = get_index_in_layer(i);
    auto d  = get_depth(n);
    return il + (1 << d) - (1 << (d - l));
}

template <index_t VL>
void solve_cyclic(const koqkatoo::ocp::LinearOCPStorage &ocp, real_t S,
                  std::span<const real_t> Σ, std::span<const bool> J,
                  std::span<const real_t> x, std::span<const real_t> grad,
                  std::span<const real_t> λ, std::span<const real_t> b,
                  std::span<const real_t> ŷ, std::span<real_t> Mxb,
                  std::span<real_t> Mᵀλ, std::span<real_t> Aᵀŷ,
                  std::span<real_t> d, std::span<real_t> Δλ,
                  std::span<real_t> MᵀΔλ) {
    using std::isfinite;
    using namespace koqkatoo;
    using namespace koqkatoo::linalg;
    using namespace koqkatoo::linalg::compact;
    constexpr auto with_index_t = guanaqo::with_index_type<index_t>;
    using compact_blas = CompactBLAS<stdx::simd_abi::deduce_t<real_t, VL>>;
    using scalar_blas  = CompactBLAS<stdx::simd_abi::scalar>;
    using matrix       = compact_blas::matrix;
    using bool_matrix  = compact_blas::bool_matrix;
    const auto [N, nx, nu, ny, ny_N] = ocp.dim;
    const auto nxu                   = nx + nu;
    constexpr auto be                = PreferredBackend::MKLScalarBatched;

    const index_t n   = (N + 1) / VL;
    const index_t lgn = get_depth(n), lgvl = get_depth(VL);
    const auto vstride = index_t{1} << lgn;
    std::cout << lgn << std::endl;
    for (index_t l = 0; l < lgn; ++l) {
        for (index_t i = 1 << l; i < n; i += (1 << (l + 1))) {
            for (index_t j = 0; j < VL; ++j)
                std::cout << std::setw(2) << (i + j * vstride) << "  ";
            std::cout << "  ";
        }
        for (index_t i = 1 << l; i < n; i += (1 << (l + 1)))
            std::cout << "  " << std::setw(2) << i << "->" << std::setw(2)
                      << get_heap_index(i, n);
        std::cout << "\n";
    }

    matrix AB{{.depth = N + 1, .rows = nx, .cols = nx + nu}},
        CD{{.depth = N + 1, .rows = ny, .cols = nx + nu}},
        H{{.depth = N + 1, .rows = nx + nu, .cols = nx + nu}},
        Σb{{.depth = N + 1, .rows = ny, .cols = 1}},
        xb{{.depth = N + 1, .rows = nx + nu, .cols = 1}},
        bb{{.depth = N + 1, .rows = nx, .cols = 1}},
        Mxbb{{.depth = N + 1, .rows = nx, .cols = 1}},
        db{{.depth = N + 1, .rows = nx + nu, .cols = 1}},
        Δλb{{.depth = N + 1, .rows = nx, .cols = 1}},
        Aᵀŷb{{.depth = N + 1, .rows = nx + nu, .cols = 1}},
        MᵀΔλb{{.depth = N + 1, .rows = nx + nu, .cols = 1}},
        Mᵀλb{{.depth = N + 1, .rows = nx + nu, .cols = 1}},
        λb{{.depth = N + 1, .rows = nx, .cols = 1}},
        ŷb{{.depth = N + 1, .rows = ny, .cols = 1}};
    bool_matrix Jb{{.depth = N + 1, .rows = ny, .cols = 1}};
    auto Σv    = as_view(guanaqo::as_eigen(Σ), with_index_t),
         gradv = as_view(guanaqo::as_eigen(grad), with_index_t),
         xv    = as_view(guanaqo::as_eigen(x), with_index_t),
         λv    = as_view(guanaqo::as_eigen(λ), with_index_t),
         ŷv    = as_view(guanaqo::as_eigen(ŷ), with_index_t),
         bv    = as_view(guanaqo::as_eigen(b), with_index_t);
    auto Mxbv  = as_view(guanaqo::as_eigen(Mxb), with_index_t),
         dv    = as_view(guanaqo::as_eigen(d), with_index_t),
         Δλv   = as_view(guanaqo::as_eigen(Δλ), with_index_t),
         Aᵀŷv  = as_view(guanaqo::as_eigen(Aᵀŷ), with_index_t),
         Mᵀλv  = as_view(guanaqo::as_eigen(Mᵀλ), with_index_t),
         MᵀΔλv = as_view(guanaqo::as_eigen(MᵀΔλ), with_index_t);
    auto Jv    = as_view(guanaqo::as_eigen(J), with_index_t);

    for (index_t i = 0; i < n; ++i) {
        auto hi = get_heap_index(i, n);
        for (index_t vi = 0; vi < VL; ++vi) {
            auto k = i + vi * vstride;
            // PRINTLN("  {} -> {}({})", k, hi, vi);
            if (k < N) {
                H.batch(hi)(vi)  = ocp.H(k);
                CD.batch(hi)(vi) = ocp.CD(k);
                AB.batch(hi)(vi) = ocp.AB(k);
                Σb.batch(hi)(vi) = Σv.middle_rows(k * ny, ny);
                Jb.batch(hi)(vi) = Jv.middle_rows(k * ny, ny);
                xb.batch(hi)(vi) = xv.middle_rows(k * nxu, nx + nu);
                bb.batch(hi)(vi) = bv.middle_rows(k * nx, nx);
                db.batch(hi)(vi) = gradv.middle_rows(k * nxu, nxu);
                λb.batch(hi)(vi) = λv.middle_rows(k * nx, nx);
                ŷb.batch(hi)(vi) = ŷv.middle_rows(k * ny, ny);
            } else if (k == N) {
                H.batch(hi)(vi).bottom_left(nu, nx).set_constant(0);
                H.batch(hi)(vi).right_cols(nu).set_constant(0);
                H.batch(hi)(vi).bottom_right(nu, nu).set_diagonal(1);
                H.batch(hi)(vi).top_left(nx, nx) = ocp.Q(k);
                CD.batch(hi)(vi).right_cols(nu).set_constant(0);
                CD.batch(hi)(vi).bottom_left(ny - ny_N, nx).set_constant(0);
                CD.batch(hi)(vi).top_left(ny_N, nx) = ocp.C(k);
                AB.batch(hi)(vi).set_constant(0);
                Σb.batch(hi)(vi).top_rows(ny_N) = Σv.middle_rows(k * ny, ny_N);
                Jb.batch(hi)(vi).top_rows(ny_N) = Jv.middle_rows(k * ny, ny_N);
                xb.batch(hi)(vi).top_rows(nx)   = xv.middle_rows(k * nxu, nx);
                bb.batch(hi)(vi)                = bv.middle_rows(k * nx, nx);
                db.batch(hi)(vi).top_rows(nx) = gradv.middle_rows(k * nxu, nx);
                λb.batch(hi)(vi)              = λv.middle_rows(k * nx, nx);
                ŷb.batch(hi)(vi).top_rows(ny_N) = ŷv.middle_rows(k * ny, ny_N);
            } else {
                H.batch(hi)(vi).set_constant(0);
                H.batch(hi)(vi).set_diagonal(1);
                CD.batch(hi)(vi).set_constant(0);
                AB.batch(hi)(vi).set_constant(0);
                Σb.batch(hi)(vi).set_constant(1e-99);
                Jb.batch(hi)(vi).set_constant(false);
            }
        }
    }

    matrix LHV{{.depth = N + 1, .rows = nx + nu + nx, .cols = nx + nu}},
        Wᵀ{{.depth = N + 1, .rows = nx + nu, .cols = nx}},
        VVᵀ{{.depth = VL, .rows = nx, .cols = nx}},
        LΨU{{.depth = N + 1, .rows = 3 * nx, .cols = nx}};
    auto LH = LHV.top_rows(nxu), V = LHV.bottom_rows(nx),
         LΨd = LΨU.top_rows(nx), U = LΨU.bottom_rows(nx),
         LΨs = LΨU.middle_rows(nx, nx);
    scalar_blas::matrix LΨU_scal{{.depth = VL, .rows = 3 * nx, .cols = nx}},
        Δλ_scal{{.depth = VL, .rows = nx, .cols = 1}};
    auto LΨd_scal = LΨU_scal.top_rows(nx), U_scal = LΨU_scal.bottom_rows(nx),
         LΨs_scal = LΨU_scal.middle_rows(nx, nx);

    KOQKATOO_OMP(parallel) {

        // Compute Aᵀŷ
    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("mat_vec_transpose_constr", i);
        auto hi = get_heap_index(i, n);
        compact_blas::xgemv_T(CD.batch(hi), ŷb.batch(hi), Aᵀŷb.batch(hi), be);
    }

    // Compute Mx - b
    KOQKATOO_OMP(single)
    compact_blas::xsub_copy(Mxbb, xb.top_rows(nx), bb);
    // TODO: move into loop
    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("residual_dynamics_constr", i);
        auto hi = get_heap_index(i, n);
        if (i + 1 < n) {
            auto hi_next = get_heap_index(i + 1, n);
            compact_blas::xgemv_sub(AB.batch(hi), xb.batch(hi),
                                    Mxbb.batch(hi_next), be);
        } else {
            // Last batch is special since we cross batch boundaries
            auto hi_next = get_heap_index(i - vstride + 1, n);
            compact_blas::xgemv_sub_shift(AB.batch(hi), xb.batch(hi),
                                          Mxbb.batch(hi_next));
        }
    }

    KOQKATOO_OMP(single)
    compact_blas::xcopy(Mxbb, Δλb);

    // Compute Mᵀλ and initialize right-hand side
    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("mat_vec_transpose_dynamics_constr", i);
        const auto hi               = get_heap_index(i, n);
        Mᵀλb.batch(hi).top_rows(nx) = λb.batch(hi);
        Mᵀλb.batch(hi).bottom_rows(nu).set_constant(0);
        if (i + 1 < n) {
            const auto hi_next = get_heap_index(i + 1, n);
            compact_blas::xgemv_T_sub(AB.batch(hi), λb.batch(hi_next),
                                      Mᵀλb.batch(hi), be);
        } else {
            const auto hi_next = get_heap_index(i - vstride + 1, n);
            compact_blas::xgemv_T_sub_shift(AB.batch(hi), λb.batch(hi_next),
                                            Mᵀλb.batch(hi));
        }
        // TODO: optimize
        for (index_t k = 0; k < nxu; ++k)
            for (index_t j = 0; j < VL; ++j)
                db.batch(hi)(j, k, 0) = -db.batch(hi)(j, k, 0) -
                                        Mᵀλb.batch(hi)(j, k, 0) -
                                        Aᵀŷb.batch(hi)(j, k, 0);
    }

    } // end parallel

#if USE_PCG
    using simd          = typename compact_blas::simd;
    using real_view     = typename compact_blas::single_batch_view;
    using mut_real_view = typename compact_blas::mut_single_batch_view;
    const auto mul_A    = [&](real_view p, mut_real_view Ap, mut_real_view w,
                           real_view L, real_view B) {
        static constexpr auto algn = stdx::vector_aligned;
        compact_blas::xcopy(p, Ap);
        compact_blas::xtrmv_T(L, Ap, be);
        compact_blas::xtrmv(L, Ap, be);
        for (index_t j = 0; j < B.cols(); ++j) {
            simd wj_accum{};
            simd zj{&p(0, j, 0), algn};
            zj = micro_kernels::shiftr<1>(zj);
            for (index_t i = 0; i < B.rows(); ++i) {
                simd zi{&p(0, i, 0), algn};
                simd Bij{&B(0, i, j), algn};
                wj_accum += Bij * micro_kernels::shiftl<1>(zi);
                simd wi = j == 0 ? simd{} : simd{&w(0, i, 0), algn};
                Bij     = micro_kernels::shiftr<1>(Bij); // TODO: rotr?
                wi += Bij * zj;
                wi.copy_to(&w(0, i, 0), algn);
            }
            simd wj{&w(0, j, 0), algn};
            wj += wj_accum;
            wj.copy_to(&w(0, j, 0), algn);
        }
        simd pAp_accum{};
        for (index_t i = 0; i < w.rows(); ++i) {
            simd Api{&Ap(0, i, 0), algn}, wi{&w(0, i, 0), algn},
                pi{&p(0, i, 0), algn};
            Api += wi;
            pAp_accum += Api * pi;
            Api.copy_to(&Ap(0, i, 0), algn);
        }
        return reduce(pAp_accum);
    };
    const auto mul_precond = [&](real_view r, mut_real_view z, mut_real_view w,
                                 real_view L, real_view B) {
        compact_blas::xcopy(r, z);
        compact_blas::xtrsv_LNN(L, z, be);
        compact_blas::xtrsv_LTN(L, z, be);
#if USE_JACOBI_PREC
        std::ignore = w;
        std::ignore = B;
        return compact_blas::xdot(r, z);
#else
        static constexpr auto algn = stdx::vector_aligned;
        for (index_t j = 0; j < B.cols(); ++j) {
            simd wj_accum{};
            simd zj{&z(0, j, 0), algn};
            zj = micro_kernels::shiftr<1>(zj);
            for (index_t i = 0; i < B.rows(); ++i) {
                simd zi{&z(0, i, 0), algn};
                simd Bij{&B(0, i, j), algn};
                wj_accum += Bij * micro_kernels::shiftl<1>(zi);
                simd wi = j == 0 ? simd{} : simd{&w(0, i, 0), algn};
                Bij     = micro_kernels::shiftr<1>(Bij); // TODO: rotr?
                wi += Bij * zj;
                wi.copy_to(&w(0, i, 0), algn);
            }
            simd wj{&w(0, j, 0), algn};
            wj += wj_accum;
            wj.copy_to(&w(0, j, 0), algn);
        }
        compact_blas::xtrsv_LNN(L, w, be);
        compact_blas::xtrsv_LTN(L, w, be);
        simd rz_accum{};
        for (index_t i = 0; i < w.rows(); ++i) {
            simd zi{&z(0, i, 0), algn}, wi{&w(0, i, 0), algn},
                ri{&r(0, i, 0), algn};
            zi -= wi;
            rz_accum += zi * ri;
            zi.copy_to(&z(0, i, 0), algn);
        }
        return reduce(rz_accum);
#endif
    };
    matrix work_pcg{{.depth = VL, .rows = nx, .cols = 5}};
#endif

#if KOQKATOO_WITH_TRACING
    koqkatoo::trace_logger.reset();
#endif

    KOQKATOO_OMP(parallel) {

    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("factor prep", i);
        const auto hi = get_heap_index(i, n);
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
        compact_blas::xtrmm_RLNN_neg(V.batch(hi), Wᵀ.batch(hi), LΨs.batch(hi),
                                     be);
        // Compute WWᵀ
        // TODO: exploit trapezoidal shape of Wᵀ
        compact_blas::xsyrk_T(Wᵀ.batch(hi), LΨd.batch(hi), be);
    }

    // Compute Ψ
    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("build Ψ", i);
        const auto hi = get_heap_index(i, n);
        if (i + 1 < n) {
            const auto hi_next = get_heap_index(i + 1, n);
            // Compute VVᵀ and add to Ψd
            compact_blas::xsyrk_add(V.batch(hi), LΨd.batch(hi_next), be);
        } else {
            // Last batch is special since we cross batch boundaries
            const auto hi_next = get_heap_index(i - vstride + 1, n);
            compact_blas::xsyrk_add_shift(V.batch(hi), LΨd.batch(hi_next));
        }
    }

    for (index_t l = 0; l < lgn; ++l) {
        const auto offset = index_t{1} << l;
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("factor Ψ", i);
            const auto hi      = get_heap_index(i, n),
                       hi_prev = get_heap_index(i - offset, n);
            auto B_prev = LΨs.batch(hi_prev), Y = LΨs.batch(hi);
            // Copy previous Bᵀ to U and clear
            compact_blas::xcopy_T(B_prev, U.batch(hi));
            // L = chol(A), Y = B L⁻ᵀ, U = B L⁻ᵀ
            compact_blas::xpotrf(LΨU.batch(hi), be);
            // Update previous diagonal and subdiagonal blocks
            // A -= UUᵀ, B = -YUᵀ
            compact_blas::xsyrk_sub(U.batch(hi), LΨd.batch(hi_prev), be);
            compact_blas::xgemm_NT_neg(Y, U.batch(hi), B_prev, be);
            // TODO: is there a way to merge these operations?
        }
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("factor Ψ YY", i);
            const auto hi = get_heap_index(i, n);
            auto Y        = LΨs.batch(hi);
            // Update next diagonal block
            if (i + offset < n) {
                const auto hi_next = get_heap_index(i + offset, n);
                compact_blas::xsyrk_sub(Y, LΨd.batch(hi_next), be);
            } else {
                // Last batch is special since we cross batch boundaries
                const auto hi_next = get_heap_index(i + offset - vstride, n);
                compact_blas::xsyrk_sub_shift(Y, LΨd.batch(hi_next));
            }
        }
    }
#if DO_PRINT
    KOQKATOO_OMP(single) {
        std::cout << "A_fac = [\n";
        for (index_t j = 0; j < VL; ++j) {
            guanaqo::print_python(std::cout, LΨd.batch(n - 1)(j), ",\n", false);
        }
        std::cout << "]\n";
        std::cout << "B_fac = [\n";
        for (index_t j = 0; j < VL - 1; ++j) {
            guanaqo::print_python(std::cout, LΨs.batch(n - 1)(j), ",\n", false);
        }
        std::cout << "]\n";
    }
#endif

#if USE_PCG
    // Factor diagonal blocks of preconditioner
    KOQKATOO_OMP(single) {
        KOQKATOO_TRACE("factor Ψ", 0);
        compact_blas::xpotrf(LΨd.batch(n - 1), be);
    }
#else
    KOQKATOO_OMP(single) {
        KOQKATOO_TRACE("unpack Ψ", 0);
        compact_blas::unpack_L(LΨU.batch(n - 1), LΨU_scal);
    }
    // TODO: use lgvl - 1 to stop once we're left with two diagonal blocks, then
    // handle that case manually
    for (index_t l = 0; l < lgvl; ++l) {
        const auto offset = index_t{1} << l;
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < VL; i += 2 * offset) {
            KOQKATOO_TRACE("factor Ψ scalar", i);
            auto B_prev = LΨs_scal.batch(i - offset), Y = LΨs_scal.batch(i);
            // Copy previous Bᵀ to U and clear
            scalar_blas::xcopy_T(B_prev, U_scal.batch(i));
            // L = chol(A), Y = B L⁻ᵀ, U = B L⁻ᵀ
            // TODO: we don't need Y in the final stage
            if ((0)) {
                scalar_blas::xpotrf(LΨU_scal.batch(i), be);
            } else {
                auto LΨi = LΨU_scal(i);
                linalg::small_potrf(LΨi.data, LΨi.outer_stride, LΨi.rows,
                                    LΨi.cols);
            }
            // Update previous diagonal and subdiagonal blocks
            // A -= UUᵀ, B = -YUᵀ
            scalar_blas::xsyrk_sub(U_scal.batch(i), LΨd_scal.batch(i - offset),
                                   be);
            scalar_blas::xgemm_NT_neg(Y, U_scal.batch(i), B_prev, be);
            // TODO: is there a way to merge these operations?
        }
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < VL - offset; i += 2 * offset) {
            KOQKATOO_TRACE("factor Ψ YY scalar", i);
            // Update next diagonal block
            auto Y = LΨs_scal.batch(i);
            scalar_blas::xsyrk_sub(Y, LΨd_scal.batch(i + offset), be);
        }
    }
    KOQKATOO_OMP(single) {
        KOQKATOO_TRACE("factor Ψ scalar", 0);
        if ((0)) {
            scalar_blas::xpotrf(LΨd_scal.batch(0), be);
        } else {
            auto LΨi = LΨd_scal(0);
            linalg::small_potrf(LΨi.data, LΨi.outer_stride, LΨi.rows, LΨi.cols);
        }
    }
#endif

    } // end parallel

    KOQKATOO_OMP(parallel) {

        // Solve Hv = -g
    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("solve Hv=g", i);
        auto hi = get_heap_index(i, n);
        PRINTLN("solve Hv({}) [{}]", i, hi);
        // Solve Lᴴ vʹ = g                                              (d ← vʹ)
        compact_blas::xtrsv_LNN(LH.batch(hi), db.batch(hi), be);
        // Solve Lᴴ⁻ᵀ v = vʹ                                            (λ ← Ev)
        PRINTLN("  add Wv({}) [{}]", i, hi);
        compact_blas::xgemv_T_add(Wᵀ.batch(hi), db.batch(hi), Δλb.batch(hi),
                                  be);
    }
    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("eval rhs Ψ", i);
        auto hi = get_heap_index(i, n);
        if (i + 1 < n) {
            auto hi_next = get_heap_index(i + 1, n);
            PRINTLN("  sub Vv({}) [{}]", i + 1, hi_next);
            compact_blas::xgemv_sub(V.batch(hi), db.batch(hi),
                                    Δλb.batch(hi_next), be);
        } else {
            // Last batch is special since we cross batch boundaries
            auto hi_next = get_heap_index(i - vstride + 1, n);
            compact_blas::xgemv_sub_shift(V.batch(hi), db.batch(hi),
                                          Δλb.batch(hi_next));
        }
    }

#if 0
    KOQKATOO_OMP(single) {
    for (index_t j = 0; j < VL; ++j)
        LΨU.batch(n - 1)(j) = LΨU_scal(j);
    // for (index_t k = N + 1; k-- > 0;) {
    std::cout << "V = [\n";
    for (index_t k = 0; k < N + 1; ++k) {
        const auto [j, i] = std::div(k, vstride);
        index_t hi        = get_heap_index(i, n);
        guanaqo::print_python(std::cout, V.batch(hi)(j), ",\n", false);
    }
    std::cout << "]\n";
    std::cout << "A_fac = [\n";
    for (index_t k = 0; k < N + 1; ++k) {
        const auto [j, i] = std::div(k, vstride);
        index_t hi        = get_heap_index(i, n);
        guanaqo::print_python(std::cout, LΨd.batch(hi)(j), ",\n", false);
    }
    std::cout << "]\n";
    std::cout << "B_fac = [\n";
    for (index_t k = 0; k < N; ++k) {
        const auto [j, i] = std::div(k, vstride);
        index_t hi        = get_heap_index(i, n);
        guanaqo::print_python(std::cout, LΨs.batch(hi)(j), ",\n", false);
    }
    std::cout << "]\n";
    std::cout << "U_fac = [\n";
    for (index_t k = 0; k < N; ++k) {
        const auto [j, i] = std::div(k, vstride);
        index_t hi        = get_heap_index(i, n);
        guanaqo::print_python(std::cout, U.batch(hi)(j), ",\n", false);
    }
    std::cout << "]\n";
    std::cout << "v = [\n";
    for (index_t k = 0; k < N + 1; ++k) {
        const auto [j, i] = std::div(k, vstride);
        index_t hi        = get_heap_index(i, n);
        guanaqo::print_python(std::cout, db.batch(hi)(j), ",\n", true);
    }
    std::cout << "]\n";
    std::cout << "rhs_psi = [\n";
    for (index_t k = 0; k < N + 1; ++k) {
        const auto [j, i] = std::div(k, vstride);
        index_t hi        = get_heap_index(i, n);
        guanaqo::print_python(std::cout, Δλb.batch(hi)(j), ",\n", true);
    }
    std::cout << "]\n";
    }
#endif

    // Forward pass Ψ (batched)
    for (index_t l = 0; l < lgn; ++l) {
        const auto offset = index_t{1} << l;
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ fwd", i);
            const auto hi      = get_heap_index(i, n),
                       hi_prev = get_heap_index(i - offset, n);
            PRINTLN("solve rhs fwd b(2i+1 = {}) -> [{}]", i, hi);
            // L⁻¹ b
            compact_blas::xtrsv_LNN(LΨd.batch(hi), Δλb.batch(hi), be);
            // Update previous even block
            PRINTLN("  update rhs fwd b(2i   = {}) -> [{}]", i - offset,
                    hi_prev);
            compact_blas::xgemv_sub(U.batch(hi), Δλb.batch(hi),
                                    Δλb.batch(hi_prev), be);
        }
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ fwd 2", i);
            const auto hi = get_heap_index(i, n);
            auto Y        = LΨs.batch(hi);
            // Update next even block
            if (i + offset < n) {
                const auto hi_next = get_heap_index(i + offset, n);
                PRINTLN("  update rhs fwd b(2i+2 = {}) -> [{}]", i + offset,
                        hi_next);
                compact_blas::xgemv_sub(Y, Δλb.batch(hi), Δλb.batch(hi_next),
                                        be);
            } else {
                // Last batch is special since we cross batch boundaries
                const auto hi_next = get_heap_index(i + offset - vstride, n);
                PRINTLN("  update rhs fwd b(2i+2 = {}) -> [{}]Δ", i + offset,
                        hi_next);
                compact_blas::xgemv_sub_shift(Y, Δλb.batch(hi),
                                              Δλb.batch(hi_next));
            }
        }
    }

#if DO_PRINT
    KOQKATOO_OMP(single) {
        std::cout << "λ_rhs = [\n";
        for (index_t j = 0; j < VL; ++j) {
            guanaqo::print_python(std::cout, Δλb.batch(n - 1)(j), ",\n", false);
        }
        std::cout << "]\n";
    }
#endif

#if USE_PCG
    KOQKATOO_OMP(single) {
        KOQKATOO_TRACE("solve Ψ pcg", 0);
        auto r  = work_pcg.batch(0).middle_cols(0, 1),
             z  = work_pcg.batch(0).middle_cols(1, 1),
             p  = work_pcg.batch(0).middle_cols(2, 1),
             Ap = work_pcg.batch(0).middle_cols(3, 1),
             w  = work_pcg.batch(0).middle_cols(4, 1);
        auto x  = Δλb.batch(n - 1);
        auto A = LΨd.batch(n - 1), B = LΨs.batch(n - 1);
        compact_blas::xcopy(x, r);
        compact_blas::xfill(0, x);
        real_t rᵀz = mul_precond(r, z, w, A, B);

#if DO_PRINT
        std::cout << "prec_rhs = [\n";
        for (index_t j = 0; j < VL; ++j) {
            guanaqo::print_python(std::cout, z(j), ",\n", false);
        }
        std::cout << "]\n";
#endif

        compact_blas::xcopy(z, p);
        for (index_t it = 0; it < 10; ++it) {
            real_t pᵀAp = mul_A(p, Ap, w, A, B);
            real_t α    = rᵀz / pᵀAp;
            compact_blas::xaxpy(+α, p, x);
            compact_blas::xaxpy(-α, Ap, r);
            real_t r2 = compact_blas::xdot(r, r);
            PRINTLN("# {}: {}", it, std::sqrt(r2));
            constexpr real_t ε = std::numeric_limits<real_t>::epsilon();
            if (r2 < ε * ε)
                break;
            real_t rᵀz_new = mul_precond(r, z, w, A, B);
            real_t β       = rᵀz_new / rᵀz;
            compact_blas::xaxpby(1, z, β, p);
            rᵀz = rᵀz_new;
        }
    }
#else
    // Forward pass Ψ (scalar)
    KOQKATOO_OMP(single) {
        KOQKATOO_TRACE("unpack Δλ", 0);
        compact_blas::unpack(Δλb.batch(n - 1), Δλ_scal);
    }
    // TODO: use lgvl - 1 to stop once we're left with two diagonal blocks, then
    // handle that case manually
    for (index_t l = 0; l < lgvl; ++l) {
        const auto offset = index_t{1} << l;
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < VL; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ fwd scalar", i);
            PRINTLN("solve rhs fwd b(2i+1 = {}) -> [{}]", i, n - 1);
            // L⁻¹ b
            scalar_blas::xtrsv_LNN(LΨd_scal.batch(i), Δλ_scal.batch(i), be);
            // Update previous even block
            PRINTLN("  update rhs fwd b(2i   = {}) -> [{}]", i - offset, n - 1);
            scalar_blas::xgemv_sub(U_scal.batch(i), Δλ_scal.batch(i),
                                   Δλ_scal.batch(i - offset), be);
        }
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < VL - offset; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ fwd scalar 2", i);
            // Update next even block
            auto Y = LΨs_scal.batch(i);
            scalar_blas::xgemv_sub(Y, Δλ_scal.batch(i),
                                   Δλ_scal.batch(i + offset), be);
            PRINTLN("  update rhs fwd b(2i+2 = {}) -> [{}]", i + offset, n - 1);
        }
    }

    KOQKATOO_OMP(single) {
        // L⁻¹ b
        PRINTLN("solve rhs fwd b({}) -> [{}]", 0, n - 1);
        {
            KOQKATOO_TRACE("solve ψ fwd scalar", 0);
            scalar_blas::xtrsv_LNN(LΨd_scal.batch(0), Δλ_scal.batch(0), be);
        }

        // Reverse pass Ψ (scalar)

        // L⁻ᵀ b
        PRINTLN("solve rhs rev b({}) -> [{}]", 0, n - 1);
        {
            KOQKATOO_TRACE("solve ψ rev scalar", 0);
            scalar_blas::xtrsv_LTN(LΨd_scal.batch(0), Δλ_scal.batch(0), be);
        }
    }

    for (index_t l = lgvl; l-- > 0;) {
        const auto offset = index_t{1} << l;
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < VL - offset; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ rev scalar 2", i);
            PRINTLN("updating rhs rev i={}", i);
            auto Y = LΨs_scal.batch(i);
            // Substitute next even block
            scalar_blas::xgemv_T_sub(Y, Δλ_scal.batch(i + offset),
                                     Δλ_scal.batch(i), be);
        }
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < VL; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ rev scalar", i);
            PRINTLN("updating rhs rev i={}", i);
            // Substitute previous even block
            scalar_blas::xgemv_T_sub(U_scal.batch(i), Δλ_scal.batch(i - offset),
                                     Δλ_scal.batch(i), be);
            // L⁻ᵀ b
            scalar_blas::xtrsv_LTN(LΨd_scal.batch(i), Δλ_scal.batch(i), be);
        }
    }

    KOQKATOO_OMP(single) {
        KOQKATOO_TRACE("pack Δλ", 0);
        for (index_t j = 0; j < VL; ++j)
            Δλb.batch(n - 1)(j) = Δλ_scal(j);
    }
#endif

#if DO_PRINT
    KOQKATOO_OMP(single) {
        std::cout << "λ_sol = [\n";
        for (index_t j = 0; j < VL; ++j) {
            guanaqo::print_python(std::cout, Δλb.batch(n - 1)(j), ",\n", false);
        }
        std::cout << "]\n";
    }
#endif

    // Reverse pass Ψ (batched)
    for (index_t l = lgn; l-- > 0;) {
        const auto offset = index_t{1} << l;
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ rev 2", i);
            const auto hi = get_heap_index(i, n);
            PRINTLN("solve rhs rev b(2i+1 = {}) -> [{}]", i, hi);
            auto Y = LΨs.batch(hi);
            // Substitute next even block
            if (i + offset < n) {
                const auto hi_next = get_heap_index(i + offset, n);
                PRINTLN("  update rhs rev b(2i+2 = {}) -> [{}]", i + offset,
                        hi_next);
                compact_blas::xgemv_T_sub(Y, Δλb.batch(hi_next), Δλb.batch(hi),
                                          be);
            } else {
                // Last batch is special since we cross batch boundaries
                const auto hi_next = get_heap_index(i + offset - vstride, n);
                PRINTLN("  update rhs rev b(2i+2 = {}) -> [{}]Δ", i + offset,
                        hi_next);
                compact_blas::xgemv_T_sub_shift(Y, Δλb.batch(hi_next),
                                                Δλb.batch(hi));
            }
        }
        KOQKATOO_OMP(for)
        for (index_t i = offset; i < n; i += 2 * offset) {
            KOQKATOO_TRACE("solve ψ rev", i);
            const auto hi      = get_heap_index(i, n),
                       hi_prev = get_heap_index(i - offset, n);
            PRINTLN("  update rhs rev b(2i = {}) -> [{}]", i + offset, hi_prev);
            // Substitute previous even block
            compact_blas::xgemv_T_sub(U.batch(hi), Δλb.batch(hi_prev),
                                      Δλb.batch(hi), be);
            // L⁻ᵀ b
            compact_blas::xtrsv_LTN(LΨd.batch(hi), Δλb.batch(hi), be);
        }
    }

    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("solve Hd=-g-MᵀΔλ", i);
        const auto hi = get_heap_index(i, n);
        compact_blas::xgemv_sub(Wᵀ.batch(hi), Δλb.batch(hi), db.batch(hi), be);
        if (i + 1 < n) {
            const auto hi_next = get_heap_index(i + 1, n);
            compact_blas::xgemv_T_add(V.batch(hi), Δλb.batch(hi_next),
                                      db.batch(hi), be);
        } else {
            const auto hi_next = get_heap_index(i - vstride + 1, n);
            compact_blas::xgemv_T_add_shift(V.batch(hi), Δλb.batch(hi_next),
                                            db.batch(hi));
        }
        compact_blas::xtrsv_LTN(LH.batch(hi), db.batch(hi), be);
        MᵀΔλb.batch(hi).top_rows(nx) = Δλb.batch(hi);
        MᵀΔλb.batch(hi).bottom_rows(nu).set_constant(0);
        if (i + 1 < n) {
            const auto hi_next = get_heap_index(i + 1, n);
            compact_blas::xgemv_T_sub(AB.batch(hi), Δλb.batch(hi_next),
                                      MᵀΔλb.batch(hi), be);
        } else {
            const auto hi_next = get_heap_index(i - vstride + 1, n);
            compact_blas::xgemv_T_sub_shift(AB.batch(hi), Δλb.batch(hi_next),
                                            MᵀΔλb.batch(hi));
        }
    }

    } // end parallel

    // Unpack return values
    for (index_t i = 0; i < n; ++i) {
        auto hi = get_heap_index(i, n);
        for (index_t vi = 0; vi < VL; ++vi) {
            auto k = i + vi * vstride;
            if (k < N) {
                Mxbv.middle_rows(k * nx, nx)    = Mxbb.batch(hi)(vi);
                Δλv.middle_rows(k * nx, nx)     = Δλb.batch(hi)(vi);
                dv.middle_rows(k * nxu, nxu)    = db.batch(hi)(vi);
                MᵀΔλv.middle_rows(k * nxu, nxu) = MᵀΔλb.batch(hi)(vi);
                Mᵀλv.middle_rows(k * nxu, nxu)  = Mᵀλb.batch(hi)(vi);
                Aᵀŷv.middle_rows(k * nxu, nxu)  = Aᵀŷb.batch(hi)(vi);
            } else if (k == N) {
                Mxbv.middle_rows(k * nx, nx)  = Mxbb.batch(hi)(vi);
                Δλv.middle_rows(k * nx, nx)   = Δλb.batch(hi)(vi);
                dv.middle_rows(k * nxu, nx)   = db.batch(hi)(vi).top_rows(nx);
                Mᵀλv.middle_rows(k * nxu, nx) = Mᵀλb.batch(hi)(vi).top_rows(nx);
                MᵀΔλv.middle_rows(k * nxu, nx) =
                    MᵀΔλb.batch(hi)(vi).top_rows(nx);
                Aᵀŷv.middle_rows(k * nxu, nx) = Aᵀŷb.batch(hi)(vi).top_rows(nx);
            }
        }
    }
}

template void
solve_cyclic<2>(const koqkatoo::ocp::LinearOCPStorage &ocp, real_t S,
                std::span<const real_t> Σ, std::span<const bool> J,
                std::span<const real_t> x, std::span<const real_t> grad,
                std::span<const real_t> λ, std::span<const real_t> b,
                std::span<const real_t> ŷ, std::span<real_t> Mxb,
                std::span<real_t> Mᵀλ, std::span<real_t> Aᵀŷ,
                std::span<real_t> d, std::span<real_t> Δλ,
                std::span<real_t> MᵀΔλ);
template void
solve_cyclic<4>(const koqkatoo::ocp::LinearOCPStorage &ocp, real_t S,
                std::span<const real_t> Σ, std::span<const bool> J,
                std::span<const real_t> x, std::span<const real_t> grad,
                std::span<const real_t> λ, std::span<const real_t> b,
                std::span<const real_t> ŷ, std::span<real_t> Mxb,
                std::span<real_t> Mᵀλ, std::span<real_t> Aᵀŷ,
                std::span<real_t> d, std::span<real_t> Δλ,
                std::span<real_t> MᵀΔλ);
template void
solve_cyclic<8>(const koqkatoo::ocp::LinearOCPStorage &ocp, real_t S,
                std::span<const real_t> Σ, std::span<const bool> J,
                std::span<const real_t> x, std::span<const real_t> grad,
                std::span<const real_t> λ, std::span<const real_t> b,
                std::span<const real_t> ŷ, std::span<real_t> Mxb,
                std::span<real_t> Mᵀλ, std::span<real_t> Aᵀŷ,
                std::span<real_t> d, std::span<real_t> Δλ,
                std::span<real_t> MᵀΔλ);
