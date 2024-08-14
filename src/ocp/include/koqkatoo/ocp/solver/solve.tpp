#pragma once

#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/openmp.h>

namespace koqkatoo::ocp {

template <simd_abi_tag Abi>
void Solver<Abi>::schur_complement_H(real_view Σ, bool_view J) {
    // Note: xsyrk_T_schur_copy has no MKL equivalent, so we always just use a
    //       (parallel) for loop.
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < CD().num_batches(); ++i)
        schur_complement_Hi(i, Σ, J);
}

template <simd_abi_tag Abi>
void Solver<Abi>::schur_complement_Hi(index_t i, real_view Σ, bool_view J) {
    compact_blas::xsyrk_T_schur_copy(CD().batch(i), Σ.batch(i), J.batch(i),
                                     H().batch(i), LH().batch(i));
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_H() {
    compact_blas::xpotrf(LH(), settings.preferred_backend);
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_Hi(index_t i) {
    compact_blas::xpotrf(LH().batch(i), settings.preferred_backend);
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve_H(mut_real_view x) {
    if (settings.prefer_single_loop) {
        KOQKATOO_OMP(parallel for)
        for (index_t i = 0; i < LH().num_batches(); ++i) {
            compact_blas::xtrsm_LLNN(LH().batch(i), x.batch(i),
                                     settings.preferred_backend);
            compact_blas::xtrsm_LLTN(LH().batch(i), x.batch(i),
                                     settings.preferred_backend);
        }
    } else {
        compact_blas::xtrsm_LLNN(LH(), x, settings.preferred_backend);
        compact_blas::xtrsm_LLTN(LH(), x, settings.preferred_backend);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::prepare_Ψi(index_t i, single_mut_real_view W,
                             single_mut_real_view V) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    auto LHi                   = LH().batch(i);
    // Solve W = LH⁻¹ [I 0]ᵀ
    auto W1 = W.top_rows(nx), W2 = W.bottom_rows(nu);
    compact_blas::xcopy(LHi.top_left(nx, nx), W1);
    compact_blas::xtrti(W1, settings.preferred_backend);
    compact_blas::xgemm_neg(LHi.bottom_left(nu, nx), W1, W2,
                            settings.preferred_backend);
    compact_blas::xtrsm_LLNN(LHi.bottom_right(nu, nu), W2,
                             settings.preferred_backend);
    compact_blas::xsyrk_T(W, LΨd().batch(i), settings.preferred_backend);
    if (i < AB().num_batches()) {
        // Solve V = [A B] LH⁻ᵀ
        compact_blas::xcopy(AB().batch(i), V);
        compact_blas::xtrsm_RLTN(LHi, V, settings.preferred_backend);
        // Store V(i) = VVᵀ
        compact_blas::xsyrk(V, VV().batch(i), settings.preferred_backend);
        // Store LΨ(i+1,i) = -VW
        compact_blas::xgemm_neg(V, W, LΨs().batch(i),
                                settings.preferred_backend);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::prepare_Ψ() {
    const int num_threads = KOQKATOO_OMP_IF_ELSE(omp_get_max_threads(), 1);
    storage.allocate_work_prepare_Ψ(num_threads);
    KOQKATOO_OMP(parallel) {
        const int thread_id = KOQKATOO_OMP_IF_ELSE(omp_get_thread_num(), 0);
        auto [W, V]         = storage.get_work_prepare_ψ(thread_id);
        KOQKATOO_OMP(for)
        for (index_t i = 0; i < LH().num_batches(); ++i)
            prepare_Ψi(i, W, V);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::prepare_all(real_t S, real_view Σ, bool_view J) {
    using std::isfinite;
    const int num_threads = KOQKATOO_OMP_IF_ELSE(omp_get_max_threads(), 1);
    storage.allocate_work_prepare_Ψ(num_threads);
    KOQKATOO_OMP(parallel) {
        const int thread_id = KOQKATOO_OMP_IF_ELSE(omp_get_thread_num(), 0);
        auto [W, V]         = storage.get_work_prepare_ψ(thread_id);
        KOQKATOO_OMP(for)
        for (index_t i = 0; i < LH().num_batches(); ++i) {
            schur_complement_Hi(i, Σ, J);
            if (isfinite(S))
                LH().batch(i).add_to_diagonal(1 / S);
            cholesky_Hi(i);
            prepare_Ψi(i, W, V);
        }
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_Ψ() {
    auto [N, nx, nu, ny, ny_N] = storage.dim;

    // First diagonal block
    storage.work_LΨd(0)(0) = LΨd()(0);
    scalar_blas::xpotrf(storage.work_LΨd(0), settings.preferred_backend);
    for (index_t i = 0; i < N; ++i) {
        const index_t j = i % simd_stride, j_next = (i + 1) % simd_stride;
        // Sub-diagonal block
        storage.work_LΨs(j)(0) = LΨs()(i);
        scalar_blas::xtrsm_RLTN(storage.work_LΨd(j), storage.work_LΨs(j),
                                settings.preferred_backend);
        // if we have read all matrices in this block
        if (j_next == 0) {
            // overwrite them with our scalar versions
            for (index_t k = 0; k < simd_stride; ++k) {
                storage.LΨd_scalar().batch(i + 1 - simd_stride + k) =
                    storage.work_LΨd(k);
                storage.LΨs_scalar().batch(i + 1 - simd_stride + k) =
                    storage.work_LΨs(k);
            }
        }
        // Next diagonal block
        storage.work_LΨd(j_next)(0) = LΨd()(i + 1);
        storage.work_LΨd(j_next)(0) += VV()(i);
        scalar_blas::xsyrk_sub(storage.work_LΨs(j), storage.work_LΨd(j_next),
                               settings.preferred_backend);
        scalar_blas::xpotrf(storage.work_LΨd(j_next),
                            settings.preferred_backend);
    }
    // Store leftover blocks
    const index_t j = N % simd_stride;
    for (index_t k = 0; k < j + 1; ++k) {
        storage.LΨd_scalar().batch(N - j + k) = storage.work_LΨd(k);
        if (k < j)
            storage.LΨs_scalar().batch(N - j + k) = storage.work_LΨs(k);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve_Ψ_scalar(std::span<real_t> λ) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    scalar_mut_real_view λ_{{.data = λ.data(), .depth = N + 1, .rows = nx}};
    for (index_t i = 0; i < N + 1; ++i) {
        // λ[i] = L[i,i]⁻¹ b[i]
        scalar_blas::xtrsm_LLNN(storage.LΨd_scalar().batch(i), λ_.batch(i),
                                settings.preferred_backend);
        if (i < N) {
            // b[i+1] -= L[i+1,i] λ[i]
            scalar_blas::xgemm_sub(storage.LΨs_scalar().batch(i), λ_.batch(i),
                                   λ_.batch(i + 1), settings.preferred_backend);
        }
    }
    for (index_t i = N + 1; i-- > 0;) {
        // λ[i] = L[i,i]⁻ᵀ b[i]
        scalar_blas::xtrsm_LLTN(storage.LΨd_scalar().batch(i), λ_.batch(i),
                                settings.preferred_backend);
        if (i > 0) {
            // b[i-1] -= L[i,i-1]ᵀ λ[i]
            scalar_blas::xgemm_TN_sub(storage.LΨs_scalar().batch(i - 1),
                                      λ_.batch(i), λ_.batch(i - 1),
                                      settings.preferred_backend);
        }
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::factor(real_t S, real_view Σ, bool_view J) {
    prepare_all(S, Σ, J);
    cholesky_Ψ();
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve(real_view grad, real_view Mᵀλ, real_view Aᵀŷ,
                        real_view Mxb, mut_real_view d, mut_real_view Δλ,
                        mut_real_view MᵀΔλ) {
    using std::isfinite;
    // d ← ∇f̃(x) + Mᵀλ + Aᵀŷ (= v)
    compact_blas::xadd_copy(d, grad, Mᵀλ, Aᵀŷ);
    // d ← H⁻¹ d
    solve_H(d);
    // Δλ ← Md - (Mx - b)
    residual_dynamics_constr(d, Mxb, Δλ);
    // Δλ ← (M H⁻¹ Mᵀ)⁻¹ Δλ
    storage.restore_dynamics_constraints(Δλ, storage.Δλ_scalar);
    solve_Ψ_scalar(storage.Δλ_scalar);
    storage.copy_dynamics_constraints(storage.Δλ_scalar, Δλ);
    // MᵀΔλ ← Mᵀ Δλ
    mat_vec_transpose_dynamics_constr(Δλ, MᵀΔλ);
    // d ← MᵀΔλ - ∇f̃(x) - Mᵀλ - Aᵀŷ
    compact_blas::xsub_copy(d, MᵀΔλ, Mᵀλ, grad, Aᵀŷ);
    // d ← H⁻¹ d
    solve_H(d);
}

template <simd_abi_tag Abi>
Solver<Abi>::Solver(const LinearOCPStorage &ocp) : storage{.dim = ocp.dim} {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    auto H = this->H(), CD = this->CD(), AB = this->AB();
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < N; ++i) {
        H(i)  = ocp.H(i);
        CD(i) = ocp.CD(i);
        AB(i) = ocp.AB(i);
    }
    H(N).set_constant(0);
    H(N).set_diagonal(1);
    H(N).top_left(nx, nx) = ocp.H(N);
    CD(N).set_constant(0);
    CD(N).top_left(ny_N, nx) = ocp.CD(N);
    for (index_t i = N + 1; i < H.ceil_depth(); ++i) {
        H(i).set_constant(0);
        H(i).set_diagonal(1);
        CD(i).set_constant(0);
    }
    for (index_t i = N; i < AB.ceil_depth(); ++i)
        AB(i).set_constant(0);
}

} // namespace koqkatoo::ocp
