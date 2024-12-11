#pragma once

#include <koqkatoo/loop.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/openmp.h>
#include <optional>

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
    if (i < AB().num_batches())
        compact_blas::xcopy(AB().batch(i), V().batch(i));
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_H() {
    compact_blas::xpotrf(LHV(), settings.preferred_backend);
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_Hi(index_t i) {
    compact_blas::xpotrf(LHV().batch(i), settings.preferred_backend);
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
void Solver<Abi>::prepare_Ψi(index_t i) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    auto LHi = LH().batch(i), Wi = Wᵀ().batch(i), Vi = V().batch(i);
    // Solve W = LH⁻¹ [I 0]ᵀ
    compact_blas::xcopy(LHi.top_left(nx + nu, nx), Wi);
    compact_blas::xtrtri(Wi, settings.preferred_backend);
    compact_blas::xtrsm_LLNN(LHi.bottom_right(nu, nu), Wi.bottom_rows(nu),
                             settings.preferred_backend);
    compact_blas::xsyrk_T(Wi, LΨd().batch(i), settings.preferred_backend);
    if (i < AB().num_batches()) {
        // Store V(i) = VVᵀ
        compact_blas::xsyrk(Vi, VV().batch(i), settings.preferred_backend);
        // Store LΨ(i+1,i) = -VW
        compact_blas::xgemm_neg(Vi, Wi, LΨs().batch(i),
                                settings.preferred_backend);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::prepare_Ψ() {
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < LH().num_batches(); ++i)
        prepare_Ψi(i);
}

template <simd_abi_tag Abi>
void Solver<Abi>::prepare_all(real_t S, real_view Σ, bool_view J) {
    using std::isfinite;
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < LH().num_batches(); ++i) {
        schur_complement_Hi(i, Σ, J);
        if (isfinite(S))
            LH().batch(i).add_to_diagonal(1 / S);
        cholesky_Hi(i);
        prepare_Ψi(i);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_Ψ() {
    const auto N = storage.dim.N_horiz;
    auto wLΨd = storage.work_LΨd(), wLΨs = storage.work_LΨs(),
         wVV = storage.work_VV();
    foreach_chunked_merged(0, N, simd_stride, [&](index_t i, auto ni) {
        // If the last batch is an incomplete one, already add Ld(N)
        for (index_t j = 0; j < std::min(ni + 1, simd_stride); ++j)
            wLΨd(j) = LΨd()(i + j);
        if (i > 0)
            wLΨd(0) += wVV(simd_stride - 1);
        for (index_t j = 0; j < ni; ++j)
            wLΨs(j) = LΨs()(i + j);
        for (index_t j = 0; j < ni; ++j)
            wVV(j) = VV()(i + j);
        for (index_t j = 0; j < ni; ++j) {
            scalar_blas::xpotrf(wLΨd.batch(j), settings.preferred_backend);
            scalar_blas::xtrsm_RLTN(wLΨd.batch(j), wLΨs.batch(j),
                                    settings.preferred_backend);
            scalar_blas::xsyrk_sub(wLΨs.batch(j), wVV.batch(j),
                                   settings.preferred_backend);
            if (j + 1 < simd_stride)
                wLΨd(j + 1) += wVV(j);
        }
        for (index_t j = 0; j < ni; ++j)
            storage.LΨd_scalar()(i + j) = wLΨd(j);
        for (index_t j = 0; j < ni; ++j)
            storage.LΨs_scalar()(i + j) = wLΨs(j);
    });
    index_t last_j = N % simd_stride;
    if (last_j == 0) {
        // If the previous batch was complete, the term VV - LsLs is in VV.
        // We load and add WW to it, factor it and store it.
        wVV(simd_stride - 1) += LΨd()(N);
        scalar_blas::xpotrf(wVV.batch(simd_stride - 1),
                            settings.preferred_backend);
        storage.LΨd_scalar()(N) = wVV(simd_stride - 1);
    } else {
        // If the previous batch was not complete, Ld has already been loaded
        // and updated by VV - LsLs.
        scalar_blas::xpotrf(wLΨd.batch(last_j), settings.preferred_backend);
        storage.LΨd_scalar()(N) = wLΨd(last_j);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_Ψ(Timings &t) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;

    // First diagonal block
    std::optional<guanaqo::Timed<typename Timings::type>> timer;
    timer.emplace(t.chol_Ψ_copy_1);
    storage.work_LΨd(0)(0) = LΨd()(0);
    timer.emplace(t.chol_Ψ_potrf);
    scalar_blas::xpotrf(storage.work_LΨd(0), settings.preferred_backend);
    for (index_t i = 0; i < N; ++i) {
        const index_t j = i % simd_stride, j_next = (i + 1) % simd_stride;
        // Sub-diagonal block
        timer.emplace(t.chol_Ψ_copy_1);
        storage.work_LΨs(j)(0) = LΨs()(i);
        timer.emplace(t.chol_Ψ_trsm);
        scalar_blas::xtrsm_RLTN(storage.work_LΨd(j), storage.work_LΨs(j),
                                settings.preferred_backend);
        // if we have read all matrices in this block
        if (j_next == 0) {
            timer.emplace(t.chol_Ψ_copy_2);
            // overwrite them with our scalar versions
            for (index_t k = 0; k < simd_stride; ++k) {
                storage.LΨd_scalar().batch(i + 1 - simd_stride + k) =
                    storage.work_LΨd(k);
                storage.LΨs_scalar().batch(i + 1 - simd_stride + k) =
                    storage.work_LΨs(k);
            }
        }
        // Next diagonal block
        timer.emplace(t.chol_Ψ_copy_1);
        storage.work_LΨd(j_next)(0) = LΨd()(i + 1);
        storage.work_LΨd(j_next)(0) += VV()(i);
        timer.emplace(t.chol_Ψ_syrk);
        scalar_blas::xsyrk_sub(storage.work_LΨs(j), storage.work_LΨd(j_next),
                               settings.preferred_backend);
        timer.emplace(t.chol_Ψ_potrf);
        scalar_blas::xpotrf(storage.work_LΨd(j_next),
                            settings.preferred_backend);
    }
    // Store leftover blocks
    const index_t j = N % simd_stride;
    timer.emplace(t.chol_Ψ_copy_2);
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
void Solver<Abi>::solve_Ψ_scalar(std::span<real_t> λ, Timings &t) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    scalar_mut_real_view λ_{{.data = λ.data(), .depth = N + 1, .rows = nx}};
    std::optional<guanaqo::Timed<typename Timings::type>> timer;
    for (index_t i = 0; i < N + 1; ++i) {
        timer.emplace(t.solve_Ψ_solve);
        // λ[i] = L[i,i]⁻¹ b[i]
        scalar_blas::xtrsm_LLNN(storage.LΨd_scalar().batch(i), λ_.batch(i),
                                settings.preferred_backend);
        if (i < N) {
            timer.emplace(t.solve_Ψ_gemm);
            // b[i+1] -= L[i+1,i] λ[i]
            scalar_blas::xgemm_sub(storage.LΨs_scalar().batch(i), λ_.batch(i),
                                   λ_.batch(i + 1), settings.preferred_backend);
        }
    }
    for (index_t i = N + 1; i-- > 0;) {
        timer.emplace(t.solve_Ψ_solve_tp);
        // λ[i] = L[i,i]⁻ᵀ b[i]
        scalar_blas::xtrsm_LLTN(storage.LΨd_scalar().batch(i), λ_.batch(i),
                                settings.preferred_backend);
        if (i > 0) {
            timer.emplace(t.solve_Ψ_gemm_tp);
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
void Solver<Abi>::factor(real_t S, real_view Σ, bool_view J, Timings &t) {
    timed(t.prepare_all, [&] { prepare_all(S, Σ, J); });
    timed(t.cholesky_Ψ, [&] { cholesky_Ψ(t); });
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve(real_view grad, real_view Mᵀλ, real_view Aᵀŷ,
                        real_view Mxb, mut_real_view d, mut_real_view Δλ,
                        mut_real_view MᵀΔλ) {
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
void Solver<Abi>::solve(real_view grad, real_view Mᵀλ, real_view Aᵀŷ,
                        real_view Mxb, mut_real_view d, mut_real_view Δλ,
                        mut_real_view MᵀΔλ, Timings &t) {
    // d ← ∇f̃(x) + Mᵀλ + Aᵀŷ (= v)
    timed(t.solve_add_rhs_1,
          [&] { compact_blas::xadd_copy(d, grad, Mᵀλ, Aᵀŷ); });
    // d ← H⁻¹ d
    timed(t.solve_H_1, &Solver::solve_H, this, d);
    // Δλ ← Md - (Mx - b)
    timed(t.solve_mat_vec, &Solver::residual_dynamics_constr, this, d, Mxb, Δλ);
    // Δλ ← (M H⁻¹ Mᵀ)⁻¹ Δλ
    timed(t.solve_unshuffle, &storage_t::restore_dynamics_constraints, storage,
          Δλ, storage.Δλ_scalar);
    timed(t.solve_Ψ, [&] { solve_Ψ_scalar(storage.Δλ_scalar, t); });
    timed(t.solve_shuffle, &storage_t::copy_dynamics_constraints, storage,
          storage.Δλ_scalar, Δλ);
    // MᵀΔλ ← Mᵀ Δλ
    timed(t.solve_mat_vec_tp, &Solver::mat_vec_transpose_dynamics_constr, this,
          Δλ, MᵀΔλ);
    // d ← MᵀΔλ - ∇f̃(x) - Mᵀλ - Aᵀŷ
    timed(t.solve_add_rhs_2,
          [&] { compact_blas::xsub_copy(d, MᵀΔλ, Mᵀλ, grad, Aᵀŷ); });
    // d ← H⁻¹ d
    timed(t.solve_H_2, &Solver::solve_H, this, d);
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
