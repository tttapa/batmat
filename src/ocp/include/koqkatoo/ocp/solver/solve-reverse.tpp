#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/thread-pool.hpp>
#include <guanaqo/trace.hpp>
#include <algorithm>
#include <atomic>

#include <koqkatoo/cholundate/householder-updowndate-common.tpp>
#include <koqkatoo/cholundate/householder-updowndate.hpp>
#include <koqkatoo/cholundate/micro-kernels/householder-updowndate.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/xshhud-diag.hpp> // TODO
#include <koqkatoo/linalg/small-potrf.hpp>

namespace koqkatoo::ocp {

template <simd_abi_tag Abi>
void Solver<Abi>::tridiagonal_factor_rev(index_t k) {
    GUANAQO_TRACE("factor Ψ", k);
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    const auto stage_idx       = k * simd_stride;
    KOQKATOO_ASSERT(stage_idx < N + 1);
    auto LΨ = LΨ_scalar(), LΨd = LΨd_scalar(), LΨs = LΨs_scalar(),
         VVᵀ = VVᵀ_scalar();
    // Default BLAS potrf is quite slow for small matrices ...
    constexpr bool merged_potrf_trsm = false;
    const bool use_small_potrf       = settings.use_serial_small_potrf;
    // Factor
    for (index_t i = std::min(N, stage_idx + simd_stride); i-- > stage_idx;) {
        // Add VVᵀ(i) to Θ(i+1)
        scalar_blas::xadd_L(VVᵀ.batch(i), LΨd.batch(i + 1));
        // Factor LΨd(i+1) = chol(Θ(i+1)) and solve LΨs(i) = -WV(i) LΨd(k+1)⁻ᵀ
        if (use_small_potrf) {
            auto LΨi = LΨ(i + 1);
            linalg::small_potrf(LΨi.data, LΨi.outer_stride, LΨi.rows, LΨi.cols);
        } else if (merged_potrf_trsm) {
            scalar_blas::xpotrf(LΨ.batch(i + 1), settings.preferred_backend);
        } else {
            scalar_blas::xpotrf(LΨd.batch(i + 1), settings.preferred_backend);
            scalar_blas::xtrsm_RLTN(LΨd.batch(i + 1), LΨs.batch(i + 1),
                                    settings.preferred_backend);
        }
        // Compute Θ(i) = WWᵀ(i) - LΨs(i) LΨs(i)ᵀ
        scalar_blas::xsyrk_sub(LΨs.batch(i + 1), LΨd.batch(i),
                               settings.preferred_backend);
    }
    // If this was the last batch, factor Θ
    if (stage_idx == 0) {
        if (use_small_potrf) {
            linalg::small_potrf(LΨd(0).data, LΨd(0).outer_stride, LΨd(0).rows,
                                LΨd(0).cols);
        } else {
            scalar_blas::xpotrf(LΨd.batch(0), settings.preferred_backend);
        }
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::factor_rev(real_t S, real_view Σ, bool_view J) {
    return factor<true>(S, Σ, J);
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve_rev(real_view grad, real_view Mᵀλ, real_view Aᵀŷ,
                            real_view Mxb, mut_real_view d, mut_real_view Δλ,
                            mut_real_view MᵀΔλ) {
    GUANAQO_TRACE("solve", 0);

    auto [N, nx, nu, ny, ny_N] = storage.dim;
    const auto be              = settings.preferred_backend;
    scalar_mut_real_view Δλ_scal{{.data  = storage.Δλ_scalar.data(),
                                  .depth = N + 1,
                                  .rows  = nx,
                                  .cols  = 1}};
    assert(storage.Δλ_scalar.size() == static_cast<size_t>(Δλ_scal.size()));
    auto Δλ1 = storage.λ1();
    auto LΨd = storage.LΨd_scalar(), LΨs = storage.LΨs_scalar();

    // Parallel solve Hv = -g
    // ----------------------
    auto solve_H1 = [&](index_t i) {
        GUANAQO_TRACE("solve Hv=g", i);
        // Initialize rhs: g = ∇ϕ + Mᵀλ = ∇f̃ + Aᵀŷ + Mᵀλ                 (d ← g)
        for (index_t j = 0; j < nx + nu; ++j)
            for (index_t ii = i * simd_stride; ii < (i + 1) * simd_stride; ++ii)
                d(ii, j, 0) = -grad(ii, j, 0) - Mᵀλ(ii, j, 0) - Aᵀŷ(ii, j, 0);
        // Solve Lᴴ vʹ = g                                              (d ← vʹ)
        compact_blas::xtrsv_LNN(LH().batch(i), d.batch(i), be);
        // Solve Lᴴ⁻ᵀ v = vʹ                                            (λ ← Ev)
        compact_blas::xgemv_T(Wᵀ().batch(i), d.batch(i), Δλ.batch(i), be);
        // Compute f = (A B) v                                          (λ1 ← f)
        if (i < AB().num_batches())
            compact_blas::xgemv(V().batch(i), d.batch(i), Δλ1.batch(i), be);
    };

    // Forward substitution Ψ
    // ----------------------
    auto solve_ψ_fwd = [&](index_t i) {
        KOQKATOO_ASSERT(i <= N);
        // Initialize rhs r + f - v = Mx-b - (AB) v + (E0)v   (λ_scal ← ...)
        if (i > 0)
            for (index_t j = 0; j < nx; ++j)
                Δλ_scal(i, j, 0) =
                    Mxb(i, j, 0) + Δλ(i, j, 0) - Δλ1(i - 1, j, 0);
        else
            for (index_t j = 0; j < nx; ++j)
                Δλ_scal(0, j, 0) = Mxb(0, j, 0) + Δλ(0, j, 0);
        if (i < N) {
            // Subtract L(s) λʹ(i)                            (λ_scal ← ...)
            scalar_blas::xgemv_sub(LΨs.batch(i + 1), Δλ_scal.batch(i + 1),
                                   Δλ_scal.batch(i), be);
        }
        // Solve L(d) Δλʹ = r + f - v - L(s) λʹ(i)            (λ_scal ← ...)
        scalar_blas::xtrsv_LNN(LΨd.batch(i), Δλ_scal.batch(i), be);
    };

    // Backward substitution Ψ
    // -----------------------
    auto solve_ψ_rev = [&](index_t i) {
        if (i > 0)
            scalar_blas::xgemv_T_sub(LΨs.batch(i), Δλ_scal.batch(i - 1),
                                     Δλ_scal.batch(i), be);
        scalar_blas::xtrsv_LTN(LΨd.batch(i), Δλ_scal.batch(i), be);
        if (i > 0)
            for (index_t j = 0; j < nx; ++j)
                Δλ1(i - 1, j, 0) = Δλ(i, j, 0) = Δλ_scal(i, j, 0);
        else
            for (index_t j = 0; j < nx; ++j)
                Δλ(0, j, 0) = Δλ_scal(0, j, 0);
    };

    // Parallel solve Hd = -g - Mᵀλ
    // -----------------------------
    auto solve_H2 = [&](index_t i) {
        GUANAQO_TRACE("solve Hd=-g-MᵀΔλ", i);
        compact_blas::xgemv_sub(Wᵀ().batch(i), Δλ.batch(i), d.batch(i), be);
        if (i < AB().num_batches())
            compact_blas::xgemv_T_add(V().batch(i), Δλ1.batch(i), d.batch(i),
                                      be);
        compact_blas::xtrsv_LTN(LH().batch(i), d.batch(i), be);
        MᵀΔλ.batch(i).top_rows(nx) = Δλ.batch(i);
        MᵀΔλ.batch(i).bottom_rows(nu).set_constant(0);
        if (i < AB().num_batches())
            compact_blas::xgemv_T_sub(AB().batch(i), Δλ1.batch(i),
                                      MᵀΔλ.batch(i), be);
    };

    index_t num_batch   = (N + 1 + simd_stride - 1) / simd_stride;
    auto &join_counters = storage.reset_join_counters();
    // process_fwd returns true if last batch was processed
    const auto process_fwd = [&](index_t k) {
        solve_H1(k);
        while (true) {
            if (k + 1 < num_batch &&
                join_counters[k + 1].value.fetch_add(1) != 1)
                return false;
            GUANAQO_TRACE("solve ψ fwd", k);
            // We introduce a delay of 1 when solving in reverse, because we
            // the right-hand side depends on f(i-1) if i > 0.
            const index_t i_start = std::min((k + 1) * simd_stride, N) + 1,
                          i_end   = k * simd_stride + 1;
            for (index_t i = i_start; i-- > i_end;)
                solve_ψ_fwd(i);
            if (k-- == 0) {
                solve_ψ_fwd(0); // final stage (i=0) has no more dependencies
                return true;
            }
        }
    };

    std::atomic<index_t> batch_counter_1{num_batch - 1}, stage_counter_2{0};
    KOQKATOO_ASSERT(std::in_range<int>(num_batch));
    std::atomic<int> stage_ready{0};
    auto thread_work = [&](index_t, index_t num_threads) {
        // Solve the first system Hv=g
        bool main_thread = false;
        while (true) {
            index_t k = batch_counter_1.fetch_sub(1, std::memory_order_relaxed);
            if (k < 0)
                break;
            main_thread = process_fwd(k);
        }
        // The thread that solves the last batch becomes the main thread that
        // performs the serial solution of Ψ in the next step.
        if (main_thread) {
            for (index_t k = 0; k < num_batch; ++k) {
                const index_t ik   = k * simd_stride,
                              ikp1 = std::min(N, ik + simd_stride);
                // When going backwards, we need a delay of one to make sure
                // that λ(i+1) is available before we notify the solve H step.
                // Concretely, we can only start solve H for batch k if
                /// Ψ((k + 1) * simd_stride) is done.
                {
                    GUANAQO_TRACE("solve ψ rev", k);
                    if (k == 0)
                        solve_ψ_rev(0);
                    for (index_t i = ik; i < ikp1; ++i)
                        solve_ψ_rev(i + 1); // delay of one for λ(i+1)
                }
                GUANAQO_TRACE("solve store-notify", k);
                // Once the block of Ψ is done, we can start the next block
                // of the solution of Hd=g-MᵀΔλ.
                stage_ready.fetch_add(1, std::memory_order_release);
                stage_ready.notify_one();
            }
            // Release all threads waiting for the next block of Ψ
            stage_ready.fetch_add(static_cast<int>(num_threads),
                                  std::memory_order_release);
            stage_ready.notify_all();
        }
        // All threads wait for blocks of Ψ to be ready so they can start
        // solving Hd=g-MᵀΔλ
        while (true) {
            // Try to decrement stage_ready to "win" the next block of H
            auto semaphore_val = stage_ready.load(std::memory_order_relaxed);
            do {
                // Wait for the semaphore value to become positive so we can
                // acquire it
                while (semaphore_val == 0) {
                    stage_ready.wait(semaphore_val, std::memory_order_relaxed);
                    semaphore_val = stage_ready.load(std::memory_order_relaxed);
                }
            } while (!stage_ready.compare_exchange_strong(
                semaphore_val, semaphore_val - 1, std::memory_order_acquire));
            // If our thread won the race for the block of H, process it
            index_t k = stage_counter_2.fetch_add(1, std::memory_order_relaxed);
            if (k >= num_batch)
                break;
            solve_H2(k);
        }
    };

    foreach_thread(thread_work);
}

template <simd_abi_tag Abi>
void Solver<Abi>::updowndate_rev(real_view Σ, bool_view J_old,
                                 bool_view J_new) {
    return updowndate<true>(Σ, J_old, J_new);
}

} // namespace koqkatoo::ocp
