#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/thread-pool.hpp>
#include <koqkatoo/trace.hpp>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <print>

#include <koqkatoo/cholundate/householder-updowndate-common.tpp>
#include <koqkatoo/cholundate/householder-updowndate.hpp>
#include <koqkatoo/cholundate/micro-kernels/householder-updowndate.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/xshhud-diag.hpp> // TODO
#include <koqkatoo/linalg/small-potrf.hpp>
#include <guanaqo/print.hpp>

namespace koqkatoo::ocp {

template <simd_abi_tag Abi>
void Solver<Abi>::prepare_factor_rev(index_t k, real_t S, real_view Σ,
                                     bool_view J) {
    using std::isfinite;
    KOQKATOO_TRACE("factor prep", k);
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    const auto be              = settings.preferred_backend;
    // Get blocks of Ψ
    const auto stage_idx = k * simd_stride;
    const auto nd        = std::min(simd_stride, N + 1 - stage_idx),
               ni        = std::min(simd_stride, N - stage_idx);
    auto LHi = LH().batch(k), Wi = Wᵀ().batch(k);
    // Initialize V = F, it is computed while factorizing H
    if (k < AB().num_batches())
        compact_blas::xcopy(AB().batch(k), V().batch(k));
    // Compute H = Hℓ + GᵀΣJ G + Γ⁻¹
    compact_blas::xsyrk_T_schur_copy(CD().batch(k), Σ.batch(k), J.batch(k),
                                     H().batch(k), LHi);
    if (isfinite(S))
        LHi.add_to_diagonal(1 / S);
    // Factorize H (and solve V)
    if (k < AB().num_batches())
        compact_blas::xpotrf(LHV().batch(k), be);
    else
        compact_blas::xpotrf(LHV().batch(k).top_rows(nx + nu), be);
    // Solve W = LH⁻¹ [I 0]ᵀ
    compact_blas::xtrtri_copy(LHi.top_left(nx + nu, nx), Wi, be);
    compact_blas::xtrsm_LLNN(LHi.bottom_right(nu, nu), Wi.bottom_rows(nu), be);
    // Compute WWᵀ
    compact_blas::xsyrk_T(Wi, WWᵀ().batch(k), be);
    // TODO: exploit trapezoidal shape of Wᵀ
    // Compute VVᵀ
    if (k < AB().num_batches()) {
        auto LΨsk = LΨs_scalar().middle_layers(stage_idx + 1, ni),
             VVᵀk = VVᵀ_scalar().middle_layers(stage_idx, ni);
        // Compute -WVᵀ
        compact_blas::xgemm_TT_neg(Wi, V().batch(k), VWᵀ().batch(k), be);
        std::println("  storing WVᵀ({}) to {}", k, stage_idx + 1);
        compact_blas::unpack(storage.VWᵀ().batch(k), LΨsk);
        // TODO: exploit trapezoidal shape of Wᵀ
        compact_blas::xsyrk(V().batch(k), VVᵀ().batch(k), be);
        compact_blas::unpack_L(VVᵀ().batch(k), VVᵀk);
    }
    auto LΨdk = LΨd_scalar().middle_layers(stage_idx, nd);
    std::println("  storing WWᵀ({}) to {}", k, stage_idx);
    compact_blas::unpack_L(WWᵀ().batch(k), LΨdk);
}

template <simd_abi_tag Abi>
void Solver<Abi>::tridiagonal_factor_rev(index_t k) {
    KOQKATOO_TRACE("factor Ψ", k);
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    const auto stage_idx       = k * simd_stride;
    KOQKATOO_ASSERT(stage_idx < N + 1);
    const auto nd = std::min(simd_stride, N + 1 - stage_idx),
               ni = std::min(simd_stride, N - stage_idx);
    auto LΨ = LΨ_scalar(), LΨd = LΨd_scalar(), LΨs = LΨs_scalar(),
         VVᵀ                         = VVᵀ_scalar();
    const auto potrf_be              = settings.preferred_backend;
    constexpr bool merged_potrf_trsm = false;
    const bool use_small_potrf       = settings.use_serial_small_potrf;
    // Factor
    for (index_t i = std::min(N, stage_idx + simd_stride); i-- > stage_idx;) {
        // Add VVᵀ(i) to Θ(i+1)
        std::println("  add VVᵀ({}) to Ψd({})", i, i + 1);
        scalar_blas::xadd_L(VVᵀ.batch(i), LΨd.batch(i + 1));
        // Factor LΨd(i+1) = chol(Θ(i+1)) and solve LΨs(i) = -WV(i) LΨd(k+1)⁻ᵀ
        if (use_small_potrf) {
            auto LΨj = LΨ(i + 1);
            linalg::small_potrf(LΨj.data, LΨj.outer_stride, LΨj.rows, LΨj.cols);
        } else if (merged_potrf_trsm) {
            scalar_blas::xpotrf(LΨ.batch(i + 1), potrf_be);
        } else {
            scalar_blas::xpotrf(LΨd.batch(i + 1), potrf_be);
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
            scalar_blas::xpotrf(LΨd.batch(0), potrf_be);
        }
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::factor_rev(real_t S, real_view Σ, bool_view J) {
    KOQKATOO_TRACE("factor", 0);
    const auto N         = storage.dim.N_horiz;
    const auto num_batch = (N + 1 + simd_stride - 1) / simd_stride;

    auto &join_counters      = storage.reset_join_counters();
    const auto process_stage = [&](index_t k) {
        prepare_factor_rev(num_batch - 1 - k, S, Σ, J);
        while (k < num_batch) {
            if (k > 0 && join_counters[k - 1].value.fetch_add(1) != 1)
                break;
            tridiagonal_factor_rev(num_batch - 1 - (k++));
        }
    };

    std::atomic<index_t> batch_counter{0};
    auto thread_work = [&](index_t, index_t) {
        while (true) {
            index_t k = batch_counter.fetch_add(1, std::memory_order_relaxed);
            if (k >= num_batch)
                break;
            process_stage(k);
        }
    };

    foreach_thread(thread_work);

    std::cout << "Wᵀ = [\n";
    for (index_t i = 0; i < N + 1; ++i)
        guanaqo::print_python(std::cout, Wᵀ()(i), ",\n", false);
    std::cout << "]" << std::endl;
    std::cout << "V = [\n";
    for (index_t i = 0; i < N; ++i)
        guanaqo::print_python(std::cout, V()(i), ",\n", false);
    std::cout << "]" << std::endl;
    std::cout << "LΨd_check = [\n";
    for (index_t i = 0; i < N + 1; ++i)
        guanaqo::print_python(std::cout, LΨd_scalar()(i), ",\n", false);
    std::cout << "]" << std::endl;
    std::cout << "LΨs_check = [\n";
    for (index_t i = 0; i < N; ++i)
        guanaqo::print_python(std::cout, LΨs_scalar()(i + 1), ",\n", false);
    std::cout << "]" << std::endl;
    std::cout << "WVᵀ_check = [\n";
    for (index_t i = 0; i < N; ++i)
        guanaqo::print_python(std::cout, VWᵀ()(i), ",\n", false);
    std::cout << "]" << std::endl;
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve_rev(real_view grad, real_view Mᵀλ, real_view Aᵀŷ,
                            real_view Mxb, mut_real_view d, mut_real_view Δλ,
                            mut_real_view MᵀΔλ) {
    KOQKATOO_TRACE("solve", 0);

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
        KOQKATOO_TRACE("solve Hv=g", i);
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
        if (i == N)
            return;
        KOQKATOO_ASSERT(i < N);
        // Initialize rhs r + f - v = Mx-b - (AB) v + (E0)v   (λ_scal ← ...)
        for (index_t j = 0; j < nx; ++j)
            Δλ_scal(i + 1, j, 0) =
                Mxb(i + 1, j, 0) + Δλ(i + 1, j, 0) - Δλ1(i, j, 0);
        if (i + 1 < N) {
            // Subtract L(s) λʹ(i + 1)                            (λ_scal ← ...)
            scalar_blas::xgemv_sub(LΨs.batch(i + 2), Δλ_scal.batch(i + 2),
                                   Δλ_scal.batch(i + 1), be);
        }
        std::println("solve fwd st.{}", i + 1);
        // Solve L(d) Δλʹ = r + f - v - L(s) λʹ(i + 1)            (λ_scal ← ...)
        scalar_blas::xtrsv_LNN(LΨd.batch(i + 1), Δλ_scal.batch(i + 1), be);
        if (i == 0) {
            // Initialize rhs r - v = Mx-b + (E0)v                (λ_scal ← ...)
            for (index_t j = 0; j < nx; ++j)
                Δλ_scal(0, j, 0) = Mxb(0, j, 0) + Δλ(0, j, 0);
            // Subtract L(s) λʹ(i + 1)                            (λ_scal ← ...)
            scalar_blas::xgemv_sub(LΨs.batch(1), Δλ_scal.batch(1),
                                   Δλ_scal.batch(0), be);
            std::println("solve fwd st.{}", 0);
            // Solve L(d) Δλʹ = r + f - v - L(s) λʹ(i + 1)        (λ_scal ← ...)
            scalar_blas::xtrsv_LNN(LΨd.batch(0), Δλ_scal.batch(0), be);
        }
    };

    // Backward substitution Ψ
    // -----------------------
    auto solve_ψ_rev = [&](index_t i) {
        if (i > 0)
            scalar_blas::xgemv_T_sub(LΨs.batch(i), Δλ_scal.batch(i - 1),
                                     Δλ_scal.batch(i), be);
        std::println("solve rev st.{}", i);
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
        KOQKATOO_TRACE("solve Hd=-g-MᵀΔλ", i);
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
        solve_H1(num_batch - 1 - k);
        while (true) {
            if (k > 0 && join_counters[k - 1].value.fetch_add(1) != 1)
                return false;
            KOQKATOO_TRACE("solve ψ fwd", k);
            index_t i_end = std::min((k + 1) * simd_stride, N + 1);
            for (index_t i = k * simd_stride; i < i_end; ++i)
                solve_ψ_fwd(N - i);
            ++k;
            if (k == num_batch)
                return true;
        }
    };

    std::atomic<index_t> batch_counter_1{0}, stage_counter_2{num_batch - 1};
    KOQKATOO_ASSERT(std::in_range<int>(num_batch));
    std::atomic<int> stage_ready{0};
    auto thread_work = [&](index_t, index_t num_threads) {
        // Solve the first system Hv=g
        bool main_thread = false;
        while (true) {
            index_t k = batch_counter_1.fetch_add(1, std::memory_order_relaxed);
            if (k >= num_batch)
                break;
            main_thread = process_fwd(k);
        }
        // The thread that solves the last batch becomes the main thread that
        // performs the serial solution of Ψ in the next step.
        if (main_thread) {
            for (index_t k = 0; k < num_batch; ++k) {
                const index_t ik   = k * simd_stride,
                              ikp1 = std::min(N + 1, ik + simd_stride);
                {
                    KOQKATOO_TRACE("solve ψ rev", k);
                    for (index_t i = ik; i < ikp1; ++i)
                        solve_ψ_rev(i);
                }
                KOQKATOO_TRACE("solve store-notify", k);
                // Once the block of Ψ is done, we can start the next block
                // of the solution of Hd=g-MᵀΔλ.
                std::println("release");
                stage_ready.fetch_add(1, std::memory_order_release);
                stage_ready.notify_one();
            }
            // Release all threads waiting for the next block of Ψ
            std::println("release {}", num_threads);
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
            index_t k = stage_counter_2.fetch_sub(1, std::memory_order_relaxed);
            if (k < 0)
                break;
            solve_H2(num_batch - 1 - k);
        }
    };

    foreach_thread(thread_work);
}

} // namespace koqkatoo::ocp
