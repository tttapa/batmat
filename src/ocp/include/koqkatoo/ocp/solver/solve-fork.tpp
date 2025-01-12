#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/thread-pool.hpp>
#include <koqkatoo/trace.hpp>
#include <libfork/core.hpp>
#include <libfork/schedule.hpp>
#include <algorithm>
#include <atomic>
#include <functional>
#include <stdexcept>

#include <koqkatoo/cholundate/householder-updowndate-common.tpp>
#include <koqkatoo/cholundate/householder-updowndate.hpp>
#include <koqkatoo/cholundate/micro-kernels/householder-updowndate.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/xshhud-diag.hpp> // TODO

/// libfork or custom thread pool
#define KOQKATOO_SOLVE_USE_LIBFORK 0
/// atomic wait (futex) or counting semaphore
#define KOQKATOO_SOLVE_USE_ATOMIC_WAIT 0

namespace koqkatoo::ocp {

inline constexpr auto do_invoke =
    []<class F, class... Args>(
        auto, F fun,
        Args... args) -> lf::task<std::invoke_result_t<F, Args...>> {
    co_return std::invoke(std::move(fun), std::move(args)...);
};

template <simd_abi_tag Abi>
void Solver<Abi>::factor_fork(real_t S, real_view Σ, bool_view J) {
    KOQKATOO_TRACE("factor", 0);

    using std::isfinite;

    const auto N = storage.dim.N_horiz;

    const auto prepare = [this, S, &Σ, &J](index_t k) {
        KOQKATOO_TRACE("factor prep", k);
        auto [N, nx, nu, ny, ny_N] = storage.dim;
        schur_complement_Hi(k, Σ, J);
        if (isfinite(S))
            LH().batch(k).add_to_diagonal(1 / S);
        cholesky_Hi(k);
        // Solve W = LH⁻¹ [I 0]ᵀ
        auto LHi = LH().batch(k), Wi = Wᵀ().batch(k);
        compact_blas::xcopy(LHi.top_left(nx + nu, nx), Wi);
        compact_blas::xtrtri(Wi, settings.preferred_backend);
        compact_blas::xtrsm_LLNN(LHi.bottom_right(nu, nu), Wi.bottom_rows(nu),
                                 settings.preferred_backend);
        // Compute WWᵀ
        compact_blas::xsyrk_T(Wᵀ().batch(k), LΨd().batch(k),
                              settings.preferred_backend);
        // Compute -VWᵀ
        if (k < AB().num_batches())
            compact_blas::xgemm_neg(V().batch(k), Wᵀ().batch(k), LΨs().batch(k),
                                    settings.preferred_backend);
        // Compute VVᵀ
        if (k < AB().num_batches())
            compact_blas::xsyrk(V().batch(k), VV().batch(k),
                                settings.preferred_backend);
    };
    const auto factor_Ψ = [this](index_t k) {
        KOQKATOO_TRACE("factor Ψ", k);
        const auto N = storage.dim.N_horiz;
        auto nd      = std::min(simd_stride, N + 1 - k * simd_stride);
        auto wLΨd = storage.work_LΨd(), wVV = storage.work_VV();
        // Copy WWᵀ, add to Θ
        compact_blas::unpack_L(LΨd().batch(k), wLΨd.first_layers(nd));
        if (k > 0)
            wLΨd(0) += wVV(simd_stride - 1);
        // Copy -VWᵀ
        auto ni = std::min(simd_stride, N - k * simd_stride);
        if (ni > 0) {
            auto wLΨs = storage.work_LΨs();
            compact_blas::unpack(LΨs().batch(k), wLΨs.first_layers(ni));
        }
        // Copy VVᵀ and then factor all batches of Ψ
        auto wLΨs = storage.work_LΨs();
        // Copy VVᵀ
        if (ni > 0)
            compact_blas::unpack_L(VV().batch(k), wVV.first_layers(ni));
        // Factor
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
            storage.LΨd_scalar()(k * simd_stride + j) = wLΨd(j);
        for (index_t j = 0; j < ni; ++j)
            storage.LΨs_scalar()(k * simd_stride + j) = wLΨs(j);
        // If this was the last batch, factor Θ
        if ((k + 1) * simd_stride > N) {
            index_t last_j = N % simd_stride;
            if (last_j == 0) {
                assert(ni <= 0);
                // If the previous batch was complete, the term VV - LsLs
                // is in VV. We load and add WW to it, then factor it and
                // store it.
                wVV(simd_stride - 1) += LΨd()(N);
                scalar_blas::xpotrf(wVV.batch(simd_stride - 1),
                                    settings.preferred_backend);
                storage.LΨd_scalar()(N) = wVV(simd_stride - 1);
            } else {
                assert(last_j <= ni);
                // If the previous batch was not complete, Ld has already
                // been loaded and updated by VV - LsLs.
                scalar_blas::xpotrf(wLΨd.batch(last_j),
                                    settings.preferred_backend);
                storage.LΨd_scalar()(N) = wLΨd(last_j);
            }
        }
    };
    index_t num_batch        = (N + 1 + simd_stride - 1) / simd_stride;
    auto &join_counters      = storage.reset_join_counters();
    const auto process_stage = [&](index_t k) {
        prepare(k);
        while (k < num_batch) {
            if (k > 0 && join_counters[k - 1].value.fetch_add(1) != 1)
                break;
            factor_Ψ(k++);
        }
    };

#if KOQKATOO_SOLVE_USE_LIBFORK
    const auto process_all = [](auto, auto process_stage,
                                index_t num_batch) -> lf::task<void> {
        for (index_t k = 0; k < num_batch; ++k)
            co_await lf::fork[do_invoke](process_stage, k);
        co_await lf::join;
    };
    lf::sync_wait(*fork_pool, process_all, process_stage, num_batch);
#else
    std::atomic<index_t> batch_counter{0};
    auto thread_work = [&] {
        while (true) {
            index_t k = batch_counter.fetch_add(1, std::memory_order_relaxed);
            if (k >= num_batch)
                break;
            process_stage(k);
        }
    };
#if KOQKATOO_WITH_OPENMP
    KOQKATOO_OMP(parallel for schedule(static))
    for (int i = 0; i < omp_get_num_threads(); ++i)
        thread_work();
#else
    pool->sync_run_all(thread_work);
#endif
#endif
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve_fork(real_view grad, real_view Mᵀλ, real_view Aᵀŷ,
                             real_view Mxb, mut_real_view d, mut_real_view Δλ,
                             mut_real_view MᵀΔλ) {
    KOQKATOO_TRACE("solve", 0);

    auto [N, nx, nu, ny, ny_N] = storage.dim;
    const auto be              = settings.preferred_backend;
    scalar_mut_real_view Δλ_scal{{
        .data  = storage.Δλ_scalar.data(),
        .depth = N + 1,
        .rows  = nx,
        .cols  = 1,
    }};
    assert(storage.Δλ_scalar.size() == static_cast<size_t>(Δλ_scal.size()));
    auto &Δλ1 = storage.λ1;
    auto LΨd = storage.LΨd_scalar(), LΨs = storage.LΨs_scalar();

    // Parallel solve Hv = g
    // ---------------------
    auto solve_H1 = [&, this](index_t i) {
        KOQKATOO_TRACE("solve Hv=g", i);
        // Initialize rhs: g = ∇ϕ + Mᵀλ = ∇f̃ + Aᵀŷ + Mᵀλ                 (d ← g)
        for (index_t j = 0; j < nx + nu; ++j)
            for (index_t ii = i * simd_stride; ii < (i + 1) * simd_stride; ++ii)
                d(ii, j, 0) = grad(ii, j, 0) + Mᵀλ(ii, j, 0) + Aᵀŷ(ii, j, 0);
        // Solve Lᴴ vʹ = g                                              (d ← vʹ)
        compact_blas::xtrsm_LLNN(LH().batch(i), d.batch(i), be);
        // Solve Lᴴ⁻ᵀ v = vʹ                                             (d ← v)
        compact_blas::xtrsm_LLTN(LH().batch(i), d.batch(i), be);
        // Compute f = (A B) v                                          (Δλ ← f)
        if (i < AB().num_batches())
            compact_blas::xgemm(AB().batch(i), d.batch(i), Δλ.batch(i), be);
    };

    // Forward substitution Ψ
    // ----------------------
    auto solve_ψ_fwd = [&](index_t i) {
        KOQKATOO_TRACE("solve ψ fwd", i);
        if (i == 0) {
            // Initialize rhs r - v = Mx - b - v                 (Δλ_scal ← ...)
            for (index_t j = 0; j < nx; ++j)
                Δλ_scal(0, j, 0) = Mxb(0, j, 0) - d(0, j, 0);
        } else {
            // Initialize rhs r + f - v = Mx - b + (A B) v - v   (Δλ_scal ← ...)
            for (index_t j = 0; j < nx; ++j)
                Δλ_scal(i, j, 0) = Mxb(i, j, 0) - d(i, j, 0) + Δλ(i - 1, j, 0);
            // Subtract L(s) Δλʹ(i - 1)                          (Δλ_scal ← ...)
            scalar_blas::xgemm_sub(LΨs.batch(i - 1), Δλ_scal.batch(i - 1),
                                   Δλ_scal.batch(i), be);
        }
        // Solve L(d) Δλʹ = r + f - v - L(s) Δλʹ(i - 1)      (Δλ_scal ← ...)
        scalar_blas::xtrsm_LLNN(LΨd.batch(i), Δλ_scal.batch(i), be);
    };

    // Backward substitution Ψ
    // -----------------------
    auto solve_ψ_rev = [&](index_t i) {
        KOQKATOO_TRACE("solve ψ rev", i);
        if (i < N)
            scalar_blas::xgemm_TN_sub(LΨs.batch(i), Δλ_scal.batch(i + 1),
                                      Δλ_scal.batch(i), be);
        scalar_blas::xtrsm_LLTN(LΨd.batch(i), Δλ_scal.batch(i), be);
        if (i > 0)
            for (index_t j = 0; j < nx; ++j)
                Δλ1(i - 1, j, 0) = Δλ(i, j, 0) = Δλ_scal(i, j, 0);
        else
            for (index_t j = 0; j < nx; ++j)
                Δλ(0, j, 0) = Δλ_scal(0, j, 0);
    };

    // Parallel solve Hd = -g - MᵀΔλ
    // -----------------------------
    auto solve_H2 = [&](index_t i) {
        KOQKATOO_TRACE("solve Hd=-g-MᵀΔλ", i);
        MᵀΔλ.batch(i).top_rows(nx) = Δλ.batch(i);
        MᵀΔλ.batch(i).bottom_rows(nu).set_constant(0);
        if (i < AB().num_batches())
            compact_blas::xgemm_TN_sub(AB().batch(i), Δλ1.batch(i),
                                       MᵀΔλ.batch(i), be);
        for (index_t j = 0; j < nx + nu; ++j)
            for (index_t ii = i * simd_stride; ii < (i + 1) * simd_stride; ++ii)
                d(ii, j, 0) = -grad(ii, j, 0) - Mᵀλ(ii, j, 0) - Aᵀŷ(ii, j, 0) -
                              MᵀΔλ(ii, j, 0);
        compact_blas::xtrsm_LLNN(LH().batch(i), d.batch(i), be);
        compact_blas::xtrsm_LLTN(LH().batch(i), d.batch(i), be);
    };

    index_t num_batch   = (N + 1 + simd_stride - 1) / simd_stride;
    auto &join_counters = storage.reset_join_counters();
    // process_fwd returns true if last batch was processed
    const auto process_fwd = [&](index_t k) {
        solve_H1(k);
        while (true) {
            if (k > 0 && join_counters[k - 1].value.fetch_add(1) != 1)
                return false;
            index_t i_end = std::min((k + 1) * simd_stride, N + 1);
            for (index_t i = k * simd_stride; i < i_end; ++i)
                solve_ψ_fwd(i);
            ++k;
            if (k == num_batch)
                return true;
        }
    };
#if KOQKATOO_SOLVE_USE_LIBFORK
    const auto process_all = [](auto process_all, auto process_fwd,
                                auto solve_ψ_rev, auto solve_H2, index_t N,
                                auto simd_stride) -> lf::task<void> {
        index_t num_batch = (N + 1 + simd_stride - 1) / simd_stride;
        for (index_t k = 0; k < num_batch; ++k)
            co_await lf::fork[do_invoke](process_fwd, k);
        co_await lf::join;
        for (index_t i = N + 1; i-- > 0;) {
            try {
                solve_ψ_rev(i);
            } catch (...) {
                process_all.stash_exception();
            }
            if (i % simd_stride == 0)
                co_await lf::fork[do_invoke](solve_H2, i / simd_stride);
        }
        co_await lf::join;
    };
    lf::sync_wait(*fork_pool, process_all, process_fwd, solve_ψ_rev, solve_H2,
                  N, std::integral_constant<index_t, simd_stride>());
#else
#if USE_ATOMIC_WAIT
    static_assert(!KOQKATOO_WITH_OPENMP);
    std::atomic<bool> main_thread_flag{false};
    std::atomic<index_t> batch_counter_1{0}, stage_counter_2{num_batch - 1},
        stage_ready{num_batch};
    std::latch join_all{static_cast<ptrdiff_t>(pool.size())};
    auto thread_work = [&] {
        while (true) {
            index_t k = batch_counter_1.fetch_add(1, std::memory_order_relaxed);
            if (k >= num_batch)
                break;
            process_fwd(k);
        }
        join_all.arrive_and_wait();
        bool main_thread =
            !main_thread_flag.exchange(true, std::memory_order_relaxed);
        if (main_thread) {
            for (index_t i = N + 1; i-- > 0;) {
                solve_ψ_rev(i);
                if (i % simd_stride == 0) {
                    KOQKATOO_TRACE("solve store-notify", i);
                    stage_ready.store(i / simd_stride,
                                      std::memory_order_release);
                    stage_ready.notify_one();
                }
            }
        }
        while (true) {
            index_t k = stage_counter_2.fetch_sub(1, std::memory_order_relaxed);
            if (k < 0)
                break;
            index_t k_ready = stage_ready.load(std::memory_order_acquire);
            while (k < k_ready) {
                stage_ready.wait(k_ready, std::memory_order_relaxed);
                k_ready = stage_ready.load(std::memory_order_acquire);
            }
            solve_H2(k);
        }
    };
#else
    std::atomic<index_t> batch_counter_1{0}, stage_counter_2{num_batch - 1};
    KOQKATOO_ASSERT(std::in_range<int>(num_batch));
    std::atomic<int> stage_ready{0};
    auto thread_work = [&](int num_threads) {
        bool main_thread = false;
        while (true) {
            index_t k = batch_counter_1.fetch_add(1, std::memory_order_relaxed);
            if (k >= num_batch)
                break;
            main_thread = process_fwd(k);
        }
        if (main_thread) {
            for (index_t i = N + 1; i-- > 0;) {
                solve_ψ_rev(i);
                if (i % simd_stride == 0) {
                    KOQKATOO_TRACE("solve store-notify", i);
                    stage_ready.fetch_add(1, std::memory_order_release);
                    stage_ready.notify_one();
                }
            }
            stage_ready.fetch_add(num_threads, std::memory_order_release);
            stage_ready.notify_all();
        }
        while (true) {
            auto semaphore_val = stage_ready.load(std::memory_order_relaxed);
            do {
                while (semaphore_val == 0) {
                    stage_ready.wait(semaphore_val, std::memory_order_relaxed);
                    semaphore_val = stage_ready.load(std::memory_order_relaxed);
                }
            } while (!stage_ready.compare_exchange_strong(
                semaphore_val, semaphore_val - 1, std::memory_order_acquire));
            index_t k = stage_counter_2.fetch_sub(1, std::memory_order_relaxed);
            if (k < 0)
                break;
            solve_H2(k);
        }
    };
#endif
#if KOQKATOO_WITH_OPENMP
    KOQKATOO_OMP(parallel) {
        int n_thr = omp_get_num_threads();
        KOQKATOO_OMP(for schedule(static))
        for (int i = 0; i < n_thr; ++i)
            thread_work(n_thr);
    }
#else
    pool->sync_run_all([&] { thread_work(static_cast<int>(pool->size())); });
#endif
#endif
}

template <simd_abi_tag Abi>
void Solver<Abi>::recompute_inner(real_t S, real_view x0, real_view x,
                                  real_view λ, real_view q,
                                  mut_real_view grad_f, mut_real_view Ax,
                                  mut_real_view Mᵀλ) {
    KOQKATOO_TRACE("recompute_inner", 0);

    auto [N, nx, nu, ny, ny_N] = storage.dim;
    const auto be              = settings.preferred_backend;
    using simd                 = types::simd;
    static constexpr auto al   = stdx::vector_aligned;
    simd invS{1 / S};
    const auto process_stage = [&](index_t k) {
        auto qi = q.batch(k), xi = x.batch(k), x0i = x0.batch(k);
        auto grad_fi = grad_f.batch(k), Axi = Ax.batch(k);
        for (index_t j = 0; j < x.rows(); ++j) {
            simd qij{&qi(0, j, 0), al}, xij{&xi(0, j, 0), al},
                x0ij{&x0i(0, j, 0), al};
            simd grad_fij = invS * (xij - x0ij) + qij;
            // q + S⁻¹(x - x0)
            grad_fij.copy_to(&grad_fi(0, j, 0), al);
        }
        // Qx + q + S⁻¹(x - x0)
        compact_blas::xsymv_add(H().batch(k), xi, grad_fi, be);
        // Cx + Du
        compact_blas::xgemm(CD().batch(k), xi, Axi, be);
        // Mᵀλ(i) = [I 0]ᵀ λ(i)
        Mᵀλ.batch(k).bottom_rows(nu).set_constant(0);
        compact_blas::xcopy(λ.batch(k), Mᵀλ.batch(k).top_rows(nx));
        if (k * simd_stride < N) {
            // Shift λ by one time step
            const auto i_end = std::min((k + 1) * simd_stride, N);
            for (index_t r = 0; r < nx; ++r)
                for (index_t i = k * simd_stride; i < i_end; ++i)
                    storage.λ1(i, r, 0) = λ(i + 1, r, 0);
            // Mᵀλ(i) = -[A B]ᵀ(i) λ(i+1) + [I 0]ᵀ λ(i)
            compact_blas::xgemm_TN_sub(AB().batch(k), storage.λ1.batch(k),
                                       Mᵀλ.batch(k), be);
        }
    };
#if KOQKATOO_SOLVE_USE_LIBFORK
    const auto process_all = [](auto, auto process_stage, index_t N,
                                auto simd_stride) -> lf::task<void> {
        index_t num_batch = (N + 1 + simd_stride - 1) / simd_stride;
        for (index_t k = 0; k < num_batch; ++k)
            co_await lf::fork[do_invoke](process_stage, k);
        co_await lf::join;
    };
    lf::sync_wait(*fork_pool, process_all, process_stage, N,
                  std::integral_constant<index_t, simd_stride>());
#else
    std::atomic<index_t> batch_counter{0};
    const index_t num_batch = (N + 1 + simd_stride - 1) / simd_stride;
    auto thread_work        = [&] {
        while (true) {
            index_t k = batch_counter.fetch_add(1, std::memory_order_relaxed);
            if (k >= num_batch)
                break;
            process_stage(k);
        }
    };
#if KOQKATOO_WITH_OPENMP
    KOQKATOO_OMP(parallel for schedule(static))
    for (int i = 0; i < omp_get_num_threads(); ++i)
        thread_work();
#else
    pool->sync_run_all(thread_work);
#endif
#endif
}

template <simd_abi_tag Abi>
real_t Solver<Abi>::recompute_outer(real_view x, real_view y, real_view λ,
                                    real_view q, mut_real_view grad_f,
                                    mut_real_view Ax, mut_real_view Aᵀy,
                                    mut_real_view Mᵀλ) {
    using std::abs;
    using std::isfinite;
    using std::max;
    KOQKATOO_TRACE("recompute_outer", 0);

    auto [N, nx, nu, ny, ny_N] = storage.dim;
    const auto be              = settings.preferred_backend;
    using simd                 = types::simd;
    static constexpr auto al   = stdx::vector_aligned;
    const auto process_stage   = [&](index_t k) {
        auto qi = q.batch(k), xi = x.batch(k), yi = y.batch(k);
        auto grad_fi = grad_f.batch(k), Axi = Ax.batch(k), Aᵀyi = Aᵀy.batch(k),
             Mᵀλi = Mᵀλ.batch(k);
        // Qx + q
        compact_blas::xcopy(qi, grad_fi);
        compact_blas::xsymv_add(H().batch(k), xi, grad_fi, be);
        // Cx + Du
        compact_blas::xgemm(CD().batch(k), xi, Axi, be);
        compact_blas::xgemm_TN(CD().batch(k), yi, Aᵀyi, be);
        // Mᵀλ(i) = [I 0]ᵀ λ(i)
        Mᵀλ.batch(k).bottom_rows(nu).set_constant(0);
        compact_blas::xcopy(λ.batch(k), Mᵀλ.batch(k).top_rows(nx));
        if (k * simd_stride < N) {
            // Shift λ by one time step
            const auto i_end = std::min((k + 1) * simd_stride, N);
            for (index_t r = 0; r < nx; ++r)
                for (index_t i = k * simd_stride; i < i_end; ++i)
                    storage.λ1(i, r, 0) = λ(i + 1, r, 0);
            // Mᵀλ(i) = -[A B]ᵀ(i) λ(i+1) + [I 0]ᵀ λ(i)
            compact_blas::xgemm_TN_sub(AB().batch(k), storage.λ1.batch(k),
                                         Mᵀλ.batch(k), be);
        }
        // Norm of augmented lagrangian gradient
        if ((k + 1) * simd_stride <= N) {
            // full batches excluding the last stage
            simd inf_norm{0}, l1_norm{0};
            for (index_t r = 0; r < nx + nu; ++r) {
                auto al_gr = simd{&grad_fi(0, r, 0), al} +
                             simd{&Aᵀyi(0, r, 0), al} +
                             simd{&Mᵀλi(0, r, 0), al};
                inf_norm = max(abs(al_gr), inf_norm);
                l1_norm += abs(al_gr);
            }
            auto l1_norm_scalar = reduce(l1_norm);
            return isfinite(l1_norm_scalar) ? hmax(inf_norm) : l1_norm_scalar;
        } else {
            // last batch (special case because final stage only has x, no u)
            real_t inf_norm = 0, l1_norm = 0;
            const auto i_end = N - k * simd_stride;
            for (index_t r = 0; r < nx + nu; ++r) {
                for (index_t i = 0; i <= i_end; ++i) {
                    if (r >= nx && i == i_end)
                        continue;
                    auto gr = grad_fi(i, r, 0) + Aᵀyi(i, r, 0) + Mᵀλi(i, r, 0);
                    inf_norm = max(abs(gr), inf_norm);
                    l1_norm += abs(gr);
                }
            }
            return isfinite(l1_norm) ? inf_norm : l1_norm;
        }
    };
#if KOQKATOO_SOLVE_USE_LIBFORK
    const auto process_all = [](auto, auto process_stage, index_t N,
                                auto simd_stride) -> lf::task<real_t> {
        index_t num_batch = (N + 1 + simd_stride - 1) / simd_stride;
        std::vector<real_t> stage_norms(num_batch);
        for (index_t k = 0; k < num_batch; ++k) {
            co_await lf::fork[&stage_norms[k], do_invoke](process_stage, k);
        }
        co_await lf::join;
        real_t inf_norm = 0, l1_norm = 0;
        for (auto stage_norm : stage_norms) {
            inf_norm = std::max(std::abs(stage_norm), inf_norm);
            l1_norm += std::abs(stage_norm);
        }
        co_return isfinite(l1_norm) ? inf_norm : l1_norm;
    };
    return lf::sync_wait(*fork_pool, process_all, process_stage, N,
                         std::integral_constant<index_t, simd_stride>());
#else
    std::atomic<index_t> batch_counter{0};
    const index_t num_batch = (N + 1 + simd_stride - 1) / simd_stride;
    std::vector<real_t> stage_norms(num_batch);
    auto thread_work = [&] {
        while (true) {
            index_t k = batch_counter.fetch_add(1, std::memory_order_relaxed);
            if (k >= num_batch)
                break;
            stage_norms[k] = process_stage(k);
        }
    };
#if KOQKATOO_WITH_OPENMP
    KOQKATOO_OMP(parallel for schedule(static))
    for (int i = 0; i < omp_get_num_threads(); ++i)
        thread_work();
#else
    pool->sync_run_all(thread_work);
#endif

    real_t inf_norm = 0, l1_norm = 0;
    for (auto stage_norm : stage_norms) {
        inf_norm = std::max(std::abs(stage_norm), inf_norm);
        l1_norm += std::abs(stage_norm);
    }
    return isfinite(l1_norm) ? inf_norm : l1_norm;
#endif
}

template <simd_abi_tag Abi>
void Solver<Abi>::updowndate_fork(real_view Σ, bool_view J_old, bool_view J_new,
                                  Timings *t) {
    std::optional<guanaqo::Timed<typename Timings::type>> t_total;
    if (t)
        t_total.emplace(t->updowndate);
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    auto &ranks                = storage.stagewise_update_counts;
    index_t rj_max             = 0;
    for (index_t k = 0; k <= N; ++k) {
        index_t rj  = 0;
        index_t nyk = k == N ? ny_N : ny;
        for (index_t r = 0; r < nyk; ++r) {
            bool up   = J_new(k, r, 0) && !J_old(k, r, 0);
            bool down = !J_new(k, r, 0) && J_old(k, r, 0);
            if (up || down) {
                storage.Σ_sgn(k, rj, 0) = up ? +real_t(0) : -real_t(0);
                storage.Σ_ud(k, rj, 0)  = Σ(k, r, 0);
                index_t nxuk            = k == N ? nx : nx + nu;
                for (index_t c = 0; c < nxuk; ++c) // TODO: pre-transpose CD?
                    storage.Z(k, c, rj) = CD()(k, r, c);
                ++rj;
            }
        }
        ranks(k, 0, 0) = rj;
        rj_max         = std::max(rj_max, rj);
    }
    if (rj_max == 0)
        return;

    const auto updowndate_HVW =
        [&](single_mut_real_view HV, single_mut_real_view Wᵀ,
            single_mut_real_view A, single_real_view D) {
            using namespace linalg::compact::micro_kernels::shhud_diag;
            static constexpr index_constant<SizeR> R;
            static constexpr index_constant<SizeS> S;

            using W_t = triangular_accessor<Abi, real_t, R>;
            alignas(W_t::alignment()) real_t W[W_t::size()];

            // Process all diagonal blocks (in multiples of R, except the last).
            foreach_chunked_merged(0, HV.cols(), R, [&](index_t k, auto rem_k) {
                // Part of A corresponding to this diagonal block
                // TODO: packing
                auto Ad = A.middle_rows(k, rem_k);
                auto Ld = HV.block(k, k, rem_k, rem_k);
                // Process the diagonal block itself
                microkernel_diag_lut<Abi>[rem_k - 1](A.cols(), W, Ld, Ad, D);
                // Process all rows below the diagonal block (in multiples of S).
                foreach_chunked_merged(
                    k + rem_k, HV.rows(), S,
                    [&](index_t i, auto rem_i) {
                        auto As = A.middle_rows(i, rem_i);
                        auto Ls = HV.block(i, k, rem_i, rem_k);
                        microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                            A.cols(), W, Ls, As, Ad, D, false);
                    },
                    LoopDir::Backward); // TODO: decide on order
                foreach_chunked_merged(
                    0, Wᵀ.cols(), S, [&](index_t iw, auto rem_i) {
                        index_t i = HV.rows() + iw;
                        auto As   = A.middle_rows(i, rem_i);
                        auto Ls   = Wᵀ.block(k, iw, rem_k, rem_i);
                        microkernel_tail_lut_2<Abi>[rem_k - 1][rem_i - 1](
                            A.cols(), W, Ls, As, Ad, D, true);
                    });
            });
        };

    const auto updowndate_stage = [&](index_t batch_idx, index_t rj_min,
                                      index_t rj_batch, auto ranksj) {
        auto Σj    = storage.Σ_ud.batch(batch_idx).top_rows(rj_batch);
        auto Sj    = storage.Σ_sgn.batch(batch_idx).top_rows(rj_batch);
        auto Zj    = storage.Z.batch(batch_idx).left_cols(rj_batch);
        auto Z1j   = storage.Z1.batch(batch_idx).left_cols(rj_batch);
        auto Luj   = storage.Lupd.batch(batch_idx).top_left(rj_batch, rj_batch);
        auto Wuj   = storage.Wupd.batch(batch_idx).left_cols(rj_batch);
        using simd = typename types::simd;
#ifdef _GLIBCXX_EXPERIMENTAL_SIMD // TODO: need general way to convert masks
        using index_simd = typename types::index_simd;
        index_simd rjks{ranksj.data, stdx::vector_aligned};
        for (index_t rj = rj_min; rj < rj_batch; ++rj) {
            index_simd rjs{rj};
            auto z0_mask = rjs >= rjks;
            simd zero{0};
            constexpr auto vec_align = stdx::vector_aligned;
            for (index_t r = 0; r < nx + nu; ++r)
                where(z0_mask.__cvt(), zero).copy_to(&Zj(0, r, rj), vec_align);
            using std::sqrt;
            zero = simd{sqrt(std::numeric_limits<real_t>::min())}; // TODO
            // TODO: why does zero sometimes result in NaN? At first sight, inf
            //       arithmetic should also work (i.e. Σ⁻¹ = ∞), but apparently
            //       I'm missing something.
            where(z0_mask.__cvt(), zero).copy_to(&Σj(0, rj, 0), vec_align);
        }
#else
#error "Fallback not yet implemented"
#endif
        const auto be = settings.preferred_backend;
        // Copy Z
        compact_blas::xcopy(Zj, Z1j.top_rows(nx + nu));
        compact_blas::xfill(real_t(0), Z1j.bottom_rows(2 * nx)); // TODO
        // Z ← L⁻¹ Z
        compact_blas::xtrsm_LLNN(LH().batch(batch_idx), Zj, be);
        // Lu ← ZᵀZ ± Σ⁻¹
        compact_blas::xsyrk_T(Zj, Luj, be);
        for (index_t r = 0; r < rj_batch; ++r) {
            simd Lujrr{&Luj(0, r, r), stdx::vector_aligned};
            simd Σjr{&Σj(0, r, 0), stdx::vector_aligned};
            simd Sjr{&Sj(0, r, 0), stdx::vector_aligned};
            Lujrr += cneg(1 / Σjr, Sjr);
            Lujrr.copy_to(&Luj(0, r, r), stdx::vector_aligned);
        }
        // Lu ← chol(ZᵀZ ± Σ⁻¹)
        compact_blas::xpntrf(Luj, Sj);
        // Z ← Z Lu⁻ᵀ
        compact_blas::xtrsm_RLTN(Luj, Zj, be);
        // Wu ← W Z
        compact_blas::xgemm_TN(Wᵀ().batch(batch_idx), Zj, Wuj, be);
        if (batch_idx < AB().num_batches()) {
            // Vu ← -V Z
            auto Vuj = storage.Vupd.batch(batch_idx).left_cols(rj_batch);
            compact_blas::xgemm_neg(V().batch(batch_idx), Zj, Vuj, be);
        }
    };

    const auto updowndate_stage_H = [&](index_t batch_idx, index_t rj_batch) {
        auto Σj    = storage.Σ_ud.batch(batch_idx).top_rows(rj_batch);
        auto Sj    = storage.Σ_sgn.batch(batch_idx).top_rows(rj_batch);
        auto Z1j   = storage.Z1.batch(batch_idx).left_cols(rj_batch);
        using simd = typename types::simd;
        for (index_t r = 0; r < rj_batch; ++r) {
            simd s{&Sj(0, r, 0), stdx::vector_aligned};
            simd σ{&Σj(0, r, 0), stdx::vector_aligned};
            cneg(σ, s).copy_to(&Σj(0, r, 0), stdx::vector_aligned);
        }
        // Update LH and V
        // TODO: no need to update V in last stage.
        updowndate_HVW(LHV().batch(batch_idx), Wᵀ().batch(batch_idx), Z1j, Σj);
        // TODO: should we always eagerly update V and W?
#ifndef NDEBUG
        auto LHi = LH().batch(batch_idx);
        // Check finiteness of LH(i)
        assert(N + 1 > batch_idx * simd_stride);
        auto i_end =
            std::min<index_t>(LHi.depth(), N + 1 - batch_idx * simd_stride);
        for (index_t i = 0; i < i_end; ++i) {
            for (index_t c = 0; c < LHi.rows(); ++c)
                for (index_t r = c; r < LHi.rows(); ++r)
                    if (!std::isfinite(LHi(i, r, c)))
                        throw std::runtime_error(std::format(
                            "inf value of LHi: {} at ({}, {}, {})",
                            LHi(i, r, c), batch_idx * simd_stride + i, r, c));
        }
#endif
    };

    // Each block of V generates fill-in in A in the block row below it, and
    // W adds new columns to A, but we only ever need two block rows of A at any
    // given time, thanks to the block-tridiagonal structure of Ψ, so we use a
    // sliding window of depth 2, and use the stage index modulo 2.
    index_t colsA = 0;
    auto &A = storage.A_ud, &D = storage.D_ud;
    using namespace cholundate;
    static constexpr index_constant<householder::DefaultSizeR> R;
    static constexpr index_constant<householder::DefaultSizeS> S;
    using W_t = micro_kernels::householder::matrix_W_storage<>;

    constinit static const auto microkernel_diag_lut =
        make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
            return householder::updowndate_diag<NR + 1, DownUpdate>;
        });
    constinit static const auto microkernel_full_lut =
        make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
            return householder::updowndate_full<NR + 1, DownUpdate>;
        });
    constinit static const auto microkernel_tail_lut =
        make_1d_lut<R>([]<index_t NR>(index_constant<NR>) {
            constexpr micro_kernels::householder::Config uConf{
                .block_size_r = NR + 1, .block_size_s = S};
            return householder::updowndate_tile_tail<uConf, DownUpdate>;
        });

    auto process_diag_block = [&](index_t k, auto rem_k, auto Ad, auto Dd,
                                  auto Ld, W_t &W) {
        auto Add = Ad.middle_rows(k, rem_k);
        auto Ldd = Ld.block(k, k, rem_k, rem_k);
        microkernel_diag_lut[rem_k - 1](colsA, W, Ldd, Add, DownUpdate{Dd});
    };
    auto process_subdiag_block_W = [&](index_t k, auto Ad, auto Dd, auto Ld,
                                       W_t &W) {
        auto Add = Ad.middle_rows(k, R);
        foreach_chunked_merged(k + R, Ld.rows, S, [&](index_t i, auto rem_i) {
            auto Ads = Ad.middle_rows(i, rem_i);
            auto Lds = Ld.block(i, k, rem_i, R);
            microkernel_tail_lut[R - 1](rem_i, 0, colsA, W, Lds, Add, Ads,
                                        DownUpdate{Dd});
        });
    };
    auto process_subdiag_block_V = [&](index_t k, index_t rem_k, index_t colsA0,
                                       auto As, auto Ls, auto Ad, auto Dd,
                                       W_t &W) {
        auto Add = Ad.middle_rows(k, rem_k);
        foreach_chunked_merged(0, Ls.rows, S, [&](index_t i, auto rem_i) {
            auto Ass = As.middle_rows(i, rem_i);
            auto Lss = Ls.block(i, k, rem_i, rem_k);
            microkernel_tail_lut[rem_k - 1](rem_i, colsA0, colsA, W, Lss, Add,
                                            Ass, DownUpdate{Dd});
        });
    };

    const auto updowndate_Ψ_stage_last = [&] {
        // Final stage has no subdiagonal block (V)
        index_t rN = ranks(N, 0, 0);
        colsA += rN;
        if (colsA > 0) {
            auto Ad = A(N % 2).left_cols(colsA), Ld = LΨd_scalar()(N);
            Ad.right_cols(rN) = storage.Wupd(N).left_cols(rN);
            W_t W;
            auto Dd            = D(0).top_rows(colsA);
            Dd.bottom_rows(rN) = storage.Σ_sgn(N).top_rows(rN);
            std::span<real_t> Dd_spn{Dd.data, static_cast<size_t>(Dd.rows)};

            // Process all diagonal blocks (in multiples of R, except the last).
            foreach_chunked(
                0, Ld.cols, R,
                [&](index_t k) {
                    process_diag_block(k, R, Ad, Dd_spn, Ld, W);
                    process_subdiag_block_W(k, Ad, Dd_spn, Ld, W);
                },
                [&](index_t k, index_t rem_k) {
                    auto Add = Ad.middle_rows(k, rem_k);
                    auto Ldd = Ld.block(k, k, rem_k, rem_k);
                    microkernel_full_lut[rem_k - 1](colsA, Ldd, Add,
                                                    DownUpdate{Dd_spn});
                });
        }
    };

    const auto updowndate_Ψ_stage = [&](index_t k) {
        if (k == N)
            return updowndate_Ψ_stage_last();
        index_t rk  = ranks(k, 0, 0);
        auto colsA0 = colsA;
        colsA += rk;
        if (colsA == 0)
            return;
        auto Ad = A(k % 2).left_cols(colsA), Ld = LΨd_scalar()(k);
        auto As = A((k + 1) % 2).left_cols(colsA), Ls = LΨs_scalar()(k);
        Ad.right_cols(rk) = storage.Wupd(k).left_cols(rk);
        As.right_cols(rk) = storage.Vupd(k).left_cols(rk);
        W_t W;
        auto Dd            = D(0).top_rows(colsA);
        Dd.bottom_rows(rk) = storage.Σ_sgn(k).top_rows(rk);
        std::span<real_t> Dd_spn{Dd.data, static_cast<size_t>(Dd.rows)};

        // Process all diagonal blocks (in multiples of R, except the last).
        foreach_chunked(
            0, Ld.cols, R,
            [&](index_t k) {
                auto c0 = k == 0 ? colsA0 : 0;
                process_diag_block(k, R, Ad, Dd_spn, Ld, W);
                process_subdiag_block_W(k, Ad, Dd_spn, Ld, W);
                process_subdiag_block_V(k, R, c0, As, Ls, Ad, Dd_spn, W);
            },
            [&](index_t k, index_t rem_k) {
                auto c0 = k == 0 ? colsA0 : 0;
                process_diag_block(k, rem_k, Ad, Dd_spn, Ld, W);
                process_subdiag_block_V(k, rem_k, c0, As, Ls, Ad, Dd_spn, W);
            });
    };

    const auto updowndate_Ψ_batch = [&](index_t batch_idx) {
        KOQKATOO_TRACE("updowndate Ψ", batch_idx);
        index_t k0 = batch_idx * simd_stride;
        index_t k1 = std::min(k0 + simd_stride, N + 1);
        for (index_t k = k0; k < k1; ++k)
            updowndate_Ψ_stage(k);
    };

    const index_t num_batch  = (N + 1 + simd_stride - 1) / simd_stride;
    auto &join_counters      = storage.reset_join_counters();
    const auto process_stage = [&](index_t batch_idx, index_t rj_min,
                                   index_t rj_batch, auto ranksj) {
        if (rj_batch > 0) {
            KOQKATOO_TRACE("updowndate", batch_idx);
            updowndate_stage(batch_idx, rj_min, rj_batch, ranksj);
        }
        auto ready = join_counters[batch_idx].value.fetch_or(2);
        join_counters[batch_idx].value.notify_one();
        // Whoever sets join_counter to 3 wins the race and gets to handle
        // the update of Ψ
        if (ready == 1 || batch_idx == 0) {
            auto bi = batch_idx;
            while (true) {
                updowndate_Ψ_batch(bi);
                if (++bi == num_batch)
                    break;
                // Indicate that this block of Ψ is done
                if (join_counters[bi].value.fetch_or(1) == 0)
                    // If the corresponding preparation is not done yet,
                    // we can't process the next block of Ψ yet.
                    break;
            };
        }
    };

    const auto process_stage_H = [&](index_t batch_idx, index_t rj_batch) {
        KOQKATOO_TRACE("updowndate H", batch_idx);
        if (rj_batch <= 0)
            return;
        auto &join_counter = join_counters[batch_idx].value;
        // Wait for the preparation to be finished (join_counter >= 2)
        while (true) {
            auto jc = join_counter.load();
            if (jc >= 2)
                break;
            join_counter.wait(jc);
        }
        updowndate_stage_H(batch_idx, rj_batch);
    };

    std::atomic<index_t> batch_counter{0};
    auto thread_work = [&] {
        while (true) {
            index_t b = batch_counter.fetch_add(1, std::memory_order_relaxed);
            if (b < num_batch) {
                auto batch_idx = b;
                auto stage_idx = batch_idx * simd_stride;
                auto nk     = std::min<index_t>(simd_stride, N + 1 - stage_idx);
                auto ranksj = ranks.batch(batch_idx);
                auto rjmm = std::minmax_element(ranksj.data, ranksj.data + nk);
                index_t rj_min = *rjmm.first, rj_batch = *rjmm.second;
                process_stage(batch_idx, rj_min, rj_batch, ranksj);
            } else if (b < 2 * num_batch) {
                // Looping backwards may give slightly better cache locality
                auto batch_idx = 2 * num_batch - 1 - b;
                auto stage_idx = batch_idx * simd_stride;
                auto nk     = std::min<index_t>(simd_stride, N + 1 - stage_idx);
                auto ranksj = ranks.batch(batch_idx);
                auto rj_batch = std::max_element(ranksj.data, ranksj.data + nk);
                process_stage_H(batch_idx, *rj_batch);
            } else {
                break;
            }
        }
    };

    std::optional<guanaqo::Timed<typename Timings::type>> t_w;
    if (t)
        t_w.emplace(t->updowndate_stages);
#if KOQKATOO_WITH_OPENMP
        KOQKATOO_OMP(parallel for schedule(static))
        for (int i = 0; i < omp_get_num_threads(); ++i)
            thread_work();
#else
    pool->sync_run_all(thread_work);
#endif
}

} // namespace koqkatoo::ocp
