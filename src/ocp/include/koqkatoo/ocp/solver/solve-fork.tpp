#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/thread-pool.hpp>
#include <koqkatoo/trace.hpp>
#include <libfork/core.hpp>
#include <libfork/schedule.hpp>
#include <algorithm>
#include <atomic>
#include <functional>
#include <stdexcept>

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

struct join_counter_t {
    alignas(64) std::atomic<int> value{};
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
    index_t num_batch = (N + 1 + simd_stride - 1) / simd_stride;
    std::vector<join_counter_t> join_counters(num_batch - 1);
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
    pool->sync_run_all(thread_work);
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
    auto solve_ψ_fwd = [&, this](index_t i) {
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
    auto solve_ψ_rev = [&, this](index_t i) {
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
    auto solve_H2 = [&, this](index_t i) {
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

    index_t num_batch = (N + 1 + simd_stride - 1) / simd_stride;
    std::vector<join_counter_t> join_counters(num_batch - 1);
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
    auto thread_work = [&] {
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
            stage_ready.fetch_add(static_cast<int>(pool->size()),
                                  std::memory_order_release);
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
    pool->sync_run_all(thread_work);
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
        auto qi = q.batch(k), grad_fi = grad_f.batch(k), xi = x.batch(k),
             x0i = x0.batch(k), Axi = Ax.batch(k);
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
    pool->sync_run_all(thread_work);
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
        auto qi = q.batch(k), grad_fi = grad_f.batch(k), xi = x.batch(k),
             Axi = Ax.batch(k), yi = y.batch(k), Aᵀyi = Aᵀy.batch(k),
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
    pool->sync_run_all(thread_work);

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
    KOQKATOO_TRACE("updowndate", 0);

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

    const auto updowndate_stage = [&](index_t batch_idx, index_t rj_min,
                                      index_t rj_batch, auto ranksj) {
        KOQKATOO_TRACE("updowndate", batch_idx);
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
        compact_blas::xfill(real_t(0), Z1j.bottom_rows(nx));
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
        for (index_t r = 0; r < rj_batch; ++r) {
            simd s{&Sj(0, r, 0), stdx::vector_aligned};
            simd σ{&Σj(0, r, 0), stdx::vector_aligned};
            cneg(σ, s).copy_to(&Σj(0, r, 0), stdx::vector_aligned);
        }
        // Update LH and V
        compact_blas::xshhud_diag(LHV().batch(batch_idx), Z1j, Σj, be);
        auto LHi = LH().batch(batch_idx), Wi = Wᵀ().batch(batch_idx);
        compact_blas::xcopy(LHi.top_left(nx + nu, nx), Wi);
        compact_blas::xtrtri(Wi, be);
        compact_blas::xtrsm_LLNN(LHi.bottom_right(nu, nu), Wi.bottom_rows(nu),
                                 be);
        // TODO: should we always eagerly update V and W?
#ifndef NDEBUG
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

#if KOQKATOO_SOLVE_USE_LIBFORK
    const auto process_all = [](auto, auto updowndate_stage, index_t N,
                                auto simd_stride,
                                auto &ranks) -> lf::task<void> {
        for (index_t k = 0; k < N + 1; k += simd_stride) {
            auto nk           = std::min<index_t>(simd_stride, N + 1 - k);
            index_t batch_idx = k / simd_stride;
            auto ranksj       = ranks.batch(batch_idx);
            auto rjmm      = std::minmax_element(ranksj.data, ranksj.data + nk);
            index_t rj_min = *rjmm.first, rj_batch = *rjmm.second;
            if (rj_batch != 0)
                co_await lf::fork[do_invoke](updowndate_stage, batch_idx,
                                             rj_min, rj_batch, ranksj);
        }
        co_await lf::join;
    };

    {
        std::optional<guanaqo::Timed<typename Timings::type>> t_w;
        if (t)
            t_w.emplace(t->updowndate_stages);
        lf::sync_wait(*fork_pool, process_all, updowndate_stage, N,
                      std::integral_constant<index_t, simd_stride>(), ranks);
    }
#else
    std::atomic<index_t> batch_counter{0};
    const index_t num_batch = (N + 1 + simd_stride - 1) / simd_stride;
    auto thread_work        = [&] {
        while (true) {
            index_t batch_idx =
                batch_counter.fetch_add(1, std::memory_order_relaxed);
            if (batch_idx >= num_batch)
                break;
            auto stage_idx = batch_idx * simd_stride;
            auto nk        = std::min<index_t>(simd_stride, N + 1 - stage_idx);
            auto ranksj    = ranks.batch(batch_idx);
            auto rjmm      = std::minmax_element(ranksj.data, ranksj.data + nk);
            index_t rj_min = *rjmm.first, rj_batch = *rjmm.second;
            if (rj_batch != 0)
                updowndate_stage(batch_idx, rj_min, rj_batch, ranksj);
        }
    };
    pool->sync_run_all(thread_work);
#endif

    {
        std::optional<guanaqo::Timed<typename Timings::type>> t_w;
        if (t)
            t_w.emplace(t->updowndate_Ψ);
        updowndate_ψ();
    }
}

} // namespace koqkatoo::ocp
