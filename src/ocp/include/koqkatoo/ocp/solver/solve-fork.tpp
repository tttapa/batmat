#pragma once

#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/trace.hpp>
#include <libfork/core.hpp>
#include <libfork/schedule.hpp>
#include <atomic>

namespace koqkatoo::ocp {

inline constexpr auto do_invoke =
    []<class F, class... Args>(
        auto, F fun,
        Args... args) -> lf::task<std::invoke_result_t<F, Args...>> {
    co_return std::invoke(std::move(fun), std::move(args)...);
};

inline lf::lazy_pool pool{4}; // TODO: make configurable/argument

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

    const auto process_all = [](auto, auto process_stage,
                                index_t num_batch) -> lf::task<void> {
        for (index_t k = 0; k < num_batch; ++k)
            co_await lf::fork[do_invoke](process_stage, k);
        co_await lf::join;
    };
    KOQKATOO_TRACE("factor wait", 0);
    lf::sync_wait(pool, process_all, process_stage, num_batch);
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
    const auto process_fwd = [&](index_t k) {
        solve_H1(k);
        while (k < num_batch) {
            if (k > 0 && join_counters[k - 1].value.fetch_add(1) != 1)
                break;
            index_t i_end = std::min((k + 1) * simd_stride, N + 1);
            for (index_t i = k * simd_stride; i < i_end; ++i)
                solve_ψ_fwd(i);
            ++k;
        }
    };
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
    KOQKATOO_TRACE("solve wait", 0);
    lf::sync_wait(pool, process_all, process_fwd, solve_ψ_rev, solve_H2, N,
                  std::integral_constant<index_t, simd_stride>());
}

} // namespace koqkatoo::ocp
