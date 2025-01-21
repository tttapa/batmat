#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/fork-pool.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/thread-pool.hpp>
#include <koqkatoo/trace.hpp>
#include <algorithm>
#include <atomic>

#include <koqkatoo/cholundate/householder-updowndate-common.tpp>
#include <koqkatoo/cholundate/householder-updowndate.hpp>
#include <koqkatoo/cholundate/micro-kernels/householder-updowndate.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/xshhud-diag.hpp> // TODO
#include <koqkatoo/linalg/small-potrf.hpp>

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
        compact_blas::unpack(storage.VWᵀ().batch(k), LΨsk);
        // TODO: exploit trapezoidal shape of Wᵀ
        compact_blas::xsyrk(V().batch(k), VVᵀ().batch(k), be);
        compact_blas::unpack_L(VVᵀ().batch(k), VVᵀk);
    }
    auto LΨdk = LΨd_scalar().middle_layers(stage_idx, nd);
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
        solve_H1(k);
        while (true) {
            if (k + 1 < num_batch &&
                join_counters[k + 1].value.fetch_add(1) != 1)
                return false;
            KOQKATOO_TRACE("solve ψ fwd", k);
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
                // that λ(i+1) is available. Concretely, we can only start batch
                // k if Ψ((k + 1) * simd_stride) is done.
                {
                    KOQKATOO_TRACE("solve ψ rev", k);
                    if (k == 0)
                        solve_ψ_rev(0);
                    for (index_t i = ik; i < ikp1; ++i)
                        solve_ψ_rev(i + 1); // delay of one for λ(i+1)
                }
                KOQKATOO_TRACE("solve store-notify", k);
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
    auto [N, nx, nu, ny, ny_N] = storage.dim;

    // Count the number of changing constraints in each stage.
    auto &ranks    = storage.stagewise_update_counts;
    index_t rj_max = 0;
    for (index_t k = 0; k <= N; ++k) { // TODO: worth parallelizing?
        index_t rj  = 0;
        index_t nyk = k == N ? ny_N : ny;
        for (index_t r = 0; r < nyk; ++r) {
            bool up   = J_new(k, r, 0) && !J_old(k, r, 0);
            bool down = !J_new(k, r, 0) && J_old(k, r, 0);
            if (up || down) {
                storage.Σ_sgn()(k, rj, 0) = up ? +real_t(0) : -real_t(0);
                storage.Σ_ud()(k, rj, 0)  = Σ(k, r, 0);
                index_t nxuk              = k == N ? nx : nx + nu;
                for (index_t c = 0; c < nxuk; ++c) // TODO: pre-transpose CD?
                    storage.Z()(k, c, rj) = CD()(k, r, c);
                ++rj;
            }
        }
        ranks(k, 0, 0) = rj;
        rj_max         = std::max(rj_max, rj);
    }
    if (rj_max == 0)
        return;

    // Factorization updates of H, V and Wᵀ
    const auto updowndate_HVW =
        [&](single_mut_real_view HV, single_mut_real_view Wᵀ,
            single_mut_real_view A, single_real_view D) {
            using namespace linalg::compact::micro_kernels::shhud_diag;
            static constexpr index_constant<SizeR> R;
            [[maybe_unused]] static constexpr index_constant<SizeS> S;

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

    // Prepare the updates of Ψ and H (Woodbury and computation of ̃W)
    const auto updowndate_stage = [&](index_t batch_idx, index_t rj_min,
                                      index_t rj_batch, auto ranksj) {
        auto Σj  = storage.Σ_ud().batch(batch_idx).top_rows(rj_batch);
        auto Sj  = storage.Σ_sgn().batch(batch_idx).top_rows(rj_batch);
        auto Zj  = storage.Z().batch(batch_idx).left_cols(rj_batch);
        auto Z1j = storage.Z1().batch(batch_idx).left_cols(rj_batch);
        auto Luj = storage.Lupd().batch(batch_idx).top_left(rj_batch, rj_batch);
        auto Wuj = storage.Wupd().batch(batch_idx).left_cols(rj_batch);
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
            auto Vuj = storage.Vupd().batch(batch_idx).left_cols(rj_batch);
            compact_blas::xgemm_neg(V().batch(batch_idx), Zj, Vuj, be);
        }
    };

    // Factorization updates of H, V and Wᵀ
    const auto updowndate_stage_H = [&](index_t batch_idx, index_t rj_batch) {
        auto Σj    = storage.Σ_ud().batch(batch_idx).top_rows(rj_batch);
        auto Sj    = storage.Σ_sgn().batch(batch_idx).top_rows(rj_batch);
        auto Z1j   = storage.Z1().batch(batch_idx).left_cols(rj_batch);
        using simd = typename types::simd;
        for (index_t r = 0; r < rj_batch; ++r) {
            simd s{&Sj(0, r, 0), stdx::vector_aligned};
            simd σ{&Σj(0, r, 0), stdx::vector_aligned};
            cneg(σ, s).copy_to(&Σj(0, r, 0), stdx::vector_aligned);
        }
        // Update LH and V
        // TODO: no need to update V in last stage.
        updowndate_HVW(LHV().batch(batch_idx), Wᵀ().batch(batch_idx), Z1j, Σj);
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

    const auto updowndate_Ψ_stage = [&](index_t k) {
        index_t rk  = ranks(k, 0, 0);
        auto colsA0 = colsA;
        colsA += rk;
        if (colsA == 0)
            return;
        auto Dd            = D(0).top_rows(colsA);
        Dd.bottom_rows(rk) = storage.Σ_sgn()(k).top_rows(rk);
        auto As            = A(k % 2).left_cols(colsA);
        As.right_cols(rk)  = storage.Wupd()(k).left_cols(rk);
        if (k == N)
            // In the final stage, we cannot donwdate anything while waiting for
            // V(N-1), we can only copy W̃(N).
            return;
        auto Ad = A((k + 1) % 2).left_cols(colsA);
        auto Ld = LΨd_scalar()(k + 1), Ls = LΨs_scalar()(k + 1);
        Ad.right_cols(rk) = storage.Vupd()(k).left_cols(rk);
        W_t W;
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

        // In the final iteration, we still need to process W̃₀
        if (k == 0) {
            auto Ld = LΨd_scalar()(0);
            auto Ad = As;
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

    const auto updowndate_Ψ_batch = [&](index_t batch_idx) {
        KOQKATOO_TRACE("updowndate Ψ", batch_idx);
        index_t k0 = batch_idx * simd_stride;
        index_t k1 = std::min(k0 + simd_stride, N + 1);
        for (index_t k = k1; k-- > k0;)
            updowndate_Ψ_stage(k);
    };

    const index_t num_batch = (N + 1 + simd_stride - 1) / simd_stride;
    auto &join_counters     = storage.reset_join_counters();
    // join_counters: A value of 2 means that the parallel preparation for this
    //                batch was done, a value of 1 means that the previous block
    //                of Ψ was done. Therefore, a value of 3 means that the
    //                current block of Ψ can be updated.
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
        if (ready == 1 || batch_idx == num_batch - 1) {
            auto bi = batch_idx;
            while (true) {
                updowndate_Ψ_batch(bi);
                if (bi-- == 0)
                    break;
                // Indicate that this block of Ψ is done
                if (join_counters[bi].value.fetch_or(1) == 0)
                    // If the corresponding preparation is not done yet,
                    // we can't process the next block of Ψ yet.
                    break;
            };
        }
    };

    // Perform the update of H (after the parallel preparation)
    const auto process_stage_H = [&](index_t batch_idx, index_t rj_batch) {
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
        KOQKATOO_TRACE("updowndate H", batch_idx);
        updowndate_stage_H(batch_idx, rj_batch);
    };

    std::atomic<index_t> batch_counter{0};
    auto thread_work = [&](index_t, index_t) {
        while (true) {
            index_t b = batch_counter.fetch_add(1, std::memory_order_relaxed);
            // First perform the parallel preparation, eagerly processing blocks
            // of Ψ as we go.
            if (b < num_batch) {
                auto batch_idx = num_batch - 1 - b;
                auto stage_idx = batch_idx * simd_stride;
                auto nk     = std::min<index_t>(simd_stride, N + 1 - stage_idx);
                auto ranksj = ranks.batch(batch_idx);
                auto rjmm = std::minmax_element(ranksj.data, ranksj.data + nk);
                index_t rj_min = *rjmm.first, rj_batch = *rjmm.second;
                process_stage(batch_idx, rj_min, rj_batch, ranksj);
            }
            // Once all parallel preparation has been started, also start
            // updating H.
            else if (b < 2 * num_batch) {
                // Looping backwards may give slightly better cache locality
                auto batch_idx = b - num_batch;
                auto stage_idx = batch_idx * simd_stride;
                auto nk     = std::min<index_t>(simd_stride, N + 1 - stage_idx);
                auto ranksj = ranks.batch(batch_idx);
                auto rj_batch = std::max_element(ranksj.data, ranksj.data + nk);
                process_stage_H(batch_idx, *rj_batch);
            }
            // If all parallel preparation and updating of H is done, return.
            else {
                break;
            }
        }
    };

    foreach_thread(thread_work);
}

} // namespace koqkatoo::ocp
