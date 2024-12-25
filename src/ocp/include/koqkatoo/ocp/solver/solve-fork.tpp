#pragma once

#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/trace.hpp>
#include <libfork/core.hpp>
#include <libfork/schedule.hpp>
#include <atomic>

namespace koqkatoo::ocp {

template <simd_abi_tag Abi>
void Solver<Abi>::factor_fork(real_t S, real_view Σ, bool_view J) {
    static lf::lazy_pool pool{8}; // TODO: make configurable/argument

    using std::isfinite;
    struct join_counter_t {
        alignas(64) std::atomic<int> value{};
    };

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
    const auto process_stage = [&](auto, index_t k) -> lf::task<void> {
        prepare(k);
        while (k < num_batch) {
            if (k > 0 && join_counters[k - 1].value.fetch_add(1) != 1)
                break;
            factor_Ψ(k++);
        }
        co_return;
    };
    const auto process_all = [&](auto) -> lf::task<void> {
        for (index_t k = 0; k < num_batch; ++k)
            co_await lf::fork[process_stage](k);
        co_await lf::join;
    };
    lf::sync_wait(pool, process_all);
}

} // namespace koqkatoo::ocp
