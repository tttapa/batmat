#pragma once

#include <koqkatoo/loop.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <oneapi/tbb/flow_graph.h>
#include <oneapi/tbb/global_control.h>

namespace koqkatoo::ocp {

template <simd_abi_tag Abi>
void Solver<Abi>::factor_tbb(real_t S, real_view Σ, bool_view J) {
    oneapi::tbb::global_control c(
        oneapi::tbb::global_control::max_allowed_parallelism, 8);

    using std::isfinite;
    namespace fl = oneapi::tbb::flow;

    struct stage_block_index_t {
        index_t k;
        operator index_t() const { return k; }
    };

    using func_node =
        fl::function_node<stage_block_index_t, stage_block_index_t>;
    using func2_node =
        fl::function_node<std::tuple<stage_block_index_t, stage_block_index_t>,
                          stage_block_index_t>;
    using join2_node =
        fl::join_node<std::tuple<stage_block_index_t, stage_block_index_t>,
                      fl::key_matching<index_t>>;

    const auto N = storage.dim.N_horiz;

    fl::graph g;
    func_node prepare(
        g, fl::unlimited,
        [this, S, &Σ, &J](stage_block_index_t k) {
            auto [N, nx, nu, ny, ny_N] = storage.dim;
            schur_complement_Hi(k, Σ, J);
            if (isfinite(S))
                LH().batch(k).add_to_diagonal(1 / S);
            cholesky_Hi(k);
            // Solve W = LH⁻¹ [I 0]ᵀ
            auto LHi = LH().batch(k), Wi = Wᵀ().batch(k);
            compact_blas::xcopy(LHi.top_left(nx + nu, nx), Wi);
            compact_blas::xtrtri(Wi, settings.preferred_backend);
            compact_blas::xtrsm_LLNN(LHi.bottom_right(nu, nu),
                                     Wi.bottom_rows(nu),
                                     settings.preferred_backend);
            // Compute WWᵀ
            compact_blas::xsyrk_T(Wᵀ().batch(k), LΨd().batch(k),
                                  settings.preferred_backend);
            // Compute -VWᵀ
            if (k < AB().num_batches())
                compact_blas::xgemm_neg(V().batch(k), Wᵀ().batch(k),
                                        LΨs().batch(k),
                                        settings.preferred_backend);
            // Compute VVᵀ
            if (k < AB().num_batches())
                compact_blas::xsyrk(V().batch(k), VV().batch(k),
                                    settings.preferred_backend);
            return k;
        },
        {}, fl::node_priority_t{0});
    func2_node factor_Ψ(
        g, fl::unlimited,
        [this](std::tuple<stage_block_index_t, stage_block_index_t> in) {
            const auto N = storage.dim.N_horiz;
            auto [k, k_] = in;
            assert(k == k_);
            auto nd   = std::min(simd_stride, N + 1 - k * simd_stride);
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
            return stage_block_index_t{k + 1};
        },
        {}, fl::node_priority_t{999});

    auto tag_hash = [](stage_block_index_t in) { return in.k; };
    join2_node join_Ψ(g, tag_hash, tag_hash);
    fl::input_node<stage_block_index_t> src(
        g, [N, k = index_t{0}](oneapi::tbb::flow_control &fc) mutable {
            if (k * simd_stride < N + 1)
                return stage_block_index_t{k++};
            fc.stop();
            return stage_block_index_t{-1};
        });

    make_edge(src, prepare);
    make_edge(factor_Ψ, input_port<0>(join_Ψ));
    make_edge(prepare, input_port<1>(join_Ψ));
    make_edge(join_Ψ, factor_Ψ);

    input_port<0>(join_Ψ).try_put(stage_block_index_t{0});
    src.activate();

    g.wait_for_all();
}

} // namespace koqkatoo::ocp
