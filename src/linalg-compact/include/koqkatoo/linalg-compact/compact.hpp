#pragma once

#include <experimental/simd>
#include <guanaqo/mat-view.hpp>
#include <cassert>
#include <concepts>

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/aligned-storage.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/linalg-compact/preferred-backend.hpp>

namespace koqkatoo::linalg::compact {

namespace stdx = std::experimental;

template <class Abi = stdx::simd_abi::native<real_t>>
struct CompactBLAS {
    using simd                       = stdx::simd<real_t, Abi>;
    using mask                       = typename simd::mask_type;
    using simd_stride_t              = stdx::simd_size<real_t, Abi>;
    static constexpr auto simd_align = stdx::memory_alignment_v<simd>;
    static constexpr auto mask_align = stdx::memory_alignment_v<mask>;
    static_assert(simd_align <= simd_stride_t() * sizeof(real_t));
    static constexpr auto simd_stride = static_cast<index_t>(simd_stride_t());
    using mut_single_batch_view =
        BatchedMatrixView<real_t, index_t, simd_stride_t, simd_stride_t>;
    using single_batch_view =
        BatchedMatrixView<const real_t, index_t, simd_stride_t, simd_stride_t>;
    using bool_single_batch_view =
        BatchedMatrixView<const bool, index_t, simd_stride_t, simd_stride_t>;
    using bool_batch_view =
        BatchedMatrixView<const bool, index_t, simd_stride_t>;
    using batch_view = BatchedMatrixView<const real_t, index_t, simd_stride_t>;
    using mut_batch_view = BatchedMatrixView<real_t, index_t, simd_stride_t>;

    static simd aligned_load(const real_t *p) {
        return {p, stdx::vector_aligned};
    }
    static auto aligned_store(real_t *p, simd v) {
        return v.copy_to(p, stdx::vector_aligned);
    }
    static simd::mask_type aligned_mask_load(const bool *p) {
        return {p, stdx::vector_aligned};
    }

    /// C ← A D Aᵀ
    static void xsyrk_schur(single_batch_view A, single_batch_view d,
                            mut_single_batch_view C);
    static void xsyrk_schur(batch_view A, batch_view d, mut_batch_view C,
                            PreferredBackend b);
    /// H_out ← H_in + Cᵀ Σ C
    /// where Σ[!mask] = 0
    static void xsyrk_T_schur_copy(single_batch_view C, single_batch_view Σ,
                                   bool_single_batch_view mask,
                                   single_batch_view H_in,
                                   mut_single_batch_view H_out);
    static void xsyrk_T_schur_copy(batch_view C, batch_view Σ,
                                   bool_batch_view mask, batch_view H_in,
                                   mut_batch_view H_out, PreferredBackend b);

    /// C = AᵀA
    static void xsyrk_T_ref(single_batch_view A, mut_single_batch_view C);
    static void xsyrk_T(single_batch_view A, mut_single_batch_view C,
                        PreferredBackend b);
    static void xsyrk_T(batch_view A, mut_batch_view C, PreferredBackend b);

    /// C = AAᵀ
    static void xsyrk_ref(single_batch_view A, mut_single_batch_view C);
    static void xsyrk(single_batch_view A, mut_single_batch_view C,
                      PreferredBackend b);
    static void xsyrk(batch_view A, mut_batch_view C, PreferredBackend b);

    /// C -= AAᵀ
    static void xsyrk_sub_ref(single_batch_view A, mut_single_batch_view C);
    static void xsyrk_sub(single_batch_view A, mut_single_batch_view C,
                          PreferredBackend b);
    static void xsyrk_sub(batch_view A, mut_batch_view C, PreferredBackend b);

    /// Hᵀ ← L⁻¹ Hᵀ or H ← H L⁻ᵀ
    static void xtrsm_RLTN(single_batch_view L, mut_single_batch_view H,
                           PreferredBackend b);
    static void xtrsm_RLTN(batch_view L, mut_batch_view H, PreferredBackend b);
    static void xtrsm_RLTN_ref(single_batch_view L, mut_single_batch_view H);

    /// H ← L⁻¹ H
    static void xtrsm_LLNN(single_batch_view L, mut_single_batch_view H,
                           PreferredBackend b);
    static void xtrsm_LLNN(batch_view L, mut_batch_view H, PreferredBackend b);
    static void xtrsm_LLNN_ref(single_batch_view L, mut_single_batch_view H);

    /// H ← L⁻ᵀ H
    static void xtrsm_LLTN(single_batch_view L, mut_single_batch_view H,
                           PreferredBackend b);
    static void xtrsm_LLTN(batch_view L, mut_batch_view H, PreferredBackend b);
    static void xtrsm_LLTN_ref(single_batch_view L, mut_single_batch_view H);

    /// H ← cholesky(H)
    static void xpotrf(mut_single_batch_view H, PreferredBackend b);
    static void xpotrf(mut_batch_view H, PreferredBackend b);
    static void xpotrf_ref(mut_single_batch_view H);

    /// x ← L x
    static void xtrmv_ref(single_batch_view L, mut_single_batch_view x);
    static void xtrmv(single_batch_view L, mut_single_batch_view x,
                      PreferredBackend b);
    static void xtrmv(batch_view L, mut_batch_view x, PreferredBackend b);

    /// L ← L⁻¹
    static void xtrti_ref(mut_single_batch_view L);
    static void xtrti(mut_single_batch_view L, PreferredBackend b);
    static void xtrti(mut_batch_view L, PreferredBackend b);

    /// C ← AB
    static void xgemm(single_batch_view A, single_batch_view B,
                      mut_single_batch_view C, PreferredBackend b);
    static void xgemm(batch_view A, batch_view B, mut_batch_view C,
                      PreferredBackend b);
    static void xgemm_ref(single_batch_view A, single_batch_view B,
                          mut_single_batch_view C);

    /// C ← -AB
    static void xgemm_neg(single_batch_view A, single_batch_view B,
                          mut_single_batch_view C, PreferredBackend b);
    static void xgemm_neg(batch_view A, batch_view B, mut_batch_view C,
                          PreferredBackend b);
    static void xgemm_neg_ref(single_batch_view A, single_batch_view B,
                              mut_single_batch_view C);

    /// C += AB
    static void xgemm_add(single_batch_view A, single_batch_view B,
                          mut_single_batch_view C, PreferredBackend b);
    static void xgemm_add(batch_view A, batch_view B, mut_batch_view C,
                          PreferredBackend b);
    static void xgemm_add_ref(single_batch_view A, single_batch_view B,
                              mut_single_batch_view C);

    /// C -= AB
    static void xgemm_sub(single_batch_view A, single_batch_view B,
                          mut_single_batch_view C, PreferredBackend b);
    static void xgemm_sub(batch_view A, batch_view B, mut_batch_view C,
                          PreferredBackend b);
    static void xgemm_sub_ref(single_batch_view A, single_batch_view B,
                              mut_single_batch_view C);

    /// C = AᵀB
    static void xgemm_TN(single_batch_view A, single_batch_view B,
                         mut_single_batch_view C, PreferredBackend b);
    static void xgemm_TN(batch_view A, batch_view B, mut_batch_view C,
                         PreferredBackend b);
    static void xgemm_TN_ref(single_batch_view A, single_batch_view B,
                             mut_single_batch_view C);

    /// C -= AᵀB
    static void xgemm_TN_sub(single_batch_view A, single_batch_view B,
                             mut_single_batch_view C, PreferredBackend b);
    static void xgemm_TN_sub(batch_view A, batch_view B, mut_batch_view C,
                             PreferredBackend b);
    static void xgemm_TN_sub_ref(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C);

    /// y += Lx
    static void xsymv_add(single_batch_view L, single_batch_view x,
                          mut_single_batch_view y, PreferredBackend b);
    static void xsymv_add(batch_view L, batch_view x, mut_batch_view y,
                          PreferredBackend b);
    static void xsymv_add_ref(single_batch_view L, single_batch_view x,
                              mut_single_batch_view y);

    /// B ← A
    static void xcopy(single_batch_view A, mut_single_batch_view B);
    static void xcopy(batch_view A, mut_batch_view B);

    /// y += a x
    static void xaxpy(real_t a, single_batch_view x, mut_single_batch_view y);
    static void xaxpy(real_t a, batch_view x, mut_batch_view y);

    /// Sum
    template <class... Views>
    static void xadd_copy_impl(mut_batch_view out, batch_view x1, Views... xs)
        requires(std::same_as<Views, batch_view> && ...);
    template <class... Views>
    static void xadd_copy(mut_batch_view out, Views... xs) {
        xadd_copy_impl(out, batch_view{xs}...);
    }
    template <class... Views>
    static void xsub_copy_impl(mut_batch_view out, batch_view x1, Views... xs)
        requires(std::same_as<Views, batch_view> && ...);
    template <class... Views>
    static void xsub_copy(mut_batch_view out, Views... xs) {
        xsub_copy_impl(out, batch_view{xs}...);
    }

    /// Dot product
    static real_t xdot(batch_view x, batch_view y);
    /// Square of the 2-norm
    static real_t xnrm2sq(batch_view x);
    /// Infinity/max norm
    static real_t xnrminf(batch_view x);

    /// y = x - clamp(x, l, u)
    static void proj_diff(single_batch_view x, single_batch_view l,
                          single_batch_view u, mut_single_batch_view y);
};

} // namespace koqkatoo::linalg::compact
