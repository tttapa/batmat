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

struct size_last_t {
    index_t rows, cols;
};

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
        BatchedMatrixView<const bool, index_t, simd_stride_t, index_t>;
    using batch_view = BatchedMatrixView<const real_t, index_t, simd_stride_t,
                                         index_t, index_t>;
    using mut_batch_view =
        BatchedMatrixView<real_t, index_t, simd_stride_t, index_t, index_t>;

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

    /// C += AAᵀ
    static void xsyrk_add_ref(single_batch_view A, mut_single_batch_view C);
    static void xsyrk_add(single_batch_view A, mut_single_batch_view C,
                          PreferredBackend b);
    static void xsyrk_add(batch_view A, mut_batch_view C, PreferredBackend b);

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
    static void xpotrf(mut_single_batch_view H, PreferredBackend b,
                       index_t n = -1);
    static void xpotrf(mut_batch_view H, PreferredBackend b, index_t n = -1);
    static void xpotrf_base(mut_batch_view H, PreferredBackend b);
    static void xpotrf_recursive(mut_batch_view H, PreferredBackend b);
    static void xpotrf_ref(mut_single_batch_view H, index_t n = -1);
    static void xpotrf_recursive_ref(mut_single_batch_view H);
    static void xpotrf_base_ref(mut_single_batch_view H, index_t n = -1);

    /// Quasidefinite LDLᵀ factorization where Dᵢᵢ = ±1. The @p signs argument
    /// should contain only the sign bit, i.e. ±0.0.
    static void xpntrf(mut_single_batch_view H, single_batch_view signs);
    static void xpntrf(mut_batch_view H, batch_view signs);
    static void xpntrf_ref(mut_single_batch_view H, single_batch_view signs);

    /// x ← L x
    static void xtrmv_ref(single_batch_view L, mut_single_batch_view x);
    static void xtrmv(single_batch_view L, mut_single_batch_view x,
                      PreferredBackend b);
    static void xtrmv(batch_view L, mut_batch_view x, PreferredBackend b);

    /// L ← L⁻¹
    static void xtrtri_ref(mut_single_batch_view L);
    static void xtrtri(mut_single_batch_view L, PreferredBackend b);
    static void xtrtri(mut_batch_view L, PreferredBackend b);

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

    /// C ← -AB (with B lower trapezoidal)
    static void xtrmm_RLNN_neg(single_batch_view A, single_batch_view B,
                               mut_single_batch_view C, PreferredBackend b);
    static void xtrmm_RLNN_neg(batch_view A, batch_view B, mut_batch_view C,
                               PreferredBackend b);
    static void xtrmm_RLNN_neg_ref(single_batch_view A, single_batch_view B,
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

    /// Cholesky downdate
    static void xshh(mut_single_batch_view L, mut_single_batch_view A,
                     PreferredBackend b);
    static void xshh(mut_batch_view L, mut_batch_view A, PreferredBackend b);
    static void xshh_ref(mut_single_batch_view L, mut_single_batch_view A);

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

    /// B ← Aᵀ
    static void xcopy_T(single_batch_view A, mut_single_batch_view B);
    static void xcopy_T(batch_view A, mut_batch_view B);

    /// B ← A
    static void xfill(real_t A, mut_single_batch_view B);
    static void xfill(real_t A, mut_batch_view B);

    /// A ← -A
    static void xneg(mut_single_batch_view A);
    static void xneg(mut_batch_view A);

    /// y += a x
    static void xaxpy(real_t a, single_batch_view x, mut_single_batch_view y);
    static void xaxpy(real_t a, batch_view x, mut_batch_view y);

    /// y ← a x + b y
    static void xaxpby(real_t a, single_batch_view x, real_t b,
                       mut_single_batch_view y);
    static void xaxpby(real_t a, batch_view x, real_t b, mut_batch_view y);

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
    static real_t xdot(size_last_t size_last, batch_view x, batch_view y);
    /// Square of the 2-norm
    static real_t xnrm2sq(batch_view x);
    static real_t xnrm2sq(size_last_t size_last, batch_view x);
    /// Infinity/max norm
    static real_t xnrminf(single_batch_view x);
    static real_t xnrminf(batch_view x);
    static real_t xnrminf(size_last_t size_last, batch_view x);

    template <class T0, class F, class R, class... Args>
    static auto xreduce(T0 init, F fun, R reduce, single_batch_view x0,
                        const Args &...xs) {
        const index_t m = x0.rows(), n = x0.cols();
        assert(((x0.rows() == xs.rows()) && ...));
        assert(((x0.cols() == xs.cols()) && ...));
        assert(((x0.depth() == xs.depth()) && ...));
        assert(((x0.batch_size() == xs.batch_size()) && ...));
        for (index_t c = 0; c < n; ++c)
            for (index_t r = 0; r < m; ++r)
                init = fun(init, aligned_load(&x0(0, r, c)),
                           aligned_load(&xs(0, r, c))...);
        return reduce(init);
    }

    template <class T0, class F, class R, class... Args>
    static auto xreduce(T0 init, F fun, R reduce, batch_view x0,
                        const Args &...xs) {
        const auto Bs   = static_cast<index_t>(x0.batch_size());
        const index_t m = x0.rows(), n = x0.cols();
        assert(((x0.rows() == xs.rows()) && ...));
        assert(((x0.cols() == xs.cols()) && ...));
        assert(((x0.depth() == xs.depth()) && ...));
        assert(((x0.batch_size() == xs.batch_size()) && ...));
        const index_t N_batched = x0.depth();
        index_t i;
        for (i = 0; i + Bs <= N_batched; i += Bs)
            for (index_t c = 0; c < n; ++c)
                for (index_t r = 0; r < m; ++r)
                    init = fun(init, aligned_load(&x0(i, r, c)),
                               aligned_load(&xs(i, r, c))...);
        auto accum_scal = reduce(init);
        for (; i < x0.depth(); ++i)
            for (index_t c = 0; c < n; ++c)
                for (index_t r = 0; r < m; ++r)
                    accum_scal = fun(accum_scal, x0(i, r, c), (xs(i, r, c))...);
        return accum_scal;
    }

    template <class T0, class F, class R, class... Args>
    static auto xreduce(size_last_t size_last, T0 init, F fun, R reduce,
                        batch_view x0, const Args &...xs) {
        const auto Bs   = static_cast<index_t>(x0.batch_size());
        const index_t m = x0.rows(), n = x0.cols();
        assert(((x0.rows() == xs.rows()) && ...));
        assert(((x0.cols() == xs.cols()) && ...));
        assert(((x0.depth() == xs.depth()) && ...));
        assert(((x0.batch_size() == xs.batch_size()) && ...));
        const index_t N_batched = x0.depth() - 1;
        index_t i;
        for (i = 0; i + Bs <= N_batched; i += Bs)
            for (index_t c = 0; c < n; ++c)
                for (index_t r = 0; r < m; ++r)
                    init = fun(init, aligned_load(&x0(i, r, c)),
                               aligned_load(&xs(i, r, c))...);
        auto accum_scal = reduce(init);
        for (; i < x0.depth() - 1; ++i)
            for (index_t c = 0; c < n; ++c)
                for (index_t r = 0; r < m; ++r)
                    accum_scal = fun(accum_scal, x0(i, r, c), (xs(i, r, c))...);
        if (i < x0.depth())
            for (index_t c = 0; c < size_last.cols; ++c)
                for (index_t r = 0; r < size_last.rows; ++r)
                    accum_scal = fun(accum_scal, x0(i, r, c), (xs(i, r, c))...);
        return accum_scal;
    }

    template <class T0, class F, class R, class... Args>
    static auto xreduce_enumerate(size_last_t size_last, T0 init, F fun,
                                  R reduce, batch_view x0, const Args &...xs) {
        const auto Bs   = static_cast<index_t>(x0.batch_size());
        const index_t m = x0.rows(), n = x0.cols();
        assert(((x0.rows() == xs.rows()) && ...));
        assert(((x0.cols() == xs.cols()) && ...));
        assert(((x0.depth() == xs.depth()) && ...));
        assert(((x0.batch_size() == xs.batch_size()) && ...));
        const index_t N_batched = x0.depth() - 1;
        index_t i;
        for (i = 0; i + Bs <= N_batched; i += Bs)
            for (index_t c = 0; c < n; ++c)
                for (index_t r = 0; r < m; ++r)
                    init = fun(std::make_tuple(i, r, c), init,
                               aligned_load(&x0(i, r, c)),
                               aligned_load(&xs(i, r, c))...);
        auto accum_scal = reduce(init);
        for (; i < x0.depth() - 1; ++i)
            for (index_t c = 0; c < n; ++c)
                for (index_t r = 0; r < m; ++r)
                    accum_scal = fun(std::make_tuple(i, r, c), accum_scal,
                                     x0(i, r, c), (xs(i, r, c))...);
        if (i < x0.depth())
            for (index_t c = 0; c < size_last.cols; ++c)
                for (index_t r = 0; r < size_last.rows; ++r)
                    accum_scal = fun(std::make_tuple(i, r, c), accum_scal,
                                     x0(i, r, c), (xs(i, r, c))...);
        return accum_scal;
    }

    /// y = x - clamp(x, l, u)
    static void proj_diff(single_batch_view x, single_batch_view l,
                          single_batch_view u, mut_single_batch_view y);
};

} // namespace koqkatoo::linalg::compact
