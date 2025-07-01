#pragma once

#include <experimental/simd>
#include <guanaqo/mat-view.hpp>
#include <cassert>
#include <concepts>

#include <batmat/config.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/matrix/view.hpp>

namespace batmat::linalg::compact {

template <class T, class Abi>
struct CompactBLAS {
    using simd_types                     = simd_view_types<T, Abi>;
    using simd                           = typename simd_types::simd;
    using single_batch_view              = typename simd_types::template view<const T>;
    using mut_single_batch_view          = typename simd_types::template view<T>;
    using mut_single_batch_view_scalar   = typename simd_types::template scalar_view<T>;
    using mut_batch_view_scalar          = typename simd_types::template multi_scalar_view<T>;
    using batch_view                     = typename simd_types::template multi_view<const T>;
    using mut_batch_view                 = typename simd_types::template multi_view<T>;
    static constexpr index_t simd_stride = simd_types::simd_stride_t();

    static void unpack(single_batch_view A, mut_single_batch_view_scalar B);
    static void unpack(single_batch_view A, mut_batch_view_scalar B);
    static void unpack(batch_view A, mut_batch_view_scalar B);

    static void unpack_L(single_batch_view A, mut_single_batch_view_scalar B);
    static void unpack_L(single_batch_view A, mut_batch_view_scalar B);
    static void unpack_L(batch_view A, mut_batch_view_scalar B);

    /// Cholesky downdate
    static void xshh(mut_single_batch_view L, mut_single_batch_view A);
    static void xshh(mut_batch_view L, mut_batch_view A);
    static void xshh_ref(mut_single_batch_view L, mut_single_batch_view A);

    /// Cholesky up/downdate
    static void xshhud_diag_ref(mut_single_batch_view L, mut_single_batch_view A,
                                single_batch_view D);
    static void xshhud_diag_2_ref(mut_single_batch_view L, mut_single_batch_view A,
                                  mut_single_batch_view L2, mut_single_batch_view A2,
                                  single_batch_view D);
    static void xshhud_diag_cyclic(mut_single_batch_view L11, mut_single_batch_view A1,
                                   mut_single_batch_view L21, single_batch_view A2,
                                   mut_single_batch_view A2_out, mut_single_batch_view L31,
                                   single_batch_view A3, mut_single_batch_view A3_out,
                                   single_batch_view D, index_t split, int rot_A2);
    static void xshhud_diag_riccati(mut_single_batch_view L11, mut_single_batch_view A1,
                                    mut_single_batch_view L21, single_batch_view A2,
                                    mut_single_batch_view A2_out, mut_single_batch_view Lu1,
                                    mut_single_batch_view Au_out, single_batch_view D,
                                    bool shift_A_out);

    /// B ← A ⊙ B
    static void xhadamard(single_batch_view A, mut_single_batch_view B);
    static void xhadamard(batch_view A, mut_batch_view B);

    /// A ← -A
    static void xneg(mut_single_batch_view A);
    static void xneg(mut_batch_view A);

    /// y += a x
    static void xaxpy(real_t a, single_batch_view x, mut_single_batch_view y);
    static void xaxpy(real_t a, batch_view x, mut_batch_view y);

    /// y ← a x + b y
    static void xaxpby(real_t a, single_batch_view x, real_t b, mut_single_batch_view y);
    static void xaxpby(real_t a, batch_view x, real_t b, mut_batch_view y);

    /// L += A
    static void xadd_L(single_batch_view A, mut_single_batch_view B);

    /// Sum
    template <int Rot = 0, class OutView, class View, class... Views>
    static void xadd_copy_impl(OutView out, View x1, Views... xs)
        requires(((std::same_as<OutView, mut_batch_view> && std::same_as<View, batch_view>) &&
                  ... && std::same_as<Views, batch_view>) ||
                 ((std::same_as<OutView, mut_single_batch_view> &&
                   std::same_as<View, single_batch_view>) &&
                  ... && std::same_as<Views, single_batch_view>));
    template <int Rot = 0, class... Views>
    static void xadd_copy(mut_batch_view out, Views... xs) {
        xadd_copy_impl<Rot>(out, batch_view{xs}...);
    }
    template <int Rot = 0, class... Views>
    static void xadd_copy(mut_single_batch_view out, Views... xs) {
        xadd_copy_impl<Rot>(out, single_batch_view{xs}...);
    }
    template <class OutView, class View, class... Views>
    static void xsub_copy_impl(OutView out, View x1, Views... xs)
        requires(((std::same_as<OutView, mut_batch_view> && std::same_as<View, batch_view>) &&
                  ... && std::same_as<Views, batch_view>) ||
                 ((std::same_as<OutView, mut_single_batch_view> &&
                   std::same_as<View, single_batch_view>) &&
                  ... && std::same_as<Views, single_batch_view>));
    template <class... Views>
    static void xsub_copy(mut_batch_view out, Views... xs) {
        xsub_copy_impl(out, batch_view{xs}...);
    }
    template <class... Views>
    static void xsub_copy(mut_single_batch_view out, Views... xs) {
        xsub_copy_impl(out, single_batch_view{xs}...);
    }
    template <int Rot = 0, class OutView, class View, class... Views>
    static void xadd_neg_copy_impl(OutView out, View x1, Views... xs)
        requires(((std::same_as<OutView, mut_batch_view> && std::same_as<View, batch_view>) &&
                  ... && std::same_as<Views, batch_view>) ||
                 ((std::same_as<OutView, mut_single_batch_view> &&
                   std::same_as<View, single_batch_view>) &&
                  ... && std::same_as<Views, single_batch_view>));
    template <int Rot = 0, class... Views>
    static void xadd_neg_copy(mut_batch_view out, Views... xs) {
        xadd_neg_copy_impl<Rot>(out, batch_view{xs}...);
    }
    template <int Rot = 0, class... Views>
    static void xadd_neg_copy(mut_single_batch_view out, Views... xs) {
        xadd_neg_copy_impl<Rot>(out, single_batch_view{xs}...);
    }

    /// Dot product
    static real_t xdot(single_batch_view x, single_batch_view y);
    static real_t xdot(batch_view x, batch_view y);
    /// Square of the 2-norm
    static real_t xnrm2sq(batch_view x);
    /// Infinity/max norm
    static real_t xnrminf(single_batch_view x);
    static real_t xnrminf(batch_view x);

    template <class T0, class F, class R, class... Args>
    static auto xreduce(T0 init, F fun, R reduce, single_batch_view x0, const Args &...xs) {
        const index_t m = x0.rows(), n = x0.cols();
        assert(((x0.rows() == xs.rows()) && ...));
        assert(((x0.cols() == xs.cols()) && ...));
        assert(((x0.depth() == xs.depth()) && ...));
        assert(((x0.batch_size() == xs.batch_size()) && ...));
        for (index_t c = 0; c < n; ++c)
            for (index_t r = 0; r < m; ++r)
                init = fun(init, simd_types::aligned_load(&x0(0, r, c)),
                           simd_types::aligned_load(&xs(0, r, c))...);
        return reduce(init);
    }

    template <class T0, class F, class R, class... Args>
    static auto xreduce(T0 init, F fun, R reduce, batch_view x0, const Args &...xs) {
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
                    init = fun(init, simd_types::aligned_load(&x0(i, r, c)),
                               simd_types::aligned_load(&xs(i, r, c))...);
        auto accum_scal = reduce(init);
        for (; i < x0.depth(); ++i)
            for (index_t c = 0; c < n; ++c)
                for (index_t r = 0; r < m; ++r)
                    accum_scal = fun(accum_scal, x0(i, r, c), (xs(i, r, c))...);
        return accum_scal;
    }

    template <class T0, class F, class R, class... Args>
    static auto xreduce_enumerate(T0 init, F fun, R reduce, batch_view x0, const Args &...xs) {
        const auto Bs   = static_cast<index_t>(x0.batch_size());
        const index_t m = x0.rows(), n = x0.cols();
        assert(((x0.rows() == xs.rows()) && ...));
        assert(((x0.cols() == xs.cols()) && ...));
        assert(((x0.depth() == xs.depth()) && ...));
        assert(((x0.batch_size() == xs.batch_size()) && ...));
        const index_t N_batched = x0.depth() - 1;
        index_t i;
        for (i = 0; i + Bs < N_batched; i += Bs)
            for (index_t c = 0; c < n; ++c)
                for (index_t r = 0; r < m; ++r)
                    init =
                        fun(std::make_tuple(i, r, c), init, simd_types::aligned_load(&x0(i, r, c)),
                            simd_types::aligned_load(&xs(i, r, c))...);
        auto accum_scal = reduce(init);
        for (; i < x0.depth(); ++i)
            for (index_t c = 0; c < n; ++c)
                for (index_t r = 0; r < m; ++r)
                    accum_scal =
                        fun(std::make_tuple(i, r, c), accum_scal, x0(i, r, c), (xs(i, r, c))...);
        return accum_scal;
    }

    template <index_t N = 8>
    static index_t compress_masks(single_batch_view A_in, single_batch_view S_in,
                                  mut_single_batch_view A_out, mut_single_batch_view S_out);
    template <index_t N = 8>
    static index_t compress_masks_count(single_batch_view S_in);

    /// y = x - clamp(x, l, u)
    static void proj_diff(single_batch_view x, single_batch_view l, single_batch_view u,
                          mut_single_batch_view y);
};

} // namespace batmat::linalg::compact

#include "compact/elementwise.tpp"
#include "compact/pack.tpp"
#include "compact/update.tpp"
