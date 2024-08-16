#pragma once

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/unroll.h>
#include <experimental/simd>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace koqkatoo::linalg::compact::micro_kernels {

namespace stdx = std::experimental;

template <class Abi>
struct simd_view_types {
    using simd                       = stdx::simd<real_t, Abi>;
    using mask                       = typename simd::mask_type;
    using simd_stride_t              = stdx::simd_size<real_t, Abi>;
    static constexpr auto simd_align = stdx::memory_alignment_v<simd>;
    static_assert(simd_align <= simd_stride_t() * sizeof(real_t));
    using mut_single_batch_view =
        BatchedMatrixView<real_t, index_t, simd_stride_t, simd_stride_t>;
    using single_batch_view =
        BatchedMatrixView<const real_t, index_t, simd_stride_t, simd_stride_t>;
    using bool_single_batch_view =
        BatchedMatrixView<const bool, index_t, simd_stride_t, simd_stride_t>;
    using batch_view = BatchedMatrixView<const real_t, index_t, simd_stride_t>;
    using mut_batch_view = BatchedMatrixView<real_t, index_t, simd_stride_t>;
};

#define KOQKATOO_STRONG_ALIAS_MICRO_KERNEL_PARAM_TYPE(type)                    \
    template <class Abi>                                                       \
    struct type : simd_view_types<Abi>::type {                                 \
        using simd_view_types<Abi>::type::type;                                \
        type(const simd_view_types<Abi>::type &o)                              \
            : simd_view_types<Abi>::type{o} {}                                 \
    }

KOQKATOO_STRONG_ALIAS_MICRO_KERNEL_PARAM_TYPE(mut_single_batch_view);
KOQKATOO_STRONG_ALIAS_MICRO_KERNEL_PARAM_TYPE(single_batch_view);
KOQKATOO_STRONG_ALIAS_MICRO_KERNEL_PARAM_TYPE(mut_batch_view);
KOQKATOO_STRONG_ALIAS_MICRO_KERNEL_PARAM_TYPE(batch_view);

#undef KOQKATOO_STRONG_ALIAS_MICRO_KERNEL_PARAM_TYPE

template <class Abi, bool Const, bool Transpose>
struct mat_access_impl {
    using value_type = std::conditional_t<Const, const real_t, real_t>;
    value_type *data;
    index_t outer_stride;

    using types                             = simd_view_types<Abi>;
    using simd                              = typename types::simd;
    static constexpr ptrdiff_t inner_stride = typename types::simd_stride_t();

    [[gnu::always_inline]] value_type &operator()(index_t r,
                                                  index_t c) const noexcept {
        ptrdiff_t i0 = Transpose ? c : r;
        ptrdiff_t i1 = Transpose ? r : c;
        return data[i0 * inner_stride +
                    i1 * static_cast<ptrdiff_t>(outer_stride)];
    }
    [[gnu::always_inline]] simd load(index_t r, index_t c) const noexcept {
        return simd{&operator()(r, c), stdx::vector_aligned};
    }
    [[gnu::always_inline]] void store(simd x, index_t r,
                                      index_t c) const noexcept
        requires(!Const)
    {
        x.copy_to(&operator()(r, c), stdx::vector_aligned);
    }
    template <class Self>
    [[gnu::always_inline]] Self block(this const Self &self, index_t r,
                                      index_t c) noexcept {
        return {&self(r, c), self.outer_stride};
    }
    template <class Self>
    [[gnu::always_inline]] Self middle_rows(this const Self &self,
                                            index_t r) noexcept {
        return {&self(r, 0), self.outer_stride};
    }
    template <class Self>
    [[gnu::always_inline]] Self middle_cols(this const Self &self,
                                            index_t c) noexcept {
        return {&self(0, c), self.outer_stride};
    }

    [[gnu::always_inline]] mat_access_impl(value_type *data,
                                           index_t outer_stride) noexcept
        : data{data}, outer_stride{outer_stride} {}
    [[gnu::always_inline]] mat_access_impl(
        const types::single_batch_view &o) noexcept
        requires Const
        : data{o.data}, outer_stride{o.outer_stride() * inner_stride} {}
    [[gnu::always_inline]] mat_access_impl(
        const types::mut_single_batch_view &o) noexcept
        : data{o.data}, outer_stride{o.outer_stride() * inner_stride} {}
};

template <index_t Size, class Abi, bool Const, bool Transpose>
struct cached_mat_access_impl {
    using value_type = std::conditional_t<Const, const real_t, real_t>;
    value_type *const data[Size];

    using types                             = simd_view_types<Abi>;
    using simd                              = typename types::simd;
    static constexpr ptrdiff_t inner_stride = typename types::simd_stride_t();

    [[gnu::always_inline]] value_type &operator()(index_t r,
                                                  index_t c) const noexcept {
        ptrdiff_t i0 = Transpose ? c : r;
        index_t i1   = Transpose ? r : c;
        assert(i1 < Size);
        return data[i1][i0 * inner_stride];
    }
    [[gnu::always_inline]] simd load(index_t r, index_t c) const noexcept {
        return simd{&operator()(r, c), stdx::vector_aligned};
    }
    [[gnu::always_inline]] void store(simd x, index_t r,
                                      index_t c) const noexcept
        requires(!Const)
    {
        x.copy_to(&operator()(r, c), stdx::vector_aligned);
    }

    template <index_t... Is>
    [[gnu::always_inline]] cached_mat_access_impl(
        const mat_access_impl<Abi, Const, Transpose> &o,
        std::integer_sequence<index_t, Is...>) noexcept
        : data{(o.data + Is * static_cast<ptrdiff_t>(o.outer_stride))...} {}
    [[gnu::always_inline]] cached_mat_access_impl(
        const mat_access_impl<Abi, Const, Transpose> &o) noexcept
        : cached_mat_access_impl{o,
                                 std::make_integer_sequence<index_t, Size>()} {}
};

template <index_t Size, class Abi, bool Const, bool Transpose>
[[gnu::always_inline]] cached_mat_access_impl<Size, Abi, Const, Transpose>
with_cached_access(const mat_access_impl<Abi, Const, Transpose> &o) noexcept {
    return {o};
}

template <class Abi, bool Trans = false>
struct single_batch_matrix_accessor : mat_access_impl<Abi, true, Trans> {
    using mat_access_impl<Abi, true, Trans>::mat_access_impl;
};

template <class Abi, bool Trans = false>
struct mut_single_batch_matrix_accessor : mat_access_impl<Abi, false, Trans> {
    using mat_access_impl<Abi, false, Trans>::mat_access_impl;
};

template <class Abi, class T>
struct vec_access_impl {
    using value_type = T;
    value_type *data;

    using types = simd_view_types<Abi>;
    using simd  = std::conditional_t<std::is_same_v<std::remove_cv_t<T>, bool>,
                                     typename types::mask, typename types::simd>;
    static constexpr ptrdiff_t inner_stride = typename types::simd_stride_t();

    [[gnu::always_inline]] value_type &operator()(index_t r) const noexcept {
        ptrdiff_t i0 = r;
        return data[i0 * inner_stride];
    }
    [[gnu::always_inline]] simd load(index_t r) const noexcept {
        return simd{&operator()(r), stdx::vector_aligned};
    }
    [[gnu::always_inline]] void store(simd x, index_t r) const noexcept
        requires(!std::is_const_v<T>)
    {
        x.copy_to(&operator()(r), stdx::vector_aligned);
    }
    template <class Self>
    [[gnu::always_inline]] Self middle_rows(this const Self &self,
                                            index_t r) noexcept {
        return {&self(r)};
    }

    [[gnu::always_inline]] vec_access_impl(value_type *data) noexcept
        : data{data} {}
    [[gnu::always_inline]] vec_access_impl(
        const types::single_batch_view &o) noexcept
        requires std::is_const_v<T>
        : data{o.data} {
        assert(o.cols() == 0);
    }
    [[gnu::always_inline]] vec_access_impl(
        const types::mut_single_batch_view &o) noexcept
        : data{o.data} {
        assert(o.cols() == 0);
    }
};

template <class Abi>
struct single_batch_vector_accessor : vec_access_impl<Abi, const real_t> {
    using vec_access_impl<Abi, const real_t>::vec_access_impl;
};

template <class Abi>
struct mut_single_batch_vector_accessor : vec_access_impl<Abi, real_t> {
    using vec_access_impl<Abi, real_t>::vec_access_impl;
};

template <class Abi>
struct single_batch_vector_mask_accessor : vec_access_impl<Abi, const bool> {
    using vec_access_impl<Abi, const bool>::vec_access_impl;
};

} // namespace koqkatoo::linalg::compact::micro_kernels
