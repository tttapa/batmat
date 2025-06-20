#pragma once

#include <batmat/config.hpp>
#include <batmat/matrix/matrix.hpp>
#include <batmat/matrix/view.hpp>
#include <batmat/unroll.h>

#include <experimental/simd>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace batmat::linalg {

using guanaqo::StorageOrder;

namespace stdx = std::experimental;

template <class T, class Abi>
struct simd_view_types {
    using value_type    = T;
    using simd          = stdx::simd<T, Abi>;
    using mask          = typename simd::mask_type;
    using isimd         = stdx::simd<index_t, stdx::simd_abi::deduce_t<index_t, simd::size()>>;
    using simd_stride_t = stdx::simd_size<T, Abi>;
    using simd_align_t  = stdx::memory_alignment<simd>;
    static constexpr auto simd_stride = simd_stride_t::value;
    static constexpr auto simd_align  = simd_align_t::value;
    static_assert(simd_align <= simd_stride * sizeof(T));
    template <class S = T, StorageOrder O = StorageOrder::ColMajor>
    using view = matrix::View<S, index_t, simd_stride_t, simd_stride_t, matrix::DefaultStride, O>;
    template <class S = T, StorageOrder O = StorageOrder::ColMajor>
    using matrix = matrix::Matrix<S, index_t, simd_stride_t, index_t, simd_align_t, O>;

    static simd aligned_load(const T *p) noexcept { return simd{p, stdx::vector_aligned}; }

    template <int MaskL = 0>
    [[gnu::always_inline]] static void aligned_store(simd x, value_type *p) noexcept {
        if constexpr (MaskL == 0) {
            x.copy_to(p, stdx::vector_aligned);
        } else if constexpr (MaskL > 0) {
            typename simd::mask_type m{};
            for (int i = 0; i < m.size(); ++i)
                m[i] = i >= MaskL;
            where(m, x).copy_to(p, stdx::vector_aligned);
        } else {
            typename simd::mask_type m{};
            for (int i = 0; i < m.size(); ++i)
                m[i] = i < m.size() - MaskL;
            where(m, x).copy_to(p, stdx::vector_aligned);
        }
    }
};

template <class T, class Abi, StorageOrder Order = StorageOrder::ColMajor>
using view = simd_view_types<std::remove_const_t<T>, Abi>::template view<T, Order>;

template <class Abi, StorageOrder Order = StorageOrder::ColMajor>
using real_view = simd_view_types<real_t, Abi>::template view<const real_t, Order>;
template <class Abi, StorageOrder Order = StorageOrder::ColMajor>
using mut_real_view = simd_view_types<real_t, Abi>::template view<real_t, Order>;

template <class T, class Abi, StorageOrder Order>
struct uview {
    using value_type = T;
    value_type *data;
    index_t outer_stride;

    using types                             = simd_view_types<std::remove_const_t<T>, Abi>;
    using view                              = types::template view<T, Order>;
    using mut_view                          = types::template view<std::remove_const_t<T>, Order>;
    using mut_uview                         = uview<std::remove_const_t<T>, Abi, Order>;
    using simd                              = typename types::simd;
    static constexpr ptrdiff_t inner_stride = typename types::simd_stride_t();

    [[gnu::always_inline]] value_type &operator()(index_t r, index_t c) const noexcept {
        ptrdiff_t i0 = Order == StorageOrder::RowMajor ? c : r;
        ptrdiff_t i1 = Order == StorageOrder::RowMajor ? r : c;
        return data[inner_stride * (i0 + i1 * static_cast<ptrdiff_t>(outer_stride))];
    }
    [[gnu::always_inline]] simd load(index_t r, index_t c) const noexcept {
        return types::aligned_load(&operator()(r, c));
    }
    template <int MaskL = 0>
    [[gnu::always_inline]] void store(simd x, index_t r, index_t c) const noexcept
        requires(!std::is_const_v<T>)
    {
        types::aligned_store(x, &operator()(r, c));
    }
    template <class Self>
    [[gnu::always_inline]] Self block(this const Self &self, index_t r, index_t c) noexcept {
        return {&self(r, c), self.outer_stride};
    }
    template <class Self>
    [[gnu::always_inline]] Self middle_rows(this const Self &self, index_t r) noexcept {
        return {&self(r, 0), self.outer_stride};
    }
    template <class Self>
    [[gnu::always_inline]] Self middle_cols(this const Self &self, index_t c) noexcept {
        return {&self(0, c), self.outer_stride};
    }

    [[gnu::always_inline]] uview(value_type *data, index_t outer_stride) noexcept
        : data{data}, outer_stride{outer_stride} {}
    [[gnu::always_inline]] uview(const mut_view &v) noexcept
        requires std::is_const_v<T>
        : data{v.data}, outer_stride{v.outer_stride()} {}
    [[gnu::always_inline]] uview(const view &v) noexcept
        : data{v.data}, outer_stride{v.outer_stride()} {}
    [[gnu::always_inline]] uview(const mut_uview &o) noexcept
        requires std::is_const_v<T>
        : data{o.data}, outer_stride{o.outer_stride} {}
    [[gnu::always_inline]] uview(const uview &o) = default;
};

template <index_t Size, class T, class Abi, StorageOrder Order>
struct cached_uview {
    using value_type = T;
    value_type *const data[Size];

    using types                             = simd_view_types<std::remove_const_t<T>, Abi>;
    using simd                              = typename types::simd;
    static constexpr ptrdiff_t inner_stride = typename types::simd_stride_t();

    [[gnu::always_inline]] value_type &operator()(index_t r, index_t c) const noexcept {
        ptrdiff_t i0 = Order == StorageOrder::RowMajor ? c : r;
        index_t i1   = Order == StorageOrder::RowMajor ? r : c;
        assert(i1 < Size);
        return data[i1][i0 * inner_stride];
    }
    [[gnu::always_inline]] simd load(index_t r, index_t c) const noexcept {
        return types::aligned_load(&operator()(r, c));
    }
    template <int MaskL = 0>
    [[gnu::always_inline]] void store(simd x, index_t r, index_t c) const noexcept
        requires(!std::is_const_v<T>)
    {
        types::aligned_store(x, &operator()(r, c));
    }

    template <index_t... Is>
    [[gnu::always_inline]] cached_uview(const uview<T, Abi, Order> &o,
                                        std::integer_sequence<index_t, Is...>) noexcept
        : data{(o.data + Is * o.inner_stride * static_cast<ptrdiff_t>(o.outer_stride))...} {}
    [[gnu::always_inline]] cached_uview(const uview<T, Abi, Order> &o) noexcept
        : cached_uview{o, std::make_integer_sequence<index_t, Size>()} {}
};

template <index_t Rows, index_t Cols, class T, class Abi, StorageOrder Order>
[[gnu::always_inline]] cached_uview<Order == StorageOrder::ColMajor ? Cols : Rows, T, Abi, Order>
with_cached_access(const uview<T, Abi, Order> &o) noexcept {
    return {o};
}

} // namespace batmat::linalg
