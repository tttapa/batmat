#pragma once

#include <batmat/matrix/matrix.hpp>
#include <batmat/matrix/view.hpp>
#include <batmat/simd.hpp>
#include <guanaqo/mat-view.hpp>
#include <type_traits>

// TODO: consistent API for guanaqo::MatrixView
namespace guanaqo {
template <class T, class I, class S, StorageOrder O>
constexpr auto data(const MatrixView<T, I, S, O> &v) {
    return v.data;
}
template <class T, class I, class S, StorageOrder O>
constexpr auto rows(const MatrixView<T, I, S, O> &v) {
    return v.rows;
}
template <class T, class I, class S, StorageOrder O>
constexpr auto cols(const MatrixView<T, I, S, O> &v) {
    return v.cols;
}
template <class T, class I, class S, StorageOrder O>
constexpr auto outer_stride(const MatrixView<T, I, S, O> &v) {
    return v.outer_stride;
}
template <class T, class I, class S, StorageOrder O>
constexpr auto depth(const MatrixView<T, I, S, O> &) {
    return std::integral_constant<I, 1>{};
}
} // namespace guanaqo

namespace batmat::linalg {

using batmat::matrix::StorageOrder;

namespace detail {

template <class V>
struct simdified_view_type;

template <class T, class S, class L, StorageOrder O>
struct simdified_view_type<batmat::matrix::View<T, index_t, S, S, L, O>> {
    using value_type   = std::remove_const_t<T>;
    using abi_type     = datapar::deduced_abi<value_type, S{}>;
    using simd_type    = datapar::simd<value_type, abi_type>;
    using stride       = datapar::simd_size<value_type, abi_type>;
    using alignment    = datapar::simd_align<value_type, abi_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<T, index_t, stride, stride, layer_stride, O>;
    static_assert(stride::value * sizeof(value_type) >= alignment::value);
};

template <class T, class S, class L, StorageOrder O>
struct simdified_view_type<const batmat::matrix::View<T, index_t, S, S, L, O>> {
    using value_type   = std::remove_const_t<T>;
    using abi_type     = datapar::deduced_abi<value_type, S{}>;
    using simd_type    = datapar::simd<value_type, abi_type>;
    using stride       = datapar::simd_size<value_type, abi_type>;
    using alignment    = datapar::simd_align<value_type, abi_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<T, index_t, stride, stride, layer_stride, O>;
    static_assert(stride::value * sizeof(value_type) >= alignment::value);
};

template <class T, class I, StorageOrder O>
struct simdified_view_type<guanaqo::MatrixView<T, I, std::integral_constant<I, 1>, O>> {
    using value_type   = std::remove_const_t<T>;
    using abi_type     = datapar::scalar_abi<value_type>;
    using simd_type    = datapar::simd<value_type, abi_type>;
    using stride       = datapar::simd_size<value_type, abi_type>;
    using alignment    = datapar::simd_align<value_type, abi_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<T, index_t, stride, stride, layer_stride, O>;
    static_assert(stride::value * sizeof(value_type) >= alignment::value);
};

template <class T, class I, StorageOrder O>
struct simdified_view_type<const guanaqo::MatrixView<T, I, std::integral_constant<I, 1>, O>> {
    using value_type   = std::remove_const_t<T>;
    using abi_type     = datapar::scalar_abi<value_type>;
    using simd_type    = datapar::simd<value_type, abi_type>;
    using stride       = datapar::simd_size<value_type, abi_type>;
    using alignment    = datapar::simd_align<value_type, abi_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<T, index_t, stride, stride, layer_stride, O>;
    static_assert(stride::value * sizeof(value_type) >= alignment::value);
};

template <class T, class I, class S, StorageOrder O, class A>
struct simdified_view_type<batmat::matrix::Matrix<T, I, S, S, O, A>> {
    using value_type   = T;
    using abi_type     = datapar::deduced_abi<value_type, S{}>;
    using simd_type    = datapar::simd<value_type, abi_type>;
    using stride       = datapar::simd_size<value_type, abi_type>;
    using alignment    = datapar::simd_align<value_type, abi_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<T, index_t, stride, stride, layer_stride, O>;
    static_assert(A{} >= alignment::value);
};

template <class T, class I, class S, class A, StorageOrder O>
struct simdified_view_type<const batmat::matrix::Matrix<T, I, S, S, O, A>> {
    using value_type   = T;
    using abi_type     = datapar::deduced_abi<value_type, S{}>;
    using simd_type    = datapar::simd<value_type, abi_type>;
    using stride       = datapar::simd_size<value_type, abi_type>;
    using alignment    = datapar::simd_align<value_type, abi_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<const T, index_t, stride, stride, layer_stride, O>;
    static_assert(A{} >= alignment::value);
};

// For multiple batches

template <class V>
struct simdified_multi_view_type;

template <class T, class S, class L, StorageOrder O>
struct simdified_multi_view_type<batmat::matrix::View<T, index_t, S, index_t, L, O>> {
    using value_type   = std::remove_const_t<T>;
    using abi_type     = datapar::deduced_abi<value_type, S{}>;
    using simd_type    = datapar::simd<value_type, abi_type>;
    using stride       = datapar::simd_size<value_type, abi_type>;
    using alignment    = datapar::simd_align<value_type, abi_type>;
    using layer_stride = L;
    using type         = batmat::matrix::View<T, index_t, stride, index_t, layer_stride, O>;
    static_assert(stride::value * sizeof(value_type) >= alignment::value);
};

template <class T, class S, class L, StorageOrder O>
struct simdified_multi_view_type<const batmat::matrix::View<T, index_t, S, index_t, L, O>> {
    using value_type   = std::remove_const_t<T>;
    using abi_type     = datapar::deduced_abi<value_type, S{}>;
    using simd_type    = datapar::simd<value_type, abi_type>;
    using stride       = datapar::simd_size<value_type, abi_type>;
    using alignment    = datapar::simd_align<value_type, abi_type>;
    using layer_stride = L;
    using type         = batmat::matrix::View<T, index_t, stride, index_t, layer_stride, O>;
    static_assert(stride::value * sizeof(value_type) >= alignment::value);
};

template <class T, class I, class S, StorageOrder O, class A>
struct simdified_multi_view_type<batmat::matrix::Matrix<T, I, S, I, O, A>> {
    using value_type   = T;
    using abi_type     = datapar::deduced_abi<value_type, S{}>;
    using simd_type    = datapar::simd<value_type, abi_type>;
    using stride       = datapar::simd_size<value_type, abi_type>;
    using alignment    = datapar::simd_align<value_type, abi_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<T, index_t, stride, index_t, layer_stride, O>;
    static_assert(A{} >= alignment::value);
};

template <class T, class I, class S, class A, StorageOrder O>
struct simdified_multi_view_type<const batmat::matrix::Matrix<T, I, S, I, O, A>> {
    using value_type   = T;
    using abi_type     = datapar::deduced_abi<value_type, S{}>;
    using simd_type    = datapar::simd<value_type, abi_type>;
    using stride       = datapar::simd_size<value_type, abi_type>;
    using alignment    = datapar::simd_align<value_type, abi_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<const T, index_t, stride, index_t, layer_stride, O>;
    static_assert(A{} >= alignment::value);
};

} // namespace detail

/// Convert the given view or matrix type @p V (batmat::matrix::View or batmat::matrix::Matrix) to
/// a batched view type using a deduced SIMD type. This conversion takes place in the wrapper around
/// the optimized implementations (which require views with a proper SIMD-compatible stride).
template <class V>
using simdified_view_type = detail::simdified_view_type<std::remove_reference_t<V>>;

template <class V>
concept simdifiable = requires { typename simdified_view_type<std::remove_reference_t<V>>::type; };

template <simdifiable V>
using simdified_view_t = typename simdified_view_type<V>::type;

namespace detail {

template <class>
struct simdified_value;

template <simdifiable V>
struct simdified_value<V> {
    using type = typename batmat::linalg::simdified_view_type<V>::value_type;
};

template <class>
struct simdified_abi;

template <simdifiable V>
struct simdified_abi<V> {
    using type = typename batmat::linalg::simdified_view_type<V>::abi_type;
};

} // namespace detail

template <class V>
using simdified_value_t = typename detail::simdified_value<V>::type;
template <class V>
using simdified_abi_t = typename detail::simdified_abi<V>::type;

template <class...>
inline constexpr bool simdify_compatible = false;

template <simdifiable V, simdifiable... Vs>
inline constexpr bool simdify_compatible<V, Vs...> =
    (std::is_same_v<simdified_value_t<V>, simdified_value_t<Vs>> && ...) &&
    (std::is_same_v<simdified_abi_t<V>, simdified_abi_t<Vs>> && ...);

constexpr auto simdify(simdifiable auto &&a) -> simdified_view_t<decltype(a)> {
    return simdified_view_t<decltype(a)>{{
        .data         = data(a),
        .rows         = rows(a),
        .cols         = cols(a),
        .outer_stride = outer_stride(a),
    }};
}

template <class V>
using simdified_multi_view_type = detail::simdified_multi_view_type<std::remove_reference_t<V>>;

template <class V>
concept simdifiable_multi =
    requires { typename simdified_multi_view_type<std::remove_reference_t<V>>::type; };

template <simdifiable_multi V>
using simdified_multi_view_t = typename simdified_multi_view_type<V>::type;

namespace detail {

template <simdifiable_multi V>
struct simdified_value<V> {
    using type = typename batmat::linalg::simdified_multi_view_type<V>::value_type;
};

template <simdifiable_multi V>
struct simdified_abi<V> {
    using type = typename batmat::linalg::simdified_multi_view_type<V>::abi_type;
};

} // namespace detail

template <simdifiable_multi V, simdifiable_multi... Vs>
inline constexpr bool simdify_compatible<V, Vs...> =
    (std::is_same_v<simdified_value_t<V>, simdified_value_t<Vs>> && ...) &&
    (std::is_same_v<simdified_abi_t<V>, simdified_abi_t<Vs>> && ...);

constexpr auto simdify(simdifiable_multi auto &&a) -> simdified_multi_view_t<decltype(a)> {
    return simdified_multi_view_t<decltype(a)>{{
        .data         = data(a),
        .depth        = depth(a),
        .rows         = rows(a),
        .cols         = cols(a),
        .outer_stride = outer_stride(a),
    }};
}

} // namespace batmat::linalg
