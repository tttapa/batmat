#pragma once

#include <batmat/matrix/matrix.hpp>
#include <batmat/matrix/view.hpp>
#include <experimental/simd>
#include <type_traits>

namespace batmat::linalg {

using batmat::matrix::StorageOrder;
namespace stdx = std::experimental;

namespace detail {

template <class V>
struct simdified_view_type;

template <class T, class S, class L, StorageOrder O>
struct simdified_view_type<batmat::matrix::View<T, index_t, S, S, L, O>> {
    using value_type   = std::remove_const_t<T>;
    using abi_type     = stdx::simd_abi::deduce_t<value_type, S{}>;
    using simd_type    = stdx::simd<value_type, abi_type>;
    using stride       = stdx::simd_size<value_type, abi_type>;
    using alignment    = stdx::memory_alignment<simd_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<T, index_t, stride, stride, layer_stride, O>;
    static_assert(stride::value * sizeof(value_type) >= alignment::value);
};

template <class T, class S, class L, StorageOrder O>
struct simdified_view_type<const batmat::matrix::View<T, index_t, S, S, L, O>> {
    using value_type   = std::remove_const_t<T>;
    using abi_type     = stdx::simd_abi::deduce_t<value_type, S{}>;
    using simd_type    = stdx::simd<value_type, abi_type>;
    using stride       = stdx::simd_size<value_type, abi_type>;
    using alignment    = stdx::memory_alignment<simd_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<T, index_t, stride, stride, layer_stride, O>;
    static_assert(stride::value * sizeof(value_type) >= alignment::value);
};

template <class T, class I, class S, StorageOrder O, class A>
struct simdified_view_type<batmat::matrix::Matrix<T, I, S, S, O, A>> {
    using value_type   = T;
    using abi_type     = stdx::simd_abi::deduce_t<value_type, S{}>;
    using simd_type    = stdx::simd<value_type, abi_type>;
    using stride       = stdx::simd_size<value_type, abi_type>;
    using alignment    = stdx::memory_alignment<simd_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<T, index_t, stride, stride, layer_stride, O>;
    static_assert(A{} >= alignment::value);
};

template <class T, class I, class S, class A, StorageOrder O>
struct simdified_view_type<const batmat::matrix::Matrix<T, I, S, S, O, A>> {
    using value_type   = T;
    using abi_type     = stdx::simd_abi::deduce_t<value_type, S{}>;
    using simd_type    = stdx::simd<value_type, abi_type>;
    using stride       = stdx::simd_size<value_type, abi_type>;
    using alignment    = stdx::memory_alignment<simd_type>;
    using layer_stride = batmat::matrix::DefaultStride;
    using type         = batmat::matrix::View<const T, index_t, stride, stride, layer_stride, O>;
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
template <simdifiable V>
using simdified_value_t = typename simdified_view_type<V>::value_type;
template <simdifiable V>
using simdified_abi_t = typename simdified_view_type<V>::abi_type;

template <simdifiable V, simdifiable... Vs>
inline constexpr bool simdify_compatible =
    (std::is_same_v<simdified_value_t<V>, simdified_value_t<Vs>> && ...) &&
    (std::is_same_v<simdified_abi_t<V>, simdified_abi_t<Vs>> && ...);

constexpr auto simdify(simdifiable auto &&a) -> simdified_view_t<decltype(a)> {
    if constexpr (requires { a.data(); }) // TODO: can we make this consistent?
        return simdified_view_t<decltype(a)>{{
            .data         = a.data(),
            .rows         = a.rows(),
            .cols         = a.cols(),
            .outer_stride = a.outer_stride(),
        }};
    else
        return simdified_view_t<decltype(a)>{{
            .data         = a.data,
            .rows         = a.rows(),
            .cols         = a.cols(),
            .outer_stride = a.outer_stride(),
        }};
}

} // namespace batmat::linalg
