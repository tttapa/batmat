#pragma once

#include <koqkatoo/assume.hpp>
#include <experimental/simd>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace koqkatoo::linalg::compact::micro_kernels {
namespace detail {
namespace stdx = std::experimental;

template <class T>
struct floating_point_to_int; // deliberately undefined

template <>
struct floating_point_to_int<float> {
    using type = std::uint32_t;
    static_assert(sizeof(float) == sizeof(type));
};

template <>
struct floating_point_to_int<double> {
    using type = std::uint64_t;
    static_assert(sizeof(double) == sizeof(type));
};

template <class T>
using floating_point_to_int_t = typename floating_point_to_int<T>::type;

/// Conditionally negates the sign bit of @p x, depending on @p signs, which
/// should contain only ±0 (i.e. only the sign bit of an IEEE-754 floating point
/// number).
template <class T, class Abi>
stdx::simd<T, Abi> cneg(stdx::simd<T, Abi> x, stdx::simd<T, Abi> signs) {
    return x * copysign(stdx::simd<T, Abi>{1}, signs);
}

template <class T, class Abi>
    requires(requires { typename floating_point_to_int_t<T>; } &&
             std::numeric_limits<T>::is_iec559 &&
             std::is_trivially_copyable_v<stdx::simd<T, Abi>>)
[[gnu::always_inline]] stdx::simd<T, Abi> cneg(stdx::simd<T, Abi> x,
                                               stdx::simd<T, Abi> signs) {
#ifndef __clang__ // TODO: enable once Clang supports operator==
    KOQKATOO_ASSUME(all_of(signs == 0));
#endif
    using flt_simd = stdx::simd<T, Abi>;
    using int_type = floating_point_to_int_t<T>;
    using int_simd = stdx::rebind_simd_t<int_type, flt_simd>;
    auto r = std::bit_cast<int_simd>(x) ^ std::bit_cast<int_simd>(signs);
    return std::bit_cast<flt_simd>(r);
}

} // namespace detail

using detail::cneg;

} // namespace koqkatoo::linalg::compact::micro_kernels
