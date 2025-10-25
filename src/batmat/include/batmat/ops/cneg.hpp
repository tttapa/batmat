#pragma once

#include <batmat/assume.hpp>
#include <batmat/simd.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace batmat::ops {
namespace detail {

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
template <class T>
T cneg(T x, T signs) {
    using std::copysign;
    return x * copysign(T{1}, signs);
}

template <class T, class Abi>
    requires(requires { typename floating_point_to_int_t<T>; } &&
             std::numeric_limits<T>::is_iec559 && std::is_trivially_copyable_v<T>)
[[gnu::always_inline]] inline T cneg(T x, T signs) {
    BATMAT_ASSUME(signs == 0);
    using int_type = floating_point_to_int_t<T>;
    auto r         = std::bit_cast<int_type>(x) ^ std::bit_cast<int_type>(signs);
    return std::bit_cast<T>(r);
}

template <class T, class Abi>
    requires(requires {
        typename floating_point_to_int_t<T>;
    } && std::numeric_limits<T>::is_iec559 && std::is_trivially_copyable_v<datapar::simd<T, Abi>>)
[[gnu::always_inline]] inline datapar::simd<T, Abi> cneg(datapar::simd<T, Abi> x,
                                                         datapar::simd<T, Abi> signs) {
#if !BATMAT_WITH_GSI_HPC_SIMD // TODO: enable once Clang supports operator==
    BATMAT_ASSUME(all_of(signs == 0));
#endif
    using flt_simd = datapar::simd<T, Abi>;
    using int_type = floating_point_to_int_t<T>;
    using int_simd = datapar::deduced_simd<int_type, flt_simd::size()>;
    auto r         = std::bit_cast<int_simd>(x) ^ std::bit_cast<int_simd>(signs);
    return std::bit_cast<flt_simd>(r);
}

} // namespace detail

using detail::cneg;

} // namespace batmat::ops
