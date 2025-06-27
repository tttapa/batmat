#pragma once

#include <batmat/simd.hpp>

namespace batmat::ops {

/// Inverse square root.
template <class T, class Abi>
datapar::simd<T, Abi> rsqrt(datapar::simd<T, Abi> x) {
    return datapar::simd<T, Abi>{1} / sqrt(x);
}

template <std::floating_point T>
T rsqrt(T x) {
    using std::sqrt;
    return 1 / sqrt(x);
}

namespace detail {

#if __AVX512F__
/// Approximation of inverse square root up to a relative error of 2⁻¹⁴.
inline auto rsqrt_0(datapar::deduced_simd<double, 8> x) {
    return decltype(x){_mm512_rsqrt14_pd(static_cast<__m512d>(x))};
}
inline auto rsqrt_0(datapar::deduced_simd<float, 16> x) {
    return decltype(x){_mm512_rsqrt14_ps(static_cast<__m512>(x))};
}
inline auto rsqrt_0(datapar::deduced_simd<double, 4> x) {
    return decltype(x){_mm256_rsqrt14_pd(static_cast<__m256d>(x))};
}
inline auto rsqrt_0(datapar::deduced_simd<float, 8> x) {
    return decltype(x){_mm256_rsqrt14_ps(static_cast<__m256>(x))};
}
inline auto rsqrt_0(datapar::deduced_simd<double, 2> x) {
    return decltype(x){_mm_rsqrt14_pd(static_cast<__m128d>(x))};
}
inline auto rsqrt_0(datapar::deduced_simd<float, 4> x) {
    return decltype(x){_mm_rsqrt14_ps(static_cast<__m128>(x))};
}
#elif __AVX2__
/// Approximation of inverse square root up to a relative error of 1.5×2⁻¹².
inline auto rsqrt_0(datapar::deduced_simd<float, 8> x) {
    return decltype(x){_mm256_rsqrt_ps(static_cast<__m256>(x))};
}
#endif

/// rsqrt_0 with a single Newton iteration of refinement.
template <class T, class Abi>
datapar::simd<T, Abi> rsqrt_1(datapar::simd<T, Abi> x) {
    auto y = rsqrt_0(x);
    const datapar::simd<T, Abi> half{T(0.5)}, three_halves{T(1.5)};
    return y * (three_halves - (half * x * y * y));
}

/// rsqrt_0 with two Newton iterations of refinement.
template <class T, class Abi>
datapar::simd<T, Abi> rsqrt_2(datapar::simd<T, Abi> x) {
    auto y = rsqrt_1(x);
    const datapar::simd<T, Abi> half{T(0.5)}, three_halves{T(1.5)};
    return y * (three_halves - (half * x * y * y));
}
} // namespace detail

#if __AVX512F__
template <>
inline datapar::deduced_simd<double, 8> rsqrt(datapar::deduced_simd<double, 8> x) {
    return detail::rsqrt_2(x);
}
template <>
inline datapar::deduced_simd<double, 4> rsqrt(datapar::deduced_simd<double, 4> x) {
    return detail::rsqrt_2(x);
}
template <>
inline datapar::deduced_simd<double, 2> rsqrt(datapar::deduced_simd<double, 2> x) {
    return detail::rsqrt_2(x);
}
template <>
inline datapar::deduced_simd<float, 16> rsqrt(datapar::deduced_simd<float, 16> x) {
    return detail::rsqrt_1(x);
}
template <>
inline datapar::deduced_simd<float, 8> rsqrt(datapar::deduced_simd<float, 8> x) {
    return detail::rsqrt_1(x);
}
template <>
inline datapar::deduced_simd<float, 4> rsqrt(datapar::deduced_simd<float, 4> x) {
    return detail::rsqrt_1(x);
}
#elif __AVX2__
template <>
inline datapar::deduced_simd<float, 8> rsqrt(datapar::deduced_simd<float, 8> x) {
    return detail::rsqrt_1(x); // TODO: one or two Newton iterations?
}
#endif

#if BATMAT_SCALAR_APPROX_INV_SQRT // Not worth it
namespace detail {

inline float rsqrt_0(float x) {
    __m128 input  = _mm_set_ss(x);
    __m128 result = _mm_rsqrt14_ss(input, input);
    return _mm_cvtss_f32(result);
}

inline double rsqrt_0(double x) {
    __m128d input  = _mm_set_sd(x);
    __m128d result = _mm_rsqrt14_sd(input, input);
    return _mm_cvtsd_f64(result);
}

/// rsqrt_0 with a single Newton iteration of refinement.
template <std::floating_point T>
T rsqrt_1(T x) {
    auto y = rsqrt_0(x);
    return y * (T(1.5) - (T(0.5) * x * y * y));
}

/// rsqrt_0 with two Newton iterations of refinement.
template <std::floating_point T>
T rsqrt_2(T x) {
    auto y = rsqrt_1(x);
    return y * (T(1.5) - (T(0.5) * x * y * y));
}

} // namespace detail

inline double rsqrt(double x) { return detail::rsqrt_2(x); }
inline float rsqrt(float x) { return detail::rsqrt_1(x); }

#endif

} // namespace batmat::ops
