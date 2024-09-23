#pragma once

#include <experimental/simd>

namespace koqkatoo::linalg::compact::micro_kernels {

namespace stdx = std::experimental;

/// Inverse square root.
template <class T, class Abi>
stdx::simd<T, Abi> rsqrt(stdx::simd<T, Abi> x) {
    return 1 / sqrt(x);
}

#if __AVX512F__
#ifdef _GLIBCXX_EXPERIMENTAL_SIMD // needed for stdx::__intrinsic_type_t
namespace detail {

inline __m512d rsqrt14(__m512d x) { return _mm512_rsqrt14_pd(x); }
inline __m512 rsqrt14(__m512 x) { return _mm512_rsqrt14_ps(x); }
inline __m256d rsqrt14(__m256d x) { return _mm256_rsqrt14_pd(x); }
inline __m256 rsqrt14(__m256 x) { return _mm256_rsqrt14_ps(x); }
inline __m128d rsqrt14(__m128d x) { return _mm_rsqrt14_pd(x); }
inline __m128 rsqrt14(__m128 x) { return _mm_rsqrt14_ps(x); }

/// Approximation of inverse square root up to a relative error of 2⁻¹⁴.
template <class T, class Abi>
stdx::simd<T, Abi> rsqrt_0(stdx::simd<T, Abi> x) {
    using intrin = stdx::__intrinsic_type_t<T, x.size()>;
    return stdx::simd<T, Abi>{rsqrt14(static_cast<intrin>(x))};
}

/// rsqrt_0 with a single Newton iteration of refinement.
template <class T, class Abi>
stdx::simd<T, Abi> rsqrt_1(stdx::simd<T, Abi> x) {
    auto y = rsqrt_0(x);
    return y * (T(1.5) - (x / 2 * y * y));
}

/// rsqrt_0 with two Newton iterations of refinement.
template <class T, class Abi>
stdx::simd<T, Abi> rsqrt_2(stdx::simd<T, Abi> x) {
    auto y = rsqrt_1(x);
    return y * (T(1.5) - (x / 2 * y * y));
}
} // namespace detail

template <>
inline stdx::simd<double, stdx::simd_abi::deduce_t<double, 8>>
rsqrt(stdx::simd<double, stdx::simd_abi::deduce_t<double, 8>> x) {
    return detail::rsqrt_2(x);
}
template <>
inline stdx::simd<double, stdx::simd_abi::deduce_t<double, 4>>
rsqrt(stdx::simd<double, stdx::simd_abi::deduce_t<double, 4>> x) {
    return detail::rsqrt_2(x);
}
template <>
inline stdx::simd<double, stdx::simd_abi::deduce_t<double, 2>>
rsqrt(stdx::simd<double, stdx::simd_abi::deduce_t<double, 2>> x) {
    return detail::rsqrt_2(x);
}
template <>
inline stdx::simd<float, stdx::simd_abi::deduce_t<float, 16>>
rsqrt(stdx::simd<float, stdx::simd_abi::deduce_t<float, 16>> x) {
    return detail::rsqrt_1(x);
}
template <>
inline stdx::simd<float, stdx::simd_abi::deduce_t<float, 8>>
rsqrt(stdx::simd<float, stdx::simd_abi::deduce_t<float, 8>> x) {
    return detail::rsqrt_1(x);
}
template <>
inline stdx::simd<float, stdx::simd_abi::deduce_t<float, 4>>
rsqrt(stdx::simd<float, stdx::simd_abi::deduce_t<float, 4>> x) {
    return detail::rsqrt_1(x);
}
#else
#pragma message("Fast inverse square roots not supported")
#endif
#endif

} // namespace koqkatoo::linalg::compact::micro_kernels
