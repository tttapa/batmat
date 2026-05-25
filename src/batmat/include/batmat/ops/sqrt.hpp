#pragma once

#include <batmat/simd.hpp>

namespace batmat::ops {

/// @addtogroup topic-low-level-ops
/// @{

/// @name Square root
/// @{

/// Square root.
template <std::floating_point T>
T sqrt(T x) {
    using std::sqrt;
    return sqrt(x);
}

#ifdef __SSE2__

/// Square root implementation that avoids setting errno.
inline float sqrt(float x) {
    __m128 input  = _mm_set_ss(x);
    __m128 result = _mm_sqrt_ss(input);
    return _mm_cvtss_f32(result);
}

/// Square root implementation that avoids setting errno.
inline double sqrt(double x) {
    __m128d input  = _mm_set_sd(x);
    __m128d result = _mm_sqrt_sd(input, input);
    return _mm_cvtsd_f64(result);
}

#elif defined(__aarch64__) && defined(__GNUC__)

/// Square root implementation that avoids setting errno.
inline float sqrt(float x) {
    float result;
    __asm__("fsqrt %s0, %s1" : "=w"(result) : "w"(x));
    return result;
}

/// Square root implementation that avoids setting errno.
inline double sqrt(double x) {
    double result;
    __asm__("fsqrt %d0, %d1" : "=w"(result) : "w"(x));
    return result;
}

#endif

/// @}

/// @}

} // namespace batmat::ops
