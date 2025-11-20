#pragma once

#include <batmat/simd.hpp>
#include <immintrin.h>

namespace batmat::ops::detail {

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 8>
gather(datapar::deduced_simd<float, 8> src, __m256 mask, datapar::deduced_simd<int32_t, 8> vindex,
       const float *base_addr) {
    return datapar::deduced_simd<float, 8>{_mm256_mask_i32gather_ps(
        static_cast<__m256>(src), base_addr, static_cast<__m256i>(vindex), mask, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 4>
gather(datapar::deduced_simd<double, 4> src, __m256d mask, datapar::deduced_simd<int64_t, 4> vindex,
       const double *base_addr) {
    return datapar::deduced_simd<double, 4>{_mm256_mask_i64gather_pd(
        static_cast<__m256d>(src), base_addr, static_cast<__m256i>(vindex), mask, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 4>
gather(datapar::deduced_simd<double, 4> src, __m256d mask, datapar::deduced_simd<int32_t, 4> vindex,
       const double *base_addr) {
    return datapar::deduced_simd<double, 4>{_mm256_mask_i32gather_pd(
        static_cast<__m256d>(src), base_addr, static_cast<__m128i>(vindex), mask, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 4>
gather(datapar::deduced_simd<float, 4> src, __m128 mask, datapar::deduced_simd<int32_t, 4> vindex,
       const float *base_addr) {
    return datapar::deduced_simd<float, 4>{_mm_mask_i32gather_ps(
        static_cast<__m128>(src), base_addr, static_cast<__m128i>(vindex), mask, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 4>
gather(datapar::deduced_simd<float, 4> src, __m128 mask, datapar::deduced_simd<int64_t, 4> vindex,
       const float *base_addr) {
    auto src128 = static_cast<__m128>(src);
    auto idx256 = static_cast<__m256i>(vindex);
    // split inputs into low and high 2x32-bit parts
    __m128 src_lo = src128, src_hi = _mm_shuffle_ps(src128, src128, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 mask_lo = mask, mask_hi = _mm_shuffle_ps(mask, mask, _MM_SHUFFLE(3, 2, 3, 2));
    __m128i idx_lo = _mm256_castsi256_si128(idx256), idx_hi = _mm256_extracti128_si256(idx256, 1);
    // two gathers
    __m128 g_lo = _mm_mask_i64gather_ps(src_lo, base_addr, idx_lo, mask_lo, Scale);
    __m128 g_hi = _mm_mask_i64gather_ps(src_hi, base_addr, idx_hi, mask_hi, Scale);
    // combine: [g_lo0, g_lo1, g_hi0, g_hi1]
    return datapar::deduced_simd<float, 4>{_mm_movelh_ps(g_lo, g_hi)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 2>
gather(datapar::deduced_simd<double, 2> src, __m128d mask, datapar::deduced_simd<int64_t, 2> vindex,
       const double *base_addr) {
    return datapar::deduced_simd<double, 2>{_mm_mask_i64gather_pd(
        static_cast<__m128d>(src), base_addr, static_cast<__m128i>(vindex), mask, Scale)};
}

#if !BATMAT_WITH_GSI_HPC_SIMD // TODO
template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 2>
gather(datapar::deduced_simd<double, 2> src, __m128d mask, datapar::deduced_simd<int32_t, 2> vindex,
       const double *base_addr) {
    return datapar::deduced_simd<double, 2>{_mm_mask_i32gather_pd(
        static_cast<__m128d>(src), base_addr, static_cast<__m128i>(vindex), mask, Scale)};
}
#endif

} // namespace batmat::ops::detail
