#pragma once

#include <batmat/simd.hpp>
#include <immintrin.h>

namespace batmat::ops::detail {

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 8>
gather(datapar::deduced_simd<float, 8> src, __m256 mask, datapar::deduced_simd<int64_t, 8> vindex,
       const float *base_addr) {
    auto src256 = static_cast<__m256>(src);
    // split src and mask into low and high 4-float parts
    __m128 src_lo = _mm256_castps256_ps128(src256), src_hi = _mm256_extractf128_ps(src256, 1);
    __m128 mask_lo = _mm256_castps256_ps128(mask), mask_hi = _mm256_extractf128_ps(mask, 1);
    // split 8 x 64-bit indices into low and high 4 x 64-bit parts
#if BATMAT_WITH_GSI_HPC_SIMD
    auto [idx_lo, idx_hi] = chunk<datapar::deduced_simd<int64_t, 4>>(vindex);
#else
    auto [idx_lo, idx_hi] = split<4, 4>(vindex);
#endif
    auto idx_lo_256 = static_cast<__m256i>(idx_lo), idx_hi_256 = static_cast<__m256i>(idx_hi);
    // two gathers of 4 floats each using 4 x 64-bit indices
    __m128 g_lo = _mm256_mask_i64gather_ps(src_lo, base_addr, idx_lo_256, mask_lo, Scale);
    __m128 g_hi = _mm256_mask_i64gather_ps(src_hi, base_addr, idx_hi_256, mask_hi, Scale);
    // combine: [g_lo0, g_lo1, g_lo2, g_lo3, g_hi0, g_hi1, g_hi2, g_hi3]
    return datapar::deduced_simd<float, 8>{_mm256_set_m128(g_hi, g_lo)};
}

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
    return datapar::deduced_simd<float, 4>{_mm256_mask_i64gather_ps(
        static_cast<__m128>(src), base_addr, static_cast<__m256i>(vindex), mask, Scale)};
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
