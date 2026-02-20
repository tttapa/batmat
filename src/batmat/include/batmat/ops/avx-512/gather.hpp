#pragma once

#include <batmat/simd.hpp>
#include <immintrin.h>

namespace batmat::ops::detail {

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 8>
gather(datapar::deduced_simd<double, 8> src, __mmask8 mask,
       datapar::deduced_simd<int64_t, 8> vindex, const double *base_addr) {
    return datapar::deduced_simd<double, 8>{_mm512_mask_i64gather_pd(
        static_cast<__m512d>(src), mask, static_cast<__m512i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 4>
gather(datapar::deduced_simd<double, 4> src, __mmask8 mask,
       datapar::deduced_simd<int64_t, 4> vindex, const double *base_addr) {
    return datapar::deduced_simd<double, 4>{_mm256_mmask_i64gather_pd(
        static_cast<__m256d>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 2>
gather(datapar::deduced_simd<double, 2> src, __mmask8 mask,
       datapar::deduced_simd<int64_t, 2> vindex, const double *base_addr) {
    return datapar::deduced_simd<double, 2>{_mm_mmask_i64gather_pd(
        static_cast<__m128d>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 8>
gather(datapar::deduced_simd<double, 8> src, __mmask8 mask,
       datapar::deduced_simd<int32_t, 8> vindex, const double *base_addr) {
    return datapar::deduced_simd<double, 8>{_mm512_mask_i32gather_pd(
        static_cast<__m512d>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 4>
gather(datapar::deduced_simd<double, 4> src, __mmask8 mask,
       datapar::deduced_simd<int32_t, 4> vindex, const double *base_addr) {
    return datapar::deduced_simd<double, 4>{_mm256_mmask_i32gather_pd(
        static_cast<__m256d>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 2>
gather(datapar::deduced_simd<double, 2> src, __mmask8 mask,
       datapar::deduced_simd<int32_t, 2> vindex, const double *base_addr) {
#if BATMAT_WITH_GSI_HPC_SIMD
    auto vindex128 = cat(vindex, decltype(vindex){}); // high 128 bits to zero
#else
    auto vindex128 = vindex;
#endif
    return datapar::deduced_simd<double, 2>{_mm_mmask_i32gather_pd(
        static_cast<__m128d>(src), mask, static_cast<__m128i>(vindex128), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 16>
gather(datapar::deduced_simd<float, 16> src, __mmask16 mask,
       datapar::deduced_simd<int32_t, 16> vindex, const float *base_addr) {
    return datapar::deduced_simd<float, 16>{_mm512_mask_i32gather_ps(
        static_cast<__m512>(src), mask, static_cast<__m512i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 8>
gather(datapar::deduced_simd<float, 8> src, __mmask8 mask, datapar::deduced_simd<int32_t, 8> vindex,
       const float *base_addr) {
    return datapar::deduced_simd<float, 8>{_mm256_mmask_i32gather_ps(
        static_cast<__m256>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 4>
gather(datapar::deduced_simd<float, 4> src, __mmask8 mask, datapar::deduced_simd<int32_t, 4> vindex,
       const float *base_addr) {
    return datapar::deduced_simd<float, 4>{_mm_mmask_i32gather_ps(
        static_cast<__m128>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 16>
gather(datapar::deduced_simd<float, 16> src, __mmask16 mask,
       datapar::deduced_simd<int64_t, 16> vindex, const float *base_addr) {
    auto src512 = static_cast<__m512>(src);
    // split src into low and high 8-float parts
    __m256 src_lo = _mm512_castps512_ps256(src512), src_hi = _mm512_extractf32x8_ps(src512, 1);
    // split mask into low and high 8-bit parts
    auto mask_lo = static_cast<__mmask8>(mask), mask_hi = static_cast<__mmask8>(mask >> 8);
    // split 16 x 64-bit indices into low and high 8 x 64-bit parts
#if BATMAT_WITH_GSI_HPC_SIMD
    auto [idx_lo, idx_hi] = chunk<datapar::deduced_simd<int64_t, 8>>(vindex);
#else
    auto [idx_lo, idx_hi] = split<8, 8>(vindex);
#endif
    auto idx_lo_512 = static_cast<__m512i>(idx_lo), idx_hi_512 = static_cast<__m512i>(idx_hi);
    // two gathers of 8 floats each using 8 x 64-bit indices
    __m256 g_lo = _mm512_mask_i64gather_ps(src_lo, mask_lo, idx_lo_512, base_addr, Scale);
    __m256 g_hi = _mm512_mask_i64gather_ps(src_hi, mask_hi, idx_hi_512, base_addr, Scale);
    // combine: [g_lo0..7, g_hi0..7]
    return datapar::deduced_simd<float, 16>{
        _mm512_insertf32x8(_mm512_castps256_ps512(g_lo), g_hi, 1)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 8>
gather(datapar::deduced_simd<float, 8> src, __mmask8 mask, datapar::deduced_simd<int64_t, 8> vindex,
       const float *base_addr) {
    return datapar::deduced_simd<float, 8>{_mm512_mask_i64gather_ps(
        static_cast<__m256>(src), mask, static_cast<__m512i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 4>
gather(datapar::deduced_simd<float, 4> src, __mmask8 mask, datapar::deduced_simd<int64_t, 4> vindex,
       const float *base_addr) {
    return datapar::deduced_simd<float, 4>{_mm256_mmask_i64gather_ps(
        static_cast<__m128>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

} // namespace batmat::ops::detail
