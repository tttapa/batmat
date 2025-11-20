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
    return datapar::deduced_simd<double, 2>{_mm_mmask_i32gather_pd(
        static_cast<__m128d>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
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
