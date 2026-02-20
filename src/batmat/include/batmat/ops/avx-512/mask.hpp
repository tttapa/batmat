#pragma once

#include <batmat/ops/mask.hpp>
#include <batmat/simd.hpp>
#include <immintrin.h>

namespace batmat::ops::detail {

template <>
struct mask_type<float, datapar::deduced_abi<float, 16>> {
    using type = __mmask16;
};
template <>
struct mask_type<float, datapar::deduced_abi<float, 8>> {
    using type = __mmask8;
};
template <>
struct mask_type<float, datapar::deduced_abi<float, 4>> {
    using type = __mmask8;
};
template <>
struct mask_type<double, datapar::deduced_abi<double, 8>> {
    using type = __mmask8;
};
template <>
struct mask_type<double, datapar::deduced_abi<double, 4>> {
    using type = __mmask8;
};
template <>
struct mask_type<double, datapar::deduced_abi<double, 2>> {
    using type = __mmask8;
};

template <>
[[gnu::always_inline]] inline __mmask16
convert_mask<float, datapar::deduced_abi<float, 16>, typename datapar::deduced_simd<int64_t, 16>>(
    typename datapar::deduced_simd<int64_t, 16> mask) {
#if BATMAT_WITH_GSI_HPC_SIMD
    auto [lo, hi] = chunk<datapar::deduced_simd<int64_t, 8>>(mask);
#else
    auto [lo, hi] = split<8, 8>(mask);
#endif
    __mmask16 mask_lo = _mm512_cmpneq_epi64_mask(static_cast<__m512i>(lo), _mm512_setzero_si512()),
              mask_hi = _mm512_cmpneq_epi64_mask(static_cast<__m512i>(hi), _mm512_setzero_si512());
    return static_cast<__mmask16>(mask_lo | (mask_hi << 8));
}

template <>
[[gnu::always_inline]] inline __mmask16
convert_mask<float, datapar::deduced_abi<float, 16>, typename datapar::deduced_simd<int32_t, 16>>(
    typename datapar::deduced_simd<int32_t, 16> mask) {
    return _mm512_cmpneq_epi32_mask(static_cast<__m512i>(mask), _mm512_setzero_si512());
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<float, datapar::deduced_abi<float, 8>, typename datapar::deduced_simd<int64_t, 8>>(
    typename datapar::deduced_simd<int64_t, 8> mask) {
    return _mm512_cmpneq_epi64_mask(static_cast<__m512i>(mask), _mm512_setzero_si512());
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<float, datapar::deduced_abi<float, 8>, typename datapar::deduced_simd<int32_t, 8>>(
    typename datapar::deduced_simd<int32_t, 8> mask) {
    return _mm256_cmpneq_epi32_mask(static_cast<__m256i>(mask), _mm256_setzero_si256());
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<double, datapar::deduced_abi<double, 8>, typename datapar::deduced_simd<int64_t, 8>>(
    typename datapar::deduced_simd<int64_t, 8> mask) {
    return _mm512_cmpneq_epi64_mask(static_cast<__m512i>(mask), _mm512_setzero_si512());
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<double, datapar::deduced_abi<double, 8>, typename datapar::deduced_simd<int32_t, 8>>(
    typename datapar::deduced_simd<int32_t, 8> mask) {
    return _mm256_cmpneq_epi32_mask(static_cast<__m256i>(mask), _mm256_setzero_si256());
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<double, datapar::deduced_abi<double, 4>, typename datapar::deduced_simd<int64_t, 4>>(
    typename datapar::deduced_simd<int64_t, 4> mask) {
    return _mm256_cmpneq_epi64_mask(static_cast<__m256i>(mask), _mm256_setzero_si256());
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<double, datapar::deduced_abi<double, 4>, typename datapar::deduced_simd<int32_t, 4>>(
    typename datapar::deduced_simd<int32_t, 4> mask) {
    return _mm_cmpneq_epi32_mask(static_cast<__m128i>(mask), _mm_setzero_si128());
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<float, datapar::deduced_abi<float, 4>, typename datapar::deduced_simd<int32_t, 4>>(
    typename datapar::deduced_simd<int32_t, 4> mask) {
    return _mm_cmpneq_epi32_mask(static_cast<__m128i>(mask), _mm_setzero_si128());
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<float, datapar::deduced_abi<float, 4>, typename datapar::deduced_simd<int64_t, 4>>(
    typename datapar::deduced_simd<int64_t, 4> mask) {
    return _mm256_cmpneq_epi64_mask(static_cast<__m256i>(mask), _mm256_setzero_si256());
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<double, datapar::deduced_abi<double, 2>, typename datapar::deduced_simd<int64_t, 2>>(
    typename datapar::deduced_simd<int64_t, 2> mask) {
    return _mm_cmpneq_epi64_mask(static_cast<__m128i>(mask), _mm_setzero_si128());
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<double, datapar::deduced_abi<double, 2>, typename datapar::deduced_simd<int32_t, 2>>(
    typename datapar::deduced_simd<int32_t, 2> mask) {
#if BATMAT_WITH_GSI_HPC_SIMD
    // TODO: cannot cast 64-bit std::datapar::simd to __m128i, so we need to extend manually
    auto w = static_cast<__m128i>(cat(mask, decltype(mask){}));
#else
    auto w = static_cast<__m128i>(mask);
#endif
    return _mm_cmpneq_epi32_mask(w, _mm_setzero_si128());
}

template <>
[[gnu::always_inline]] inline __mmask16
convert_mask<float, datapar::deduced_abi<float, 16>, typename datapar::deduced_simd<float, 16>>(
    typename datapar::deduced_simd<float, 16> mask) {
    return _mm512_cmp_ps_mask(static_cast<__m512>(mask), _mm512_setzero_ps(), _CMP_NEQ_OQ);
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<float, datapar::deduced_abi<float, 8>, typename datapar::deduced_simd<float, 8>>(
    typename datapar::deduced_simd<float, 8> mask) {
    return _mm256_cmp_ps_mask(static_cast<__m256>(mask), _mm256_setzero_ps(), _CMP_NEQ_OQ);
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<double, datapar::deduced_abi<double, 8>, typename datapar::deduced_simd<double, 8>>(
    typename datapar::deduced_simd<double, 8> mask) {
    return _mm512_cmp_pd_mask(static_cast<__m512d>(mask), _mm512_setzero_pd(), _CMP_NEQ_OQ);
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<double, datapar::deduced_abi<double, 4>, typename datapar::deduced_simd<double, 4>>(
    typename datapar::deduced_simd<double, 4> mask) {
    return _mm256_cmp_pd_mask(static_cast<__m256d>(mask), _mm256_setzero_pd(), _CMP_NEQ_OQ);
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<float, datapar::deduced_abi<float, 4>, typename datapar::deduced_simd<float, 4>>(
    typename datapar::deduced_simd<float, 4> mask) {
    return _mm_cmp_ps_mask(static_cast<__m128>(mask), _mm_setzero_ps(), _CMP_NEQ_OQ);
}

template <>
[[gnu::always_inline]] inline __mmask8
convert_mask<double, datapar::deduced_abi<double, 2>, typename datapar::deduced_simd<double, 2>>(
    typename datapar::deduced_simd<double, 2> mask) {
    return _mm_cmp_pd_mask(static_cast<__m128d>(mask), _mm_setzero_pd(), _CMP_NEQ_OQ);
}

[[gnu::always_inline]] inline __mmask8 compare_ge_0(datapar::deduced_simd<int32_t, 4> x) {
    return _mm_cmp_epi32_mask(static_cast<__m128i>(x), _mm_setzero_si128(), _MM_CMPINT_NLT);
}

[[gnu::always_inline]] inline __mmask8 compare_ge_0(datapar::deduced_simd<int32_t, 2> x) {
#if BATMAT_WITH_GSI_HPC_SIMD
    // TODO: cannot cast 64-bit std::datapar::simd to __m128i, so we need to extend manually
    auto w = static_cast<__m128i>(cat(x, decltype(x){-1, -1}));
#else
    auto w = static_cast<__m128i>(x);
#endif
    return _mm_cmp_epi32_mask(w, _mm_setzero_si128(), _MM_CMPINT_NLT);
}

[[gnu::always_inline]] inline __mmask8 compare_ge_0(datapar::deduced_simd<int32_t, 8> x) {
    return _mm256_cmp_epi32_mask(static_cast<__m256i>(x), _mm256_setzero_si256(), _MM_CMPINT_NLT);
}

[[gnu::always_inline]] inline __mmask16 compare_ge_0(datapar::deduced_simd<int32_t, 16> x) {
    return _mm512_cmp_epi32_mask(static_cast<__m512i>(x), _mm512_setzero_si512(), _MM_CMPINT_NLT);
}

[[gnu::always_inline]] inline __mmask8 compare_ge_0(datapar::deduced_simd<int64_t, 2> x) {
    return _mm_cmp_epi64_mask(static_cast<__m128i>(x), _mm_setzero_si128(), _MM_CMPINT_NLT);
}
[[gnu::always_inline]] inline __mmask8 compare_ge_0(datapar::deduced_simd<int64_t, 4> x) {
    return _mm256_cmp_epi64_mask(static_cast<__m256i>(x), _mm256_setzero_si256(), _MM_CMPINT_NLT);
}

[[gnu::always_inline]] inline __mmask8 compare_ge_0(datapar::deduced_simd<int64_t, 8> x) {
    return _mm512_cmp_epi64_mask(static_cast<__m512i>(x), _mm512_setzero_si512(), _MM_CMPINT_NLT);
}

} // namespace batmat::ops::detail
