#pragma once

#include <batmat/ops/mask.hpp>
#include <batmat/simd.hpp>
#include <immintrin.h>

namespace batmat::ops::detail {

template <>
struct mask_type<float, datapar::deduced_abi<float, 8>> {
    using type = __m256;
};
template <>
struct mask_type<float, datapar::deduced_abi<float, 4>> {
    using type = __m128;
};
template <>
struct mask_type<double, datapar::deduced_abi<double, 4>> {
    using type = __m256d;
};
template <>
struct mask_type<double, datapar::deduced_abi<double, 2>> {
    using type = __m128d;
};

#if 0 // TODO: cannot cast std::experimental::simd_mask to intrinsic types
template <>
[[gnu::always_inline]] inline __m256
convert_mask<float, datapar::deduced_abi<float, 8>,
             typename datapar::deduced_simd<int32_t, 8>::mask_type>(
    typename datapar::deduced_simd<int32_t, 8>::mask_type mask) {
    return _mm256_castsi256_ps(static_cast<__m256i>(mask));
}

template <>
[[gnu::always_inline]] inline __m256d
convert_mask<double, datapar::deduced_abi<double, 4>,
             typename datapar::deduced_simd<int64_t, 4>::mask_type>(
    typename datapar::deduced_simd<int64_t, 4>::mask_type mask) {
    return _mm256_castsi256_pd(static_cast<__m256i>(mask));
}

template <>
[[gnu::always_inline]] inline __m256d
convert_mask<double, datapar::deduced_abi<double, 4>,
             typename datapar::deduced_simd<int32_t, 4>::mask_type>(
    typename datapar::deduced_simd<int32_t, 4>::mask_type mask) {
    return _mm256_castsi256_pd(_mm256_cvtepi32_epi64(static_cast<__m128i>(mask)));
}

template <>
[[gnu::always_inline]] inline __m128
convert_mask<float, datapar::deduced_abi<float, 4>,
             typename datapar::deduced_simd<int32_t, 4>::mask_type>(
    typename datapar::deduced_simd<int32_t, 4>::mask_type mask) {
    return _mm_castsi128_ps(static_cast<__m128i>(mask));
}

template <>
[[gnu::always_inline]] inline __m128d
convert_mask<double, datapar::deduced_abi<double, 2>,
             typename datapar::deduced_simd<int64_t, 2>::mask_type>(
    typename datapar::deduced_simd<int64_t, 2>::mask_type mask) {
    return _mm_castsi128_pd(static_cast<__m128i>(mask));
}
template <>
[[gnu::always_inline]] inline __m128d
convert_mask<double, datapar::deduced_abi<double, 2>,
             typename datapar::deduced_simd<int32_t, 2>::mask_type>(
    typename datapar::deduced_simd<int32_t, 2>::mask_type mask) {
    return _mm_castsi128_pd(_mm_cvtepi32_epi64(static_cast<__m128i>(mask)));
}

template <>
[[gnu::always_inline]] inline __m256
convert_mask<float, datapar::deduced_abi<float, 8>,
             typename datapar::deduced_simd<float, 8>::mask_type>(
    typename datapar::deduced_simd<float, 8>::mask_type mask) {
    return static_cast<__m256>(mask);
}

template <>
[[gnu::always_inline]] inline __m256d
convert_mask<double, datapar::deduced_abi<double, 4>,
             typename datapar::deduced_simd<double, 4>::mask_type>(
    typename datapar::deduced_simd<double, 4>::mask_type mask) {
    return static_cast<__m256d>(mask);
}

template <>
[[gnu::always_inline]] inline __m128
convert_mask<float, datapar::deduced_abi<float, 4>,
             typename datapar::deduced_simd<float, 4>::mask_type>(
    typename datapar::deduced_simd<float, 4>::mask_type mask) {
    return static_cast<__m128>(mask);
}

template <>
[[gnu::always_inline]] inline __m128d
convert_mask<double, datapar::deduced_abi<double, 2>,
             typename datapar::deduced_simd<double, 2>::mask_type>(
    typename datapar::deduced_simd<double, 2>::mask_type mask) {
    return static_cast<__m128d>(mask);
}
#endif

template <>
[[gnu::always_inline]] inline __m256
convert_mask<float, datapar::deduced_abi<float, 8>, typename datapar::deduced_simd<int32_t, 8>>(
    typename datapar::deduced_simd<int32_t, 8> mask) {
    return _mm256_castsi256_ps(
        _mm256_xor_si256(_mm256_cmpeq_epi32(static_cast<__m256i>(mask), _mm256_setzero_si256()),
                         _mm256_set1_epi32(-1)));
}

template <>
[[gnu::always_inline]] inline __m256d
convert_mask<double, datapar::deduced_abi<double, 4>, typename datapar::deduced_simd<int64_t, 4>>(
    typename datapar::deduced_simd<int64_t, 4> mask) {
    return _mm256_castsi256_pd(
        _mm256_xor_si256(_mm256_cmpeq_epi64(static_cast<__m256i>(mask), _mm256_setzero_si256()),
                         _mm256_set1_epi64x(-1)));
}

template <>
[[gnu::always_inline]] inline __m256d
convert_mask<double, datapar::deduced_abi<double, 4>, typename datapar::deduced_simd<int32_t, 4>>(
    typename datapar::deduced_simd<int32_t, 4> mask) {
    return _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_xor_si128(
        _mm_cmpeq_epi32(static_cast<__m128i>(mask), _mm_setzero_si128()), _mm_set1_epi32(-1))));
}

template <>
[[gnu::always_inline]] inline __m128
convert_mask<float, datapar::deduced_abi<float, 4>, typename datapar::deduced_simd<int32_t, 4>>(
    typename datapar::deduced_simd<int32_t, 4> mask) {
    return _mm_castsi128_ps(_mm_xor_si128(
        _mm_cmpeq_epi32(static_cast<__m128i>(mask), _mm_setzero_si128()), _mm_set1_epi32(-1)));
}

template <>
[[gnu::always_inline]] inline __m128d
convert_mask<double, datapar::deduced_abi<double, 2>, typename datapar::deduced_simd<int64_t, 2>>(
    typename datapar::deduced_simd<int64_t, 2> mask) {
    return _mm_castsi128_pd(_mm_xor_si128(
        _mm_cmpeq_epi64(static_cast<__m128i>(mask), _mm_setzero_si128()), _mm_set1_epi64x(-1)));
}
template <>
[[gnu::always_inline]] inline __m128d
convert_mask<double, datapar::deduced_abi<double, 2>, typename datapar::deduced_simd<int32_t, 2>>(
    typename datapar::deduced_simd<int32_t, 2> mask) {
    return _mm_castsi128_pd(_mm_cvtepi32_epi64(_mm_xor_si128(
        _mm_cmpeq_epi32(static_cast<__m128i>(mask), _mm_setzero_si128()), _mm_set1_epi32(-1))));
}

template <>
[[gnu::always_inline]] inline __m256
convert_mask<float, datapar::deduced_abi<float, 8>, typename datapar::deduced_simd<float, 8>>(
    typename datapar::deduced_simd<float, 8> mask) {
    return _mm256_cmp_ps(static_cast<__m256>(mask), _mm256_setzero_ps(), _CMP_NEQ_OQ);
}

template <>
[[gnu::always_inline]] inline __m256d
convert_mask<double, datapar::deduced_abi<double, 4>, typename datapar::deduced_simd<double, 4>>(
    typename datapar::deduced_simd<double, 4> mask) {
    return _mm256_cmp_pd(static_cast<__m256d>(mask), _mm256_setzero_pd(), _CMP_NEQ_OQ);
}

template <>
[[gnu::always_inline]] inline __m128
convert_mask<float, datapar::deduced_abi<float, 4>, typename datapar::deduced_simd<float, 4>>(
    typename datapar::deduced_simd<float, 4> mask) {
    return _mm_cmp_ps(static_cast<__m128>(mask), _mm_setzero_ps(), _CMP_NEQ_OQ);
}

template <>
[[gnu::always_inline]] inline __m128d
convert_mask<double, datapar::deduced_abi<double, 2>, typename datapar::deduced_simd<double, 2>>(
    typename datapar::deduced_simd<double, 2> mask) {
    return _mm_cmp_pd(static_cast<__m128d>(mask), _mm_setzero_pd(), _CMP_NEQ_OQ);
}

[[gnu::always_inline]] inline __m256i compare_ge_0(datapar::deduced_simd<int32_t, 8> x) {
    __m256i zero = _mm256_setzero_si256();
    return _mm256_or_si256(_mm256_cmpgt_epi32(static_cast<__m256i>(x), zero),
                           _mm256_cmpeq_epi32(static_cast<__m256i>(x), zero));
}

[[gnu::always_inline]] inline __m128i compare_ge_0(datapar::deduced_simd<int32_t, 4> x) {
    __m128i zero = _mm_setzero_si128();
    return _mm_or_si128(_mm_cmpgt_epi32(static_cast<__m128i>(x), zero),
                        _mm_cmpeq_epi32(static_cast<__m128i>(x), zero));
}

#if !BATMAT_WITH_GSI_HPC_SIMD // TODO
[[gnu::always_inline]] inline __m128i compare_ge_0(datapar::deduced_simd<int32_t, 2> x) {
    __m128i zero = _mm_setzero_si128();
    return _mm_or_si128(_mm_cmpgt_epi64(static_cast<__m128i>(x), zero),
                        _mm_cmpeq_epi64(static_cast<__m128i>(x), zero));
}
#endif

[[gnu::always_inline]] inline __m256i compare_ge_0(datapar::deduced_simd<int64_t, 4> x) {
    __m256i zero = _mm256_setzero_si256();
    return _mm256_or_si256(_mm256_cmpgt_epi64(static_cast<__m256i>(x), zero),
                           _mm256_cmpeq_epi64(static_cast<__m256i>(x), zero));
}

[[gnu::always_inline]] inline __m128i compare_ge_0(datapar::deduced_simd<int64_t, 2> x) {
    __m128i zero = _mm_setzero_si128();
    return _mm_or_si128(_mm_cmpgt_epi64(static_cast<__m128i>(x), zero),
                        _mm_cmpeq_epi64(static_cast<__m128i>(x), zero));
}

} // namespace batmat::ops::detail
