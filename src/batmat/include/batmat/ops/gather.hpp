#pragma once

#include <experimental/simd>
#include <concepts>

namespace batmat::ops {

namespace stdx = std::experimental;

template <class T, size_t N>
using deduce_simd = stdx::simd<T, stdx::simd_abi::deduce_t<T, N>>;

namespace detail {

template <class T, class Abi>
[[gnu::always_inline]] inline stdx::simd_mask<T, Abi> compare_ge_0(stdx::simd<T, Abi> x) {
    return x >= stdx::simd<T, Abi>{};
}

template <int Scale, class T, class AbiT, class I, class AbiI>
[[gnu::always_inline]] inline stdx::simd<T, AbiT>
gather(stdx::simd<T, AbiT> src, stdx::simd_mask<I, AbiI> mask, stdx::simd<I, AbiI> vindex,
       const T *base_addr) {
    return stdx::simd<T, AbiT>{[=](auto i) { return mask[i] ? base_addr[vindex[i]] : src[i]; }};
}

#if defined(__AVX512F__)

[[gnu::always_inline]] inline __mmask8 compare_ge_0(deduce_simd<int32_t, 4> x) {
    return _mm_cmp_epi32_mask(static_cast<__m128i>(x), static_cast<__m128i>(decltype(x){}),
                              _MM_CMPINT_NLT);
}

[[gnu::always_inline]] inline __mmask8 compare_ge_0(deduce_simd<int32_t, 8> x) {
    return _mm256_cmp_epi32_mask(static_cast<__m256i>(x), static_cast<__m256i>(decltype(x){}),
                                 _MM_CMPINT_NLT);
}

[[gnu::always_inline]] inline __mmask16 compare_ge_0(deduce_simd<int32_t, 16> x) {
    return _mm512_cmp_epi32_mask(static_cast<__m512i>(x), static_cast<__m512i>(decltype(x){}),
                                 _MM_CMPINT_NLT);
}

[[gnu::always_inline]] inline __mmask8 compare_ge_0(deduce_simd<int64_t, 2> x) {
    return _mm_cmp_epi64_mask(static_cast<__m128i>(x), static_cast<__m128i>(decltype(x){}),
                              _MM_CMPINT_NLT);
}
[[gnu::always_inline]] inline __mmask8 compare_ge_0(deduce_simd<int64_t, 4> x) {
    return _mm256_cmp_epi64_mask(static_cast<__m256i>(x), static_cast<__m256i>(decltype(x){}),
                                 _MM_CMPINT_NLT);
}

[[gnu::always_inline]] inline __mmask8 compare_ge_0(deduce_simd<int64_t, 8> x) {
    return _mm512_cmp_epi64_mask(static_cast<__m512i>(x), static_cast<__m512i>(decltype(x){}),
                                 _MM_CMPINT_NLT);
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline deduce_simd<double, 8>
gather(deduce_simd<double, 8> src, __mmask8 mask, deduce_simd<int64_t, 8> vindex,
       const void *base_addr) {
    return deduce_simd<double, 8>{_mm512_mask_i64gather_pd(
        static_cast<__m512d>(src), mask, static_cast<__m512i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline deduce_simd<double, 4>
gather(deduce_simd<double, 4> src, __mmask8 mask, deduce_simd<int64_t, 4> vindex,
       const void *base_addr) {
    return deduce_simd<double, 4>{_mm256_mmask_i64gather_pd(
        static_cast<__m256d>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline deduce_simd<double, 2>
gather(deduce_simd<double, 2> src, __mmask8 mask, deduce_simd<int64_t, 2> vindex,
       const void *base_addr) {
    return deduce_simd<double, 2>{_mm_mmask_i64gather_pd(
        static_cast<__m128d>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline deduce_simd<double, 8>
gather(deduce_simd<double, 8> src, __mmask8 mask, deduce_simd<int32_t, 8> vindex,
       const void *base_addr) {
    return deduce_simd<double, 8>{_mm512_mask_i32gather_pd(
        static_cast<__m512d>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline deduce_simd<double, 4>
gather(deduce_simd<double, 4> src, __mmask8 mask, deduce_simd<int32_t, 4> vindex,
       const void *base_addr) {
    return deduce_simd<double, 4>{_mm256_mmask_i32gather_pd(
        static_cast<__m256d>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline deduce_simd<double, 2>
gather(deduce_simd<double, 2> src, __mmask8 mask, deduce_simd<int32_t, 2> vindex,
       const void *base_addr) {
    return deduce_simd<double, 2>{_mm_mmask_i32gather_pd(
        static_cast<__m128d>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline deduce_simd<float, 16>
gather(deduce_simd<float, 16> src, __mmask16 mask, deduce_simd<int32_t, 16> vindex,
       const void *base_addr) {
    return deduce_simd<float, 16>{_mm512_mask_i32gather_ps(
        static_cast<__m512>(src), mask, static_cast<__m512i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline deduce_simd<float, 8> gather(deduce_simd<float, 8> src, __mmask8 mask,
                                                           deduce_simd<int32_t, 8> vindex,
                                                           const void *base_addr) {
    return deduce_simd<float, 8>{_mm256_mmask_i32gather_ps(
        static_cast<__m256>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline deduce_simd<float, 4> gather(deduce_simd<float, 4> src, __mmask8 mask,
                                                           deduce_simd<int32_t, 4> vindex,
                                                           const void *base_addr) {
    return deduce_simd<float, 4>{_mm_mmask_i32gather_ps(
        static_cast<__m128>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline deduce_simd<float, 8> gather(deduce_simd<float, 8> src, __mmask8 mask,
                                                           deduce_simd<int64_t, 8> vindex,
                                                           const void *base_addr) {
    return deduce_simd<float, 8>{_mm512_mask_i64gather_ps(
        static_cast<__m256>(src), mask, static_cast<__m512i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline deduce_simd<float, 4> gather(deduce_simd<float, 4> src, __mmask8 mask,
                                                           deduce_simd<int64_t, 4> vindex,
                                                           const void *base_addr) {
    return deduce_simd<float, 4>{_mm256_mmask_i64gather_ps(
        static_cast<__m128>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

#elif defined(__AVX2__)

[[gnu::always_inline]] inline __m256i compare_ge_0(deduce_simd<int32_t, 8> x) {
    __m256i zero = _mm256_setzero_si256();
    return _mm256_or_si256(_mm256_cmpgt_epi32(static_cast<__m256i>(x), zero),
                           _mm256_cmpeq_epi32(static_cast<__m256i>(x), zero));
}

[[gnu::always_inline]] inline __m128i compare_ge_0(deduce_simd<int32_t, 4> x) {
    __m128i zero = _mm_setzero_si128();
    return _mm_or_si128(_mm_cmpgt_epi32(static_cast<__m128i>(x), zero),
                        _mm_cmpeq_epi32(static_cast<__m128i>(x), zero));
}

[[gnu::always_inline]] inline __m128i compare_ge_0(deduce_simd<int32_t, 2> x) {
    __m128i zero = _mm_setzero_si128();
    return _mm_or_si128(_mm_cmpgt_epi64(static_cast<__m128i>(x), zero),
                        _mm_cmpeq_epi64(static_cast<__m128i>(x), zero));
}

[[gnu::always_inline]] inline __m256i compare_ge_0(deduce_simd<int64_t, 4> x) {
    __m256i zero = _mm256_setzero_si256();
    return _mm256_or_si256(_mm256_cmpgt_epi64(static_cast<__m256i>(x), zero),
                           _mm256_cmpeq_epi64(static_cast<__m256i>(x), zero));
}

[[gnu::always_inline]] inline __m128i compare_ge_0(deduce_simd<int64_t, 2> x) {
    __m128i zero = _mm_setzero_si128();
    return _mm_or_si128(_mm_cmpgt_epi64(static_cast<__m128i>(x), zero),
                        _mm_cmpeq_epi64(static_cast<__m128i>(x), zero));
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline deduce_simd<float, 8> gather(deduce_simd<float, 8> src, __m256i mask,
                                                           deduce_simd<int32_t, 8> vindex,
                                                           const float *base_addr) {
    __m256i active     = mask;
    __m256i safe_index = _mm256_and_si256(static_cast<__m256i>(vindex), active);
    __m256 gathered    = _mm256_i32gather_ps(base_addr, safe_index, Scale);
    __m256 result =
        _mm256_blendv_ps(static_cast<__m256>(src), gathered, _mm256_castsi256_ps(active));
    return deduce_simd<float, 8>{result};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline deduce_simd<double, 4>
gather(deduce_simd<double, 4> src, __m256i mask, deduce_simd<int64_t, 4> vindex,
       const double *base_addr) {
    __m256i active     = mask;
    __m256i safe_index = _mm256_and_si256(static_cast<__m256i>(vindex), active);
    __m256d gathered   = _mm256_i64gather_pd(base_addr, safe_index, Scale);
    __m256d result =
        _mm256_blendv_pd(static_cast<__m256d>(src), gathered, _mm256_castsi256_pd(active));
    return deduce_simd<double, 4>{result};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline deduce_simd<double, 4>
gather(deduce_simd<double, 4> src, __m128i mask, deduce_simd<int32_t, 4> vindex,
       const double *base_addr) {
    __m128i active     = mask;
    __m128i safe_index = _mm_and_si128(static_cast<__m128i>(vindex), active);
    __m256d gathered   = _mm256_i32gather_pd(base_addr, safe_index, Scale);
    __m256d result     = _mm256_blendv_pd(static_cast<__m256d>(src), gathered,
                                          _mm256_castsi256_pd(_mm256_cvtepi32_epi64(active)));
    return deduce_simd<double, 4>{result};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline deduce_simd<float, 4> gather(deduce_simd<float, 4> src, __m128i mask,
                                                           deduce_simd<int32_t, 4> vindex,
                                                           const float *base_addr) {
    __m128i active     = mask;
    __m128i safe_index = _mm_and_si128(static_cast<__m128i>(vindex), active);
    __m128 gathered    = _mm_i32gather_ps(base_addr, safe_index, Scale);
    __m128 result = _mm_blendv_ps(static_cast<__m128>(src), gathered, _mm_castsi128_ps(active));
    return deduce_simd<float, 4>{result};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline deduce_simd<double, 2>
gather(deduce_simd<double, 2> src, __m128i mask, deduce_simd<int64_t, 2> vindex,
       const double *base_addr) {
    __m128i active     = mask;
    __m128i safe_index = _mm_and_si128(static_cast<__m128i>(vindex), active);
    __m128d gathered   = _mm_i64gather_pd(base_addr, safe_index, Scale);
    __m128d result = _mm_blendv_pd(static_cast<__m128d>(src), gathered, _mm_castsi128_pd(active));
    return deduce_simd<double, 2>{result};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline deduce_simd<double, 2>
gather(deduce_simd<double, 2> src, __m128i mask, deduce_simd<int32_t, 2> vindex,
       const double *base_addr) {
    __m128i active     = mask;
    __m128i safe_index = _mm_and_si128(static_cast<__m128i>(vindex), active);
    __m128d gathered   = _mm_i32gather_pd(base_addr, safe_index, Scale);
    __m128d result     = _mm_blendv_pd(static_cast<__m128d>(src), gathered,
                                       _mm_castsi128_pd(_mm_cvtepi32_epi64(active)));
    return deduce_simd<double, 2>{result};
}

#endif

template <class T1, class T2>
concept same_size_but_different_ints =
    std::integral<T1> && std::integral<T2> && sizeof(T1) == sizeof(T2) && !std::same_as<T1, T2>;

} // namespace detail

template <class T, size_t N, class I>
deduce_simd<T, N> gather(const T *p, deduce_simd<I, N> idx) {
    if constexpr (detail::same_size_but_different_ints<I, int32_t>) {
        const auto idx_ = simd_cast<deduce_simd<int32_t, N>>(idx);
        return detail::gather(deduce_simd<T, N>{}, detail::compare_ge_0(idx_), idx_, p);
    } else if constexpr (detail::same_size_but_different_ints<I, int64_t>) {
        const auto idx_ = simd_cast<deduce_simd<int64_t, N>>(idx);
        return detail::gather(deduce_simd<T, N>{}, detail::compare_ge_0(idx_), idx_, p);
    } else {
        return detail::gather(deduce_simd<T, N>{}, detail::compare_ge_0(idx), idx, p);
    }
}

} // namespace batmat::ops
