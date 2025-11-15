#pragma once

#include <batmat/simd.hpp>
#include <immintrin.h>
#include <concepts>

namespace batmat::ops {

namespace detail {

template <class T, class AbiT, class I, class AbiI>
[[gnu::always_inline]] inline datapar::simd<T, AbiT>
gather(datapar::simd<T, AbiT> src, typename datapar::simd<I, AbiI>::mask_type mask,
       datapar::simd<I, AbiI> vindex, const T *base_addr) {
    return datapar::simd<T, AbiT>{[=](auto i) { return mask[i] ? base_addr[vindex[i]] : src[i]; }};
}

template <class T, class I, class AbiI,
          class AbiT = datapar::deduced_abi<T, datapar::simd_size<I, AbiI>::value>>
[[gnu::always_inline]] inline datapar::simd<T, AbiT> gather(datapar::simd<I, AbiI> vindex,
                                                            const T *base_addr) {
    return datapar::simd<T, AbiT>{[=](auto i) { return base_addr[vindex[i]]; }};
}

#if defined(__AVX512F__)

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 8>
gather(datapar::deduced_simd<double, 8> src, __mmask8 mask,
       datapar::deduced_simd<int64_t, 8> vindex, const void *base_addr) {
    return datapar::deduced_simd<double, 8>{_mm512_mask_i64gather_pd(
        static_cast<__m512d>(src), mask, static_cast<__m512i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 4>
gather(datapar::deduced_simd<double, 4> src, __mmask8 mask,
       datapar::deduced_simd<int64_t, 4> vindex, const void *base_addr) {
    return datapar::deduced_simd<double, 4>{_mm256_mmask_i64gather_pd(
        static_cast<__m256d>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 2>
gather(datapar::deduced_simd<double, 2> src, __mmask8 mask,
       datapar::deduced_simd<int64_t, 2> vindex, const void *base_addr) {
    return datapar::deduced_simd<double, 2>{_mm_mmask_i64gather_pd(
        static_cast<__m128d>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 8>
gather(datapar::deduced_simd<double, 8> src, __mmask8 mask,
       datapar::deduced_simd<int32_t, 8> vindex, const void *base_addr) {
    return datapar::deduced_simd<double, 8>{_mm512_mask_i32gather_pd(
        static_cast<__m512d>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 4>
gather(datapar::deduced_simd<double, 4> src, __mmask8 mask,
       datapar::deduced_simd<int32_t, 4> vindex, const void *base_addr) {
    return datapar::deduced_simd<double, 4>{_mm256_mmask_i32gather_pd(
        static_cast<__m256d>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 2>
gather(datapar::deduced_simd<double, 2> src, __mmask8 mask,
       datapar::deduced_simd<int32_t, 2> vindex, const void *base_addr) {
    return datapar::deduced_simd<double, 2>{_mm_mmask_i32gather_pd(
        static_cast<__m128d>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 16>
gather(datapar::deduced_simd<float, 16> src, __mmask16 mask,
       datapar::deduced_simd<int32_t, 16> vindex, const void *base_addr) {
    return datapar::deduced_simd<float, 16>{_mm512_mask_i32gather_ps(
        static_cast<__m512>(src), mask, static_cast<__m512i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 8>
gather(datapar::deduced_simd<float, 8> src, __mmask8 mask, datapar::deduced_simd<int32_t, 8> vindex,
       const void *base_addr) {
    return datapar::deduced_simd<float, 8>{_mm256_mmask_i32gather_ps(
        static_cast<__m256>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 4>
gather(datapar::deduced_simd<float, 4> src, __mmask8 mask, datapar::deduced_simd<int32_t, 4> vindex,
       const void *base_addr) {
    return datapar::deduced_simd<float, 4>{_mm_mmask_i32gather_ps(
        static_cast<__m128>(src), mask, static_cast<__m128i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 8>
gather(datapar::deduced_simd<float, 8> src, __mmask8 mask, datapar::deduced_simd<int64_t, 8> vindex,
       const void *base_addr) {
    return datapar::deduced_simd<float, 8>{_mm512_mask_i64gather_ps(
        static_cast<__m256>(src), mask, static_cast<__m512i>(vindex), base_addr, Scale)};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 4>
gather(datapar::deduced_simd<float, 4> src, __mmask8 mask, datapar::deduced_simd<int64_t, 4> vindex,
       const void *base_addr) {
    return datapar::deduced_simd<float, 4>{_mm256_mmask_i64gather_ps(
        static_cast<__m128>(src), mask, static_cast<__m256i>(vindex), base_addr, Scale)};
}

#elif defined(__AVX2__)

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 8>
gather(datapar::deduced_simd<float, 8> src,
       typename datapar::deduced_simd<float, 8>::mask_type mask,
       datapar::deduced_simd<int32_t, 8> vindex, const float *base_addr) {
    auto active        = static_cast<__m256>(mask);
    auto unsafe_index  = static_cast<__m256i>(vindex);
    __m256i safe_index = _mm256_and_si256(unsafe_index, _mm256_castps_si256(active));
    __m256 gathered    = _mm256_i32gather_ps(base_addr, safe_index, Scale);
    __m256 result      = _mm256_blendv_ps(static_cast<__m256>(src), gathered, active);
    return datapar::deduced_simd<float, 8>{result};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 4>
gather(datapar::deduced_simd<double, 4> src,
       typename datapar::deduced_simd<double, 4>::mask_type mask,
       datapar::deduced_simd<int64_t, 4> vindex, const double *base_addr) {
    auto active        = static_cast<__m256d>(mask);
    auto unsafe_index  = static_cast<__m256i>(vindex);
    __m256i safe_index = _mm256_and_si256(unsafe_index, _mm256_castpd_si256(active));
    __m256d gathered   = _mm256_i64gather_pd(base_addr, safe_index, Scale);
    __m256d result     = _mm256_blendv_pd(static_cast<__m256d>(src), gathered, active);
    return datapar::deduced_simd<double, 4>{result};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 4>
gather(datapar::deduced_simd<double, 4> src,
       typename datapar::deduced_simd<double, 4>::mask_type mask,
       datapar::deduced_simd<int32_t, 4> vindex, const double *base_addr) {
    auto active        = static_cast<__m256d>(mask);
    auto active_i32    = _mm256_cvtepi64_epi32(_mm256_castpd_si256(active));
    auto unsafe_index  = static_cast<__m128i>(vindex);
    __m128i safe_index = _mm_and_si128(unsafe_index, active_i32);
    __m256d gathered   = _mm256_i32gather_pd(base_addr, safe_index, Scale);
    __m256d result     = _mm256_blendv_pd(static_cast<__m256d>(src), gathered, active);
    return datapar::deduced_simd<double, 4>{result};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 4>
gather(datapar::deduced_simd<float, 4> src,
       typename datapar::deduced_simd<float, 4>::mask_type mask,
       datapar::deduced_simd<int32_t, 4> vindex, const float *base_addr) {
    auto active        = static_cast<__m128>(mask);
    auto unsafe_index  = static_cast<__m128i>(vindex);
    __m128i safe_index = _mm_and_si128(unsafe_index, _mm_castps_si128(active));
    __m128 gathered    = _mm_i32gather_ps(base_addr, safe_index, Scale);
    __m128 result      = _mm_blendv_ps(static_cast<__m128>(src), gathered, active);
    return datapar::deduced_simd<float, 4>{result};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 2>
gather(datapar::deduced_simd<double, 2> src,
       typename datapar::deduced_simd<double, 2>::mask_type mask,
       datapar::deduced_simd<int64_t, 2> vindex, const double *base_addr) {
    auto active        = static_cast<__m128d>(mask);
    auto unsafe_index  = static_cast<__m128i>(vindex);
    __m128i safe_index = _mm_and_si128(unsafe_index, _mm_castpd_si128(active));
    __m128d gathered   = _mm_i64gather_pd(base_addr, safe_index, Scale);
    __m128d result     = _mm_blendv_pd(static_cast<__m128d>(src), gathered, active);
    return datapar::deduced_simd<double, 2>{result};
}

#if !BATMAT_WITH_GSI_HPC_SIMD // TODO
template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 2>
gather(datapar::deduced_simd<double, 2> src,
       typename datapar::deduced_simd<double, 2>::mask_type mask,
       datapar::deduced_simd<int32_t, 2> vindex, const double *base_addr) {
    auto active        = static_cast<__m128d>(mask);
    auto active_i32    = _mm_cvtepi64_epi32(_mm_castpd_si128(active));
    auto unsafe_index  = static_cast<__m128i>(vindex);
    __m128i safe_index = _mm_and_si128(unsafe_index, active_i32);
    __m128d gathered   = _mm_i32gather_pd(base_addr, safe_index, Scale);
    __m128d result     = _mm_blendv_pd(static_cast<__m128d>(src), gathered, active);
    return datapar::deduced_simd<double, 2>{result};
}
#endif

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 8>
gather(datapar::deduced_simd<int32_t, 8> vindex, const float *base_addr) {
    auto unsafe_index = static_cast<__m256i>(vindex);
    __m256 gathered   = _mm256_i32gather_ps(base_addr, unsafe_index, Scale);
    return datapar::deduced_simd<float, 8>{gathered};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 4>
gather(datapar::deduced_simd<int64_t, 4> vindex, const double *base_addr) {
    auto unsafe_index = static_cast<__m256i>(vindex);
    __m256d gathered  = _mm256_i64gather_pd(base_addr, unsafe_index, Scale);
    return datapar::deduced_simd<double, 4>{gathered};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 4>
gather(datapar::deduced_simd<int32_t, 4> vindex, const double *base_addr) {
    auto unsafe_index = static_cast<__m128i>(vindex);
    __m256d gathered  = _mm256_i32gather_pd(base_addr, unsafe_index, Scale);
    return datapar::deduced_simd<double, 4>{gathered};
}

template <int Scale = sizeof(float)>
[[gnu::always_inline]] inline datapar::deduced_simd<float, 4>
gather(datapar::deduced_simd<int32_t, 4> vindex, const float *base_addr) {
    auto unsafe_index = static_cast<__m128i>(vindex);
    __m128 gathered   = _mm_i32gather_ps(base_addr, unsafe_index, Scale);
    return datapar::deduced_simd<float, 4>{gathered};
}

template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 2>
gather(datapar::deduced_simd<int64_t, 2> vindex, const double *base_addr) {
    auto unsafe_index = static_cast<__m128i>(vindex);
    __m128d gathered  = _mm_i64gather_pd(base_addr, unsafe_index, Scale);
    return datapar::deduced_simd<double, 2>{gathered};
}

#if !BATMAT_WITH_GSI_HPC_SIMD // TODO
template <int Scale = sizeof(double)>
[[gnu::always_inline]] inline datapar::deduced_simd<double, 2>
gather(datapar::deduced_simd<int32_t, 2> vindex, const double *base_addr) {
    auto unsafe_index = static_cast<__m128i>(vindex);
    __m128d gathered  = _mm_i32gather_pd(base_addr, unsafe_index, Scale);
    return datapar::deduced_simd<double, 2>{gathered};
}
#endif

#endif

template <class T1, class T2>
concept same_size_but_different_ints =
    std::integral<T1> && std::integral<T2> && sizeof(T1) == sizeof(T2) && !std::same_as<T1, T2>;

} // namespace detail

template <class T, size_t N, class I>
datapar::deduced_simd<T, N> gather(const T *p, datapar::deduced_simd<I, N> idx,
                                   typename datapar::deduced_simd<T, N>::mask_type msk) {
    if constexpr (detail::same_size_but_different_ints<I, int32_t>) {
#if BATMAT_WITH_GSI_HPC_SIMD // TODO: add simd_cast wrapper
        const datapar::deduced_simd<int32_t, N> idx_{idx};
#else
        const auto idx_ = simd_cast<datapar::deduced_simd<int32_t, N>>(idx);
#endif
        return detail::gather(datapar::deduced_simd<T, N>{}, msk, idx_, p);
    } else if constexpr (detail::same_size_but_different_ints<I, int64_t>) {
#if BATMAT_WITH_GSI_HPC_SIMD // TODO: add simd_cast wrapper
        const datapar::deduced_simd<int64_t, N> idx_{idx};
#else
        const auto idx_ = simd_cast<datapar::deduced_simd<int64_t, N>>(idx);
#endif
        return detail::gather(datapar::deduced_simd<T, N>{}, msk, idx_, p);
    } else {
        return detail::gather(datapar::deduced_simd<T, N>{}, msk, idx, p);
    }
}

template <class T, size_t N, class I>
datapar::deduced_simd<T, N> gather(const T *p, datapar::deduced_simd<I, N> idx) {
    if constexpr (detail::same_size_but_different_ints<I, int32_t>) {
#if BATMAT_WITH_GSI_HPC_SIMD // TODO: add simd_cast wrapper
        const datapar::deduced_simd<int32_t, N> idx_{idx};
#else
        const auto idx_ = simd_cast<datapar::deduced_simd<int32_t, N>>(idx);
#endif
        return detail::gather(idx_, p);
    } else if constexpr (detail::same_size_but_different_ints<I, int64_t>) {
#if BATMAT_WITH_GSI_HPC_SIMD // TODO: add simd_cast wrapper
        const datapar::deduced_simd<int64_t, N> idx_{idx};
#else
        const auto idx_ = simd_cast<datapar::deduced_simd<int64_t, N>>(idx);
#endif
        return detail::gather(idx_, p);
    } else {
        return detail::gather(idx, p);
    }
}

#if __AVX2__ // TODO: write optimized intrinsic variants
template <>
inline datapar::deduced_simd<double, 8>
gather<double, 8, int>(const double *p, datapar::deduced_simd<int, 8> idx,
                       typename datapar::deduced_simd<double, 8>::mask_type msk) {
    return datapar::deduced_simd<double, 8>{[=](auto i) { return msk[i] ? p[idx[i]] : 0.0; }};
}
template <>
inline datapar::deduced_simd<double, 2>
gather<double, 2, int>(const double *p, datapar::deduced_simd<int, 2> idx,
                       typename datapar::deduced_simd<double, 2>::mask_type msk) {
    return datapar::deduced_simd<double, 2>{[=](auto i) { return msk[i] ? p[idx[i]] : 0.0; }};
}
template <>
inline datapar::deduced_simd<float, 4>
gather<float, 4, long long>(const float *p, datapar::deduced_simd<long long, 4> idx,
                            typename datapar::deduced_simd<float, 4>::mask_type msk) {
    return datapar::deduced_simd<float, 4>{[=](auto i) { return msk[i] ? p[idx[i]] : 0.0f; }};
}

template <>
inline datapar::deduced_simd<double, 8> gather<double, 8, int>(const double *p,
                                                               datapar::deduced_simd<int, 8> idx) {
    return datapar::deduced_simd<double, 8>{[=](auto i) { return p[idx[i]]; }};
}
template <>
inline datapar::deduced_simd<double, 2> gather<double, 2, int>(const double *p,
                                                               datapar::deduced_simd<int, 2> idx) {
    return datapar::deduced_simd<double, 2>{[=](auto i) { return p[idx[i]]; }};
}
template <>
inline datapar::deduced_simd<float, 4>
gather<float, 4, long long>(const float *p, datapar::deduced_simd<long long, 4> idx) {
    return datapar::deduced_simd<float, 4>{[=](auto i) { return p[idx[i]]; }};
}
#endif

} // namespace batmat::ops
