#pragma once

#include <batmat/simd.hpp>
#include <cassert>

namespace batmat::ops {

namespace detail {

template <class F, class Abi>
[[gnu::always_inline]] inline datapar::simd<F, Abi> rot(datapar::simd<F, Abi> x, int s) {
    assert(s <= 0 || static_cast<size_t>(+s) < x.size());
    assert(s >= 0 || static_cast<size_t>(-s) < x.size());
    datapar::simd<F, Abi> y;
    for (size_t j = 0; j < x.size(); ++j)
        y[j] = x[(x.size() + j - s) % x.size()];
    return y;
}

template <int S, class F, class Abi>
[[gnu::always_inline]] inline datapar::simd<F, Abi> rotl(datapar::simd<F, Abi> x) {
    static_assert(S > 0 && S < x.size());
    datapar::simd<F, Abi> y;
    for (size_t j = 0; j < x.size() - S; ++j)
        y[j] = x[j + S];
    for (size_t j = x.size() - S; j < x.size(); ++j)
        y[j] = x[j + S - x.size()];
    return y;
}

template <int S, class F, class Abi>
[[gnu::always_inline]] inline datapar::simd<F, Abi> rotr(datapar::simd<F, Abi> x) {
    static_assert(S > 0 && S < x.size());
    datapar::simd<F, Abi> y;
    for (size_t j = x.size() - S; j < x.size(); ++j)
        y[j + S - x.size()] = x[j];
    for (size_t j = 0; j < x.size() - S; ++j)
        y[j + S] = x[j];
    return y;
}

template <int S, class F, class Abi>
[[gnu::always_inline]] inline datapar::simd<F, Abi> shiftl(datapar::simd<F, Abi> x) {
    static_assert(S > 0 && S < x.size());
    datapar::simd<F, Abi> y;
    for (size_t j = 0; j < x.size() - S; ++j)
        y[j] = x[j + S];
    for (size_t j = x.size() - S; j < x.size(); ++j)
        y[j] = 0;
    return y;
}

template <int S, class F, class Abi>
[[gnu::always_inline]] inline datapar::simd<F, Abi> shiftr(datapar::simd<F, Abi> x) {
    static_assert(S > 0 && S < x.size());
    datapar::simd<F, Abi> y;
    for (size_t j = x.size() - S; j < x.size(); ++j)
        y[j + S - x.size()] = 0;
    for (size_t j = 0; j < x.size() - S; ++j)
        y[j + S] = x[j];
    return y;
}

#if defined(__AVX512F__)

[[gnu::always_inline]] inline auto rot(datapar::deduced_simd<double, 8> x, int s) {
    assert(s <= 0 || static_cast<size_t>(+s) < x.size());
    assert(s >= 0 || static_cast<size_t>(-s) < x.size());
    constexpr size_t N                                          = x.size();
    static constinit std::array<int64_t, 2 * N - 1> indices_lut = [] {
        std::array<int64_t, 2 * N - 1> indices_lut{};
        for (size_t i = 0; i < 2 * N - 1; ++i)
            indices_lut[i] = static_cast<int64_t>((i + 1) % N);
        return indices_lut;
    }();
    static constinit const int64_t *p = indices_lut.data() + N - 1;
    const __m512i indices             = _mm512_loadu_epi64(p - s);
    __m512d y                         = _mm512_permutexvar_pd(indices, static_cast<__m512d>(x));
    return decltype(x){y};
}

[[gnu::always_inline]] inline auto rot(datapar::deduced_simd<double, 4> x, int s) {
    assert(s <= 0 || static_cast<size_t>(+s) < x.size());
    assert(s >= 0 || static_cast<size_t>(-s) < x.size());
    constexpr size_t N                                          = x.size();
    static constinit std::array<int64_t, 2 * N - 1> indices_lut = [] {
        std::array<int64_t, 2 * N - 1> indices_lut{};
        for (size_t i = 0; i < 2 * N - 1; ++i)
            indices_lut[i] = static_cast<int64_t>((i + 1) % N);
        return indices_lut;
    }();
    static constinit const int64_t *p = indices_lut.data() + N - 1;
    const __m256i indices             = _mm256_loadu_epi64(p - s);
    __m256d y                         = _mm256_permutexvar_pd(indices, static_cast<__m256d>(x));
    return decltype(x){y};
}

template <int S>
[[gnu::always_inline]] inline auto rotl(datapar::deduced_simd<double, 8> x) {
    static_assert(S > 0 && S < x.size());
    constexpr size_t N    = x.size();
    const __m512i indices = _mm512_set_epi64((S + 7) % N, (S + 6) % N, (S + 5) % N, (S + 4) % N,
                                             (S + 3) % N, (S + 2) % N, (S + 1) % N, S % N);
    __m512d y             = _mm512_permutexvar_pd(indices, static_cast<__m512d>(x));
    return decltype(x){y};
}

template <int S>
[[gnu::always_inline]] inline auto rotr(datapar::deduced_simd<double, 8> x) {
    static_assert(S > 0 && S < x.size());
    constexpr size_t N = x.size();
    const __m512i indices =
        _mm512_set_epi64((N - S + 7) % N, (N - S + 6) % N, (N - S + 5) % N, (N - S + 4) % N,
                         (N - S + 3) % N, (N - S + 2) % N, (N - S + 1) % N, (N - S) % N);
    __m512d y = _mm512_permutexvar_pd(indices, static_cast<__m512d>(x));
    return decltype(x){y};
}

template <int S>
[[gnu::always_inline]] inline auto shiftl(datapar::deduced_simd<double, 8> x) {
    static_assert(S > 0 && S < x.size());
    constexpr uint8_t mask = (1u << (x.size() - S)) - 1u;
    auto y                 = static_cast<__m512d>(rotl<S>(x));
    y                      = _mm512_mask_blend_pd(mask, _mm512_set1_pd(0), y);
    return decltype(x){y};
}

template <int S>
[[gnu::always_inline]] inline auto shiftr(datapar::deduced_simd<double, 8> x) {
    static_assert(S > 0 && S < x.size());
    constexpr uint8_t mask = (1u << S) - 1u;
    auto y                 = static_cast<__m512d>(rotr<S>(x));
    y                      = _mm512_mask_blend_pd(mask, y, _mm512_set1_pd(0));
    return decltype(x){y};
}

#endif

#if defined(__AVX2__)

template <int S>
[[gnu::always_inline]] inline auto shiftl(datapar::deduced_simd<double, 4> x) {
    static_assert(S > 0 && S < x.size());
    constexpr uint8_t mask = (1u << (x.size() - S)) - 1u;
    auto y                 = static_cast<__m256d>(rotl<S>(x));
    y                      = _mm256_blend_pd(_mm256_set1_pd(0), y, mask);
    return decltype(x){y};
}

template <int S>
[[gnu::always_inline]] inline auto shiftr(datapar::deduced_simd<double, 4> x) {
    static_assert(S > 0 && S < x.size());
    constexpr uint8_t mask = (1u << S) - 1u;
    auto y                 = static_cast<__m256d>(rotr<S>(x));
    y                      = _mm256_blend_pd(y, _mm256_set1_pd(0), mask);
    return decltype(x){y};
}

#endif

#if defined(__AVX512F__)

template <int S>
[[gnu::always_inline]] inline auto rotl(datapar::deduced_simd<double, 4> x) {
    static_assert(S > 0 && S < x.size());
    constexpr size_t N    = x.size();
    const __m256i indices = _mm256_set_epi64x((S + 3) % N, (S + 2) % N, (S + 1) % N, S % N);
    __m256d y             = _mm256_permutexvar_pd(indices, static_cast<__m256d>(x));
    return decltype(x){y};
}

template <int S>
[[gnu::always_inline]] inline auto rotr(datapar::deduced_simd<double, 4> x) {
    static_assert(S > 0 && S < x.size());
    constexpr size_t N = x.size();
    const __m256i indices =
        _mm256_set_epi64x((N - S + 3) % N, (N - S + 2) % N, (N - S + 1) % N, (N - S) % N);
    __m256d y = _mm256_permutexvar_pd(indices, static_cast<__m256d>(x));
    return decltype(x){y};
}

#elif defined(__AVX2__)

template <int S>
[[gnu::always_inline]] inline auto rotl(datapar::deduced_simd<double, 4> x) {
    static_assert(S > 0 && S < x.size());
    constexpr size_t N = x.size();
    constexpr int indices =
        (((S + 3) % N) << 6) | (((S + 2) % N) << 4) | (((S + 1) % N) << 2) | (S % N);
    __m256d y = _mm256_permute4x64_pd(static_cast<__m256d>(x), indices);
    return decltype(x){y};
}

template <int S>
[[gnu::always_inline]] inline auto rotr(datapar::deduced_simd<double, 4> x) {
    static_assert(S > 0 && S < x.size());
    constexpr size_t N    = x.size();
    constexpr int indices = (((N - S + 3) % N) << 6) | (((N - S + 2) % N) << 4) |
                            (((N - S + 1) % N) << 2) | ((N - S) % N);
    __m256d y = _mm256_permute4x64_pd(static_cast<__m256d>(x), indices);
    return decltype(x){y};
}

#endif

} // namespace detail

template <int S, class F, class Abi>
[[gnu::always_inline]] inline datapar::simd<F, Abi> rotl(datapar::simd<F, Abi> x) {
    if constexpr (S % x.size() == 0)
        return x;
    else if constexpr (S < 0)
        return detail::rotr<-S>(x);
    else
        return detail::rotl<S>(x);
}

template <int S, class F, class Abi>
[[gnu::always_inline]] inline datapar::simd<F, Abi> rotr(datapar::simd<F, Abi> x) {
    if constexpr (S % x.size() == 0)
        return x;
    else if constexpr (S < 0)
        return detail::rotl<-S>(x);
    else
        return detail::rotr<S>(x);
}

template <int S, class F, class Abi>
[[gnu::always_inline]] inline datapar::simd<F, Abi> shiftl(datapar::simd<F, Abi> x) {
    if constexpr (S == 0)
        return x;
    else if constexpr (S >= static_cast<int>(x.size()) || -S >= static_cast<int>(x.size()))
        return datapar::simd<F, Abi>{0};
    else if constexpr (S < 0)
        return detail::shiftr<-S>(x);
    else
        return detail::shiftl<S>(x);
}

template <int S, class F, class Abi>
[[gnu::always_inline]] inline datapar::simd<F, Abi> shiftr(datapar::simd<F, Abi> x) {
    if constexpr (S == 0)
        return x;
    else if constexpr (S >= static_cast<int>(x.size()) || -S >= static_cast<int>(x.size()))
        return datapar::simd<F, Abi>{0};
    else if constexpr (S < 0)
        return detail::shiftl<-S>(x);
    else
        return detail::shiftr<S>(x);
}

using detail::rot;

} // namespace batmat::ops
