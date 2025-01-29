#pragma once

#include <experimental/simd>

namespace koqkatoo::linalg::compact::micro_kernels {

namespace stdx = std::experimental;

template <size_t S, class F, class Abi>
[[gnu::always_inline]] inline stdx::simd<F, Abi> rotl(stdx::simd<F, Abi> x) {
    if constexpr (S % x.size() == 0) {
        return x;
    } else {
        static_assert(S < x.size());
        stdx::simd<F, Abi> y;
        for (size_t j = 0; j < x.size() - S; ++j)
            y[j] = x[j + S];
        for (size_t j = x.size() - S; j < x.size(); ++j)
            y[j] = x[j + S - x.size()];
        return y;
    }
}

template <size_t S, class F, class Abi>
[[gnu::always_inline]] inline stdx::simd<F, Abi> rotr(stdx::simd<F, Abi> x) {
    if constexpr (S % x.size() == 0) {
        return x;
    } else {
        static_assert(S < x.size());
        stdx::simd<F, Abi> y;
        for (size_t j = x.size() - S; j < x.size(); ++j)
            y[j + S - x.size()] = x[j];
        for (size_t j = 0; j < x.size() - S; ++j)
            y[j + S] = x[j];
        return y;
    }
}

#if defined(__AVX512F__)

template <size_t S>
[[gnu::always_inline]] inline auto
rotl(stdx::simd<double, stdx::simd_abi::deduce_t<double, 8>> x) {
    if constexpr (S % x.size() == 0) {
        return x;
    } else {
        static_assert(S < x.size());
        constexpr size_t N = x.size();
        const __m512i indices =
            _mm512_set_epi64((S + 7) % N, (S + 6) % N, (S + 5) % N, (S + 4) % N,
                             (S + 3) % N, (S + 2) % N, (S + 1) % N, S % N);
        __m512d y = _mm512_permutexvar_pd(indices, static_cast<__m512d>(x));
        return decltype(x){y};
    }
}

template <size_t S>
[[gnu::always_inline]] inline auto
rotr(stdx::simd<double, stdx::simd_abi::deduce_t<double, 8>> x) {
    if constexpr (S % x.size() == 0) {
        return x;
    } else {
        static_assert(S < x.size());
        constexpr size_t N    = x.size();
        const __m512i indices = _mm512_set_epi64(
            (N - S + 7) % N, (N - S + 6) % N, (N - S + 5) % N, (N - S + 4) % N,
            (N - S + 3) % N, (N - S + 2) % N, (N - S + 1) % N, (N - S) % N);
        __m512d y = _mm512_permutexvar_pd(indices, static_cast<__m512d>(x));
        return decltype(x){y};
    }
}

#endif

#if defined(__AVX512F__) && 0

template <size_t S>
[[gnu::always_inline]] inline auto
rotl(stdx::simd<double, stdx::simd_abi::deduce_t<double, 4>> x) {
    if constexpr (S % x.size() == 0) {
        return x;
    } else {
        static_assert(S < x.size());
        constexpr size_t N = x.size();
        const __m256i indices =
            _mm256_set_epi64x((S + 3) % N, (S + 2) % N, (S + 1) % N, S % N);
        __m256d y = _mm256_permutexvar_pd(indices, static_cast<__m256d>(x));
        return decltype(x){y};
    }
}

template <size_t S>
[[gnu::always_inline]] inline auto
rotr(stdx::simd<double, stdx::simd_abi::deduce_t<double, 4>> x) {
    if constexpr (S % x.size() == 0) {
        return x;
    } else {
        static_assert(S < x.size());
        constexpr size_t N    = x.size();
        const __m256i indices = _mm256_set_epi64x(
            (N - S + 3) % N, (N - S + 2) % N, (N - S + 1) % N, (N - S) % N);
        __m256d y = _mm256_permutexvar_pd(indices, static_cast<__m256d>(x));
        return decltype(x){y};
    }
}

#elif defined(__AVX2__)

template <size_t S>
[[gnu::always_inline]] inline auto
rotl(stdx::simd<double, stdx::simd_abi::deduce_t<double, 4>> x) {
    if constexpr (S % x.size() == 0) {
        return x;
    } else {
        static_assert(S < x.size());
        constexpr size_t N    = x.size();
        constexpr int indices = (((S + 3) % N) << 6) | (((S + 2) % N) << 4) |
                                (((S + 1) % N) << 2) | (S % N);
        __m256d y = _mm256_permute4x64_pd(static_cast<__m256d>(x), indices);
        return decltype(x){y};
    }
}

template <size_t S>
[[gnu::always_inline]] inline auto
rotr(stdx::simd<double, stdx::simd_abi::deduce_t<double, 4>> x) {
    if constexpr (S % x.size() == 0) {
        return x;
    } else {
        static_assert(S < x.size());
        constexpr size_t N    = x.size();
        constexpr int indices = (((N - S + 3) % N) << 6) |
                                (((N - S + 2) % N) << 4) |
                                (((N - S + 1) % N) << 2) | ((N - S) % N);
        __m256d y = _mm256_permute4x64_pd(static_cast<__m256d>(x), indices);
        return decltype(x){y};
    }
}

#endif

} // namespace koqkatoo::linalg::compact::micro_kernels
