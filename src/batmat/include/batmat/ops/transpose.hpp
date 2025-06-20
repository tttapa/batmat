#pragma once

#include <batmat/assume.hpp>
#include <batmat/config.hpp>
#include <batmat/unroll.h>

#include <experimental/simd>

namespace batmat::ops {

template <index_t R, index_t C, class T>
[[gnu::always_inline]]
inline void transpose_dyn(const T *pa, index_t lda, T *pb, index_t ldb, index_t d) {
    BATMAT_ASSUME(d <= R);
    T r[C][R];
    BATMAT_FULLY_UNROLLED_FOR (int i = 0; i < C; ++i)
        BATMAT_FULLY_UNROLLED_FOR (int j = 0; j < R; ++j)
            r[i][j] = pa[i * lda + j];
    BATMAT_FULLY_UNROLLED_FOR (int i = 0; i < d; ++i)
        BATMAT_FULLY_UNROLLED_FOR (int j = 0; j < C; ++j)
            pb[i * ldb + j] = r[j][i];
}

#ifdef __AVX2__
template <>
[[gnu::always_inline]]
inline void transpose_dyn<4, 4>(const double *pa, index_t lda, double *pb, index_t ldb, index_t d) {
    namespace stdx = std::experimental;
    using simd     = stdx::simd<double, stdx::simd_abi::deduce_t<double, 4>>;
    simd cols[4], shuf[4];
    BATMAT_FULLY_UNROLLED_FOR (int i = 0; i < 4; ++i)
        cols[i].copy_from(pa + i * lda, stdx::element_aligned);
    // clang-format off
    shuf[0] = simd{_mm256_shuffle_pd((__m256d)cols[0], (__m256d)cols[1], 0b0000)};
    shuf[1] = simd{_mm256_shuffle_pd((__m256d)cols[0], (__m256d)cols[1], 0b1111)};
    shuf[2] = simd{_mm256_shuffle_pd((__m256d)cols[2], (__m256d)cols[3], 0b0000)};
    shuf[3] = simd{_mm256_shuffle_pd((__m256d)cols[2], (__m256d)cols[3], 0b1111)};
    cols[0] = simd{_mm256_permute2f128_pd((__m256d)shuf[0], (__m256d)shuf[2], 0b00100000)};
    cols[0].copy_to(pb + 0 * ldb, stdx::element_aligned);
    if (d < 2) [[unlikely]] return;
    cols[1] = simd{_mm256_permute2f128_pd((__m256d)shuf[1], (__m256d)shuf[3], 0b00100000)};
    cols[1].copy_to(pb + 1 * ldb, stdx::element_aligned);
    if (d < 3) [[unlikely]] return;
    cols[2] = simd{_mm256_permute2f128_pd((__m256d)shuf[0], (__m256d)shuf[2], 0b00110001)};
    cols[2].copy_to(pb + 2 * ldb, stdx::element_aligned);
    if (d < 4) [[unlikely]] return;
    cols[3] = simd{_mm256_permute2f128_pd((__m256d)shuf[1], (__m256d)shuf[3], 0b00110001)};
    cols[3].copy_to(pb + 3 * ldb, stdx::element_aligned);
    // clang-format on
}
#endif

template <index_t R, index_t C, class T>
[[gnu::always_inline]]
inline void transpose(const T *pa, index_t lda, T *pb, index_t ldb) {
    transpose_dyn<R, C>(pa, lda, pb, ldb, R);
}

} // namespace batmat::ops
