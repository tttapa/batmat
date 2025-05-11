#pragma once

#include <koqkatoo/linalg-compact/compact.hpp>

#include <koqkatoo/assume.hpp>
#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/mkl.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/lapack.hpp>
#include <koqkatoo/lut.hpp>
#include <koqkatoo/openmp.h>
#include <guanaqo/trace.hpp>

#include "compact/micro-kernels/rotate.hpp"
#include "compact/xgemm.tpp"
#include "compact/xpntrf.tpp"
#include "compact/xpotrf.tpp"
#include "compact/xshh.tpp"
#include "compact/xshhud-diag.tpp"
#include "compact/xsyrk.tpp"
#include "compact/xtrsm.tpp"
#include "compact/xtrtri.tpp"

#include <algorithm>
#include <array>

namespace koqkatoo::linalg::compact {

template <index_t R, index_t C, class T>
[[gnu::always_inline]]
inline void transpose(const T *pa, index_t lda, T *pb, index_t ldb, index_t d) {
    GUANAQO_ASSUME(d <= R);
    T r[C][R];
    for (int i = 0; i < C; ++i)
        for (int j = 0; j < R; ++j)
            r[i][j] = pa[i * lda + j];
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < C; ++j)
            pb[i * ldb + j] = r[j][i];
}

#ifdef __AVX2__ // TODO
template <>
[[gnu::always_inline]]
inline void transpose<4, 4>(const double *pa, index_t lda, double *pb,
                            index_t ldb, index_t d) {
    using simd = stdx::simd<double, stdx::simd_abi::deduce_t<double, 4>>;
    simd cols[4], shuf[4];
    for (int i = 0; i < 4; ++i)
        cols[i].copy_from(pa + i * lda, stdx::vector_aligned);
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
#ifdef __AVX512F__
// TODO
#endif

template <class Abi>
void CompactBLAS<Abi>::unpack(single_batch_view A,
                              mut_single_batch_view_scalar B) {
    GUANAQO_TRACE("unpack", 0, A.rows() * A.cols() * A.depth());
    constexpr auto lut =
        make_1d_lut<simd_stride>([]<index_t R>(index_constant<R>) {
            return transpose<simd_stride, R + 1, real_t>;
        });
    for (index_t c = 0; c < A.cols(); ++c)
        foreach_chunked(
            0, A.rows(), simd_stride,
            [&](index_t r) {
                transpose<simd_stride, simd_stride>(
                    &A(0, r, c), simd_stride, &B(0, r, c), B.layer_stride(),
                    simd_stride);
            },
            [&](index_t r, index_t nr) {
                lut[nr - 1](A.block(r, c, nr, 1).data, simd_stride,
                            B.block(r, c, nr, 1).data, B.layer_stride(),
                            simd_stride);
            });
}

template <class Abi>
void CompactBLAS<Abi>::unpack(single_batch_view A, mut_batch_view_scalar B) {
    GUANAQO_TRACE("unpack", 0, A.rows() * A.cols() * A.depth());
    constexpr auto lut =
        make_1d_lut<simd_stride>([]<index_t R>(index_constant<R>) {
            return transpose<simd_stride, R + 1, real_t>;
        });
    for (index_t c = 0; c < A.cols(); ++c)
        foreach_chunked(
            0, A.rows(), simd_stride,
            [&](index_t r) {
                transpose<simd_stride, simd_stride>(
                    &A(0, r, c), simd_stride, &B(0, r, c), B.layer_stride(),
                    B.depth());
            },
            [&](index_t r, index_t nr) {
                lut[nr - 1](A.block(r, c, nr, 1).data, simd_stride,
                            B.block(r, c, nr, 1).data, B.layer_stride(),
                            B.depth());
            });
}

template <class Abi>
void CompactBLAS<Abi>::unpack(batch_view A, mut_batch_view_scalar B) {
    assert(A.depth() == B.depth());
    foreach_chunked(
        0, B.depth(), simd_stride,
        [&](index_t l) {
            unpack(A.batch(l / simd_stride), B.middle_layers(l, simd_stride));
        },
        [&](index_t l, index_t nl) {
            unpack(A.batch(l / simd_stride), B.middle_layers(l, nl));
        });
}

template <class Abi>
void CompactBLAS<Abi>::unpack_L(single_batch_view A,
                                mut_single_batch_view_scalar B) {
    [[maybe_unused]] auto [m, M] = std::minmax({A.rows(), A.cols()});
    GUANAQO_TRACE("unpack_L", 0, (m * (m + 1) / 2 + (M - m) * m) * A.depth());
    constexpr auto lut =
        make_1d_lut<simd_stride>([]<index_t R>(index_constant<R>) {
            return transpose<simd_stride, R + 1, real_t>;
        });
    for (index_t c = 0; c < A.cols(); ++c)
        foreach_chunked(
            c, A.rows(), simd_stride,
            [&](index_t r) {
                transpose<simd_stride, simd_stride>(
                    &A(0, r, c), simd_stride, &B(0, r, c), B.layer_stride(),
                    simd_stride);
            },
            [&](index_t r, index_t nr) {
                lut[nr - 1](A.block(r, c, nr, 1).data, simd_stride,
                            B.block(r, c, nr, 1).data, B.layer_stride(),
                            simd_stride);
            });
}

template <class Abi>
void CompactBLAS<Abi>::unpack_L(single_batch_view A, mut_batch_view_scalar B) {
    [[maybe_unused]] auto [m, M] = std::minmax({A.rows(), A.cols()});
    GUANAQO_TRACE("unpack_L", 0, (m * (m + 1) / 2 + (M - m) * m) * A.depth());
    constexpr auto S   = simd_stride;
    constexpr auto lut = make_1d_lut<S>([]<index_t R>(index_constant<R>) {
        return transpose<S, R + 1, real_t>;
    });
#if 1
    const auto colstrA = A.outer_stride() * S;
    const auto colstrB = B.outer_stride();
    const auto laystrB = B.layer_stride();
    const auto d       = B.depth();
    auto pA            = A.data;
    const auto pAend   = pA + A.cols() * colstrA;
    real_t *pB         = B.data;
    auto rowcnt        = A.rows();
    while (pA < pAend) {
        const real_t *pA_ = pA;
        real_t *pB_       = pB;
        index_t r;
        for (r = 0; r + S <= rowcnt; r += S) {
            transpose<S, S>(pA_, S, pB_, laystrB, d);
            pA_ += S * S;
            pB_ += S;
        }
        index_t nr = rowcnt - r;
        if (nr > 0) [[unlikely]]
            lut[nr - 1](pA_, S, pB_, laystrB, d);
        pA += colstrA + S;
        pB += colstrB + 1;
        --rowcnt;
    }
#else
    for (index_t c = 0; c < A.cols(); ++c) {
        index_t r;
        for (r = c; r + S <= A.rows(); r += S)
            transpose<S, S>(&A(0, r, c), S, &B(0, r, c), B.layer_stride(),
                            B.depth());
        index_t nr = A.rows() - r;
        if (nr > 0) [[unlikely]]
            lut[nr - 1](&A(0, r, c), S, &B(0, r, c), B.layer_stride(),
                        B.depth());
    }
#endif
}

template <class Abi>
void CompactBLAS<Abi>::unpack_L(batch_view A, mut_batch_view_scalar B) {
    assert(A.depth() >= B.depth());
    foreach_chunked(
        0, B.depth(), simd_stride,
        [&](index_t l) {
            unpack_L(A.batch(l / simd_stride), B.middle_layers(l, simd_stride));
        },
        [&](index_t l, index_t nl) {
            unpack_L(A.batch(l / simd_stride), B.middle_layers(l, nl));
        });
}

template <>
void CompactBLAS<stdx::simd_abi::scalar>::unpack(batch_view A,
                                                 mut_batch_view_scalar B) {
    GUANAQO_TRACE("unpack", 0, B.rows() * B.cols() * B.depth());
    assert(A.depth() >= B.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    if (A.rows() == A.outer_stride() && B.rows() == B.outer_stride())
        for (index_t l = 0; l < B.depth(); ++l)
            std::copy_n(A(l).data, A.rows() * A.cols(), B(l).data);
    else
        for (index_t l = 0; l < B.depth(); ++l)
            for (index_t c = 0; c < B.cols(); ++c)
                std::copy_n(&A(l, 0, c), A.rows(), &B(l, 0, c));
}

template <>
void CompactBLAS<stdx::simd_abi::scalar>::unpack_L(batch_view A,
                                                   mut_batch_view_scalar B) {
    GUANAQO_TRACE("unpack", 0, B.rows() * B.cols() * B.depth());
    assert(A.depth() >= B.depth());
    for (index_t l = 0; l < B.depth(); ++l)
        for (index_t c = 0; c < B.cols(); ++c)
            std::copy_n(&A(l, c, c), A.rows() - c, &B(l, c, c));
}

template <class Abi>
void CompactBLAS<Abi>::xtrmv_ref(single_batch_view L, mut_single_batch_view x) {
    GUANAQO_TRACE("xtrmv", 0, (L.rows() * (L.rows() + 1) / 2) * L.depth());
    assert(L.rows() == L.cols());
    assert(x.rows() == L.cols());
    for (index_t j = L.cols(); j-- > 0;) {
        auto xj = aligned_load(&x(0, j, 0));
        for (index_t i = L.rows(); i-- > j;) {
            auto Lij = aligned_load(&L(0, i, j));
            aligned_store(&x(0, i, 0), fma(Lij, xj, aligned_load(&x(0, i, 0))));
        }
        aligned_store(&x(0, j, 0), aligned_load(&L(0, j, j)) * xj);
    }
}

template <class Abi>
void CompactBLAS<Abi>::xtrmv_T_ref(single_batch_view L,
                                   mut_single_batch_view x) {
    GUANAQO_TRACE("xtrmv_T", 0, (L.rows() * (L.rows() + 1) / 2) * L.depth());
    assert(L.rows() == L.cols());
    assert(x.rows() == L.cols());
    for (index_t j = 0; j < L.cols(); ++j) {
        simd accum{};
        for (index_t i = j; i < L.rows(); ++i) {
            auto Lij = aligned_load(&L(0, i, j));
            auto xi  = aligned_load(&x(0, i, 0));
            accum += Lij * xi;
        }
        aligned_store(&x(0, j, 0), accum);
    }
}

template <class Abi>
void CompactBLAS<Abi>::xtrmv(batch_view L, mut_batch_view x,
                             PreferredBackend b) {
    std::ignore = b; // TODO
    assert(L.ceil_depth() == x.ceil_depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrmv_ref(L.batch(i), x.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrmv_T(batch_view L, mut_batch_view x,
                               PreferredBackend b) {
    std::ignore = b; // TODO
    assert(L.ceil_depth() == x.ceil_depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrmv_T_ref(L.batch(i), x.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrmv(single_batch_view L, mut_single_batch_view x,
                             PreferredBackend b) {
    std::ignore = b; // TODO
    xtrmv_ref(L, x);
}

template <class Abi>
void CompactBLAS<Abi>::xtrmv_T(single_batch_view L, mut_single_batch_view x,
                               PreferredBackend b) {
    std::ignore = b; // TODO
    xtrmv_T_ref(L, x);
}

template <class Abi>
void CompactBLAS<Abi>::xsymv_add_ref(single_batch_view L, single_batch_view x,
                                     mut_single_batch_view y) {
    GUANAQO_TRACE("xsymv", 0, L.rows() * L.rows() * L.depth());
    assert(L.cols() == L.rows());
    assert(x.rows() == L.rows());
    assert(y.rows() == L.rows());
    assert(x.cols() == 1);
    assert(y.cols() == 1);
    const auto n = L.rows(), m = L.cols();
    for (index_t j = 0; j < m; ++j) {
        auto xj  = aligned_load(&x(0, j, 0));
        auto Ljj = aligned_load(&L(0, j, j));
        auto yj  = xj * Ljj + aligned_load(&y(0, j, 0));
        KOQKATOO_UNROLLED_IVDEP_FOR (4, index_t i = j + 1; i < n; ++i) {
            auto Lij = aligned_load(&L(0, i, j));
            auto xi  = aligned_load(&x(0, i, 0));
            aligned_store(&y(0, i, 0), xj * Lij + aligned_load(&y(0, i, 0)));
            yj += Lij * xi;
        }
        aligned_store(&y(0, j, 0), yj);
    }
}

template <class Abi>
void CompactBLAS<Abi>::xsymv_add(batch_view L, batch_view x, mut_batch_view y,
                                 PreferredBackend b) {
    std::ignore = b; // TODO
    assert(L.ceil_depth() == x.ceil_depth());
    assert(x.ceil_depth() == y.ceil_depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xsymv_add_ref(L.batch(i), x.batch(i), y.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xsymv_add(single_batch_view L, single_batch_view x,
                                 mut_single_batch_view y, PreferredBackend b) {
    std::ignore = b; // TODO
    xsymv_add_ref(L, x, y);
}

template <class Abi>
void CompactBLAS<Abi>::xcopy(single_batch_view A, mut_single_batch_view B) {
    GUANAQO_TRACE("xcopy", 0, A.rows() * A.cols() * A.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    const index_t n = A.rows(), m = A.cols();
    for (index_t j = 0; j < m; ++j)
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < n; ++i)
            aligned_store(&B(0, i, j), aligned_load(&A(0, i, j)));
    // std::copy_n(&A(0, 0, j), simd_stride * n, &B(0, 0, j));
}

template <class Abi>
void CompactBLAS<Abi>::xcopy(batch_view A, mut_batch_view B) {
    assert(A.ceil_depth() == B.ceil_depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xcopy(A.batch(i), B.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xcopy_L(single_batch_view A, mut_single_batch_view B) {
    [[maybe_unused]] auto [m, M] = std::minmax({A.rows(), A.cols()});
    GUANAQO_TRACE("xcopy_L", 0, (m * (m + 1) / 2 + (M - m) * m) * A.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    assert(A.rows() >= A.cols());
    const index_t rows = A.rows(), cols = A.cols();
    for (index_t j = 0; j < cols; ++j)
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = j; i < rows; ++i)
            aligned_store(&B(0, i, j), aligned_load(&A(0, i, j)));
}

template <class Abi>
void CompactBLAS<Abi>::xcopy_L(batch_view A, mut_batch_view B) {
    assert(A.ceil_depth() == B.ceil_depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xcopy_L(A.batch(i), B.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xcopy_T(single_batch_view A, mut_single_batch_view B) {
    GUANAQO_TRACE("xcopy_T", 0, A.rows() * A.cols() * A.depth());
    assert(A.rows() == B.cols());
    assert(A.cols() == B.rows());
    const index_t n = A.rows(), m = A.cols();
    for (index_t j = 0; j < m; ++j)
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < n; ++i)
            aligned_store(&B(0, j, i), aligned_load(&A(0, i, j)));
}

template <class Abi>
void CompactBLAS<Abi>::xcopy_T(batch_view A, mut_batch_view B) {
    assert(A.ceil_depth() == B.ceil_depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xcopy_T(A.batch(i), B.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xfill(real_t a, mut_single_batch_view B) {
    GUANAQO_TRACE("xcopy", 0, B.rows() * B.cols() * B.depth());
    const index_t n = B.rows(), m = B.cols();
    for (index_t j = 0; j < m; ++j)
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < n; ++i)
            aligned_store(&B(0, i, j), simd{a});
}

template <class Abi>
void CompactBLAS<Abi>::xfill(real_t a, mut_batch_view B) {
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < B.num_batches(); ++i)
        xfill(a, B.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xneg(mut_single_batch_view A) {
    GUANAQO_TRACE("xneg", 0, A.rows() * A.cols() * A.depth());
    const index_t n = A.rows(), m = A.cols();
    for (index_t j = 0; j < m; ++j)
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < n; ++i)
            aligned_store(&A(0, i, j), -aligned_load(&A(0, i, j)));
}

template <class Abi>
void CompactBLAS<Abi>::xneg(mut_batch_view A) {
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xneg(A.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xadd_L(single_batch_view A, mut_single_batch_view B) {
    [[maybe_unused]] const auto tri_op_cnt  = A.cols() * (A.cols() + 1) / 2,
                                rect_op_cnt = A.cols() * (A.rows() - A.cols());
    GUANAQO_TRACE("xadd_L", 0, (tri_op_cnt + rect_op_cnt) * A.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    assert(A.rows() >= A.cols());
    const index_t n = A.rows(), m = A.cols();
    for (index_t j = 0; j < m; ++j)
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = j; i < n; ++i)
            aligned_store(&B(0, i, j), aligned_load(&B(0, i, j)) +
                                           aligned_load(&A(0, i, j)));
}

template <class Abi>
void CompactBLAS<Abi>::xaxpy(real_t a, single_batch_view x,
                             mut_single_batch_view y) {
    GUANAQO_TRACE("xaxpy", 0, x.rows() * x.depth());
    assert(x.rows() == y.rows());
    assert(x.cols() == y.cols());
    simd a_simd{a};
    const index_t n = x.rows();
    for (index_t j = 0; j < x.cols(); ++j) {
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < n; ++i) {
            aligned_store(&y(0, i, j), a_simd * aligned_load(&x(0, i, j)) +
                                           aligned_load(&y(0, i, j)));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xaxpy(real_t a, batch_view x, mut_batch_view y) {
    assert(x.ceil_depth() == y.ceil_depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < x.num_batches(); ++i)
        xaxpy(a, x.batch(i), y.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xaxpby(real_t a, single_batch_view x, real_t b,
                              mut_single_batch_view y) {
    assert(x.rows() == y.rows());
    assert(x.cols() == y.cols());
    const index_t n = x.rows();
    if (b == 0) {
        GUANAQO_TRACE("xaxpby", 0, x.rows() * x.depth());
        simd a_simd{a};
        for (index_t j = 0; j < x.cols(); ++j) {
            KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < n; ++i) {
                aligned_store(&y(0, i, j), a_simd * aligned_load(&x(0, i, j)));
            }
        }
    } else {
        GUANAQO_TRACE("xaxpby", 0, 2 * x.rows() * x.depth());
        simd a_simd{a}, b_simd{b};
        for (index_t j = 0; j < x.cols(); ++j) {
            KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < n; ++i) {
                aligned_store(&y(0, i, j),
                              a_simd * aligned_load(&x(0, i, j)) +
                                  b_simd * aligned_load(&y(0, i, j)));
            }
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xaxpby(real_t a, batch_view x, real_t b,
                              mut_batch_view y) {
    assert(x.ceil_depth() == y.ceil_depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < x.num_batches(); ++i)
        xaxpby(a, x.batch(i), b, y.batch(i));
}

template <class Abi>
template <int Rot, class OutView, class View, class... Views>
void CompactBLAS<Abi>::xadd_copy_impl(OutView out, View x1, Views... xs)
    requires(((std::same_as<OutView, mut_batch_view> &&
               std::same_as<View, batch_view>) &&
              ... && std::same_as<Views, batch_view>) ||
             ((std::same_as<OutView, mut_single_batch_view> &&
               std::same_as<View, single_batch_view>) &&
              ... && std::same_as<Views, single_batch_view>))
{
    assert(((x1.batch_size() == xs.batch_size()) && ...));
    assert(x1.batch_size() == out.batch_size());
    assert(((x1.depth() == xs.depth()) && ...));
    assert(x1.depth() == out.depth());
    assert(((x1.rows() == xs.rows()) && ...));
    assert(x1.rows() == out.rows());
    assert(((x1.cols() == xs.cols()) && ...));
    assert(x1.cols() == out.cols());
    index_t i;
    const auto Bs   = static_cast<index_t>(x1.batch_size());
    const index_t n = x1.rows(), m = x1.cols();
    using micro_kernels::rotr;
    KOQKATOO_OMP(parallel for lastprivate(i))
    for (i = 0; i <= static_cast<index_t>(x1.depth()) - Bs; i += Bs) {
        for (index_t c = 0; c < m; ++c) {
            KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t r = 0; r < n; ++r) {
                aligned_store(&out(i, r, c),
                              (rotr<Rot>(aligned_load(&x1(i, r, c))) + ... +
                               aligned_load(&xs(i, r, c))));
            }
        }
    }
    for (; i < static_cast<index_t>(x1.depth()); ++i)
        for (index_t c = 0; c < m; ++c)
            for (index_t r = 0; r < n; ++r)
                out(i, r, c) = (x1(i, r, c) + ... + xs(i, r, c));
}

template <class Abi>
template <class OutView, class View, class... Views>
void CompactBLAS<Abi>::xsub_copy_impl(OutView out, View x1, Views... xs)
    requires(((std::same_as<OutView, mut_batch_view> &&
               std::same_as<View, batch_view>) &&
              ... && std::same_as<Views, batch_view>) ||
             ((std::same_as<OutView, mut_single_batch_view> &&
               std::same_as<View, single_batch_view>) &&
              ... && std::same_as<Views, single_batch_view>))
{
    assert(((x1.batch_size() == xs.batch_size()) && ...));
    assert(x1.batch_size() == out.batch_size());
    assert(((x1.depth() == xs.depth()) && ...));
    assert(x1.depth() == out.depth());
    assert(((x1.rows() == xs.rows()) && ...));
    assert(x1.rows() == out.rows());
    assert(((x1.cols() == xs.cols()) && ...));
    assert(x1.cols() == out.cols());
    index_t i;
    const auto Bs   = static_cast<index_t>(x1.batch_size());
    const index_t n = x1.rows(), m = x1.cols();
    KOQKATOO_OMP(parallel for lastprivate(i))
    for (i = 0; i <= static_cast<index_t>(x1.depth()) - Bs; i += Bs) {
        for (index_t c = 0; c < m; ++c) {
            KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t r = 0; r < n; ++r) {
                aligned_store(&out(i, r, c),
                              aligned_load(&x1(i, r, c)) -
                                  (... + aligned_load(&xs(i, r, c))));
            }
        }
    }
    for (; i < static_cast<index_t>(x1.depth()); ++i)
        for (index_t c = 0; c < m; ++c)
            for (index_t r = 0; r < n; ++r)
                out(i, r, c) = x1(i, r, c) - (... + xs(i, r, c));
}

template <class Abi>
template <class OutView, class... Views>
void CompactBLAS<Abi>::xadd_neg_copy_impl(OutView out, Views... xs)
    requires((std::same_as<OutView, mut_batch_view> && ... &&
              std::same_as<Views, batch_view>) ||
             (std::same_as<OutView, mut_single_batch_view> && ... &&
              std::same_as<Views, single_batch_view>))
{
    assert(((out.batch_size() == xs.batch_size()) && ...));
    assert(out.batch_size() == out.batch_size());
    assert(((out.depth() == xs.depth()) && ...));
    assert(out.depth() == out.depth());
    assert(((out.rows() == xs.rows()) && ...));
    assert(out.rows() == out.rows());
    assert(((out.cols() == xs.cols()) && ...));
    assert(out.cols() == out.cols());
    index_t i;
    const auto Bs   = static_cast<index_t>(out.batch_size());
    const index_t n = out.rows(), m = out.cols();
    KOQKATOO_OMP(parallel for lastprivate(i))
    for (i = 0; i <= static_cast<index_t>(out.depth()) - Bs; i += Bs) {
        for (index_t c = 0; c < m; ++c) {
            KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t r = 0; r < n; ++r) {
                aligned_store(&out(i, r, c),
                              -(... + aligned_load(&xs(i, r, c))));
            }
        }
    }
    for (; i < static_cast<index_t>(out.depth()); ++i)
        for (index_t c = 0; c < m; ++c)
            for (index_t r = 0; r < n; ++r)
                out(i, r, c) = -(... + xs(i, r, c));
}

template <class Abi>
real_t CompactBLAS<Abi>::xdot(single_batch_view x, single_batch_view y) {
    using std::fma;
    // TODO: why does fma(xi, yi, accum) give such terrible code gen?
    return xreduce(
        simd{0}, [](auto accum, auto xi, auto yi) { return xi * yi + accum; },
        [](auto accum) { return reduce(accum); }, x, y);
}

template <class Abi>
real_t CompactBLAS<Abi>::xdot(batch_view x, batch_view y) {
    using std::fma;
    // TODO: why does fma(xi, yi, accum) give such terrible code gen?
    return xreduce(
        simd{0}, [](auto accum, auto xi, auto yi) { return xi * yi + accum; },
        [](auto accum) { return reduce(accum); }, x, y);
}

template <class Abi>
real_t CompactBLAS<Abi>::xdot(size_last_t size_last, batch_view x,
                              batch_view y) {
    using std::fma;
    return xreduce(
        size_last, simd{0},
        [](auto accum, auto xi, auto yi) { return xi * yi + accum; },
        [](auto accum) { return reduce(accum); }, x, y);
}

template <class Abi>
real_t CompactBLAS<Abi>::xnrm2sq(batch_view x) {
    using std::fma;
    return xreduce(
        simd{0}, [](auto accum, auto xi) { return xi * xi + accum; },
        [](auto accum) { return reduce(accum); }, x);
}

template <class Abi>
real_t CompactBLAS<Abi>::xnrm2sq(size_last_t size_last, batch_view x) {
    using std::fma;
    return xreduce(
        size_last, simd{0}, [](auto accum, auto xi) { return xi * xi + accum; },
        [](auto accum) { return reduce(accum); }, x);
}

template <class Abi>
real_t CompactBLAS<Abi>::xnrminf(single_batch_view x) {
    using std::abs;
    using std::fma;
    using std::isfinite;
    using std::max;
    auto [inf_nrm, l1_norm] = xreduce(
        std::array<simd, 2>{0, 0},
        [](auto accum, auto xi) {
            return std::array{max(abs(xi), accum[0]), abs(xi) + accum[1]};
        },
        [](auto accum) { return std::array{hmax(accum[0]), reduce(accum[1])}; },
        x);
    return isfinite(l1_norm) ? inf_nrm : l1_norm;
}

template <class Abi>
real_t CompactBLAS<Abi>::xnrminf(batch_view x) {
    using std::abs;
    using std::fma;
    using std::isfinite;
    using std::max;
    auto [inf_nrm, l1_norm] = xreduce(
        std::array<simd, 2>{0, 0},
        [](auto accum, auto xi) {
            return std::array{max(abs(xi), accum[0]), abs(xi) + accum[1]};
        },
        [](auto accum) { return std::array{hmax(accum[0]), reduce(accum[1])}; },
        x);
    return isfinite(l1_norm) ? inf_nrm : l1_norm;
}

template <class Abi>
real_t CompactBLAS<Abi>::xnrminf(size_last_t size_last, batch_view x) {
    using std::abs;
    using std::fma;
    using std::isfinite;
    using std::max;
    auto [inf_nrm, l1_norm] = xreduce(
        size_last, std::array<simd, 2>{0, 0},
        [](auto accum, auto xi) {
            return std::array{max(abs(xi), accum[0]), abs(xi) + accum[1]};
        },
        [](auto accum) { return std::array{hmax(accum[0]), reduce(accum[1])}; },
        x);
    return isfinite(l1_norm) ? inf_nrm : l1_norm;
}

template <class Abi>
void CompactBLAS<Abi>::proj_diff(single_batch_view x, single_batch_view l,
                                 single_batch_view u, mut_single_batch_view y) {
    assert(x.rows() == y.rows());
    assert(x.rows() == l.rows());
    assert(x.rows() == u.rows());
    assert(x.cols() == y.cols());
    assert(x.cols() == l.cols());
    assert(x.cols() == u.cols());
    const index_t n = x.rows();
    for (index_t j = 0; j < x.cols(); ++j) {
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < n; ++i) {
            auto xij = aligned_load(&x(0, i, j));
            auto lij = aligned_load(&l(0, i, j));
            auto uij = aligned_load(&u(0, i, j));
            aligned_store(&y(0, i, j), xij - max(lij, min(xij, uij)));
        }
    }
}

} // namespace koqkatoo::linalg::compact
