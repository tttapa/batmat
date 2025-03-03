#pragma once

#include <koqkatoo/linalg/blas.hpp>
#include <experimental/simd>

namespace koqkatoo::linalg::compact {

namespace stdx = std::experimental;

template <class RealT, class Abi>
struct vector_format_mkl;

#if KOQKATOO_WITH_MKL

template <>
struct vector_format_mkl<double, stdx::simd_abi::deduce_t<double, 8>> {
    static constexpr MKL_COMPACT_PACK format = MKL_COMPACT_AVX512;
};
template <>
struct vector_format_mkl<double, stdx::simd_abi::deduce_t<double, 4>> {
    static constexpr MKL_COMPACT_PACK format = MKL_COMPACT_AVX;
};
template <>
struct vector_format_mkl<double, stdx::simd_abi::deduce_t<double, 2>> {
    static constexpr MKL_COMPACT_PACK format = MKL_COMPACT_SSE;
};
template <>
struct vector_format_mkl<float, stdx::simd_abi::deduce_t<float, 16>> {
    static constexpr MKL_COMPACT_PACK format = MKL_COMPACT_AVX512;
};
template <>
struct vector_format_mkl<float, stdx::simd_abi::deduce_t<float, 8>> {
    static constexpr MKL_COMPACT_PACK format = MKL_COMPACT_AVX;
};
template <>
struct vector_format_mkl<float, stdx::simd_abi::deduce_t<float, 4>> {
    static constexpr MKL_COMPACT_PACK format = MKL_COMPACT_SSE;
};

inline void xgemm_compact(MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                          MKL_TRANSPOSE transb, MKL_INT m, MKL_INT n, MKL_INT k,
                          double alpha, const double *a, MKL_INT ldap,
                          const double *b, MKL_INT ldbp, double beta, double *c,
                          MKL_INT ldcp, MKL_COMPACT_PACK format, MKL_INT nm) {
    ::mkl_dgemm_compact(layout, transa, transb, m, n, k, alpha, a, ldap, b,
                        ldbp, beta, c, ldcp, format, nm);
}
inline void xtrsm_compact(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,
                          MKL_TRANSPOSE transa, MKL_DIAG diag, MKL_INT m,
                          MKL_INT n, double alpha, const double *a,
                          MKL_INT ldap, double *b, MKL_INT ldbp,
                          MKL_COMPACT_PACK format, MKL_INT nm) {
    ::mkl_dtrsm_compact(layout, side, uplo, transa, diag, m, n, alpha, a, ldap,
                        b, ldbp, format, nm);
}
inline void xgemm_compact(MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                          MKL_TRANSPOSE transb, MKL_INT m, MKL_INT n, MKL_INT k,
                          float alpha, const float *a, MKL_INT ldap,
                          const float *b, MKL_INT ldbp, float beta, float *c,
                          MKL_INT ldcp, MKL_COMPACT_PACK format, MKL_INT nm) {
    ::mkl_sgemm_compact(layout, transa, transb, m, n, k, alpha, a, ldap, b,
                        ldbp, beta, c, ldcp, format, nm);
}
inline void xtrsm_compact(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,
                          MKL_TRANSPOSE transa, MKL_DIAG diag, MKL_INT m,
                          MKL_INT n, float alpha, const float *a, MKL_INT ldap,
                          float *b, MKL_INT ldbp, MKL_COMPACT_PACK format,
                          MKL_INT nm) {
    ::mkl_strsm_compact(layout, side, uplo, transa, diag, m, n, alpha, a, ldap,
                        b, ldbp, format, nm);
}
inline void xpotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
                           double *ap, MKL_INT ldap, MKL_INT *info,
                           MKL_COMPACT_PACK format, MKL_INT nm) {
    ::mkl_dpotrf_compact(layout, uplo, n, ap, ldap, info, format, nm);
}
inline void xpotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
                           float *ap, MKL_INT ldap, MKL_INT *info,
                           MKL_COMPACT_PACK format, MKL_INT nm) {
    ::mkl_spotrf_compact(layout, uplo, n, ap, ldap, info, format, nm);
}

#endif // KOQKATOO_WITH_MKL

template <class RealT, class Abi>
concept supports_mkl_packed =
    requires { vector_format_mkl<RealT, Abi>::format; };

} // namespace koqkatoo::linalg::compact

#if KOQKATOO_WITH_MKL
#define KOQKATOO_MKL_IF_ELSE(X, Y) X
#define KOQKATOO_MKL_IF(X) X
#else
#define KOQKATOO_MKL_IF_ELSE(X, Y) Y
#define KOQKATOO_MKL_IF(X)
#endif
