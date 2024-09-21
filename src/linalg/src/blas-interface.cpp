#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/export.h>
#include <koqkatoo/linalg/lapack.hpp>
#include <koqkatoo/openmp.h>

#define DO_INSTANTIATE 0

namespace koqkatoo::linalg {

template <>
KOQKATOO_LINALG_EXPORT void xgemv(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                                  blas_index_t M, blas_index_t N, double alpha,
                                  const double *A, blas_index_t lda,
                                  const double *X, blas_index_t incX,
                                  double beta, double *Y, blas_index_t incY) {
    return cblas_dgemv(Layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                       incY);
}
template <>
KOQKATOO_LINALG_EXPORT void xgemv(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                                  blas_index_t M, blas_index_t N, float alpha,
                                  const float *A, blas_index_t lda,
                                  const float *X, blas_index_t incX, float beta,
                                  float *Y, blas_index_t incY) {
    return cblas_sgemv(Layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                       incY);
}

template <>
KOQKATOO_LINALG_EXPORT void xgemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                                  CBLAS_TRANSPOSE TransB, index_t M, index_t N,
                                  index_t K, double alpha, const double *A,
                                  index_t lda, const double *B, index_t ldb,
                                  double beta, double *C, index_t ldc) {
    cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                ldc);
}
template <>
KOQKATOO_LINALG_EXPORT void
xgemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
      index_t M, index_t N, index_t K, float alpha, const float *A, index_t lda,
      const float *B, index_t ldb, float beta, float *C, index_t ldc) {
    cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                ldc);
}

#if DO_INSTANTIATE
template KOQKATOO_LINALG_EXPORT void xgemm<real_t, index_t>(
    CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    index_t M, index_t N, index_t K, real_t alpha, const real_t *A, index_t lda,
    const real_t *B, index_t ldb, real_t beta, real_t *C, index_t ldc);
#endif

template <>
KOQKATOO_LINALG_EXPORT void xsyrk(CBLAS_LAYOUT Layout, CBLAS_UPLO Uplo,
                                  CBLAS_TRANSPOSE Trans, index_t N, index_t K,
                                  double alpha, const double *A, index_t lda,
                                  double beta, double *C, index_t ldc) {
    cblas_dsyrk(Layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}
template <>
KOQKATOO_LINALG_EXPORT void xsyrk(CBLAS_LAYOUT Layout, CBLAS_UPLO Uplo,
                                  CBLAS_TRANSPOSE Trans, index_t N, index_t K,
                                  float alpha, const float *A, index_t lda,
                                  float beta, float *C, index_t ldc) {
    cblas_ssyrk(Layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
}

#if DO_INSTANTIATE
template KOQKATOO_LINALG_EXPORT void
xsyrk<real_t, index_t>(CBLAS_LAYOUT Layout, CBLAS_UPLO Uplo,
                       CBLAS_TRANSPOSE Trans, index_t N, index_t K,
                       real_t alpha, const real_t *A, index_t lda, real_t beta,
                       real_t *C, index_t ldc);
#endif

template <>
KOQKATOO_LINALG_EXPORT void
xtrsm(CBLAS_LAYOUT Layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
      CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, index_t M, index_t N,
      double alpha, const double *A, index_t lda, double *B, index_t ldb) {
    cblas_dtrsm(Layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}
template <>
KOQKATOO_LINALG_EXPORT void
xtrsm(CBLAS_LAYOUT Layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
      CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, index_t M, index_t N,
      float alpha, const float *A, index_t lda, float *B, index_t ldb) {
    cblas_strsm(Layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

#if DO_INSTANTIATE
template KOQKATOO_LINALG_EXPORT void
xtrsm<real_t, index_t>(CBLAS_LAYOUT Layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                       CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, index_t M,
                       index_t N, real_t alpha, const real_t *A, index_t lda,
                       real_t *B, index_t ldb);
#endif

template <>
void xsytrf_rk(const char *uplo, const index_t *n, double *a,
               const index_t *lda, double *e, index_t *ipiv, double *work,
               const index_t *lwork, index_t *info) {
    dsytrf_rk(uplo, n, a, lda, e, ipiv, work, lwork, info);
}

template <>
void xsytrf_rk(const char *uplo, const index_t *n, float *a, const index_t *lda,
               float *e, index_t *ipiv, float *work, const index_t *lwork,
               index_t *info) {
    ssytrf_rk(uplo, n, a, lda, e, ipiv, work, lwork, info);
}

template <>
void xtrtrs(const char *uplo, const char *trans, const char *diag,
            const index_t *n, const index_t *nrhs, const double *A,
            const index_t *ldA, double *B, const index_t *ldB, index_t *info) {
    dtrtrs(uplo, trans, diag, n, nrhs, A, ldA, B, ldB, info);
}

template <>
void xtrtrs(const char *uplo, const char *trans, const char *diag,
            const index_t *n, const index_t *nrhs, const float *A,
            const index_t *ldA, float *B, const index_t *ldB, index_t *info) {
    strtrs(uplo, trans, diag, n, nrhs, A, ldA, B, ldB, info);
}

#if DO_INSTANTIATE
template KOQKATOO_LINALG_EXPORT void
xtrtrs<real_t, index_t>(const char *uplo, const char *trans, const char *diag,
                        const index_t *n, const index_t *nrhs, const real_t *A,
                        const index_t *ldA, real_t *B, const index_t *ldB,
                        index_t *info);
#endif

template <>
void xscal(index_t N, double alpha, double *X, index_t incX) {
    cblas_dscal(N, alpha, X, incX);
}

template <>
void xscal(index_t N, float alpha, float *X, index_t incX) {
    cblas_sscal(N, alpha, X, incX);
}

#if DO_INSTANTIATE
template KOQKATOO_LINALG_EXPORT void
xscal<real_t, index_t>(index_t N, real_t alpha, real_t *X, index_t incX);
#endif

template <>
void KOQKATOO_LINALG_EXPORT xpotrf(const char *uplo, index_t n, double *a,
                                   index_t lda, index_t *info) {
    dpotrf(uplo, &n, a, &lda, info);
}
template <>
void KOQKATOO_LINALG_EXPORT xpotrf(const char *uplo, index_t n, float *a,
                                   index_t lda, index_t *info) {
    spotrf(uplo, &n, a, &lda, info);
}
#if DO_INSTANTIATE
template void KOQKATOO_LINALG_EXPORT xpotrf<real_t, index_t>(const char *uplo,
                                                             const index_t *n,
                                                             real_t *a,
                                                             const index_t *lda,
                                                             index_t *info);
#endif

template <>
void KOQKATOO_LINALG_EXPORT xtrtri(const char *uplo, const char *diag,
                                   index_t n, double *a, index_t lda,
                                   index_t *info) {
    dtrtri(uplo, diag, &n, a, &lda, info);
}
template <>
void KOQKATOO_LINALG_EXPORT xtrtri(const char *uplo, const char *diag,
                                   index_t n, float *a, index_t lda,
                                   index_t *info) {
    strtri(uplo, diag, &n, a, &lda, info);
}
#if DO_INSTANTIATE
template void KOQKATOO_LINALG_EXPORT
xtrtri<real_t, index_t>(const char *uplo, const char *diag, const index_t *n,
                        real_t *a, const index_t *lda, index_t *info);
#endif

template <class T, class I>
void xgemv_batch_strided(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, I m, I n,
                         T alpha, const T *a, I lda, I stridea, const T *x,
                         I incx, I stridex, T beta, T *y, I incy, I stridey,
                         I batch_size) {
    KOQKATOO_OMP(parallel for)
    for (I i = 0; i < batch_size; ++i) {
        auto offset_a = i * stridea;
        auto offset_x = i * stridex;
        auto offset_y = i * stridey;
        xgemv(layout, trans, m, n, alpha, a + offset_a, lda, x + offset_x, incx,
              beta, y + offset_y, incy);
    }
}

#if KOQKATOO_WITH_MKL
template <>
void KOQKATOO_LINALG_EXPORT xgemv_batch_strided(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, index_t m, index_t n,
    double alpha, const double *a, index_t lda, index_t stridea,
    const double *x, index_t incx, index_t stridex, double beta, double *y,
    index_t incy, index_t stridey, index_t batch_size) {
    cblas_dgemv_batch_strided(layout, trans, m, n, alpha, a, lda, stridea, x,
                              incx, stridex, beta, y, incy, stridey,
                              batch_size);
}

template <>
void KOQKATOO_LINALG_EXPORT xgemv_batch_strided(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, index_t m, index_t n,
    float alpha, const float *a, index_t lda, index_t stridea, const float *x,
    index_t incx, index_t stridex, float beta, float *y, index_t incy,
    index_t stridey, index_t batch_size) {
    cblas_sgemv_batch_strided(layout, trans, m, n, alpha, a, lda, stridea, x,
                              incx, stridex, beta, y, incy, stridey,
                              batch_size);
}
#endif

template void KOQKATOO_LINALG_EXPORT xgemv_batch_strided<real_t, index_t>(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, index_t m, index_t n,
    real_t alpha, const real_t *a, index_t lda, index_t stridea,
    const real_t *x, index_t incx, index_t stridex, real_t beta, real_t *y,
    index_t incy, index_t stridey, index_t batch_size);

template <class T, class I>
void xgemm_batch_strided(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
                         CBLAS_TRANSPOSE TransB, I M, I N, I K, T alpha,
                         const T *A, I lda, I stridea, const T *B, I ldb,
                         I strideb, T beta, T *C, I ldc, I stridec,
                         I batch_size) {
    KOQKATOO_OMP(parallel for)
    for (I i = 0; i < batch_size; ++i) {
        xgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
        A += stridea;
        B += strideb;
        C += stridec;
    }
}

#if KOQKATOO_WITH_MKL
template <>
void KOQKATOO_LINALG_EXPORT xgemm_batch_strided(
    CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    index_t M, index_t N, index_t K, double alpha, const double *A, index_t lda,
    index_t stridea, const double *B, index_t ldb, index_t strideb, double beta,
    double *C, index_t ldc, index_t stridec, index_t batch_size) {
    cblas_dgemm_batch_strided(Layout, TransA, TransB, M, N, K, alpha, A, lda,
                              stridea, B, ldb, strideb, beta, C, ldc, stridec,
                              batch_size);
}

template <>
void KOQKATOO_LINALG_EXPORT xgemm_batch_strided(
    CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    index_t M, index_t N, index_t K, float alpha, const float *A, index_t lda,
    index_t stridea, const float *B, index_t ldb, index_t strideb, float beta,
    float *C, index_t ldc, index_t stridec, index_t batch_size) {
    cblas_sgemm_batch_strided(Layout, TransA, TransB, M, N, K, alpha, A, lda,
                              stridea, B, ldb, strideb, beta, C, ldc, stridec,
                              batch_size);
}
#endif

template void KOQKATOO_LINALG_EXPORT xgemm_batch_strided<real_t, index_t>(
    CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    index_t M, index_t N, index_t K, real_t alpha, const real_t *A, index_t lda,
    index_t stridea, const real_t *B, index_t ldb, index_t strideb, real_t beta,
    real_t *C, index_t ldc, index_t stridec, index_t batch_size);

template <class T, class I>
void xsyrk_batch_strided(CBLAS_LAYOUT Layout, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE Trans, I N, I K, T alpha, const T *A,
                         I lda, I stridea, T beta, T *C, I ldc, I stridec,
                         I batch_size) {
    KOQKATOO_OMP(parallel for)
    for (I i = 0; i < batch_size; ++i) {
        xsyrk(Layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
        A += stridea;
        C += stridec;
    }
}

#if KOQKATOO_WITH_MKL
template <>
void KOQKATOO_LINALG_EXPORT xsyrk_batch_strided(
    CBLAS_LAYOUT Layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, index_t N,
    index_t K, double alpha, const double *A, index_t lda, index_t stridea,
    double beta, double *C, index_t ldc, index_t stridec, index_t batch_size) {
    cblas_dsyrk_batch_strided(Layout, Uplo, Trans, N, K, alpha, A, lda, stridea,
                              beta, C, ldc, stridec, batch_size);
}

template <>
void KOQKATOO_LINALG_EXPORT xsyrk_batch_strided(
    CBLAS_LAYOUT Layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, index_t N,
    index_t K, float alpha, const float *A, index_t lda, index_t stridea,
    float beta, float *C, index_t ldc, index_t stridec, index_t batch_size) {
    cblas_ssyrk_batch_strided(Layout, Uplo, Trans, N, K, alpha, A, lda, stridea,
                              beta, C, ldc, stridec, batch_size);
}
#endif

template void KOQKATOO_LINALG_EXPORT xsyrk_batch_strided<real_t, index_t>(
    CBLAS_LAYOUT Layout, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, index_t N,
    index_t K, real_t alpha, const real_t *A, index_t lda, index_t stridea,
    real_t beta, real_t *C, index_t ldc, index_t stridec, index_t batch_size);

template <class T, class I>
void xtrsm_batch_strided(CBLAS_LAYOUT Layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                         CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, I M, I N,
                         T alpha, const T *A, I lda, I stridea, T *B, I ldb,
                         I strideb, I batch_size) {
    KOQKATOO_OMP(parallel for)
    for (I i = 0; i < batch_size; ++i) {
        xtrsm(Layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
        A += stridea;
        B += strideb;
    }
}

#if KOQKATOO_WITH_MKL
template <>
void KOQKATOO_LINALG_EXPORT xtrsm_batch_strided(
    CBLAS_LAYOUT Layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
    CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, index_t M, index_t N, double alpha,
    const double *A, index_t lda, index_t stridea, double *B, index_t ldb,
    index_t strideb, index_t batch_size) {
    cblas_dtrsm_batch_strided(Layout, Side, Uplo, TransA, Diag, M, N, alpha, A,
                              lda, stridea, B, ldb, strideb, batch_size);
}

template <>
void KOQKATOO_LINALG_EXPORT xtrsm_batch_strided(
    CBLAS_LAYOUT Layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
    CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, index_t M, index_t N, float alpha,
    const float *A, index_t lda, index_t stridea, float *B, index_t ldb,
    index_t strideb, index_t batch_size) {
    cblas_strsm_batch_strided(Layout, Side, Uplo, TransA, Diag, M, N, alpha, A,
                              lda, stridea, B, ldb, strideb, batch_size);
}
#endif

template void KOQKATOO_LINALG_EXPORT xtrsm_batch_strided<real_t, index_t>(
    CBLAS_LAYOUT Layout, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
    CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, index_t M, index_t N, real_t alpha,
    const real_t *A, index_t lda, index_t stridea, real_t *B, index_t ldb,
    index_t strideb, index_t batch_size);

template <class T, class I>
void xpotrf_batch_strided(const char *Uplo, I N, T *A, I lda, I stridea,
                          I batch_size) {
    I info_all = 0;
    KOQKATOO_OMP(parallel for reduction(+:info_all))
    for (I i = 0; i < batch_size; ++i) {
        I info   = 0;
        I offset = i * stridea;
        xpotrf(Uplo, N, A + offset, lda, &info);
        if (info > 0)
            info = 0; // Ignore factorization failure
        info_all += info;
    }
    // TODO: proper error handling
    lapack_throw_on_err("xpotrf_batch_strided", info_all);
}

template KOQKATOO_LINALG_EXPORT void
xpotrf_batch_strided<real_t, index_t>(const char *Uplo, index_t N, real_t *A,
                                      index_t lda, index_t stridea,
                                      index_t batch_size);

} // namespace koqkatoo::linalg
