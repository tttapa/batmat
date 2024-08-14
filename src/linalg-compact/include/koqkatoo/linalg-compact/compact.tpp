#pragma once

#include <koqkatoo/linalg-compact/compact.hpp>

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/mkl.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/lapack.hpp>
#include <koqkatoo/openmp.h>

#include "compact/xgemm.tpp"
#include "compact/xpotrf.tpp"
#include "compact/xsyrk.tpp"
#include "compact/xtrsm.tpp"

namespace koqkatoo::linalg::compact {

template <class Abi>
void CompactBLAS<Abi>::xtrmv_ref(single_batch_view L, mut_single_batch_view x) {
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
void CompactBLAS<Abi>::xtrmv(batch_view L, mut_batch_view x,
                             PreferredBackend b) {
    std::ignore = b; // TODO
    assert(L.ceil_depth() == x.ceil_depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrmv_ref(L.batch(i), x.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrmv(single_batch_view L, mut_single_batch_view x,
                             PreferredBackend b) {
    std::ignore = b; // TODO
    xtrmv_ref(L, x);
}

template <class Abi>
void CompactBLAS<Abi>::xtrti_ref(mut_single_batch_view L) {
    assert(L.rows() == L.cols());
    const index_t n = L.rows();
    for (index_t j = n; j-- > 0;) {
        auto Ljj = simd{1} / aligned_load(&L(0, j, j));
        aligned_store(&L(0, j, j), Ljj);
        Ljj = -Ljj;
        if (j + 1 < n) {
            xtrmv_ref(L.bottom_right(n - j - 1, n - j - 1),
                      L.bottom_rows(n - j - 1).middle_cols(j, 1));
            KOQKATOO_UNROLLED_IVDEP_FOR (4, index_t i = j + 1; i < n; ++i)
                aligned_store(&L(0, i, j), aligned_load(&L(0, i, j)) * Ljj);
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xtrti(mut_batch_view L, PreferredBackend b) {
    std::ignore = b; // TODO
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrti_ref(L.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrti(mut_single_batch_view L, PreferredBackend b) {
    std::ignore = b; // TODO
    xtrti_ref(L);
}

template <class Abi>
void CompactBLAS<Abi>::xsymv_add_ref(single_batch_view L, single_batch_view x,
                                     mut_single_batch_view y) {
    assert(L.cols() == L.rows());
    assert(x.rows() == L.rows());
    assert(y.rows() == L.rows());
    assert(x.cols() == 1);
    assert(y.cols() == 1);
    const auto n = L.rows(), m = L.cols();
    for (index_t j = 0; j < m; ++j) {
        auto xj = aligned_load(&x(0, j, 0));
        simd t{0};
        auto Ljj = aligned_load(&L(0, j, j));
        aligned_store(&y(0, j, 0), fma(xj, Ljj, aligned_load(&y(0, j, 0))));
        KOQKATOO_UNROLLED_IVDEP_FOR (4, index_t i = j + 1; i < n; ++i) {
            auto Lij = aligned_load(&L(0, i, j));
            auto xi  = aligned_load(&x(0, i, 0));
            aligned_store(&y(0, i, 0), fma(xj, Lij, aligned_load(&y(0, i, 0))));
            t = fma(Lij, xi, t);
        }
        aligned_store(&y(0, j, 0), aligned_load(&y(0, j, 0)) + t);
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
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    const index_t n = A.rows(), m = A.cols();
    for (index_t j = 0; j < m; ++j)
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < n; ++i)
            aligned_store(&B(0, i, j), aligned_load(&A(0, i, j)));
}

template <class Abi>
void CompactBLAS<Abi>::xcopy(batch_view A, mut_batch_view B) {
    assert(A.ceil_depth() == B.ceil_depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xcopy(A.batch(i), B.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xaxpy(real_t a, single_batch_view x,
                             mut_single_batch_view y) {
    assert(x.rows() == y.rows());
    assert(x.cols() == y.cols());
    simd a_simd{a};
    const index_t n = x.rows();
    for (index_t j = 0; j < x.cols(); ++j) {
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < n; ++i) {
            aligned_store(&y(0, i, j), fma(a_simd, aligned_load(&x(0, i, j)),
                                           aligned_load(&y(0, i, j))));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xaxpy(real_t a, batch_view x, mut_batch_view y) {
    assert(x.rows() == y.rows());
    assert(x.cols() == y.cols());
    assert(x.cols() == 1);
    simd a_simd{a};
    index_t i;
    const auto Bs   = static_cast<index_t>(x.batch_size());
    const index_t n = x.rows();
    KOQKATOO_OMP(parallel for lastprivate(i))
    for (i = 0; i <= x.depth() - Bs; i += Bs) {
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t r = 0; r < n; ++r) {
            aligned_store(&y(i, r, 0), fma(a_simd, aligned_load(&x(i, r, 0)),
                                           aligned_load(&y(i, r, 0))));
        }
    }
    for (; i < x.depth(); ++i)
        for (index_t r = 0; r < n; ++r)
            y(i, r, 0) += a * x(i, r, 0);
}

template <class Abi>
template <class... Views>
void CompactBLAS<Abi>::xadd_copy_impl(mut_batch_view out, batch_view x1,
                                      Views... xs)
    requires(std::same_as<Views, batch_view> && ...)
{
    assert(((x1.batch_size() == xs.batch_size()) && ...));
    assert(x1.batch_size() == out.batch_size());
    assert(((x1.depth() == xs.depth()) && ...));
    assert(x1.depth() == out.depth());
    assert(((x1.rows() == xs.rows()) && ...));
    assert(x1.rows() == out.rows());
    assert(((x1.cols() == xs.cols()) && ...));
    assert(x1.cols() == out.cols());
    assert(x1.cols() == 1);
    index_t i;
    const auto Bs   = static_cast<index_t>(x1.batch_size());
    const index_t n = x1.rows();
    KOQKATOO_OMP(parallel for lastprivate(i))
    for (i = 0; i <= x1.depth() - Bs; i += Bs) {
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t r = 0; r < n; ++r) {
            aligned_store(&out(i, r, 0), (aligned_load(&x1(i, r, 0)) + ... +
                                          aligned_load(&xs(i, r, 0))));
        }
    }
    for (; i < x1.depth(); ++i)
        for (index_t r = 0; r < n; ++r)
            out(i, r, 0) = (x1(i, r, 0) + ... + xs(i, r, 0));
}

template <class Abi>
template <class... Views>
void CompactBLAS<Abi>::xsub_copy_impl(mut_batch_view out, batch_view x1,
                                      Views... xs)
    requires(std::same_as<Views, batch_view> && ...)
{
    assert(((x1.batch_size() == xs.batch_size()) && ...));
    assert(x1.batch_size() == out.batch_size());
    assert(((x1.depth() == xs.depth()) && ...));
    assert(x1.depth() == out.depth());
    assert(((x1.rows() == xs.rows()) && ...));
    assert(x1.rows() == out.rows());
    assert(((x1.cols() == xs.cols()) && ...));
    assert(x1.cols() == out.cols());
    assert(x1.cols() == 1);
    index_t i;
    const auto Bs   = static_cast<index_t>(x1.batch_size());
    const index_t n = x1.rows();
    KOQKATOO_OMP(parallel for lastprivate(i))
    for (i = 0; i <= x1.depth() - Bs; i += Bs) {
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t r = 0; r < n; ++r) {
            aligned_store(&out(i, r, 0),
                          aligned_load(&x1(i, r, 0)) -
                              (... + aligned_load(&xs(i, r, 0))));
        }
    }
    for (; i < x1.depth(); ++i)
        for (index_t r = 0; r < n; ++r)
            out(i, r, 0) = x1(i, r, 0) - (... + xs(i, r, 0));
}

template <class Abi>
real_t CompactBLAS<Abi>::xdot(batch_view x, batch_view y) {
    using std::fma;
    assert(x.depth() == y.depth());
    assert(x.batch_size() == y.batch_size());
    assert(x.rows() == y.rows());
    assert(x.cols() == y.cols());
    assert(x.cols() == 1);
    simd dot_simd{0};
    index_t i;
    const auto Bs   = static_cast<index_t>(x.batch_size());
    const index_t n = x.rows();
    // KOQKATOO_OMP(parallel for reduction(+:dot_simd) lastprivate(i)) // TODO
    for (i = 0; i <= x.depth() - Bs; i += Bs) {
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t r = 0; r < n; ++r) {
            dot_simd = fma(aligned_load(&x(i, r, 0)), aligned_load(&y(i, r, 0)),
                           dot_simd);
        }
    }
    real_t dot_scal{0};
    for (; i < x.depth(); ++i)
        for (index_t r = 0; r < n; ++r)
            dot_scal = fma(x(i, r, 0), y(i, r, 0), dot_scal);
    return reduce(dot_simd) + dot_scal;
}

template <class Abi>
real_t CompactBLAS<Abi>::xnrm2sq(batch_view x) {
    using std::fma;
    assert(x.cols() == 1);
    simd dot_simd{0};
    index_t i;
    const auto Bs   = static_cast<index_t>(x.batch_size());
    const index_t n = x.rows();
    // KOQKATOO_OMP(parallel for reduction(+:dot_simd) lastprivate(i)) // TODO
    for (i = 0; i <= x.depth() - Bs; i += Bs) {
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t r = 0; r < n; ++r) {
            auto xir = aligned_load(&x(i, r, 0));
            dot_simd = fma(xir, xir, dot_simd);
        }
    }
    real_t dot_scal{0};
    for (; i < x.depth(); ++i)
        for (index_t r = 0; r < n; ++r)
            dot_scal = fma(x(i, r, 0), x(i, r, 0), dot_scal);
    return reduce(dot_simd) + dot_scal;
}

template <class Abi>
real_t CompactBLAS<Abi>::xnrminf(batch_view x) {
    using std::abs;
    using std::fma;
    using std::isfinite;
    using std::max;
    assert(x.cols() == 1);
    simd nrm_simd{0}, l1_nrm_simd{0};
    index_t i;
    const auto Bs   = static_cast<index_t>(x.batch_size());
    const index_t n = x.rows();
    // KOQKATOO_OMP(parallel for reduction(max:nrm_simd) reduction(+:l1_nrm_simd) lastprivate(i)) // TODO
    for (i = 0; i <= x.depth() - Bs; i += Bs) {
        KOQKATOO_UNROLLED_IVDEP_FOR (8, index_t r = 0; r < n; ++r) {
            auto axir = abs(aligned_load(&x(i, r, 0)));
            nrm_simd  = max(axir, nrm_simd);
            l1_nrm_simd += axir;
        }
    }
    real_t nrm_scal = hmax(nrm_simd), l1_nrm_scal = reduce(l1_nrm_simd);
    for (; i < x.depth(); ++i)
        for (index_t r = 0; r < n; ++r) {
            auto axir = abs(x(i, r, 0));
            nrm_scal  = max(axir, nrm_scal);
            l1_nrm_scal += axir;
        }
    return isfinite(l1_nrm_scal) ? nrm_scal : l1_nrm_scal;
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

template <>
void CompactBLAS<stdx::simd_abi::scalar>::xtrti(mut_single_batch_view H,
                                                PreferredBackend b) {
    std::ignore = b; // TODO
    assert(H.rows() == H.cols());
    index_t info = 0;
    index_t n = H.rows(), ldh = H.outer_stride();
    linalg::xtrtri("L", "N", &n, H.data, &ldh, &info);
    lapack_throw_on_err("xtrtri", info);
}

} // namespace koqkatoo::linalg::compact
