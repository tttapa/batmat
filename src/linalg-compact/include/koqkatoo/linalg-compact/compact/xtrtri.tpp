#pragma once

#include <koqkatoo/kib.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/lapack.hpp>
#include <koqkatoo/loop.hpp>

#include <cmath>
#include <concepts>

#include <koqkatoo/linalg-compact/compact/micro-kernels/xtrtri.hpp>
#include "util.hpp"

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xtrtri_ref(mut_single_batch_view L) {
    using namespace micro_kernels;
    assert(L.rows() >= L.cols());
    const index_t m = L.rows(), n = L.cols();
    [[maybe_unused]] const auto //
        op_cnt_trtri = n * (n - 1) * (n - 2) / 6 + n * (n - 1) + n,
        op_cnt_trmm  = n * (n - 1) * (m - n) / 2 + n * (m - n);
    KOQKATOO_TRACE("xtrtri", 0, (op_cnt_trtri + op_cnt_trmm) * L.depth());
    static constexpr index_t R               = micro_kernels::trtri::RowsReg;
    mut_single_batch_matrix_accessor<Abi> L_ = L;
    foreach_chunked(
        0, n, R,
        [&](index_t j) {
            trtri::xtrtri_trmm_microkernel<Abi, R>(L_.block(j, j), m - j);
            foreach_chunked(
                j + R, n, R,
                [&](index_t k) {
                    trtri::xtrmm_microkernel<Abi, R, R>(L_.block(k, k),
                                                        L_.block(k, j), m - k);
                },
                [&](index_t k, index_t nk) {
                    trtri::microkernel_trmm_lut<Abi>[nk - 1](
                        L_.block(k, k), L_.block(k, j), m - k);
                },
                LoopDir::Backward);
        },
        [&](index_t j, index_t nj) {
            trtri::microkernel_lut<Abi>[nj - 1](L_.block(j, j), m - j);
        },
        LoopDir::Backward);
}

template <class Abi>
void CompactBLAS<Abi>::xtrtri_copy_ref(single_batch_view Lin,
                                       mut_single_batch_view L) {
    using namespace micro_kernels;
    assert(Lin.rows() == L.rows());
    assert(Lin.cols() == L.cols());
    assert(L.rows() >= L.cols());
    const index_t m = L.rows(), n = L.cols();
    [[maybe_unused]] const auto //
        op_cnt_trtri = n * (n - 1) * (n - 2) / 6 + n * (n - 1) + n,
        op_cnt_trmm  = n * (n - 1) * (m - n) / 2 + n * (m - n);
    KOQKATOO_TRACE("xtrtri_copy", 0, (op_cnt_trtri + op_cnt_trmm) * L.depth());
    static constexpr index_t R               = micro_kernels::trtri::RowsReg;
    single_batch_matrix_accessor<Abi> Lin_   = Lin;
    mut_single_batch_matrix_accessor<Abi> L_ = L;
    foreach_chunked(
        0, n, R,
        [&](index_t j) {
            trtri::xtrtri_trmm_copy_microkernel<Abi, R>(Lin_.block(j, j),
                                                        L_.block(j, j), m - j);
            foreach_chunked(
                j + R, n, R,
                [&](index_t k) {
                    trtri::xtrmm_microkernel<Abi, R, R>(L_.block(k, k),
                                                        L_.block(k, j), m - k);
                },
                [&](index_t k, index_t nk) {
                    trtri::microkernel_trmm_lut<Abi>[nk - 1](
                        L_.block(k, k), L_.block(k, j), m - k);
                },
                LoopDir::Backward);
        },
        [&](index_t j, index_t nj) {
            trtri::microkernel_copy_lut<Abi>[nj - 1](Lin_.block(j, j),
                                                     L_.block(j, j), m - j);
        },
        LoopDir::Backward);
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xtrtri(mut_batch_view L, PreferredBackend b) {
    assert(L.rows() >= L.cols());
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < L.num_batches(); ++i)
            xtrtri(L.batch(i), b);
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrtri(L.batch(i), b);
}

template <class Abi>
void CompactBLAS<Abi>::xtrtri_copy(batch_view Lin, mut_batch_view L,
                                   PreferredBackend b) {
    assert(Lin.num_batches() == L.num_batches());
    assert(L.rows() >= L.cols());
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < L.num_batches(); ++i)
            xtrtri_copy(Lin.batch(i), L.batch(i), b);
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrtri_copy(Lin.batch(i), L.batch(i), b);
}

// BLAS specializations (scalar)
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xtrtri(mut_single_batch_view L, PreferredBackend b) {
    assert(L.rows() >= L.cols());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            index_t info;
            const auto m = L.rows(), n = L.cols();
            [[maybe_unused]] const auto //
                op_cnt_trtri = n * (n - 1) * (n - 2) / 6 + n * (n - 1) + n,
                op_cnt_trmm  = n * (n - 1) * (m - n) / 2 + n * (m - n);
            KOQKATOO_TRACE("xtrtri_blas", 0,
                           (op_cnt_trtri + op_cnt_trmm) * L.depth());
            linalg::xtrtri("L", "N", n, L.data, L.outer_stride(), &info);
            lapack_throw_on_err("xtrtri", info);
            if (m > n)
                linalg::xtrmm(CblasColMajor, CblasRight, CblasLower,
                              CblasNoTrans, CblasNonUnit, m - n, n, real_t(-1),
                              L.top_rows(n).data, L.outer_stride(),
                              L.bottom_rows(m - n).data, L.outer_stride());
            return;
        }
    }
    xtrtri_ref(L);
}

template <class Abi>
void CompactBLAS<Abi>::xtrtri_copy(single_batch_view Lin,
                                   mut_single_batch_view L,
                                   PreferredBackend b) {
    assert(L.rows() >= L.cols());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            index_t info;
            const auto m = L.rows(), n = L.cols();
            [[maybe_unused]] const auto //
                op_cnt_trtri = n * (n - 1) * (n - 2) / 6 + n * (n - 1) + n,
                op_cnt_trmm  = n * (n - 1) * (m - n) / 2 + n * (m - n);
            KOQKATOO_TRACE("xtrtri_copy_blas", 0,
                           (op_cnt_trtri + op_cnt_trmm) * L.depth());
            L = Lin;
            linalg::xtrtri("L", "N", n, L.data, L.outer_stride(), &info);
            lapack_throw_on_err("xtrtri", info);
            if (m > n)
                linalg::xtrmm(CblasColMajor, CblasRight, CblasLower,
                              CblasNoTrans, CblasNonUnit, m - n, n, real_t(-1),
                              L.top_rows(n).data, L.outer_stride(),
                              L.bottom_rows(m - n).data, L.outer_stride());
            return;
        }
    }
    xtrtri_copy_ref(Lin, L);
}

} // namespace koqkatoo::linalg::compact
