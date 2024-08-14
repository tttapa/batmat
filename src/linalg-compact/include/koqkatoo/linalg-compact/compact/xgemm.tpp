#pragma once

#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>

#include "micro_kernels/xgemm.tpp"
#include "util.hpp"

#include <concepts>

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

// TODO: pick optimal values depending on real_t and system's cache sizes
static constexpr index_t GemmBlockSizeRows = 32;
static constexpr index_t GemmBlockSizeCols = 48;

template <class Abi>
void CompactBLAS<Abi>::xgemm_ref(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = 0; i < C.rows(); ++i)
            simd{0}.copy_to(&C(0, i, j), stdx::vector_aligned);
    for (index_t l = 0; l < A.cols(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.cols() - l);
        for (index_t i = 0; i < A.rows(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.rows() - i);
            micro_kernels::gemm::xgemm_register<simd, {}>(A.block(i, l, ni, nl),
                                                          B.middle_rows(l, nl),
                                                          C.middle_rows(i, ni));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_neg_ref(single_batch_view A, single_batch_view B,
                                     mut_single_batch_view C) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = 0; i < C.rows(); ++i)
            simd{0}.copy_to(&C(0, i, j), stdx::vector_aligned);
    for (index_t l = 0; l < A.cols(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.cols() - l);
        for (index_t i = 0; i < A.rows(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.rows() - i);
            micro_kernels::gemm::xgemm_register<simd, {.negate = true}>(
                A.block(i, l, ni, nl), B.middle_rows(l, nl),
                C.middle_rows(i, ni));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_add_ref(single_batch_view A, single_batch_view B,
                                     mut_single_batch_view C) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate = false};
    using mat   = BatchedMatrix<real_t, index_t, simd_stride_t, simd_stride_t>;
    bool packed = B.outer_stride() * B.batch_size() * sizeof(real_t) > 4096;
    mat B_packed{{
        .rows = packed ? GemmBlockSizeCols : 0,
        .cols = B.cols(),
    }};
    for (index_t l = 0; l < A.cols(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.cols() - l);
        if (packed)
            xcopy(B.middle_rows(l, nl), B_packed.view.top_rows(nl));
        auto Bl = packed ? B_packed.view.top_rows(nl) : B.middle_rows(l, nl);
        for (index_t i = 0; i < A.rows(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.rows() - i);
            micro_kernels::gemm::xgemm_register<simd, conf>(
                A.block(i, l, ni, nl), Bl, C.middle_rows(i, ni));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_sub_ref(single_batch_view A, single_batch_view B,
                                     mut_single_batch_view C) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate = true};
    for (index_t l = 0; l < A.cols(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.cols() - l);
        for (index_t i = 0; i < A.rows(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.rows() - i);
            micro_kernels::gemm::xgemm_register<simd, conf>(
                A.block(i, l, ni, nl), B.middle_rows(l, nl),
                C.middle_rows(i, ni));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_sub_ref(single_batch_view A,
                                        single_batch_view B,
                                        mut_single_batch_view C) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate  = true,
                                                            .trans_A = true};
    for (index_t l = 0; l < A.rows(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.rows() - l);
        for (index_t i = 0; i < A.cols(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.cols() - i);
            micro_kernels::gemm::xgemm_register<simd, conf>(
                A.block(l, i, nl, ni), B.middle_rows(l, nl),
                C.middle_rows(i, ni));
        }
    }
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_ref(single_batch_view A, single_batch_view B,
                                    mut_single_batch_view C) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    static constexpr micro_kernels::gemm::KernelConfig conf{.negate  = false,
                                                            .trans_A = true};
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = 0; i < C.rows(); ++i)
            simd{0}.copy_to(&C(0, i, j), stdx::vector_aligned);
    for (index_t l = 0; l < A.rows(); l += GemmBlockSizeCols) {
        auto nl = std::min<index_t>(GemmBlockSizeCols, A.rows() - l);
        for (index_t i = 0; i < A.cols(); i += GemmBlockSizeRows) {
            auto ni = std::min<index_t>(GemmBlockSizeRows, A.cols() - i);
            micro_kernels::gemm::xgemm_register<simd, conf>(
                A.block(l, i, nl, ni), B.middle_rows(l, nl),
                C.middle_rows(i, ni));
        }
    }
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xgemm(batch_view A, batch_view B, mut_batch_view C,
                             PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.rows(), B.data, B.rows(),
                real_t{0}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{0}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_neg(batch_view A, batch_view B, mut_batch_view C,
                                 PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.rows(), B.data, B.rows(),
                real_t{0}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{0}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_neg_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_add(batch_view A, batch_view B, mut_batch_view C,
                                 PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.rows(), B.data, B.rows(),
                real_t{1}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{1}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_add_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_sub(batch_view A, batch_view B, mut_batch_view C,
                                 PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.rows(), B.data, B.rows(),
                real_t{1}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return linalg::xgemm_batch_strided(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{1}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_sub_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_sub(batch_view A, batch_view B,
                                    mut_batch_view C, PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_TRANS, MKL_NOTRANS, A.cols(), B.cols(),
                A.rows(), real_t{-1}, A.data, A.rows(), B.data, B.rows(),
                real_t{1}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return linalg::xgemm_batch_strided(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{-1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{1}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_TN_sub_ref(A.batch(i), B.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN(batch_view A, batch_view B, mut_batch_view C,
                                PreferredBackend b) {
    assert(A.depth() == B.depth());
    assert(B.depth() == C.depth());
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b))
            return xgemm_compact(
                MKL_COL_MAJOR, MKL_TRANS, MKL_NOTRANS, A.cols(), B.cols(),
                A.rows(), real_t{1}, A.data, A.rows(), B.data, B.rows(),
                real_t{0}, C.data, C.rows(),
                vector_format_mkl<real_t, Abi>::format, A.ceil_depth());
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b))
            return linalg::xgemm_batch_strided(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{1}, A.data, A.rows(), A.rows() * A.cols(),
                B.data, B.rows(), B.rows() * B.cols(), real_t{0}, C.data,
                C.rows(), C.rows() * C.cols(), A.depth());
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xgemm_TN_ref(A.batch(i), B.batch(i), C.batch(i));
}

// BLAS specializations (scalar)
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xgemm(single_batch_view A, single_batch_view B,
                             mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride());
    xgemm_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_neg(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride());
    xgemm_neg_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_add(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride());
    xgemm_add_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_sub(single_batch_view A, single_batch_view B,
                                 mut_single_batch_view C, PreferredBackend b) {
    assert(A.rows() == C.rows());
    assert(A.cols() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(),
                A.cols(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride());
    xgemm_sub_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN_sub(single_batch_view A, single_batch_view B,
                                    mut_single_batch_view C,
                                    PreferredBackend b) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{-1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{1}, C.data, C.outer_stride());
    xgemm_TN_sub_ref(A, B, C);
}

template <class Abi>
void CompactBLAS<Abi>::xgemm_TN(single_batch_view A, single_batch_view B,
                                mut_single_batch_view C, PreferredBackend b) {
    assert(A.cols() == C.rows());
    assert(A.rows() == B.rows());
    assert(B.cols() == C.cols());
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b))
            return linalg::xgemm(
                CblasColMajor, CblasTrans, CblasNoTrans, A.cols(), B.cols(),
                A.rows(), real_t{1}, A.data, A.outer_stride(), B.data,
                B.outer_stride(), real_t{0}, C.data, C.outer_stride());
    xgemm_TN_ref(A, B, C);
}

} // namespace koqkatoo::linalg::compact
