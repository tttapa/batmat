#pragma once

#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>

#include <koqkatoo/linalg-compact/compact-new/micro-kernels/xgemm.hpp>
#include "util.hpp"

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xsyrk_schur(single_batch_view A, single_batch_view d,
                                   mut_single_batch_view C) {
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    assert(d.rows() == A.cols());
    constexpr micro_kernels::gemm::KernelConfig conf{.trans_B = true};
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = j; i < C.rows(); ++i)
            simd{0}.copy_to(&C(0, i, j), stdx::vector_aligned);
    // TODO: cache blocking
    micro_kernels::gemm::xgemmt_diag_register<Abi, conf>(A, A, C, d);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_T_schur_copy(single_batch_view C,
                                          single_batch_view Σ,
                                          bool_single_batch_view mask,
                                          single_batch_view H_in,
                                          mut_single_batch_view H_out) {
    for (index_t j = 0; j < H_out.cols(); ++j)
        for (index_t i = j; i < H_out.rows(); ++i)
            aligned_store(&H_out(0, i, j), aligned_load(&H_in(0, i, j)));
    // TODO: cache blocking
    constexpr micro_kernels::gemm::KernelConfig conf{.trans_A = true};
    micro_kernels::gemm::xgemmt_diag_mask_register<Abi, conf>(C, C, H_out, Σ,
                                                              mask);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_T_ref(single_batch_view A,
                                   mut_single_batch_view C) {
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = j; i < C.rows(); ++i)
            aligned_store(&C(0, i, j), simd{0});
    // TODO: cache blocking
    micro_kernels::gemm::xgemmt_register<Abi, {.trans_A = true}>(A, A, C);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_ref(single_batch_view A, mut_single_batch_view C) {
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    for (index_t j = 0; j < C.cols(); ++j)
        for (index_t i = j; i < C.rows(); ++i)
            aligned_store(&C(0, i, j), simd{0});
    // TODO: cache blocking
    micro_kernels::gemm::xgemmt_register<Abi, {.trans_B = true}>(A, A, C);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_sub_ref(single_batch_view A,
                                     mut_single_batch_view C) {
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    // TODO: cache blocking
    constexpr micro_kernels::gemm::KernelConfig conf{.negate  = true,
                                                     .trans_B = true};
    micro_kernels::gemm::xgemmt_register<Abi, conf>(A, A, C);
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xsyrk_T(batch_view A, mut_batch_view C,
                               PreferredBackend b) {
    assert(A.ceil_depth() == C.ceil_depth());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            linalg::xsyrk_batch_strided(
                CblasColMajor, CblasLower, CblasTrans, C.rows(), A.rows(),
                real_t{1}, A.data, A.rows(), A.rows() * A.cols(), real_t{0},
                C.data, C.rows(), C.rows() * C.cols(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xsyrk_T_ref(A.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk(batch_view A, mut_batch_view C,
                             PreferredBackend b) {
    assert(A.ceil_depth() == C.ceil_depth());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            linalg::xsyrk_batch_strided(
                CblasColMajor, CblasLower, CblasNoTrans, C.rows(), A.cols(),
                real_t{1}, A.data, A.rows(), A.rows() * A.cols(), real_t{0},
                C.data, C.rows(), C.rows() * C.cols(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xsyrk_ref(A.batch(i), C.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_sub(batch_view A, mut_batch_view C,
                                 PreferredBackend b) {
    assert(A.ceil_depth() == C.ceil_depth());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            linalg::xsyrk_batch_strided(
                CblasColMajor, CblasLower, CblasNoTrans, C.rows(), A.cols(),
                real_t{-1}, A.data, A.rows(), A.rows() * A.cols(), real_t{1},
                C.data, C.rows(), C.rows() * C.cols(), A.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < A.num_batches(); ++i)
        xsyrk_sub_ref(A.batch(i), C.batch(i));
}

// BLAS specializations (scalar)
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xsyrk_T(single_batch_view A, mut_single_batch_view C,
                               PreferredBackend b) {
    assert(C.rows() == C.cols());
    assert(C.rows() == A.cols());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            linalg::xsyrk(CblasColMajor, CblasLower, CblasTrans, C.rows(),
                          A.rows(), real_t{1}, A.data, A.outer_stride(),
                          real_t{0}, C.data, C.outer_stride());
            return;
        }
    }
    xsyrk_T_ref(A, C);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk(single_batch_view A, mut_single_batch_view C,
                             PreferredBackend b) {
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            linalg::xsyrk(CblasColMajor, CblasLower, CblasNoTrans, C.rows(),
                          A.cols(), real_t{1}, A.data, A.outer_stride(),
                          real_t{0}, C.data, C.outer_stride());
            return;
        }
    }
    xsyrk_ref(A, C);
}

template <class Abi>
void CompactBLAS<Abi>::xsyrk_sub(single_batch_view A, mut_single_batch_view C,
                                 PreferredBackend b) {
    assert(C.rows() == C.cols());
    assert(C.rows() == A.rows());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            linalg::xsyrk(CblasColMajor, CblasLower, CblasNoTrans, C.rows(),
                          A.cols(), real_t{-1}, A.data, A.outer_stride(),
                          real_t{1}, C.data, C.outer_stride());
            return;
        }
    }
    xsyrk_sub_ref(A, C);
}

} // namespace koqkatoo::linalg::compact
