#pragma once

#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/trace.hpp>

#include <koqkatoo/linalg-compact/compact/micro-kernels/xtrsm.hpp>
#include "util.hpp"

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xtrsm_RLTN_ref(single_batch_view L,
                                      mut_single_batch_view H) {
    [[maybe_unused]] const auto op_cnt_trsm =
        L.rows() * (L.rows() + 1) * H.rows() / 2 + L.rows();
    KOQKATOO_TRACE("xtrsm_RLTN", 0, op_cnt_trsm * L.depth());
    assert(L.rows() == L.cols());
    assert(H.cols() == L.rows());
    micro_kernels::trsm::xtrsm_register<Abi, {.trans = true}>(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLNN_ref(single_batch_view L,
                                      mut_single_batch_view H) {
    [[maybe_unused]] const auto op_cnt_trsm =
        L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
    KOQKATOO_TRACE("xtrsm_LLNN", 0, op_cnt_trsm * L.depth());
    assert(L.rows() == L.cols());
    assert(L.rows() == H.rows());
    micro_kernels::trsm::xtrsm_llnn_register<Abi>(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsv_LNN_ref(single_batch_view L,
                                     mut_single_batch_view H) {
    [[maybe_unused]] const auto op_cnt_trsm =
        L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
    KOQKATOO_TRACE("xtrsv_LNN", 0, op_cnt_trsm * L.depth());
    assert(L.rows() == L.cols());
    assert(L.rows() == H.rows());
    assert(H.cols() == 1);
    micro_kernels::trsm::xtrsm_llnn_register<Abi>(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLTN_ref(single_batch_view L,
                                      mut_single_batch_view H) {
    [[maybe_unused]] const auto op_cnt_trsm =
        L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
    KOQKATOO_TRACE("xtrsm_LLTN", 0, op_cnt_trsm * L.depth());
    assert(L.rows() == L.cols());
    assert(L.rows() == H.rows());
    micro_kernels::trsm::xtrsm_lltn_register<Abi>(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsv_LTN_ref(single_batch_view L,
                                     mut_single_batch_view H) {
    [[maybe_unused]] const auto op_cnt_trsm =
        L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
    KOQKATOO_TRACE("xtrsv_LTN", 0, op_cnt_trsm * L.depth());
    assert(L.rows() == L.cols());
    assert(L.rows() == H.rows());
    assert(H.cols() == 1);
    const auto n = L.rows();
    micro_kernels::trsm::xtrsm_lltn_register<Abi>(L, H);
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xtrsm_RLTN(batch_view L, mut_batch_view H,
                                  PreferredBackend b) {
    assert(L.depth() == H.depth());
    assert(L.rows() == L.cols());
    assert(H.cols() == L.rows());
    if (H.rows() == 0 || H.cols() == 0)
        return;
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && L.has_full_layer_stride() &&
            H.has_full_layer_stride()) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.rows() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsm_RLTN_mkl_compact", 0,
                           op_cnt_trsm * L.depth());
            return xtrsm_compact(
                MKL_COL_MAJOR, MKL_RIGHT, MKL_LOWER, MKL_TRANS, MKL_NONUNIT,
                H.rows(), H.cols(), real_t{1}, L.data, L.outer_stride(), H.data,
                H.outer_stride(), vector_format_mkl<real_t, Abi>::format,
                L.ceil_depth());
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.rows() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsm_RLTN_batched", 0, op_cnt_trsm * L.depth());
            return xtrsm_batch_strided(
                CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                H.rows(), H.cols(), real_t{1}, L.data, L.outer_stride(),
                L.layer_stride(), H.data, H.outer_stride(), H.layer_stride(),
                L.depth());
        }
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < L.num_batches(); ++i)
            xtrsm_RLTN_ref(L.batch(i), H.batch(i));
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrsm_RLTN_ref(L.batch(i), H.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLNN(batch_view L, mut_batch_view H,
                                  PreferredBackend b) {
    assert(L.depth() == H.depth());
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    if (H.rows() == 0 || H.cols() == 0)
        return;
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && L.has_full_layer_stride() &&
            H.has_full_layer_stride()) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsm_LLNN_mkl_compact", 0,
                           op_cnt_trsm * L.depth());
            return xtrsm_compact(
                MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_NOTRANS, MKL_NONUNIT,
                H.rows(), H.cols(), real_t{1}, L.data, L.outer_stride(), H.data,
                H.outer_stride(), vector_format_mkl<real_t, Abi>::format,
                L.ceil_depth());
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsm_LLNN_batched", 0, op_cnt_trsm * L.depth());
            return xtrsm_batch_strided(
                CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, H.rows(), H.cols(), real_t{1}, L.data,
                L.outer_stride(), L.layer_stride(), H.data, H.outer_stride(),
                H.layer_stride(), L.depth());
        }
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < L.num_batches(); ++i)
            xtrsm_LLNN_ref(L.batch(i), H.batch(i));
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrsm_LLNN_ref(L.batch(i), H.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrsv_LNN(batch_view L, mut_batch_view H,
                                 PreferredBackend b) {
    assert(L.depth() == H.depth());
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    assert(H.cols() == 1);
    if (H.rows() == 0)
        return;
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && L.has_full_layer_stride() &&
            H.has_full_layer_stride()) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsv_LNN_mkl_compact", 0, op_cnt_trsm * L.depth());
            return xtrsm_compact(
                MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_NOTRANS, MKL_NONUNIT,
                H.rows(), H.cols(), real_t{1}, L.data, L.outer_stride(), H.data,
                H.outer_stride(), vector_format_mkl<real_t, Abi>::format,
                L.ceil_depth());
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsv_LNN_batched", 0, op_cnt_trsm * L.depth());
            return xtrsm_batch_strided(
                CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, H.rows(), H.cols(), real_t{1}, L.data,
                L.outer_stride(), L.layer_stride(), H.data, H.outer_stride(),
                H.layer_stride(), L.depth());
        }
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < L.num_batches(); ++i)
            xtrsv_LNN_ref(L.batch(i), H.batch(i));
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrsv_LNN_ref(L.batch(i), H.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLTN(batch_view L, mut_batch_view H,
                                  PreferredBackend b) {
    assert(L.depth() == H.depth());
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    if (H.rows() == 0 || H.cols() == 0)
        return;
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && L.has_full_layer_stride() &&
            H.has_full_layer_stride()) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsm_LLTN_mkl_compact", 0,
                           op_cnt_trsm * L.depth());
            return xtrsm_compact(
                MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_TRANS, MKL_NONUNIT,
                H.rows(), H.cols(), real_t{1}, L.data, L.outer_stride(), H.data,
                H.outer_stride(), vector_format_mkl<real_t, Abi>::format,
                L.ceil_depth());
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsm_LLTN_batched", 0, op_cnt_trsm * L.depth());
            return xtrsm_batch_strided(
                CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                H.rows(), H.cols(), real_t{1}, L.data, L.outer_stride(),
                L.layer_stride(), H.data, H.outer_stride(), H.layer_stride(),
                L.depth());
        }
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < L.num_batches(); ++i)
            xtrsm_LLTN_ref(L.batch(i), H.batch(i));
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrsm_LLTN_ref(L.batch(i), H.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xtrsv_LTN(batch_view L, mut_batch_view H,
                                 PreferredBackend b) {
    assert(L.depth() == H.depth());
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    assert(H.cols() == 1);
    if (H.rows() == 0)
        return;
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && L.has_full_layer_stride() &&
            H.has_full_layer_stride()) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsv_LTN_mkl_compact", 0, op_cnt_trsm * L.depth());
            return xtrsm_compact(
                MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_TRANS, MKL_NONUNIT,
                H.rows(), H.cols(), real_t{1}, L.data, L.outer_stride(), H.data,
                H.outer_stride(), vector_format_mkl<real_t, Abi>::format,
                L.ceil_depth());
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_mkl_batched(b)) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsv_LTN_batched", 0, op_cnt_trsm * L.depth());
            return xtrsm_batch_strided(
                CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                H.rows(), H.cols(), real_t{1}, L.data, L.outer_stride(),
                L.layer_stride(), H.data, H.outer_stride(), H.layer_stride(),
                L.depth());
        }
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < L.num_batches(); ++i)
            xtrsv_LTN_ref(L.batch(i), H.batch(i));
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < L.num_batches(); ++i)
        xtrsv_LTN_ref(L.batch(i), H.batch(i));
}

// BLAS specializations (scalar)
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xtrsm_RLTN(single_batch_view L, mut_single_batch_view H,
                                  PreferredBackend b) {
    assert(L.rows() == L.cols());
    assert(H.cols() == L.rows());
    if (H.rows() == 0 || H.cols() == 0)
        return;
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.rows() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsm_RLTN_blas", 0, op_cnt_trsm * L.depth());
            return linalg::xtrsm(CblasColMajor, CblasRight, CblasLower,
                                 CblasTrans, CblasNonUnit, H.rows(), H.cols(),
                                 real_t{1}, L.data, L.outer_stride(), H.data,
                                 H.outer_stride());
        }
    xtrsm_RLTN_ref(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLNN(single_batch_view L, mut_single_batch_view H,
                                  PreferredBackend b) {
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    if (H.rows() == 0 || H.cols() == 0)
        return;
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsm_LLNN_blas", 0, op_cnt_trsm * L.depth());
            return linalg::xtrsm(CblasColMajor, CblasLeft, CblasLower,
                                 CblasNoTrans, CblasNonUnit, H.rows(), H.cols(),
                                 real_t{1}, L.data, L.outer_stride(), H.data,
                                 H.outer_stride());
        }
    xtrsm_LLNN_ref(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsv_LNN(single_batch_view L, mut_single_batch_view h,
                                 PreferredBackend b) {
    assert(L.rows() == L.cols());
    assert(h.rows() == L.rows());
    assert(h.cols() == 1);
    if (h.rows() == 0)
        return;
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * h.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsv_LNN_blas", 0, op_cnt_trsm * L.depth());
            return linalg::xtrsv(CblasColMajor, CblasLower, CblasNoTrans,
                                 CblasNonUnit, h.rows(), L.data,
                                 L.outer_stride(), h.data, index_t{1});
        }
    xtrsv_LNN_ref(L, h);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsm_LLTN(single_batch_view L, mut_single_batch_view H,
                                  PreferredBackend b) {
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    if (H.rows() == 0 || H.cols() == 0)
        return;
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsm_LLTN_blas", 0, op_cnt_trsm * L.depth());
            return linalg::xtrsm(CblasColMajor, CblasLeft, CblasLower,
                                 CblasTrans, CblasNonUnit, H.rows(), H.cols(),
                                 real_t{1}, L.data, L.outer_stride(), H.data,
                                 H.outer_stride());
        }
    xtrsm_LLTN_ref(L, H);
}

template <class Abi>
void CompactBLAS<Abi>::xtrsv_LTN(single_batch_view L, mut_single_batch_view H,
                                 PreferredBackend b) {
    assert(L.rows() == L.cols());
    assert(H.rows() == L.rows());
    assert(H.cols() == 1);
    if (H.rows() == 0)
        return;
    if constexpr (std::same_as<Abi, scalar_abi>)
        if (use_blas_scalar(b)) {
            [[maybe_unused]] const auto op_cnt_trsm =
                L.rows() * (L.rows() + 1) * H.cols() / 2 + L.rows();
            KOQKATOO_TRACE("xtrsv_LTN_blas", 0, op_cnt_trsm * L.depth());
            return linalg::xtrsv(CblasColMajor, CblasLower, CblasTrans,
                                 CblasNonUnit, H.rows(), L.data,
                                 L.outer_stride(), H.data, index_t{1});
        }
    xtrsv_LTN_ref(L, H);
}

} // namespace koqkatoo::linalg::compact
