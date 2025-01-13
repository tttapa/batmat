#pragma once

#include <koqkatoo/kib.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/lapack.hpp>
#include <koqkatoo/loop.hpp>

#include <cmath>
#include <concepts>

#include <koqkatoo/linalg-compact/compact/micro-kernels/xpotrf.hpp>
#include "util.hpp"

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpotrf_recursive_ref(mut_single_batch_view H) {
    using std::sqrt;
    const index_t m = H.rows(), n = H.cols();
    // Base case
    if (n == 0)
        return;
    else if (n == 1)
        return sqrt(simd{&H(0, 0, 0), stdx::vector_aligned})
            .copy_to(&H(0, 0, 0), stdx::vector_aligned);
    else if (n <= micro_kernels::potrf::RowsReg)
        return micro_kernels::potrf::microkernel_lut<Abi>[n - 1](H);
    // Recursively factor as 2×2 block matrix
    index_t n1 = (n + 1) / 2, n2 = n - n1, m2 = m - n1;
    auto H11 = H.top_left(n1, n1), H21 = H.bottom_left(m2, n1),
         H22 = H.bottom_right(m2, n2);
    // Factor H₁₁
    xpotrf_recursive_ref(H11);
    // Compute L₂₁ = H₂₁ L₁₁⁻ᵀ
    xtrsm_RLTN_ref(H11, H21);
    // Update H₂₂ -= L₂₁ L₂₁ᵀ
    xsyrk_sub_ref(H21, H22);
    // Factor H₂₂
    xpotrf_recursive_ref(H22);
}

inline index_t xpotrf_op_cnt(auto H, index_t n) {
    [[maybe_unused]] const index_t m = H.rows(), N = H.cols();
    if (n < 0)
        n = N;
    // TODO: I should probably double-check these ...
    [[maybe_unused]] const auto op_cnt_chol = (n + 1) * n * (n - 1) / 6 +
                                              n * (n - 1) / 2 + 2 * n,
                                op_cnt_trsm = n * (n + 1) * (m - n) / 2,
                                op_cnt_syrk = (N - n) * (N - n + 1) * n / 2;
    return (op_cnt_chol + op_cnt_trsm + op_cnt_syrk) * H.depth();
}

template <class Abi>
void CompactBLAS<Abi>::xpotrf_ref(mut_single_batch_view H, index_t n) {
    KOQKATOO_TRACE("xpotrf", 0, xpotrf_op_cnt(H, n));
    xpotrf_ref_impl(H, n);
}

template <class Abi>
void CompactBLAS<Abi>::xpotrf_ref_impl(mut_single_batch_view H, index_t n) {
    const index_t m = H.rows(), N = H.cols();
    if (n < 0)
        n = N;
    // Base case
    if (simd_stride * m * N * sizeof(real_t) <= 48_KiB * 32) // TODO: tune
        return xpotrf_base_ref(H, n);
    // Recursively factor as 2×2 block matrix
    // TODO: the min(..., n) is not ideal, but it keeps things simpler.
    index_t n1 = std::min<index_t>((N + 1) / 2, n), n2 = N - n1, m2 = m - n1;
    auto H11 = H.top_left(n1, n1), H21 = H.bottom_left(m2, n1),
         H22 = H.bottom_right(m2, n2);
    // Factor H₁₁
    xpotrf_ref_impl(H11);
    // Compute L₂₁ = H₂₁ L₁₁⁻ᵀ
    xtrsm_RLTN_ref(H11, H21);
    // Update H₂₂ -= L₂₁ L₂₁ᵀ
    xsyrk_sub_ref(H21, H22);
    // Factor H₂₂
    if (n > n1)
        xpotrf_ref_impl(H22, n - n1);
}

template <class Abi>
void CompactBLAS<Abi>::xpotrf_base_ref(mut_single_batch_view H, index_t n) {
    using namespace micro_kernels;
    constexpr index_t R = potrf::RowsReg; // Block size
    const index_t m = H.rows(), N = H.cols();
    assert(m >= N);
    assert((n == m && m == N) || (n == N && m >= N) || (n < m && m == N));
    if (n < 0)
        n = N;
    mut_single_batch_matrix_accessor<Abi> H_ = H;

    // TODO: run benchmark to see if we need a special case for
    // n == H.rows() == H.cols().

    // Compute the Cholesky factorization of the very last block (right before
    // the Schur complement block), which has size r×r rather than R×R.
    // If requested, also update the rows below the Cholesky factor, and the
    // Schur complement to the bottom right of the given block.
    // These extra blocks are always sizes (m-n)×r and (m-n)×(m-n) respectively.
    auto process_bottom_right = [m, N, n](auto Hii, index_t r) {
        // If we're factorizing the full matrix, just use Cholesky micro-kernel.
        if (m == n) {
            potrf::microkernel_lut<Abi>[r - 1](Hii);
        }
        // If we're not factorizing the full matrix, we need either the sub-
        // diagonal block, or the sub-diagonal block and the Schur complement.
        else {
            // Cholesky of last block to be factorized + triangular solve with
            // sub-diagonal block.
            auto H21 = Hii.middle_rows(r);
            potrf::microkernel_trsm_lut<Abi>[r - 1](Hii, H21, m - n);
            // Update the Schur complement (bottom right) with the outer product
            // of the sub-diagonal block column.
            if (n < N) {
                auto H22 = Hii.block(r, r);
                foreach_chunked(
                    0, m - n, R,
                    [&](index_t j) {
                        auto Hj1 = H21.block(j, 0), Hjj = H22.block(j, j);
                        potrf::microkernel_syrk_lut_2<Abi>[R - 1][r - 1](
                            Hj1, Hjj, m - n - j);
                    },
                    [&](index_t j, index_t rem) {
                        auto Hj1 = H21.block(j, 0), Hjj = H22.block(j, j);
                        potrf::microkernel_syrk_lut_2<Abi>[rem - 1][r - 1](
                            Hj1, Hjj, m - n - j);
                    });
            }
        }
    };

    // Base case
    if (n == 0) {
        return;
    } else if (n <= potrf::RowsReg) {
        process_bottom_right(H_, n);
        return;
    }
    // Loop over columns of H with block size R.
    index_t i;
    for (i = 0; i + R <= n; i += R) {
        const index_t m2 = m - R - i;
        auto H11         = H_.block(i, i);
        auto H21         = H_.block(i + R, i);
        // Factor the diagonal block and update the subdiagonal block
        potrf::xpotrf_xtrsm_microkernel<Abi, R>(H11, H21, m2);
        // Update the Schur complement (bottom right) with the outer product of
        // the subdiagonal block.
        foreach_chunked(
            i + R, N, R,
            [&](index_t j) {
                auto H21 = H_.block(j, i), H22 = H_.block(j, j);
                potrf::xpotrf_xsyrk_microkernel<Abi, R, R>(H21, H22, m - j);
            },
            [&](index_t j, index_t rem) {
                auto H21 = H_.block(j, i), H22 = H_.block(j, j);
                potrf::microkernel_syrk_lut<Abi>[rem - 1](H21, H22, m - j);
            },
            LoopDir::Backward);
        // Loop backwards for cache locality (we'll use the next column in the
        // next interation, so we want the syrk operation to leave it in cache).
        // TODO: verify in benchmark.
    }
    const index_t rem = n - i;
    if (rem > 0) {
        auto Hii = H_.block(i, i);
        process_bottom_right(Hii, rem);
    }
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpotrf(mut_batch_view H, PreferredBackend b, index_t n) {
    assert(H.rows() >= H.cols());
    if (n < 0)
        n = H.cols();
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && H.has_full_layer_stride() && n == H.cols()) {
            KOQKATOO_TRACE("xpotrf_mkl_compact", 0, xpotrf_op_cnt(H, n));
            auto H11     = H.top_left(n, n);
            MKL_INT info = 0;
            xpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, H11.rows(), H11.data,
                           H11.outer_stride(), &info,
                           vector_format_mkl<real_t, Abi>::format,
                           H11.ceil_depth());
            lapack_throw_on_err("xpotrf_compact", info);
            if (n < H.rows()) {
                auto H21 = H.bottom_left(H.rows() - n, n);
                xtrsm_compact(
                    MKL_COL_MAJOR, MKL_RIGHT, MKL_LOWER, MKL_TRANS, MKL_NONUNIT,
                    H21.rows(), H21.cols(), real_t(1), H11.data,
                    H11.outer_stride(), H21.data, H21.outer_stride(),
                    vector_format_mkl<real_t, Abi>::format, H.ceil_depth());
            }
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            KOQKATOO_TRACE("xpotrf_batched", 0, xpotrf_op_cnt(H, n));
            auto H11 = H.top_left(n, n);
            xpotrf_batch_strided("L", H11.rows(), H11.data, H11.outer_stride(),
                                 H11.layer_stride(), H11.depth());
            if (n < H.rows()) {
                auto H21 = H.bottom_left(H.rows() - n, n);
                xtrsm_batch_strided(
                    CblasColMajor, CblasRight, CblasLower, CblasTrans,
                    CblasNonUnit, H21.rows(), H21.cols(), real_t(1), H11.data,
                    H11.outer_stride(), H11.layer_stride(), H21.data,
                    H21.outer_stride(), H21.layer_stride(), H.depth());
                if (n < H.cols()) {
                    auto H22 = H.bottom_right(H.rows() - n, H.cols() - n);
                    assert(H22.rows() == H22.cols());
                    xsyrk_sub(H21, H22, b);
                }
            }
            return;
        }
    }
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < H.num_batches(); ++i)
            xpotrf_ref(H.batch(i), n);
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < H.num_batches(); ++i)
        xpotrf_ref(H.batch(i), n);
}

template <class Abi>
void CompactBLAS<Abi>::xpotrf_base(mut_batch_view H, PreferredBackend b) {
    assert(H.rows() >= H.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && H.rows() == H.cols() &&
            H.has_full_layer_stride()) {
            KOQKATOO_TRACE("xpotrf_base_mkl_compact", 0, xpotrf_op_cnt(H, -1));
            MKL_INT info = 0;
            xpotrf_compact(
                MKL_COL_MAJOR, MKL_LOWER, H.rows(), H.data, H.outer_stride(),
                &info, vector_format_mkl<real_t, Abi>::format, H.ceil_depth());
            lapack_throw_on_err("xpotrf_compact", info);
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b) && H.rows() == H.cols()) {
            KOQKATOO_TRACE("xpotrf_base_batched", 0, xpotrf_op_cnt(H, -1));
            xpotrf_batch_strided("L", H.rows(), H.data, H.outer_stride(),
                                 H.layer_stride(), H.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < H.num_batches(); ++i)
        xpotrf_base_ref(H.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xpotrf_recursive(mut_batch_view H, PreferredBackend b) {
    assert(H.rows() >= H.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && H.rows() == H.cols() &&
            H.has_full_layer_stride()) {
            KOQKATOO_TRACE("xpotrf_recursive_mkl_compact", 0,
                           xpotrf_op_cnt(H, -1));
            MKL_INT info = 0;
            xpotrf_compact(
                MKL_COL_MAJOR, MKL_LOWER, H.rows(), H.data, H.outer_stride(),
                &info, vector_format_mkl<real_t, Abi>::format, H.ceil_depth());
            lapack_throw_on_err("xpotrf_compact", info);
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b) && H.rows() == H.cols()) {
            KOQKATOO_TRACE("xpotrf_recursive_batched", 0, xpotrf_op_cnt(H, -1));
            xpotrf_batch_strided("L", H.rows(), H.data, H.rows(),
                                 H.rows() * H.cols(), H.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < H.num_batches(); ++i)
        xpotrf_recursive_ref(H.batch(i));
}

// BLAS specializations (scalar)
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpotrf(mut_single_batch_view H, PreferredBackend b,
                              index_t n) {
    assert(H.rows() >= H.cols());
    if (n < 0)
        n = H.cols();
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            KOQKATOO_TRACE("xpotrf_blas", 0, xpotrf_op_cnt(H, n));
            auto H11     = H.top_left(n, n);
            index_t info = 0;
            linalg::xpotrf("L", H11.rows(), H11.data, H11.outer_stride(),
                           &info);
            lapack_throw_on_err("xpotrf", info);
            if (n < H.rows()) {
                auto H21 = H.bottom_left(H.rows() - n, n);
                linalg::xtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                              CblasNonUnit, H21.rows(), H21.cols(), real_t(1),
                              H11.data, H11.outer_stride(), H21.data,
                              H21.outer_stride());
                if (n < H.cols()) {
                    auto H22 = H.bottom_right(H.rows() - n, H.cols() - n);
                    assert(H22.rows() == H22.cols());
                    linalg::xsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                  H22.rows(), H21.cols(), real_t{-1}, H21.data,
                                  H21.outer_stride(), real_t{1}, H22.data,
                                  H22.outer_stride());
                }
            }
            return;
        }
    }
    xpotrf_ref(H, n);
}

} // namespace koqkatoo::linalg::compact
