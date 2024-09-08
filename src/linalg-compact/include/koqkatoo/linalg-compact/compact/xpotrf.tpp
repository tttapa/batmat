#pragma once

#include <koqkatoo/kib.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/lapack.hpp>

#include <cmath>
#include <concepts>
#include <type_traits>

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

template <class Abi>
void CompactBLAS<Abi>::xpotrf_ref(mut_single_batch_view H) {
    const index_t m = H.rows(), n = H.cols();
    // Base case
    if (simd_stride * m * n * sizeof(real_t) <= 48_KiB * 32) // TODO: tune
        return xpotrf_base_ref(H);
    // Recursively factor as 2×2 block matrix
    index_t n1 = (n + 1) / 2, n2 = n - n1, m2 = m - n1;
    auto H11 = H.top_left(n1, n1), H21 = H.bottom_left(m2, n1),
         H22 = H.bottom_right(m2, n2);
    // Factor H₁₁
    xpotrf_ref(H11);
    // Compute L₂₁ = H₂₁ L₁₁⁻ᵀ
    xtrsm_RLTN_ref(H11, H21);
    // Update H₂₂ -= L₂₁ L₂₁ᵀ
    xsyrk_sub_ref(H21, H22);
    // Factor H₂₂
    xpotrf_ref(H22);
}

template <class Abi>
void CompactBLAS<Abi>::xpotrf_base_ref(mut_single_batch_view H) {
    using namespace micro_kernels;
    constexpr index_t R = potrf::RowsReg; // Block size
    const index_t m = H.rows(), n = H.cols();
    mut_single_batch_matrix_accessor<Abi> H_ = H;
    // Base case
    if (n == 0) {
        return;
    } else if (n <= potrf::RowsReg) {
        auto H11 = H_.block(0, 0);
        auto H21 = H_.block(n, 0);
        m == n ? potrf::microkernel_lut<Abi>[n - 1](H11)
               : potrf::microkernel_trsm_lut<Abi>[n - 1](H11, H21, m - n);
        return;
    }
    // Loop over columns of H with block size R
    index_t i;
    for (i = 0; i + R <= n; i += R) {
        const index_t m2 = m - R - i;
        auto H11         = H_.block(i, i);
        auto H21         = H_.block(i + R, i);
        // Factor the diagonal block and update the subdiagonal block
        potrf::xpotrf_xtrsm_microkernel<Abi, R>(H11, H21, m2);
        // Update the tail with the outer product of the subdiagonal block
        const index_t cols = n - (i + R);
        const index_t rem  = cols % R;
        // Remainder block (bottom right) with size less than R
        if (rem > 0) {
            index_t j = n - rem;
            auto H21  = H_.block(j, i);
            auto H22  = H_.block(j, j);
            potrf::microkernel_syrk_lut<Abi>[rem - 1](H21, H22, m - j);
        }
        // Loop backwards for cache locality (we'll use the next column in the
        // next interation, so we want the syrk operation to leave it in cache).
        for (index_t j = n - rem - R; j >= i + R; j -= R) {
            static_assert(std::is_signed_v<index_t>);
            auto H21 = H_.block(j, i);
            auto H22 = H_.block(j, j);
            potrf::xpotrf_xsyrk_microkernel<Abi, R, R>(H21, H22, m - j);
        }
    }
    const index_t rem = n - i;
    if (rem > 0) {
        auto H11 = H_.block(i, i);
        auto H21 = H_.block(n, i);
        m == n ? potrf::microkernel_lut<Abi>[rem - 1](H11)
               : potrf::microkernel_trsm_lut<Abi>[rem - 1](H11, H21, m - n);
    }
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpotrf(mut_batch_view H, PreferredBackend b) {
    assert(H.rows() >= H.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && H.rows() == H.cols()) {
            MKL_INT info = 0;
            xpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, H.rows(), H.data, H.rows(),
                           &info, vector_format_mkl<real_t, Abi>::format,
                           H.ceil_depth());
            lapack_throw_on_err("xpotrf_compact", info);
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b) && H.rows() == H.cols()) {
            xpotrf_batch_strided("L", H.rows(), H.data, H.rows(),
                                 H.rows() * H.cols(), H.depth());
            return;
        }
    }
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < H.num_batches(); ++i)
            xpotrf_ref(H.batch(i));
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < H.num_batches(); ++i)
        xpotrf_ref(H.batch(i));
}

template <class Abi>
void CompactBLAS<Abi>::xpotrf_base(mut_batch_view H, PreferredBackend b) {
    assert(H.rows() >= H.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b) && H.rows() == H.cols()) {
            MKL_INT info = 0;
            xpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, H.rows(), H.data, H.rows(),
                           &info, vector_format_mkl<real_t, Abi>::format,
                           H.ceil_depth());
            lapack_throw_on_err("xpotrf_compact", info);
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b) && H.rows() == H.cols()) {
            xpotrf_batch_strided("L", H.rows(), H.data, H.rows(),
                                 H.rows() * H.cols(), H.depth());
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
        if (use_mkl_compact(b) && H.rows() == H.cols()) {
            MKL_INT info = 0;
            xpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, H.rows(), H.data, H.rows(),
                           &info, vector_format_mkl<real_t, Abi>::format,
                           H.ceil_depth());
            lapack_throw_on_err("xpotrf_compact", info);
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b) && H.rows() == H.cols()) {
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
void CompactBLAS<Abi>::xpotrf(mut_single_batch_view H, PreferredBackend b) {
    assert(H.rows() >= H.cols());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b) && H.rows() == H.cols()) {
            index_t info = 0;
            index_t n = H.cols(), ldh = H.outer_stride();
            linalg::xpotrf("L", &n, H.data, &ldh, &info);
            lapack_throw_on_err("xpotrf", info);
            return;
        }
    }
    xpotrf_ref(H);
}

} // namespace koqkatoo::linalg::compact
