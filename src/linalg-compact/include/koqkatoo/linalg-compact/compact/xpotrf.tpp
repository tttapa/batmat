#pragma once

#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/lapack.hpp>

#include <cmath>
#include <concepts>

#include "micro_kernels/xpotrf.tpp"

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpotrf_ref(mut_single_batch_view H) {
    using std::sqrt;
    const index_t n = H.rows();
    [[assume(n >= 0)]];
    // Base case
    if (n == 0)
        return;
    else if (n == 1)
        return sqrt(simd{&H(0, 0, 0), stdx::vector_aligned})
            .copy_to(&H(0, 0, 0), stdx::vector_aligned);
    else if (n <= micro_kernels::potrf::RowsReg)
        return micro_kernels::potrf::xpotrf_register<simd>(H);
    // Recursively factor as 2×2 block matrix
    index_t n1 = (n + 1) / 2, n2 = n - n1;
    auto H11 = H.top_left(n1, n1), H21 = H.bottom_left(n2, n1),
         H22 = H.bottom_right(n2, n2);
    // Factor H₁₁
    xpotrf_ref(H11);
    // Compute L₂₁ = H₂₁ L₁₁⁻¹
    xtrsm_RLTN_ref(H11, H21);
    // Update H₂₂ -= L₂₁ L₂₁ᵀ
    xsyrk_sub_ref(H21, H22);
    // Factor H₂₂
    xpotrf_ref(H22);
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpotrf(mut_batch_view H, PreferredBackend b) {
    assert(H.rows() == H.cols());
    KOQKATOO_MKL_IF(if constexpr (supports_mkl_packed<real_t, Abi>) {
        if (use_mkl_compact(b)) {
            MKL_INT info = 0;
            xpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, H.rows(), H.data, H.rows(),
                           &info, vector_format_mkl<real_t, Abi>::format,
                           H.ceil_depth());
            lapack_throw_on_err("xpotrf_compact", info);
            return;
        }
    })
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_mkl_batched(b)) {
            xpotrf_batch_strided("L", H.rows(), H.data, H.rows(),
                                 H.rows() * H.cols(), H.depth());
            return;
        }
    }
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < H.num_batches(); ++i)
        xpotrf_ref(H.batch(i));
}

// BLAS specializations (scalar)
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpotrf(mut_single_batch_view H, PreferredBackend b) {
    assert(H.rows() == H.cols());
    if constexpr (std::same_as<Abi, scalar_abi>) {
        if (use_blas_scalar(b)) {
            index_t info = 0;
            index_t n = H.rows(), ldh = H.outer_stride();
            linalg::xpotrf("L", &n, H.data, &ldh, &info);
            lapack_throw_on_err("xpotrf", info);
            return;
        }
    }
    xpotrf_ref(H);
}

} // namespace koqkatoo::linalg::compact
