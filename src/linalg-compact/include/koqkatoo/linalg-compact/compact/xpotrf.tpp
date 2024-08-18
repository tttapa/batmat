#pragma once

#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <koqkatoo/linalg/lapack.hpp>

#include <cmath>
#include <concepts>

#include <koqkatoo/linalg-compact/compact-new/micro-kernels/xgemm.hpp>
#include <koqkatoo/linalg-compact/compact-new/micro-kernels/xpotrf.hpp>
#include "util.hpp"

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpotrf_recursive_ref(mut_single_batch_view H) {
    using std::sqrt;
    const index_t n = H.rows();
    // Base case
    if (n == 0)
        return;
    else if (n == 1)
        return sqrt(simd{&H(0, 0, 0), stdx::vector_aligned})
            .copy_to(&H(0, 0, 0), stdx::vector_aligned);
    else if (n <= micro_kernels::potrf::RowsReg)
        return micro_kernels::potrf::microkernel_lut<Abi>[n - 1](H);
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

template <class Abi>
void CompactBLAS<Abi>::xpotrf_ref(mut_single_batch_view H) {
    using std::sqrt;
    using namespace micro_kernels;
    const index_t n = H.rows();
    // Base case
    if (n == 0)
        return;
    else if (n == 1)
        return sqrt(simd{&H(0, 0, 0), stdx::vector_aligned})
            .copy_to(&H(0, 0, 0), stdx::vector_aligned);
    else if (n <= potrf::RowsReg)
        return potrf::microkernel_lut<Abi>[n - 1](H);
    mut_single_batch_matrix_accessor<Abi> H_ = H;
    constexpr index_t R                      = gemm::RowsReg;
    // Loop over columns of H with block size R
    index_t i;
    for (i = 0; i + R <= n; i += R) {
        const index_t n2 = n - R - i;
        auto H11         = H_.block(i, i);
        auto H21         = H_.block(i + R, i);
        // Factor the diagonal block and update the subdiagonal block
        potrf::microkernel_trsm_lut<Abi>[R - 1](H11, H21, n2);
        // Update the tail with the outer product of the subdiagonal block
        for (index_t j = i + R; j < n; j += R) {
            auto H21         = H_.block(j, i);
            auto H22         = H_.block(j, j);
            const index_t nj = std::min(R, n - j);
            potrf::microkernel_syrk_lut<Abi>[nj - 1][R - 1](H21, H22, n - j);
        }
    }
    const index_t rem = n - i;
    if (rem > 0)
        potrf::microkernel_lut<Abi>[rem - 1](H_.block(i, i));
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
