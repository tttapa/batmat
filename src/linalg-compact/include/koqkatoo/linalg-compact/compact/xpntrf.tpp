#pragma once

#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/loop.hpp>
#include <koqkatoo/openmp.h>

#include <cmath>

#include <koqkatoo/linalg-compact/compact/micro-kernels/xpntrf.hpp>

namespace koqkatoo::linalg::compact {

// Reference implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpntrf_ref(mut_single_batch_view H,
                                  single_batch_view signs) {
    using namespace micro_kernels;
    constexpr index_t R = pntrf::RowsReg; // Block size
    const index_t m = H.rows(), n = H.cols();
    [[maybe_unused]] const auto op_cnt_chol = (n + 1) * n * (n - 1) / 6 +
                                              n * (n - 1) / 2 + 2 * n,
                                op_cnt_trsm = n * (n + 1) * (m - n) / 2;
    KOQKATOO_TRACE("xpntrf", 0, (op_cnt_chol + op_cnt_trsm) * H.depth());
    assert(m >= n);
    assert(n == signs.rows());
    mut_single_batch_matrix_accessor<Abi> H_ = H;
    single_batch_vector_accessor<Abi> s_     = signs;

    // Base case
    if (n == 0) {
        return;
    } else if (n <= pntrf::RowsReg) {
        const auto s1  = s_.middle_rows(0);
        const auto H11 = H_.block(0, 0), H21 = H_.block(n, 0);
        pntrf::microkernel_trsm_lut<Abi>[n - 1](H11, H21, m - n, s1);
        return;
    }
    // Loop over columns of H with block size R.
    index_t i;
    for (i = 0; i + R <= n; i += R) {
        const index_t m2 = m - R - i;
        const auto s1    = s_.middle_rows(i);
        const auto H11 = H_.block(i, i), H21 = H_.block(i + R, i);
        // Factor the diagonal block and update the subdiagonal block
        pntrf::xpntrf_xtrsm_microkernel<Abi, R>(H11, H21, m2, s1);
        // Update the Schur complement (bottom right) with the outer product of
        // the subdiagonal block.
        foreach_chunked(
            i + R, n, R,
            [&](index_t j) {
                auto H21 = H_.block(j, i), H22 = H_.block(j, j);
                pntrf::xpntrf_xsyrk_microkernel<Abi, R, R>(H21, H22, m - j, s1);
            },
            [&](index_t j, index_t rem) {
                auto H21 = H_.block(j, i), H22 = H_.block(j, j);
                pntrf::microkernel_syrk_lut<Abi>[rem - 1](H21, H22, m - j, s1);
            },
            LoopDir::Backward);
        // Loop backwards for cache locality (we'll use the next column in the
        // next interation, so we want the syrk operation to leave it in cache).
        // TODO: verify in benchmark.
    }
    const index_t rem = n - i;
    if (rem > 0) {
        auto Hii       = H_.block(i, i);
        const auto si  = s_.middle_rows(i);
        const auto H11 = Hii.block(0, 0), H21 = Hii.block(rem, 0);
        pntrf::microkernel_trsm_lut<Abi>[rem - 1](H11, H21, m - n, si);
    }
}

// Parallel batched implementations
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpntrf(mut_batch_view H, batch_view signs) {
    assert(H.rows() >= H.cols());
    assert(H.cols() == signs.rows());
    assert(H.ceil_depth() == signs.ceil_depth());
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        for (index_t i = 0; i < H.num_batches(); ++i)
            xpntrf_ref(H.batch(i), signs.batch(i));
        return;
    }
#endif
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < H.num_batches(); ++i)
        xpntrf_ref(H.batch(i), signs.batch(i));
}

// BLAS specializations (scalar)
// -----------------------------------------------------------------------------

template <class Abi>
void CompactBLAS<Abi>::xpntrf(mut_single_batch_view H,
                              single_batch_view signs) {
    assert(H.rows() >= H.cols());
    assert(H.cols() == signs.rows());
    xpntrf_ref(H, signs);
    // TODO: vectorized scalar implementation
}

} // namespace koqkatoo::linalg::compact
