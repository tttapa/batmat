#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/syomv.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/ops/rotate.hpp>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::syomv {

/// Symmetric off-diagonal block multiply. Single register block.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder OA, StorageOrder OB,
          StorageOrder OD>
[[gnu::hot, gnu::flatten]] void
syomv_microkernel(const uview<const T, Abi, OA> A, const uview<const T, Abi, OB> B,
                  const uview<T, Abi, OD> D, const index_t l0, const index_t k) noexcept {
    static_assert(RowsReg > 0);
    static_assert(Conf.struc_A == MatrixStructure::LowerTriangular); // TODO
    using simd = datapar::simd<T, Abi>;
    BATMAT_ASSUME(k > 0);
    // Pre-compute the offsets of the columns of A
    auto A_cached = with_cached_access<0, RowsReg>(A);
    // Initialize accumulator
    simd accum[RowsReg]{}; // NOLINT(*-c-arrays)
    // Column weights for non-transposed block
    simd Bl[RowsReg];
    UNROLL_FOR (index_t l = 0; l < RowsReg; ++l)
        Bl[l] = ops::shiftr<1>(B.load(l + l0, 0));
    // Actual multiplication kernel
    for (index_t i = 0; i < k; ++i) {
        simd Bi = B.load(i, 0);
        simd Di = D.load(i, 0);
        UNROLL_FOR (index_t l = 0; l < RowsReg; ++l) {
            // Dot product between first lane of A and second lane of x
            simd Ail = A_cached.load(i, l);
            accum[l] += Ail * ops::shiftl<1>(Bi);
            // Linear combination of columns of first lane of A, weighted by
            // the rows of the first lane of x, added to the second lane of v
            Ail = ops::shiftr<1>(Ail); // TODO: rotr?
            Conf.negate ? (Di -= Ail * Bl[l]) : (Di += Ail * Bl[l]);
        }
        D.store(Di, i, 0);
    }
    // Subtract dot products of transposed block
    UNROLL_FOR (index_t l = 0; l < RowsReg; ++l) {
        simd vl = D.load(l + l0, 0);
        Conf.negate ? (vl -= accum[l]) : (vl += accum[l]);
        D.store(vl, l + l0, 0);
    }
}

/// Generalized matrix multiplication D = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Using register blocking.
template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OD>
void syomv_register(const view<const T, Abi, OA> A, const view<const T, Abi, OB> B,
                    const view<T, Abi, OD> D) noexcept {
    using enum MatrixStructure;
    constexpr auto Rows = RowsReg<T, Abi>;
    // Check dimensions
    const index_t I = D.rows(), J = D.cols(), K = A.cols();
    BATMAT_ASSUME(A.rows() == I);
    BATMAT_ASSUME(B.rows() == K);
    BATMAT_ASSUME(B.cols() == J);
    BATMAT_ASSUME(I == K);
    BATMAT_ASSUME(I > 0);
    BATMAT_ASSUME(J > 0);
    BATMAT_ASSUME(K > 0);
    BATMAT_ASSUME(J == 1); // TODO
    static const auto microkernel = syomv_lut<T, Abi, Conf, OA, OB, OD>;
    // Sizeless views to partition and pass to the micro-kernels
    const uview<const T, Abi, OA> A_ = A;
    const uview<const T, Abi, OB> B_ = B;
    const uview<T, Abi, OD> D_       = D;

    // Optimization for very small matrices
    if (I <= Rows)
        return microkernel[I - 1](A_, B_, D_, 0, K);
    // Simply loop over all block columns in the matrix.
    foreach_chunked_merged(0, K, Rows, [&](index_t l, index_t nl) {
        const auto Al = A_.middle_cols(l);
        microkernel[nl - 1](Al, B_, D_, l, A.rows());
    });
}

} // namespace batmat::linalg::micro_kernels::syomv
