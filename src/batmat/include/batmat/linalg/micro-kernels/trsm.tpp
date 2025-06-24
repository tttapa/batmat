#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/trsm.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::trsm {

/// @param  A Lower trapezoidal RowsReg×(k+RowsReg).
/// @param  B RowsReg×ColsReg.
/// @param  D (k+RowsReg)×ColsReg.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OA,
          StorageOrder OB, StorageOrder OD>
[[gnu::hot, gnu::flatten]] void
trsm_copy_microkernel(const uview<const T, Abi, OA> A, const uview<const T, Abi, OB> B,
                      const uview<T, Abi, OD> D, const index_t k) noexcept {
    static_assert(Conf.struc_A == MatrixStructure::LowerTriangular); // TODO
    static_assert(RowsReg > 0 && ColsReg > 0);
    using namespace ops;
    using simd = stdx::simd<T, Abi>;
    // Pre-compute the offsets of the columns/rows of B
    const auto B_cached = with_cached_access<RowsReg, ColsReg>(B);
    // Load accumulator into registers
    simd B_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            B_reg[ii][jj] = B_cached.load(ii, jj);
    // Matrix multiplication
    const auto D_cached = with_cached_access<0, ColsReg>(D);
    for (index_t l = 0; l < k; ++l)
        UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
            simd Xlj = D_cached.load(l, jj);
            UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
                simd Ail  = A.load(ii, l);
                simd &Bij = B_reg[ii][jj];
                Bij -= Ail * Xlj;
            }
        }
    // Triangular solve
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
        simd Aii = 1 / A.load(ii, k + ii);
        UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
            simd &Xij = B_reg[ii][jj];
            UNROLL_FOR (index_t l = 0; l < ii; ++l) {
                simd Ail  = A.load(ii, k + l);
                simd &Xlj = B_reg[l][jj];
                Xij -= Ail * Xlj;
            }
            Xij *= Aii; // Diagonal already inverted
        }
    }
    // Store accumulator to memory again
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            D_cached.store(B_reg[ii][jj], k + ii, jj);
}

/// Triangular solve D = (A⁽ᵀ⁾)⁻¹ B⁽ᵀ⁾ where A⁽ᵀ⁾ is lower triangular. Using register blocking.
/// Note: D = A⁻¹ B  <=>  Dᵀ = Bᵀ A⁻ᵀ
template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OD>
void trsm_copy_register(const view<const T, Abi, OA> A, const view<const T, Abi, OB> B,
                        const view<T, Abi, OD> D) noexcept {
    static_assert(Conf.struc_A == MatrixStructure::LowerTriangular); // TODO
    constexpr auto Rows = RowsReg<T, Abi>, Cols = ColsReg<T, Abi>;
    // Check dimensions
    const index_t I = A.rows(), K = A.cols(), J = B.cols();
    BATMAT_ASSUME(K >= I);
    BATMAT_ASSUME(B.rows() == I);
    BATMAT_ASSUME(D.rows() == K);
    BATMAT_ASSUME(D.cols() == J);
    BATMAT_ASSUME(I > 0);
    BATMAT_ASSUME(J > 0);
    BATMAT_ASSUME(K > 0);
    static const auto microkernel = trsm_copy_lut<T, Abi, Conf, OA, OB, OD>;
    // Sizeless views to partition and pass to the micro-kernels
    const uview<const T, Abi, OA> A_ = A;
    const uview<const T, Abi, OB> B_ = B;
    const uview<T, Abi, OD> D_       = D;

    // Optimization for very small matrices
    if (I <= Rows && J <= Cols)
        return microkernel[I - 1][J - 1](A_, B_, D_, 0);

    // Simply loop over all blocks in the given matrices.
    auto run = [&] [[gnu::always_inline]] (index_t i, index_t ni, index_t j, index_t nj) {
        auto Ai0 = A_.middle_rows(i); // subdiagonal block row
        auto Bij = B_.block(i, j);    // rhs block to solve now
        auto X0j = D_.middle_cols(j); // solution up to i and solution block to fill in
        microkernel[ni - 1][nj - 1](Ai0, Bij, X0j, i + K - I);
    };
    if constexpr (OD == StorageOrder::ColMajor)
        // Loop over block columns of B and D
        foreach_chunked_merged(0, J, Cols, [&](index_t j, auto nj) {
            // Loop over the diagonal blocks of A
            foreach_chunked_merged(0, I, Rows, [&](index_t i, auto ni) { run(i, ni, j, nj); });
        });
    else
        // Loop over the diagonal blocks of A
        foreach_chunked_merged(0, I, Rows, [&](index_t i, auto ni) {
            // Loop over block columns of B and D
            foreach_chunked_merged(0, J, Cols, [&](index_t j, auto nj) { run(i, ni, j, nj); });
        });
}

} // namespace batmat::linalg::micro_kernels::trsm
