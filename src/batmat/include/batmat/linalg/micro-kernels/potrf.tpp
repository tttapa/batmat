#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/potrf.hpp>
#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/ops/rsqrt.hpp>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::potrf {

/// @param  A1 RowsReg×k1.
/// @param  A2 RowsReg×k2.
/// @param  C RowsReg×RowsReg.
/// @param  D RowsReg×RowsReg.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder O1, StorageOrder O2>
[[gnu::hot, gnu::flatten]] void
potrf_copy_microkernel(const uview<const T, Abi, O1> A1, const uview<const T, Abi, O2> A2,
                       const uview<const T, Abi, O2> C, const uview<T, Abi, O2> D, T *const invD,
                       const index_t k1, const index_t k2) noexcept {
    static_assert(Conf.struc_C == MatrixStructure::LowerTriangular); // TODO
    static_assert(RowsReg > 0);
    using ops::rsqrt;
    using simd = datapar::simd<T, Abi>;
    // Pre-compute the offsets of the columns of C
    const auto C_cached = with_cached_access<RowsReg, RowsReg>(C);
    // Load matrix into registers
    simd C_reg[RowsReg * (RowsReg + 1) / 2]; // NOLINT(*-c-arrays)
    const auto index = [](index_t r, index_t c) { return c * (2 * RowsReg - 1 - c) / 2 + r; };
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (index_t jj = 0; jj <= ii; ++jj)
            C_reg[index(ii, jj)] = C_cached.load(ii, jj);
    // Perform syrk operation of A
    const auto A1_cached = with_cached_access<RowsReg, 0>(A1);
    for (index_t l = 0; l < k1; ++l)
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A1_cached.load(ii, l);
            UNROLL_FOR (index_t jj = 0; jj <= ii; ++jj) {
                simd &Cij = C_reg[index(ii, jj)];
                simd Blj  = A1_cached.load(jj, l);
                Conf.negate_A ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
            }
        }
    const auto A2_cached = with_cached_access<RowsReg, 0>(A2);
    for (index_t l = 0; l < k2; ++l)
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A2_cached.load(ii, l);
            UNROLL_FOR (index_t jj = 0; jj <= ii; ++jj) {
                simd &Cij = C_reg[index(ii, jj)];
                simd Blj  = A2_cached.load(jj, l);
                Cij -= Ail * Blj;
            }
        }
#if 1
    // Actual Cholesky kernel (Cholesky–Crout)
    UNROLL_FOR (index_t j = 0; j < RowsReg; ++j) {
        UNROLL_FOR (index_t k = 0; k < j; ++k)
            C_reg[index(j, j)] -= C_reg[index(j, k)] * C_reg[index(j, k)];
        simd inv_pivot     = rsqrt(C_reg[index(j, j)]);
        C_reg[index(j, j)] = sqrt(C_reg[index(j, j)]);
        datapar::aligned_store(inv_pivot, invD + j * simd::size());
        UNROLL_FOR (index_t i = j + 1; i < RowsReg; ++i) {
            UNROLL_FOR (index_t k = 0; k < j; ++k)
                C_reg[index(i, j)] -= C_reg[index(i, k)] * C_reg[index(j, k)];
            C_reg[index(i, j)] = inv_pivot * C_reg[index(i, j)];
        }
    }
#elif 0
    // Actual Cholesky kernel (naive, sqrt/rsqrt in critical path)
    UNROLL_FOR (index_t j = 0; j < RowsReg; ++j) {
        simd inv_pivot     = rsqrt(C_reg[index(j, j)]);
        C_reg[index(j, j)] = sqrt(C_reg[index(j, j)]);
        datapar::aligned_store(inv_pivot, invD + j * simd::size());
        UNROLL_FOR (index_t i = j + 1; i < RowsReg; ++i)
            C_reg[index(i, j)] *= inv_pivot;
        UNROLL_FOR (index_t i = j + 1; i < RowsReg; ++i)
            UNROLL_FOR (index_t k = j + 1; k <= i; ++k)
                C_reg[index(i, k)] -= C_reg[index(i, j)] * C_reg[index(k, j)];
    }
#else
    // Actual Cholesky kernel (naive, but hiding the latency of sqrt/rsqrt)
    simd inv_pivot     = rsqrt(C_reg[index(0, 0)]);
    C_reg[index(0, 0)] = sqrt(C_reg[index(0, 0)]);
    UNROLL_FOR (index_t j = 0; j < RowsReg; ++j) {
        datapar::aligned_store(inv_pivot, invD + j * simd::size());
        UNROLL_FOR (index_t i = j + 1; i < RowsReg; ++i)
            C_reg[index(i, j)] *= inv_pivot;
        UNROLL_FOR (index_t i = j + 1; i < RowsReg; ++i)
            UNROLL_FOR (index_t k = j + 1; k <= i; ++k) {
                C_reg[index(i, k)] -= C_reg[index(i, j)] * C_reg[index(k, j)];
                if (k == j + 1 && i == j + 1) {
                    inv_pivot                  = rsqrt(C_reg[index(j + 1, j + 1)]);
                    C_reg[index(j + 1, j + 1)] = sqrt(C_reg[index(j + 1, j + 1)]);
                }
            }
    }
#endif
    // Store result to memory
    auto D_cached = with_cached_access<RowsReg, RowsReg>(D);
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (index_t jj = 0; jj <= ii; ++jj)
            D_cached.store(C_reg[index(ii, jj)], ii, jj);
}

/// @param  A1 RowsReg×k1.
/// @param  B1 ColsReg×k1.
/// @param  A2 RowsReg×k2.
/// @param  B2 ColsReg×k2.
/// @param  L ColsReg×ColsReg.
/// @param  C RowsReg×ColsReg.
/// @param  D RowsReg×ColsReg.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder O1,
          StorageOrder O2>
[[gnu::hot, gnu::flatten]]
void trsm_copy_microkernel(const uview<const T, Abi, O1> A1, const uview<const T, Abi, O1> B1,
                           const uview<const T, Abi, O2> A2, const uview<const T, Abi, O2> B2,
                           const uview<const T, Abi, O2> L, const T *invL,
                           const uview<const T, Abi, O2> C, const uview<T, Abi, O2> D,
                           const index_t k1, const index_t k2) noexcept {
    static_assert(Conf.struc_C == MatrixStructure::LowerTriangular); // TODO
    static_assert(RowsReg > 0 && ColsReg > 0);
    using ops::rsqrt;
    using simd = datapar::simd<T, Abi>;
    // Pre-compute the offsets of the columns of C
    const auto C_cached = with_cached_access<RowsReg, ColsReg>(C);
    // Load matrix into registers
    simd C_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            C_reg[ii][jj] = C_cached.load(ii, jj);
    // Perform gemm operation of A and B
    const auto A1_cached = with_cached_access<RowsReg, 0>(A1);
    const auto B1_cached = with_cached_access<ColsReg, 0>(B1);
    for (index_t l = 0; l < k1; ++l)
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A1_cached.load(ii, l);
            UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj  = B1_cached.load(jj, l);
                Conf.negate_A ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
            }
        }
    const auto A2_cached = with_cached_access<RowsReg, 0>(A2);
    const auto B2_cached = with_cached_access<ColsReg, 0>(B2);
    for (index_t l = 0; l < k2; ++l)
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A2_cached.load(ii, l);
            UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj  = B2_cached.load(jj, l);
                Cij -= Ail * Blj;
            }
        }
    // Triangular solve
    UNROLL_FOR (index_t jj = 0; jj < RowsReg; ++jj)
        UNROLL_FOR (index_t ii = 0; ii < ColsReg; ++ii) {
            simd &Xij    = C_reg[jj][ii];
            simd inv_piv = datapar::aligned_load<simd>(invL + ii * simd::size());
            UNROLL_FOR (index_t kk = 0; kk < ii; ++kk) {
                simd Aik = L.load(ii, kk);
                simd Xkj = C_reg[jj][kk];
                Xij -= Aik * Xkj;
            }
            Xij *= inv_piv;
        }
    // Store result to memory
    auto D_cached = with_cached_access<RowsReg, ColsReg>(D);
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            D_cached.store(C_reg[ii][jj], ii, jj);
}

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OCD>
void potrf_copy_register(const view<const T, Abi, OA> A, const view<const T, Abi, OCD> C,
                         const view<T, Abi, OCD> D) noexcept {
    using enum MatrixStructure;
    static_assert(Conf.struc_C == LowerTriangular); // TODO
    constexpr auto Rows = RowsReg<T, Abi>, Cols = ColsReg<T, Abi>;
    // Check dimensions
    const index_t I = C.rows(), K = A.cols(), J = C.cols();
    BATMAT_ASSUME(I >= J);
    BATMAT_ASSUME(A.cols() == 0 || A.rows() == I);
    BATMAT_ASSUME(D.rows() == I);
    BATMAT_ASSUME(D.cols() == J);
    BATMAT_ASSUME(I > 0);
    BATMAT_ASSUME(J > 0);
    static const auto potrf_microkernel = potrf_copy_lut<T, Abi, Conf, OA, OCD>;
    static const auto trsm_microkernel  = trsm_copy_lut<T, Abi, Conf, OA, OCD>;
    // Sizeless views to partition and pass to the micro-kernels
    const uview<const T, Abi, OA> A_  = A;
    const uview<const T, Abi, OCD> C_ = C;
    const uview<T, Abi, OCD> D_       = D;
    alignas(datapar::simd_align<T, Abi>::value) T invD[Cols * datapar::simd_size<T, Abi>::value];

    // Optimization for very small matrices
    if (I <= Rows && J <= Cols && I == J)
        return potrf_microkernel[J - 1](A_, C_, C_, D_, invD, K, 0);

    foreach_chunked_merged( // Loop over the diagonal blocks of C
        0, J, Cols, [&](index_t j, auto nj) {
            const auto Aj  = A_.middle_rows(j);
            const auto Dj  = D_.middle_rows(j);
            const auto Djj = D_.block(j, j);
            // Djj = chol(Cjj ± Aj Ajᵀ - Dj Djᵀ)
            potrf_microkernel[nj - 1](Aj, Dj, C_.block(j, j), Djj, invD, K, j);
            foreach_chunked_merged( // Loop over the subdiagonal rows
                j + nj, I, Rows, [&](index_t i, auto ni) {
                    const auto Ai  = A_.middle_rows(i);
                    const auto Di  = D_.middle_rows(i);
                    const auto Cij = C_.block(i, j);
                    const auto Dij = D_.block(i, j);
                    // Dij = (Cij ± Ai Ajᵀ - Di Djᵀ) Djj⁻ᵀ
                    trsm_microkernel[ni - 1][nj - 1](Ai, Aj, Di, Dj, Djj, invD, Cij, Dij, K, j);
                });
        });
}

} // namespace batmat::linalg::micro_kernels::potrf
