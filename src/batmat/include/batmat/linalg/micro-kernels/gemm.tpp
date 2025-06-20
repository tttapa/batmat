#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/gemm.hpp>
#include <batmat/ops/rotate.hpp>

namespace batmat::linalg::micro_kernels::gemm {

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Single register block.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
[[gnu::hot, gnu::flatten]] void gemm_microkernel(const uview<const T, Abi, Conf.order_A> A,
                                                 const uview<const T, Abi, Conf.order_B> B,
                                                 const uview<T, Abi, Conf.order_C> C,
                                                 const index_t k, bool init_zero) noexcept {
    using namespace ops;
    using simd = stdx::simd<T, Abi>;
    // The following assumption ensures that there is no unnecessary branch
    // for k == 0 in between the loops. This is crucial for good code
    // generation, otherwise the compiler inserts jumps and labels between
    // the matmul kernel and the loading/storing of C, which will cause it to
    // place C_reg on the stack, resulting in many unnecessary loads and stores.
    BATMAT_ASSUME(k > 0);
    // Pre-compute the offsets of the columns of C
    auto C_cached = with_cached_access<RowsReg, ColsReg>(C);
    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg]{}; // NOLINT(*-c-arrays)
    if (!init_zero) [[likely]]
        BATMAT_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            BATMAT_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
                C_reg[ii][jj] = rotl<Conf.shift_C>(C_cached.load(ii, jj));
    // Actual matrix multiplication kernel
    for (index_t l = 0; l < k; ++l) {
        BATMAT_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A.load(ii, l);
            BATMAT_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj  = shiftl<Conf.shift_B>(B.load(l, jj));
                if constexpr (Conf.negate)
                    Cij -= Ail * Blj;
                else
                    Cij += Ail * Blj;
            }
        }
    }
    // Store accumulator to memory again
    BATMAT_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        BATMAT_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            C_cached.template store<Conf.shift_D>(rotr<Conf.shift_D>(C_reg[ii][jj]), ii, jj);
}

/// Generalized matrix multiplication D = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Single register block.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg>
[[gnu::hot, gnu::flatten]] void gemm_copy_microkernel(const uview<const T, Abi, Conf.order_A> A,
                                                      const uview<const T, Abi, Conf.order_B> B,
                                                      const uview<const T, Abi, Conf.order_C> C,
                                                      const uview<T, Abi, Conf.order_D> D,
                                                      const index_t k) noexcept {
    using namespace ops;
    using simd = stdx::simd<T, Abi>;
    // The following assumption ensures that there is no unnecessary branch
    // for k == 0 in between the loops. This is crucial for good code
    // generation, otherwise the compiler inserts jumps and labels between
    // the matmul kernel and the loading/storing of C, which will cause it to
    // place C_reg on the stack, resulting in many unnecessary loads and stores.
    BATMAT_ASSUME(k > 0);
    // Pre-compute the offsets of the columns of C
    auto C_cached = with_cached_access<RowsReg, ColsReg>(C);
    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    BATMAT_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        BATMAT_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            C_reg[ii][jj] = rotl<Conf.shift_C>(C_cached.load(ii, jj));
    // Actual matrix multiplication kernel
    for (index_t l = 0; l < k; ++l) {
        BATMAT_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A.load(ii, l);
            BATMAT_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj  = shiftl<Conf.shift_B>(B.load(l, jj));
                if constexpr (Conf.negate)
                    Cij -= Ail * Blj;
                else
                    Cij += Ail * Blj;
            }
        }
    }
    auto D_cached = with_cached_access<RowsReg, ColsReg>(D);
    // Store accumulator to memory again
    BATMAT_FULLY_UNROLLED_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        BATMAT_FULLY_UNROLLED_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            D_cached.template store<Conf.shift_D>(rotr<Conf.shift_D>(C_reg[ii][jj]), ii, jj);
}

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Using register blocking.
template <class T, class Abi, KernelConfig Conf>
void gemm_register(const view<const T, Abi, Conf.order_A> A,
                   const view<const T, Abi, Conf.order_B> B, const view<T, Abi, Conf.order_C> C,
                   const bool init_zero) noexcept {
    constexpr auto Rows = RowsReg<T, Abi>, Cols = ColsReg<T, Abi>;
    const index_t I = C.rows(), J = C.cols(), K = A.cols();
    BATMAT_ASSUME(I > 0);
    BATMAT_ASSUME(J > 0);
    BATMAT_ASSUME(K > 0);
    static const auto microkernel              = gemm_lut<T, Abi, Conf>;
    const uview<const T, Abi, Conf.order_A> A_ = A;
    const uview<const T, Abi, Conf.order_B> B_ = B;
    const uview<T, Abi, Conf.order_C> C_       = C;
    // Optimization for very small matrices
    if (I <= Rows && J <= Cols)
        return microkernel[I - 1][J - 1](A_, B_, C_, K, init_zero);
    // Simply loop over all blocks in the given matrices.
    for (index_t j = 0; j < J; j += Cols) {
        const auto nj = std::min<index_t>(Cols, J - j);
        const auto Bj = B_.middle_cols(j);
        for (index_t i = 0; i < I; i += Rows) {
            const index_t ni = std::min<index_t>(Rows, I - i);
            const auto Ai    = A_.middle_rows(i);
            const auto Cij   = C_.block(i, j);
            microkernel[ni - 1][nj - 1](Ai, Bj, Cij, K, init_zero);
        }
    }
}

/// Generalized matrix multiplication D = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Using register blocking.
template <class T, class Abi, KernelConfig Conf>
void gemm_copy_register(const view<const T, Abi, Conf.order_A> A,
                        const view<const T, Abi, Conf.order_B> B,
                        const view<const T, Abi, Conf.order_C> C,
                        const view<T, Abi, Conf.order_D> D) noexcept {
    constexpr auto Rows = RowsReg<T, Abi>, Cols = ColsReg<T, Abi>;
    const index_t I = C.rows(), J = C.cols(), K = A.cols();
    BATMAT_ASSUME(I > 0);
    BATMAT_ASSUME(J > 0);
    BATMAT_ASSUME(K > 0);
    static const auto microkernel              = gemm_copy_lut<T, Abi, Conf>;
    const uview<const T, Abi, Conf.order_A> A_ = A;
    const uview<const T, Abi, Conf.order_B> B_ = B;
    const uview<const T, Abi, Conf.order_C> C_ = C;
    const uview<T, Abi, Conf.order_D> D_       = D;
    // Optimization for very small matrices
    if (I <= Rows && J <= Cols)
        return microkernel[I - 1][J - 1](A_, B_, C_, D_, K);
    // Simply loop over all blocks in the given matrices.
    for (index_t j = 0; j < J; j += Cols) {
        const auto nj = std::min<index_t>(Cols, J - j);
        const auto Bj = B_.middle_cols(j);
        for (index_t i = 0; i < I; i += Rows) {
            const index_t ni = std::min<index_t>(Rows, I - i);
            const auto Ai    = A_.middle_rows(i);
            const auto Cij   = C_.block(i, j);
            const auto Dij   = D_.block(i, j);
            microkernel[ni - 1][nj - 1](Ai, Bj, Cij, Dij, K);
        }
    }
}

} // namespace batmat::linalg::micro_kernels::gemm
