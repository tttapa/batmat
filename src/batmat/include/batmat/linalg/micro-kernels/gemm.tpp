#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/gemm.hpp>
#include <batmat/ops/rotate.hpp>
#include "batmat/linalg/uview.hpp"

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::gemm {

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Single register block.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OA,
          StorageOrder OB, StorageOrder OC>
[[gnu::hot, gnu::flatten]] void
gemm_microkernel(const uview<const T, Abi, OA> A, const uview<const T, Abi, OB> B,
                 const uview<T, Abi, OC> C, const index_t k, bool init_zero) noexcept {
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
        UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj)
            UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
                C_reg[ii][jj] = rotl<Conf.shift_C>(C_cached.load(ii, jj));
    // Actual matrix multiplication kernel
    for (index_t l = 0; l < k; ++l) {
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = shiftl<Conf.shift_A>(A.load(ii, l));
            UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj  = shiftl<Conf.shift_B>(B.load(l, jj));
                Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
            }
        }
    }
    // Store accumulator to memory again
    UNROLL_FOR (index_t jj = 0; jj < ColsReg; ++jj)
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            C_cached.template store<Conf.shift_D>(rotr<Conf.shift_D>(C_reg[ii][jj]), ii, jj);
}

template <MatrixStructure Struc>
inline constexpr auto first_column =
    [](index_t row_index) { return Struc == MatrixStructure::UpperTriangular ? row_index : 0; };

template <index_t ColsReg, MatrixStructure Struc>
inline constexpr auto last_column = [](index_t row_index) {
    return Struc == MatrixStructure::LowerTriangular ? std::min(row_index, ColsReg - 1)
                                                     : ColsReg - 1;
};

/// Generalized matrix multiplication D = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Single register block.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OA,
          StorageOrder OB, StorageOrder OC, StorageOrder OD>
[[gnu::hot, gnu::flatten]] void
gemm_copy_microkernel(const uview<const T, Abi, OA> A, const uview<const T, Abi, OB> B,
                      const std::optional<uview<const T, Abi, OC>> C, const uview<T, Abi, OD> D,
                      const index_t k) noexcept {
    static_assert(RowsReg > 0 && ColsReg > 0);
    using enum MatrixStructure;
    using namespace ops;
    using simd = stdx::simd<T, Abi>;
    // Column range for triangular matrix C (gemmt)
    static constexpr auto min_col = first_column<Conf.struc_C>;
    static constexpr auto max_col = last_column<ColsReg, Conf.struc_C>;
    // The following assumption ensures that there is no unnecessary branch
    // for k == 0 in between the loops. This is crucial for good code
    // generation, otherwise the compiler inserts jumps and labels between
    // the matmul kernel and the loading/storing of C, which will cause it to
    // place C_reg on the stack, resulting in many unnecessary loads and stores.
    BATMAT_ASSUME(k > 0);
    // Check dimensions in the triangular case
    if constexpr (Conf.struc_A != General)
        BATMAT_ASSUME(k >= RowsReg);
    if constexpr (Conf.struc_B != General)
        BATMAT_ASSUME(k >= ColsReg);
    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    if (C) {
        const auto C_cached = with_cached_access<RowsReg, ColsReg>(*C);
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            UNROLL_FOR (index_t jj = min_col(ii); jj <= max_col(ii); ++jj)
                C_reg[ii][jj] = rotl<Conf.shift_C>(C_cached.load(ii, jj));
    } else {
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            UNROLL_FOR (index_t jj = min_col(ii); jj <= max_col(ii); ++jj)
                C_reg[ii][jj] = simd{0};
    }

    const auto A_cached = with_cached_access<RowsReg, 0>(A);
    const auto B_cached = with_cached_access<0, ColsReg>(B);

    // Triangular matrix multiplication kernel
    index_t l = 0;
    if constexpr (Conf.struc_A == UpperTriangular && Conf.struc_B == LowerTriangular) {
        l += std::max(RowsReg, ColsReg);
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            UNROLL_FOR (index_t ll = ii; ll < std::max(RowsReg, ColsReg); ++ll) {
                simd Ail = shiftl<Conf.shift_A>(A_cached.load(ii, ll));
                UNROLL_FOR (index_t jj = min_col(ii); jj <= std::min(ll, max_col(ii)); ++jj) {
                    simd &Cij = C_reg[ii][jj];
                    simd Blj  = shiftl<Conf.shift_B>(B_cached.load(ll, jj));
                    Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
                }
            }
        }
    } else if constexpr (Conf.struc_A == UpperTriangular) {
        l += RowsReg;
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            UNROLL_FOR (index_t ll = ii; ll < RowsReg; ++ll) {
                simd Ail = shiftl<Conf.shift_A>(A_cached.load(ii, ll));
                UNROLL_FOR (index_t jj = min_col(ii); jj <= max_col(ii); ++jj) {
                    simd &Cij = C_reg[ii][jj];
                    simd Blj  = shiftl<Conf.shift_B>(B_cached.load(ll, jj));
                    Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
                }
            }
        }
    } else if constexpr (Conf.struc_B == LowerTriangular) {
        l += ColsReg;
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            UNROLL_FOR (index_t ll = 0; ll < ColsReg; ++ll) {
                simd Ail = shiftl<Conf.shift_A>(A_cached.load(ii, ll));
                UNROLL_FOR (index_t jj = min_col(ii); jj <= std::min(ll, max_col(ii)); ++jj) {
                    simd &Cij = C_reg[ii][jj];
                    simd Blj  = shiftl<Conf.shift_B>(B_cached.load(ll, jj));
                    Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
                }
            }
        }
    }

    // Rectangular matrix multiplication kernel
    const index_t l_end_A = Conf.struc_A == LowerTriangular ? k - RowsReg : k;
    const index_t l_end_B = Conf.struc_B == UpperTriangular ? k - ColsReg : k;
    for (; l < std::min(l_end_A, l_end_B); ++l) {
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = shiftl<Conf.shift_A>(A_cached.load(ii, l));
            UNROLL_FOR (index_t jj = min_col(ii); jj <= max_col(ii); ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj  = shiftl<Conf.shift_B>(B_cached.load(l, jj));
                Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
            }
        }
    }

    // Triangular matrix multiplication kernel
    if constexpr (Conf.struc_A == LowerTriangular && Conf.struc_B == UpperTriangular) {
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            UNROLL_FOR (index_t ll = 0; ll <= ii + std::max(ColsReg, RowsReg) - RowsReg; ++ll) {
                simd Ail = shiftl<Conf.shift_A>(A_cached.load(ii, l + ll));
                static_assert(std::is_signed_v<index_t>);
                const index_t j0 = ll - (std::max(RowsReg, ColsReg) - ColsReg);
                UNROLL_FOR (index_t jj = std::max(j0, min_col(ii)); jj <= max_col(ii); ++jj) {
                    simd &Cij = C_reg[ii][jj];
                    simd Blj  = shiftl<Conf.shift_B>(B_cached.load(l + ll, jj));
                    Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
                }
            }
        }
    } else if constexpr (Conf.struc_A == LowerTriangular) {
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            UNROLL_FOR (index_t ll = 0; ll <= ii; ++ll) {
                simd Ail = shiftl<Conf.shift_A>(A_cached.load(ii, l + ll));
                UNROLL_FOR (index_t jj = min_col(ii); jj <= max_col(ii); ++jj) {
                    simd &Cij = C_reg[ii][jj];
                    simd Blj  = shiftl<Conf.shift_B>(B_cached.load(l + ll, jj));
                    Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
                }
            }
        }
    } else if constexpr (Conf.struc_B == UpperTriangular) {
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            UNROLL_FOR (index_t ll = 0; ll < ColsReg; ++ll) {
                simd Ail = shiftl<Conf.shift_A>(A_cached.load(ii, l + ll));
                UNROLL_FOR (index_t jj = std::max(ll, min_col(ii)); jj <= max_col(ii); ++jj) {
                    simd &Cij = C_reg[ii][jj];
                    simd Blj  = shiftl<Conf.shift_B>(B_cached.load(l + ll, jj));
                    Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
                }
            }
        }
    }

    const auto D_cached = with_cached_access<RowsReg, ColsReg>(D);
    // Store accumulator to memory again
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (index_t jj = min_col(ii); jj <= max_col(ii); ++jj)
            D_cached.template store<Conf.shift_D>(rotr<Conf.shift_D>(C_reg[ii][jj]), ii, jj);
}

/// Generalized matrix multiplication C = C ± A⁽ᵀ⁾ B⁽ᵀ⁾. Using register blocking.
template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC>
void gemm_register(const view<const T, Abi, OA> A, const view<const T, Abi, OB> B,
                   const view<T, Abi, OC> C, const bool init_zero) noexcept {
    constexpr auto Rows = RowsReg<T, Abi>, Cols = ColsReg<T, Abi>;
    const index_t I = C.rows(), J = C.cols(), K = A.cols();
    BATMAT_ASSUME(I > 0);
    BATMAT_ASSUME(J > 0);
    BATMAT_ASSUME(K > 0);
    static const auto microkernel    = gemm_lut<T, Abi, Conf, OA, OB, OC>;
    const uview<const T, Abi, OA> A_ = A;
    const uview<const T, Abi, OB> B_ = B;
    const uview<T, Abi, OC> C_       = C;
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
template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC,
          StorageOrder OD>
void gemm_copy_register(const view<const T, Abi, OA> A, const view<const T, Abi, OB> B,
                        const std::optional<view<const T, Abi, OC>> C,
                        const view<T, Abi, OD> D) noexcept {
    using enum MatrixStructure;
    constexpr auto Rows = RowsReg<T, Abi>, Cols = ColsReg<T, Abi>;
    const index_t I = D.rows(), J = D.cols(), K = A.cols();
    BATMAT_ASSUME(A.rows() == I);
    BATMAT_ASSUME(B.rows() == K);
    BATMAT_ASSUME(B.cols() == J);
    if constexpr (Conf.struc_A != General)
        BATMAT_ASSUME(I == K);
    if constexpr (Conf.struc_B != General)
        BATMAT_ASSUME(K == J);
    if constexpr (Conf.struc_C != General)
        BATMAT_ASSUME(I == J);
    BATMAT_ASSUME(I > 0);
    BATMAT_ASSUME(J > 0);
    BATMAT_ASSUME(K > 0);
    constexpr KernelConfig ConfGXG{.negate  = Conf.negate,
                                   .shift_A = Conf.shift_A,
                                   .shift_B = Conf.shift_B,
                                   .shift_C = Conf.shift_C,
                                   .shift_D = Conf.shift_D,
                                   .struc_A = General,
                                   .struc_B = Conf.struc_B,
                                   .struc_C = General};
    constexpr KernelConfig ConfXGG{.negate  = Conf.negate,
                                   .shift_A = Conf.shift_A,
                                   .shift_B = Conf.shift_B,
                                   .shift_C = Conf.shift_C,
                                   .shift_D = Conf.shift_D,
                                   .struc_A = Conf.struc_A,
                                   .struc_B = General,
                                   .struc_C = General};
    constexpr KernelConfig ConfXXG{.negate  = Conf.negate,
                                   .shift_A = Conf.shift_A,
                                   .shift_B = Conf.shift_B,
                                   .shift_C = Conf.shift_C,
                                   .shift_D = Conf.shift_D,
                                   .struc_A = Conf.struc_A,
                                   .struc_B = Conf.struc_B,
                                   .struc_C = General};
    static const auto microkernel     = gemm_copy_lut<T, Abi, Conf, OA, OB, OC, OD>;
    static const auto microkernel_GXG = gemm_copy_lut<T, Abi, ConfGXG, OA, OB, OC, OD>;
    static const auto microkernel_XGG = gemm_copy_lut<T, Abi, ConfXGG, OA, OB, OC, OD>;
    static const auto microkernel_XXG = gemm_copy_lut<T, Abi, ConfXXG, OA, OB, OC, OD>;
    const uview<const T, Abi, OA> A_  = A;
    const uview<const T, Abi, OB> B_  = B;
    const std::optional<uview<const T, Abi, OC>> C_ = C;
    const uview<T, Abi, OD> D_                      = D;
    // Optimization for very small matrices
    if (I <= Rows && J <= Cols)
        return microkernel[I - 1][J - 1](A_, B_, C_, D_, K);
    // Simply loop over all blocks in the given matrices.
    for (index_t j = 0; j < J; j += Cols) {
        const auto nj = std::min<index_t>(Cols, J - j);
        const auto Bj = B_.middle_cols(j);
        const auto i0 = Conf.struc_C == LowerTriangular ? j : 0;
        const auto i1 = Conf.struc_C == UpperTriangular ? j + nj : I;
        for (index_t i = i0; i < i1; i += Rows) {
            const index_t ni = std::min<index_t>(Rows, I - i);
            const auto l0A   = Conf.struc_A == UpperTriangular ? i : 0;
            const auto l1A   = Conf.struc_A == LowerTriangular ? i + ni + std::max(K, I) - I : K;
            const auto l0B   = Conf.struc_B == LowerTriangular ? j : 0;
            const auto l1B   = Conf.struc_B == UpperTriangular ? j + nj + std::max(K, J) - J : K;
            const auto l0    = std::max(l0A, l0B);
            const auto l1    = std::min(l1A, l1B);
            const auto Ai    = A_.middle_rows(i);
            const auto Cij   = C_ ? std::make_optional(C_->block(i, j)) : std::nullopt;
            const auto Dij   = D_.block(i, j);
            const auto Ail   = Ai.middle_cols(l0);
            const auto Blj   = Bj.middle_rows(l0);

            if (l1 == l0)
                continue;
            if constexpr (Conf.struc_A == LowerTriangular && Conf.struc_B == UpperTriangular) {
                if (l1A > l1B) {
                    microkernel_GXG[ni - 1][nj - 1](Ail, Blj, Cij, Dij, l1 - l0);
                    continue;
                } else if (l1A < l1B) {
                    microkernel_XGG[ni - 1][nj - 1](Ail, Blj, Cij, Dij, l1 - l0);
                    continue;
                }
            }
            if constexpr (Conf.struc_A == UpperTriangular && Conf.struc_B == LowerTriangular) {
                if (l0A > l0B) {
                    microkernel_XGG[ni - 1][nj - 1](Ail, Blj, Cij, Dij, l1 - l0);
                    continue;
                } else if (l0A < l0B) {
                    microkernel_GXG[ni - 1][nj - 1](Ail, Blj, Cij, Dij, l1 - l0);
                    continue;
                }
            }
            if constexpr (Conf.struc_C != General) {
                if (i != j) {
                    microkernel_XXG[ni - 1][nj - 1](Ail, Blj, Cij, Dij, l1 - l0);
                    continue;
                }
            }
            microkernel[ni - 1][nj - 1](Ail, Blj, Cij, Dij, l1 - l0);
        }
    }
}

} // namespace batmat::linalg::micro_kernels::gemm
