#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/gemm-diag.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/ops/rotate.hpp>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::gemm_diag {

template <MatrixStructure Struc>
inline constexpr auto first_column =
    [](index_t row_index) { return Struc == MatrixStructure::UpperTriangular ? row_index : 0; };

template <index_t ColsReg, MatrixStructure Struc>
inline constexpr auto last_column = [](index_t row_index) {
    return Struc == MatrixStructure::LowerTriangular ? std::min(row_index, ColsReg - 1)
                                                     : ColsReg - 1;
};

/// Generalized matrix multiplication D = C ± A⁽ᵀ⁾ diag(d) B⁽ᵀ⁾. Single register block.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder OA,
          StorageOrder OB, StorageOrder OC, StorageOrder OD>
[[gnu::hot, gnu::flatten]] std::conditional_t<Conf.track_zeros, std::pair<index_t, index_t>, void>
gemm_diag_copy_microkernel(const uview<const T, Abi, OA> A, const uview<const T, Abi, OB> B,
                           const std::optional<uview<const T, Abi, OC>> C,
                           const uview<T, Abi, OD> D, const uview_vec<const T, Abi> d,
                           const index_t k) noexcept {
    static_assert(RowsReg > 0 && ColsReg > 0);
    using enum MatrixStructure;
    using namespace ops;
    using simd = datapar::simd<T, Abi>;
    // Column range for triangular matrix C (gemmt)
    static constexpr auto min_col = first_column<Conf.struc_C>;
    static constexpr auto max_col = last_column<ColsReg, Conf.struc_C>;
    // The following assumption ensures that there is no unnecessary branch
    // for k == 0 in between the loops. This is crucial for good code
    // generation, otherwise the compiler inserts jumps and labels between
    // the matmul kernel and the loading/storing of C, which will cause it to
    // place C_reg on the stack, resulting in many unnecessary loads and stores.
    BATMAT_ASSUME(k > 0);
    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg]; // NOLINT(*-c-arrays)
    if (C) [[likely]] {
        const auto C_cached = with_cached_access<RowsReg, ColsReg>(*C);
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            UNROLL_FOR (index_t jj = min_col(ii); jj <= max_col(ii); ++jj)
                C_reg[ii][jj] = C_cached.load(ii, jj);
    } else {
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            UNROLL_FOR (index_t jj = min_col(ii); jj <= max_col(ii); ++jj)
                C_reg[ii][jj] = simd{0};
    }

    const auto A_cached = with_cached_access<RowsReg, 0>(A);
    const auto B_cached = with_cached_access<0, ColsReg>(B);

    // Rectangular matrix multiplication kernel
    index_t first_nonzero = -1, last_nonzero = -1;
    for (index_t l = 0; l < k; ++l) {
        bool all_zero = true;
        simd dl       = d.load(l);
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
            simd Ail = dl * A_cached.load(ii, l);
            if constexpr (Conf.track_zeros)
                all_zero &= all_of(Ail == simd{0});
            UNROLL_FOR (index_t jj = min_col(ii); jj <= max_col(ii); ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj  = B_cached.load(l, jj);
                Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
            }
        }
        if constexpr (Conf.track_zeros)
            if (!all_zero) {
                last_nonzero = l;
                if (first_nonzero < 0)
                    first_nonzero = l;
            }
    }

    const auto D_cached = with_cached_access<RowsReg, ColsReg>(D);
    // Store accumulator to memory again
    UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (index_t jj = min_col(ii); jj <= max_col(ii); ++jj)
            D_cached.store(C_reg[ii][jj], ii, jj);

    if constexpr (Conf.track_zeros) {
        if (first_nonzero < 0)
            return {k, k};
        else
            return {first_nonzero, last_nonzero + 1};
    }
}

/// Generalized matrix multiplication D = C ± A⁽ᵀ⁾ diag(d) B⁽ᵀ⁾. Using register blocking.
template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OB, StorageOrder OC,
          StorageOrder OD>
void gemm_diag_copy_register(const view<const T, Abi, OA> A, const view<const T, Abi, OB> B,
                             const std::optional<view<const T, Abi, OC>> C,
                             const view<T, Abi, OD> D, view<const T, Abi> d) noexcept {
    using enum MatrixStructure;
    constexpr auto Rows = RowsReg<T, Abi>, Cols = ColsReg<T, Abi>;
    // Check dimensions
    const index_t I = D.rows(), J = D.cols(), K = A.cols();
    BATMAT_ASSUME(A.rows() == I);
    BATMAT_ASSUME(B.rows() == K);
    BATMAT_ASSUME(B.cols() == J);
    BATMAT_ASSUME(d.rows() == K);
    BATMAT_ASSUME(d.cols() == 1);
    if constexpr (Conf.struc_C != General)
        BATMAT_ASSUME(I == J);
    BATMAT_ASSUME(I > 0);
    BATMAT_ASSUME(J > 0);
    BATMAT_ASSUME(K > 0);
    // Configurations for the various micro-kernels
    constexpr KernelConfig ConfSmall{.negate = Conf.negate, .struc_C = Conf.struc_C};
    constexpr KernelConfig ConfSub{.negate = Conf.negate, .struc_C = General};
    static const auto microkernel       = gemm_diag_copy_lut<T, Abi, Conf, OA, OB, OC, OD>;
    static const auto microkernel_small = gemm_diag_copy_lut<T, Abi, ConfSmall, OA, OB, OC, OD>;
    static const auto microkernel_sub   = gemm_diag_copy_lut<T, Abi, ConfSub, OA, OB, OC, OD>;
    (void)microkernel_sub; // GCC incorrectly warns about unused variable
    // Sizeless views to partition and pass to the micro-kernels
    const uview<const T, Abi, OA> A_                = A;
    const uview<const T, Abi, OB> B_                = B;
    const std::optional<uview<const T, Abi, OC>> C_ = C;
    const uview<T, Abi, OD> D_                      = D;
    const uview_vec<const T, Abi> d_{d};

    // Optimization for very small matrices
    if (I <= Rows && J <= Cols)
        return microkernel_small[I - 1][J - 1](A_, B_, C_, D_, d_, K);

    // Loop over block rows of A and block columns of B
    foreach_chunked_merged(0, I, index_constant<Rows>(), [&](index_t i, auto ni) {
        const auto Ai = A_.middle_rows(i);
        // If triangular: use diagonal block (i, i) for counting zeros.
        // If general: use first block column (i, 0).
        const auto j0 = Conf.struc_C == UpperTriangular   ? i + ni
                        : Conf.struc_C == LowerTriangular ? 0
                                                          : std::min(Cols, J),
                   j1 = Conf.struc_C == LowerTriangular ? i : J;
        // First micro-kernel call that keeps track of the leading/trailing zeros in A diag(d)
        auto [l0, l1] = [&] {
            const auto j   = Conf.struc_C == General ? 0 : i;
            const auto nj  = std::min<index_t>(Cols, J - j);
            const auto Bj  = B_.middle_cols(j);
            const auto Cij = C_ ? std::make_optional(C_->block(i, j)) : std::nullopt;
            const auto Dij = D_.block(i, j);
            if constexpr (Conf.track_zeros)
                return microkernel[ni - 1][nj - 1](Ai, Bj, Cij, Dij, d_, K);
            else {
                microkernel[ni - 1][nj - 1](Ai, Bj, Cij, Dij, d_, K);
                return std::pair<index_t, index_t>{0, K};
            }
        }();
        if (l1 == l0) {
            if (!C)
                D.block(i, j0, ni, j1 - j0).set_constant(T{});
            else if (C->data() == D.data() && C->outer_stride() == D.outer_stride())
                BATMAT_ASSUME(C->storage_order == D.storage_order); // Nothing to do
            else if constexpr (OC == StorageOrder::ColMajor)
                for (index_t jj = j0; jj < j1; ++jj) // TODO: suboptimal when transpose required
                    for (index_t ii = i; ii < i + ni; ++ii)
                        D_.store(C_->load(ii, jj), ii, jj);
            else
                for (index_t ii = i; ii < i + ni; ++ii) // TODO: suboptimal when transpose required
                    for (index_t jj = j0; jj < j1; ++jj)
                        D_.store(C_->load(ii, jj), ii, jj);
            return;
        }
        // Process other blocks, trimming any leading/trailing zeros (before l0 or after l1)
        foreach_chunked_merged(j0, j1, index_constant<Cols>(), [&](index_t j, auto nj) {
            const auto Bj  = B_.middle_cols(j);
            const auto Cij = C_ ? std::make_optional(C_->block(i, j)) : std::nullopt;
            const auto Dij = D_.block(i, j);
            const auto Ail = Ai.middle_cols(l0);
            const auto Blj = Bj.middle_rows(l0);
            const auto dl  = d_.segment(l0);
            microkernel_sub[ni - 1][nj - 1](Ail, Blj, Cij, Dij, dl, l1 - l0);
        });
    });
}

} // namespace batmat::linalg::micro_kernels::gemm_diag
