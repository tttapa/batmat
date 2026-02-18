#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/gemv.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/ops/rotate.hpp>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::gemv {

/// Generalized matrix-vector multiplication d = c ± A⁽ᵀ⁾ b. Single register block.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder OA>
[[gnu::hot, gnu::flatten]] void
gemv_copy_microkernel(const uview<const T, Abi, OA> A,
                      const uview<const T, Abi, StorageOrder::ColMajor> B,
                      const std::optional<uview<const T, Abi, StorageOrder::ColMajor>> C,
                      const uview<T, Abi, StorageOrder::ColMajor> D, const index_t k) noexcept {
    static_assert(RowsReg > 0);
    using enum MatrixStructure;
    using namespace ops;
    using simd = datapar::simd<T, Abi>;
    BATMAT_ASSUME(k > 0);
    if constexpr (OA == StorageOrder::RowMajor) {
        // Load accumulator into registers
        simd C_reg[RowsReg]; // NOLINT(*-c-arrays)
        if (C) [[likely]] {
            UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
                C_reg[ii] = rotl<Conf.rotate_C>(C->load(ii, 0));
        } else {
            UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
                C_reg[ii] = simd{0};
        }
        // Matrix-vector multiplication kernel
        const auto A_cached = with_cached_access<RowsReg, 0>(A);
        for (index_t l = 0; l < k; ++l) {
            UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii) {
                simd Ail  = shiftl<Conf.shift_A>(A_cached.load(ii, l));
                simd &Cij = C_reg[ii];
                simd Blj  = rotl<Conf.rotate_B>(B.load(l, 0));
                Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
            }
        }
        // Store accumulator to memory again
        UNROLL_FOR (index_t ii = 0; ii < RowsReg; ++ii)
            D.template store<Conf.mask_D>(rotr<Conf.rotate_D>(C_reg[ii]), ii, 0);
    } else {
        // Load B into registers
        simd B_reg[RowsReg]; // NOLINT(*-c-arrays)
        UNROLL_FOR (index_t l = 0; l < RowsReg; ++l)
            B_reg[l] = rotl<Conf.rotate_B>(B.load(l, 0));
        // Matrix-vector multiplication kernel
        const auto A_cached = with_cached_access<0, RowsReg>(A);
        if (C) [[likely]] {
            for (index_t i = 0; i < k; ++i) {
                simd Cij = rotl<Conf.rotate_C>(C->load(i, 0));
                UNROLL_FOR (index_t ll = 0; ll < RowsReg; ++ll) {
                    simd Ail = shiftl<Conf.shift_A>(A_cached.load(i, ll));
                    Conf.negate ? (Cij -= Ail * B_reg[ll]) : (Cij += Ail * B_reg[ll]);
                }
                D.template store<Conf.mask_D>(rotr<Conf.rotate_D>(Cij), i, 0);
            }
        } else {
            for (index_t i = 0; i < k; ++i) {
                simd Cij{0};
                UNROLL_FOR (index_t ll = 0; ll < RowsReg; ++ll) {
                    simd Ail = shiftl<Conf.shift_A>(A_cached.load(i, ll));
                    Conf.negate ? (Cij -= Ail * B_reg[ll]) : (Cij += Ail * B_reg[ll]);
                }
                D.template store<Conf.mask_D>(rotr<Conf.rotate_D>(Cij), i, 0);
            }
        }
    }
}

/// Generalized matrix multiplication d = c ± A⁽ᵀ⁾ b. Using register blocking.
template <class T, class Abi, KernelConfig Conf, StorageOrder OA>
void gemv_copy_register(const view<const T, Abi, OA> A, const view<const T, Abi> B,
                        const std::optional<view<const T, Abi>> C, const view<T, Abi> D) noexcept {
    using enum MatrixStructure;
    constexpr auto Rows = RowsReg<T, Abi>;
    // Check dimensions
    const index_t I = D.rows(), K = A.cols();
    BATMAT_ASSUME(A.rows() == I);
    BATMAT_ASSUME(B.rows() == K);
    BATMAT_ASSUME(B.cols() == 1);
    BATMAT_ASSUME(D.cols() == 1);
    BATMAT_ASSUME(I > 0);
    BATMAT_ASSUME(K > 0);
    static const auto microkernel = gemv_copy_lut<T, Abi, Conf, OA>;
    // Sizeless views to partition and pass to the micro-kernels
    const uview<const T, Abi, OA> A_                                    = A;
    const uview<const T, Abi, StorageOrder::ColMajor> B_                = B;
    const std::optional<uview<const T, Abi, StorageOrder::ColMajor>> C_ = C;
    const uview<T, Abi, StorageOrder::ColMajor> D_                      = D;

    if constexpr (OA == StorageOrder::RowMajor) {
        if (I <= Rows)
            return microkernel[I - 1](A_, B_, C_, D_, K);
        foreach_chunked_merged(0, I, Rows, [&](index_t i, auto ni) {
            auto Cj = C_ ? std::make_optional(C_->middle_rows(i)) : std::nullopt;
            microkernel[ni - 1](A_.middle_rows(i), B_, Cj, D_.middle_rows(i), K);
        });
    } else {
        if (K <= Rows)
            return microkernel[K - 1](A_, B_, C_, D_, I);
        microkernel[Rows - 1](A_.middle_cols(0), B_.middle_rows(0), C_, D_, I);
        foreach_chunked_merged(Rows, K, Rows, [&](index_t k, auto nk) {
            microkernel[nk - 1](A_.middle_cols(k), B_.middle_rows(k), D_, D_, I);
        });
    }
}

} // namespace batmat::linalg::micro_kernels::gemv
