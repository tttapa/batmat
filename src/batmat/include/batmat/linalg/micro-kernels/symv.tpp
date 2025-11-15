#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/symv.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/ops/rotate.hpp>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::symv {

/// Symmetric matrix-vector multiplication d = c ± A b. Single register block.
template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder OA>
[[gnu::hot, gnu::flatten]] void
symv_copy_microkernel(const uview<const T, Abi, OA> A,
                      const uview<const T, Abi, StorageOrder::ColMajor> B,
                      const std::optional<uview<const T, Abi, StorageOrder::ColMajor>> C,
                      const uview<T, Abi, StorageOrder::ColMajor> D, const index_t k) noexcept {
    static_assert(RowsReg > 0);
    using enum MatrixStructure;
    using namespace ops;
    using simd = datapar::simd<T, Abi>;
    BATMAT_ASSUME(k >= RowsReg);

    // TODO: optimize for row-major case

    // Load B and C into registers
    simd B_reg[RowsReg], C_reg[RowsReg]; // NOLINT(*-c-arrays)
    UNROLL_FOR (index_t l = 0; l < RowsReg; ++l) {
        B_reg[l] = B.load(l, 0);
        C_reg[l] = C ? C->load(l, 0) : simd{0};
    }
    // Matrix-vector multiplication kernel (diagonal block)
    const auto A_cached = with_cached_access<0, RowsReg>(A);
    UNROLL_FOR (index_t ll = 0; ll < RowsReg; ++ll) {
        auto Blj = B_reg[ll];
        auto All = A_cached.load(ll, ll);
        Conf.negate ? (C_reg[ll] -= All * Blj) : (C_reg[ll] += All * Blj);
        UNROLL_FOR (index_t ii = ll + 1; ii < RowsReg; ++ii) {
            auto Ail = A_cached.load(ii, ll);
            auto Bil = B_reg[ii];
            Conf.negate ? (C_reg[ii] -= Ail * Blj) : (C_reg[ii] += Ail * Blj);
            Conf.negate ? (C_reg[ll] -= Ail * Bil) : (C_reg[ll] += Ail * Bil);
        }
    }
    // Matrix-vector multiplication kernel (subdiagonal block)
    for (index_t i = RowsReg; i < k; ++i) {
        auto Cij = C ? C->load(i, 0) : simd{0};
        UNROLL_FOR (index_t ll = 0; ll < RowsReg; ++ll) {
            auto Blj = B_reg[ll];
            auto Ail = A_cached.load(i, ll);
            auto Bil = B.load(i, 0);
            Conf.negate ? (Cij -= Ail * Blj) : (Cij += Ail * Blj);
            Conf.negate ? (C_reg[ll] -= Ail * Bil) : (C_reg[ll] += Ail * Bil);
        }
        D.store(Cij, i, 0);
    }
    UNROLL_FOR (index_t ll = 0; ll < RowsReg; ++ll)
        D.store(C_reg[ll], ll, 0);
}

/// Generalized matrix multiplication d = c ± A⁽ᵀ⁾ b. Using register blocking.
template <class T, class Abi, KernelConfig Conf, StorageOrder OA>
void symv_copy_register(const view<const T, Abi, OA> A, const view<const T, Abi> B,
                        const std::optional<view<const T, Abi>> C, const view<T, Abi> D) noexcept {
    using enum MatrixStructure;
    constexpr auto Rows = RowsReg<T, Abi>;
    // Check dimensions
    const index_t I = D.rows();
    BATMAT_ASSUME(A.rows() == I);
    BATMAT_ASSUME(A.cols() == I);
    BATMAT_ASSUME(B.rows() == I);
    BATMAT_ASSUME(B.cols() == 1);
    BATMAT_ASSUME(D.cols() == 1);
    BATMAT_ASSUME(I > 0);
    static const auto microkernel = symv_copy_lut<T, Abi, Conf, OA>;
    // Sizeless views to partition and pass to the micro-kernels
    const uview<const T, Abi, OA> A_                                    = A;
    const uview<const T, Abi, StorageOrder::ColMajor> B_                = B;
    const std::optional<uview<const T, Abi, StorageOrder::ColMajor>> C_ = C;
    const uview<T, Abi, StorageOrder::ColMajor> D_                      = D;

    if (I <= Rows)
        return microkernel[I - 1](A_, B_, C_, D_, I);
    microkernel[Rows - 1](A_, B_, C_, D_, I);
    foreach_chunked_merged(Rows, I, Rows, [&](index_t k, auto nk) {
        auto Dk = D_.middle_rows(k);
        microkernel[nk - 1](A_.block(k, k), B_.middle_rows(k), Dk, Dk, I - k);
    });
}

} // namespace batmat::linalg::micro_kernels::symv
