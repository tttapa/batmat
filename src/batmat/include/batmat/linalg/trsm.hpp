#pragma once

#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/micro-kernels/trsm.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

// TODO: move these to shift.hpp (but make sure the concepts for gemm are still correct)
template <int I>
struct with_rotate_A_t : std::integral_constant<int, I> {};
template <int I>
struct with_rotate_B_t : std::integral_constant<int, I> {};
template <int I>
inline constexpr with_rotate_A_t<I> with_rotate_A;
template <int I>
inline constexpr with_rotate_B_t<I> with_rotate_B;

namespace detail {
template <class T, class Abi, micro_kernels::trsm::KernelConfig Conf, StorageOrder OA,
          StorageOrder OB, StorageOrder OD>
    requires(Conf.struc_A != MatrixStructure::General)
void trsm(view<const T, Abi, OA> A, view<const T, Abi, OB> B, view<T, Abi, OD> D) {
    // Check dimensions
    BATMAT_ASSERT(A.rows() == A.cols()); // TODO: could be relaxed
    BATMAT_ASSERT(A.cols() == B.rows());
    BATMAT_ASSERT(B.rows() == D.rows());
    BATMAT_ASSERT(B.cols() == D.cols());
    const index_t M = A.rows(), K = A.cols(), N = B.cols();
    [[maybe_unused]] const auto fc = flops::trsm(M, N);
    GUANAQO_TRACE("trsm", 0, total(fc) * D.depth());
    // Degenerate case
    if (M == 0 || N == 0 || K == 0) [[unlikely]]
        return;
    return micro_kernels::trsm::trsm_copy_register<T, Abi, Conf>(A, B, D);
}
} // namespace detail

/// D = A⁻¹ B with A triangular
template <MatrixStructure SA, simdifiable VA, simdifiable VB, simdifiable VD, int RotB = 0>
    requires simdify_compatible<VA, VB, VD>
void trsm(Structured<VA, SA> A, VB &&B, VD &&D, with_rotate_B_t<RotB> = {}) {
    detail::trsm<simdified_value_t<VA>, simdified_abi_t<VA>, {.struc_A = SA, .rotate_B = RotB}>(
        simdify(A.value).as_const(), simdify(B).as_const(), simdify(D));
}
/// D = A⁻¹ D with A triangular
template <MatrixStructure SA, simdifiable VA, simdifiable VD, int RotB = 0>
    requires simdify_compatible<VA, VD>
void trsm(Structured<VA, SA> A, VD &&D, with_rotate_B_t<RotB> shift = {}) {
    trsm(A.ref(), D, D, shift);
}

/// D = A B⁻¹ with B triangular
template <MatrixStructure SB, simdifiable VA, simdifiable VB, simdifiable VD, int RotA = 0>
    requires simdify_compatible<VA, VB, VD>
void trsm(VA &&A, Structured<VB, SB> B, VD &&D, with_rotate_A_t<RotA> = {}) {
    // D = B A⁻¹  <=>  Dᵀ = A⁻ᵀ Bᵀ
    detail::trsm<simdified_value_t<VA>, simdified_abi_t<VA>,
                 {.struc_A = transpose(SB), .rotate_B = RotA}>(
        simdify(B.value).transposed().as_const(), simdify(A).transposed().as_const(),
        simdify(D).transposed());
}
/// D = D B⁻¹ with B triangular
template <MatrixStructure SB, simdifiable VB, simdifiable VD, int RotA = 0>
    requires simdify_compatible<VB, VD>
void trsm(VD &&D, Structured<VB, SB> B, with_rotate_A_t<RotA> shift = {}) {
    trsm(D, B.ref(), D, shift);
}

} // namespace batmat::linalg
