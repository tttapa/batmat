#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/platform/platform.hpp>

namespace batmat::linalg::micro_kernels::potrf {

struct KernelConfig {
    bool negate_A = false; ///< Whether to compute chol(C - AAᵀ) instead of chol(C + AAᵀ)
    enum {
        none,           ///< chol(C ± AAᵀ)
        diag,           ///< chol(C ± AΣAᵀ) with Σ diagonal
        diag_sign_only, ///< chol(C ± AΣAᵀ) with Σ diagonal and containing only ±0 (just sign bits)
    } diag_A                = none;
    MatrixStructure struc_C = MatrixStructure::LowerTriangular;
    [[nodiscard]] constexpr bool with_diag() const noexcept { return diag_A != none; }
};

template <class T, class Abi, KernelConfig Conf>
using diag_uview_type = std::conditional_t<Conf.with_diag(), uview_vec<T, Abi>, std::false_type>;
template <class T, class Abi, KernelConfig Conf>
using diag_view_type = std::conditional_t<Conf.with_diag(), view<T, Abi>, std::false_type>;

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder O1, StorageOrder O2>
void potrf_copy_microkernel(uview<const T, Abi, O1> A1, uview<const T, Abi, O2> A2,
                            uview<const T, Abi, O2> C, uview<T, Abi, O2> D, T *invD, index_t k1,
                            index_t k2, T regularization,
                            diag_uview_type<const T, Abi, Conf> diag) noexcept;

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder O1,
          StorageOrder O2>
void trsm_copy_microkernel(uview<const T, Abi, O1> A1, uview<const T, Abi, O1> B1,
                           uview<const T, Abi, O2> A2, uview<const T, Abi, O2> B2,
                           uview<const T, Abi, O2> L, const T *invL, uview<const T, Abi, O2> C,
                           uview<T, Abi, O2> D, index_t k1, index_t k2,
                           diag_uview_type<const T, Abi, Conf> diag) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OCD>
void potrf_copy_register(view<const T, Abi, OA> A, view<const T, Abi, OCD> C, view<T, Abi, OCD> D,
                         T regularization, diag_view_type<const T, Abi, Conf> diag) noexcept;

// Square block sizes greatly simplify handling of triangular matrices.
using gemm::RowsReg;
template <class T, class Abi>
constexpr index_t ColsReg = RowsReg<T, Abi>;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OC>
inline const constinit auto potrf_copy_lut =
    make_1d_lut<RowsReg<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return potrf_copy_microkernel<T, Abi, Conf, Row + 1, OA, OC>;
    });

template <class T, class Abi, KernelConfig Conf, StorageOrder O1, StorageOrder O2>
inline const constinit auto trsm_copy_lut = make_2d_lut<RowsReg<T, Abi>, ColsReg<T, Abi>>(
    []<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
        return trsm_copy_microkernel<T, Abi, Conf, Row + 1, Col + 1, O1, O2>;
    });

} // namespace batmat::linalg::micro_kernels::potrf
