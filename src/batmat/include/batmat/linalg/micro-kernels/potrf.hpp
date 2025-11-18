#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/platform/platform.hpp>

namespace batmat::linalg::micro_kernels::potrf {

struct KernelConfig {
    bool negate_A           = false;
    MatrixStructure struc_C = MatrixStructure::LowerTriangular;
};

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, StorageOrder O1, StorageOrder O2>
void potrf_copy_microkernel(uview<const T, Abi, O1> A1, uview<const T, Abi, O2> A2,
                            uview<const T, Abi, O2> C, uview<T, Abi, O2> D, T *invD, index_t k1,
                            index_t k2, real_t regularization) noexcept;

template <class T, class Abi, KernelConfig Conf, index_t RowsReg, index_t ColsReg, StorageOrder O1,
          StorageOrder O2>
void trsm_copy_microkernel(uview<const T, Abi, O1> A1, uview<const T, Abi, O1> B1,
                           uview<const T, Abi, O2> A2, uview<const T, Abi, O2> B2,
                           uview<const T, Abi, O2> L, const T *invL, uview<const T, Abi, O2> C,
                           uview<T, Abi, O2> D, index_t k1, index_t k2) noexcept;

template <class T, class Abi, KernelConfig Conf, StorageOrder OA, StorageOrder OCD>
void potrf_copy_register(view<const T, Abi, OA> A, view<const T, Abi, OCD> C, view<T, Abi, OCD> D,
                         real_t regularization) noexcept;

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
