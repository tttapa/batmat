#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/platform/platform.hpp>
#include <batmat/simd.hpp>

namespace batmat::linalg::micro_kernels::hyhound {

template <class T, class Abi, index_t R>
struct triangular_accessor {
    using value_type = T;
    value_type *data;

    using simd = datapar::simd<std::remove_const_t<T>, Abi>;
    static constexpr ptrdiff_t inner_stride =
        datapar::simd_size<std::remove_const_t<T>, Abi>::value;

    static constexpr index_t num_elem_per_layer() { return R * (R + 1) / 2; }
    static constexpr size_t size() {
        return simd::size() * static_cast<size_t>(num_elem_per_layer());
    }
    static constexpr size_t alignment() {
        return datapar::simd_align<std::remove_const_t<T>, Abi>::value;
    }

    [[gnu::always_inline]] value_type &operator()(index_t r, index_t c) const noexcept {
        assert(r <= c);
        return data[(r + c * (c + 1) / 2) * inner_stride];
    }
    [[gnu::always_inline]] simd load(index_t r, index_t c) const noexcept {
        return datapar::aligned_load<simd>(&operator()(r, c));
    }
    [[gnu::always_inline]] void store(simd x, index_t r, index_t c) const noexcept
        requires(!std::is_const_v<T>)
    {
        datapar::aligned_store(x, &operator()(r, c));
    }

    [[gnu::always_inline]] triangular_accessor(value_type *data) noexcept : data{data} {}
    operator triangular_accessor<const T, Abi, R>() const noexcept { return {data}; }
};

template <class T, class Abi>
inline constexpr index_t SizeR = gemm::RowsReg<T, Abi>; // TODO
template <class T, class Abi>
inline constexpr index_t SizeS = gemm::RowsReg<T, Abi>; // TODO

template <class T, class Abi, index_t R, StorageOrder OL, StorageOrder OA, bool SignOnly>
void xshhud_diag_diag_microkernel(index_t colsA, triangular_accessor<T, Abi, SizeR<T, Abi>> W,
                                  uview<T, Abi, OL> L, uview<T, Abi, OA> A,
                                  uview<const T, Abi, StorageOrder::ColMajor> diag) noexcept;

template <class T, class Abi, index_t R, StorageOrder OL, StorageOrder OA, bool SignOnly>
void xshhud_diag_full_microkernel(index_t colsA, uview<T, Abi, OL> L, uview<T, Abi, OA> A,
                                  uview<const T, Abi, StorageOrder::ColMajor> diag) noexcept;

enum class Structure {
    General = 0,
    Zero    = 1,
    Upper   = 2,
};

template <class T, class Abi, index_t R, index_t S, StorageOrder OL, StorageOrder OA, bool SignOnly>
void xshhud_diag_tail_microkernel(index_t kA_nonzero_start, index_t kA_nonzero_end, index_t colsA,
                                  triangular_accessor<const T, Abi, SizeR<T, Abi>> W,
                                  uview<T, Abi, OL> L, uview<const T, Abi, OA> A_in,
                                  uview<T, Abi, OA> A_out, uview<const T, Abi, OA> B,
                                  uview<const T, Abi, StorageOrder::ColMajor> diag,
                                  Structure struc_L, int rotate_A) noexcept;

template <class T, class Abi, StorageOrder OL, StorageOrder OA, bool SignOnly>
inline const constinit auto microkernel_diag_lut =
    make_1d_lut<SizeR<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return xshhud_diag_diag_microkernel<T, Abi, Row + 1, OL, OA, SignOnly>;
    });

template <class T, class Abi, StorageOrder OL, StorageOrder OA, bool SignOnly>
inline const constinit auto microkernel_full_lut =
    make_1d_lut<SizeR<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return xshhud_diag_full_microkernel<T, Abi, Row + 1, OL, OA, SignOnly>;
    });

template <class T, class Abi, StorageOrder OL, StorageOrder OA, bool SignOnly>
inline const constinit auto microkernel_tail_lut =
    make_1d_lut<SizeS<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return xshhud_diag_tail_microkernel<T, Abi, SizeR<T, Abi>, Row + 1, OL, OA, SignOnly>;
    });

template <class T, class Abi, StorageOrder OL, StorageOrder OA, bool SignOnly>
inline const constinit auto microkernel_tail_lut_2 = make_2d_lut<SizeR<T, Abi>, SizeS<T, Abi>>(
    []<index_t NR, index_t NS>(index_constant<NR>, index_constant<NS>) {
        return xshhud_diag_tail_microkernel<T, Abi, NR + 1, NS + 1, OL, OA, SignOnly>;
    });

} // namespace batmat::linalg::micro_kernels::hyhound
