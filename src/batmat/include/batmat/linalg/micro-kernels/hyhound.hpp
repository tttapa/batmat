#pragma once

#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/lut.hpp>
#include <batmat/platform/platform.hpp>
#include <batmat/simd.hpp>

namespace batmat::linalg::micro_kernels::hyhound {

struct KernelConfig {
    bool sign_only = false;
};

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

template <class T, class Abi, index_t R, StorageOrder OL, StorageOrder OA, KernelConfig Conf>
void hyhound_diag_diag_microkernel(index_t kA, triangular_accessor<T, Abi, SizeR<T, Abi>> W,
                                   uview<T, Abi, OL> L, uview<T, Abi, OA> A,
                                   uview<const T, Abi, StorageOrder::ColMajor> diag) noexcept;

template <class T, class Abi, index_t R, StorageOrder OL, StorageOrder OA, KernelConfig Conf>
void hyhound_diag_full_microkernel(index_t kA, uview<T, Abi, OL> L, uview<T, Abi, OA> A,
                                   uview<const T, Abi, StorageOrder::ColMajor> diag) noexcept;

enum class Structure {
    General = 0,
    Zero    = 1,
    Upper   = 2,
};

template <class T, class Abi, index_t R, index_t S, StorageOrder OL, StorageOrder OA,
          StorageOrder OB, KernelConfig Conf>
void hyhound_diag_tail_microkernel(index_t kA_nonzero_start, index_t kA_nonzero_end, index_t kA,
                                   triangular_accessor<const T, Abi, SizeR<T, Abi>> W,
                                   uview<T, Abi, OL> L, uview<const T, Abi, OA> A_in,
                                   uview<T, Abi, OA> A_out, uview<const T, Abi, OB> B,
                                   uview<const T, Abi, StorageOrder::ColMajor> diag,
                                   Structure struc_L, int rotate_A) noexcept;

template <class T, class Abi, StorageOrder OL, StorageOrder OA, KernelConfig Conf>
inline const constinit auto microkernel_diag_lut =
    make_1d_lut<SizeR<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return hyhound_diag_diag_microkernel<T, Abi, Row + 1, OL, OA, Conf>;
    });

template <class T, class Abi, StorageOrder OL, StorageOrder OA, KernelConfig Conf>
inline const constinit auto microkernel_full_lut =
    make_1d_lut<SizeR<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return hyhound_diag_full_microkernel<T, Abi, Row + 1, OL, OA, Conf>;
    });

template <class T, class Abi, StorageOrder OL, StorageOrder OA, StorageOrder OB, KernelConfig Conf>
inline const constinit auto microkernel_tail_lut =
    make_1d_lut<SizeS<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return hyhound_diag_tail_microkernel<T, Abi, SizeR<T, Abi>, Row + 1, OL, OA, OB, Conf>;
    });

template <class T, class Abi, StorageOrder OL, StorageOrder OA, StorageOrder OB, KernelConfig Conf>
inline const constinit auto microkernel_tail_lut_2 = make_2d_lut<SizeR<T, Abi>, SizeS<T, Abi>>(
    []<index_t NR, index_t NS>(index_constant<NR>, index_constant<NS>) {
        return hyhound_diag_tail_microkernel<T, Abi, NR + 1, NS + 1, OL, OA, OB, Conf>;
    });

// Helper function to compute size of the storage for the matrix W (part of the hyperbolic
// Householder representation).
template <class T, class Abi, StorageOrder OL>
constexpr std::pair<index_t, index_t> xshhud_W_size(view<T, Abi, OL> L) {
    static constexpr index_constant<SizeR<std::remove_const_t<T>, Abi>> R;
    using W_t = triangular_accessor<std::remove_const_t<T>, Abi, R>;
    return {W_t::num_elem_per_layer(), (L.cols() + R - 1) / R};
}

// Low-level register-blocked routines
template <class T, class Abi, KernelConfig Conf = {}, StorageOrder OL = StorageOrder::ColMajor,
          StorageOrder OA = StorageOrder::ColMajor>
void hyhound_diag_register(view<T, Abi, OL> L, view<T, Abi, OA> A, view<const T, Abi> D) noexcept;

template <class T, class Abi, KernelConfig Conf = {}, StorageOrder OL = StorageOrder::ColMajor,
          StorageOrder OA = StorageOrder::ColMajor>
void hyhound_diag_register(view<T, Abi, OL> L, view<T, Abi, OA> A, view<const T, Abi> D,
                           view<T, Abi> W) noexcept;

template <class T, class Abi, KernelConfig Conf = {}, StorageOrder OL = StorageOrder::ColMajor,
          StorageOrder OA = StorageOrder::ColMajor>
void hyhound_diag_apply_register(view<T, Abi, OL> L, view<const T, Abi, OA> Ain,
                                 view<T, Abi, OA> Aout, view<const T, Abi, OA> B,
                                 view<const T, Abi> D, view<const T, Abi> W,
                                 index_t kA_nonzero_start = 0,
                                 index_t kA_nonzero_end   = -1) noexcept;

template <class T, class Abi, StorageOrder OL1 = StorageOrder::ColMajor,
          StorageOrder OA1 = StorageOrder::ColMajor, StorageOrder OL2 = StorageOrder::ColMajor,
          StorageOrder OA2 = StorageOrder::ColMajor, KernelConfig Conf = {}>
void hyhound_diag_2_register(view<T, Abi, OL1> L11, view<T, Abi, OA1> A1, view<T, Abi, OL2> L21,
                             view<T, Abi, OA2> A2, view<const T, Abi> D) noexcept;

template <class T, class Abi, StorageOrder OL = StorageOrder::ColMajor,
          StorageOrder OW = StorageOrder::ColMajor, StorageOrder OY = StorageOrder::ColMajor,
          StorageOrder OU = StorageOrder::ColMajor, KernelConfig Conf = {}>
void hyhound_diag_cyclic_register(view<T, Abi, OL> L11, view<T, Abi, OW> A1, view<T, Abi, OY> L21,
                                  view<const T, Abi, OW> A2, view<T, Abi, OW> A2_out,
                                  view<T, Abi, OU> L31, view<const T, Abi, OW> A3,
                                  view<T, Abi, OW> A3_out, view<const T, Abi> D,
                                  index_t split_A) noexcept;

template <class T, class Abi, StorageOrder OL = StorageOrder::ColMajor,
          StorageOrder OA = StorageOrder::ColMajor, StorageOrder OLu = StorageOrder::ColMajor,
          StorageOrder OAu = StorageOrder::ColMajor, KernelConfig Conf = {}>
void hyhound_diag_riccati_register(view<T, Abi, OL> L11, view<T, Abi, OA> A1, view<T, Abi, OL> L21,
                                   view<const T, Abi, OA> A2, view<T, Abi, OA> A2_out,
                                   view<T, Abi, OLu> Lu1, view<T, Abi, OAu> Au_out,
                                   view<const T, Abi> D, bool shift_A_out) noexcept;

} // namespace batmat::linalg::micro_kernels::hyhound
