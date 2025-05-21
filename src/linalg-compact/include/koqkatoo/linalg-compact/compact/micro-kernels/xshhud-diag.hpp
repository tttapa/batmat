#pragma once

#include <koqkatoo/linalg-compact/compact/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>
#include <type_traits>

namespace koqkatoo::linalg::compact::micro_kernels::shhud_diag {

template <class Abi, class T, index_t R>
struct triangular_accessor {
    using value_type = T;
    value_type *data;

    using simd = stdx::simd<std::remove_const_t<T>, Abi>;
    static constexpr ptrdiff_t inner_stride = simd::size();

    static constexpr size_t size() {
        auto r = static_cast<size_t>(R);
        return simd::size() * (r * (r + 1) / 2);
    }
    static constexpr size_t alignment() {
        return stdx::memory_alignment_v<simd>;
    }

    [[gnu::always_inline]] value_type &operator()(index_t r,
                                                  index_t c) const noexcept {
        assert(r <= c);
        return data[(r + c * (c + 1) / 2) * inner_stride];
    }
    [[gnu::always_inline]] simd load(index_t r, index_t c) const noexcept {
        return simd{&operator()(r, c), stdx::vector_aligned};
    }
    [[gnu::always_inline]] void store(simd x, index_t r,
                                      index_t c) const noexcept
        requires(!std::is_const_v<T>)
    {
        x.copy_to(&operator()(r, c), stdx::vector_aligned);
    }

    [[gnu::always_inline]] triangular_accessor(value_type *data) noexcept
        : data{data} {}
};

#ifdef __AVX512F__
// AVX512 has 32 vector registers, TODO:
static constexpr index_t SizeR = 5;
static constexpr index_t SizeS = 5;
#elif defined(__ARM_NEON)
// NEON has 32 vector registers, TODO:
static constexpr index_t SizeR = 5;
static constexpr index_t SizeS = 5;
#else
// AVX2 has 16 vector registers, TODO:
static constexpr index_t SizeR = 4;
static constexpr index_t SizeS = 3;
#endif

template <class Abi, index_t R>
void xshhud_diag_diag_microkernel(
    index_t colsA, triangular_accessor<Abi, real_t, SizeR> W,
    mut_single_batch_matrix_accessor<Abi> L,
    mut_single_batch_matrix_accessor<Abi> A,
    single_batch_vector_accessor<Abi> diag) noexcept;

template <class Abi, index_t R>
void xshhud_diag_full_microkernel(
    index_t colsA, mut_single_batch_matrix_accessor<Abi> L,
    mut_single_batch_matrix_accessor<Abi> A,
    single_batch_vector_accessor<Abi> diag) noexcept;

template <class Abi, index_t R, index_t S>
void xshhud_diag_tail_microkernel(
    index_t colsA, triangular_accessor<Abi, const real_t, SizeR> W,
    mut_single_batch_matrix_accessor<Abi> L,
    mut_single_batch_matrix_accessor<Abi> A,
    single_batch_matrix_accessor<Abi> B, single_batch_vector_accessor<Abi> diag,
    bool trans_L) noexcept;

template <class Abi>
inline const constinit auto microkernel_diag_lut =
    make_1d_lut<SizeR>([]<index_t Row>(index_constant<Row>) {
        return xshhud_diag_diag_microkernel<Abi, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_full_lut =
    make_1d_lut<SizeR>([]<index_t Row>(index_constant<Row>) {
        return xshhud_diag_full_microkernel<Abi, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_tail_lut =
    make_1d_lut<SizeS>([]<index_t Row>(index_constant<Row>) {
        return xshhud_diag_tail_microkernel<Abi, SizeR, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_tail_lut_2 = make_2d_lut<SizeR, SizeS>(
    []<index_t NR, index_t NS>(index_constant<NR>, index_constant<NS>) {
        return xshhud_diag_tail_microkernel<Abi, NR + 1, NS + 1>;
    });

} // namespace koqkatoo::linalg::compact::micro_kernels::shhud_diag
