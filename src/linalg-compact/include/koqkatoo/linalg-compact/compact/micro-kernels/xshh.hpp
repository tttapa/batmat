#pragma once

#include <koqkatoo/linalg-compact/compact/micro-kernels/common.hpp>
#include <koqkatoo/lut.hpp>
#include <type_traits>

namespace koqkatoo::linalg::compact::micro_kernels::shh {

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

template <class Abi, index_t R>
void xshh_diag_microkernel(index_t colsA, triangular_accessor<Abi, real_t, R> W,
                           mut_single_batch_matrix_accessor<Abi> L,
                           mut_single_batch_matrix_accessor<Abi> A) noexcept;

template <class Abi, index_t R>
void xshh_full_microkernel(index_t colsA,
                           mut_single_batch_matrix_accessor<Abi> L,
                           mut_single_batch_matrix_accessor<Abi> A) noexcept;

template <class Abi, index_t R, index_t S>
void xshh_tail_microkernel(index_t colsA,
                           triangular_accessor<Abi, const real_t, R> W,
                           mut_single_batch_matrix_accessor<Abi> L,
                           mut_single_batch_matrix_accessor<Abi> A,
                           single_batch_matrix_accessor<Abi> B) noexcept;

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

template <class Abi>
inline const constinit auto microkernel_full_lut =
    make_1d_lut<SizeR>([]<index_t Row>(index_constant<Row>) {
        return xshh_full_microkernel<Abi, Row + 1>;
    });

template <class Abi>
inline const constinit auto microkernel_tail_lut =
    make_1d_lut<SizeS>([]<index_t Row>(index_constant<Row>) {
        return xshh_tail_microkernel<Abi, SizeR, Row + 1>;
    });

} // namespace koqkatoo::linalg::compact::micro_kernels::shh
