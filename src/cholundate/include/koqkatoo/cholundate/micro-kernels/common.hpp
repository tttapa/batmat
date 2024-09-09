#pragma once

#include <koqkatoo/config.hpp>
#include <koqkatoo/matrix-view.hpp>
#include <koqkatoo/unroll.h>
#include <experimental/simd>
#include <cstddef>
#include <type_traits>

namespace koqkatoo::cholundate::micro_kernels {

namespace stdx                     = std::experimental;
using native_abi                   = stdx::simd_abi::native<real_t>;
constexpr index_t native_simd_size = stdx::simd_size_v<real_t, native_abi>;

template <index_t Bs, index_t MaxVecLen = 0>
struct optimal_simd_type {
    static constexpr index_t max_vec_len =
        MaxVecLen > 0 ? MaxVecLen : native_simd_size;
    static constexpr index_t vec_len =
        ((Bs > max_vec_len) && (Bs % max_vec_len == 0)) ? max_vec_len : Bs;
    using simd_abi = stdx::simd_abi::deduce_t<real_t, vec_len>;
    using type     = stdx::simd<real_t, simd_abi>;
};
template <index_t Bs, index_t MaxVecLen = 0>
using optimal_simd_type_t = typename optimal_simd_type<Bs, MaxVecLen>::type;

template <class T, class OuterStrideT = index_t>
struct mat_access_impl {
    using value_type = T;
    value_type *data;
    [[no_unique_address]] OuterStrideT outer_stride;
    static constexpr ptrdiff_t inner_stride = 1;
    static constexpr bool transpose         = false;

    [[gnu::always_inline]] value_type &operator()(index_t r,
                                                  index_t c) const noexcept {
        ptrdiff_t i0 = transpose ? c : r;
        ptrdiff_t i1 = transpose ? r : c;
        return data[i0 * inner_stride +
                    i1 * static_cast<ptrdiff_t>(outer_stride)];
    }
    template <class Simd>
    [[gnu::always_inline]] Simd load(index_t r, index_t c) const noexcept {
        return Simd{&operator()(r, c), stdx::element_aligned};
    }
    template <class Simd>
    [[gnu::always_inline]] void store(Simd x, index_t r,
                                      index_t c) const noexcept
        requires(!std::is_const_v<T>)
    {
        x.copy_to(&operator()(r, c), stdx::element_aligned);
    }
    template <class Simd, class Align>
    [[gnu::always_inline]] Simd load(index_t r, index_t c,
                                     Align align) const noexcept {
        return Simd{&operator()(r, c), align};
    }
    template <class Simd, class Align>
    [[gnu::always_inline]] void store(Simd x, index_t r, index_t c,
                                      Align align) const noexcept
        requires(!std::is_const_v<T>)
    {
        x.copy_to(&operator()(r, c), align);
    }
    template <class Self>
    [[gnu::always_inline]] constexpr Self block(this const Self &self,
                                                index_t r, index_t c) noexcept {
        return {&self(r, c), self.outer_stride};
    }
    template <class Self>
    [[gnu::always_inline]] constexpr Self middle_rows(this const Self &self,
                                                      index_t r) noexcept {
        return {&self(r, 0), self.outer_stride};
    }
    template <class Self>
    [[gnu::always_inline]] constexpr Self middle_cols(this const Self &self,
                                                      index_t c) noexcept {
        return {&self(0, c), self.outer_stride};
    }

    [[gnu::always_inline]] constexpr mat_access_impl(
        value_type *data, OuterStrideT outer_stride = {}) noexcept
        : data{data}, outer_stride{outer_stride} {}
    [[gnu::always_inline]] constexpr mat_access_impl(
        const guanaqo::MatrixView<T, index_t> &o) noexcept
        : data{o.data},
          outer_stride{o.outer_stride * static_cast<index_t>(inner_stride)} {}
    [[gnu::always_inline]] constexpr mat_access_impl(
        const mat_access_impl<std::remove_const_t<T>> &o) noexcept
        requires(std::is_const_v<T>)
        : data{o.data}, outer_stride{o.outer_stride} {}
    [[gnu::always_inline]] constexpr mat_access_impl(const mat_access_impl &o) =
        default;
};

struct matrix_accessor : mat_access_impl<const real_t> {
    using mat_access_impl<const real_t>::mat_access_impl;
};

struct mut_matrix_accessor : mat_access_impl<real_t> {
    using mat_access_impl<real_t>::mat_access_impl;
};

} // namespace koqkatoo::cholundate::micro_kernels
