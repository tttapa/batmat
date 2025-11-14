#pragma once

#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/micro-kernels/gemv.hpp>
#include <batmat/linalg/shift.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail {
template <class T, class Abi, micro_kernels::gemv::KernelConfig Conf = {}, StorageOrder OA>
void gemv(view<const T, Abi, OA> A, view<const T, Abi> B, std::optional<view<const T, Abi>> C,
          view<T, Abi> D) {
    GUANAQO_TRACE("gemv", 0, A.rows() * A.cols() * B.cols() * A.depth());
    // Check dimensions
    BATMAT_ASSERT(!C || C->rows() == D.rows());
    BATMAT_ASSERT(!C || C->cols() == D.cols());
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == B.rows());
    BATMAT_ASSERT(B.cols() == D.cols());
    BATMAT_ASSERT(B.cols() == 1);
    const index_t M = D.rows(), K = A.cols();

    // Degenerate case
    if (M == 0) [[unlikely]]
        return;
    if (K == 0) [[unlikely]] {
        // https://github.com/llvm/llvm-project/issues/146272
        constexpr detail::copy::CopyConfig rot{.rotate = Conf.rotate_C - Conf.rotate_D,
                                               .mask   = Conf.mask_D};
        constexpr detail::copy::FillConfig msk{.mask = Conf.mask_D};
        if (C)
            detail::copy::copy<T, Abi, rot>(*C, D);
        else
            detail::copy::fill<T, Abi, msk>(T{}, D);
        return;
    }
    micro_kernels::gemv::gemv_copy_register<T, Abi, Conf, OA>(A, B, C, D);
}

template <shift_opt... Opts>
constexpr micro_kernels::gemv::KernelConfig
apply_gemv_options(micro_kernels::gemv::KernelConfig conf, Opts...) {
    if (auto s = shift_A<Opts...>)
        conf.shift_A = *s;
    if (auto s = shift_B<Opts...>)
        conf.shift_B = *s;
    if (auto s = rotate_C<Opts...>)
        conf.rotate_C = *s;
    if (auto s = rotate_D<Opts...>)
        conf.rotate_D = *s;
    if (auto s = mask_D<Opts...>)
        conf.mask_D = *s;
    return conf;
}

} // namespace detail

/// d = A b
template <simdifiable VA, simdifiable VB, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VD>
void gemv(VA &&A, VB &&B, VD &&D, Opts... opts) {
    constexpr auto conf = detail::apply_gemv_options({.negate = false}, opts...);
    std::optional<decltype(simdify(D).as_const())> null;
    detail::gemv<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(B).as_const(), null, simdify(D));
}

/// d = -A b
template <simdifiable VA, simdifiable VB, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VD>
void gemv_neg(VA &&A, VB &&B, VD &&D, Opts... opts) {
    constexpr auto conf = detail::apply_gemv_options({.negate = true}, opts...);
    std::optional<decltype(simdify(D).as_const())> null;
    detail::gemv<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(B).as_const(), null, simdify(D));
}

/// d = c + A b
template <simdifiable VA, simdifiable VB, simdifiable VC, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VC, VD>
void gemv_add(VA &&A, VB &&B, VC &&C, VD &&D, Opts... opts) {
    constexpr auto conf = detail::apply_gemv_options({.negate = false}, opts...);
    detail::gemv<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(B).as_const(), simdify(C).as_const(), simdify(D));
}
/// d = d + A b
template <simdifiable VA, simdifiable VB, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VD>
void gemv_add(VA &&A, VB &&B, VD &&D, Opts... opts) {
    gemv_add(A, B, D, D, opts...);
}

/// d = c - A b
template <simdifiable VA, simdifiable VB, simdifiable VC, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VC, VD>
void gemv_sub(VA &&A, VB &&B, VC &&C, VD &&D, Opts... opts) {
    constexpr auto conf = detail::apply_gemv_options({.negate = true}, opts...);
    detail::gemv<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(B).as_const(), simdify(C).as_const(), simdify(D));
}
/// d = d - A b
template <simdifiable VA, simdifiable VB, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VD>
void gemv_sub(VA &&A, VB &&B, VD &&D, Opts... opts) {
    gemv_sub(A, B, D, D, opts...);
}

} // namespace batmat::linalg
