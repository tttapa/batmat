#pragma once

#include <batmat/kib.hpp>
#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/micro-kernels/gemm-diag.hpp>
#include <batmat/linalg/shift.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>
#include <optional>

namespace batmat::linalg {

namespace detail {
template <class T, class Abi, micro_kernels::gemm_diag::KernelConfig Conf = {}, StorageOrder OA,
          StorageOrder OB, StorageOrder OC, StorageOrder OD>
void gemm_diag(view<const T, Abi, OA> A, view<const T, Abi, OB> B,
               std::optional<view<const T, Abi, OC>> C, view<T, Abi, OD> D, view<const T, Abi> d) {
    GUANAQO_TRACE("gemm_diag", 0, A.rows() * A.cols() * B.cols() * A.depth());
    // Check dimensions
    BATMAT_ASSERT(!C || C->rows() == D.rows());
    BATMAT_ASSERT(!C || C->cols() == D.cols());
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == B.rows());
    BATMAT_ASSERT(A.cols() == d.rows());
    BATMAT_ASSERT(d.cols() == 1);
    BATMAT_ASSERT(B.cols() == D.cols());
    const index_t M = D.rows(), N = D.cols(), K = A.cols();

    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;
    if (K == 0) [[unlikely]] {
        constexpr detail::copy::CopyConfig rot{.struc = Conf.struc_C};
        constexpr detail::copy::FillConfig msk{.struc = Conf.struc_C};
        if (C)
            detail::copy::copy<T, Abi, rot>(*C, D);
        else
            detail::copy::copy<T, Abi, msk>(T{}, D);
        return;
    }
    // TODO: cache blocking
    return micro_kernels::gemm_diag::gemm_diag_copy_register<T, Abi, Conf>(A, B, C, D, d);
}
} // namespace detail

template <bool Z>
struct track_zeros_t : std::bool_constant<Z> {};

template <bool Z = true>
inline constexpr track_zeros_t<Z> track_zeros;

namespace detail {
template <class...>
inline constexpr std::optional<bool> get_track_zeros = std::nullopt;
template <class T, class... Ts>
inline constexpr std::optional<bool> get_track_zeros<T, Ts...> = get_track_zeros<Ts...>;
template <bool Z, class... Ts>
inline constexpr std::optional<bool> get_track_zeros<track_zeros_t<Z>, Ts...> = Z;

template <class>
inline constexpr bool is_track_zeros_opt = false;
template <bool Z>
inline constexpr bool is_track_zeros_opt<track_zeros_t<Z>> = true;

template <class Opt>
concept track_zeros_opt = is_track_zeros_opt<Opt>;

template <class... Opts>
constexpr micro_kernels::gemm_diag::KernelConfig
apply_options(micro_kernels::gemm_diag::KernelConfig conf, Opts...) {
    if (auto z = get_track_zeros<Opts...>)
        conf.track_zeros = *z;
    return conf;
}
} // namespace detail

/// D = A diag(d) B
template <simdifiable VA, simdifiable VB, simdifiable VD, simdifiable Vd,
          detail::track_zeros_opt... Opts>
    requires simdify_compatible<VA, VB, VD, Vd>
void gemm_diag(VA &&A, VB &&B, VD &&D, Vd &&d, Opts... opts) {
    std::optional<decltype(simdify(D).as_const())> null;
    constexpr auto conf = detail::apply_options({.negate = false}, opts...);
    detail::gemm_diag<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(B).as_const(), null, simdify(D), simdify(d).as_const());
}

/// D = C + A diag(d) Aᵀ with C, D symmetric
template <MatrixStructure SC, simdifiable VA, simdifiable VC, simdifiable VD, simdifiable Vd,
          detail::track_zeros_opt... Opts>
    requires simdify_compatible<VA, VC, VD, Vd>
void syrk_diag_add(VA &&A, Structured<VC, SC> C, Structured<VD, SC> D, Vd &&d, Opts... opts) {
    static_assert(SC != MatrixStructure::General);
    constexpr auto conf = detail::apply_options({.negate = false, .struc_C = SC}, opts...);
    detail::gemm_diag<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(A).as_const().transposed(),
        std::make_optional(simdify(C.value).as_const()), simdify(D.value), simdify(d).as_const());
}
/// D += A diag(d) Aᵀ with D symmetric
template <MatrixStructure SC, simdifiable VA, simdifiable VD, simdifiable Vd,
          detail::track_zeros_opt... Opts>
    requires simdify_compatible<VA, VD, Vd>
void syrk_diag_add(VA &&A, Structured<VD, SC> D, Vd &&d, Opts... opts) {
    syrk_diag_add(A, D, D, d, opts...);
}

} // namespace batmat::linalg
