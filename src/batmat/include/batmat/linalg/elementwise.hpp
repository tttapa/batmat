#pragma once

#include <batmat/assume.hpp>
#include <batmat/config.hpp>
#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/shift.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/ops/rotate.hpp>
#include <batmat/simd.hpp>
#include <guanaqo/trace.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <tuple>
#include <utility>

namespace batmat::linalg {

/// @cond DETAIL

namespace detail {

constexpr index_t num_elem(const auto &A) { return A.rows() * A.cols() * A.depth(); }

template <class T, class Abi, StorageOrder O, class F, class X, class... Xs>
[[gnu::always_inline]] inline void iter_elems(F &&fun, X &&x, Xs &&...xs) {
    using types = simd_view_types<T, Abi>;
    if constexpr (O == StorageOrder::ColMajor) {
        for (index_t c = 0; c < x.cols(); ++c)
            for (index_t r = 0; r < x.rows(); ++r)
                fun(types::aligned_load(&x(0, r, c)), types::aligned_load(&xs(0, r, c))...);
    } else {
        for (index_t r = 0; r < x.rows(); ++r)
            for (index_t c = 0; c < x.cols(); ++c)
                fun(types::aligned_load(&x(0, r, c)), types::aligned_load(&xs(0, r, c))...);
    }
}

template <class T, class Abi, StorageOrder O, class F, class X, class... Xs>
[[gnu::always_inline]] inline void iter_elems_store(F &&fun, X &&x, Xs &&...xs) {
    using types = simd_view_types<T, Abi>;
    if constexpr (O == StorageOrder::ColMajor) {
        for (index_t c = 0; c < x.cols(); ++c)
            for (index_t r = 0; r < x.rows(); ++r)
                types::aligned_store(fun(types::aligned_load(&xs(0, r, c))...), &x(0, r, c));
    } else {
        for (index_t r = 0; r < x.rows(); ++r)
            for (index_t c = 0; c < x.cols(); ++c)
                types::aligned_store(fun(types::aligned_load(&xs(0, r, c))...), &x(0, r, c));
    }
}

template <class T, class Abi, StorageOrder O, class F, class X0, class X1, class... Xs>
[[gnu::always_inline]] inline void iter_elems_store2(F &&fun, X0 &&x0, X1 &&x1, Xs &&...xs) {
    using types = simd_view_types<T, Abi>;
    if constexpr (O == StorageOrder::ColMajor) {
        for (index_t c = 0; c < x0.cols(); ++c)
            for (index_t r = 0; r < x0.rows(); ++r) {
                auto [r0, r1] = fun(types::aligned_load(&xs(0, r, c))...);
                types::aligned_store(r0, &x0(0, r, c));
                types::aligned_store(r1, &x1(0, r, c));
            }
    } else {
        for (index_t r = 0; r < x0.rows(); ++r)
            for (index_t c = 0; c < x0.cols(); ++c) {
                auto [r0, r1] = fun(types::aligned_load(&xs(0, r, c))...);
                types::aligned_store(r0, &x0(0, r, c));
                types::aligned_store(r1, &x1(0, r, c));
            }
    }
}

template <class T, class Abi, StorageOrder O, class F, class... Ys, class... Xs>
[[gnu::always_inline]] inline void iter_elems_store_n(F &&fun, std::tuple<Ys...> ys, Xs &&...xs) {
    using std::get;
    using types        = simd_view_types<T, Abi>;
    const index_t rows = std::get<0>(ys).rows(), cols = std::get<0>(ys).cols();
    if constexpr (O == StorageOrder::ColMajor) {
        for (index_t c = 0; c < cols; ++c)
            for (index_t r = 0; r < rows; ++r) {
                auto rs = fun(types::aligned_load(&xs(0, r, c))...);
                static_assert(std::tuple_size_v<decltype(rs)> == sizeof...(Ys));
                [&]<size_t... Is>(std::index_sequence<Is...>) {
                    ((types::aligned_store(get<Is>(rs), &get<Is>(ys)(0, r, c))), ...);
                }(std::index_sequence_for<Ys...>());
            }
    } else {
        for (index_t r = 0; r < rows; ++r)
            for (index_t c = 0; c < cols; ++c) {
                auto rs = fun(types::aligned_load(&xs(0, r, c))...);
                static_assert(std::tuple_size_v<decltype(rs)> == sizeof...(Ys));
                [&]<size_t... Is>(std::index_sequence<Is...>) {
                    ((types::aligned_store(get<Is>(rs), &get<Is>(ys)(0, r, c))), ...);
                }(std::index_sequence_for<Ys...>());
            }
    }
}

/// Iterate element-wise over the diagonal elements of matrices and over the elements of vectors.
/// Any argument can either be a square matrix or a vector, but this function has a fast path
/// when the first input and output arguments are square matrices and all others are column vectors.
template <class T, class Abi, class F, class... Ys, class... Xs>
[[gnu::always_inline]] inline void iter_diag_store_n(F &&fun, std::tuple<Ys...> ys, Xs &&...xs) {
    using std::get;
    using types        = simd_view_types<T, Abi>;
    const index_t rows = std::get<0>(ys).rows(), cols = std::get<0>(ys).cols();
    const index_t n = std::get<0>(ys).storage_order == StorageOrder::ColMajor
                          ? (cols > 1 ? std::min(rows, cols) : rows)
                          : (rows > 1 ? std::min(rows, cols) : cols);
    // Optimized implementation for the special case where the first input and the first output are
    // square matrices and all others are column vectors.
    static constexpr auto all_vectors_except_first = [](auto &x0, auto &...x1s) {
        return x0.rows() == x0.cols() &&
               ((x1s.storage_order == StorageOrder::ColMajor && x1s.cols() == 1) && ...);
    };
    const bool all_xs_vectors_except_first = all_vectors_except_first(xs...);
    const bool all_ys_vector_except_first  = std::apply(all_vectors_except_first, ys);
    if (all_xs_vectors_except_first && all_ys_vector_except_first) {
        for (index_t r = 0; r < n; ++r) {
            auto rs = [&](auto &x0, auto &...x1s) {
                return fun(types::aligned_load(&x0(0, r, r)),
                           types::aligned_load(&x1s(0, r, 0))...);
            }(xs...);
            static_assert(std::tuple_size_v<decltype(rs)> == sizeof...(Ys));
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ((types::aligned_store(get<Is>(rs), &get<Is>(ys)(0, r, Is == 0 ? r : 0))), ...);
            }(std::index_sequence_for<Ys...>());
        }
    }
    // Fully generic implementation
    // (GCC does not seem to hoist the conditionals in the access function out of the loop, so we
    //  pay for some cmovs here, even at -O3. Should be fine though, since we're most likely memory-
    //  bound anyway.)
    else {
        static constexpr auto access = [](auto &x, index_t r) -> auto & {
            return x.storage_order == StorageOrder::ColMajor
                       ? (x.cols() > 1 ? x(0, r, r) : x(0, r, 0))
                       : (x.rows() > 1 ? x(0, r, r) : x(0, 0, r));
        };
        for (index_t r = 0; r < n; ++r) {
            auto rs = fun(types::aligned_load(&access(xs, r))...);
            static_assert(std::tuple_size_v<decltype(rs)> == sizeof...(Ys));
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ((types::aligned_store(get<Is>(rs), &access(get<Is>(ys), r))), ...);
            }(std::index_sequence_for<Ys...>());
        }
    }
}

/// Scalar product.
template <class T, class Abi, StorageOrder OB, StorageOrder OC>
[[gnu::flatten]] void scale(datapar::simd<T, Abi> a, view<const T, Abi, OB> B, view<T, Abi, OC> C) {
    BATMAT_ASSERT(B.rows() == C.rows());
    BATMAT_ASSERT(B.cols() == C.cols());
    iter_elems_store<T, Abi, OC>([&](auto Bi) { return a * Bi; }, C, B);
}

/// Hadamard (elementwise) product.
template <class T, class Abi, StorageOrder OA, StorageOrder OB, StorageOrder OC>
[[gnu::flatten]] void hadamard(view<const T, Abi, OA> A, view<const T, Abi, OB> B,
                               view<T, Abi, OC> C) {
    BATMAT_ASSERT(A.rows() == B.rows());
    BATMAT_ASSERT(A.cols() == B.cols());
    BATMAT_ASSERT(A.rows() == C.rows());
    BATMAT_ASSERT(A.cols() == C.cols());
    iter_elems_store<T, Abi, OC>([&](auto Ai, auto Bi) { return Ai * Bi; }, C, A, B);
}

/// Elementwise clamping z = max(lo, min(x, hi)).
template <class T, class Abi, StorageOrder O>
[[gnu::flatten]] void clamp(view<const T, Abi, O> x, view<const T, Abi, O> lo,
                            view<const T, Abi, O> hi, view<T, Abi, O> z) {
    BATMAT_ASSERT(x.rows() == lo.rows());
    BATMAT_ASSERT(x.cols() == lo.cols());
    BATMAT_ASSERT(x.rows() == hi.rows());
    BATMAT_ASSERT(x.cols() == hi.cols());
    BATMAT_ASSERT(x.rows() == z.rows());
    BATMAT_ASSERT(x.cols() == z.cols());
    const auto clamp = [&](auto xi, auto loi, auto hii) { return fmax(loi, fmin(xi, hii)); };
    iter_elems_store<T, Abi, O>(clamp, z, x, lo, hi);
}

/// Elementwise clamping z = max(lo, min(x, hi)), with scalar lo and hi.
template <class T, class Abi, StorageOrder O>
[[gnu::flatten]] void clamp(view<const T, Abi, O> x, datapar::simd<T, Abi> lo,
                            datapar::simd<T, Abi> hi, view<T, Abi, O> z) {
    BATMAT_ASSERT(x.rows() == z.rows());
    BATMAT_ASSERT(x.cols() == z.cols());
    const auto clamp = [&](auto xi) { return fmax(lo, fmin(xi, hi)); };
    iter_elems_store<T, Abi, O>(clamp, z, x);
}

/// Elementwise clamping residual z = x - max(lo, min(x, hi)).
template <class T, class Abi, StorageOrder O>
[[gnu::flatten]] void clamp_resid(view<const T, Abi, O> x, view<const T, Abi, O> lo,
                                  view<const T, Abi, O> hi, view<T, Abi, O> z) {
    BATMAT_ASSERT(x.rows() == lo.rows());
    BATMAT_ASSERT(x.cols() == lo.cols());
    BATMAT_ASSERT(x.rows() == hi.rows());
    BATMAT_ASSERT(x.cols() == hi.cols());
    BATMAT_ASSERT(x.rows() == z.rows());
    BATMAT_ASSERT(x.cols() == z.cols());
    using simd             = datapar::simd<T, Abi>;
    const auto clamp_resid = [&](auto xi, auto loi, auto hii) {
        return fmax(xi - hii, fmin(simd{0}, xi - loi));
    };
    iter_elems_store<T, Abi, O>(clamp_resid, z, x, lo, hi);
}

/// Linear combination of vectors z = beta * z + sum_i alpha_i * x_i.
template <class T, class Abi, T Beta, StorageOrder O, class... Xs>
[[gnu::flatten]] void gaxpby(view<T, Abi, O> z,
                             const std::array<datapar::simd<T, Abi>, sizeof...(Xs)> &alphas,
                             const Xs &...xs) {
    BATMAT_ASSERT(((z.rows() == xs.rows()) && ...));
    BATMAT_ASSERT(((z.cols() == xs.cols()) && ...));
    if constexpr (Beta == 0)
        iter_elems_store<T, Abi, O>(
            [&](auto... xis) {
                return [&]<std::size_t... Is>(std::index_sequence<Is...>, auto... xis) {
                    return ((xis * alphas[Is]) + ...);
                }(std::make_index_sequence<sizeof...(Xs)>(), xis...);
            },
            z, xs...);
    else
        iter_elems_store<T, Abi, O>(
            [&](auto zi, auto... xis) {
                return [&]<std::size_t... Is>(std::index_sequence<Is...>, auto... xis) {
                    return zi * Beta + ((xis * alphas[Is]) + ...);
                }(std::make_index_sequence<sizeof...(Xs)>(), xis...);
            },
            z, z, xs...);
}

/// Negate a matrix or vector.
/// @todo: add Negate option to batmat::linalg::copy and remove this function, then this also
///        supports transposition.
template <class T, class Abi, int Rotate, StorageOrder OA, StorageOrder OB>
[[gnu::flatten]] void negate(view<const T, Abi, OA> A, view<T, Abi, OB> B) {
    BATMAT_ASSERT(A.rows() == B.rows());
    BATMAT_ASSERT(A.cols() == B.cols());
    using ops::rotl;
    iter_elems_store<T, Abi, OB>([&](auto Ai) { return -rotl<Rotate>(Ai); }, B, A);
}

/// Subtract two matrices or vectors C = A - B.
template <class T, class Abi, int Rotate, StorageOrder OA, StorageOrder OB, StorageOrder OC>
[[gnu::flatten]] void sub(view<const T, Abi, OA> A, view<const T, Abi, OB> B, view<T, Abi, OC> C) {
    BATMAT_ASSERT(A.rows() == B.rows());
    BATMAT_ASSERT(A.cols() == B.cols());
    BATMAT_ASSERT(A.rows() == C.rows());
    BATMAT_ASSERT(A.cols() == C.cols());
    using ops::rotl;
    iter_elems_store<T, Abi, OC>([&](auto Ai, auto Bi) { return Ai - rotl<Rotate>(Bi); }, C, A, B);
}

/// Add two matrices or vectors C = A + B.
template <class T, class Abi, int Rotate, StorageOrder OA, StorageOrder OB, StorageOrder OC>
[[gnu::flatten]] void add(view<const T, Abi, OA> A, view<const T, Abi, OB> B, view<T, Abi, OC> C) {
    BATMAT_ASSERT(A.rows() == B.rows());
    BATMAT_ASSERT(A.cols() == B.cols());
    BATMAT_ASSERT(A.rows() == C.rows());
    BATMAT_ASSERT(A.cols() == C.cols());
    using ops::rotl;
    iter_elems_store<T, Abi, OC>([&](auto Ai, auto Bi) { return Ai + rotl<Rotate>(Bi); }, C, A, B);
}

} // namespace detail

/// @endcond

/// @addtogroup topic-linalg
/// @{

/// @name Single-batch elementwise operations
/// @{

/// Multiply a vector by a scalar z = αx.
template <simdifiable Vx, simdifiable Vz, std::convertible_to<simdified_simd_t<Vx>> T>
    requires simdify_compatible<Vx, Vz>
void scale(T alpha, Vx &&x, Vz &&z) {
    GUANAQO_TRACE_LINALG("scale", detail::num_elem(simdify(x)));
    detail::scale<simdified_value_t<Vx>, simdified_abi_t<Vx>>(alpha, simdify(x).as_const(),
                                                              simdify(z));
}

/// Multiply a vector by a scalar x = αx.
template <simdifiable Vx, std::convertible_to<simdified_simd_t<Vx>> T>
void scale(T alpha, Vx &&x) {
    GUANAQO_TRACE_LINALG("scale", detail::num_elem(simdify(x)));
    detail::scale<simdified_value_t<Vx>, simdified_abi_t<Vx>>(alpha, simdify(x).as_const(),
                                                              simdify(x));
}

/// Compute the Hadamard (elementwise) product of two vectors z = x ⊙ y.
template <simdifiable Vx, simdifiable Vy, simdifiable Vz>
    requires simdify_compatible<Vx, Vy, Vz>
void hadamard(Vx &&x, Vy &&y, Vz &&z) {
    GUANAQO_TRACE_LINALG("hadamard", detail::num_elem(simdify(x)));
    detail::hadamard<simdified_value_t<Vx>, simdified_abi_t<Vx>>(simdify(x).as_const(),
                                                                 simdify(y).as_const(), simdify(z));
}

/// Compute the Hadamard (elementwise) product of two vectors x = x ⊙ y.
template <simdifiable Vx, simdifiable Vy>
    requires simdify_compatible<Vx, Vy>
void hadamard(Vx &&x, Vy &&y) {
    GUANAQO_TRACE_LINALG("hadamard", detail::num_elem(simdify(x)));
    detail::hadamard<simdified_value_t<Vx>, simdified_abi_t<Vx>>(simdify(x).as_const(),
                                                                 simdify(y).as_const(), simdify(x));
}

/// Elementwise clamping z = max(lo, min(x, hi)).
template <simdifiable Vx, simdifiable Vlo, simdifiable Vhi, simdifiable Vz>
    requires simdify_compatible<Vx, Vlo, Vhi, Vz>
void clamp(Vx &&x, Vlo &&lo, Vhi &&hi, Vz &&z) {
    GUANAQO_TRACE_LINALG("clamp", 2 * detail::num_elem(simdify(x))); // max, min
    detail::clamp<simdified_value_t<Vx>, simdified_abi_t<Vx>>(
        simdify(x).as_const(), simdify(lo).as_const(), simdify(hi).as_const(), simdify(z));
}

/// Elementwise clamping residual z = x - max(lo, min(x, hi)).
template <simdifiable Vx, simdifiable Vlo, simdifiable Vhi, simdifiable Vz>
    requires simdify_compatible<Vx, Vlo, Vhi, Vz>
void clamp_resid(Vx &&x, Vlo &&lo, Vhi &&hi, Vz &&z) {
    GUANAQO_TRACE_LINALG("clamp_resid", 3 * detail::num_elem(simdify(x))); // sub, max, min
    detail::clamp_resid<simdified_value_t<Vx>, simdified_abi_t<Vx>>(
        simdify(x).as_const(), simdify(lo).as_const(), simdify(hi).as_const(), simdify(z));
}

/// Elementwise clamping z = max(lo, min(x, hi)), with scalar lo and hi.
template <simdifiable Vx, simdifiable Vz>
    requires simdify_compatible<Vx, Vz>
void clamp(Vx &&x, simdified_simd_t<Vx> lo, simdified_simd_t<Vx> hi, Vz &&z) {
    GUANAQO_TRACE_LINALG("clamp", 2 * detail::num_elem(simdify(x))); // max, min
    detail::clamp<simdified_value_t<Vx>, simdified_abi_t<Vx>>(simdify(x).as_const(), lo, hi,
                                                              simdify(z));
}

/// Add scaled vector z = αx + βy.
template <simdifiable Vx, simdifiable Vy, simdifiable Vz, //
          std::convertible_to<simdified_simd_t<Vx>> Ta,
          std::convertible_to<simdified_simd_t<Vx>> Tb>
    requires simdify_compatible<Vx, Vy, Vz>
void axpby(Ta alpha, Vx &&x, Tb beta, Vy &&y, Vz &&z) {
    GUANAQO_TRACE_LINALG("axpby", 2 * detail::num_elem(simdify(x))); // mul, fma
    detail::gaxpby<simdified_value_t<Vx>, simdified_abi_t<Vx>, simdified_value_t<Vx>{0}>(
        simdify(z), {{alpha, beta}}, simdify(x).as_const(), simdify(y).as_const());
}

/// Add scaled vector y = αx + βy.
template <simdifiable Vx, simdifiable Vy, //
          std::convertible_to<simdified_simd_t<Vx>> Ta,
          std::convertible_to<simdified_simd_t<Vx>> Tb>
    requires simdify_compatible<Vx, Vy>
void axpby(Ta alpha, Vx &&x, Tb beta, Vy &&y) {
    GUANAQO_TRACE_LINALG("axpby", 2 * detail::num_elem(simdify(x))); // mul, fma
    detail::gaxpby<simdified_value_t<Vx>, simdified_abi_t<Vx>, simdified_value_t<Vx>{0}>(
        simdify(y), {{alpha, beta}}, simdify(x).as_const(), simdify(y).as_const());
}

/// Add scaled vector y = ∑ᵢ αᵢxᵢ + βy.
template <auto Beta = 1, simdifiable Vy, simdifiable... Vx>
    requires simdify_compatible<Vy, Vx...>
void axpy(Vy &&y, const std::array<simdified_simd_t<Vy>, sizeof...(Vx)> &alphas, Vx &&...x) {
    [[maybe_unused]] static constexpr index_t num_mul = Beta != 1 && Beta != 0 ? 1 : 0;
    [[maybe_unused]] static constexpr index_t num_fma = sizeof...(Vx);
    GUANAQO_TRACE_LINALG("axpy", (num_mul + num_fma) * detail::num_elem(simdify(y))); // mul, fma
    detail::gaxpby<simdified_value_t<Vy>, simdified_abi_t<Vy>, simdified_value_t<Vy>{Beta}>(
        simdify(y), alphas, simdify(x).as_const()...);
}

/// Add scaled vector z = αx + y.
template <simdifiable Vx, simdifiable Vy, simdifiable Vz,
          std::convertible_to<simdified_simd_t<Vx>> Ta>
    requires simdify_compatible<Vx, Vy, Vz>
void axpy(Ta alpha, Vx &&x, Vy &&y, Vz &&z) {
    axpby(alpha, x, Ta{1}, y, z);
}

/// Add scaled vector y = αx + βy (where β is a compile-time constant).
template <auto Beta = 1, simdifiable Vx, simdifiable Vy,
          std::convertible_to<simdified_simd_t<Vx>> Ta>
    requires simdify_compatible<Vx, Vy>
void axpy(Ta alpha, Vx &&x, Vy &&y) {
    [[maybe_unused]] static constexpr index_t num_mul = Beta != 1 && Beta != 0 ? 1 : 0;
    [[maybe_unused]] static constexpr index_t num_fma = 1;
    GUANAQO_TRACE_LINALG("axpy", (num_mul + num_fma) * detail::num_elem(simdify(y))); // mul, fma
    detail::gaxpby<simdified_value_t<Vx>, simdified_abi_t<Vx>, simdified_value_t<Vx>{Beta}>(
        simdify(y), {{alpha}}, simdify(x).as_const());
}

/// Negate a matrix or vector B = -A.
template <simdifiable VA, simdifiable VB, int Rotate = 0>
    requires simdify_compatible<VA, VB>
void negate(VA &&A, VB &&B, with_rotate_t<Rotate> = {}) {
    GUANAQO_TRACE_LINALG("negate", detail::num_elem(simdify(A)));
    detail::negate<simdified_value_t<VA>, simdified_abi_t<VA>, Rotate>(simdify(A).as_const(),
                                                                       simdify(B));
}

/// Negate a matrix or vector A = -A.
template <simdifiable VA, int Rotate = 0>
void negate(VA &&A, with_rotate_t<Rotate> = {}) {
    GUANAQO_TRACE_LINALG("negate", detail::num_elem(simdify(A)));
    detail::negate<simdified_value_t<VA>, simdified_abi_t<VA>, Rotate>(simdify(A).as_const(),
                                                                       simdify(A));
}

/// Subtract two matrices or vectors C = A - B. Rotate affects B.
template <simdifiable VA, simdifiable VB, simdifiable VC, int Rotate = 0>
    requires simdify_compatible<VA, VB, VC>
void sub(VA &&A, VB &&B, VC &&C, with_rotate_t<Rotate> = {}) {
    GUANAQO_TRACE_LINALG("sub", detail::num_elem(simdify(A)));
    detail::sub<simdified_value_t<VA>, simdified_abi_t<VA>, Rotate>(
        simdify(A).as_const(), simdify(B).as_const(), simdify(C));
}

/// Subtract two matrices or vectors A = A - B. Rotate affects B.
template <simdifiable VA, simdifiable VB, int Rotate = 0>
    requires simdify_compatible<VA, VB>
void sub(VA &&A, VB &&B, with_rotate_t<Rotate> = {}) {
    GUANAQO_TRACE_LINALG("sub", detail::num_elem(simdify(A)));
    detail::sub<simdified_value_t<VA>, simdified_abi_t<VA>, Rotate>(
        simdify(A).as_const(), simdify(B).as_const(), simdify(A));
}

/// Add two matrices or vectors C = A + B. Rotate affects B.
template <simdifiable VA, simdifiable VB, simdifiable VC, int Rotate = 0>
    requires simdify_compatible<VA, VB, VC>
void add(VA &&A, VB &&B, VC &&C, with_rotate_t<Rotate> = {}) {
    GUANAQO_TRACE_LINALG("add", detail::num_elem(simdify(A)));
    detail::add<simdified_value_t<VA>, simdified_abi_t<VA>, Rotate>(
        simdify(A).as_const(), simdify(B).as_const(), simdify(C));
}

/// Add two matrices or vectors A = A + B. Rotate affects B.
template <simdifiable VA, simdifiable VB, int Rotate = 0>
    requires simdify_compatible<VA, VB>
void add(VA &&A, VB &&B, with_rotate_t<Rotate> = {}) {
    GUANAQO_TRACE_LINALG("add", detail::num_elem(simdify(A)));
    detail::add<simdified_value_t<VA>, simdified_abi_t<VA>, Rotate>(
        simdify(A).as_const(), simdify(B).as_const(), simdify(A));
}

/// Apply a function to all elements of the given matrices or vectors.
template <class F, simdifiable VA, simdifiable... VAs>
    requires simdify_compatible<VA, VAs...>
void for_each_elementwise(F &&fun, VA &&A, VAs &&...As) {
    static constexpr auto storage_order = simdified_view_t<VA>::storage_order;
    detail::iter_elems<simdified_value_t<VA>, simdified_abi_t<VA>, storage_order>(
        std::forward<F>(fun), simdify(A).as_const(), simdify(As).as_const()...);
}

/// Apply a function to all elements of the given matrices or vectors, storing the result in the
/// first argument.
template <class F, simdifiable VA, simdifiable... VAs>
    requires simdify_compatible<VA, VAs...>
void transform_elementwise(F &&fun, VA &&A, VAs &&...As) {
    static constexpr auto storage_order = simdified_view_t<VA>::storage_order;
    detail::iter_elems_store<simdified_value_t<VA>, simdified_abi_t<VA>, storage_order>(
        std::forward<F>(fun), simdify(A), simdify(As).as_const()...);
}

/// Apply a function to all elements of the given matrices or vectors, storing the results in the
/// first two arguments.
template <class F, simdifiable VA, simdifiable VB, simdifiable... VAs>
    requires simdify_compatible<VA, VB, VAs...>
void transform2_elementwise(F &&fun, VA &&A, VB &&B, VAs &&...As) {
    static constexpr auto storage_order = simdified_view_t<VA>::storage_order;
    detail::iter_elems_store2<simdified_value_t<VA>, simdified_abi_t<VA>, storage_order>(
        std::forward<F>(fun), simdify(A), simdify(B), simdify(As).as_const()...);
}

/// Apply a function to all elements of the given matrices or vectors, storing the results in the
/// tuple of matrices given as the first argument.
template <class F, simdifiable... VAs, simdifiable... VBs>
    requires simdify_compatible<VAs..., VBs...>
void transform_n_elementwise(F &&fun, std::tuple<VAs...> As, VBs &&...Bs) {
    using VA0                           = std::tuple_element_t<0, decltype(As)>;
    static constexpr auto storage_order = simdified_view_t<VA0>::storage_order;
    detail::iter_elems_store_n<simdified_value_t<VA0>, simdified_abi_t<VA0>, storage_order>(
        std::forward<F>(fun),
        std::apply([](auto &&...a) { return std::make_tuple(simdify(a)...); }, As),
        simdify(Bs).as_const()...);
}

/// Apply a function to all elements of the given vectors and the diagonal elements of the given
/// square matrices, storing the results in the tuple of vectors or matrices given as the first
/// argument. Most efficient if only the first argument contains matrices, and all other arguments
/// are column vectors.
template <class F, simdifiable... VAs, simdifiable... VBs>
    requires simdify_compatible<VAs..., VBs...>
void transform_n_diag(F &&fun, std::tuple<VAs...> As, VBs &&...Bs) {
    constexpr auto check_size = [](auto &x) {
        if (x.rows() == x.cols())
            return x.rows();
        else if (x.storage_order == StorageOrder::ColMajor) {
            BATMAT_ASSERT(x.cols() == 1);
            return x.rows();
        } else {
            BATMAT_ASSERT(x.rows() == 1);
            return x.cols();
        }
    };
    [[maybe_unused]] const index_t n = check_size(std::get<0>(As));
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        BATMAT_ASSERT(((check_size(get<Is>(As)) == n) && ...));
    }(std::index_sequence_for<VAs...>());
    BATMAT_ASSERT(((check_size(Bs) == n) && ...));
    using VA0 = std::tuple_element_t<0, decltype(As)>;
    detail::iter_diag_store_n<simdified_value_t<VA0>, simdified_abi_t<VA0>>(
        std::forward<F>(fun),
        std::apply([](auto &&...a) { return std::make_tuple(simdify(a)...); }, As),
        simdify(Bs).as_const()...);
}

/// Copy the diagonal elements of a matrix. The arguments @p A and @p B must either be square
/// matrices or vectors. This function supports setting the diagonal of a matrix to the values of
/// a vector, copying the diagonal of one matrix to the diagonal of another, or copying the diagonal
/// elements of a matrix to a vector.
template <class F, simdifiable VA, simdifiable VB>
    requires simdify_compatible<VA, VB>
void copy_diag(VA &&A, VB &&B) {
    [[maybe_unused]] const index_t n =
        A.storage_order == StorageOrder::ColMajor ? A.rows() : A.cols();
    if constexpr (A.storage_order == StorageOrder::ColMajor) {
        BATMAT_ASSERT(A.rows() == n);
        BATMAT_ASSERT(A.cols() == n || A.cols() == 1);
    } else {
        BATMAT_ASSERT(A.rows() == n || A.rows() == 1);
        BATMAT_ASSERT(A.cols() == n);
    }
    if constexpr (B.storage_order == StorageOrder::ColMajor) {
        BATMAT_ASSERT(B.rows() == n);
        BATMAT_ASSERT(B.cols() == n || B.cols() == 1);
    } else {
        BATMAT_ASSERT(B.rows() == n || B.rows() == 1);
        BATMAT_ASSERT(B.cols() == n);
    }
    GUANAQO_TRACE_LINALG("copy_diag", n * A.depth());
    detail::iter_diag_store_n<simdified_value_t<VA>, simdified_abi_t<VA>>(
        [](auto Ai) { return std::make_tuple(Ai); }, std::make_tuple(simdify(B)),
        simdify(A).as_const());
}

/// C = A + diag(b).
template <simdifiable VA, simdifiable VB, simdifiable VC>
    requires simdify_compatible<VA, VB, VC>
void add_diag(VA &&A, VB &&b, VC &&C) {
    BATMAT_ASSERT(A.rows() == A.cols());
    BATMAT_ASSERT(A.rows() == C.rows());
    BATMAT_ASSERT(A.cols() == C.cols());
    if constexpr (b.storage_order == StorageOrder::ColMajor) {
        BATMAT_ASSERT(b.rows() == A.rows());
        BATMAT_ASSERT(b.cols() == 1);
    } else {
        BATMAT_ASSERT(b.rows() == 1);
        BATMAT_ASSERT(b.cols() == A.cols());
    }
    GUANAQO_TRACE_LINALG("add_diag", detail::num_elem(simdify(b)));
    detail::iter_diag_store_n<simdified_value_t<VA>, simdified_abi_t<VA>>(
        [](auto Ai, auto bi) { return std::make_tuple(Ai + bi); }, std::make_tuple(simdify(C)),
        simdify(A).as_const(), simdify(b).as_const());
}

/// A += diag(b).
template <simdifiable VA, simdifiable VB>
    requires simdify_compatible<VA, VB>
void add_diag(VA &&A, VB &&b) {
    add_diag(std::forward<VA>(A), std::forward<VB>(b), std::forward<VA>(A));
}

/// @}

/// @}

// TODO: doxygen gets confused because the template parameters are the same as the single-batch
// versions, so put in a separate namespace
inline namespace multi {

/// @addtogroup topic-linalg
/// @{

/// @name Multi-batch elementwise operations
/// @{

/// Multiply a vector by a scalar z = αx.
template <simdifiable_multi Vx, simdifiable_multi Vz, std::convertible_to<simdified_simd_t<Vx>> T>
    requires simdify_compatible<Vx, Vz>
void scale(T alpha, Vx &&x, Vz &&z) {
    BATMAT_ASSERT(x.num_batches() == z.num_batches());
    for (index_t b = 0; b < x.num_batches(); ++b)
        linalg::scale(alpha, x.batch(b), z.batch(b));
}

/// Multiply a vector by a scalar x = αx.
template <simdifiable_multi Vx, std::convertible_to<simdified_simd_t<Vx>> T>
void scale(T alpha, Vx &&x) {
    for (index_t b = 0; b < x.num_batches(); ++b)
        linalg::scale(alpha, x.batch(b));
}

/// Compute the Hadamard (elementwise) product of two vectors z = x ⊙ y.
template <simdifiable_multi Vx, simdifiable_multi Vy, simdifiable_multi Vz>
    requires simdify_compatible<Vx, Vy, Vz>
void hadamard(Vx &&x, Vy &&y, Vz &&z) {
    BATMAT_ASSERT(x.num_batches() == y.num_batches());
    BATMAT_ASSERT(x.num_batches() == z.num_batches());
    for (index_t b = 0; b < x.num_batches(); ++b)
        linalg::hadamard(x.batch(b), y.batch(b), z.batch(b));
}

/// Compute the Hadamard (elementwise) product of two vectors x = x ⊙ y.
template <simdifiable_multi Vx, simdifiable_multi Vy>
    requires simdify_compatible<Vx, Vy>
void hadamard(Vx &&x, Vy &&y) {
    BATMAT_ASSERT(x.num_batches() == y.num_batches());
    for (index_t b = 0; b < x.num_batches(); ++b)
        linalg::hadamard(x.batch(b), y.batch(b));
}

/// Elementwise clamping z = max(lo, min(x, hi)).
template <simdifiable_multi Vx, simdifiable_multi Vlo, simdifiable_multi Vhi, simdifiable_multi Vz>
    requires simdify_compatible<Vx, Vlo, Vhi, Vz>
void clamp(Vx &&x, Vlo &&lo, Vhi &&hi, Vz &&z) {
    BATMAT_ASSERT(x.num_batches() == lo.num_batches());
    BATMAT_ASSERT(x.num_batches() == hi.num_batches());
    BATMAT_ASSERT(x.num_batches() == z.num_batches());
    for (index_t b = 0; b < x.num_batches(); ++b)
        linalg::clamp(x.batch(b), lo.batch(b), hi.batch(b), z.batch(b));
}

/// Elementwise clamping residual z = x - max(lo, min(x, hi)).
template <simdifiable_multi Vx, simdifiable_multi Vlo, simdifiable_multi Vhi, simdifiable_multi Vz>
    requires simdify_compatible<Vx, Vlo, Vhi, Vz>
void clamp_resid(Vx &&x, Vlo &&lo, Vhi &&hi, Vz &&z) {
    BATMAT_ASSERT(x.num_batches() == lo.num_batches());
    BATMAT_ASSERT(x.num_batches() == hi.num_batches());
    BATMAT_ASSERT(x.num_batches() == z.num_batches());
    for (index_t b = 0; b < x.num_batches(); ++b)
        linalg::clamp_resid(x.batch(b), lo.batch(b), hi.batch(b), z.batch(b));
}

/// Elementwise clamping z = max(lo, min(x, hi)), with scalar lo and hi.
template <simdifiable_multi Vx, simdifiable_multi Vz>
    requires simdify_compatible<Vx, Vz>
void clamp(Vx &&x, simdified_simd_t<Vx> lo, simdified_simd_t<Vx> hi, Vz &&z) {
    BATMAT_ASSERT(x.num_batches() == z.num_batches());
    for (index_t b = 0; b < x.num_batches(); ++b)
        linalg::clamp(x.batch(b), lo, hi, z.batch(b));
}

/// Add scaled vector z = αx + βy.
template <simdifiable_multi Vx, simdifiable_multi Vy, simdifiable_multi Vz, //
          std::convertible_to<simdified_simd_t<Vx>> Ta,
          std::convertible_to<simdified_simd_t<Vx>> Tb>
    requires simdify_compatible<Vx, Vy, Vz>
void axpby(Ta alpha, Vx &&x, Tb beta, Vy &&y, Vz &&z) {
    BATMAT_ASSERT(x.num_batches() == y.num_batches());
    BATMAT_ASSERT(x.num_batches() == z.num_batches());
    for (index_t b = 0; b < x.num_batches(); ++b)
        linalg::axpby(alpha, x.batch(b), beta, y.batch(b), z.batch(b));
}

/// Add scaled vector y = αx + βy.
template <simdifiable_multi Vx, simdifiable_multi Vy, //
          std::convertible_to<simdified_simd_t<Vx>> Ta,
          std::convertible_to<simdified_simd_t<Vx>> Tb>
    requires simdify_compatible<Vx, Vy>
void axpby(Ta alpha, Vx &&x, Tb beta, Vy &&y) {
    BATMAT_ASSERT(x.num_batches() == y.num_batches());
    for (index_t b = 0; b < x.num_batches(); ++b)
        linalg::axpby(alpha, x.batch(b), beta, y.batch(b));
}

/// Add scaled vector y = ∑ᵢ αᵢxᵢ + βy.
template <auto Beta = 1, simdifiable_multi Vy, simdifiable_multi... Vx>
    requires simdify_compatible<Vy, Vx...>
void axpy(Vy &&y, const std::array<simdified_simd_t<Vy>, sizeof...(Vx)> &alphas, Vx &&...x) {
    BATMAT_ASSERT(((y.num_batches() == x.num_batches()) && ...));
    for (index_t b = 0; b < y.num_batches(); ++b)
        linalg::axpy<Beta>(y.batch(b), alphas, x.batch(b)...);
}

/// Add scaled vector z = αx + y.
template <simdifiable_multi Vx, simdifiable_multi Vy, simdifiable_multi Vz,
          std::convertible_to<simdified_simd_t<Vx>> Ta>
    requires simdify_compatible<Vx, Vy, Vz>
void axpy(Ta alpha, Vx &&x, Vy &&y, Vz &&z) {
    axpby(alpha, x, 1, y, z);
}

/// Add scaled vector y = αx + βy (where β is a compile-time constant).
template <auto Beta = 1, simdifiable_multi Vx, simdifiable_multi Vy,
          std::convertible_to<simdified_simd_t<Vx>> Ta>
    requires simdify_compatible<Vx, Vy>
void axpy(Ta alpha, Vx &&x, Vy &&y) {
    BATMAT_ASSERT(x.num_batches() == y.num_batches());
    for (index_t b = 0; b < x.num_batches(); ++b)
        linalg::axpy<Beta>(alpha, x.batch(b), y.batch(b));
}

/// Negate a matrix or vector B = -A.
template <simdifiable_multi VA, simdifiable_multi VB, int Rotate = 0>
    requires simdify_compatible<VA, VB>
void negate(VA &&A, VB &&B, with_rotate_t<Rotate> rot = {}) {
    BATMAT_ASSERT(A.num_batches() == B.num_batches());
    for (index_t b = 0; b < A.num_batches(); ++b)
        linalg::negate(A.batch(b), B.batch(b), rot);
}

/// Negate a matrix or vector A = -A.
template <simdifiable_multi VA, int Rotate = 0>
void negate(VA &&A, with_rotate_t<Rotate> rot = {}) {
    for (index_t b = 0; b < A.num_batches(); ++b)
        linalg::negate(A.batch(b), rot);
}

/// Subtract two matrices or vectors C = A - B. Rotate affects B.
template <simdifiable_multi VA, simdifiable_multi VB, simdifiable_multi VC, int Rotate = 0>
    requires simdify_compatible<VA, VB, VC>
void sub(VA &&A, VB &&B, VC &&C, with_rotate_t<Rotate> rot = {}) {
    BATMAT_ASSERT(A.num_batches() == B.num_batches());
    BATMAT_ASSERT(A.num_batches() == C.num_batches());
    for (index_t b = 0; b < A.num_batches(); ++b)
        linalg::sub(A.batch(b), B.batch(b), C.batch(b), rot);
}

/// Subtract two matrices or vectors A = A - B. Rotate affects B.
template <simdifiable_multi VA, simdifiable_multi VB, int Rotate = 0>
    requires simdify_compatible<VA, VB>
void sub(VA &&A, VB &&B, with_rotate_t<Rotate> rot = {}) {
    BATMAT_ASSERT(A.num_batches() == B.num_batches());
    for (index_t b = 0; b < A.num_batches(); ++b)
        linalg::sub(A.batch(b), B.batch(b), rot);
}

/// Add two matrices or vectors C = A + B. Rotate affects B.
template <simdifiable_multi VA, simdifiable_multi VB, simdifiable_multi VC, int Rotate = 0>
    requires simdify_compatible<VA, VB, VC>
void add(VA &&A, VB &&B, VC &&C, with_rotate_t<Rotate> rot = {}) {
    BATMAT_ASSERT(A.num_batches() == B.num_batches());
    BATMAT_ASSERT(A.num_batches() == C.num_batches());
    for (index_t b = 0; b < A.num_batches(); ++b)
        linalg::add(A.batch(b), B.batch(b), C.batch(b), rot);
}

/// Add two matrices or vectors A = A + B. Rotate affects B.
template <simdifiable_multi VA, simdifiable_multi VB, int Rotate = 0>
    requires simdify_compatible<VA, VB>
void add(VA &&A, VB &&B, with_rotate_t<Rotate> rot = {}) {
    BATMAT_ASSERT(A.num_batches() == B.num_batches());
    for (index_t b = 0; b < A.num_batches(); ++b)
        linalg::add(A.batch(b), B.batch(b), rot);
}

/// Apply a function to all elements of the given matrices or vectors.
template <class F, simdifiable_multi VA, simdifiable_multi... VAs>
    requires simdify_compatible<VA, VAs...>
void for_each_elementwise(F &&fun, VA &&A, VAs &&...As) {
    BATMAT_ASSERT(((A.num_batches() == As.num_batches()) && ...));
    for (index_t b = 0; b < A.num_batches(); ++b)
        linalg::for_each_elementwise(fun, A.batch(b), As.batch(b)...);
}

/// Apply a function to all elements of the given matrices or vectors, storing the result in the
/// first argument.
template <class F, simdifiable_multi VA, simdifiable_multi... VAs>
    requires simdify_compatible<VA, VAs...>
void transform_elementwise(F &&fun, VA &&A, VAs &&...As) {
    BATMAT_ASSERT(((A.num_batches() == As.num_batches()) && ...));
    for (index_t b = 0; b < A.num_batches(); ++b)
        linalg::transform_elementwise(fun, A.batch(b), As.batch(b)...);
}

/// Apply a function to all elements of the given matrices or vectors, storing the results in the
/// first two arguments.
template <class F, simdifiable_multi VA, simdifiable_multi VB, simdifiable_multi... VAs>
    requires simdify_compatible<VA, VB, VAs...>
void transform2_elementwise(F &&fun, VA &&A, VB &&B, VAs &&...As) {
    BATMAT_ASSERT(A.num_batches() == B.num_batches());
    BATMAT_ASSERT(((A.num_batches() == As.num_batches()) && ...));
    for (index_t b = 0; b < A.num_batches(); ++b)
        linalg::transform2_elementwise(fun, A.batch(b), B.batch(b), As.batch(b)...);
}

/// Apply a function to all elements of the given matrices or vectors, storing the results in the
/// tuple of matrices given as the first argument.
template <class F, simdifiable_multi... VAs, simdifiable_multi... VBs>
    requires simdify_compatible<VAs..., VBs...>
void transform_n_elementwise(F &&fun, std::tuple<VAs...> As, VBs &&...Bs) {
    using std::get;
    auto &&a0 = get<0>(As);
    BATMAT_ASSERT(((a0.num_batches() == Bs.num_batches()) && ...));
    BATMAT_ASSERT([&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ((a0.num_batches() == get<Is>(As).num_batches()) && ...);
    }(std::make_index_sequence<sizeof...(VAs)>()));
    for (index_t b = 0; b < a0.num_batches(); ++b)
        linalg::transform_n_elementwise(
            fun, std::apply([&](auto &&...a) { return std::make_tuple(a.batch(b)...); }, As),
            Bs.batch(b)...);
}

/// @}

/// @}

} // namespace multi

} // namespace batmat::linalg
