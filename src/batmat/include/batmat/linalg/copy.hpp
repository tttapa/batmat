#pragma once

#include <batmat/linalg/shift.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/structure.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/lut.hpp>
#include <batmat/ops/rotate.hpp>
#include <batmat/ops/transpose.hpp>
#include <batmat/platform/platform.hpp>
#include <batmat/unroll.h>
#include <guanaqo/trace.hpp>
#include <algorithm>
#include <concepts>

namespace batmat::linalg {

namespace detail::copy {
struct FillConfig {
    int mask              = 0;
    MatrixStructure struc = MatrixStructure::General;
};

template <class T, class Abi, FillConfig Conf = {}, StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void fill(T a, view<T, Abi, OB> B) {
    using std::max;
    using std::min;
    using enum MatrixStructure;
    [[maybe_unused]] static constexpr const auto trace_name =
        Conf.struc == General           ? GUANAQO_TRACE_STATIC_STR("fill")
        : Conf.struc == LowerTriangular ? GUANAQO_TRACE_STATIC_STR("fill(L)")
        : Conf.struc == UpperTriangular ? GUANAQO_TRACE_STATIC_STR("fill(U)")
                                        : GUANAQO_TRACE_STATIC_STR("fill(?)");
    GUANAQO_TRACE_LINALG(trace_name,
                         B.rows() * B.cols() * B.depth()); // TODO
    const auto I = B.rows(), J = B.cols();
    if (I == 0 || J == 0 || B.depth() == 0)
        return;

    using types = simd_view_types<T, Abi>;
    typename types::simd A{a};
    const index_t JI_adif = max<index_t>(0, J - I), IJ_adif = max<index_t>(0, I - J);
    if constexpr (OB == StorageOrder::ColMajor)
        for (index_t j = 0; j < J; ++j) {
            const index_t i0 = Conf.struc == LowerTriangular ? max<index_t>(0, j - JI_adif) : 0;
            const index_t i1 = Conf.struc == UpperTriangular ? min(j + 1 + IJ_adif, I) : I;
            BATMAT_UNROLLED_IVDEP_FOR (8, index_t i = i0; i < i1; ++i)
                types::template aligned_store<Conf.mask>(A, &B(0, i, j));
        }
    else
        for (index_t i = 0; i < I; ++i) {
            const index_t j0 = Conf.struc == UpperTriangular ? max<index_t>(0, i - IJ_adif) : 0;
            const index_t j1 = Conf.struc == LowerTriangular ? min(i + 1 + JI_adif, J) : J;
            BATMAT_UNROLLED_IVDEP_FOR (8, index_t j = j0; j < j1; ++j)
                types::template aligned_store<Conf.mask>(A, &B(0, i, j));
        }
}

struct CopyConfig {
    int rotate            = 0;
    int mask              = rotate;
    MatrixStructure struc = MatrixStructure::General;
};

template <class T, class Abi, CopyConfig Conf = {}, StorageOrder OA, StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void copy(view<const T, Abi, OA> A, view<T, Abi, OB> B)
    requires(!std::same_as<Abi, datapar::scalar_abi<T>> || Conf.struc != MatrixStructure::General)
{
    using ops::rotl;
    using ops::rotr;
    using std::max;
    using std::min;
    using enum MatrixStructure;
    [[maybe_unused]] static constexpr const auto trace_name =
        Conf.struc == General           ? GUANAQO_TRACE_STATIC_STR("copy")
        : Conf.struc == LowerTriangular ? GUANAQO_TRACE_STATIC_STR("copy(L)")
        : Conf.struc == UpperTriangular ? GUANAQO_TRACE_STATIC_STR("copy(U)")
                                        : GUANAQO_TRACE_STATIC_STR("copy(?)");
    GUANAQO_TRACE_LINALG(trace_name,
                         A.rows() * A.cols() * A.depth()); // TODO
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    const auto I = A.rows(), J = A.cols();
    if (I == 0 || J == 0 || A.depth() == 0)
        return;

    using types           = simd_view_types<T, Abi>;
    const index_t JI_adif = max<index_t>(0, J - I), IJ_adif = max<index_t>(0, I - J);
    if constexpr (OA == StorageOrder::ColMajor)
        for (index_t j = 0; j < J; ++j) {
            const index_t i0 = Conf.struc == LowerTriangular ? max<index_t>(0, j - JI_adif) : 0;
            const index_t i1 = Conf.struc == UpperTriangular ? min(j + 1 + IJ_adif, I) : I;
            BATMAT_UNROLLED_IVDEP_FOR (8, index_t i = i0; i < i1; ++i)
                types::template aligned_store<Conf.mask>(
                    rotl<Conf.rotate>(types::aligned_load(&A(0, i, j))), &B(0, i, j));
        }
    else
        for (index_t i = 0; i < I; ++i) {
            const index_t j0 = Conf.struc == UpperTriangular ? max<index_t>(0, i - IJ_adif) : 0;
            const index_t j1 = Conf.struc == LowerTriangular ? min(i + 1 + JI_adif, J) : J;
            BATMAT_UNROLLED_IVDEP_FOR (8, index_t j = j0; j < j1; ++j)
                types::template aligned_store<Conf.mask>(
                    rotl<Conf.rotate>(types::aligned_load(&A(0, i, j))), &B(0, i, j));
        }
}

template <class T, class Abi, CopyConfig Conf = {}, StorageOrder OA, StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void copy(view<const T, Abi, OA> A, view<T, Abi, OB> B)
    requires(std::same_as<Abi, datapar::scalar_abi<T>> && OA == OB &&
             Conf.struc == MatrixStructure::General)
{
    GUANAQO_TRACE_LINALG("copy", A.rows() * A.cols() * A.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    if constexpr (Conf.mask != 0) // Scalar only
        return;
    if (A.rows() == 0 || A.cols() == 0 || A.depth() == 0)
        return;

    static_assert(typename decltype(A)::batch_size_type() == 1);
    static_assert(typename decltype(B)::batch_size_type() == 1);
    if constexpr (OA == StorageOrder::ColMajor)
        for (index_t j = 0; j < A.cols(); ++j)
            std::copy_n(&A(0, 0, j), A.rows(), &B(0, 0, j));
    else
        for (index_t i = 0; i < A.rows(); ++i)
            std::copy_n(&A(0, i, 0), A.cols(), &B(0, i, 0));
}

template <class T, class Abi, CopyConfig Conf = {}, StorageOrder OA, StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void copy(view<const T, Abi, OA> A, view<T, Abi, OB> B)
    requires(std::same_as<Abi, datapar::scalar_abi<T>> && OA != OB &&
             Conf.struc == MatrixStructure::General)
{
    GUANAQO_TRACE_LINALG("copy(T)", A.rows() * A.cols() * A.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    if constexpr (Conf.mask != 0) // Scalar only
        return;
    if (A.rows() == 0 || A.cols() == 0 || A.depth() == 0)
        return;

    constexpr index_t R = ops::RowsRegTranspose<T>;
    constexpr index_t C = ops::ColsRegTranspose<T>;
    [[maybe_unused]] static const constinit auto lut =
        make_2d_lut<R, C>([]<index_t Row, index_t Col>(index_constant<Row>, index_constant<Col>) {
            return ops::transpose<Row + 1, Col + 1, T>;
        });

    // Always access A contiguously in the inner loop
    if constexpr (OA == StorageOrder::ColMajor)
        // Tiled transposition
        foreach_chunked_merged(0, A.cols(), C, [&](index_t c, auto nc) {
            foreach_chunked_merged(0, A.rows(), R, [&](index_t r, auto nr) {
                lut[nr - 1][nc - 1](&A(0, r, c), A.outer_stride(), &B(0, r, c), B.outer_stride());
            });
        });
    else
        foreach_chunked_merged(0, A.rows(), R, [&](index_t r, auto nr) {
            foreach_chunked_merged(0, A.cols(), C, [&](index_t c, auto nc) {
                lut[nc - 1][nr - 1](&A(0, r, c), A.outer_stride(), &B(0, r, c), B.outer_stride());
            });
        });
}

template <class... Opts>
constexpr CopyConfig apply_options(CopyConfig conf, Opts...) {
    if (auto s = get_rotate<Opts...>)
        conf.rotate = *s;
    if (auto s = get_mask<Opts...>)
        conf.mask = *s;
    return conf;
}
} // namespace detail::copy

/// @addtogroup topic-linalg
/// @{

/// @name Copying and filling batches of matrices
/// @{

/// B = A
template <simdifiable VA, simdifiable VB, rotate_opt... Opts>
    requires simdify_compatible<VA, VB>
void copy(VA &&A, VB &&B, Opts... opts) {
    constexpr auto conf = detail::copy::apply_options({}, opts...);
    detail::copy::copy<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(simdify(A).as_const(),
                                                                         simdify(B));
}

/// B = A
template <MatrixStructure S, simdifiable VA, simdifiable VB, rotate_opt... Opts>
    requires simdify_compatible<VA, VB>
void copy(Structured<VA, S> A, Structured<VB, S> B, Opts... opts) {
    constexpr auto conf = detail::copy::apply_options({.struc = S}, opts...);
    detail::copy::copy<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B.value));
}

/// B = a
template <simdifiable VB>
void fill(simdified_value_t<VB> a, VB &&B) {
    detail::copy::fill<simdified_value_t<VB>, simdified_abi_t<VB>, {}>(a, simdify(B));
}

/// B = a
template <MatrixStructure S, simdifiable VB>
void fill(simdified_value_t<VB> a, Structured<VB, S> B) {
    detail::copy::fill<simdified_value_t<VB>, simdified_abi_t<VB>, {.struc = S}>(a,
                                                                                 simdify(B.value));
}

/// @}

/// @}

} // namespace batmat::linalg
