#pragma once

#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/lut.hpp>
#include <batmat/ops/transpose.hpp>
#include <batmat/platform/platform.hpp>
#include <batmat/unroll.h>
#include <guanaqo/trace.hpp>
#include <algorithm>
#include <concepts>

namespace batmat::linalg {

template <class T, class Abi, StorageOrder OA, StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void copy(view<const T, Abi, OA> A, view<T, Abi, OB> B)
    requires(!std::same_as<Abi, datapar::scalar_abi<T>>)
{
    GUANAQO_TRACE("copy", 0, A.rows() * A.cols() * A.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    if (A.rows() == 0 || A.cols() == 0 || A.depth() == 0)
        return;

    using types = simd_view_types<T, Abi>;
    for (index_t j = 0; j < A.cols(); ++j)
        BATMAT_UNROLLED_IVDEP_FOR (8, index_t i = 0; i < A.rows(); ++i)
            types::aligned_store(types::aligned_load(&A(0, i, j)), &B(0, i, j));
}

template <class T, class Abi, StorageOrder OA, StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void copy(view<const T, Abi, OA> A, view<T, Abi, OB> B)
    requires(std::same_as<Abi, datapar::scalar_abi<T>> && OA == OB)
{
    GUANAQO_TRACE("copy", 0, A.rows() * A.cols() * A.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
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

template <class T, class Abi, StorageOrder OA, StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void copy(view<const T, Abi, OA> A, view<T, Abi, OB> B)
    requires(std::same_as<Abi, datapar::scalar_abi<T>> && OA != OB)
{
    GUANAQO_TRACE("copy(T)", 0, A.rows() * A.cols() * A.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
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

} // namespace batmat::linalg
