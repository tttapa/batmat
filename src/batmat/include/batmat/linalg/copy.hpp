#pragma once

#include <batmat/linalg/structure.hpp>
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

struct FillMask {
    int mask = 0;
};

template <class T, class Abi, MatrixStructure S = MatrixStructure::General, FillMask Msk = {},
          StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void copy(T a, view<T, Abi, OB> B) {
    using enum MatrixStructure;
    GUANAQO_TRACE(S == General           ? "fill"
                  : S == LowerTriangular ? "fill(L)"
                  : S == UpperTriangular ? "fill(U)"
                                         : "fill(?)",
                  0, B.rows() * B.cols() * B.depth()); // TODO
    const auto I = B.rows(), J = B.cols();
    if (I == 0 || J == 0 || B.depth() == 0)
        return;

    using types = simd_view_types<T, Abi>;
    typename types::simd A{a};
    const index_t JI_adif = std::max<index_t>(0, J - I), IJ_adif = std::max<index_t>(0, I - J);
    if constexpr (OB == StorageOrder::ColMajor)
        for (index_t j = 0; j < J; ++j) {
            const index_t i0 = S == LowerTriangular ? std::max<index_t>(0, j - JI_adif) : 0;
            const index_t i1 = S == UpperTriangular ? std::min(j + 1 + IJ_adif, I) : I;
            BATMAT_UNROLLED_IVDEP_FOR (8, index_t i = i0; i < i1; ++i)
                types::template aligned_store<Msk.mask>(A, &B(0, i, j));
        }
    else
        for (index_t i = 0; i < I; ++i) {
            const index_t j0 = S == UpperTriangular ? std::max<index_t>(0, i - IJ_adif) : 0;
            const index_t j1 = S == LowerTriangular ? std::min(i + 1 + JI_adif, J) : J;
            BATMAT_UNROLLED_IVDEP_FOR (8, index_t j = j0; j < j1; ++j)
                types::template aligned_store<Msk.mask>(A, &B(0, i, j));
        }
}

struct CopyRotate {
    int rotate = 0;
    int mask   = rotate;
};

template <class T, class Abi, MatrixStructure S = MatrixStructure::General, CopyRotate Rot = {},
          StorageOrder OA, StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void copy(view<const T, Abi, OA> A, view<T, Abi, OB> B)
    requires(!std::same_as<Abi, datapar::scalar_abi<T>> || S != MatrixStructure::General)
{
    using ops::rotl;
    using ops::rotr;
    using enum MatrixStructure;
    GUANAQO_TRACE(S == General           ? "copy"
                  : S == LowerTriangular ? "copy(L)"
                  : S == UpperTriangular ? "copy(U)"
                                         : "copy(?)",
                  0, A.rows() * A.cols() * A.depth()); // TODO
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    const auto I = A.rows(), J = A.cols();
    if (I == 0 || J == 0 || A.depth() == 0)
        return;

    using types           = simd_view_types<T, Abi>;
    const index_t JI_adif = std::max<index_t>(0, J - I), IJ_adif = std::max<index_t>(0, I - J);
    if constexpr (OA == StorageOrder::ColMajor)
        for (index_t j = 0; j < J; ++j) {
            const index_t i0 = S == LowerTriangular ? std::max<index_t>(0, j - JI_adif) : 0;
            const index_t i1 = S == UpperTriangular ? std::min(j + 1 + IJ_adif, I) : I;
            BATMAT_UNROLLED_IVDEP_FOR (8, index_t i = i0; i < i1; ++i)
                types::template aligned_store<Rot.mask>(
                    rotl<Rot.rotate>(types::aligned_load(&A(0, i, j))), &B(0, i, j));
        }
    else
        for (index_t i = 0; i < I; ++i) {
            const index_t j0 = S == UpperTriangular ? std::max<index_t>(0, i - IJ_adif) : 0;
            const index_t j1 = S == LowerTriangular ? std::min(i + 1 + JI_adif, J) : J;
            BATMAT_UNROLLED_IVDEP_FOR (8, index_t j = j0; j < j1; ++j)
                types::template aligned_store<Rot.mask>(
                    rotl<Rot.rotate>(types::aligned_load(&A(0, i, j))), &B(0, i, j));
        }
}

template <class T, class Abi, MatrixStructure S = MatrixStructure::General, CopyRotate Rot = {},
          StorageOrder OA, StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void copy(view<const T, Abi, OA> A, view<T, Abi, OB> B)
    requires(std::same_as<Abi, datapar::scalar_abi<T>> && OA == OB && S == MatrixStructure::General)
{
    GUANAQO_TRACE("copy", 0, A.rows() * A.cols() * A.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    if constexpr (Rot.mask != 0) // Scalar only
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

template <class T, class Abi, MatrixStructure S = MatrixStructure::General, CopyRotate Rot = {},
          StorageOrder OA, StorageOrder OB>
[[gnu::flatten, gnu::noinline]] void copy(view<const T, Abi, OA> A, view<T, Abi, OB> B)
    requires(std::same_as<Abi, datapar::scalar_abi<T>> && OA != OB && S == MatrixStructure::General)
{
    GUANAQO_TRACE("copy(T)", 0, A.rows() * A.cols() * A.depth());
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    if constexpr (Rot.mask != 0) // Scalar only
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

} // namespace batmat::linalg
