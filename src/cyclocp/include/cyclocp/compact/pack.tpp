#include <cyclocp/compact.hpp>

#include <batmat/loop.hpp>
#include <batmat/lut.hpp>
#include <batmat/ops/transpose.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg::compact {

template <class T, class Abi>
void CompactBLAS<T, Abi>::unpack(single_batch_view A, mut_single_batch_view_scalar B) {
    GUANAQO_TRACE("unpack", 0, A.rows() * A.cols() * A.depth());
    static constexpr auto lut = make_1d_lut<simd_stride>(
        []<index_t R>(index_constant<R>) { return ops::transpose<simd_stride, R + 1, real_t>; });
    for (index_t c = 0; c < A.cols(); ++c)
        foreach_chunked(
            0, A.rows(), simd_stride,
            [&](index_t r) {
                ops::transpose<simd_stride, simd_stride>(&A(0, r, c), simd_stride, &B(0, r, c),
                                                         B.layer_stride());
            },
            [&](index_t r, index_t nr) {
                lut[nr - 1](A.block(r, c, nr, 1).data, simd_stride, B.block(r, c, nr, 1).data,
                            B.layer_stride());
            });
}

template <class T, class Abi>
void CompactBLAS<T, Abi>::unpack(single_batch_view A, mut_batch_view_scalar B) {
    GUANAQO_TRACE("unpack", 0, A.rows() * A.cols() * A.depth());
    static constexpr auto lut = make_1d_lut<simd_stride>([]<index_t R>(index_constant<R>) {
        return ops::transpose_dyn<simd_stride, R + 1, real_t>;
    });
    for (index_t c = 0; c < A.cols(); ++c)
        foreach_chunked(
            0, A.rows(), simd_stride,
            [&](index_t r) {
                ops::transpose_dyn<simd_stride, simd_stride>(&A(0, r, c), simd_stride, &B(0, r, c),
                                                             B.layer_stride(), B.depth());
            },
            [&](index_t r, index_t nr) {
                lut[nr - 1](A.block(r, c, nr, 1).data, simd_stride, B.block(r, c, nr, 1).data,
                            B.layer_stride(), B.depth());
            });
}

template <class T, class Abi>
void CompactBLAS<T, Abi>::unpack(batch_view A, mut_batch_view_scalar B) {
    assert(A.depth() == B.depth());
    foreach_chunked(
        0, B.depth(), simd_stride,
        [&](index_t l) { unpack(A.batch(l / simd_stride), B.middle_layers(l, simd_stride)); },
        [&](index_t l, index_t nl) { unpack(A.batch(l / simd_stride), B.middle_layers(l, nl)); });
}

template <class T, class Abi>
void CompactBLAS<T, Abi>::unpack_L(single_batch_view A, mut_single_batch_view_scalar B) {
    [[maybe_unused]] auto [m, M] = std::minmax({A.rows(), A.cols()});
    GUANAQO_TRACE("unpack_L", 0, (m * (m + 1) / 2 + (M - m) * m) * A.depth());
    static constexpr auto lut = make_1d_lut<simd_stride>(
        []<index_t R>(index_constant<R>) { return ops::transpose<simd_stride, R + 1, real_t>; });
    for (index_t c = 0; c < A.cols(); ++c)
        foreach_chunked(
            c, A.rows(), simd_stride,
            [&](index_t r) {
                ops::transpose<simd_stride, simd_stride>(&A(0, r, c), simd_stride, &B(0, r, c),
                                                         B.layer_stride());
            },
            [&](index_t r, index_t nr) {
                lut[nr - 1](A.block(r, c, nr, 1).data, simd_stride, B.block(r, c, nr, 1).data,
                            B.layer_stride());
            });
}

template <class T, class Abi>
void CompactBLAS<T, Abi>::unpack_L(single_batch_view A, mut_batch_view_scalar B) {
    [[maybe_unused]] auto [m, M] = std::minmax({A.rows(), A.cols()});
    GUANAQO_TRACE("unpack_L", 0, (m * (m + 1) / 2 + (M - m) * m) * A.depth());
    constexpr auto S   = simd_stride;
    constexpr auto lut = make_1d_lut<S>(
        []<index_t R>(index_constant<R>) { return ops::transpose_dyn<S, R + 1, real_t>; });
#if 1
    const auto colstrA = A.outer_stride() * S;
    const auto colstrB = B.outer_stride();
    const auto laystrB = B.layer_stride();
    const auto d       = B.depth();
    auto pA            = A.data;
    const auto pAend   = pA + A.cols() * colstrA;
    real_t *pB         = B.data;
    auto rowcnt        = A.rows();
    while (pA < pAend) {
        const real_t *pA_ = pA;
        real_t *pB_       = pB;
        index_t r;
        for (r = 0; r + S <= rowcnt; r += S) {
            ops::transpose_dyn<S, S>(pA_, S, pB_, laystrB, d);
            pA_ += S * S;
            pB_ += S;
        }
        index_t nr = rowcnt - r;
        if (nr > 0) [[unlikely]]
            lut[nr - 1](pA_, S, pB_, laystrB, d);
        pA += colstrA + S;
        pB += colstrB + 1;
        --rowcnt;
    }
#else
    for (index_t c = 0; c < A.cols(); ++c) {
        index_t r;
        for (r = c; r + S <= A.rows(); r += S)
            ops::transpose_dyn<S, S>(&A(0, r, c), S, &B(0, r, c), B.layer_stride(), B.depth());
        index_t nr = A.rows() - r;
        if (nr > 0) [[unlikely]]
            lut[nr - 1](&A(0, r, c), S, &B(0, r, c), B.layer_stride(), B.depth());
    }
#endif
}

template <class T, class Abi>
void CompactBLAS<T, Abi>::unpack_L(batch_view A, mut_batch_view_scalar B) {
    assert(A.depth() >= B.depth());
    foreach_chunked_merged(0, B.depth(), simd_stride, [&](index_t l, index_t nl) {
        unpack_L(A.batch(l / simd_stride), B.middle_layers(l, nl));
    });
}

} // namespace batmat::linalg::compact
