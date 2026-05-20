#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/micro-kernels/sytrd.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/lut.hpp>
#include <batmat/ops/cneg.hpp>
#include <batmat/ops/rotate.hpp>
#include <guanaqo/trace.hpp>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

namespace batmat::linalg::micro_kernels::sytrd {

template <class T, class Abi, KernelConfig Conf, StorageOrder OD>
inline const constinit auto microkernel_diag_lut =
    make_1d_lut<SizeR<T, Abi>>([]<index_t Row>(index_constant<Row>) {
        return sytrd_diag_microkernel<T, Abi, Conf, Row + 1, OD>;
    });

template <class T, class Abi, KernelConfig Conf, index_t R, StorageOrder OD>
[[gnu::hot, gnu::flatten]] void
sytrd_diag_microkernel(index_t k, triangular_accessor<T, Abi, SizeR<T, Abi>> W, uview<T, Abi, OD> D,
                       uview<T, Abi, StorageOrder::ColMajor> Y) noexcept {
    using std::copysign;
    using std::sqrt;
    using simd = datapar::simd<T, Abi>;
    BATMAT_ASSUME(k > R);

    //          j    j+1   j+2    R
    // ┌─────┬─────┬─────┬─────┐
    // │  D  │  ·  │  ·  │  ·  │
    // ├─────┼─────┼─────┼─────┤
    // │  d1 │ a11 │  ×  │  ×  │  j
    // ├─────┼─────┼─────┼─────┤
    // │  b2 │ a21 │ a22 │  ×  │  j+1
    // ├─────┼─────┼─────┼─────┤
    // │  B3 │ a31 │ a32 │  A3 │  j+2
    // └─────┴─────┴─────┴─────┘

    //                   j   j+1  j+2
    // ┌──────────────┬────┬────┬───────────────┐
    // │  d    ·    · │  · │  · │  ·    ·    ·  │     d: diagonal elements of tridiagonal matrix
    // │              │    │    │               │     e: off-diagonal elements of tridiagonal matrix
    // │  c    d    · │  · │  · │  ·    ·    ·  │     b: Householder reflectors
    // │              │    │    │               │     a: original matrix
    // │  b    c    d │  · │  · │  ·    ·    ·  │     ×: implicitly symmetric part
    // │              ├────┼────┼───────────────┤
    // │  b    b    c │  a │  × │  ×    ×    ×  │  j
    // ├──────────────┼────┼────┼───────────────┤
    // │  b    b    b │  a │  a │  ×    ×    ×  │  j+1
    // ├──────────────┼────┼────┼───────────────┤
    // │  b    b    b │  a │  a │  a    ×    ×  │  j+2
    // │              │    │    │               │
    // │  b    b    b │  a │  a │  a    a    ×  │
    // │              │    │    │               │
    // │  b    b    b │  a │  a │  a    a    a  │
    // └──────────────┴────┴────┴───────────────┘
    //    │              │    │    ╰─ symv A3 by a31
    //    │              │    ╰─ dot a32 with a31
    //    │              ╰─ ±norm a31 and a21 becomes c(j)
    //    ╰─ dot B3 with a31 for block Householder

    // symv:
    // [  W(j+2, j)  ]   [  A(j+2, j+2)   A(j+3, j+2)   A(j+4, j+2)   ...  ]  [  A(j+2, j)  ]
    // [  W(j+3, j)  ] = [  A(j+3, j+2)   A(j+3, j+3)   A(j+4, j+3)   ...  ]  [  A(j+3, j)  ]
    // [  W(j+4, j)  ]   [  A(j+4, j+2)   A(j+4, j+3)   A(j+4, j+4)   ...  ]  [  A(j+4, j)  ]
    // [    ...      ]   [    ...           ...           ...         ...  ]  [    ...      ]
    //
    // W(j+2, j) = sum(l=j+2..k) A(l, j+2) A(l, j)
    // W(j+3, j) = A(j+3, j+2) A(j+2, j) + sum(l=j+3..k) A(l, j+3) A(l, j)
    // W(j+4, j) = A(j+4, j+2) A(j+2, j) + A(j+4, j+3) A(j+3, j) + sum(l=j+4..k) A(l, j+4) A(l, j)
    // W(q, j) = sum(p=j+2..q-1) A(q, p) A(p, j) + sum(l=q..k) A(l, q) A(l, j)

    UNROLL_FOR (index_t j = 0; j < R; ++j) {
        using std::max;
        simd Axj[R + 1];
        UNROLL_FOR (index_t i = j + 1; i < R + 1; ++i)
            Axj[i] = D.load(i, j);
        // Compute inner products between a(j) and b(i<j), a(j), and a(i>j) (symv)
        simd bb[R + 1]{};
        // Triangular part
        UNROLL_FOR (index_t q = j + 2; q < R + 1; ++q) {
            UNROLL_FOR (index_t i = 0; i <= q; ++i)
                bb[i] += D.load(q, i) * Axj[q];
            UNROLL_FOR (index_t p = j + 2; p < q; ++p)
                bb[q] += D.load(q, p) * Axj[p];
        }
        // Rectangular part
        for (index_t q = max(R + 1, j + 2); q < k; ++q) {
            simd Aqx[R + 1];
            UNROLL_FOR (index_t i = 0; i < R + 1; ++i)
                Aqx[i] = D.load(q, i);
            UNROLL_FOR (index_t i = 0; i < R + 1; ++i)
                bb[i] += Aqx[i] * Aqx[j];
            simd Yl{};
            UNROLL_FOR (index_t p = j + 2; p < R + 1; ++p)
                Yl += Aqx[p] * Axj[p];
            Y.store(Yl, q, j); // W(q, j) = sum(p=j+2..q-1) A(q, p) A(p, j)
        }
        const simd a21 = Axj[j + 1];
        bb[j] += a21 * a21;
        // bb[i<j] now contain the inner products of a31 with the previous Householder vectors
        //         (except for the first components, which are implicitly 1 and are added later).
        // bb[j]   contains the squared norm of (a21, a31).
        // bb[j+1] contains the dot product of a31 and a32.
        // bb[i>j+1] contain the top rows of the symmetric product A3 a31 (complete).
        // Y[i>=R, j] contains part of the symmetric product A3 a31, but still requires adding
        //            the contributions from all columns >=R (including dot products with the upper
        //            triangle of A3)

        // Energy condition and Householder coefficients
        const simd c̃j = copysign(sqrt(bb[j]), a21), β = a21 + c̃j;
        const simd inv_τ = β / c̃j, inv_β = simd{1} / β;

        // Save block Householder matrix W
        UNROLL_FOR (index_t i = 0; i < j; ++i)
            // Multiply implicit first component of the current Householder vector by the
            // corresponding row of the previous Householder vectors, and add it to the previously
            // computed inner products with a31, scaled by β⁻¹ to go from a31 to the normalized
            // Householder vector.
            W.store(bb[i] * inv_β + D.load(j + 1, i), i, j);
        W.store(inv_τ, j, j); // inverse of diagonal

        // Finish the symmetric product A3 a31
        for (index_t i = max(R + 1, j + 2); i < k; ++i) {
            simd yi       = Y.load(i, j);
            const simd xi = D.load(i, j);
            yi += D.load(i, i) * xi;                          // diagonal term
            for (index_t l = max(R + 1, j + 2); l < i; ++l) { // lower triangle l < i
                simd yl       = Y.load(l, j);
                const simd xl = D.load(l, j), ail = D.load(i, l); // TODO: access D column-wise
                yi += ail * xl;
                yl += ail * xi;    // symmetric contribution to y[j]
                Y.store(yl, l, j); // TODO: optimize by unrolling to avoid load/store of yl
            }
            Y.store(yi, i, j);
        }
        // Y[i>=R, j] now contains the complete bottom rows of the symmetric product A3 a31.

        // Now compute the vector w = τ⁻¹(A3 b + a32) = τ⁻¹(β⁻¹ A3 a31 + a32).
        simd Axj1[R + 1];
        UNROLL_FOR (index_t i = j + 1; i < R + 1; ++i)
            Axj1[i] = D.load(i, j + 1);
        UNROLL_FOR (index_t i = j + 2; i < R + 1; ++i)
            Y.store(bb[i] = inv_τ * (inv_β * bb[i] + Axj1[i]), i, j);
        for (index_t i = max(R + 1, j + 2); i < k; ++i) {
            simd yi = Y.load(i, j);
            Y.store(inv_τ * (inv_β * yi + D.load(i, j + 1)), i, j);
        }
        // bb[i>j+1] now contain w[i].

        const simd a2      = Axj1[j + 1];
        const simd a31ᵀa32 = bb[j + 1];
        const simd ω       = inv_τ * (inv_β * a31ᵀa32 + a2); // ω = τ⁻¹(a32ᵀb + a22)
        // Scale a31 to obtain b, and dot it with w.
        simd wᵀb_ω = ω; // accumulator for wᵀb + ω
        simd b[R + 1];
        UNROLL_FOR (index_t l = j + 2; l < R + 1; ++l) {
            b[l] = inv_β * Axj[l];
            D.store(b[l], l, j);
            wᵀb_ω += b[l] * bb[l];
        }
        for (index_t l = max(R + 1, j + 2); l < k; ++l) {
            simd bl = inv_β * D.load(l, j);
            D.store(bl, l, j);
            wᵀb_ω += bl * Y.load(l, j);
        }
        const simd γ  = inv_τ * wᵀb_ω; // γ = τ⁻¹ (wᵀb + ω)
        const simd d2 = a2 - 2 * ω + γ;
        D.store(-c̃j, j + 1, j);
        D.store(d2, j + 1, j + 1);

        // Compute and store ã32 = a32 + (γ - ω) b - w and y = w - ½γ b
        const simd γ_ω = γ - ω;
        UNROLL_FOR (index_t l = j + 2; l < R + 1; ++l) {
            simd ã32 = Axj1[l] + γ_ω * b[l] - bb[l];
            D.store(ã32, l, j + 1);
            simd yl = bb[l] - simd{0.5} * γ * b[l];
            Y.store(yl, l, j);
        }
        for (index_t l = max(R + 1, j + 2); l < k; ++l) {
            simd bl  = D.load(l, j);
            simd ã32 = D.load(l, j + 1) + γ_ω * bl - Y.load(l, j);
            D.store(ã32, l, j + 1);
            simd yl = Y.load(l, j) - simd{0.5} * γ * bl;
            Y.store(yl, l, j);
        }

        // Update the trailing submatrix A3 = A3 - byᵀ - ybᵀ
        // TODO: optimize memory accesses
        for (index_t i = j + 2; i < k; ++i)   // column of A3
            for (index_t l = i; l < k; ++l) { // row of A3 (lower triangle)
                simd Ail = D.load(l, i);
                Ail -= Y.load(i, j) * D.load(l, j) + Y.load(l, j) * D.load(i, j);
                D.store(Ail, l, i);
            }
    }
}

/// Symmetric block tridiagonalization.
template <class T, class Abi, KernelConfig Conf, StorageOrder OD>
void sytrd_register(const view<T, Abi, OD> D, const view<T, Abi> W, const view<T, Abi> Y) noexcept {
    static constexpr index_constant<SizeR<T, Abi>> R;
    const index_t k = D.rows();
    BATMAT_ASSUME(k > 0);
    BATMAT_ASSUME(D.rows() == D.cols());
    BATMAT_ASSUME(W.rows() == 0 ||
                  (W.cols() == 1 && W.rows() == std::max<index_t>(D.cols(), 1) - 1) ||
                  std::make_pair(W.rows(), W.cols()) == (sytrd_W_size<T, Abi>)(D));
    BATMAT_ASSUME(std::make_pair(Y.rows(), Y.cols()) == (sytrd_Y_size<T, Abi>)(D));

    using W_t = triangular_accessor<T, Abi, R>;
    alignas(W_t::alignment()) T W_sto[W_t::size()];

    // Sizeless views to partition and pass to the micro-kernels
    const uview<T, Abi, OD> D_                     = D;
    const uview<T, Abi, StorageOrder::ColMajor> W_ = W;
    const uview<T, Abi, StorageOrder::ColMajor> Y_ = Y;
    const bool store_full_W = std::make_pair(W.rows(), W.cols()) == (sytrd_W_size<T, Abi>)(D);

    // Process all diagonal blocks (in multiples of R, except the last).
    foreach_chunked_merged(0, k - 1, R, [&](index_t j, auto rem_j) {
        auto Wj  = store_full_W ? W_t{W_.middle_cols(j / R).data} : W_t{W_sto};
        auto Djj = D_.block(j, j);
        microkernel_diag_lut<T, Abi, Conf, OD>[rem_j - 1](k - j, Wj, Djj, Y_);
        if (!store_full_W && W.rows() > 0) [[unlikely]]
            for (index_t l = 0; l < rem_j; ++l)
                W_.store(Wj.load(l, l), j + l, 0);
    });
}

} // namespace batmat::linalg::micro_kernels::sytrd
