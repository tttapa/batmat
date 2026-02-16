#pragma once

#include <batmat/assume.hpp>
#include <batmat/config.hpp>
#include <batmat/linalg/structure.hpp>
#include <algorithm>

namespace batmat::linalg::flops {

/// @addtogroup topic-linalg-flops
/// @{

/// Count of individual floating point operations, broken down by type.
struct FlopCount {
    index_t fma  = 0;
    index_t mul  = 0;
    index_t add  = 0;
    index_t div  = 0;
    index_t sqrt = 0;
};

/// Combine two flop counts by summing the counts of each operation type.
constexpr FlopCount operator+(FlopCount a, FlopCount b) {
    return {.fma  = a.fma + b.fma,
            .mul  = a.mul + b.mul,
            .add  = a.add + b.add,
            .div  = a.div + b.div,
            .sqrt = a.sqrt + b.sqrt};
}

/// Compute the total number of floating point operations by summing the counts of all operation
/// types.
constexpr index_t total(FlopCount c) { return c.fma + c.mul + c.add + c.div + c.sqrt; }

/// Matrix-matrix multiplication of m×k and k×n matrices.
/// @implementation{flops-gemm}
// [flops-gemm]
constexpr FlopCount gemm(index_t m, index_t n, index_t k) { return {.fma = m * k * n}; }
// [flops-gemm]

/// Matrix-matrix multiplication of m×k and k×n matrices where one or more of the matrices
/// are triangular or trapezoidal.
/// @implementation{flops-trmm}
// [flops-trmm]
constexpr FlopCount trmm(index_t m, index_t n, index_t k, MatrixStructure sA, MatrixStructure sB,
                         MatrixStructure sC) {
    using enum MatrixStructure;
    if (sB == General && sC == General) {
        if (sA == LowerTriangular || sA == UpperTriangular) { // trapezoidal A
            if (m >= k)                                       // tall A
                // x       x x x x          x x x   x x x x
                // x x     x x x x          x x x   x x x x
                // x x x   x x x x          x x x   x x x x
                // x x x                      x x
                // x x x                        x
                return {.fma = k * (k + 1) / 2 * n + (m - k) * k * n};
            else // wide A
                // x x x       x x x        x x x x x   x x x
                // x x x x     x x x          x x x x   x x x
                // x x x x x   x x x            x x x   x x x
                //             x x x                    x x x
                //             x x x                    x x x
                return {.fma = m * (m + 1) / 2 * n + (k - m) * (k - m) * n};
        } else if (sA == General) {
            return {.fma = m * k * n};
        } else {
            BATMAT_ASSUME(!"invalid structure");
        }
    } else if (sA == General && sC == General) {
        if (sB == LowerTriangular || sB == UpperTriangular) { // trapezoidal B
            if (n >= k)                                       // wide B
                return {.fma = k * (k + 1) / 2 * m + (n - k) * k * m};
            else // tall B
                return {.fma = n * (n + 1) / 2 * m + (k - n) * (k - n) * m};
        } else {
            BATMAT_ASSUME(!"invalid structure");
        }
    } else if (sA == General && sB == General) {
        if (sC == LowerTriangular || sC == UpperTriangular) {
            BATMAT_ASSUME(m == n);
            return {.fma = m * (m + 1) / 2 * k};
        } else {
            BATMAT_ASSUME(!"invalid structure");
        }
    } else if (sC == LowerTriangular || sC == UpperTriangular) {
        if (sA == transpose(sB)) {
            BATMAT_ASSUME(m == n);
            BATMAT_ASSUME(m == k);
            return {.fma = m * (m + 1) * (m + 2) / 6};
        } else {
            BATMAT_ASSUME(!"invalid structure");
        }
    } else {
        BATMAT_ASSUME(!"unsupported structure");
    }
    return {};
}
// [flops-trmm]

/// Matrix-matrix multiplication of m×k and k×n matrices where the result is symmetric.
/// @implementation{flops-gemmt}
// [flops-gemmt]
constexpr FlopCount gemmt(index_t m, index_t n, index_t k, MatrixStructure sA, MatrixStructure sB,
                          MatrixStructure sC) {
    return trmm(m, n, k, sA, sB, sC);
}
// [flops-gemmt]

/// Symmetric rank-k update of n×n matrices.
/// @implementation{flops-syrk}
// [flops-syrk]
constexpr FlopCount syrk(index_t n, index_t k) {
    return gemmt(n, n, k, MatrixStructure::General, MatrixStructure::General,
                 MatrixStructure::LowerTriangular);
}
// [flops-syrk]

/// Matrix-matrix multiplication of m×k and k×n matrices with a diagonal k×k matrix in the middle,
/// where the result is symmetric.
/// @implementation{flops-gemmt-diag}
// [flops-gemmt-diag]
constexpr FlopCount gemmt_diag(index_t m, index_t n, index_t k, MatrixStructure sC) {
    constexpr auto sA = MatrixStructure::General, sB = sA;
    return trmm(m, n, k, sA, sB, sC) + FlopCount{.mul = std::min(m, n) * k};
}
// [flops-gemmt-diag]

/// Cholesky factorization and triangular solve for an m×n matrix with m≥n.
/// @implementation{flops-potrf}
// [flops-potrf]
constexpr FlopCount potrf(index_t m, index_t n) {
    BATMAT_ASSUME(m >= n);
    return {
        .fma = (n + 1) * n * (n - 1) / 6    // Schur complement (square)
               + (m - n) * n * (n - 1) / 2, //                  (bottom)
        .mul = n * (n - 1) / 2              // multiplication by inverse pivot (square)
               + (m - n) * n,               //                                 (bottom)
        .div  = n,                          // inverting pivot
        .sqrt = n,                          // square root pivot
    };
}
// [flops-potrf]

/// Hyperbolic Householder factorization update with L n×n and A nr×k.
/// @implementation{flops-hyh-square}
// [flops-hyh-square]
constexpr FlopCount hyh_square(index_t n, index_t k) {
    return {
        .fma  = k * n * n + 2 * n,
        .mul  = k * n + (n + 1) * n / 2 + n,
        .add  = (n + 1) * n / 2 + n,
        .div  = 2 * n,
        .sqrt = n,
    };
}
// [flops-hyh-square]

/// Hyperbolic Householder factorization application to L2 nr×nc and A2 nr×k.
/// @implementation{flops-hyh-apply}
// [flops-hyh-apply]
constexpr FlopCount hyh_apply(index_t nr, index_t nc, index_t k) {
    return {
        .fma = 2 * nr * k * nc,
        .mul = nr * nc,
        .add = nr * nc,
    };
}
// [flops-hyh-apply]

/// Hyperbolic Householder factorization update with L nr×nc and A nr×k.
/// @implementation{flops-hyh}
// [flops-hyh]
constexpr FlopCount hyh(index_t nr, index_t nc, index_t k) {
    BATMAT_ASSUME(nr >= nc);
    return hyh_square(nc, k) + hyh_apply(nr - nc, nc, k);
}
// [flops-hyh]

/// Fused symmetric rank-k update and Cholesky factorization of an m×n matrix with m≥n.
/// @implementation{flops-syrk-potrf}
// [flops-syrk-potrf]
constexpr FlopCount syrk_potrf(index_t m, index_t n, index_t k) {
    BATMAT_ASSUME(m >= n);
    return potrf(m, n) + FlopCount{.fma = n * (n + 1) * k / 2 + (m - n) * n * k};
}
// [flops-syrk-potrf]

/// Triangular solve of m×n matrices.
/// @implementation{flops-trsm}
// [flops-trsm]
constexpr FlopCount trsm(index_t m, index_t n) {
    return {.fma = m * (m - 1) * n / 2, .mul = m * n, .div = m};
}
// [flops-trsm]

/// Triangular inversion of an m×m matrix.
/// @implementation{flops-trtri}
// [flops-trtri]
constexpr FlopCount trtri(index_t m) {
    return {.fma = (m + 1) * m * (m - 1) / 6, .div = m}; // TODO
}
// [flops-trtri]

/// @}

} // namespace batmat::linalg::flops
