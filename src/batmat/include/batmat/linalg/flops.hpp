#pragma once

#include <batmat/assume.hpp>
#include <batmat/config.hpp>
#include <batmat/linalg/structure.hpp>
#include <algorithm>

namespace batmat::linalg::flops {

struct FlopCount {
    index_t fma  = 0;
    index_t mul  = 0;
    index_t add  = 0;
    index_t div  = 0;
    index_t sqrt = 0;
};

constexpr FlopCount operator+(FlopCount a, FlopCount b) {
    return {.fma  = a.fma + b.fma,
            .mul  = a.mul + b.mul,
            .add  = a.add + b.add,
            .div  = a.div + b.div,
            .sqrt = a.sqrt + b.sqrt};
}

constexpr index_t total(FlopCount c) { return c.fma + c.mul + c.add + c.div + c.sqrt; }

constexpr FlopCount gemm(index_t m, index_t n, index_t k) { return {.fma = m * k * n}; }

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

constexpr FlopCount gemmt(index_t m, index_t n, index_t k, MatrixStructure sA, MatrixStructure sB,
                          MatrixStructure sC) {
    return trmm(m, n, k, sA, sB, sC);
}

constexpr FlopCount gemmt_diag(index_t m, index_t n, index_t k, MatrixStructure sC) {
    constexpr auto sA = MatrixStructure::General, sB = sA;
    return trmm(m, n, k, sA, sB, sC) + FlopCount{.mul = std::min(m, n) * k};
}

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

constexpr FlopCount syrk_potrf(index_t m, index_t n, index_t k) {
    BATMAT_ASSUME(m >= n);
    return potrf(m, n) + FlopCount{.fma = n * (n + 1) * k / 2 + (m - n) * n * k};
}

constexpr FlopCount trsm(index_t m, index_t n) {
    return {.fma = m * (m - 1) * n / 2, .mul = m * n, .div = m};
}

constexpr FlopCount trtri(index_t m) {
    return {.fma = (m + 1) * m * (m - 1) / 6, .div = m}; // TODO
}

} // namespace batmat::linalg::flops
