#include <experimental/simd>
#include <iostream>
#include <print>

#include <guanaqo/print.hpp>

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/rsqrt.hpp>
#include <koqkatoo/unroll.h>

namespace stdx = std::experimental;
using real_t   = koqkatoo::real_t;
using index_t  = koqkatoo::index_t;

#define UNROLL_FOR(...) KOQKATOO_FULLY_UNROLLED_FOR (__VA_ARGS__)

#if 0

template <index_t ND> // number of diagonal subblocks
void potrf_trsm_kernel(real_t *L, index_t ldL, index_t rows) {
    using koqkatoo::linalg::compact::micro_kernels::rsqrt;
    constexpr index_t BD  = 4; // block size of diagonal subblocks
    constexpr auto nalign = stdx::element_aligned;
    using simd_diag = stdx::simd<real_t, stdx::simd_abi::deduce_t<real_t, BD>>;
    simd_diag Dr[BD * ND * (ND + 1) / 2 + ND];
    auto index = [](index_t r, index_t c, index_t k) {
        return BD * (c * (2 * ND - 1 - c) / 2 + r) + k;
    };
    auto inv_index = [](index_t c) { return BD * ND * (ND + 1) / 2 + c; };
    {
        const real_t *pL = L;
        UNROLL_FOR (index_t j = 0; j < ND; ++j) {        // block column
            UNROLL_FOR (index_t jj = 0; jj < BD; ++jj) { // column
                UNROLL_FOR (index_t i = j; i < ND; ++i)  // block row
                    Dr[index(i, j, jj)].copy_from(pL + i * BD, nalign);
                pL += ldL;
            }
        }
    }
    UNROLL_FOR (index_t j = 0; j < ND; ++j) {        // block column
        UNROLL_FOR (index_t jj = 0; jj < BD; ++jj) { // column
            auto scal_pivot = rsqrt(real_t{Dr[index(j, j, jj)][jj]});
            simd_diag pivot{scal_pivot};
            UNROLL_FOR (index_t i = j; i < ND; ++i)
                Dr[index(i, j, jj)] *= pivot;
            Dr[inv_index(j)][jj] = scal_pivot;
            UNROLL_FOR (index_t k = j; k < ND; ++k) { // block column syrk
                const index_t k0 = k == j ? jj + 1 : 0;
                UNROLL_FOR (index_t kk = k0; kk < BD; ++kk) { // column syrk
                    simd_diag fac{Dr[index(k, j, jj)][kk]};
                    // std::println("[{}, {}, {}] = {}", k, j, kk) << fac[0] << std::endl;
                    UNROLL_FOR (index_t i = k; i < ND; ++i)
                        Dr[index(i, k, kk)] -= Dr[index(i, j, jj)] * fac;
                }
            }
        }
    }
    {
        real_t *pL = L;
        UNROLL_FOR (index_t j = 0; j < ND; ++j) {        // block column
            UNROLL_FOR (index_t jj = 0; jj < BD; ++jj) { // column
                UNROLL_FOR (index_t i = j; i < ND; ++i)  // block row
                    Dr[index(i, j, jj)].copy_to(pL + i * BD, nalign);
                pL += ldL;
            }
        }
    }
    constexpr index_t BT = 8;
    using simd_tail = stdx::simd<real_t, stdx::simd_abi::deduce_t<real_t, BT>>;
    for (index_t br = ND * BD; br + BT <= rows; br += BT) { // block row
        UNROLL_FOR (index_t bc = 0; bc < ND; ++bc) {        // block column
            UNROLL_FOR (index_t c = 0; c < BD; ++c) {       // column
                simd_tail Xij{&L[br + (bc * BD + c) * ldL], nalign};
                UNROLL_FOR (index_t k = 0; k <= bc; ++k) { // block column inner
                    const index_t k0 = k == bc ? c : BD;
                    UNROLL_FOR (index_t kk = 0; kk < k0; ++kk) { // column inner
                        // This is slow because of the extract+broadcasts
                        real_t Aik = Dr[index(bc, k, kk)][c];
                        simd_tail Xkj{&L[br + (k * BD + kk) * ldL], nalign};
                        Xij -= Aik * Xkj;
                    }
                }
                Xij *= Dr[inv_index(bc)][c];
                Xij.copy_to(&L[br + (bc * BD + c) * ldL], nalign);
            }
        }
    }
}

template void potrf_trsm_kernel<1>(real_t *L, index_t ldL, index_t rows);
template void potrf_trsm_kernel<2>(real_t *L, index_t ldL, index_t rows);
template void potrf_trsm_kernel<3>(real_t *L, index_t ldL, index_t rows);
template void potrf_trsm_kernel<4>(real_t *L, index_t ldL, index_t rows);

#else

template <index_t NS> // number of subdiagonal blocks
void potrf_trsm_kernel(real_t *L, index_t ldL, index_t rows) {
    using koqkatoo::linalg::compact::micro_kernels::rsqrt;
    constexpr index_t BD  = 8; // block size of diagonal subblocks // TODO
    constexpr index_t BT  = 8; // block size of subdiagonal blocks
    constexpr auto nalign = stdx::element_aligned;
    using simd_diag = stdx::simd<real_t, stdx::simd_abi::deduce_t<real_t, BD>>;
    using simd_tail = stdx::simd<real_t, stdx::simd_abi::deduce_t<real_t, BT>>;
    simd_diag Dr[BD];
    simd_tail Sr[NS][BD];
    {
        const real_t *pL = L;
        UNROLL_FOR (index_t jj = 0; jj < BD; ++jj) { // column
            Dr[jj].copy_from(pL, nalign);
            UNROLL_FOR (index_t i = 0; i < NS; ++i)
                Sr[i][jj].copy_from(pL + BD + i * BT, nalign);
            pL += ldL;
        }
    }
    UNROLL_FOR (index_t jj = 0; jj < BD; ++jj) { // column
        auto scal_pivot = rsqrt(real_t{Dr[jj][jj]});
        simd_diag pivot{scal_pivot};
        Dr[jj] *= pivot;
        UNROLL_FOR (index_t i = 0; i < NS; ++i)
            Sr[i][jj] *= pivot;
        UNROLL_FOR (index_t kk = jj + 1; kk < BD; ++kk) { // column syrk
            simd_diag fac{Dr[jj][kk]};
            Dr[kk] -= Dr[jj] * fac;
            UNROLL_FOR (index_t i = 0; i < NS; ++i)
                Sr[i][kk] -= Sr[i][jj] * fac;
        }
    }
    {
        real_t *pL = L;
        UNROLL_FOR (index_t jj = 0; jj < BD; ++jj) { // column
            Dr[jj].copy_to(pL, nalign);
            UNROLL_FOR (index_t i = 0; i < NS; ++i)
                Sr[i][jj].copy_to(pL + BD + i * BT, nalign);
            pL += ldL;
        }
    }
}

#endif

template void potrf_trsm_kernel<1>(real_t *L, index_t ldL, index_t rows);
template void potrf_trsm_kernel<2>(real_t *L, index_t ldL, index_t rows);
template void potrf_trsm_kernel<3>(real_t *L, index_t ldL, index_t rows);
template void potrf_trsm_kernel<4>(real_t *L, index_t ldL, index_t rows);