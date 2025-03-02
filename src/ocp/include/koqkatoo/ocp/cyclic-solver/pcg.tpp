#pragma once
#include <koqkatoo/ocp/cyclic-solver/cyclic-solver.hpp>

#include <koqkatoo/linalg-compact/compact/micro-kernels/xgemm.hpp> // TODO

namespace koqkatoo::ocp {

template <class Abi>
auto CyclicOCPSolver<Abi>::mul_A(real_view_single p, mut_real_view_single Ap,
                                 real_view_single L, real_view_single B) const
    -> real_t {
    compact_blas::xcopy(p, Ap);
    compact_blas::xtrmv_T(L, Ap, be);
    compact_blas::xtrmv(L, Ap, be);
    using linalg::compact::micro_kernels::gemm::xsyomv_register; // TODO
    xsyomv_register<simd_abi, false>(B, p, Ap);
    return compact_blas::xdot(p, Ap);
}

template <class Abi>
auto CyclicOCPSolver<Abi>::mul_precond(real_view_single r,
                                       mut_real_view_single z,
                                       mut_real_view_single w,
                                       real_view_single L,
                                       real_view_single B) const -> real_t {
    compact_blas::xcopy(r, z);
#if USE_JACOBI_PREC
    std::ignore = w;
    std::ignore = B;
#else
    compact_blas::xcopy(r, w);
    compact_blas::xtrsv_LNN(L, w, be);
    compact_blas::xtrsv_LTN(L, w, be);
    using linalg::compact::micro_kernels::gemm::xsyomv_register; // TODO
    xsyomv_register<simd_abi, true>(B, w.as_const(), z);
#endif
    compact_blas::xtrsv_LNN(L, z, be);
    compact_blas::xtrsv_LTN(L, z, be);
    return compact_blas::xdot(r, z);
}

} // namespace koqkatoo::ocp