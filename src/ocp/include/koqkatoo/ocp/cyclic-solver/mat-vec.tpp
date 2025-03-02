#pragma once

#include <koqkatoo/ocp/cyclic-solver/cyclic-solver.hpp>
#include <print>

namespace koqkatoo::ocp {

template <class Abi>
void CyclicOCPSolver<Abi>::mat_vec_constr_tp(real_view ŷb,
                                             mut_real_view Aᵀŷb) const {
    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("mat_vec_transpose_constr", i);
        auto hi = get_batch_index(i);
        compact_blas::xgemv_T(CD.batch(hi), ŷb.batch(hi), Aᵀŷb.batch(hi), be);
    }
}

template <class Abi>
void CyclicOCPSolver<Abi>::mat_vec_dyn(real_view xb, real_view bb,
                                       mut_real_view Mxbb) const {
    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("residual_dynamics_constr", i);
        auto hi = get_batch_index(i);
        if (i + 1 < n) {
            auto hi_next = get_batch_index(i + 1);
            compact_blas::xsub_copy(Mxbb.batch(hi_next),
                                    xb.batch(hi_next).top_rows(dim.nx),
                                    bb.batch(hi_next));
            compact_blas::xgemv_sub(AB.batch(hi), xb.batch(hi),
                                    Mxbb.batch(hi_next), be);
        } else {
            // Last batch is special since we cross batch boundaries
            auto hi_next = get_batch_index(i - vstride + 1);
            compact_blas::xsub_copy(Mxbb.batch(hi_next),
                                    xb.batch(hi_next).top_rows(dim.nx),
                                    bb.batch(hi_next));
            compact_blas::xgemv_sub_shift(AB.batch(hi), xb.batch(hi),
                                          Mxbb.batch(hi_next));
        }
    }
}

template <class Abi>
void CyclicOCPSolver<Abi>::mat_vec_dyn_tp(real_view λb,
                                          mut_real_view Mᵀλb) const {
    KOQKATOO_OMP(for)
    for (index_t i = 0; i < n; ++i) {
        KOQKATOO_TRACE("mat_vec_transpose_dynamics_constr", i);
        const auto hi                   = get_batch_index(i);
        Mᵀλb.batch(hi).top_rows(dim.nx) = λb.batch(hi);
        Mᵀλb.batch(hi).bottom_rows(dim.nu).set_constant(0);
        if (i + 1 < n) {
            const auto hi_next = get_batch_index(i + 1);
            compact_blas::xgemv_T_sub(AB.batch(hi), λb.batch(hi_next),
                                      Mᵀλb.batch(hi), be);
        } else {
            const auto hi_next = get_batch_index(i - vstride + 1);
            compact_blas::xgemv_T_sub_shift(AB.batch(hi), λb.batch(hi_next),
                                            Mᵀλb.batch(hi));
        }
    }
}

} // namespace koqkatoo::ocp
