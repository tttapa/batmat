#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/assume.hpp>
#include <print>

namespace koqkatoo::ocp::cyclocp {

template <index_t VL>
auto CyclicOCPSolver<VL>::mul_A(matrix_view_batch p, mut_matrix_view_batch Ap,
                                matrix_view_batch L, matrix_view_batch B) const
    -> real_t {
    compact_blas::xcopy(p, Ap);
    compact_blas::xtrmv_T(L, Ap, backend);
    compact_blas::xtrmv(L, Ap, backend);
    compact_blas::xsyomv(B, p, Ap);
    return compact_blas::xdot(p, Ap);
}

template <index_t VL>
auto CyclicOCPSolver<VL>::mul_precond(matrix_view_batch r,
                                      mut_matrix_view_batch z,
                                      mut_matrix_view_batch w,
                                      matrix_view_batch L,
                                      matrix_view_batch B) const -> real_t {
    compact_blas::xcopy(r, z);
    if (use_stair_preconditioner) {
        compact_blas::xcopy(r, w);
        compact_blas::xtrsv_LNN(L, w, backend);
        compact_blas::xtrsv_LTN(L, w, backend);
        compact_blas::xsyomv_neg(B, w.as_const(), z);
    }
    compact_blas::xtrsv_LNN(L, z, backend);
    compact_blas::xtrsv_LTN(L, z, backend);
    return compact_blas::xdot(r, z);
}

template <index_t VL>
void CyclicOCPSolver<VL>::solve_pcg(mut_matrix_view_batch λ,
                                    mut_matrix_view_batch work_pcg) const {
    auto r = work_pcg.middle_cols(0, 1), z = work_pcg.middle_cols(1, 1),
         p = work_pcg.middle_cols(2, 1), Ap = work_pcg.middle_cols(3, 1);
    auto A = coupling_D.batch(0), B = coupling_Y.batch(0);
    real_t rᵀz = [&] {
        GUANAQO_TRACE("solve Ψ pcg", 0);
        compact_blas::xcopy(λ, r);
        compact_blas::xfill(0, λ);
        real_t rᵀz = mul_precond(r, z, Ap, A, B);
        compact_blas::xcopy(z, p);
        return rᵀz;
    }();
    const auto ε2 = pcg_tolerance * pcg_tolerance;
    for (index_t it = 0; it < pcg_max_iter; ++it) { // TODO
        GUANAQO_TRACE("solve Ψ pcg", it + 1);
        real_t pᵀAp = mul_A(p, Ap, A, B);
        real_t α    = rᵀz / pᵀAp;
        compact_blas::xaxpy(+α, p, λ);
        compact_blas::xaxpy(-α, Ap, r);
        real_t r2 = compact_blas::xdot(r, r);
        if (pcg_print_resid)
            std::println("{:>4}) pcg resid = {:15.6e}", it, std::sqrt(r2));
        if (r2 < ε2)
            break;
        real_t rᵀz_new = mul_precond(r, z, Ap, A, B);
        real_t β       = rᵀz_new / rᵀz;
        compact_blas::xaxpby(1, z, β, p);
        rᵀz = rᵀz_new;
    }
}

} // namespace koqkatoo::ocp::cyclocp
