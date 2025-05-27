#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/assume.hpp>
#include <koqkatoo/loop.hpp>
#include <stdexcept>

namespace koqkatoo::ocp::cyclocp {

template <index_t VL>
void CyclicOCPSolver<VL>::residual_dynamics_constr(matrix_view x, matrix_view b,
                                                   mut_matrix_view Mxb) const {
    koqkatoo::foreach_thread([&](index_t ti, index_t P) {
        if (P < (1 << (lP - lvl)))
            throw std::logic_error("Incorrect number of threads");
        if (ti >= (1 << (lP - lvl)))
            return;
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t k0         = ti * num_stages; // stage index
        {
            GUANAQO_TRACE("resid_dyn_constr init", k0);
            index_t ti_next = add_wrap_PmV(ti, 1);
            index_t di_next = ti_next * num_stages + num_stages - 1;
            auto x_next     = x.batch(di_next).bottom_rows(nx);
            auto Mxb0       = Mxb.batch(di0);
            auto b0         = b.batch(di0);

            ti_next == 0
                ? compact_blas::template xadd_neg_copy<-1>(Mxb0, x_next, b0)
                : compact_blas::template xadd_neg_copy<0>(Mxb0, x_next, b0);
        }
        for (index_t i = 0; i < num_stages; ++i) {
            [[maybe_unused]] index_t k = sub_wrap_N(k0, i);
            GUANAQO_TRACE("resid_dyn_constr", k);
            index_t di = di0 + i;
            auto BAi   = data_BA.batch(di);
            auto uxi   = x.batch(di);
            compact_blas::xgemv_add(BAi, uxi, Mxb.batch(di), backend);
            if (i + 1 < num_stages) {
                index_t di_next = di + 1;
                compact_blas::xadd_neg_copy(
                    Mxb.batch(di_next), b.batch(di_next), uxi.bottom_rows(nx));
            }
        }
    });
}

template <index_t VL>
void CyclicOCPSolver<VL>::transposed_dynamics_constr(
    matrix_view λ, mut_matrix_view Mᵀλ) const {
    koqkatoo::foreach_thread([&](index_t ti, index_t P) {
        if (P < (1 << (lP - lvl)))
            throw std::logic_error("Incorrect number of threads");
        if (ti >= (1 << (lP - lvl)))
            return;
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t k0         = ti * num_stages; // stage index
        for (index_t i = 0; i < num_stages; ++i) {
            [[maybe_unused]] index_t k = sub_wrap_N(k0, i);
            GUANAQO_TRACE("transposed_dynamics_constr", k);
            index_t di = di0 + i;
            auto BAi   = data_BA.batch(di);
            Mᵀλ.batch(di).top_rows(nu).set_constant(0);
            if (i + 1 < num_stages) {
                index_t di_next = di + 1;
                compact_blas::xadd_neg_copy(Mᵀλ.batch(di).bottom_rows(nx),
                                            λ.batch(di_next));
            } else {
                const index_t ti_next = sub_wrap_PmV(ti, 1);
                index_t di_next       = ti_next * num_stages;
                ti == 0 ? compact_blas::template xadd_neg_copy<1>(
                              Mᵀλ.batch(di).bottom_rows(nx), λ.batch(di_next))
                              : compact_blas::template xadd_neg_copy<0>(
                              Mᵀλ.batch(di).bottom_rows(nx), λ.batch(di_next));
            }
            compact_blas::xgemv_T_add(BAi, λ.batch(di), Mᵀλ.batch(di), backend);
        }
    });
}

template <index_t VL>
void CyclicOCPSolver<VL>::cost_gradient(matrix_view ux, real_t a, matrix_view q,
                                        real_t b, mut_matrix_view grad_f) const {
    koqkatoo::foreach_thread([&](index_t ti, index_t P) {
        if (P < (1 << (lP - lvl)))
            throw std::logic_error("Incorrect number of threads");
        if (ti >= (1 << (lP - lvl)))
            return;
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t k0         = ti * num_stages; // stage index
        for (index_t i = 0; i < num_stages; ++i) {
            [[maybe_unused]] index_t k = sub_wrap_N(k0, i);
            GUANAQO_TRACE("cost_gradient", k);
            index_t di = di0 + i;
            compact_blas::xaxpby(a, q.batch(di), b, grad_f.batch(di));
            compact_blas::xsymv_add(data_RSQ.batch(di), ux.batch(di),
                                    grad_f.batch(di), backend);
        }
    });
}

template <index_t VL>
void CyclicOCPSolver<VL>::general_constr(matrix_view ux,
                                         mut_matrix_view DCux) const {
    koqkatoo::foreach_thread([&](index_t ti, index_t P) {
        if (P < (1 << (lP - lvl)))
            throw std::logic_error("Incorrect number of threads");
        if (ti >= (1 << (lP - lvl)))
            return;
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t k0         = ti * num_stages; // stage index
        for (index_t i = 0; i < num_stages; ++i) {
            [[maybe_unused]] index_t k = sub_wrap_N(k0, i);
            GUANAQO_TRACE("general_constr", k);
            index_t di = di0 + i;
            compact_blas::xgemv_T(data_DCᵀ.batch(di), ux.batch(di),
                                  DCux.batch(di), backend);
        }
    });
}

template <index_t VL>
void CyclicOCPSolver<VL>::transposed_general_constr(
    matrix_view y, mut_matrix_view DCᵀy) const {
    koqkatoo::foreach_thread([&](index_t ti, index_t P) {
        if (P < (1 << (lP - lvl)))
            throw std::logic_error("Incorrect number of threads");
        if (ti >= (1 << (lP - lvl)))
            return;
        const auto [N, nx, nu, ny, ny_N] = dim;
        const index_t num_stages = N >> lP; // number of stages per thread
        const index_t di0        = ti * num_stages; // data batch index
        const index_t k0         = ti * num_stages; // stage index
        for (index_t i = 0; i < num_stages; ++i) {
            [[maybe_unused]] index_t k = sub_wrap_N(k0, i);
            GUANAQO_TRACE("transposed_general_constr", k);
            index_t di = di0 + i;
            compact_blas::xgemv(data_DCᵀ.batch(di), y.batch(di), DCᵀy.batch(di),
                                backend);
        }
    });
}

} // namespace koqkatoo::ocp::cyclocp
