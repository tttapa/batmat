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

} // namespace koqkatoo::ocp::cyclocp
