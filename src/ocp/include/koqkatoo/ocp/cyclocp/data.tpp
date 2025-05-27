#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/assume.hpp>

namespace koqkatoo::ocp::cyclocp {

template <class T1, class I1, class S1, class T2, class I2, class S2>
void copy_T(guanaqo::MatrixView<T1, I1, S1> src,
            guanaqo::MatrixView<T2, I2, S2> dst) {
    assert(src.rows == dst.cols);
    assert(src.cols == dst.rows);
    for (index_t r = 0; r < src.rows; ++r) // TODO: optimize
        for (index_t c = 0; c < src.cols; ++c)
            dst(c, r) = src(r, c);
}

template <index_t VL>
void CyclicOCPSolver<VL>::initialize(const LinearOCPStorage &ocp) {
    KOQKATOO_ASSERT(ocp.dim == dim);
    auto [N, nx, nu, ny, ny_N] = dim;
    const auto vstride         = N >> lvl;
    const index_t num_stages   = N >> lP; // number of stages per thread
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k = sub_wrap_N(k0 + vi * vstride, i);
                copy_T(ocp.D(k), data_DCᵀ.batch(di)(vi).top_left(nu, dim.ny));
                if (k < dim.N_horiz) {
                    data_BA.batch(di)(vi).left_cols(nu) = ocp.B(k);
                    if (k == 0) {
                        auto data_DCᵀi = data_DCᵀ.batch(di)(vi);
                        copy_T(ocp.C(N), data_DCᵀi.bottom_right(nx, dim.ny_N));
                        data_DCᵀi // TODO: check user input
                            .top_right(nu, dim.ny_N)
                            .set_constant(0);
                        data_DCᵀi.bottom_left(nx, data_DCᵀ.cols() - dim.ny_N)
                            .set_constant(0);
                    } else {
                        auto data_DCᵀi = data_DCᵀ.batch(di)(vi);
                        copy_T(ocp.C(k), data_DCᵀi.bottom_left(nx, dim.ny));
                        data_DCᵀi.right_cols(data_DCᵀ.cols() - dim.ny)
                            .set_constant(0);
                    }
                    data_RSQ.batch(di)(vi).top_left(nu, nu) = ocp.R(k);
                    if (k == 0) {
                        data_BA.batch(di)(vi)
                            .right_cols(nx) // A
                            .set_constant(0);
                        data_RSQ.batch(di)(vi)
                            .bottom_left(nx, nu) // S
                            .set_constant(0);
                        data_RSQ.batch(di)(vi).bottom_right(nx, nx) = ocp.Q(N);
                    } else {
                        data_BA.batch(di)(vi).right_cols(nx) = ocp.A(k);
                        data_RSQ.batch(di)(vi).bottom_left(nx, nu) =
                            ocp.S_trans(k); // TODO
                        data_RSQ.batch(di)(vi).bottom_right(nx, nx) = ocp.Q(k);
                    }
                }
            }
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::initialize_Σ(std::span<const real_t> Σ_lin,
                                       mut_matrix_view Σ) const {
    auto [N, nx, nu, ny, ny_N] = dim;

    KOQKATOO_ASSERT(static_cast<index_t>(Σ_lin.size()) == N * ny + ny_N);
    KOQKATOO_ASSERT(Σ.depth() == N);
    KOQKATOO_ASSERT(Σ.rows() == std::max(ny, ny_N));
    KOQKATOO_ASSERT(Σ.cols() == 1);

    const auto vstride       = N >> lvl;
    const index_t num_stages = N >> lP; // number of stages per thread
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k       = sub_wrap_N(k0 + vi * vstride, i);
                using crview = guanaqo::MatrixView<const real_t, index_t>;
                if (k < N) {
                    Σ.batch(di)(vi).top_rows(dim.ny) =
                        crview::as_column(Σ_lin.subspan(k * ny, ny));
                    Σ.batch(di)(vi).bottom_rows(Σ.rows() - ny).set_constant(0);
                    if (k == 0)
                        Σ.batch(di)(vi).bottom_rows(ny_N) =
                            crview::as_column(Σ_lin.subspan(N * ny, ny_N));
                }
            }
        }
    }
}

} // namespace koqkatoo::ocp::cyclocp
