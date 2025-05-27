#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/assume.hpp>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <stdexcept>
#include <vector>

namespace koqkatoo::ocp::cyclocp {

CyclicOCPStorage CyclicOCPStorage::build(const LinearOCPStorage &ocp,
                                         std::span<const real_t> qr,
                                         std::span<const real_t> b_eq,
                                         std::span<const real_t> b_lb,
                                         std::span<const real_t> b_ub) {
    using crview = guanaqo::MatrixView<const real_t, index_t>;
    const auto [N, nx, nu, ny, ny_N] = ocp.dim;
    // Count the number of input constraints in the first stage
    std::vector<bool> Ju0(ny);
    for (index_t c = 0; c < nu; ++c)
        for (index_t r = 0; r < ny; ++r)
            if (ocp.D(0)(r, c) != 0)
                Ju0[r] = true;
    const index_t ny_0 = std::ranges::count(Ju0, index_t{0});
    CyclicOCPStorage res{
        .N_horiz = N, .nx = nx, .nu = nu, .ny = ny, .ny_0 = ny_0, .ny_N = ny_N};
    // H₀ = [ R₀ 0 ]
    //      [ 0  Qₙ]
    res.data_H(0).top_left(nu, nu)     = ocp.R(0);
    res.data_H(0).bottom_right(nx, nx) = ocp.Q(N);
    res.data_H(0).bottom_left(nx, nu).set_constant(0);
    res.data_H(0).top_right(nu, nx).set_constant(0);
    // F₀ = [ B₀ 0 ]
    res.data_F(0).left_cols(nu) = ocp.B(0);
    res.data_F(0).right_cols(nx).set_constant(0);
    // G₀ = [ D₀ 0 ]
    //      [ 0  Cₙ]
    res.data_G(0).bottom_left(ny_N, nu).set_constant(0);
    res.data_G(0).top_right(ny_0, nx).set_constant(0);
    for (index_t r = 0, j = 0; r < ny; ++r) {
        if (Ju0[r]) {
            res.data_G0N(0).block(j, 0, 1, nu) = ocp.D(0).middle_rows(r, 1);
            real_t t                           = 0;
            for (index_t c = 0; c < nx; ++c) // lb - C₀ x₀
                t += ocp.C(0)(r, c) * b_eq[c];
            res.data_lb0N(0, j, 0) = b_lb[r] - t;
            res.data_ub0N(0, j, 0) = b_ub[r] - t;
            ++j;
        }
    }
    res.data_G0N(0).bottom_right(ny_N, nx) = ocp.C(N);
    res.data_lb0N(0).bottom_rows(ny_N) =
        crview::as_column(b_lb.subspan(N * ny, ny_N));
    res.data_ub0N(0).bottom_rows(ny_N) =
        crview::as_column(b_ub.subspan(N * ny, ny_N));
    // c̃₀ = c₀ + A₀ x₀      (b_eq = [x₀, c₀, ... cₙ₋₁])
    res.data_c(0) = crview::as_column(b_eq.subspan(nx, nx));
    for (index_t r = 0; r < nx; ++r)
        for (index_t c = 0; c < nx; ++c)
            res.data_c(0, r, 0) += ocp.A(0)(r, c) * b_eq[c];
    // r̃₀ = r₀ + S₀ x₀
    res.data_rq(0).bottom_rows(nx) = crview::as_column(qr.first(nx));
    res.data_rq(0).top_rows(nu)    = crview::as_column(qr.subspan(nx, nu));
    for (index_t r = 0; r < nu; ++r)
        for (index_t c = 0; c < nx; ++c)
            res.data_rq(0, r, 0) += ocp.S_trans(0)(c, r) * b_eq[c];
    for (index_t i = 1; i < N; ++i) {
        res.data_H(i).top_left(nu, nu)     = ocp.R(i);
        res.data_H(i).bottom_left(nx, nu)  = ocp.S_trans(i);
        res.data_H(i).top_right(nu, nx)    = ocp.S(i);
        res.data_H(i).bottom_right(nx, nx) = ocp.Q(i);
        res.data_F(i).left_cols(nu)        = ocp.B(i);
        res.data_F(i).right_cols(nx)       = ocp.A(i);
        res.data_G(i - 1).left_cols(nu)    = ocp.D(i);
        res.data_G(i - 1).right_cols(nx)   = ocp.C(i);
        res.data_lb(i - 1) = crview::as_column(b_lb.subspan(i * ny, ny));
        res.data_ub(i - 1) = crview::as_column(b_ub.subspan(i * ny, ny));
        res.data_c(i)      = crview::as_column(b_eq.subspan((i + 1) * nx, nx));
    }
    return res;
}

template <index_t VL>
CyclicOCPSolver<VL> CyclicOCPSolver<VL>::build(const CyclicOCPStorage &ocp,
                                               index_t lP) {
    CyclicOCPSolver<VL> res{
        .N_horiz = ocp.N_horiz,
        .nx      = ocp.nx,
        .nu      = ocp.nu,
        .ny      = ocp.ny,
        .ny_0    = ocp.ny_0,
        .ny_N    = ocp.ny_N,
        .lP      = lP,
    };
    const auto vstride       = res.N_horiz >> lvl;
    const index_t num_stages = res.N_horiz >> lP; // number of stages per thread
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k = res.sub_wrap_N(k0 + vi * vstride, i);
                if (k < res.N_horiz) {
                    res.data_BA.batch(di)(vi)  = ocp.data_F(k);
                    res.data_RSQ.batch(di)(vi) = ocp.data_H(k);
                    if (k == 0) {
                        detail::copy_T(ocp.data_G0N(0),
                                       res.data_DCᵀ.batch(di)(vi).left_cols(
                                           res.ny_0 + res.ny_N));
                    } else {
                        detail::copy_T(
                            ocp.data_G(k - 1),
                            res.data_DCᵀ.batch(di)(vi).left_cols(res.ny));
                    }
                }
            }
        }
    }
    return res;
}

template <index_t VL>
void CyclicOCPSolver<VL>::initialize_rhs(const CyclicOCPStorage &ocp,
                                         mut_matrix_view rhs) const {
    KOQKATOO_ASSERT(rhs.depth() == N_horiz);
    KOQKATOO_ASSERT(rhs.rows() == nx);
    KOQKATOO_ASSERT(rhs.cols() == 1);
    const auto vstride       = N_horiz >> lvl;
    const index_t num_stages = N_horiz >> lP; // number of stages per thread
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k = sub_wrap_N(k0 + vi * vstride, i);
                if (k < N_horiz) {
                    rhs.batch(di)(vi) = ocp.data_c(k);
                }
            }
            compact_blas::xneg(rhs.batch(di)); // TODO
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::pack_variables(std::span<const real_t> ux_lin,
                                         mut_matrix_view ux) const {
    const index_t nux = nu + nx;
    KOQKATOO_ASSERT(static_cast<index_t>(ux_lin.size()) == nux * N_horiz);
    KOQKATOO_ASSERT(ux.depth() == N_horiz);
    KOQKATOO_ASSERT(ux.rows() == nux);
    KOQKATOO_ASSERT(ux.cols() == 1);
    const auto vstride       = N_horiz >> lvl;
    const index_t num_stages = N_horiz >> lP; // number of stages per thread
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k       = sub_wrap_N(k0 + vi * vstride, i);
                using crview = guanaqo::MatrixView<const real_t, index_t>;
                if (k < N_horiz) {
                    ux.batch(di)(vi) =
                        crview::as_column(ux_lin.subspan(k * nux, nux));
                }
            }
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::unpack_variables(matrix_view ux,
                                           std::span<real_t> ux_lin) const {
    const index_t nux = nu + nx;
    KOQKATOO_ASSERT(static_cast<index_t>(ux_lin.size()) == nux * N_horiz);
    KOQKATOO_ASSERT(ux.depth() == N_horiz);
    KOQKATOO_ASSERT(ux.rows() == nux);
    KOQKATOO_ASSERT(ux.cols() == 1);
    const auto vstride       = N_horiz >> lvl;
    const index_t num_stages = N_horiz >> lP; // number of stages per thread
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k      = sub_wrap_N(k0 + vi * vstride, i);
                using rview = guanaqo::MatrixView<real_t, index_t>;
                if (k < N_horiz) {
                    rview::as_column(ux_lin.subspan(k * nux, nux)) =
                        ux.batch(di)(vi);
                }
            }
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::pack_dynamics(std::span<const real_t> λ_lin,
                                        mut_matrix_view λ) const {
    const index_t nλ = nx;
    KOQKATOO_ASSERT(static_cast<index_t>(λ_lin.size()) == nλ * N_horiz);
    KOQKATOO_ASSERT(λ.depth() == N_horiz);
    KOQKATOO_ASSERT(λ.rows() == nλ);
    KOQKATOO_ASSERT(λ.cols() == 1);
    const auto vstride       = N_horiz >> lvl;
    const index_t num_stages = N_horiz >> lP; // number of stages per thread
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k       = sub_wrap_N(k0 + vi * vstride, i);
                using crview = guanaqo::MatrixView<const real_t, index_t>;
                if (k < N_horiz) {
                    λ.batch(di)(vi) =
                        crview::as_column(λ_lin.subspan(k * nλ, nλ));
                }
            }
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::unpack_dynamics(matrix_view λ,
                                          std::span<real_t> λ_lin) const {
    const index_t nλ = nu + nx;
    KOQKATOO_ASSERT(static_cast<index_t>(λ_lin.size()) == nλ * N_horiz);
    KOQKATOO_ASSERT(λ.depth() == N_horiz);
    KOQKATOO_ASSERT(λ.rows() == nλ);
    KOQKATOO_ASSERT(λ.cols() == 1);
    const auto vstride       = N_horiz >> lvl;
    const index_t num_stages = N_horiz >> lP; // number of stages per thread
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k      = sub_wrap_N(k0 + vi * vstride, i);
                using rview = guanaqo::MatrixView<real_t, index_t>;
                if (k < N_horiz) {
                    rview::as_column(λ_lin.subspan(k * nλ, nλ)) =
                        λ.batch(di)(vi);
                }
            }
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::pack_constraints(std::span<const real_t> y_lin,
                                           mut_matrix_view y,
                                           real_t fill) const {
    KOQKATOO_ASSERT(static_cast<index_t>(y_lin.size()) ==
                    ny * (N_horiz - 1) + ny_0 + ny_N);
    KOQKATOO_ASSERT(y.depth() == N_horiz);
    KOQKATOO_ASSERT(y.rows() == std::max(ny, ny_0 + ny_N));
    KOQKATOO_ASSERT(y.cols() == 1);
    const auto vstride       = N_horiz >> lvl;
    const index_t num_stages = N_horiz >> lP; // number of stages per thread
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k       = sub_wrap_N(k0 + vi * vstride, i);
                using crview = guanaqo::MatrixView<const real_t, index_t>;
                if (k == 0) {
                    index_t ny_pad = std::max(ny, ny_0 + ny_N) - (ny_0 + ny_N);
                    y.batch(di)(vi).top_rows(ny_0) =
                        crview::as_column(y_lin.first(ny_0));
                    y.batch(di)(vi).bottom_rows(ny_N) = crview::as_column(
                        y_lin.subspan(ny_0 + (N_horiz - 1) * ny, ny_N));
                    y.batch(di)(vi).bottom_rows(ny_pad).set_constant(fill);
                } else if (k < N_horiz) {
                    index_t ny_pad = std::max(ny, ny_0 + ny_N) - ny;
                    y.batch(di)(vi).top_rows(ny) = crview::as_column(
                        y_lin.subspan(ny_0 + (k - 1) * ny, ny));
                    y.batch(di)(vi).bottom_rows(ny_pad).set_constant(fill);
                }
            }
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::unpack_constraints(matrix_view y,
                                             std::span<real_t> y_lin) const {
    KOQKATOO_ASSERT(static_cast<index_t>(y_lin.size()) ==
                    ny * (N_horiz - 1) + ny_0 + ny_N);
    KOQKATOO_ASSERT(y.depth() == N_horiz);
    KOQKATOO_ASSERT(y.rows() == std::max(ny, ny_N));
    KOQKATOO_ASSERT(y.cols() == 1);
    const auto vstride       = N_horiz >> lvl;
    const index_t num_stages = N_horiz >> lP; // number of stages per thread
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k      = sub_wrap_N(k0 + vi * vstride, i);
                using rview = guanaqo::MatrixView<real_t, index_t>;
                if (k == 0) {
                    rview::as_column(y_lin.first(ny_0)) =
                        y.batch(di)(vi).top_rows(ny_0);
                    rview::as_column(
                        y_lin.subspan(ny_0 + (N_horiz - 1) * ny, ny_N)) =
                        y.batch(di)(vi).bottom_rows(ny_N);
                } else if (k < N_horiz) {
                    rview::as_column(y_lin.subspan(ny_0 + (k - 1) * ny, ny)) =
                        y.batch(di)(vi).top_rows(ny);
                }
            }
        }
    }
}

} // namespace koqkatoo::ocp::cyclocp
