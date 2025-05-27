#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/assume.hpp>
#include <guanaqo/blas/hl-blas-interface.hpp>
#include <stdexcept>

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
    const auto all_zero        = [](auto X) {
        for (typename decltype(X)::index_type c = 0; c < X.cols; ++c)
            for (typename decltype(X)::index_type r = 0; r < X.rows; ++r)
                if (X(r, c) != 0)
                    return false;
        return true;
    };
    const auto num_nonzero_rows = [](auto X) {
        typename decltype(X)::index_type nzr = 0;
        for (typename decltype(X)::index_type c = 0; c < X.cols; ++c)
            for (typename decltype(X)::index_type r = 0; r < X.rows; ++r)
                if (X(r, c) != 0)
                    nzr = std::max(nzr, r + 1);
        return nzr;
    };
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k         = sub_wrap_N(k0 + vi * vstride, i);
                auto data_DCᵀi = data_DCᵀ.batch(di)(vi);
                copy_T(ocp.D(k), data_DCᵀi.top_left(nu, dim.ny));
                if (k < dim.N_horiz) {
                    data_BA.batch(di)(vi).left_cols(nu) = ocp.B(k);
                    if (k == 0) {
                        const index_t ny0 = num_nonzero_rows(ocp.D(0));
                        if (!all_zero(ocp.C(0)))
                            throw std::invalid_argument(
                                "State constraints C(0) should be zero.");
                        if (!all_zero(ocp.C(N).top_rows(ny0)))
                            throw std::invalid_argument(
                                "Top of state constraints C(N) should be "
                                "zero. You can work around this limitation by "
                                "moving all input-only constraints to the top, "
                                "increasing ny_N and adding padding rows to "
                                "C(N).");
                        copy_T(ocp.C(N), data_DCᵀi.bottom_right(nx, dim.ny_N));
                        data_DCᵀi.top_right(nu, dim.ny_N).set_constant(0);
                        data_DCᵀi.bottom_left(nx, ny0).set_constant(0);
                    } else {
                        const index_t ny_pad = std::max(ny, ny_N) - dim.ny;
                        copy_T(ocp.C(k), data_DCᵀi.bottom_left(nx, dim.ny));
                        data_DCᵀi.right_cols(ny_pad).set_constant(0);
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
void CyclicOCPSolver<VL>::initialize_rhs(const LinearOCPStorage &ocp,
                                         std::span<const real_t> x0,
                                         std::span<const real_t> c,
                                         mut_matrix_view rhs) const {
    KOQKATOO_ASSERT(ocp.dim == dim);
    initialize_rhs(ocp.A(0), x0, c, rhs);
}

template <index_t VL>
void CyclicOCPSolver<VL>::initialize_rhs(
    guanaqo::MatrixView<const real_t, index_t> A0, std::span<const real_t> x0,
    std::span<const real_t> c, mut_matrix_view rhs) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    KOQKATOO_ASSERT(A0.rows == nx);
    KOQKATOO_ASSERT(A0.cols == nx);
    KOQKATOO_ASSERT(static_cast<index_t>(x0.size()) == nx);
    KOQKATOO_ASSERT(static_cast<index_t>(c.size()) == nx * N);
    KOQKATOO_ASSERT(rhs.depth() == N);
    KOQKATOO_ASSERT(rhs.rows() == nx);
    KOQKATOO_ASSERT(rhs.cols() == 1);
    const auto vstride       = N >> lvl;
    const index_t num_stages = N >> lP; // number of stages per thread
    std::vector<real_t> work(nx);
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k       = sub_wrap_N(k0 + vi * vstride, i);
                using rview  = guanaqo::MatrixView<real_t, index_t>;
                using crview = guanaqo::MatrixView<const real_t, index_t>;
                if (k == 0) {
                    auto c0 = c.first(nx);
                    std::ranges::copy(c0, work.begin());
                    auto A0x0c0 = rview::as_column(work); // A₀ x₀ + c₀
                    guanaqo::blas::xgemv_N(real_t{1}, A0, crview::as_column(x0),
                                           real_t{1}, A0x0c0);
                    rhs.batch(di)(vi) = A0x0c0;
                } else if (k < dim.N_horiz) {
                    rhs.batch(di)(vi) =
                        crview::as_column(c.subspan(k * nx, nx));
                }
            }
            compact_blas::xneg(rhs.batch(di)); // TODO
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::pack_variables(std::span<const real_t> ux_lin,
                                         mut_matrix_view ux) const {
    const auto [N, nx, nu, ny, ny_N] = dim;
    const index_t nux                = nu + nx;
    KOQKATOO_ASSERT(static_cast<index_t>(ux_lin.size()) == nux * N);
    KOQKATOO_ASSERT(ux.depth() == N);
    KOQKATOO_ASSERT(ux.rows() == nux);
    KOQKATOO_ASSERT(ux.cols() == 1);
    const auto vstride       = N >> lvl;
    const index_t num_stages = N >> lP; // number of stages per thread
    std::vector<real_t> work(nx);
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k       = sub_wrap_N(k0 + vi * vstride, i);
                using crview = guanaqo::MatrixView<const real_t, index_t>;
                if (k < dim.N_horiz) {
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
    const auto [N, nx, nu, ny, ny_N] = dim;
    const index_t nux                = nu + nx;
    KOQKATOO_ASSERT(static_cast<index_t>(ux_lin.size()) == nux * N);
    KOQKATOO_ASSERT(ux.depth() == N);
    KOQKATOO_ASSERT(ux.rows() == nux);
    KOQKATOO_ASSERT(ux.cols() == 1);
    const auto vstride       = N >> lvl;
    const index_t num_stages = N >> lP; // number of stages per thread
    std::vector<real_t> work(nx);
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k      = sub_wrap_N(k0 + vi * vstride, i);
                using rview = guanaqo::MatrixView<real_t, index_t>;
                if (k < dim.N_horiz) {
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
    const auto [N, nx, nu, ny, ny_N] = dim;
    const index_t nλ                 = nx;
    KOQKATOO_ASSERT(static_cast<index_t>(λ_lin.size()) == nλ * N);
    KOQKATOO_ASSERT(λ.depth() == N);
    KOQKATOO_ASSERT(λ.rows() == nλ);
    KOQKATOO_ASSERT(λ.cols() == 1);
    const auto vstride       = N >> lvl;
    const index_t num_stages = N >> lP; // number of stages per thread
    std::vector<real_t> work(nx);
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k       = sub_wrap_N(k0 + vi * vstride, i);
                using crview = guanaqo::MatrixView<const real_t, index_t>;
                if (k < dim.N_horiz) {
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
    const auto [N, nx, nu, ny, ny_N] = dim;
    const index_t nλ                 = nu + nx;
    KOQKATOO_ASSERT(static_cast<index_t>(λ_lin.size()) == nλ * N);
    KOQKATOO_ASSERT(λ.depth() == N);
    KOQKATOO_ASSERT(λ.rows() == nλ);
    KOQKATOO_ASSERT(λ.cols() == 1);
    const auto vstride       = N >> lvl;
    const index_t num_stages = N >> lP; // number of stages per thread
    std::vector<real_t> work(nx);
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k      = sub_wrap_N(k0 + vi * vstride, i);
                using rview = guanaqo::MatrixView<real_t, index_t>;
                if (k < dim.N_horiz) {
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
    const auto [N, nx, nu, ny, ny_N] = dim;
    KOQKATOO_ASSERT(static_cast<index_t>(y_lin.size()) == ny * N + ny_N);
    KOQKATOO_ASSERT(y.depth() == N);
    KOQKATOO_ASSERT(y.rows() == std::max(ny, ny_N));
    KOQKATOO_ASSERT(y.cols() == 1);
    const auto vstride       = N >> lvl;
    const index_t num_stages = N >> lP; // number of stages per thread
    std::vector<real_t> work(nx);
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k       = sub_wrap_N(k0 + vi * vstride, i);
                using crview = guanaqo::MatrixView<const real_t, index_t>;
                if (k == 0) {
                    const index_t ny0 = std::max(ny, ny_N) - ny_N;
                    y.batch(di)(vi).top_rows(ny0) =
                        crview::as_column(y_lin.first(ny0));
                    y.batch(di)(vi).bottom_rows(ny_N) =
                        crview::as_column(y_lin.subspan(N * ny, ny_N));
                } else if (k < dim.N_horiz) {
                    const index_t ny_pad = std::max(ny, ny_N) - ny;
                    y.batch(di)(vi).top_rows(ny) =
                        crview::as_column(y_lin.subspan(k * ny, ny));
                    y.batch(di)(vi).bottom_rows(ny_pad).set_constant(fill);
                }
            }
        }
    }
}

template <index_t VL>
void CyclicOCPSolver<VL>::unpack_constraints(matrix_view y,
                                             std::span<real_t> y_lin) const {
    const auto [N, nx, nu, ny, ny_N] = dim;
    KOQKATOO_ASSERT(static_cast<index_t>(y_lin.size()) == ny * N + ny_N);
    KOQKATOO_ASSERT(y.depth() == N);
    KOQKATOO_ASSERT(y.rows() == std::max(ny, ny_N));
    KOQKATOO_ASSERT(y.cols() == 1);
    const auto vstride       = N >> lvl;
    const index_t num_stages = N >> lP; // number of stages per thread
    std::vector<real_t> work(nx);
    for (index_t ti = 0; ti < (1 << (lP - lvl)); ++ti) {
        const index_t k0  = ti * num_stages;
        const index_t di0 = ti * num_stages;
        for (index_t i = 0; i < num_stages; ++i) {
            index_t di = di0 + i;
            for (index_t vi = 0; vi < vl; ++vi) {
                auto k      = sub_wrap_N(k0 + vi * vstride, i);
                using rview = guanaqo::MatrixView<real_t, index_t>;
                if (k == 0) {
                    const index_t ny0 = std::max(ny, ny_N) - ny_N;
                    rview::as_column(y_lin.first(ny0)) =
                        y.batch(di)(vi).top_rows(ny0);
                    rview::as_column(y_lin.subspan(N * ny, ny_N)) =
                        y.batch(di)(vi).bottom_rows(ny_N);
                } else if (k < dim.N_horiz) {
                    rview::as_column(y_lin.subspan(k * ny, ny)) =
                        y.batch(di)(vi).top_rows(ny);
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
