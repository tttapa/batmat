#pragma once

#include <koqkatoo/ocp/cyclic-solver/cyclic-solver.hpp>

namespace koqkatoo::ocp {

template <class Abi>
void CyclicOCPSolver<Abi>::initialize(const LinearOCPStorage &ocp) {
    auto [N, nx, nu, ny, ny_N] = dim;
    for (index_t i = 0; i < n; ++i) {
        auto bi = get_batch_index(i);
        for (index_t vi = 0; vi < vl; ++vi) {
            auto k = i + vi * vstride;
            if (k < dim.N_horiz) {
                H.batch(bi)(vi)  = ocp.H(k);
                AB.batch(bi)(vi) = ocp.AB(k);
                CD.batch(bi)(vi) = ocp.CD(k);
            } else if (k == dim.N_horiz) {
                H.batch(bi)(vi).bottom_left(nu, nx).set_constant(0);
                H.batch(bi)(vi).right_cols(nu).set_constant(0);
                H.batch(bi)(vi).bottom_right(nu, nu).set_diagonal(1);
                H.batch(bi)(vi).top_left(nx, nx) = ocp.Q(k);
                AB.batch(bi)(vi).set_constant(0);
                CD.batch(bi)(vi).right_cols(nu).set_constant(0);
                CD.batch(bi)(vi).bottom_left(ny - ny_N, nx).set_constant(0);
                CD.batch(bi)(vi).top_left(ny_N, nx) = ocp.C(k).top_rows(ny_N);
            } else {
                H.batch(bi)(vi).set_constant(0);
                H.batch(bi)(vi).set_diagonal(1);
                AB.batch(bi)(vi).set_constant(0);
                CD.batch(bi)(vi).set_constant(0);
            }
        }
    }
}

template <class Abi>
template <class Tin, class Tout>
void CyclicOCPSolver<Abi>::pack_vectors(
    std::span<Tin> in, typename compact_blas::template batch_view_t<Tout> out,
    index_t rows, index_t rows_N) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    using view = typename compact_blas::template batch_view_scalar_t<Tin>;
    view vw{{.data = in.data(), .depth = N + 1, .rows = rows, .cols = 1}};
    assert(in.size() == static_cast<size_t>(N * rows + rows_N));
    for (index_t i = 0; i < n; ++i) {
        auto bi = get_batch_index(i);
        for (index_t vi = 0; vi < vl; ++vi) {
            auto k = i + vi * vstride;
            if (k < N) {
                out.batch(bi)(vi) = vw(k);
            } else if (k == N) {
                out.batch(bi)(vi).top_rows(rows_N) = vw(k).top_rows(rows_N);
                out.batch(bi)(vi).bottom_rows(rows - rows_N).set_constant(0);
            } else {
                out.batch(bi)(vi).set_constant(0);
            }
        }
    }
}

template <class Abi>
template <class Tin, class Tout>
void CyclicOCPSolver<Abi>::unpack_vectors(
    typename compact_blas::template batch_view_t<Tin> in, std::span<Tout> out,
    index_t rows, index_t rows_N) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    using view = typename compact_blas::template batch_view_scalar_t<Tout>;
    view vw{{.data = out.data(), .depth = N + 1, .rows = rows, .cols = 1}};
    assert(out.size() == static_cast<size_t>(N * rows + rows_N));
    for (index_t i = 0; i < n; ++i) {
        auto bi = get_batch_index(i);
        for (index_t vi = 0; vi < vl; ++vi) {
            auto k = i + vi * vstride;
            if (k < N) {
                vw(k) = in.batch(bi)(vi);
            } else if (k == N) {
                vw(k).top_rows(rows_N) = in.batch(bi)(vi).top_rows(rows_N);
            }
        }
    }
}

template <class Abi>
template <class Tin, class Tout>
void CyclicOCPSolver<Abi>::pack_dyn(
    std::span<Tin> in,
    typename compact_blas::template batch_view_t<Tout> out) const {
    pack_vectors(in, out, dim.nx, dim.nx);
}

template <class Abi>
auto CyclicOCPSolver<Abi>::pack_dyn(std::span<const real_t> in) const
    -> real_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    real_matrix out{{.depth = N + 1, .rows = nx, .cols = 1}};
    if (!in.empty())
        this->pack_dyn<const real_t, real_t>(in, out);
    return out;
}

template <class Abi>
template <class Tin, class Tout>
void CyclicOCPSolver<Abi>::unpack_dyn(
    typename compact_blas::template batch_view_t<Tin> in,
    std::span<Tout> out) const {
    unpack_vectors(in, out, dim.nx, dim.nx);
}

template <class Abi>
template <class Tin, class Tout>
void CyclicOCPSolver<Abi>::pack_constr(
    std::span<Tin> in,
    typename compact_blas::template batch_view_t<Tout> out) const {
    pack_vectors(in, out, dim.ny, dim.ny_N);
}

template <class Abi>
auto CyclicOCPSolver<Abi>::pack_constr(std::span<const real_t> in) const
    -> real_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    real_matrix out{{.depth = N + 1, .rows = ny, .cols = 1}};
    if (!in.empty())
        this->pack_constr<const real_t, real_t>(in, out);
    return out;
}

template <class Abi>
auto CyclicOCPSolver<Abi>::pack_constr(std::span<const bool> in) const
    -> bool_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    bool_matrix out{{.depth = N + 1, .rows = ny, .cols = 1}};
    if (!in.empty())
        this->pack_constr<const bool, bool>(in, out);
    return out;
}

template <class Abi>
template <class Tin, class Tout>
void CyclicOCPSolver<Abi>::unpack_constr(
    typename compact_blas::template batch_view_t<Tin> in,
    std::span<Tout> out) const {
    unpack_vectors(in, out, dim.ny, dim.ny_N);
}

template <class Abi>
template <class Tin, class Tout>
void CyclicOCPSolver<Abi>::pack_var(
    std::span<Tin> in,
    typename compact_blas::template batch_view_t<Tout> out) const {
    pack_vectors(in, out, dim.nx + dim.nu, dim.nx);
}

template <class Abi>
auto CyclicOCPSolver<Abi>::pack_var(std::span<const real_t> in) const
    -> real_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    real_matrix out{{.depth = N + 1, .rows = nx + nu, .cols = 1}};
    if (!in.empty())
        this->pack_var<const real_t, real_t>(in, out);
    return out;
}

template <class Abi>
template <class Tin, class Tout>
void CyclicOCPSolver<Abi>::unpack_var(
    typename compact_blas::template batch_view_t<Tin> in,
    std::span<Tout> out) const {
    unpack_vectors(in, out, dim.nx + dim.nu, dim.nx);
}

} // namespace koqkatoo::ocp
