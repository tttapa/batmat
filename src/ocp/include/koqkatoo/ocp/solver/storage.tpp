#pragma once

#include <koqkatoo/ocp/solver/storage.hpp>

namespace koqkatoo::ocp {

template <simd_abi_tag Abi>
void SolverStorage<Abi>::copy_active_set(std::span<const bool> in,
                                         view_type<bool> out) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    assert(in.size() == static_cast<size_t>(N * ny + ny_N));
    assert(out.rows() == ny);
    assert(out.cols() == 1);
    assert(out.depth() == N + 1);
    auto in_vw = guanaqo::MatrixView<const bool, index_t>::as_column(in);
    for (index_t i = 0; i < N; ++i)
        out(i) = in_vw.middle_rows(i * ny, ny);
    out(N).top_rows(ny_N) = in_vw.middle_rows(N * ny, ny_N);
    out(N).bottom_rows(ny - ny_N).set_constant(0);
    for (index_t i = N + 1; i < out.ceil_depth(); ++i)
        out(i).set_constant(0);
}

template <simd_abi_tag Abi>
void SolverStorage<Abi>::restore_active_set(view_type<const bool> in,
                                            std::span<bool> out) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    assert(out.size() == static_cast<size_t>(N * ny + ny_N));
    assert(in.rows() == ny);
    assert(in.cols() == 1);
    assert(in.depth() == N + 1);
    auto out_vw = guanaqo::MatrixView<bool, index_t>::as_column(out);
    for (index_t i = 0; i < N; ++i)
        out_vw.middle_rows(i * ny, ny) = in(i);
    out_vw.middle_rows(N * ny, ny_N) = in(N).top_rows(ny_N);
}

template <simd_abi_tag Abi>
void SolverStorage<Abi>::copy_constraints(std::span<const real_t> in,
                                          view_type<real_t> out) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    assert(in.size() == static_cast<size_t>(N * ny + ny_N));
    assert(out.rows() == ny);
    assert(out.cols() == 1);
    assert(out.depth() == N + 1);
    auto in_vw = guanaqo::MatrixView<const real_t, index_t>::as_column(in);
    for (index_t i = 0; i < N; ++i)
        out(i) = in_vw.middle_rows(i * ny, ny);
    out(N).top_rows(ny_N) = in_vw.middle_rows(N * ny, ny_N);
    out(N).bottom_rows(ny - ny_N).set_constant(0);
    for (index_t i = N + 1; i < out.ceil_depth(); ++i)
        out(i).set_constant(0);
}

template <simd_abi_tag Abi>
void SolverStorage<Abi>::restore_constraints(view_type<const real_t> in,
                                             std::span<real_t> out) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    assert(out.size() == static_cast<size_t>(N * ny + ny_N));
    assert(in.rows() == ny);
    assert(in.cols() == 1);
    assert(in.depth() == N + 1);
    auto out_vw = guanaqo::MatrixView<real_t, index_t>::as_column(out);
    for (index_t i = 0; i < N; ++i)
        out_vw.middle_rows(i * ny, ny) = in(i);
    out_vw.middle_rows(N * ny, ny_N) = in(N).top_rows(ny_N);
}

template <simd_abi_tag Abi>
void SolverStorage<Abi>::copy_dynamics_constraints(
    std::span<const real_t> in, view_type<real_t> out) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    assert(in.size() == static_cast<size_t>(N * nx + nx));
    assert(out.rows() == nx);
    assert(out.cols() == 1);
    assert(out.depth() == N + 1);
    auto in_vw = guanaqo::MatrixView<const real_t, index_t>::as_column(in);
    for (index_t i = 0; i < N + 1; ++i)
        out(i) = in_vw.middle_rows(i * nx, nx);
    for (index_t i = N + 1; i < out.ceil_depth(); ++i)
        out(i).set_constant(0);
}

template <simd_abi_tag Abi>
void SolverStorage<Abi>::restore_dynamics_constraints(
    view_type<const real_t> in, std::span<real_t> out) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    assert(out.size() == static_cast<size_t>(N * nx + nx));
    assert(in.rows() == nx);
    assert(in.cols() == 1);
    assert(in.depth() == N + 1);
    auto out_vw = guanaqo::MatrixView<real_t, index_t>::as_column(out);
    for (index_t i = 0; i < N + 1; ++i)
        out_vw.middle_rows(i * nx, nx) = in(i);
}

template <simd_abi_tag Abi>
void SolverStorage<Abi>::copy_variables(std::span<const real_t> in,
                                        view_type<real_t> out) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    assert(in.size() == static_cast<size_t>(N * (nx + nu) + nx));
    assert(out.rows() == nx + nu);
    assert(out.cols() == 1);
    assert(out.depth() == N + 1);
    auto in_vw = guanaqo::MatrixView<const real_t, index_t>::as_column(in);
    for (index_t i = 0; i < N; ++i)
        out(i) = in_vw.middle_rows(i * (nx + nu), nx + nu);
    out(N).top_rows(nx) = in_vw.middle_rows(N * (nx + nu), nx);
    out(N).bottom_rows(nu).set_constant(0);
    for (index_t i = N + 1; i < out.ceil_depth(); ++i)
        out(i).set_constant(0);
}

template <simd_abi_tag Abi>
void SolverStorage<Abi>::restore_variables(view_type<const real_t> in,
                                           std::span<real_t> out) const {
    auto [N, nx, nu, ny, ny_N] = dim;
    assert(out.size() == static_cast<size_t>(N * (nx + nu) + nx));
    assert(in.rows() == nx + nu);
    assert(in.cols() == 1);
    assert(in.depth() == N + 1);
    auto out_vw = guanaqo::MatrixView<real_t, index_t>::as_column(out);
    for (index_t i = 0; i < N; ++i)
        out_vw.middle_rows(i * (nx + nu), nx + nu) = in(i);
    out_vw.middle_rows(N * (nx + nu), nx) = in(N).top_rows(nx);
}

template <simd_abi_tag Abi>
[[nodiscard]] auto SolverStorage<Abi>::initialize_constraints(
    std::span<const real_t> in) const -> real_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    real_matrix out{{.depth = N + 1, .rows = ny, .cols = 1}};
    copy_constraints(in, out);
    return out;
}

template <simd_abi_tag Abi>
[[nodiscard]] auto SolverStorage<Abi>::initialize_dynamics_constraints(
    std::span<const real_t> in) const -> real_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    real_matrix out{{.depth = N + 1, .rows = nx, .cols = 1}};
    copy_dynamics_constraints(in, out);
    return out;
}

template <simd_abi_tag Abi>
[[nodiscard]] auto SolverStorage<Abi>::initialize_active_set(
    std::span<const bool> in) const -> mask_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    mask_matrix out{{.depth = N + 1, .rows = ny, .cols = 1}};
    copy_active_set(in, out);
    return out;
}

template <simd_abi_tag Abi>
[[nodiscard]] auto SolverStorage<Abi>::initialize_variables(
    std::span<const real_t> in) const -> real_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    real_matrix out{{.depth = N + 1, .rows = nx + nu, .cols = 1}};
    copy_variables(in, out);
    return out;
}

template <simd_abi_tag Abi>
[[nodiscard]] auto
SolverStorage<Abi>::initialize_constraints(real_t x) const -> real_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    real_matrix out{{.depth = N + 1, .rows = ny, .cols = 1}};
    if (x != 0)
        out.set_constant(x);
    return out;
}

template <simd_abi_tag Abi>
[[nodiscard]] auto SolverStorage<Abi>::initialize_dynamics_constraints(
    real_t x) const -> real_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    real_matrix out{{.depth = N + 1, .rows = nx, .cols = 1}};
    if (x != 0)
        out.set_constant(x);
    return out;
}

template <simd_abi_tag Abi>
[[nodiscard]] auto
SolverStorage<Abi>::initialize_active_set(bool x) const -> mask_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    mask_matrix out{{.depth = N + 1, .rows = ny, .cols = 1}};
    if (x)
        out.set_constant(x);
    return out;
}

template <simd_abi_tag Abi>
[[nodiscard]] auto
SolverStorage<Abi>::initialize_variables(real_t x) const -> real_matrix {
    auto [N, nx, nu, ny, ny_N] = dim;
    real_matrix out{{.depth = N + 1, .rows = nx + nu, .cols = 1}};
    if (x != 0)
        out.set_constant(x);
    return out;
}

} // namespace koqkatoo::ocp
