#pragma once

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/aligned-storage.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/ocp/ocp.hpp>

#include <experimental/simd>
#include <atomic>

namespace koqkatoo::ocp {

using linalg::compact::aligned_simd_storage;
using linalg::compact::BatchedMatrix;
using linalg::compact::BatchedMatrixLayout;
using linalg::compact::BatchedMatrixView;

namespace stdx = std::experimental;

template <class Abi>
concept simd_abi_tag = stdx::is_abi_tag_v<Abi>;

/// ed in other classes.
template <simd_abi_tag Abi>
struct SolverTypes {
    using simd          = stdx::simd<real_t, Abi>;
    using mask          = typename simd::mask_type;
    using simd_stride_t = stdx::simd_size<real_t, Abi>;
    using simd_align_t  = stdx::memory_alignment<simd>;
    using mask_align_t  = stdx::memory_alignment<mask>;
    using index_simd =
        stdx::simd<index_t, stdx::simd_abi::deduce_t<index_t, simd_stride_t{}>>;
    using index_align_t = stdx::memory_alignment<index_simd>;
    static_assert(simd_align_t() <= simd_stride_t() * sizeof(real_t));
    using scalar_abi           = stdx::simd_abi::scalar;
    using scalar_simd          = stdx::simd<real_t, scalar_abi>;
    using scalar_simd_stride_t = stdx::simd_size<real_t, scalar_abi>;
    using scalar_simd_align_t  = stdx::memory_alignment<scalar_simd>;

    /// View of an arbitrary number of batches of matrices.
    template <class T>
    using view_type = BatchedMatrixView<T, index_t, simd_stride_t>;
    /// View of an arbitrary number of batches of matrices.
    using real_view = view_type<const real_t>;
    /// View of an arbitrary number of batches of matrices.
    using bool_view = view_type<const bool>;
    /// Mutable view of an arbitrary number of batches of matrices.
    using mut_real_view = view_type<real_t>;
    /// Mutable view of a single batch of matrices.
    using single_mut_real_view =
        BatchedMatrixView<real_t, index_t, simd_stride_t, simd_stride_t>;
    using single_real_view =
        BatchedMatrixView<const real_t, index_t, simd_stride_t, simd_stride_t>;

    /// Type that owns an arbitrary number of batches of matrices.
    using real_matrix =
        BatchedMatrix<real_t, index_t, simd_stride_t, index_t, simd_align_t>;

    /// Type that owns an arbitrary number of matrices.
    using scalar_real_matrix = BatchedMatrix<real_t, index_t>;
    /// Type that owns an arbitrary number of batches of boolean matrices.
    using mask_matrix =
        BatchedMatrix<bool, index_t, simd_stride_t, index_t, mask_align_t>;
    /// Type that owns an arbitrary number of batches of integer matrices.
    using index_matrix =
        BatchedMatrix<index_t, index_t, simd_stride_t, index_t, index_align_t>;

    /// View of a single scalar matrix.
    using single_scalar_mut_real_view =
        BatchedMatrixView<real_t, index_t, scalar_simd_stride_t,
                          scalar_simd_stride_t>;
    /// View of a scalar, non-interleaved batch of matrices.
    using scalar_mut_real_view =
        BatchedMatrixView<real_t, index_t, scalar_simd_stride_t>;
    /// Layout of a scalar, non-interleaved batch of matrices.
    using scalar_layout = BatchedMatrixLayout<index_t, scalar_simd_stride_t>;
};

/// Workspaces and other storage of matrices/vectors used by the OCP solver.
template <simd_abi_tag Abi>
struct SolverStorage {
    using types = SolverTypes<Abi>;

    template <class T>
    using view_type            = typename types::template view_type<T>;
    using real_view            = typename types::real_view;
    using bool_view            = typename types::bool_view;
    using mut_real_view        = typename types::mut_real_view;
    using single_mut_real_view = typename types::single_mut_real_view;
    using single_scalar_mut_real_view =
        typename types::single_scalar_mut_real_view;
    using scalar_mut_real_view           = typename types::scalar_mut_real_view;
    using scalar_layout                  = typename types::scalar_layout;
    using scalar_real_matrix             = typename types::scalar_real_matrix;
    using real_matrix                    = typename types::real_matrix;
    using mask_matrix                    = typename types::mask_matrix;
    using index_matrix                   = typename types::index_matrix;
    static constexpr index_t simd_stride = typename types::simd_stride_t();

    const OCPDim dim;

    real_matrix H = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.nx + dim.nu,
                            .cols  = dim.nx + dim.nu}};
    }();
    real_matrix LHV = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.nx + dim.nu + dim.nx,
                            .cols  = dim.nx + dim.nu}};
    }();
    real_matrix CD = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1,
                            .rows  = dim.ny, // assuming ny >= ny_N
                            .cols  = dim.nx + dim.nu}};
    }();
    real_matrix LΨd = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.nx,
                            .cols  = dim.nx}};
    }();
    scalar_layout scalar_layout_LΨd = [this] {
        return scalar_layout{{.depth = dim.N_horiz + 1, //
                              .rows  = dim.nx,
                              .cols  = dim.nx}};
    }();
    real_matrix LΨs = [this] {
        return real_matrix{{.depth = dim.N_horiz, //
                            .rows  = dim.nx,
                            .cols  = dim.nx}};
    }();
    scalar_layout scalar_layout_LΨs = [this] {
        return scalar_layout{{.depth = dim.N_horiz, //
                              .rows  = dim.nx,
                              .cols  = dim.nx}};
    }();
    real_matrix Wᵀ = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.nx + dim.nu,
                            .cols  = dim.nx}};
    }();
    real_matrix Z = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.nx + dim.nu,
                            .cols  = dim.ny}}; // assuming ny >= ny_N
    }();
    real_matrix Z1 = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.nx + dim.nu + dim.nx + dim.nx,
                            .cols  = dim.ny}}; // assuming ny >= ny_N
    }();
    real_matrix Σ_sgn = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.ny,          // assuming ny >= ny_N
                            .cols  = 1}};
    }();
    real_matrix Σ_ud = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.ny,          // assuming ny >= ny_N
                            .cols  = 1}};
    }();
    real_matrix Lupd = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.ny,          // assuming ny >= ny_N
                            .cols  = dim.ny}};
    }();
    real_matrix Wupd = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.nx,          // assuming ny >= ny_N
                            .cols  = dim.ny}};
    }();
    real_matrix Vupd = [this] {
        return real_matrix{{.depth = dim.N_horiz, //
                            .rows  = dim.nx,      // assuming ny >= ny_N
                            .cols  = dim.ny}};
    }();
    scalar_real_matrix A_ud = [this] {
        return scalar_real_matrix{{.depth = 2, //
                                   .rows  = dim.nx,
                                   .cols  = dim.N_horiz * dim.ny + dim.ny_N}};
    }();
    scalar_real_matrix D_ud = [this] {
        return scalar_real_matrix{{.depth = 1, //
                                   .rows  = dim.N_horiz * dim.ny + dim.ny_N,
                                   .cols  = 1}};
    }();
    real_matrix VV = [this] {
        return real_matrix{{.depth = dim.N_horiz + 1, //
                            .rows  = dim.nx,
                            .cols  = dim.nx}};
    }();
    real_matrix AB = [this] {
        return real_matrix{{.depth = dim.N_horiz, //
                            .rows  = dim.nx,
                            .cols  = dim.nx + dim.nu}};
    }();
    real_matrix λ1 = [this] {
        return real_matrix{{.depth = dim.N_horiz, //
                            .rows  = dim.nx,
                            .cols  = 1}};
    }();
    real_matrix mFx = [this] {
        return real_matrix{{.depth = dim.N_horiz, //
                            .rows  = dim.nx,
                            .cols  = 1}};
    }();
    index_matrix stagewise_update_counts = [this] {
        return index_matrix{{.depth = dim.N_horiz + 1, .rows = 1, .cols = 1}};
    }();

    struct join_counter_t {
        alignas(64) std::atomic<int> value{};
    };

    std::vector<join_counter_t> join_counters = std::vector<join_counter_t>(
        (dim.N_horiz + 1 + simd_stride - 1) / simd_stride);

    std::vector<real_t> work_batch =
        std::vector<real_t>((dim.N_horiz + 1 + simd_stride - 1) / simd_stride);

    std::vector<join_counter_t> &reset_join_counters() {
        for (auto &jc : join_counters)
            jc.value.store(0, std::memory_order_relaxed);
        return join_counters;
    }

    std::vector<real_t> work_chol_Ψ =
        std::vector<real_t>(3 * simd_stride * dim.nx * dim.nx);
    std::vector<real_t> Δλ_scalar =
        std::vector<real_t>((dim.N_horiz + 1) * dim.nx);

    auto work_LΨd(index_t i) -> single_scalar_mut_real_view {
        auto [N, nx, nu, ny, ny_N] = dim;
        assert(i < simd_stride);
        auto offset = i * 2 * nx * nx;
        return {{.data = &work_chol_Ψ[offset], .rows = nx, .cols = nx}};
    }

    auto work_LΨs(index_t i) -> single_scalar_mut_real_view {
        auto [N, nx, nu, ny, ny_N] = dim;
        assert(i < simd_stride);
        auto offset = (i * 2 + 1) * nx * nx;
        return {{.data = &work_chol_Ψ[offset], .rows = nx, .cols = nx}};
    }
    auto work_LΨd() -> scalar_mut_real_view {
        auto [N, nx, nu, ny, ny_N] = dim;
        auto offset                = 0;
        return {{.data  = &work_chol_Ψ[offset],
                 .depth = simd_stride,
                 .rows  = nx,
                 .cols  = nx}};
    }
    auto work_LΨs() -> scalar_mut_real_view {
        auto [N, nx, nu, ny, ny_N] = dim;
        auto offset                = nx * nx * simd_stride;
        return {{.data  = &work_chol_Ψ[offset],
                 .depth = simd_stride,
                 .rows  = nx,
                 .cols  = nx}};
    }
    auto work_VV() -> scalar_mut_real_view {
        auto [N, nx, nu, ny, ny_N] = dim;
        auto offset                = 2 * nx * nx * simd_stride;
        return {{.data  = &work_chol_Ψ[offset],
                 .depth = simd_stride,
                 .rows  = nx,
                 .cols  = nx}};
    }

    [[nodiscard]] scalar_mut_real_view LΨd_scalar() {
        return {LΨd.data(), scalar_layout_LΨd};
    }

    [[nodiscard]] scalar_mut_real_view LΨs_scalar() {
        return {LΨs.data(), scalar_layout_LΨs};
    }

    void copy_active_set(std::span<const bool> in, view_type<bool> out) const;
    void restore_active_set(view_type<const bool> in,
                            std::span<bool> out) const;
    void copy_constraints(std::span<const real_t> in,
                          view_type<real_t> out) const;
    void restore_constraints(view_type<const real_t> in,
                             std::span<real_t> out) const;
    void copy_dynamics_constraints(std::span<const real_t> in,
                                   view_type<real_t> out) const;
    void restore_dynamics_constraints(view_type<const real_t> in,
                                      std::span<real_t> out) const;
    void copy_variables(std::span<const real_t> in,
                        view_type<real_t> out) const;
    void restore_variables(view_type<const real_t> in,
                           std::span<real_t> out) const;

    [[nodiscard]] real_matrix
    initialize_constraints(std::span<const real_t> in) const;
    [[nodiscard]] real_matrix
    initialize_dynamics_constraints(std::span<const real_t> in) const;
    [[nodiscard]] mask_matrix
    initialize_active_set(std::span<const bool> in) const;
    [[nodiscard]] real_matrix
    initialize_variables(std::span<const real_t> in) const;

    [[nodiscard]] real_matrix initialize_constraints(real_t x = 0) const;
    [[nodiscard]] real_matrix
    initialize_dynamics_constraints(real_t x = 0) const;
    [[nodiscard]] mask_matrix initialize_active_set(bool x = false) const;
    [[nodiscard]] real_matrix initialize_variables(real_t x = 0) const;
};

} // namespace koqkatoo::ocp
