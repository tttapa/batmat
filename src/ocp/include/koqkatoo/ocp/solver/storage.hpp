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
    using view_type =
        BatchedMatrixView<T, index_t, simd_stride_t, index_t, index_t>;
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
    using scalar_real_matrix =
        BatchedMatrix<real_t, index_t, scalar_simd_stride_t>;
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
        BatchedMatrixView<real_t, index_t, scalar_simd_stride_t, index_t,
                          index_t>;
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

    // clang-format off
    real_matrix stagewise_storage = [this] {
        auto [N, nx, nu, ny, ny_N] = dim;
        return real_matrix{{
            .depth = N + 1,
            .rows = nx*(2*nx) + 2*nx*ny + nx*(nu + nx) + nx*nx + 2*nx*1 + ny*(nu + nx) + ny*(nu + nx) + ny*(nu + 3*nx) + ny*ny + 2*ny*1 + 2*(nu + nx)*(nu + 2*nx),
            .cols = 1,
        }};
    }();

    #define KQT_PRECOMPUTE_OFFSET(name, value)                                 \
        const index_t name ## _offset = [this] {                               \
            [[maybe_unused]] auto [N, nx, nu, ny, ny_N] = dim;                 \
            return value;                                                      \
        }()

    static constexpr index_t CD_offset = 0;
    KQT_PRECOMPUTE_OFFSET(HAB, nu*ny + nx*ny);
    KQT_PRECOMPUTE_OFFSET(LHV, 3*nu*nx + nu*ny + nu*nu + nx*ny + 2*(nx*nx));
    KQT_PRECOMPUTE_OFFSET(λ1, 6*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 4*(nx*nx));
    KQT_PRECOMPUTE_OFFSET(mFx, 6*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + nx + 4*(nx*nx));
    KQT_PRECOMPUTE_OFFSET(Wᵀ, 6*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 2*nx + 4*(nx*nx));
    KQT_PRECOMPUTE_OFFSET(WWᵀVWᵀ, 7*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 2*nx + 5*(nx*nx));
    KQT_PRECOMPUTE_OFFSET(VVᵀ, 7*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 2*nx + 7*(nx*nx));
    KQT_PRECOMPUTE_OFFSET(Z, 7*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 2*nx + 8*(nx*nx));
    KQT_PRECOMPUTE_OFFSET(Z1, 7*nu*nx + 2*nu*ny + 2*(nu*nu) + 2*nx*ny + 2*nx + 8*(nx*nx));
    KQT_PRECOMPUTE_OFFSET(Σ_sgn, 7*nu*nx + 3*nu*ny + 2*(nu*nu) + 5*nx*ny + 2*nx + 8*(nx*nx));
    KQT_PRECOMPUTE_OFFSET(Σ_ud, 7*nu*nx + 3*nu*ny + 2*(nu*nu) + 5*nx*ny + 2*nx + 8*(nx*nx) + ny);
    KQT_PRECOMPUTE_OFFSET(Lupd, 7*nu*nx + 3*nu*ny + 2*(nu*nu) + 5*nx*ny + 2*nx + 8*(nx*nx) + 2*ny);
    KQT_PRECOMPUTE_OFFSET(Wupd, 7*nu*nx + 3*nu*ny + 2*(nu*nu) + 5*nx*ny + 2*nx + 8*(nx*nx) + ny*ny + 2*ny);
    KQT_PRECOMPUTE_OFFSET(Vupd, 7*nu*nx + 3*nu*ny + 2*(nu*nu) + 6*nx*ny + 2*nx + 8*(nx*nx) + ny*ny + 2*ny);

    mut_real_view CD() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(CD_offset, ny*(nu + nx)).reshaped(ny, nu + nx);
    }

    mut_real_view HAB() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(HAB_offset, (nu + nx)*(nu + 2*nx)).reshaped(nu + 2*nx, nu + nx);
    }

    mut_real_view LHV() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(LHV_offset, (nu + nx)*(nu + 2*nx)).reshaped(nu + 2*nx, nu + nx);
    }

    mut_real_view λ1() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(λ1_offset, nx*1).reshaped(nx, 1);
    }

    mut_real_view mFx() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(mFx_offset, nx*1).reshaped(nx, 1);
    }

    mut_real_view Wᵀ() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(Wᵀ_offset, nx*(nu + nx)).reshaped(nu + nx, nx);
    }

    mut_real_view WWᵀVWᵀ() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(WWᵀVWᵀ_offset, nx*(2*nx)).reshaped(2*nx, nx);
    }

    mut_real_view VVᵀ() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(VVᵀ_offset, nx*nx).reshaped(nx, nx);
    }

    mut_real_view Z() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(Z_offset, ny*(nu + nx)).reshaped(nu + nx, ny);
    }

    mut_real_view Z1() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(Z1_offset, ny*(nu + 3*nx)).reshaped(nu + 3*nx, ny);
    }

    mut_real_view Σ_sgn() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(Σ_sgn_offset, ny*1).reshaped(ny, 1);
    }

    mut_real_view Σ_ud() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(Σ_ud_offset, ny*1).reshaped(ny, 1);
    }

    mut_real_view Lupd() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(Lupd_offset, ny*ny).reshaped(ny, ny);
    }

    mut_real_view Wupd() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(Wupd_offset, nx*ny).reshaped(nx, ny);
    }

    mut_real_view Vupd() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(Vupd_offset, nx*ny).reshaped(nx, ny);
    }

    #undef KQT_PRECOMPUTE_OFFSET
    // clang-format on

    mut_real_view H() { return HAB().top_rows(dim.nx + dim.nu); }
    mut_real_view AB() { return HAB().bottom_rows(dim.nx); }
    mut_real_view WWᵀ() { return WWᵀVWᵀ().top_rows(dim.nx); }
    mut_real_view VWᵀ() { return WWᵀVWᵀ().bottom_rows(dim.nx); }

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

    scalar_real_matrix work_chol_Ψ = [this] {
        auto [N, nx, nu, ny, ny_N] = dim;
        return scalar_real_matrix{{
            .depth = N + 1,
            .rows  = 3 * nx * nx,
            .cols  = 1,
        }};
    }();
    std::vector<real_t> Δλ_scalar =
        std::vector<real_t>((dim.N_horiz + 1) * dim.nx);

    [[nodiscard]] scalar_mut_real_view LΨ_scalar() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return work_chol_Ψ.view.top_rows(2 * nx * nx).reshaped(2 * nx, nx);
    }
    [[nodiscard]] scalar_mut_real_view LΨd_scalar() {
        return LΨ_scalar().top_rows(dim.nx);
    }
    [[nodiscard]] scalar_mut_real_view LΨs_scalar() {
        return LΨ_scalar().bottom_rows(dim.nx);
    }
    [[nodiscard]] scalar_mut_real_view VVᵀ_scalar() {
        auto [N, nx, nu, ny, ny_N] = dim;
        return work_chol_Ψ.view.bottom_rows(nx * nx).reshaped(nx, nx);
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
