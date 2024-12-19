#pragma once

#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/preferred-backend.hpp>
#include <koqkatoo/ocp/solver/storage.hpp>
#include <koqkatoo/timing.hpp>

namespace koqkatoo::ocp {

struct SolverOptions {
    /// This selects the difference between doing one loop that operates on a
    /// single batch per iteration, and doing multiple batched BLAS calls that
    /// all have their own internal loops.
    bool prefer_single_loop = true;
    /// Configures which functions are forwarded to BLAS or MKL, and which
    /// functions use the custom micro-kernels.
    linalg::compact::PreferredBackend preferred_backend =
        linalg::compact::PreferredBackend::Reference;
};

template <simd_abi_tag Abi>
struct Solver {
    using types = SolverTypes<Abi>;

    template <class T>
    using view_type            = typename types::template view_type<T>;
    using real_view            = typename types::real_view;
    using bool_view            = typename types::bool_view;
    using mut_real_view        = typename types::mut_real_view;
    using single_mut_real_view = typename types::single_mut_real_view;
    using scalar_mut_real_view = typename types::scalar_mut_real_view;
    using scalar_layout        = typename types::scalar_layout;
    using scalar_real_matrix   = typename types::scalar_real_matrix;
    using real_matrix          = typename types::real_matrix;
    using mask_matrix          = typename types::mask_matrix;
    static constexpr index_t simd_stride = typename types::simd_stride_t();

    using compact_blas = linalg::compact::CompactBLAS<Abi>;
    using scalar_blas  = linalg::compact::CompactBLAS<stdx::simd_abi::scalar>;

    using storage_t = SolverStorage<Abi>;
    storage_t storage;

    SolverOptions settings{};

    struct Timings {
        using type = koqkatoo::DefaultTimings;
        type prepare_all, cholesky_Ψ, solve_add_rhs_1, solve_H_1, solve_mat_vec,
            solve_unshuffle, solve_Ψ, solve_shuffle, solve_mat_vec_tp,
            solve_add_rhs_2, solve_H_2, updowndate, updowndate_stages,
            updowndate_Ψ, chol_Ψ_copy_1, chol_Ψ_potrf, chol_Ψ_trsm,
            chol_Ψ_copy_2, chol_Ψ_syrk, solve_Ψ_solve, solve_Ψ_gemm,
            solve_Ψ_solve_tp, solve_Ψ_gemm_tp, schur_complement, cholesky_H,
            prepare_Ψ;
    };

    mut_real_view H() { return storage.H; }
    mut_real_view LHV() { return storage.LHV; }
    mut_real_view LH() {
        auto [N, nx, nu, ny, ny_N] = storage.dim;
        return LHV().top_rows(nx + nu);
    }
    mut_real_view V() { return LHV().bottom_rows(storage.dim.nx); }
    mut_real_view CD() { return storage.CD; }
    mut_real_view LΨd() { return storage.LΨd; }
    mut_real_view LΨs() { return storage.LΨs; }
    scalar_mut_real_view LΨd_scalar() { return storage.LΨd_scalar(); }
    scalar_mut_real_view LΨs_scalar() { return storage.LΨs_scalar(); }
    mut_real_view VV() { return storage.VV; }
    mut_real_view AB() { return storage.AB; }
    mut_real_view Wᵀ() { return storage.Wᵀ; }

    /// Overwrites the Cholesky factors of H (LH) with H + CDᵀ Σ CD.
    void schur_complement_H(real_view Σ, bool_view J);
    void schur_complement_Hi(index_t i, real_view Σ, bool_view J);
    /// Compute the Cholesky factorization of H (in-place in LH).
    void cholesky_H();
    void cholesky_Hi(index_t i);
    /// Solve a system of equations with H (with Cholesky factor in LH).
    void solve_H(mut_real_view x);

    void prepare_Ψ();
    void prepare_Ψi(index_t i);
    void prepare_all(real_t S, real_view Σ, bool_view J);
    void prepare_all(real_t S, real_view Σ, bool_view J, Timings &t);

    void cholesky_Ψ();
    void cholesky_Ψ(Timings &t);
    void solve_Ψ_scalar(std::span<real_t> λ);
    void solve_Ψ_scalar(std::span<real_t> λ, Timings &t);

    void factor(real_t S, real_view Σ, bool_view J);
    void factor(real_t S, real_view Σ, bool_view J, Timings &t);
    void factor_tbb(real_t S, real_view Σ, bool_view J);
    void factor_fork(real_t S, real_view Σ, bool_view J);
    void factor_omp(real_t S, real_view Σ, bool_view J);
    void solve(real_view grad, real_view Mᵀλ, real_view Aᵀŷ, real_view Mxb,
               mut_real_view d, mut_real_view Δλ, mut_real_view MᵀΔλ);
    void solve(real_view grad, real_view Mᵀλ, real_view Aᵀŷ, real_view Mxb,
               mut_real_view d, mut_real_view Δλ, mut_real_view MᵀΔλ,
               Timings &t);

    /// Mᵀλ
    void mat_vec_transpose_dynamics_constr(real_view λ, mut_real_view Mᵀλ);
    /// Mxb = Mx - b
    void residual_dynamics_constr(real_view x, real_view b, mut_real_view Mxb);
    /// Aᵀy
    void mat_vec_transpose_constr(real_view y, mut_real_view Aᵀy);
    /// Qx
    void mat_vec_cost_add(real_view x, mut_real_view Qx);
    /// Qx + q
    void cost_gradient(real_view x, real_view q, mut_real_view grad_f);
    /// Qx + q + S⁻¹(x - x₀)
    void cost_gradient_regularized(real_t S, real_view x, real_view x0,
                                   real_view q, mut_real_view grad_f);
    /// grad_f -= S⁻¹(x - x₀)
    void cost_gradient_remove_regularization(real_t S, real_view x,
                                             real_view x0,
                                             mut_real_view grad_f);

    void updowndate(real_view Σ, bool_view J_old, bool_view J_new, Timings *t);
    void updowndate_ψ();

    [[nodiscard]] index_t num_variables() const {
        auto [N, nx, nu, ny, ny_N] = storage.dim;
        return N * (nx + nu) + nx;
    }
    [[nodiscard]] index_t num_constraints() const {
        auto [N, nx, nu, ny, ny_N] = storage.dim;
        return N * ny + ny_N;
    }
    [[nodiscard]] index_t num_dynamics_constraints() const {
        auto [N, nx, nu, ny, ny_N] = storage.dim;
        return (N + 1) * nx;
    }

    Solver(const LinearOCPStorage &ocp);
};

} // namespace koqkatoo::ocp
