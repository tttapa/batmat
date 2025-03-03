#pragma once

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/preferred-backend.hpp>
#include <koqkatoo/ocp/ocp.hpp>
#include <koqkatoo/ocp/random-ocp.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/trace.hpp>

#include <experimental/simd>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <type_traits>

namespace koqkatoo::ocp {

namespace stdx = std::experimental;

[[nodiscard]] constexpr index_t get_depth(index_t n) {
    assert(n > 0);
    auto un = static_cast<std::make_unsigned_t<index_t>>(n);
    return static_cast<index_t>(std::bit_width(un - 1));
}

[[nodiscard]] constexpr index_t get_level(index_t i) {
    assert(i > 0);
    auto ui = static_cast<std::make_unsigned_t<index_t>>(i);
    return static_cast<index_t>(std::countr_zero(ui));
}

template <class Abi>
struct CyclicOCPSolver {
    using simd_abi      = Abi;
    using compact_blas  = linalg::compact::CompactBLAS<simd_abi>;
    using scalar_blas   = linalg::compact::CompactBLAS<stdx::simd_abi::scalar>;
    using real_matrix   = compact_blas::matrix;
    using real_view     = compact_blas::batch_view;
    using mut_real_view = compact_blas::mut_batch_view;
    using bool_view     = compact_blas::bool_batch_view;
    using bool_matrix   = compact_blas::bool_matrix;
    using real_view_single     = compact_blas::single_batch_view;
    using mut_real_view_single = compact_blas::mut_single_batch_view;

    static constexpr index_t vl  = stdx::simd_size_v<real_t, Abi>;
    static constexpr index_t lvl = get_depth(vl);

    struct Options {
        bool use_pcg = true; // TODO: implement use_pcg=false
    };

    const OCPDim dim;
    const Options opts;
    const index_t lN      = get_depth(dim.N_horiz + 1);
    const index_t ln      = lN - lvl;
    const index_t n       = index_t{1} << ln;
    const index_t vstride = index_t{1} << ln;

    linalg::compact::PreferredBackend be =
        linalg::compact::PreferredBackend::MKLScalarBatched;

    [[nodiscard]] static constexpr index_t get_index_in_level(index_t i) {
        auto l = get_level(i);
        return i >> (l + 1);
    }

    [[nodiscard]] constexpr index_t get_level_width(index_t l) const {
        assert(l < ln);
        return index_t{1} << (ln - l - 1);
    }

    [[nodiscard]] constexpr index_t get_batch_index(index_t i) const {
        assert(i < n);
        if (i == 0)
            return n - 1;
        auto l  = get_level(i);
        auto il = get_index_in_level(i);
        return il + (1 << ln) - (1 << (ln - l));
    }

    CyclicOCPSolver(OCPDim dim, const Options &opts = {})
        : dim{dim}, opts{opts} {}
    CyclicOCPSolver(const LinearOCPStorage &ocp, const Options &opts = {})
        : CyclicOCPSolver{ocp.dim, opts} {
        initialize(ocp);
    }
    CyclicOCPSolver(const CyclicOCPSolver &)            = delete;
    CyclicOCPSolver &operator=(const CyclicOCPSolver &) = delete;

    /// @name Packing for vectorization
    /// @{

    void initialize(const LinearOCPStorage &ocp);

    template <class Tin, class Tout>
    void pack_vectors(std::span<Tin> in,
                      typename compact_blas::template batch_view_t<Tout> out,
                      index_t rows, index_t rows_N) const;
    template <class Tin, class Tout>
    void unpack_vectors(typename compact_blas::template batch_view_t<Tin> in,
                        std::span<Tout> out, index_t rows,
                        index_t rows_N) const;

    template <class Tin, class Tout>
    void pack_dyn(std::span<Tin> in,
                  typename compact_blas::template batch_view_t<Tout> out) const;
    real_matrix pack_dyn(std::span<const real_t> in = {}) const;
    template <class Tin, class Tout>
    void unpack_dyn(typename compact_blas::template batch_view_t<Tin> in,
                    std::span<Tout> out) const;
    template <class Tin, class Tout>
    void
    pack_constr(std::span<Tin> in,
                typename compact_blas::template batch_view_t<Tout> out) const;
    real_matrix pack_constr(std::span<const real_t> in = {}) const;
    bool_matrix pack_constr(std::span<const bool> in = {}) const;
    template <class Tin, class Tout>
    void unpack_constr(typename compact_blas::template batch_view_t<Tin> in,
                       std::span<Tout> out) const;
    template <class Tin, class Tout>
    void pack_var(std::span<Tin> in,
                  typename compact_blas::template batch_view_t<Tout> out) const;
    real_matrix pack_var(std::span<const real_t> in = {}) const;
    template <class Tin, class Tout>
    void unpack_var(typename compact_blas::template batch_view_t<Tin> in,
                    std::span<Tout> out) const;

    /// @}

    /// @name Matrix-vector products
    /// @{

    // Compute Aᵀŷ.
    void mat_vec_constr_tp(real_view ŷb, mut_real_view Aᵀŷb) const;
    /// Compute Mx - b
    void mat_vec_dyn(real_view xb, real_view bb, mut_real_view Mxbb) const;
    // Compute Mᵀλ
    void mat_vec_dyn_tp(real_view λb, mut_real_view Mᵀλb) const;

    /// @}

    /// @name Preconditioned conjugate gradient solver
    /// @{

    real_t mul_A(real_view_single p, // NOLINT(*-nodiscard)
                 mut_real_view_single Ap, real_view_single L,
                 real_view_single B) const;

    real_t mul_precond(real_view_single r, // NOLINT(*-nodiscard)
                       mut_real_view_single z, mut_real_view_single w,
                       real_view_single L, real_view_single B) const;

    /// @}

    /// @name Factorization and solution
    /// @{

    void compute_Ψ(real_t S, real_view Σb, bool_view Jb);
    void factor_Ψ();
    void solve_H_fwd(real_view grad, real_view Mᵀλ, real_view Aᵀŷ,
                     mut_real_view d, mut_real_view Δλ) const;
    void solve_H_rev(mut_real_view d, real_view Δλ, mut_real_view MᵀΔλ) const;
    void solve_Ψ_work(mut_real_view Δλ, mut_real_view work_pcg) const;
    void solve_Ψ(mut_real_view Δλ) { solve_Ψ_work(Δλ, work_pcg); }

    /// @}

#include "cyclic-storage.ipp"

    const mut_real_view H   = HAB.top_rows(dim.nx + dim.nu),
                        AB  = HAB.bottom_rows(dim.nx);
    const mut_real_view LH  = LHV.top_rows(dim.nx + dim.nu),
                        V   = LHV.bottom_rows(dim.nx);
    const mut_real_view LΨd = LΨU.top_rows(dim.nx),
                        Ut  = LΨU.middle_rows(dim.nx, dim.nx),
                        Ub  = LΨU.bottom_rows(dim.nx);
    struct alignas(64) atomic_counter {
        std::atomic<index_t> counter;
    };
    mutable std::vector<atomic_counter> counters =
        std::vector<atomic_counter>(n);
    real_matrix work_pcg{{.depth = vl, .rows = dim.nx, .cols = 4}};
};

} // namespace koqkatoo::ocp
