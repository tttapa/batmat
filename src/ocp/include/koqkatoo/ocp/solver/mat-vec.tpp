#pragma once

#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/openmp.h>

namespace koqkatoo::ocp {

template <simd_abi_tag Abi>
void Solver<Abi>::mat_vec_transpose_dynamics_constr(real_view λ,
                                                    mut_real_view Mᵀλ) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    assert(λ.depth() == N + 1);
    assert(Mᵀλ.depth() == N + 1);
    assert(λ.rows() == nx);
    assert(Mᵀλ.rows() == nx + nu);
    for (index_t i = 0; i < N; ++i)
        storage.λ1(i) = λ(i + 1);
    Mᵀλ.set_constant(0);
    for (index_t i = 0; i < λ.num_batches(); ++i)
        // Mᵀλ(i) = [I 0]ᵀ λ(i)
        compact_blas::xcopy(λ.batch(i), Mᵀλ.batch(i).top_rows(nx));
    // Mᵀλ(i) = -[A B]ᵀ(i) λ(i+1) + [I 0]ᵀ λ(i)
    compact_blas::xgemm_TN_sub(AB(), storage.λ1, Mᵀλ.first_layers(N),
                               settings.preferred_backend);
}

template <simd_abi_tag Abi>
void Solver<Abi>::residual_dynamics_constr(real_view x, real_view b,
                                           mut_real_view Mxb) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    // mFx = -Ax -Bu
    compact_blas::xgemm_neg(AB(), x.first_layers(N), storage.mFx,
                            settings.preferred_backend);
    for (index_t i = 0; i < N + 1; ++i) {
        // Mxb(i) = x(i) - b(i)
        for (index_t j = 0; j < nx; ++j)
            Mxb(i, j, 0) = x(i, j, 0) - b(i, j, 0);
        if (i > 0)
            // Mxb(i) = x(i) - b(i) - Ax(i-1) - Bu(i-1)
            for (index_t j = 0; j < nx; ++j)
                Mxb(i, j, 0) += storage.mFx(i - 1, j, 0);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::mat_vec_transpose_constr(real_view y, mut_real_view Aᵀy) {
    // Ax = Cx + Du
    compact_blas::xgemm_TN(CD(), y, Aᵀy, settings.preferred_backend);
}

template <simd_abi_tag Abi>
void Solver<Abi>::mat_vec_cost_add(real_view x, mut_real_view Qx) {
    // Qx += Q * x
    compact_blas::xsymv_add(H(), x, Qx, settings.preferred_backend);
}

template <simd_abi_tag Abi>
void Solver<Abi>::cost_gradient(real_view x, real_view q,
                                mut_real_view grad_f) {
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < H().num_batches(); ++i) {
        compact_blas::xcopy(q.batch(i), grad_f.batch(i));
        compact_blas::xsymv_add(H().batch(i), x.batch(i), grad_f.batch(i),
                                settings.preferred_backend);
    }
}

} // namespace koqkatoo::ocp
