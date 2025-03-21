#pragma once

#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/openmp.h>
#include <guanaqo/trace.hpp>

namespace koqkatoo::ocp {

template <simd_abi_tag Abi>
void Solver<Abi>::mat_vec_transpose_dynamics_constr(real_view λ,
                                                    mut_real_view Mᵀλ) {
    GUANAQO_TRACE("mat_vec_transpose_dynamics_constr", 0);
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    assert(λ.depth() == N + 1);
    assert(Mᵀλ.depth() == N + 1);
    assert(λ.rows() == nx);
    assert(Mᵀλ.rows() == nx + nu);
    for (index_t i = 0; i < N; ++i)
        storage.λ1()(i) = λ(i + 1);
    Mᵀλ.set_constant(0);
    for (index_t i = 0; i < λ.num_batches(); ++i)
        // Mᵀλ(i) = [I 0]ᵀ λ(i)
        compact_blas::xcopy(λ.batch(i), Mᵀλ.batch(i).top_rows(nx));
    // Mᵀλ(i) = -[A B]ᵀ(i) λ(i+1) + [I 0]ᵀ λ(i)
    compact_blas::xgemv_T_sub(AB(), storage.λ1().first_layers(N),
                              Mᵀλ.first_layers(N), settings.preferred_backend);
}

template <simd_abi_tag Abi>
void Solver<Abi>::residual_dynamics_constr(real_view x, real_view b,
                                           mut_real_view Mxb) {
    GUANAQO_TRACE("residual_dynamics_constr", 0);
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    // mFx = -Ax -Bu
    compact_blas::xgemv_neg(AB(), x.first_layers(N),
                            storage.mFx().first_layers(N),
                            settings.preferred_backend);
    for (index_t i = 0; i < N + 1; ++i) {
        // Mxb(i) = x(i) - b(i)
        for (index_t j = 0; j < nx; ++j)
            Mxb(i, j, 0) = x(i, j, 0) - b(i, j, 0);
        if (i > 0)
            // Mxb(i) = x(i) - b(i) - Ax(i-1) - Bu(i-1)
            for (index_t j = 0; j < nx; ++j)
                Mxb(i, j, 0) += storage.mFx()(i - 1, j, 0);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::mat_vec_transpose_constr(real_view y, mut_real_view Aᵀy) {
    GUANAQO_TRACE("mat_vec_transpose_constr", 0);
    // Ax = Cx + Du
    compact_blas::xgemv_T(CD(), y, Aᵀy, settings.preferred_backend);
}

template <simd_abi_tag Abi>
void Solver<Abi>::mat_vec_cost_add(real_view x, mut_real_view Qx) {
    GUANAQO_TRACE("mat_vec_cost_add", 0);
    // Qx += Q * x
    compact_blas::xsymv_add(H(), x, Qx, settings.preferred_backend);
}

template <simd_abi_tag Abi>
void Solver<Abi>::cost_gradient(real_view x, real_view q,
                                mut_real_view grad_f) {
    GUANAQO_TRACE("cost_gradient", 0);
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < H().num_batches(); ++i) {
        compact_blas::xcopy(q.batch(i), grad_f.batch(i));
        compact_blas::xsymv_add(H().batch(i), x.batch(i), grad_f.batch(i),
                                settings.preferred_backend);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::cost_gradient_regularized(real_t S, real_view x, real_view x0,
                                            real_view q, mut_real_view grad_f) {
    GUANAQO_TRACE("cost_gradient_regularized", 0);
    assert(x.depth() == x0.depth());
    assert(x.depth() == q.depth());
    assert(x.depth() == grad_f.depth());
    assert(x.rows() == x0.rows());
    assert(x.rows() == q.rows());
    assert(x.rows() == grad_f.rows());
    assert(x.cols() == 1);
    assert(x.cols() == x0.cols());
    assert(x.cols() == q.cols());
    assert(x.cols() == grad_f.cols());
    using simd = types::simd;
    simd invS{1 / S};
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < H().num_batches(); ++i) {
        auto qi = q.batch(i), xi = x.batch(i), x0i = x0.batch(i);
        auto grad_fi = grad_f.batch(i);
        for (index_t j = 0; j < x.rows(); ++j) {
            simd qij{&qi(0, j, 0), stdx::vector_aligned},
                xij{&xi(0, j, 0), stdx::vector_aligned},
                x0ij{&x0i(0, j, 0), stdx::vector_aligned};
            simd grad_fij = invS * (xij - x0ij) + qij;
            grad_fij.copy_to(&grad_fi(0, j, 0), stdx::vector_aligned);
        }
        compact_blas::xsymv_add(H().batch(i), x.batch(i), grad_f.batch(i),
                                settings.preferred_backend);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::cost_gradient_remove_regularization(real_t S, real_view x,
                                                      real_view x0,
                                                      mut_real_view grad_f) {
    GUANAQO_TRACE("cost_gradient_remove_regularization", 0);
    assert(x.depth() == x0.depth());
    assert(x.depth() == grad_f.depth());
    assert(x.rows() == x0.rows());
    assert(x.rows() == grad_f.rows());
    assert(x.cols() == 1);
    assert(x.cols() == x0.cols());
    assert(x.cols() == grad_f.cols());
    using simd = types::simd;
    simd invS{1 / S};
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < x.num_batches(); ++i) {
        auto xi = x.batch(i), x0i = x0.batch(i);
        auto grad_fi = grad_f.batch(i);
        for (index_t j = 0; j < x.rows(); ++j) {
            simd grad_fij{&grad_fi(0, j, 0), stdx::vector_aligned},
                xij{&xi(0, j, 0), stdx::vector_aligned},
                x0ij{&x0i(0, j, 0), stdx::vector_aligned};
            grad_fij += invS * (x0ij - xij);
            grad_fij.copy_to(&grad_fi(0, j, 0), stdx::vector_aligned);
        }
    }
}

} // namespace koqkatoo::ocp
