#pragma once

#include "householder-downdate.hpp"

namespace koqkatoo::cholundate::micro_kernels::householder {

template <index_t R>
void updowndate_diag(index_t colsA, mut_W_accessor<R> W, real_t *Ld,
                     index_t ldL, real_t *Ad, index_t ldA,
                     const real_t *Sp) noexcept;

template <index_t R>
void updowndate_full(index_t colsA, real_t *Ld, index_t ldL, real_t *Ad,
                     index_t ldA, const real_t *Sp) noexcept;

template <Config Conf>
void updowndate_tail(index_t colsA, mut_W_accessor<Conf.block_size_r> W,
                     real_t *Lp, index_t ldL, const real_t *Bp, index_t ldB,
                     real_t *Ap, index_t ldA, const real_t *Sp) noexcept;

} // namespace koqkatoo::cholundate::micro_kernels::householder
