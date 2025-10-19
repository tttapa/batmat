#pragma once

#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/flops.hpp>
#include <batmat/linalg/micro-kernels/hyhound.tpp> // TODO
#include <batmat/linalg/shift.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

/// Update Cholesky factor L using low-rank term A diag(d) Aᵀ.
template <MatrixStructure SL, simdifiable VL, simdifiable VA, simdifiable Vd>
    requires simdify_compatible<VL, VA, Vd>
void hyhound_diag(Structured<VL, SL> L, VA &&A, Vd &&d) {
    micro_kernels::hyhound::xshhud_diag_ref<simdified_value_t<VL>, simdified_abi_t<VL>>(
        simdify(L.value), simdify(A), simdify(d).as_const());
}

/// Update Cholesky factor L using low-rank term A diag(copysign(1, d)) Aᵀ,
/// where d contains only ±0 values.
template <MatrixStructure SL, simdifiable VL, simdifiable VA, simdifiable Vd>
    requires simdify_compatible<VL, VA, Vd>
void hyhound_sign(Structured<VL, SL> L, VA &&A, Vd &&d) {
    micro_kernels::hyhound::xshhud_diag_ref<simdified_value_t<VL>, simdified_abi_t<VL>, true>(
        simdify(L.value), simdify(A), simdify(d).as_const());
}

/// Update Cholesky factor L using low-rank term A diag(d) Aᵀ, where L and A are stored as two
/// separate block rows.
template <MatrixStructure SL, simdifiable VL1, simdifiable VA1, simdifiable VL2, simdifiable VA2,
          simdifiable Vd>
    requires simdify_compatible<VL1, VA1, VL2, VA2, Vd>
void hyhound_diag_2(Structured<VL1, SL> L1, VA1 &&A1, VL2 &&L2, VA2 &&A2, Vd &&d) {
    micro_kernels::hyhound::xshhud_diag_2_ref<simdified_value_t<VL1>, simdified_abi_t<VL1>>(
        simdify(L1.value), simdify(A1), simdify(L2), simdify(A2), simdify(d).as_const());
}

/// @todo Docs
template <MatrixStructure SL, simdifiable VL11, simdifiable VA1, simdifiable VL21, simdifiable VA2,
          simdifiable VA2o, simdifiable VU, simdifiable VA3, simdifiable VA3o, simdifiable Vd>
    requires simdify_compatible<VL11, VA1, VL21, VA2, VA2o, VU, VA3, VA3o, Vd>
void hyhound_diag_cyclic(Structured<VL11, SL> L11, VA1 &&A1, VL21 &&L21, VA2 &&A2, VA2o &&A2_out,
                         VU &&L31, VA3 &&A3, VA3o &&A3_out, Vd &&d, index_t split_A, int rot_A2) {
    micro_kernels::hyhound::xshhud_diag_cyclic<simdified_value_t<VL11>, simdified_abi_t<VL11>>(
        simdify(L11.value), simdify(A1), simdify(L21), simdify(A2).as_const(), simdify(A2_out),
        simdify(L31), simdify(A3).as_const(), simdify(A3_out), simdify(d).as_const(), split_A,
        rot_A2);
}

/// @todo Docs
template <MatrixStructure SL, simdifiable VL11, simdifiable VA1, simdifiable VL21, simdifiable VA2,
          simdifiable VA2o, simdifiable VLu1, simdifiable VAuo, simdifiable Vd>
    requires simdify_compatible<VL11, VA1, VL21, VA2, VA2o, VLu1, VAuo, Vd>
void hyhound_diag_riccati(Structured<VL11, SL> L11, VA1 &&A1, VL21 &&L21, VA2 &&A2, VA2o &&A2_out,
                          VLu1 &&Lu1, VAuo &&Au_out, Vd &&d, bool shift_A_out) {
    micro_kernels::hyhound::xshhud_diag_riccati<simdified_value_t<VL11>, simdified_abi_t<VL11>>(
        simdify(L11.value), simdify(A1), simdify(L21), simdify(A2).as_const(), simdify(A2_out),
        simdify(Lu1), simdify(Au_out), simdify(d).as_const(), shift_A_out);
}

} // namespace batmat::linalg
