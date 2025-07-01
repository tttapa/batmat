#include <cyclocp/compact.hpp>

#include <batmat/linalg/micro-kernels/hyhound.tpp>

namespace batmat::linalg::compact {

template <class T, class Abi>
void CompactBLAS<T, Abi>::xshhud_diag_ref(mut_single_batch_view L, mut_single_batch_view A,
                                          single_batch_view D) {
    batmat::linalg::micro_kernels::hyhound::xshhud_diag_ref<T, Abi>(L, A, D);
}

template <class T, class Abi>
void CompactBLAS<T, Abi>::xshhud_diag_2_ref(mut_single_batch_view L, mut_single_batch_view A,
                                            mut_single_batch_view L2, mut_single_batch_view A2,
                                            single_batch_view D) {
    batmat::linalg::micro_kernels::hyhound::xshhud_diag_2_ref<T, Abi>(L, A, L2, A2, D);
}

template <class T, class Abi>
void CompactBLAS<T, Abi>::xshhud_diag_cyclic(mut_single_batch_view L11, mut_single_batch_view A1,
                                             mut_single_batch_view L21, single_batch_view A2,
                                             mut_single_batch_view A2_out,
                                             mut_single_batch_view L31, single_batch_view A3,
                                             mut_single_batch_view A3_out, single_batch_view D,
                                             index_t split, int rot_A2) {
    batmat::linalg::micro_kernels::hyhound::xshhud_diag_cyclic<T, Abi>(
        L11, A1, L21, A2, A2_out, L31, A3, A3_out, D, split, rot_A2);
}

template <class T, class Abi>
void CompactBLAS<T, Abi>::xshhud_diag_riccati(mut_single_batch_view L11, mut_single_batch_view A1,
                                              mut_single_batch_view L21, single_batch_view A2,
                                              mut_single_batch_view A2_out,
                                              mut_single_batch_view Lu1,
                                              mut_single_batch_view Au_out, single_batch_view D,
                                              bool shift_A_out) {
    batmat::linalg::micro_kernels::hyhound::xshhud_diag_riccati<T, Abi>(
        L11, A1, L21, A2, A2_out, Lu1, Au_out, D, shift_A_out);
}

} // namespace batmat::linalg::compact
