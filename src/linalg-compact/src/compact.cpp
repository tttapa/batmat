#ifdef __clang__
#pragma clang fp contract(fast)
#endif

#include <koqkatoo/linalg-compact/compact.tpp>

namespace koqkatoo::linalg::compact {

#define INSTANTIATE(...)                                                       \
    template struct CompactBLAS<__VA_ARGS__>;                                  \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl(                    \
        mut_batch_view, batch_view, batch_view);                               \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl(                    \
        mut_single_batch_view, single_batch_view);                             \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl<1>(                 \
        mut_single_batch_view, single_batch_view);                             \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl<-1>(                \
        mut_single_batch_view, single_batch_view);                             \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl(                    \
        mut_single_batch_view, single_batch_view, single_batch_view);          \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl<-1>(                \
        mut_single_batch_view, single_batch_view, single_batch_view);          \
    template void CompactBLAS<__VA_ARGS__>::xsub_copy_impl(                    \
        mut_batch_view, batch_view, batch_view);                               \
    template void CompactBLAS<__VA_ARGS__>::xadd_neg_copy_impl(                \
        mut_batch_view, batch_view, batch_view);                               \
    template void CompactBLAS<__VA_ARGS__>::xadd_neg_copy_impl(                \
        mut_single_batch_view, single_batch_view);                             \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl(                    \
        mut_batch_view, batch_view, batch_view, batch_view);                   \
    template void CompactBLAS<__VA_ARGS__>::xsub_copy_impl(                    \
        mut_batch_view, batch_view, batch_view, batch_view);                   \
    template void CompactBLAS<__VA_ARGS__>::xadd_neg_copy_impl(                \
        mut_batch_view, batch_view, batch_view, batch_view);                   \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl(                    \
        mut_batch_view, batch_view, batch_view, batch_view, batch_view);       \
    template void CompactBLAS<__VA_ARGS__>::xsub_copy_impl(                    \
        mut_batch_view, batch_view, batch_view, batch_view, batch_view);       \
    template void CompactBLAS<__VA_ARGS__>::xadd_neg_copy_impl(                \
        mut_batch_view, batch_view, batch_view, batch_view, batch_view);       \
    template void CompactBLAS<__VA_ARGS__>::xsub_copy_impl(                    \
        mut_single_batch_view, single_batch_view, single_batch_view);          \
    template void CompactBLAS<__VA_ARGS__>::xadd_neg_copy_impl(                \
        mut_single_batch_view, single_batch_view, single_batch_view);          \
    template void CompactBLAS<__VA_ARGS__>::xsub_copy_impl(                    \
        mut_single_batch_view, single_batch_view, single_batch_view,           \
        single_batch_view);                                                    \
    template void CompactBLAS<__VA_ARGS__>::xadd_neg_copy_impl(                \
        mut_single_batch_view, single_batch_view, single_batch_view,           \
        single_batch_view);                                                    \
    template void CompactBLAS<__VA_ARGS__>::xsub_copy_impl(                    \
        mut_single_batch_view, single_batch_view, single_batch_view,           \
        single_batch_view, single_batch_view);                                 \
    template void CompactBLAS<__VA_ARGS__>::xadd_neg_copy_impl(                \
        mut_single_batch_view, single_batch_view, single_batch_view,           \
        single_batch_view, single_batch_view);                                 \
    template void CompactBLAS<__VA_ARGS__>::xadd_neg_copy_impl<1>(             \
        mut_single_batch_view, single_batch_view);                             \
    template void CompactBLAS<__VA_ARGS__>::xadd_neg_copy_impl<-1>(            \
        mut_single_batch_view, single_batch_view);                             \
    template index_t CompactBLAS<__VA_ARGS__>::compress_masks(                 \
        single_batch_view, single_batch_view, mut_single_batch_view,           \
        mut_single_batch_view);                                                \
    template index_t CompactBLAS<__VA_ARGS__>::compress_masks<4>(              \
        single_batch_view, single_batch_view, mut_single_batch_view,           \
        mut_single_batch_view);                                                \
    template index_t CompactBLAS<__VA_ARGS__>::compress_masks_count(           \
        single_batch_view);                                                    \
    template index_t CompactBLAS<__VA_ARGS__>::compress_masks_count<4>(        \
        single_batch_view);                                                    \
    template void CompactBLAS<__VA_ARGS__>::xshhud_diag_riccati(               \
        mut_single_batch_view, mut_single_batch_view, mut_single_batch_view,   \
        single_batch_view, mut_single_batch_view, mut_single_batch_view,       \
        mut_single_batch_view, single_batch_view, bool);                       \
    template void CompactBLAS<__VA_ARGS__>::xshhud_diag_riccati<true>(         \
        mut_single_batch_view, mut_single_batch_view, mut_single_batch_view,   \
        single_batch_view, mut_single_batch_view, mut_single_batch_view,       \
        mut_single_batch_view, single_batch_view, bool)

INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 16>);
INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 8>);
INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 4>);
INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 2>);
INSTANTIATE(stdx::simd_abi::scalar);

} // namespace koqkatoo::linalg::compact
