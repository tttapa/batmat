#include <koqkatoo/linalg-compact/compact.tpp>

namespace koqkatoo::linalg::compact {

#define INSTANTIATE(...)                                                       \
    template struct CompactBLAS<__VA_ARGS__>;                                  \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl(                    \
        mut_batch_view, batch_view, batch_view);                               \
    template void CompactBLAS<__VA_ARGS__>::xsub_copy_impl(                    \
        mut_batch_view, batch_view, batch_view);                               \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl(                    \
        mut_batch_view, batch_view, batch_view, batch_view);                   \
    template void CompactBLAS<__VA_ARGS__>::xsub_copy_impl(                    \
        mut_batch_view, batch_view, batch_view, batch_view);                   \
    template void CompactBLAS<__VA_ARGS__>::xadd_copy_impl(                    \
        mut_batch_view, batch_view, batch_view, batch_view, batch_view);       \
    template void CompactBLAS<__VA_ARGS__>::xsub_copy_impl(                    \
        mut_batch_view, batch_view, batch_view, batch_view, batch_view)

INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 16>);
INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 8>);
INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 4>);
INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 2>);
INSTANTIATE(stdx::simd_abi::scalar);

} // namespace koqkatoo::linalg::compact
