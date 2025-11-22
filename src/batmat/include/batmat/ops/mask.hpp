#pragma once

#include <batmat/simd.hpp>

namespace batmat::ops::detail {

template <class T, class Abi>
struct mask_type {
    using type = typename datapar::simd<T, Abi>::mask_type;
};

template <class T, class AbiT>
using mask_type_t = typename mask_type<T, AbiT>::type;

/// Convert a SIMD mask to the appropriate intrinsic type. If @p M is not a mask type, its values
/// are compared to zero to create a mask.
/// @todo   GCC's <experimental/simd> does not allow casting from std::experimental::simd_mask to
///         intrinsic types, so I'm not using simd_mask for now.
template <class T, class AbiT, class M>
[[gnu::always_inline]] inline mask_type_t<T, AbiT> convert_mask(M mask) {
#if BATMAT_WITH_GSI_HPC_SIMD // TODO
    return mask != M{0};
#else
    return (mask != 0).__cvt();
#endif
}

template <class T, class Abi>
[[gnu::always_inline]] inline auto compare_ge_0(datapar::simd<T, Abi> x) {
    return x >= datapar::simd<T, Abi>{};
}

} // namespace batmat::ops::detail

#if defined(__AVX512F__)
#include <batmat/ops/avx-512/mask.hpp>
#elif defined(__AVX2__)
#include <batmat/ops/avx2/mask.hpp>
#endif
