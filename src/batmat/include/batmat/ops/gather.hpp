#pragma once

#include <batmat/ops/mask.hpp>
#include <batmat/simd.hpp>
#include <concepts>
#include <type_traits>

namespace batmat::ops ::detail {

template <class T, class AbiT, class I, class AbiI>
[[gnu::always_inline]] inline datapar::simd<T, AbiT>
gather(datapar::simd<T, AbiT> src, typename datapar::simd<T, AbiT>::mask_type mask,
       datapar::simd<I, AbiI> vindex, const T *base_addr) {
    return datapar::simd<T, AbiT>{[=](auto i) { return mask[i] ? base_addr[vindex[i]] : src[i]; }};
}

template <class T1, class T2>
concept same_size_but_different_ints =
    std::integral<T1> && std::integral<T2> && sizeof(T1) == sizeof(T2) && !std::same_as<T1, T2>;

template <std::integral I>
auto convert_int(I i) {
    if constexpr (std::is_signed_v<I> && sizeof(I) == sizeof(int32_t)) {
        return static_cast<int32_t>(i);
    } else if constexpr (std::is_unsigned_v<I> && sizeof(I) == sizeof(int32_t)) {
        return static_cast<uint32_t>(i);
    } else if constexpr (std::is_signed_v<I> && sizeof(I) == sizeof(int64_t)) {
        return static_cast<int64_t>(i);
    } else if constexpr (std::is_unsigned_v<I> && sizeof(I) == sizeof(int64_t)) {
        return static_cast<uint64_t>(i);
    }
}
template <std::integral I>
using convert_int_t = decltype(convert_int(std::declval<I>()));

} // namespace batmat::ops::detail

#if defined(__AVX512F__)
#include <batmat/ops/avx-512/gather.hpp>
#elif defined(__AVX2__)
#include <batmat/ops/avx2/gather.hpp>
#endif

namespace batmat::ops {

template <class T, class AbiT, class I, class AbiI, class M>
[[gnu::always_inline]] inline datapar::simd<T, AbiT> gather(const T *p, datapar::simd<I, AbiI> idx,
                                                            M mask) {
    using simd      = datapar::simd<T, AbiT>;
    using isimd     = datapar::rebind_simd_t<detail::convert_int_t<I>, simd>;
    using msimd     = datapar::rebind_simd_t<detail::convert_int_t<typename M::value_type>, simd>;
    auto mask_      = detail::convert_mask<T, AbiT>(simd_cast<msimd>(mask));
    const auto idx_ = simd_cast<isimd>(idx);
    return detail::gather(simd{}, mask_, idx_, p);
}

} // namespace batmat::ops
