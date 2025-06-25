#pragma once

#include <batmat/config.hpp>

#if BATMAT_WITH_GSI_HPC_SIMD

#include <cstddef>
#include <simd>

namespace batmat::datapar {

template <class Tp, class Abi>
using simd = std::datapar::basic_simd<Tp, Abi>;
template <class Tp, std::size_t Np>
using deduced_simd = std::datapar::simd<Tp, Np>;
template <class Tp, std::size_t Np>
using deduced_abi = deduced_simd<Tp, Np>::abi_type;
template <class Tp>
using scalar_abi = deduced_abi<Tp, 1>;

template <class V>
V unaligned_load(const typename V::value_type *p) {
    return std::datapar::unchecked_load<V>(p, V::size());
}

template <class V>
V aligned_load(const typename V::value_type *p) {
    return std::datapar::unchecked_load<V>(p, V::size(), std::datapar::flag_aligned);
}

template <class V>
void unaligned_store(V v, typename V::value_type *p) {
    std::datapar::unchecked_store(v, p, V::size());
}

template <class V>
void aligned_store(V v, typename V::value_type *p) {
    std::datapar::unchecked_store(v, p, V::size(), std::datapar::flag_aligned);
}

template <class V>
void masked_aligned_store(V v, typename V::mask_type m, typename V::value_type *p) {
    std::datapar::unchecked_store(v, p, V::size(), m, std::datapar::flag_aligned);
}

#if defined(__x86_64__) || defined(_M_X64)
template <class V>
auto to_intrin(V v) {
    return std::__detail::__to_x86_intrin(v);
}
#define BATMAT_HAVE_SIMD_TO_INTRIN 1
#endif

template <class Tp, class Abi>
using simd_size = decltype(simd<Tp, Abi>::size);
template <class Tp, class Abi>
using simd_align = std::simd_alignment<simd<Tp, Abi>>;

} // namespace batmat::datapar

#else

#include <experimental/simd>
#include <cstddef>

namespace batmat::datapar {
namespace stdx = std::experimental;

template <class Tp, class Abi>
using simd = stdx::simd<Tp, Abi>;
template <class Tp, std::size_t Np>
using deduced_abi = stdx::simd_abi::deduce_t<Tp, Np>;
template <class Tp, std::size_t Np>
using deduced_simd = simd<Tp, deduced_abi<Tp, Np>>;

template <class V>
V unaligned_load(const typename V::value_type *p) {
    return V{p, stdx::element_aligned};
}

template <class V>
V aligned_load(const typename V::value_type *p) {
    return V{p, stdx::vector_aligned};
}

template <class V>
void unaligned_store(V v, typename V::value_type *p) {
    v.copy_to(p, stdx::element_aligned);
}

template <class V>
void aligned_store(V v, typename V::value_type *p) {
    v.copy_to(p, stdx::vector_aligned);
}

template <class V>
void masked_aligned_store(V v, typename V::mask_type m, typename V::value_type *p) {
    where(m, v).copy_to(p, stdx::vector_aligned);
}

template <class V>
auto to_intrin(V v) {
    return static_cast<stdx::__intrinsic_type_t<V, v.size()>>(v);
}
#define BATMAT_HAVE_SIMD_TO_INTRIN 1

template <class Tp, class Abi>
using simd_size = stdx::simd_size<Tp, Abi>;
template <class Tp, class Abi>
using simd_align = stdx::memory_alignment<simd<Tp, Abi>>;
template <class T, class V>
using rebind_simd_t = stdx::rebind_simd_t<T, V>;
template <class Tp>
using scalar_abi = deduced_abi<Tp, 1>;

} // namespace batmat::datapar

#endif
