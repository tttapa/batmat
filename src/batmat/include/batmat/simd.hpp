#pragma once

#include <batmat/config.hpp>
#include <batmat/unroll.h>

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
    std::span<typename V::value_type, V::size()> sp{const_cast<V::value_type *>(p), V::size()};
    return std::datapar::unchecked_load<V>(sp);
}

template <class V>
V aligned_load(const typename V::value_type *p) {
    std::span<typename V::value_type, V::size()> sp{const_cast<V::value_type *>(p), V::size()};
    return std::datapar::unchecked_load<V>(sp, std::datapar::flag_aligned);
}

template <class V>
void unaligned_store(V v, typename V::value_type *p) {
    std::datapar::unchecked_store(v, p, V::size());
}

template <class V>
void aligned_store(V v, typename V::value_type *p) {
    std::span<typename V::value_type, V::size()> sp{p, V::size()};
    std::datapar::unchecked_store(v, sp, std::datapar::flag_aligned);
}

template <class V>
void masked_aligned_store(V v, typename V::mask_type m, typename V::value_type *p) {
    std::span<typename V::value_type, V::size()> sp{p, V::size()};
    if constexpr (V::size() == 1) {
        if (m[0])
            std::datapar::unchecked_store(v, sp, std::datapar::flag_aligned);
    } else {
        std::datapar::unchecked_store(v, sp, m, std::datapar::flag_aligned);
    }
}

#if defined(__x86_64__) || defined(_M_X64)
template <class V>
auto to_intrin(V v) {
    return std::__detail::__to_x86_intrin(v);
}
#define BATMAT_HAVE_SIMD_TO_INTRIN 1
#endif

template <class Tp, class Abi>
using simd_size = std::remove_cvref_t<decltype(simd<Tp, Abi>::size)>;
template <class Tp, class Abi>
using simd_align = std::datapar::alignment<simd<Tp, Abi>>;
template <class T, class V>
using rebind_simd_t = deduced_simd<T, V::size()>;

template <class V>
auto hmax(V v) { // TODO
    using value_type = V::value_type;
    value_type m     = v[0];
    BATMAT_FULLY_UNROLLED_FOR (int i = 1; i < v.size(); ++i)
        m = std::max(v[i], m);
    return m;
}
template <class V>
auto hmin(V v) { // TODO
    using value_type = V::value_type;
    value_type m     = v[0];
    BATMAT_FULLY_UNROLLED_FOR (int i = 1; i < v.size(); ++i)
        m = std::min(v[i], m);
    return m;
}

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

using stdx::hmax;
using stdx::hmin;

} // namespace batmat::datapar

#endif

namespace batmat::datapar {

template <class V>
constexpr V from_values(auto... values) {
    alignas(simd_align<typename V::value_type, typename V::abi_type>::value)
        const typename V::value_type data[]{values...};
    return aligned_load<V>(data);
}

} // namespace batmat::datapar
