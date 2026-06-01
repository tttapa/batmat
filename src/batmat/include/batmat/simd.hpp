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
    // const_cast required because unchecked_load seems to lack a std::remove_const_t somewhere
    // in its metaprogramming
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
V masked_aligned_load(const typename V::value_type *p, typename V::mask_type m) {
    std::span<const typename V::value_type, V::size()> sp{p, V::size()};
    return std::datapar::unchecked_load<V>(sp, m, std::datapar::flag_aligned);
}

template <class V>
V masked_unaligned_load(const typename V::value_type *p, typename V::mask_type m) {
    std::span<const typename V::value_type, V::size()> sp{p, V::size()};
    return std::datapar::unchecked_load<V>(sp, m);
}

template <class V, int N>
V partial_load(const typename V::value_type *p) {
    std::span<const typename V::value_type, N> sp{p, N};
    return std::datapar::partial_load<V>(sp);
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

template <class V>
void masked_unaligned_store(V v, typename V::mask_type m, typename V::value_type *p) {
    std::span<typename V::value_type, V::size()> sp{p, V::size()};
    if constexpr (V::size() == 1) {
        if (m[0])
            std::datapar::unchecked_store(v, sp);
    } else {
        std::datapar::unchecked_store(v, sp, m);
    }
}

template <class V, int I, bool Value = true>
auto generate_mask() {
    return typename V::mask_type{[](int i) -> bool { return (i != I) ^ Value; }};
}

template <class V, bool Value = true>
auto generate_mask(int i) {
    return typename V::mask_type{[i](int j) -> bool { return (j != i) ^ Value; }};
}

template <class V, int N, bool Value = true>
auto generate_mask_until() {
    return typename V::mask_type{[](int i) -> bool { return (i >= N) ^ Value; }};
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

using std::datapar::reduce_count;
using std::datapar::select;

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
V masked_aligned_load(const typename V::value_type *p, typename V::mask_type m) {
    V v{};
    where(m, v).copy_from(p, stdx::vector_aligned);
    return v;
}

template <class V>
V masked_unaligned_load(const typename V::value_type *p, typename V::mask_type m) {
    V v{};
    where(m, v).copy_from(p, stdx::element_aligned);
    return v;
}

template <class V>
void masked_aligned_store(V v, typename V::mask_type m, typename V::value_type *p) {
    where(m, v).copy_to(p, stdx::vector_aligned);
}

template <class V>
void masked_unaligned_store(V v, typename V::mask_type m, typename V::value_type *p) {
    where(m, v).copy_to(p, stdx::element_aligned);
}

template <class V, size_t I, bool Value = true>
auto generate_mask() {
    typename V::mask_type m{!Value};
    m[I] = Value;
    return m;
}

template <class V, bool Value = true>
auto generate_mask(size_t i) {
    typename V::mask_type m{!Value};
    m[i] = Value;
    return m;
}

template <class V, size_t N, bool Value = true>
auto generate_mask_until() {
    typename V::mask_type m{Value};
    BATMAT_FULLY_UNROLLED_FOR (size_t i = N; i < V::size(); ++i)
        m[i] = !Value;
    return m;
}

template <class V, size_t N>
V partial_load(const typename V::value_type *p) {
    const auto mask = generate_mask_until<V, N>();
    return masked_unaligned_load<V>(p, mask);
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

auto reduce_count(auto v) { return popcount(v); }
auto select(auto cond, auto t, auto f) {
    where(cond, f) = t;
    return f;
}

} // namespace batmat::datapar

#endif

namespace batmat::datapar {

template <class V>
constexpr V from_values(auto... values) {
    alignas(simd_align<typename V::value_type, typename V::abi_type>::value)
        const typename V::value_type data[]{values...};
    return aligned_load<V>(data);
}

auto select(bool cond, auto t, auto f) { return cond ? t : f; }

} // namespace batmat::datapar
