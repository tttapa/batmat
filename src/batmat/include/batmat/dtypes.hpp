#pragma once

#include <batmat/config.hpp>
#include <algorithm>
#include <array>
#include <functional>
#include <type_traits>

namespace batmat::types {

template <class... Ts>
struct Types {
    template <template <class...> class Template, class... Args>
    using into = Template<Args..., Ts...>;
};

template <class>
struct Head;
template <class T, class... Ts>
struct Head<Types<T, Ts...>> {
    using type = Types<T>;
};

template <class>
struct Tail;
template <class T, class... Ts>
struct Tail<Types<T, Ts...>> {
    using type = Types<Ts...>;
};

template <class...>
struct Concat;
template <>
struct Concat<> {
    using type = Types<>;
};
template <class... Ts>
struct Concat<Types<Ts...>> {
    using type = Types<Ts...>;
};
template <class... Ts1, class... Ts2, class... Rest>
struct Concat<Types<Ts1...>, Types<Ts2...>, Rest...> {
    using type = typename Concat<Types<Ts1..., Ts2...>, Rest...>::type;
};

template <template <class> class Func, class>
struct Map;
template <template <class> class Func, class... Ts>
struct Map<Func, Types<Ts...>> {
    using type = Types<Func<Ts>...>;
};
template <template <class> class Func, class List>
using Map_t = typename Map<Func, List>::type;

template <template <class> class Func, class>
struct FlatMap;
template <template <class> class Func, class... Ts>
struct FlatMap<Func, Types<Ts...>> {
    using type = typename Concat<Func<Ts>...>::type;
};
template <template <class> class Func, class List>
using FlatMap_t = typename FlatMap<Func, List>::type;

template <template <class> class Pred, class>
struct Filter;
template <template <class> class Pred, class... Ts>
struct Filter<Pred, Types<Ts...>> {
    using type = typename Concat<std::conditional_t<Pred<Ts>::value, Types<Ts>, Types<>>...>::type;
};
template <template <class> class Pred, class List>
using Filter_t = typename Filter<Pred, List>::type;

template <class T, index_t VL>
struct DTypeVectorLength {
    using dtype                 = T;
    using vl_t                  = std::integral_constant<index_t, VL>;
    static constexpr index_t vl = VL;
};

template <index_t VL>
struct VectorLengthIs {
    template <class T>
    using type = std::bool_constant<VL == T::vl>;
};

template <class DT>
struct DTypeIs {
    template <class T>
    using type = std::is_same<DT, typename T::dtype>;
};

template <index_t VL, class List>
using FilterVL = FlatMap_t<VectorLengthIs<VL>::template type, List>;

template <class Ts>
using GetDType = typename Ts::dtype;

#define BATMAT_PREFIX_COMMA(...) , __VA_ARGS__
/// @ref Types containing all supported dtypes.
using dtype_all = Tail<Types<void BATMAT_FOREACH_DTYPE(BATMAT_PREFIX_COMMA)>>::type;

/// @ref Types containing @ref DTypeVectorLength for all supported (dtype, VL) combinations.
#define BATMAT_INST_DT_VL(DT, VL) , DTypeVectorLength<DT, VL>
using dtype_vl_all = Tail<Types<void BATMAT_FOREACH_DTYPE_VL(BATMAT_INST_DT_VL)>>::type;
#undef BATMAT_INST_DT_VL

/// Array of supported vector lengths for a given dtype @p T.
template <class DT>
constexpr std::array vl_for_dtype = []<class... Dtvls>(Types<Dtvls...>) {
    return std::array<index_t, sizeof...(Dtvls)>{Dtvls::vl...};
}(Filter_t<DTypeIs<DT>::template type, dtype_vl_all>{});

/// Array of supported vector lengths for the default @ref real_t.
constexpr std::array vl_for_real_t = vl_for_dtype<real_t>;

/// @ref Types containing all supported dtypes for a given vector length @p VL.
template <index_t VL>
using dtypes_for_vl = Map_t<GetDType, Filter_t<VectorLengthIs<VL>::template type, dtype_vl_all>>;

/// @ref Types containing the given dtype and vector length combination, if supported.
template <class DT, index_t VL>
using lookup_dtype_vl = Filter_t<DTypeIs<DT>::template type, //
                                 Filter_t<VectorLengthIs<VL>::template type, dtype_vl_all>>;

/// Check if a given (dtype, VL) combination is supported.
template <class DT, index_t VL>
constexpr bool is_supported_dtype_vl = !std::is_same_v<lookup_dtype_vl<DT, VL>, Types<>>;

/// The smallest supported vector length for dtype @p DT that is greater than or equal to @p VL.
/// Returns 0 if no supported vector length is large enough.
template <class DT, index_t VL>
constexpr index_t vl_at_least = [] {
    if constexpr (is_supported_dtype_vl<DT, VL>) {
        return VL;
    } else {
        auto options = vl_for_dtype<DT>;
        std::ranges::sort(options, std::less{});
        for (auto v : options)
            if (v >= VL)
                return v;
        return index_t{0};
    }
}();

/// The largest supported vector length for dtype @p DT that is less than or equal to @p VL.
/// Returns 0 if no supported vector length is small enough.
template <class DT, index_t VL>
constexpr index_t vl_at_most = [] {
    if constexpr (is_supported_dtype_vl<DT, VL>) {
        return VL;
    } else {
        auto options = vl_for_dtype<DT>;
        std::ranges::sort(options, std::greater{});
        for (auto v : options)
            if (v <= VL)
                return v;
        return index_t{0};
    }
}();

/// @p VL if it is a supported vector length for dtype @p DT, otherwise the largest supported vector
/// length for @p DT.
template <class DT, index_t VL>
constexpr index_t vl_or_largest = [] {
    if constexpr (is_supported_dtype_vl<DT, VL>) {
        return VL;
    } else {
        auto options = vl_for_dtype<DT>;
        std::ranges::sort(options, std::greater{});
        return options.empty() ? index_t{0} : options.front();
    }
}();

/// Call a given function @p f for all supported (dtype, VL) combinations. @p f should be callable
/// with signature `void(DTypeVectorLength)`.
template <class F>
constexpr auto foreach_dtype_vl(F &&f) {
    return [&f]<class... Ts>(Types<Ts...>) { (f(Ts{}), ...); }(dtype_vl_all{});
}

} // namespace batmat::types
