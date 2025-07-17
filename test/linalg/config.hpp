#pragma once

#include <batmat/config.hpp>
#include <batmat/matrix/layout.hpp>
#include <batmat/simd.hpp>
#include <gtest/gtest.h>

namespace batmat::tests {

#if BATMAT_EXTENSIVE_TESTS

using Abis = ::testing::Types<datapar::deduced_abi<real_t, 1>, //
                              datapar::deduced_abi<real_t, 2>, //
                              datapar::deduced_abi<real_t, 4>, //
                              datapar::deduced_abi<real_t, 8>, //
                              datapar::deduced_abi<real_t, 16>>;

constexpr index_t sizes[] // NOLINT(*-c-arrays)
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 29, 63, 71, 293, 311};

#else

using Abis = ::testing::Types<datapar::deduced_abi<real_t, 1>, //
                              datapar::deduced_abi<real_t, 4>, //
                              datapar::deduced_abi<real_t, 8>>;

constexpr index_t sizes[] // NOLINT(*-c-arrays)
    {0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 17, 71, 131, 311};

#endif

using batmat::matrix::StorageOrder;
using enum StorageOrder;

template <class T, index_t N, StorageOrder... Orders>
struct TestConfig {
    using value_type = T;
    using batch_size = std::integral_constant<index_t, N>;
    static constexpr StorageOrder orders[sizeof...(Orders)]{Orders...};
};

template <class...>
struct CatTypes;

template <class... T1>
struct CatTypes<::testing::Types<T1...>> {
    using type = ::testing::Types<T1...>;
};
template <class... T1, class... T2, class... Others>
struct CatTypes<::testing::Types<T1...>, ::testing::Types<T2...>, Others...> {
    using type = typename CatTypes<::testing::Types<T1..., T2...>, Others...>::type;
};

template <class T, index_t N>
using OrderConfigs3 = ::testing::Types<
    TestConfig<T, N, ColMajor, ColMajor, ColMajor>,
#if BATMAT_EXTENSIVE_TESTS
    TestConfig<T, N, ColMajor, ColMajor, RowMajor>, TestConfig<T, N, ColMajor, RowMajor, ColMajor>,
    TestConfig<T, N, ColMajor, RowMajor, RowMajor>, TestConfig<T, N, RowMajor, ColMajor, ColMajor>,
    TestConfig<T, N, RowMajor, ColMajor, RowMajor>, TestConfig<T, N, RowMajor, RowMajor, ColMajor>,
#endif
    TestConfig<T, N, RowMajor, RowMajor, RowMajor>>;

template <class T, index_t N>
using OrderConfigs2 =
    ::testing::Types<TestConfig<T, N, ColMajor, ColMajor>,
#if BATMAT_EXTENSIVE_TESTS
                     TestConfig<T, N, ColMajor, ColMajor>, TestConfig<T, N, ColMajor, RowMajor>,
                     TestConfig<T, N, RowMajor, ColMajor>,
#endif
                     TestConfig<T, N, RowMajor, RowMajor>>;

template <class T, index_t N>
using OrderConfigs1 = ::testing::Types<TestConfig<T, N, ColMajor>, TestConfig<T, N, RowMajor>>;

template <template <class T, index_t N> class OrderConfigs>
#if BATMAT_WITH_SINGLE
using TestConfigs = typename CatTypes<OrderConfigs<double, 1>, OrderConfigs<double, 4>,
                                      OrderConfigs<float, 1>, OrderConfigs<float, 8>>::type;
#else
using TestConfigs = typename CatTypes<OrderConfigs<double, 1>, OrderConfigs<double, 4>>::type;
#endif

} // namespace batmat::tests
