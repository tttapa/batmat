#pragma once

#include <batmat/config.hpp>
#include <batmat/dtypes.hpp>
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

template <class T, index_t N>
using OrderConfigs3 = types::Types<
    TestConfig<T, N, ColMajor, ColMajor, ColMajor>,
#if BATMAT_EXTENSIVE_TESTS
    TestConfig<T, N, ColMajor, ColMajor, RowMajor>, TestConfig<T, N, ColMajor, RowMajor, ColMajor>,
    TestConfig<T, N, ColMajor, RowMajor, RowMajor>, TestConfig<T, N, RowMajor, ColMajor, ColMajor>,
    TestConfig<T, N, RowMajor, ColMajor, RowMajor>, TestConfig<T, N, RowMajor, RowMajor, ColMajor>,
#endif
    TestConfig<T, N, RowMajor, RowMajor, RowMajor>>;

template <class T, index_t N>
using OrderConfigs2 =
    types::Types<TestConfig<T, N, ColMajor, ColMajor>,
#if BATMAT_EXTENSIVE_TESTS
                 TestConfig<T, N, ColMajor, ColMajor>, TestConfig<T, N, ColMajor, RowMajor>,
                 TestConfig<T, N, RowMajor, ColMajor>,
#endif
                 TestConfig<T, N, RowMajor, RowMajor>>;

template <class T, index_t N>
using OrderConfigs1 = types::Types<TestConfig<T, N, ColMajor>, TestConfig<T, N, RowMajor>>;

/// Meta-function that unpacks the dtypes and vector lengths of the given DtVls types,
/// and applies the given ConfigsForDtVl template to them.
template <template <class T, index_t N> class ConfigsForDtVl>
struct UncurryDTypeVL {
    template <class DtVl>
    using type = ConfigsForDtVl<typename DtVl::dtype, DtVl::vl>;
};

/// Apply the given @p ConfigsForDtVl template to all supported combinations of dtypes and vector
/// lengths and convert the result to a Google Test type list.
template <template <class T, index_t N> class ConfigsForDtVl>
using TestConfigs = types::FlatMap_t<UncurryDTypeVL<ConfigsForDtVl>::template type,
                                     types::dtype_vl_all>::template into<::testing::Types>;

} // namespace batmat::tests
