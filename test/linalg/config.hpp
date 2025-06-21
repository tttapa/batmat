#pragma once

#include <batmat/config.hpp>
#include <gtest/gtest.h>

#include <experimental/simd>

namespace batmat::tests {

namespace stdx = std::experimental;

#if KOQKATOO_EXTENSIVE_TESTS

using Abis = ::testing::Types<stdx::simd_abi::scalar,              //
                              stdx::simd_abi::deduce_t<real_t, 2>, //
                              stdx::simd_abi::deduce_t<real_t, 4>, //
                              stdx::simd_abi::deduce_t<real_t, 8>, //
                              stdx::simd_abi::deduce_t<real_t, 16>>;

constexpr index_t sizes[] // NOLINT(*-c-arrays)
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 29, 63, 71, 293};

#else

using Abis = ::testing::Types<stdx::simd_abi::scalar,              //
                              stdx::simd_abi::deduce_t<real_t, 4>, //
                              stdx::simd_abi::deduce_t<real_t, 8>>;

constexpr index_t sizes[] // NOLINT(*-c-arrays)
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 71, 131, 311};

#endif

} // namespace batmat::tests
