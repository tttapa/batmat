#include <gtest/gtest.h>

#include <koqkatoo/config.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/rotate.hpp>

namespace stdx = std::experimental;
using koqkatoo::real_t;
using koqkatoo::linalg::compact::micro_kernels::rotl;
using koqkatoo::linalg::compact::micro_kernels::rotr;
using koqkatoo::linalg::compact::micro_kernels::shiftl;
using koqkatoo::linalg::compact::micro_kernels::shiftr;

TEST(Rotate, rotl4) {
    using abi  = stdx::simd_abi::deduce_t<real_t, 4>;
    using simd = stdx::simd<real_t, abi>;
    simd x;
    x[0]   = 1;
    x[1]   = 2;
    x[2]   = 3;
    x[3]   = 4;
    simd y = rotl<1>(x);
    simd y_expected;
    y_expected[0] = 2;
    y_expected[1] = 3;
    y_expected[2] = 4;
    y_expected[3] = 1;
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotr4) {
    using abi  = stdx::simd_abi::deduce_t<real_t, 4>;
    using simd = stdx::simd<real_t, abi>;
    simd x;
    x[0]   = 1;
    x[1]   = 2;
    x[2]   = 3;
    x[3]   = 4;
    simd y = rotr<1>(x);
    simd y_expected;
    y_expected[0] = 4;
    y_expected[1] = 1;
    y_expected[2] = 2;
    y_expected[3] = 3;
    EXPECT_TRUE(all_of(y == y_expected));
}

using TestVectorLengths =
    ::testing::Types<stdx::simd_abi::scalar,              //
                     stdx::simd_abi::deduce_t<real_t, 1>, //
                     stdx::simd_abi::deduce_t<real_t, 2>, //
                     stdx::simd_abi::deduce_t<real_t, 4>, //
                     stdx::simd_abi::deduce_t<real_t, 8>, //
                     stdx::simd_abi::deduce_t<real_t, 16> //
                     >;

template <typename Abi>
class RotateTest : public ::testing::Test {};
TYPED_TEST_SUITE(RotateTest, TestVectorLengths);

TYPED_TEST(RotateTest, rotl) {
    using simd         = stdx::simd<real_t, TypeParam>;
    constexpr size_t N = simd::size();
    simd x;
    for (size_t i = 0; i < N; ++i)
        x[i] = static_cast<double>(i + 1);

    simd y = rotl<1>(x);
    simd y_expected;
    for (size_t i = 0; i < N - 1; ++i)
        y_expected[i] = x[i + 1];
    y_expected[N - 1] = x[0];

    EXPECT_TRUE(all_of(y == y_expected));
}

TYPED_TEST(RotateTest, rotr) {
    using simd         = stdx::simd<real_t, TypeParam>;
    constexpr size_t N = simd::size();
    simd x;
    for (size_t i = 0; i < N; ++i)
        x[i] = static_cast<double>(i + 1);

    simd y = rotr<1>(x);
    simd y_expected;
    y_expected[0] = x[N - 1];
    for (size_t i = 1; i < N; ++i)
        y_expected[i] = x[i - 1];

    EXPECT_TRUE(all_of(y == y_expected));
}

TYPED_TEST(RotateTest, shiftl) {
    using simd         = stdx::simd<real_t, TypeParam>;
    constexpr size_t N = simd::size();
    simd x;
    for (size_t i = 0; i < N; ++i)
        x[i] = static_cast<double>(i + 1);

    simd y = shiftl<1>(x);
    simd y_expected;
    for (size_t i = 0; i < N - 1; ++i)
        y_expected[i] = x[i + 1];
    y_expected[N - 1] = 0.0;

    EXPECT_TRUE(all_of(y == y_expected));
}

TYPED_TEST(RotateTest, shiftr) {
    using simd         = stdx::simd<real_t, TypeParam>;
    constexpr size_t N = simd::size();
    simd x;
    for (size_t i = 0; i < N; ++i)
        x[i] = static_cast<double>(i + 1);

    simd y = shiftr<1>(x);
    simd y_expected;
    y_expected[0] = 0.0;
    for (size_t i = 1; i < N; ++i)
        y_expected[i] = x[i - 1];

    EXPECT_TRUE(all_of(y == y_expected));
}
