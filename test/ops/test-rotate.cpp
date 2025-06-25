#include <gtest/gtest.h>

#include <batmat/config.hpp>
#include <batmat/ops/rotate.hpp>
#include <print>

using batmat::real_t;
using batmat::ops::rot;
using batmat::ops::rotl;
using batmat::ops::rotr;
using batmat::ops::shiftl;
using batmat::ops::shiftr;

TEST(Rotate, rotl4) {
    using simd = batmat::datapar::deduced_simd<real_t, 4>;
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

TEST(Rotate, rotl4var) {
    using simd = batmat::datapar::deduced_simd<real_t, 4>;
    simd x;
    x[0]   = 1;
    x[1]   = 2;
    x[2]   = 3;
    x[3]   = 4;
    simd y = rot(x, -1);
    simd y_expected;
    y_expected[0] = 2;
    y_expected[1] = 3;
    y_expected[2] = 4;
    y_expected[3] = 1;
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotl8var) {
    using simd = batmat::datapar::deduced_simd<real_t, 8>;
    simd x;
    x[0]   = 1;
    x[1]   = 2;
    x[2]   = 3;
    x[3]   = 4;
    x[4]   = 5;
    x[5]   = 6;
    x[6]   = 7;
    x[7]   = 8;
    simd y = rot(x, -1);
    simd y_expected;
    y_expected[0] = 2;
    y_expected[1] = 3;
    y_expected[2] = 4;
    y_expected[3] = 5;
    y_expected[4] = 6;
    y_expected[5] = 7;
    y_expected[6] = 8;
    y_expected[7] = 1;
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotl8var2) {
    using simd = batmat::datapar::deduced_simd<real_t, 8>;
    simd x;
    x[0]   = 1;
    x[1]   = 2;
    x[2]   = 3;
    x[3]   = 4;
    x[4]   = 5;
    x[5]   = 6;
    x[6]   = 7;
    x[7]   = 8;
    simd y = rot(x, -1);
    simd y_expected;
    y_expected[0] = 2;
    y_expected[1] = 3;
    y_expected[2] = 4;
    y_expected[3] = 5;
    y_expected[4] = 6;
    y_expected[5] = 7;
    y_expected[6] = 8;
    y_expected[7] = 1;
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotr4) {
    using simd = batmat::datapar::deduced_simd<real_t, 4>;
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

TEST(Rotate, rotr4var) {
    using simd = batmat::datapar::deduced_simd<real_t, 4>;
    simd x;
    x[0]   = 1;
    x[1]   = 2;
    x[2]   = 3;
    x[3]   = 4;
    simd y = rot(x, 1);
    simd y_expected;
    y_expected[0] = 4;
    y_expected[1] = 1;
    y_expected[2] = 2;
    y_expected[3] = 3;
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotr8var) {
    using simd = batmat::datapar::deduced_simd<real_t, 8>;
    simd x;
    x[0]   = 1;
    x[1]   = 2;
    x[2]   = 3;
    x[3]   = 4;
    x[4]   = 5;
    x[5]   = 6;
    x[6]   = 7;
    x[7]   = 8;
    simd y = rot(x, 1);
    simd y_expected;
    y_expected[0] = 8;
    y_expected[1] = 1;
    y_expected[2] = 2;
    y_expected[3] = 3;
    y_expected[4] = 4;
    y_expected[5] = 5;
    y_expected[6] = 6;
    y_expected[7] = 7;
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotr8var2) {
    using simd = batmat::datapar::deduced_simd<real_t, 8>;
    simd x;
    x[0]   = 1;
    x[1]   = 2;
    x[2]   = 3;
    x[3]   = 4;
    x[4]   = 5;
    x[5]   = 6;
    x[6]   = 7;
    x[7]   = 8;
    simd y = rot(x, 2);
    simd y_expected;
    y_expected[0] = 7;
    y_expected[1] = 8;
    y_expected[2] = 1;
    y_expected[3] = 2;
    y_expected[4] = 3;
    y_expected[5] = 4;
    y_expected[6] = 5;
    y_expected[7] = 6;
    EXPECT_TRUE(all_of(y == y_expected));
}

using TestAbis = ::testing::Types<batmat::datapar::deduced_abi<real_t, 1>, //
                                  batmat::datapar::deduced_abi<real_t, 2>, //
                                  batmat::datapar::deduced_abi<real_t, 4>, //
                                  batmat::datapar::deduced_abi<real_t, 8>, //
                                  batmat::datapar::deduced_abi<real_t, 16> //
                                  >;

template <typename Abi, size_t Shift>
struct TestParams {
    using abi                     = Abi;
    static constexpr size_t shift = Shift;
};

template <typename Abi, typename Indices>
struct ExpandShiftCounts;
template <typename Abi, size_t... Indices>
struct ExpandShiftCounts<Abi, std::index_sequence<Indices...>> {
    using type = ::testing::Types<TestParams<Abi, Indices>...>;
};

template <typename... Abis>
struct ConcatTestTypes;
template <typename... R, typename... L>
struct ConcatTestTypes<::testing::Types<L...>, ::testing::Types<R...>> {
    using type = ::testing::Types<L..., R...>;
};

template <typename... Abis>
struct GenerateTestTypes;
template <>
struct GenerateTestTypes<::testing::Types<>> {
    using type = ::testing::Types<>;
};
template <typename FirstAbi, typename... RestAbis>
struct GenerateTestTypes<::testing::Types<FirstAbi, RestAbis...>> {
    using expanded = ExpandShiftCounts<
        FirstAbi, std::make_index_sequence<batmat::datapar::simd<real_t, FirstAbi>::size()>>;
    using type = typename ConcatTestTypes<
        typename expanded::type,
        typename GenerateTestTypes<::testing::Types<RestAbis...>>::type>::type;
};

// Instantiate all test cases with SIMD ABIs and shift values
using TestCases = typename GenerateTestTypes<TestAbis>::type;

// Define the test fixture with both ABI and shift count
template <typename Param>
class RotateTest : public ::testing::Test {};

TYPED_TEST_SUITE(RotateTest, TestCases);

TYPED_TEST(RotateTest, rotl) {
    using simd         = batmat::datapar::simd<real_t, typename TypeParam::abi>;
    constexpr size_t N = simd::size();
    constexpr size_t S = TypeParam::shift;
    simd x;
    for (size_t i = 0; i < N; ++i)
        x[i] = static_cast<double>(i + 1);

    simd y = rotl<S>(x);
    simd y_expected;
    for (size_t i = 0; i < N; ++i)
        y_expected[i] = x[(i + S) % N];

    EXPECT_TRUE(all_of(y == y_expected));
}

TYPED_TEST(RotateTest, rotr) {
    using simd         = batmat::datapar::simd<real_t, typename TypeParam::abi>;
    constexpr size_t N = simd::size();
    constexpr size_t S = TypeParam::shift;
    simd x;
    for (size_t i = 0; i < N; ++i)
        x[i] = static_cast<double>(i + 1);

    simd y = rotr<S>(x);
    simd y_expected;
    for (size_t i = 0; i < N; ++i)
        y_expected[i] = x[(i + N - S) % N];

    EXPECT_TRUE(all_of(y == y_expected));
}

TYPED_TEST(RotateTest, shiftl) {
    using simd         = batmat::datapar::simd<real_t, typename TypeParam::abi>;
    constexpr size_t N = simd::size();
    constexpr size_t S = TypeParam::shift;
    simd x;
    for (size_t i = 0; i < N; ++i)
        x[i] = static_cast<double>(i + 1);

    simd y = shiftl<S>(x);
    simd y_expected;
    for (size_t i = 0; i < N; ++i)
        y_expected[i] = (i + S < N) ? x[i + S] : 0.0;

    for (size_t i = 0; i < N; ++i) {
        std::println("y[{}] = {},    expected {}", i, (real_t)y[i], (real_t)y_expected[i]);
    }

    EXPECT_TRUE(all_of(y == y_expected));
}

TYPED_TEST(RotateTest, shiftr) {
    using simd         = batmat::datapar::simd<real_t, typename TypeParam::abi>;
    constexpr size_t N = simd::size();
    constexpr size_t S = TypeParam::shift;
    simd x;
    for (size_t i = 0; i < N; ++i)
        x[i] = static_cast<double>(i + 1);

    simd y = shiftr<S>(x);
    simd y_expected;
    for (size_t i = 0; i < N; ++i)
        y_expected[i] = (i >= S) ? x[i - S] : 0.0;

    EXPECT_TRUE(all_of(y == y_expected));
}
