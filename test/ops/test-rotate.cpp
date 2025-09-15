#include <gtest/gtest.h>

#include <batmat/config.hpp>
#include <batmat/ops/rotate.hpp>
#include <batmat/simd.hpp>

using batmat::real_t;
using batmat::ops::rot;
using batmat::ops::rotl;
using batmat::ops::rotr;
using batmat::ops::shiftl;
using batmat::ops::shiftr;

TEST(Rotate, rotl4) {
    using simd      = batmat::datapar::deduced_simd<real_t, 4>;
    simd x          = batmat::datapar::from_values<simd>(1., 2., 3., 4.);
    simd y_expected = batmat::datapar::from_values<simd>(2., 3., 4., 1.);
    simd y          = rotl<1>(x);
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotl4var) {
    using simd      = batmat::datapar::deduced_simd<real_t, 4>;
    simd x          = batmat::datapar::from_values<simd>(1., 2., 3., 4.);
    simd y_expected = batmat::datapar::from_values<simd>(2., 3., 4., 1.);
    simd y          = rot(x, -1);
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotl8var) {
    using simd      = batmat::datapar::deduced_simd<real_t, 8>;
    simd x          = batmat::datapar::from_values<simd>(1., 2., 3., 4., 5., 6., 7., 8.);
    simd y_expected = batmat::datapar::from_values<simd>(2., 3., 4., 5., 6., 7., 8., 1.);
    simd y          = rot(x, -1);
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotl8var2) {
    using simd      = batmat::datapar::deduced_simd<real_t, 8>;
    simd x          = batmat::datapar::from_values<simd>(1., 2., 3., 4., 5., 6., 7., 8.);
    simd y_expected = batmat::datapar::from_values<simd>(2., 3., 4., 5., 6., 7., 8., 1.);
    simd y          = rot(x, -1);
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotr4) {
    using simd      = batmat::datapar::deduced_simd<real_t, 4>;
    simd x          = batmat::datapar::from_values<simd>(1., 2., 3., 4.);
    simd y_expected = batmat::datapar::from_values<simd>(4., 1., 2., 3.);
    simd y          = rotr<1>(x);
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotr4var) {
    using simd      = batmat::datapar::deduced_simd<real_t, 4>;
    simd x          = batmat::datapar::from_values<simd>(1., 2., 3., 4.);
    simd y_expected = batmat::datapar::from_values<simd>(4., 1., 2., 3.);
    simd y          = rot(x, 1);
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotr8var) {
    using simd      = batmat::datapar::deduced_simd<real_t, 8>;
    simd x          = batmat::datapar::from_values<simd>(1., 2., 3., 4., 5., 6., 7., 8.);
    simd y_expected = batmat::datapar::from_values<simd>(8., 1., 2., 3., 4., 5., 6., 7.);
    simd y          = rot(x, 1);
    EXPECT_TRUE(all_of(y == y_expected));
}

TEST(Rotate, rotr8var2) {
    using simd      = batmat::datapar::deduced_simd<real_t, 8>;
    simd x          = batmat::datapar::from_values<simd>(1., 2., 3., 4., 5., 6., 7., 8.);
    simd y_expected = batmat::datapar::from_values<simd>(7., 8., 1., 2., 3., 4., 5., 6.);
    simd y          = rot(x, 2);
    EXPECT_TRUE(all_of(y == y_expected));
}

using TestAbis = ::testing::Types<batmat::datapar::deduced_abi<real_t, 1>, //
                                  batmat::datapar::deduced_abi<real_t, 2>, //
                                  batmat::datapar::deduced_abi<real_t, 4>, //
                                  batmat::datapar::deduced_abi<real_t, 8>, //
                                  batmat::datapar::deduced_abi<real_t, 16> //
                                  >;

template <typename Abi, int Shift>
struct TestParams {
    using abi                  = Abi;
    static constexpr int shift = Shift;
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
    using simd      = batmat::datapar::simd<real_t, typename TypeParam::abi>;
    constexpr int N = simd::size();
    constexpr int S = TypeParam::shift;
    simd x{[&](int i) { return static_cast<real_t>(i + 1); }};

    simd y = rotl<S>(x);
    simd y_expected{[&](int i) { return x[(i + S) % N]; }};

    EXPECT_TRUE(all_of(y == y_expected));
}

TYPED_TEST(RotateTest, rotr) {
    using simd      = batmat::datapar::simd<real_t, typename TypeParam::abi>;
    constexpr int N = simd::size();
    constexpr int S = TypeParam::shift;
    simd x{[&](int i) { return static_cast<real_t>(i + 1); }};

    simd y = rotr<S>(x);
    simd y_expected{[&](int i) { return x[(i + N - S) % N]; }};

    EXPECT_TRUE(all_of(y == y_expected));
}

TYPED_TEST(RotateTest, shiftl) {
    using simd      = batmat::datapar::simd<real_t, typename TypeParam::abi>;
    constexpr int N = simd::size();
    constexpr int S = TypeParam::shift;
    simd x{[&](int i) { return static_cast<real_t>(i + 1); }};

    simd y = shiftl<S>(x);
    simd y_expected{[&](int i) { return (i + S < N) ? x[i + S] : 0.0; }};

    EXPECT_TRUE(all_of(y == y_expected));
}

TYPED_TEST(RotateTest, shiftr) {
    using simd      = batmat::datapar::simd<real_t, typename TypeParam::abi>;
    constexpr int S = TypeParam::shift;
    simd x{[&](int i) { return static_cast<real_t>(i + 1); }};

    simd y = shiftr<S>(x);
    simd y_expected{[&](int i) { return (i >= S) ? x[i - S] : 0.0; }};

    EXPECT_TRUE(all_of(y == y_expected));
}
