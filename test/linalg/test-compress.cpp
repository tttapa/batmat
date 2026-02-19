#include <gtest/gtest.h>
#include <algorithm>
#include <cassert>
#include <print>
#include <random>

#include <batmat/dtypes.hpp>
#include <batmat/linalg/compress.hpp>
#include <batmat/linalg/gemm-diag.hpp>
#include <batmat/linalg/gemm.hpp>
#include <batmat/linalg/uview.hpp>
#include <guanaqo/demangled-typename.hpp>

#include "eigen-matchers.hpp"

constexpr batmat::real_t operator""_r(long double x) { return static_cast<batmat::real_t>(x); }

TEST(linalg, compress) {
    using namespace batmat;
    using namespace batmat::linalg;
    const index_t ny            = 25;
    const index_t nr            = 6;
    static constexpr index_t VL = 4;
    using abi                   = datapar::deduced_abi<real_t, VL>;
    using types                 = simd_view_types<real_t, abi>;
    using mat                   = types::matrix<>;
    mat A_in{{.depth = VL, .rows = nr, .cols = ny}}, A_out{{.depth = VL, .rows = nr, .cols = ny}};
    mat S_in{{.depth = VL, .rows = ny}}, S_out{{.depth = VL, .rows = ny}};

    static constexpr const real_t S_in_data[]{
        0.0_r,  0.1_r,  0.2_r,  0.0_r,  // (0)
        1.0_r,  0.0_r,  0.0_r,  0.0_r,  // (1)
        0.0_r,  0.0_r,  0.0_r,  2.3_r,  // (2)
        0.0_r,  3.1_r,  3.2_r,  0.0_r,  // (3)
        4.0_r,  4.1_r,  0.0_r,  0.0_r,  // (4)
        5.0_r,  5.1_r,  0.0_r,  5.3_r,  // (5)
        0.0_r,  6.1_r,  6.2_r,  0.0_r,  // (6)
        7.0_r,  7.1_r,  0.0_r,  0.0_r,  // (7)
        0.0_r,  8.1_r,  0.0_r,  0.0_r,  // (8)
        0.0_r,  9.1_r,  9.2_r,  0.0_r,  // (9)
        10.0_r, 10.1_r, 0.0_r,  0.0_r,  // (10)
        0.0_r,  11.1_r, 0.0_r,  0.0_r,  // (11)
        0.0_r,  12.1_r, 12.2_r, 0.0_r,  // (12)
        13.0_r, 13.1_r, 0.0_r,  0.0_r,  // (13)
        0.0_r,  14.1_r, 0.0_r,  0.0_r,  // (14)
        0.0_r,  15.1_r, 15.2_r, 15.3_r, // (15)
        16.0_r, 0.0_r,  16.2_r, 0.0_r,  // (16)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (17)
        0.0_r,  0.0_r,  18.2_r, 0.0_r,  // (18)
        0.0_r,  19.1_r, 19.2_r, 19.3_r, // (19)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (20)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (21)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (22)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (23)
        24.0_r, 24.1_r, 24.2_r, 24.3_r, // (24)
    };
    static constexpr const real_t S_expected_data[]{
        1.0_r,  0.1_r,  0.2_r,  2.3_r,  // (0)
        4.0_r,  3.1_r,  3.2_r,  5.3_r,  // (1)
        5.0_r,  4.1_r,  6.2_r,  0.0_r,  // (2)
        7.0_r,  5.1_r,  9.2_r,  0.0_r,  // (3)
        10.0_r, 6.1_r,  0.0_r,  0.0_r,  // (4)
        0.0_r,  7.1_r,  0.0_r,  0.0_r,  // (5)
        0.0_r,  8.1_r,  12.2_r, 0.0_r,  // (6)
        13.0_r, 9.1_r,  0.0_r,  0.0_r,  // (7)
        0.0_r,  10.1_r, 0.0_r,  0.0_r,  // (8)
        0.0_r,  11.1_r, 15.2_r, 15.3_r, // (9)
        16.0_r, 12.1_r, 16.2_r, 19.3_r, // (10)
        24.0_r, 13.1_r, 18.2_r, 24.3_r, // (11)
        0.0_r,  14.1_r, 19.2_r, 0.0_r,  // (12)
        0.0_r,  15.1_r, 24.2_r, 0.0_r,  // (13)
        0.0_r,  19.1_r, 0.0_r,  0.0_r,  // (14)
        0.0_r,  24.1_r, 0.0_r,  0.0_r,  // (15)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (16)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (17)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (18)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (19)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (20)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (21)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (22)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (23)
        0.0_r,  0.0_r,  0.0_r,  0.0_r,  // (24)
    };
    ASSERT_EQ(std::ranges::ssize(S_in_data), VL * ny);
    std::ranges::copy(S_in_data, S_in.data());

    auto nj  = compress_masks<4>(A_in.batch(0), S_in.batch(0), A_out.batch(0), S_out.batch(0));
    auto njc = compress_masks_count<4>(S_in.batch(0));

    EXPECT_EQ(nj, njc);

    for (index_t i = 0; i < ny; ++i) {
        std::print("        ");
        for (index_t l = 0; l < VL; ++l)
            std::print("{:5.1f},", S_in(l, i, 0));
        std::print("  // ({})\n", i);
    }
    std::println();

    for (index_t i = 0; i < ny; ++i) {
        std::print("        ");
        for (index_t l = 0; l < VL; ++l)
            std::print("{:5.1f},", S_out(l, i, 0));
        std::print("  // ({})\n", i);
    }

    EXPECT_EQ(nj, 16);
    EXPECT_TRUE(std::ranges::equal(S_expected_data, S_out));

    // TODO: check A_out
}

TEST(linalg, compress8) {
    using namespace batmat;
    using namespace batmat::linalg;
    const index_t ny            = 25;
    const index_t nr            = 6;
    static constexpr index_t VL = 8;
    using abi                   = datapar::deduced_abi<real_t, VL>;
    using types                 = simd_view_types<real_t, abi>;
    using mat                   = types::matrix<>;
    mat A_in{{.depth = VL, .rows = nr, .cols = ny}}, A_out{{.depth = VL, .rows = nr, .cols = ny}};
    mat S_in{{.depth = VL, .rows = ny}}, S_out{{.depth = VL, .rows = ny}};

    static constexpr const real_t S_in_data[]{
        0.0_r, 0.0_r,  0.1_r,  0.0_r, 0.2_r,  0.0_r, 0.0_r, 0.0_r,  // (0)
        0.0_r, 1.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (1)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 2.3_r,  // (2)
        0.0_r, 0.0_r,  3.1_r,  0.0_r, 3.2_r,  0.0_r, 0.0_r, 0.0_r,  // (3)
        0.0_r, 4.0_r,  4.1_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (4)
        0.0_r, 5.0_r,  5.1_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 5.3_r,  // (5)
        0.0_r, 0.0_r,  6.1_r,  0.0_r, 6.2_r,  0.0_r, 0.0_r, 0.0_r,  // (6)
        0.0_r, 7.0_r,  7.1_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (7)
        0.0_r, 0.0_r,  8.1_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (8)
        0.0_r, 0.0_r,  9.1_r,  0.0_r, 9.2_r,  0.0_r, 0.0_r, 0.0_r,  // (9)
        0.0_r, 10.0_r, 10.1_r, 0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (10)
        0.0_r, 0.0_r,  11.1_r, 0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (11)
        0.0_r, 0.0_r,  12.1_r, 0.0_r, 12.2_r, 0.0_r, 0.0_r, 0.0_r,  // (12)
        0.0_r, 13.0_r, 13.1_r, 0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (13)
        0.0_r, 0.0_r,  14.1_r, 0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (14)
        0.0_r, 0.0_r,  15.1_r, 0.0_r, 15.2_r, 0.0_r, 0.0_r, 15.3_r, // (15)
        0.0_r, 16.0_r, 0.0_r,  0.0_r, 16.2_r, 0.0_r, 0.0_r, 0.0_r,  // (16)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (17)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 18.2_r, 0.0_r, 0.0_r, 0.0_r,  // (18)
        0.0_r, 0.0_r,  19.1_r, 0.0_r, 19.2_r, 0.0_r, 0.0_r, 19.3_r, // (19)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (20)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (21)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (22)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (23)
        0.0_r, 24.0_r, 24.1_r, 0.0_r, 24.2_r, 0.0_r, 0.0_r, 24.3_r, // (24)
    };
    static constexpr const real_t S_expected_data[]{
        0.0_r, 1.0_r,  0.1_r,  0.0_r, 0.2_r,  0.0_r, 0.0_r, 2.3_r,  // (0)
        0.0_r, 4.0_r,  3.1_r,  0.0_r, 3.2_r,  0.0_r, 0.0_r, 5.3_r,  // (1)
        0.0_r, 5.0_r,  4.1_r,  0.0_r, 6.2_r,  0.0_r, 0.0_r, 0.0_r,  // (2)
        0.0_r, 7.0_r,  5.1_r,  0.0_r, 9.2_r,  0.0_r, 0.0_r, 0.0_r,  // (3)
        0.0_r, 10.0_r, 6.1_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (4)
        0.0_r, 0.0_r,  7.1_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (5)
        0.0_r, 0.0_r,  8.1_r,  0.0_r, 12.2_r, 0.0_r, 0.0_r, 0.0_r,  // (6)
        0.0_r, 13.0_r, 9.1_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (7)
        0.0_r, 0.0_r,  10.1_r, 0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (8)
        0.0_r, 0.0_r,  11.1_r, 0.0_r, 15.2_r, 0.0_r, 0.0_r, 15.3_r, // (9)
        0.0_r, 16.0_r, 12.1_r, 0.0_r, 16.2_r, 0.0_r, 0.0_r, 19.3_r, // (10)
        0.0_r, 24.0_r, 13.1_r, 0.0_r, 18.2_r, 0.0_r, 0.0_r, 24.3_r, // (11)
        0.0_r, 0.0_r,  14.1_r, 0.0_r, 19.2_r, 0.0_r, 0.0_r, 0.0_r,  // (12)
        0.0_r, 0.0_r,  15.1_r, 0.0_r, 24.2_r, 0.0_r, 0.0_r, 0.0_r,  // (13)
        0.0_r, 0.0_r,  19.1_r, 0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (14)
        0.0_r, 0.0_r,  24.1_r, 0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (15)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (16)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (17)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (18)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (19)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (20)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (21)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (22)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (23)
        0.0_r, 0.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r, 0.0_r, 0.0_r,  // (24)
    };
    ASSERT_EQ(std::ranges::ssize(S_in_data), VL * ny);
    std::ranges::copy(S_in_data, S_in.data());

    auto nj  = compress_masks<4>(A_in.batch(0), S_in.batch(0), A_out.batch(0), S_out.batch(0));
    auto njc = compress_masks_count<4>(S_in.batch(0));

    EXPECT_EQ(nj, njc);

    for (index_t i = 0; i < ny; ++i) {
        std::print("        ");
        for (index_t l = 0; l < VL; ++l)
            std::print("{:5.1f},", S_in(l, i, 0));
        std::print("  // ({})\n", i);
    }
    std::println();

    for (index_t i = 0; i < ny; ++i) {
        std::print("        ");
        for (index_t l = 0; l < VL; ++l)
            std::print("{:5.1f},", S_out(l, i, 0));
        std::print("  // ({})\n", i);
    }

    EXPECT_EQ(nj, 16);
    EXPECT_TRUE(std::ranges::equal(S_expected_data, S_out));

    // TODO: check A_out
}

TEST(linalg, compress2) {
    using namespace batmat;
    using namespace batmat::linalg;
    const index_t ny            = 25;
    const index_t nr            = 6;
    static constexpr index_t VL = 2;
    using abi                   = datapar::deduced_abi<real_t, VL>;
    using types                 = simd_view_types<real_t, abi>;
    using mat                   = types::matrix<>;
    mat A_in{{.depth = VL, .rows = nr, .cols = ny}}, A_out{{.depth = VL, .rows = nr, .cols = ny}};
    mat S_in{{.depth = VL, .rows = ny}}, S_out{{.depth = VL, .rows = ny}};

    static constexpr const real_t S_in_data[]{
        0.0_r,  0.1_r,  // (0)
        1.0_r,  0.0_r,  // (1)
        0.0_r,  0.0_r,  // (2)
        0.0_r,  3.1_r,  // (3)
        4.0_r,  4.1_r,  // (4)
        5.0_r,  5.1_r,  // (5)
        0.0_r,  6.1_r,  // (6)
        7.0_r,  7.1_r,  // (7)
        0.0_r,  8.1_r,  // (8)
        0.0_r,  9.1_r,  // (9)
        10.0_r, 10.1_r, // (10)
        0.0_r,  11.1_r, // (11)
        0.0_r,  12.1_r, // (12)
        13.0_r, 13.1_r, // (13)
        0.0_r,  14.1_r, // (14)
        0.0_r,  15.1_r, // (15)
        16.0_r, 0.0_r,  // (16)
        0.0_r,  0.0_r,  // (17)
        0.0_r,  0.0_r,  // (18)
        0.0_r,  19.1_r, // (19)
        0.0_r,  0.0_r,  // (20)
        0.0_r,  0.0_r,  // (21)
        0.0_r,  0.0_r,  // (22)
        0.0_r,  0.0_r,  // (23)
        24.0_r, 24.1_r, // (24)
    };
    static constexpr const real_t S_expected_data[]{
        1.0_r,  0.1_r,  // (0)
        4.0_r,  3.1_r,  // (1)
        5.0_r,  4.1_r,  // (2)
        7.0_r,  5.1_r,  // (3)
        10.0_r, 6.1_r,  // (4)
        0.0_r,  7.1_r,  // (5)
        0.0_r,  8.1_r,  // (6)
        13.0_r, 9.1_r,  // (7)
        0.0_r,  10.1_r, // (8)
        0.0_r,  11.1_r, // (9)
        16.0_r, 12.1_r, // (10)
        24.0_r, 13.1_r, // (11)
        0.0_r,  14.1_r, // (12)
        0.0_r,  15.1_r, // (13)
        0.0_r,  19.1_r, // (14)
        0.0_r,  24.1_r, // (15)
        0.0_r,  0.0_r,  // (16)
        0.0_r,  0.0_r,  // (17)
        0.0_r,  0.0_r,  // (18)
        0.0_r,  0.0_r,  // (19)
        0.0_r,  0.0_r,  // (20)
        0.0_r,  0.0_r,  // (21)
        0.0_r,  0.0_r,  // (22)
        0.0_r,  0.0_r,  // (23)
        0.0_r,  0.0_r,  // (24)
    };
    ASSERT_EQ(std::ranges::ssize(S_in_data), VL * ny);
    std::ranges::copy(S_in_data, S_in.data());

    auto nj  = compress_masks<4>(A_in.batch(0), S_in.batch(0), A_out.batch(0), S_out.batch(0));
    auto njc = compress_masks_count<4>(S_in.batch(0));

    EXPECT_EQ(nj, njc);

    for (index_t i = 0; i < ny; ++i) {
        std::print("        ");
        for (index_t l = 0; l < VL; ++l)
            std::print("{:5.1f},", S_in(l, i, 0));
        std::print("  // ({})\n", i);
    }
    std::println();

    for (index_t i = 0; i < ny; ++i) {
        std::print("        ");
        for (index_t l = 0; l < VL; ++l)
            std::print("{:5.1f},", S_out(l, i, 0));
        std::print("  // ({})\n", i);
    }

    EXPECT_EQ(nj, 16);
    EXPECT_TRUE(std::ranges::equal(S_expected_data, S_out));

    // TODO: check A_out
}

template <class Config>
struct CompressSqrtTest : ::testing::Test {};

template <class T, class V>
void test_compress_sqrt() {
    using namespace batmat;
    using namespace batmat::linalg;
    using Mat = batmat::matrix::Matrix<T, index_t, V, V>;
    Mat A{{.rows = 19, .cols = 231}}, Acompact{{.rows = 19, .cols = 231}};
    Mat S{{.rows = 231, .cols = 1}};
    std::mt19937_64 rng{12345};
    std::uniform_real_distribution<T> dist{0, 1};
    std::ranges::generate(A.begin(), A.end(), [&]() { return dist(rng); });
    std::ranges::generate(S.begin(), S.end(), [&]() { return dist(rng); });
    std::bernoulli_distribution bdist{0.333};
    for (auto &s : S)
        if (bdist(rng))
            s = 0;
    Mat ASAᵀ{{.rows = 19, .cols = 19}}, ASAᵀcompact{{.rows = 19, .cols = 19}};
    syrk_diag_add(A, tril(ASAᵀ), S);
    auto m_compact = compress_masks_sqrt(A, S, Acompact);
    syrk_add(Acompact.left_cols(m_compact), tril(ASAᵀcompact));

    const auto ε = std::numeric_limits<T>::epsilon() * 100;
    for (index_t l = 0; l < ASAᵀ.depth(); ++l) {
        EXPECT_THAT(as_eigen(ASAᵀcompact(l)), EigenAlmostEqual(as_eigen(ASAᵀ(l)), ε))
            << "at layer " << l;
    }
}

template <class T, class V>
void test_compress_sqrt_inplace() {
    using namespace batmat;
    using namespace batmat::linalg;
    using Mat = batmat::matrix::Matrix<T, index_t, V, V>;
    Mat A{{.rows = 19, .cols = 231}};
    Mat S{{.rows = 231, .cols = 1}};
    std::mt19937_64 rng{12345};
    std::uniform_real_distribution<T> dist{0, 1};
    std::ranges::generate(A.begin(), A.end(), [&]() { return dist(rng); });
    std::ranges::generate(S.begin(), S.end(), [&]() { return dist(rng); });
    std::bernoulli_distribution bdist{0.333};
    for (auto &s : S)
        if (bdist(rng))
            s = 0;
    Mat ASAᵀ{{.rows = 19, .cols = 19}}, ASAᵀcompact{{.rows = 19, .cols = 19}};
    syrk_diag_add(A, tril(ASAᵀ), S);
    auto m_compact = compress_masks_sqrt(A, S, A);
    syrk_add(A.left_cols(m_compact), tril(ASAᵀcompact));

    const auto ε = std::numeric_limits<T>::epsilon() * 100;
    for (index_t l = 0; l < ASAᵀ.depth(); ++l) {
        EXPECT_THAT(as_eigen(ASAᵀcompact(l)), EigenAlmostEqual(as_eigen(ASAᵀ(l)), ε))
            << "at layer " << l;
    }
}

TYPED_TEST_SUITE(CompressSqrtTest, batmat::types::dtype_vl_all::into<::testing::Types>);

TYPED_TEST(CompressSqrtTest, compressSqrt) {
    test_compress_sqrt<typename TypeParam::dtype, typename TypeParam::vl_t>();
}
TYPED_TEST(CompressSqrtTest, compressSqrtInplace) {
    test_compress_sqrt_inplace<typename TypeParam::dtype, typename TypeParam::vl_t>();
}
