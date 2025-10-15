#include <gtest/gtest.h>
#include <algorithm>
#include <cassert>
#include <print>
#include <random>

#include <batmat/linalg/compress.hpp>
#include <batmat/linalg/gemm-diag.hpp>
#include <batmat/linalg/gemm.hpp>
#include <batmat/linalg/uview.hpp>

#include "eigen-matchers.hpp"

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
        0.0,  0.1,  0.2,  0.0,  // (0)
        1.0,  0.0,  0.0,  0.0,  // (1)
        0.0,  0.0,  0.0,  2.3,  // (2)
        0.0,  3.1,  3.2,  0.0,  // (3)
        4.0,  4.1,  0.0,  0.0,  // (4)
        5.0,  5.1,  0.0,  5.3,  // (5)
        0.0,  6.1,  6.2,  0.0,  // (6)
        7.0,  7.1,  0.0,  0.0,  // (7)
        0.0,  8.1,  0.0,  0.0,  // (8)
        0.0,  9.1,  9.2,  0.0,  // (9)
        10.0, 10.1, 0.0,  0.0,  // (10)
        0.0,  11.1, 0.0,  0.0,  // (11)
        0.0,  12.1, 12.2, 0.0,  // (12)
        13.0, 13.1, 0.0,  0.0,  // (13)
        0.0,  14.1, 0.0,  0.0,  // (14)
        0.0,  15.1, 15.2, 15.3, // (15)
        16.0, 0.0,  16.2, 0.0,  // (16)
        0.0,  0.0,  0.0,  0.0,  // (17)
        0.0,  0.0,  18.2, 0.0,  // (18)
        0.0,  19.1, 19.2, 19.3, // (19)
        0.0,  0.0,  0.0,  0.0,  // (20)
        0.0,  0.0,  0.0,  0.0,  // (21)
        0.0,  0.0,  0.0,  0.0,  // (22)
        0.0,  0.0,  0.0,  0.0,  // (23)
        24.0, 24.1, 24.2, 24.3, // (24)
    };
    static constexpr const real_t S_expected_data[]{
        1.0,  0.1,  0.2,  2.3,  // (0)
        4.0,  3.1,  3.2,  5.3,  // (1)
        5.0,  4.1,  6.2,  0.0,  // (2)
        7.0,  5.1,  9.2,  0.0,  // (3)
        10.0, 6.1,  0.0,  0.0,  // (4)
        0.0,  7.1,  0.0,  0.0,  // (5)
        0.0,  8.1,  12.2, 0.0,  // (6)
        13.0, 9.1,  0.0,  0.0,  // (7)
        0.0,  10.1, 0.0,  0.0,  // (8)
        0.0,  11.1, 15.2, 15.3, // (9)
        16.0, 12.1, 16.2, 19.3, // (10)
        24.0, 13.1, 18.2, 24.3, // (11)
        0.0,  14.1, 19.2, 0.0,  // (12)
        0.0,  15.1, 24.2, 0.0,  // (13)
        0.0,  19.1, 0.0,  0.0,  // (14)
        0.0,  24.1, 0.0,  0.0,  // (15)
        0.0,  0.0,  0.0,  0.0,  // (16)
        0.0,  0.0,  0.0,  0.0,  // (17)
        0.0,  0.0,  0.0,  0.0,  // (18)
        0.0,  0.0,  0.0,  0.0,  // (19)
        0.0,  0.0,  0.0,  0.0,  // (20)
        0.0,  0.0,  0.0,  0.0,  // (21)
        0.0,  0.0,  0.0,  0.0,  // (22)
        0.0,  0.0,  0.0,  0.0,  // (23)
        0.0,  0.0,  0.0,  0.0,  // (24)
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
        0.0,  0.1,  0.2,  0.0,  0.0, 0.0, 0.0, 0.0, // (0)
        1.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (1)
        0.0,  0.0,  0.0,  2.3,  0.0, 0.0, 0.0, 0.0, // (2)
        0.0,  3.1,  3.2,  0.0,  0.0, 0.0, 0.0, 0.0, // (3)
        4.0,  4.1,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (4)
        5.0,  5.1,  0.0,  5.3,  0.0, 0.0, 0.0, 0.0, // (5)
        0.0,  6.1,  6.2,  0.0,  0.0, 0.0, 0.0, 0.0, // (6)
        7.0,  7.1,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (7)
        0.0,  8.1,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (8)
        0.0,  9.1,  9.2,  0.0,  0.0, 0.0, 0.0, 0.0, // (9)
        10.0, 10.1, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (10)
        0.0,  11.1, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (11)
        0.0,  12.1, 12.2, 0.0,  0.0, 0.0, 0.0, 0.0, // (12)
        13.0, 13.1, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (13)
        0.0,  14.1, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (14)
        0.0,  15.1, 15.2, 15.3, 0.0, 0.0, 0.0, 0.0, // (15)
        16.0, 0.0,  16.2, 0.0,  0.0, 0.0, 0.0, 0.0, // (16)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (17)
        0.0,  0.0,  18.2, 0.0,  0.0, 0.0, 0.0, 0.0, // (18)
        0.0,  19.1, 19.2, 19.3, 0.0, 0.0, 0.0, 0.0, // (19)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (20)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (21)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (22)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (23)
        24.0, 24.1, 24.2, 24.3, 0.0, 0.0, 0.0, 0.0, // (24)
    };
    static constexpr const real_t S_expected_data[]{
        1.0,  0.1,  0.2,  2.3,  0.0, 0.0, 0.0, 0.0, // (0)
        4.0,  3.1,  3.2,  5.3,  0.0, 0.0, 0.0, 0.0, // (1)
        5.0,  4.1,  6.2,  0.0,  0.0, 0.0, 0.0, 0.0, // (2)
        7.0,  5.1,  9.2,  0.0,  0.0, 0.0, 0.0, 0.0, // (3)
        10.0, 6.1,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (4)
        0.0,  7.1,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (5)
        0.0,  8.1,  12.2, 0.0,  0.0, 0.0, 0.0, 0.0, // (6)
        13.0, 9.1,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (7)
        0.0,  10.1, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (8)
        0.0,  11.1, 15.2, 15.3, 0.0, 0.0, 0.0, 0.0, // (9)
        16.0, 12.1, 16.2, 19.3, 0.0, 0.0, 0.0, 0.0, // (10)
        24.0, 13.1, 18.2, 24.3, 0.0, 0.0, 0.0, 0.0, // (11)
        0.0,  14.1, 19.2, 0.0,  0.0, 0.0, 0.0, 0.0, // (12)
        0.0,  15.1, 24.2, 0.0,  0.0, 0.0, 0.0, 0.0, // (13)
        0.0,  19.1, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (14)
        0.0,  24.1, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (15)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (16)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (17)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (18)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (19)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (20)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (21)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (22)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (23)
        0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, // (24)
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
        0.0,  0.1,  // (0)
        1.0,  0.0,  // (1)
        0.0,  0.0,  // (2)
        0.0,  3.1,  // (3)
        4.0,  4.1,  // (4)
        5.0,  5.1,  // (5)
        0.0,  6.1,  // (6)
        7.0,  7.1,  // (7)
        0.0,  8.1,  // (8)
        0.0,  9.1,  // (9)
        10.0, 10.1, // (10)
        0.0,  11.1, // (11)
        0.0,  12.1, // (12)
        13.0, 13.1, // (13)
        0.0,  14.1, // (14)
        0.0,  15.1, // (15)
        16.0, 0.0,  // (16)
        0.0,  0.0,  // (17)
        0.0,  0.0,  // (18)
        0.0,  19.1, // (19)
        0.0,  0.0,  // (20)
        0.0,  0.0,  // (21)
        0.0,  0.0,  // (22)
        0.0,  0.0,  // (23)
        24.0, 24.1, // (24)
    };
    static constexpr const real_t S_expected_data[]{
        1.0,  0.1,  // (0)
        4.0,  3.1,  // (1)
        5.0,  4.1,  // (2)
        7.0,  5.1,  // (3)
        10.0, 6.1,  // (4)
        0.0,  7.1,  // (5)
        0.0,  8.1,  // (6)
        13.0, 9.1,  // (7)
        0.0,  10.1, // (8)
        0.0,  11.1, // (9)
        16.0, 12.1, // (10)
        24.0, 13.1, // (11)
        0.0,  14.1, // (12)
        0.0,  15.1, // (13)
        0.0,  19.1, // (14)
        0.0,  24.1, // (15)
        0.0,  0.0,  // (16)
        0.0,  0.0,  // (17)
        0.0,  0.0,  // (18)
        0.0,  0.0,  // (19)
        0.0,  0.0,  // (20)
        0.0,  0.0,  // (21)
        0.0,  0.0,  // (22)
        0.0,  0.0,  // (23)
        0.0,  0.0,  // (24)
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

using I4 = std::integral_constant<batmat::index_t, 4>;

TEST(linalg, compressSqrt) {
    using namespace batmat;
    using namespace batmat::linalg;
    using Mat = batmat::matrix::Matrix<real_t, index_t, I4, I4>;
    Mat A{{.rows = 19, .cols = 231}}, Acompact{{.rows = 19, .cols = 231}};
    Mat S{{.rows = 231, .cols = 1}};
    std::mt19937_64 rng{12345};
    std::uniform_real_distribution<real_t> dist{0, 1};
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

    const auto ε = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t l = 0; l < ASAᵀ.depth(); ++l) {
        EXPECT_THAT(as_eigen(ASAᵀcompact(l)), EigenAlmostEqual(as_eigen(ASAᵀ(l)), ε))
            << "at layer " << l;
    }
}

TEST(linalg, compressSqrtInplace) {
    using namespace batmat;
    using namespace batmat::linalg;
    using Mat = batmat::matrix::Matrix<real_t, index_t, I4, I4>;
    Mat A{{.rows = 19, .cols = 231}};
    Mat S{{.rows = 231, .cols = 1}};
    std::mt19937_64 rng{12345};
    std::uniform_real_distribution<real_t> dist{0, 1};
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

    const auto ε = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t l = 0; l < ASAᵀ.depth(); ++l) {
        EXPECT_THAT(as_eigen(ASAᵀcompact(l)), EigenAlmostEqual(as_eigen(ASAᵀ(l)), ε))
            << "at layer " << l;
    }
}
