#include <gtest/gtest.h>
#include <algorithm>
#include <cassert>
#include <print>

#include <koqkatoo/linalg-compact/compact.hpp>

TEST(Compact, compress) {
    using namespace koqkatoo;
    using namespace koqkatoo::linalg::compact;
    const index_t ny            = 20;
    const index_t nr            = 6;
    static constexpr index_t VL = 4;
    using abi                   = stdx::simd_abi::deduce_t<real_t, VL>;
    using mat                   = CompactBLAS<abi>::matrix;
    mat A_in{{.depth = VL, .rows = nr, .cols = ny}},
        A_out{{.depth = VL, .rows = nr, .cols = ny}};
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
        0.0,  13.1, 18.2, 0.0,  // (11)
        0.0,  14.1, 19.2, 0.0,  // (12)
        0.0,  15.1, 0.0,  0.0,  // (13)
        0.0,  19.1, 0.0,  0.0,  // (14)
        0.0,  0.0,  0.0,  0.0,  // (15)
        0.0,  0.0,  0.0,  0.0,  // (16)
        0.0,  0.0,  0.0,  0.0,  // (17)
        0.0,  0.0,  0.0,  0.0,  // (18)
        0.0,  0.0,  0.0,  0.0,  // (19)
    };
    ASSERT_EQ(std::ranges::ssize(S_in_data), VL * ny);
    std::ranges::copy(S_in_data, S_in.data());

    auto nj = CompactBLAS<abi>::compress_masks<4>(
        A_in.batch(0), S_in.batch(0), A_out.batch(0), S_out.batch(0));

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

    EXPECT_EQ(nj, 15);
    EXPECT_TRUE(std::ranges::equal(S_expected_data, S_out));
}
