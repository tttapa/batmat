#include <gtest/gtest.h>
#include <algorithm>
#include <cassert>
#include <print>

#include <koqkatoo/assume.hpp>
#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/gather.hpp>
#include <koqkatoo/unroll.h>

namespace koqkatoo::linalg::compact {

template <class Abi, index_t N = 8>
[[gnu::used]] index_t
compress_masks(typename CompactBLAS<Abi>::single_batch_view A_in,
               typename CompactBLAS<Abi>::single_batch_view S_in,
               typename CompactBLAS<Abi>::mut_single_batch_view A_out,
               typename CompactBLAS<Abi>::mut_single_batch_view S_out) {
    using koqkatoo::linalg::compact::micro_kernels::gather;
    assert(A_in.rows() == A_out.rows());
    assert(A_in.cols() == A_out.cols());
    assert(A_in.depth() == A_out.depth());
    assert(A_in.depth() == S_in.depth());
    assert(A_in.depth() == S_out.depth());
    assert(A_in.cols() == S_in.rows());
    assert(A_out.cols() == S_out.rows());
    const auto C             = A_in.cols();
    const auto R             = A_in.rows();
    static constexpr auto VL = stdx::simd_size_v<real_t, Abi>;
    if (C == 0)
        return 0;
    KOQKATOO_ASSUME(R > 0);
    using simd = typename CompactBLAS<Abi>::simd;
    using isimd =
        stdx::simd<index_t, stdx::simd_abi::deduce_t<index_t, simd::size()>>;
    isimd hist[N]{};
    index_t j = 0;
    static const isimd iota{[](auto i) { return i; }};

    const auto commit_no_shift = [&](auto h_commit) {
        h_commit -= 1;
        {
            const auto bs       = static_cast<index_t>(S_in.batch_size());
            const isimd offsets = h_commit * bs + iota;
            auto gather_S       = gather<real_t, VL>(&S_in(0, 0, 0), offsets);
            gather_S.copy_to(&S_out(0, j, 0), stdx::vector_aligned);
        }
        for (index_t r = 0; r < R; ++r) {
            const auto bs       = static_cast<index_t>(A_in.batch_size());
            const isimd offsets = h_commit * bs * A_in.outer_stride() + iota;
            auto gather_A       = gather<real_t, VL>(&A_in(0, 0, r), offsets);
            gather_A.copy_to(&A_out(0, j, r), stdx::vector_aligned);
        }
        ++j;
    };
    const auto commit = [&] [[gnu::always_inline]] () {
        const isimd h = hist[0];
        KOQKATOO_FULLY_UNROLLED_FOR (index_t k = 1; k < N; ++k)
            hist[k - 1] = hist[k];
        hist[N - 1] = 0;
        commit_no_shift(h);
    };

    for (index_t c = 0; c < C; ++c) {
        const simd Sc{&S_in(0, c, 0), stdx::vector_aligned};
        auto Sc_msk = Sc != 0;
        KOQKATOO_FULLY_UNROLLED_FOR (auto &h : hist) {
            if (none_of(Sc_msk))
                break;
            const auto msk = (h == 0) && Sc_msk.__cvt();
            where(msk, h)  = isimd{c + 1};
            Sc_msk         = Sc_msk && (!msk).__cvt();
        }
        // Masks of all ones can already be written to memory
        while (none_of(hist[0] == 0) || (c + 1 == C && any_of(hist[0] != 0)))
            commit();
        // If there are still bits set in the mask.
        if (any_of(Sc_msk)) {
            // Check if there's an empty slot (always at the end)
            if (any_of(hist[N - 1] != 0))
                // If not, commit the first slot to make room.
                commit();
            // Find the first empty slot, and add the remaining bits.
            KOQKATOO_FULLY_UNROLLED_FOR (auto &h : hist)
                if (all_of(h == 0))
                    where(Sc_msk.__cvt(), h) = isimd{c + 1};
        }
        // Invariant: first registers in the buffer contain fewest zeros
        KOQKATOO_FULLY_UNROLLED_FOR (index_t i = 1; i < N; ++i)
            assert(popcount(hist[i] != 0) <= popcount(hist[i - 1] != 0));
    }
    if (any_of(hist[0] != 0))
        commit_no_shift(hist[0]);
    return j;
}

} // namespace koqkatoo::linalg::compact

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

    auto nj = compress_masks<abi, 4>(A_in.batch(0), S_in.batch(0),
                                     A_out.batch(0), S_out.batch(0));

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
