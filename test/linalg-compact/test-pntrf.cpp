#include <koqkatoo/linalg-compact/compact.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <random>

#include "config.hpp"

using namespace koqkatoo::linalg::compact;
using koqkatoo::index_t;
using koqkatoo::real_t;

const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));

template <class Abi>
class PntrfTest : public ::testing::Test {
  protected:
    using abi_t         = Abi;
    using simd          = stdx::simd<real_t, abi_t>;
    using simd_stride_t = stdx::simd_size<real_t, abi_t>;
    using Mat           = BatchedMatrix<real_t, index_t, simd_stride_t>;
    using View = BatchedMatrixView<const real_t, index_t, simd_stride_t,
                                   index_t, index_t>;
    using MutView =
        BatchedMatrixView<real_t, index_t, simd_stride_t, index_t, index_t>;
    using func_t        = void(MutView, View);
    using CompactBLAS_t = CompactBLAS<abi_t>;

    void SetUp() override { rng.seed(12345); }

    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};
    std::bernoulli_distribution brnl{0.5};

    void RunTestTrsm(index_t n, func_t func) {
        using std::copysign;
        const auto backend = PreferredBackend::Reference;
        Mat L{{.depth = 15, .rows = n, .cols = n}};
        Mat B{{.depth = 15, .rows = n + 13, .cols = n}};
        Mat AB{{.depth = 15, .rows = 2 * n + 13, .cols = n}};
        Mat D{{.depth = 15, .rows = n, .cols = 1}};
        Mat S{{.depth = 15, .rows = n, .cols = 1}};
        std::ranges::generate(L, [&] { return uni(rng); });
        std::ranges::generate(B, [&] { return uni(rng); });
        std::ranges::generate(D, [&] { return brnl(rng) ? +1 : -1; });
        std::ranges::transform(D, S.begin(),
                               [](real_t d) { return copysign(0, d); });
        L.view.add_to_diagonal(10);
        for (index_t c = 1; c < L.cols(); ++c) // Make A lower triangular
            L.view.block(0, c, c, 1).set_constant(0);
        CompactBLAS_t::xsyrk_schur(L, D, AB.view.top_rows(n), backend);
        AB.view.bottom_rows(n + 13) = B;

        // Perform the operation
        func(AB, S);

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < L.depth(); ++i)
            for (index_t j = 0; j < L.rows(); ++j)
                for (index_t k = 0; k < L.cols(); ++k)
                    ASSERT_NEAR(AB(i, j, k), L(i, j, k), ε)
                        << "[" << n << "]: PNTRF mismatch at (" << i << ", "
                        << j << ", " << k << ")";

        Mat LD = L;
        for (index_t k = 0; k < LD.depth(); ++k)
            for (index_t c = 0; c < LD.cols(); ++c)
                for (index_t r = c; r < LD.rows(); ++r)
                    LD(k, r, c) *= D(k, c, 0);
        // Perform the reference operation for comparison
        CompactBLAS_t::xtrsm_RLTN(LD, B, backend);

        for (index_t i = 0; i < L.depth(); ++i)
            for (index_t j = 0; j < B.rows(); ++j)
                for (index_t k = 0; k < B.cols(); ++k)
                    ASSERT_NEAR(AB(i, n + j, k), B(i, j, k), ε)
                        << "[" << n << "]: PNTRF TRSM mismatch at (" << i
                        << ", " << j << ", " << k << ")";
    }
};

TYPED_TEST_SUITE(PntrfTest, koqkatoo::tests::Abis);

TYPED_TEST(PntrfTest, PntrfTrsm) {
    for (index_t i : koqkatoo::tests::sizes)
        this->RunTestTrsm(i, []<class... Args>(Args... args) {
            return TestFixture::CompactBLAS_t::xpntrf(
                std::forward<Args>(args)...);
        });
}
