#include <koqkatoo/linalg-compact/compact.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <random>

#include "config.hpp"

using namespace koqkatoo::linalg::compact;
using koqkatoo::index_t;
using koqkatoo::real_t;

const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));

template <class Abi>
class ShhTest : public ::testing::Test {
  protected:
    using abi_t         = Abi;
    using simd          = stdx::simd<real_t, abi_t>;
    using simd_stride_t = stdx::simd_size<real_t, abi_t>;
    using Mat           = BatchedMatrix<real_t, index_t, simd_stride_t>;
    using View = BatchedMatrixView<const real_t, index_t, simd_stride_t,
                                   index_t, index_t>;
    using MutView =
        BatchedMatrixView<real_t, index_t, simd_stride_t, index_t, index_t>;
    using func_t        = void(MutView, MutView, PreferredBackend);
    using CompactBLAS_t = CompactBLAS<abi_t>;

    void SetUp() override { rng.seed(12345); }

    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    void RunTest(PreferredBackend backend, index_t n, func_t func) {
        Mat K̃{{
            .depth = 15,
            .rows  = n,
            .cols  = n,
        }};
        Mat K{{
            .depth = 15,
            .rows  = n,
            .cols  = n,
        }};
        Mat L̃{{
            .depth = 15,
            .rows  = n,
            .cols  = n,
        }};
        Mat L{{
            .depth = 15,
            .rows  = n,
            .cols  = n,
        }};
        Mat A{{
            .depth = 15,
            .rows  = n,
            .cols  = 13,
        }};
        std::ranges::generate(K, [&] { return nrml(rng); });
        std::ranges::generate(A, [&] { return nrml(rng); });
        CompactBLAS_t::xsyrk(K, K̃, backend);
        K̃.view.add_to_diagonal(1);
        CompactBLAS_t::xcopy(K̃, K);
        CompactBLAS_t::xsyrk_add(A, K, backend);
        CompactBLAS_t::xcopy(K, L);
        CompactBLAS_t::xpotrf(L, backend);
        CompactBLAS_t::xcopy(K̃, L̃);
        CompactBLAS_t::xpotrf(L̃, backend);

        // Perform the operation
        func(L, A, backend);

        // Flip signs if necessary
        for (index_t i = 0; i < L.depth(); ++i)
            for (index_t k = 0; k < L.cols(); ++k)
                if (L(i, k, k) < 0)
                    for (index_t j = k; j < L.rows(); ++j)
                        L(i, j, k) *= -1;

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < L.depth(); ++i)
            for (index_t j = 0; j < L.rows(); ++j)
                for (index_t k = 0; k <= j; ++k)
                    ASSERT_NEAR(L(i, j, k), L̃(i, j, k), ε)
                        << enum_name(backend) << "[" << n
                        << "]: SHH mismatch at (" << i << ", " << j << ", " << k
                        << ")";
    }
};

TYPED_TEST_SUITE(ShhTest, koqkatoo::tests::Abis);

TYPED_TEST(ShhTest, Shh) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(backend, i, []<class... Args>(Args... args) {
                return TestFixture::CompactBLAS_t::xshh(
                    std::forward<Args>(args)...);
            });
}
