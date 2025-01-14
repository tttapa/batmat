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
class ShhUdDiagTest : public ::testing::Test {
  protected:
    using abi_t         = Abi;
    using simd          = stdx::simd<real_t, abi_t>;
    using simd_stride_t = stdx::simd_size<real_t, abi_t>;
    using Mat           = BatchedMatrix<real_t, index_t, simd_stride_t>;
    using View = BatchedMatrixView<const real_t, index_t, simd_stride_t,
                                   index_t, index_t>;
    using MutView =
        BatchedMatrixView<real_t, index_t, simd_stride_t, index_t, index_t>;
    using func_t        = void(MutView, MutView, View, PreferredBackend);
    using CompactBLAS_t = CompactBLAS<abi_t>;

    void SetUp() override { rng.seed(12345); }

    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};
    std::uniform_real_distribution<real_t> uni{-0.5, 0.5};

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
        Mat D{{
            .depth = 15,
            .rows  = 13,
            .cols  = 1,
        }};
        std::ranges::generate(K̃, [&] { return nrml(rng); });
        std::ranges::generate(A, [&] { return uni(rng); });
        std::ranges::generate(D, [&] { return uni(rng); });
        CompactBLAS_t::xsyrk(K̃, K, backend);
        K.view.add_to_diagonal(10);
        CompactBLAS_t::xcopy_L(K, K̃);
        CompactBLAS_t::xsyrk_schur_add(A, D, K̃, backend);
        CompactBLAS_t::xcopy_L(K, L);
        CompactBLAS_t::xpotrf(L, backend);
        CompactBLAS_t::xcopy_L(K̃, L̃);
        CompactBLAS_t::xpotrf(L̃, backend);

        // Perform the operation
        func(L, A, D, backend);

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < L.depth(); ++i)
            for (index_t j = 0; j < L.rows(); ++j)
                for (index_t k = 0; k <= j; ++k)
                    ASSERT_NEAR(L(i, j, k), L̃(i, j, k), ε)
                        << enum_name(backend) << "[" << n
                        << "]: SHHUD mismatch at (" << i << ", " << j << ", "
                        << k << ")";
    }
};

TYPED_TEST_SUITE(ShhUdDiagTest, koqkatoo::tests::Abis);

TYPED_TEST(ShhUdDiagTest, Shh) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(backend, i, []<class... Args>(Args... args) {
                return TestFixture::CompactBLAS_t::xshhud_diag(
                    std::forward<Args>(args)...);
            });
}
