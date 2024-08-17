#include <koqkatoo/linalg-compact/compact.hpp>
#include <gtest/gtest.h>
#include <functional>
#include <limits>
#include <random>

#include "config.hpp"

using namespace koqkatoo::linalg::compact;
using koqkatoo::index_t;
using koqkatoo::real_t;

const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));

template <class Abi>
class TrsmTest : public ::testing::Test {
  protected:
    using abi_t         = Abi;
    using simd          = stdx::simd<real_t, abi_t>;
    using simd_stride_t = stdx::simd_size<real_t, abi_t>;
    using Mat           = BatchedMatrix<real_t, index_t, simd_stride_t>;
    using View    = BatchedMatrixView<const real_t, index_t, simd_stride_t>;
    using MutView = BatchedMatrixView<real_t, index_t, simd_stride_t>;
    using func_t  = void(View, MutView, PreferredBackend);
    using naive_func_t  = std::function<void(View, MutView)>;
    using CompactBLAS_t = CompactBLAS<abi_t>;

    void SetUp() override { rng.seed(12345); }

    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    void RunTest(PreferredBackend backend, index_t n, func_t func,
                 naive_func_t naive_func) {
        Mat A{{.depth = 15, .rows = n, .cols = n}};
        Mat B{{.depth = 15, .rows = n, .cols = n}};
        std::ranges::generate(A, [&] { return nrml(rng); });
        std::ranges::generate(B, [&] { return nrml(rng); });
        A.view.add_to_diagonal(10);

        // Save the original C matrix for comparison
        Mat B_reference = B;

        // Perform the operation
        func(A, B, backend);

        // Perform the reference operation for comparison
        naive_func(A, B_reference);

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < B.depth(); ++i)
            for (index_t j = 0; j < B.rows(); ++j)
                for (index_t k = 0; k < B.cols(); ++k)
                    ASSERT_NEAR(B(i, j, k), B_reference(i, j, k), ε)
                        << enum_name(backend) << "[" << n
                        << "]: TRSM mismatch at (" << i << ", " << j << ", "
                        << k << ")";
    }

    static void naive_trsm_RLTN(View L, MutView H) {
        for (index_t l = 0; l < L.depth(); ++l) {
            for (index_t k = 0; k < L.rows(); ++k) {
                auto pivot     = L(l, k, k);
                auto inv_pivot = 1 / pivot;
                for (index_t i = 0; i < H.rows(); ++i)
                    H(l, i, k) = H(l, i, k) * inv_pivot;
                for (index_t j = k + 1; j < L.rows(); ++j) {
                    auto Ljk = L(l, j, k);
                    for (index_t i = 0; i < H.rows(); ++i) {
                        auto Hij = H(l, i, j);
                        Hij -= Ljk * H(l, i, k);
                        H(l, i, j) = Hij;
                    }
                }
            }
        }
    }
    static void naive_trsm_LLNN(View L, MutView H) {
        for (index_t l = 0; l < L.depth(); ++l) {
            for (index_t j = 0; j < H.cols(); ++j) {
                for (index_t k = 0; k < L.rows(); ++k) {
                    auto pivot = L(l, k, k);
                    auto Hkj   = H(l, k, j);
                    Hkj /= pivot;
                    H(l, k, j) = Hkj;
                    for (index_t i = k + 1; i < H.rows(); ++i) {
                        auto Hij = H(l, i, j);
                        Hij -= Hkj * L(l, i, k);
                        H(l, i, j) = Hij;
                    }
                }
            }
        }
    }
    static void naive_trsm_LLTN(View L, MutView H) {
        for (index_t l = 0; l < L.depth(); ++l) {
            for (index_t j = 0; j < H.cols(); ++j) {
                for (index_t i = H.rows(); i-- > 0;) {
                    auto Hij = H(l, i, j);
                    for (index_t k = i + 1; k < L.rows(); ++k) {
                        auto Hkj = H(l, k, j);
                        Hij -= Hkj * L(l, k, i);
                    }
                    auto pivot = L(l, i, i);
                    Hij /= pivot;
                    H(l, i, j) = Hij;
                }
            }
        }
    }
};

TYPED_TEST_SUITE(TrsmTest, koqkatoo::tests::Abis);

TYPED_TEST(TrsmTest, TrsmRLTN) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(backend, i, TestFixture::CompactBLAS_t::xtrsm_RLTN,
                          TestFixture::naive_trsm_RLTN);
}
TYPED_TEST(TrsmTest, TrsmLLNN) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(backend, i, TestFixture::CompactBLAS_t::xtrsm_LLNN,
                          TestFixture::naive_trsm_LLNN);
}
TYPED_TEST(TrsmTest, TrsmLLTN) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(backend, i, TestFixture::CompactBLAS_t::xtrsm_LLTN,
                          TestFixture::naive_trsm_LLTN);
}
