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
class GemmTest : public ::testing::Test {
  protected:
    using abi_t         = Abi;
    using simd          = stdx::simd<real_t, abi_t>;
    using simd_stride_t = stdx::simd_size<real_t, abi_t>;
    using Mat           = BatchedMatrix<real_t, index_t, simd_stride_t>;
    using View = BatchedMatrixView<const real_t, index_t, simd_stride_t,
                                   index_t, index_t>;
    using MutView =
        BatchedMatrixView<real_t, index_t, simd_stride_t, index_t, index_t>;
    using func_t        = void(View, View, MutView, PreferredBackend);
    using naive_func_t  = std::function<void(View, View, MutView)>;
    using CompactBLAS_t = CompactBLAS<abi_t>;

    void SetUp() override { rng.seed(12345); }

    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    void RunTest(PreferredBackend backend, index_t n, func_t func,
                 naive_func_t naive_func) {
        Mat A{{.depth = 15, .rows = n, .cols = n}};
        Mat B{{.depth = 15, .rows = n, .cols = n}};
        Mat C{{.depth = 15, .rows = n, .cols = n}};
        std::ranges::generate(A, [&] { return nrml(rng); });
        std::ranges::generate(B, [&] { return nrml(rng); });
        std::ranges::generate(C, [&] { return nrml(rng); });

        // Save the original C matrix for comparison
        Mat C_reference = C;

        // Perform the operation
        func(A, B, C, backend);

        // Perform the reference operation for comparison
        naive_func(A, B, C_reference);

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < C.depth(); ++i)
            for (index_t j = 0; j < C.rows(); ++j)
                for (index_t k = 0; k < C.cols(); ++k)
                    ASSERT_NEAR(C(i, j, k), C_reference(i, j, k), ε)
                        << enum_name(backend) << "[" << n
                        << "]: GEMM mismatch at (" << i << ", " << j << ", "
                        << k << ")";
    }

    static void naive_gemm(bool transA, bool transB, real_t α, real_t β, View A,
                           View B, MutView C) {
        auto M = transA ? A.cols() : A.rows();
        auto N = transB ? B.rows() : B.cols();
        auto K = transA ? A.rows() : A.cols();
        for (index_t l = 0; l < A.depth(); ++l) {
            for (index_t r = 0; r < M; ++r) {
                for (index_t c = 0; c < N; ++c) {
                    auto Crc = C(l, r, c);
                    Crc *= β;
                    for (index_t k = 0; k < K; ++k) {
                        auto Ark = transA ? A(l, k, r) : A(l, r, k);
                        auto Bkc = transB ? B(l, c, k) : B(l, k, c);
                        Crc += α * Ark * Bkc;
                    }
                    C(l, r, c) = Crc;
                }
            }
        }
    }
};

TYPED_TEST_SUITE(GemmTest, koqkatoo::tests::Abis);

TYPED_TEST(GemmTest, Gemm) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(
                backend, i, TestFixture::CompactBLAS_t::xgemm,
                std::bind_front(TestFixture::naive_gemm, false, false, 1, 0));
}
TYPED_TEST(GemmTest, GemmNeg) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(
                backend, i, TestFixture::CompactBLAS_t::xgemm_neg,
                std::bind_front(TestFixture::naive_gemm, false, false, -1, 0));
}
TYPED_TEST(GemmTest, GemmAdd) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(
                backend, i, TestFixture::CompactBLAS_t::xgemm_add,
                std::bind_front(TestFixture::naive_gemm, false, false, 1, 1));
}
TYPED_TEST(GemmTest, GemmSub) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(
                backend, i, TestFixture::CompactBLAS_t::xgemm_sub,
                std::bind_front(TestFixture::naive_gemm, false, false, -1, 1));
}
TYPED_TEST(GemmTest, GemmTN) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(
                backend, i, TestFixture::CompactBLAS_t::xgemm_TN,
                std::bind_front(TestFixture::naive_gemm, true, false, 1, 0));
}
TYPED_TEST(GemmTest, GemmTNSub) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(
                backend, i, TestFixture::CompactBLAS_t::xgemm_TN_sub,
                std::bind_front(TestFixture::naive_gemm, true, false, -1, 1));
}
