#include <batmat/linalg/gemm.hpp>
#include <gtest/gtest.h>
#include <functional>
#include <limits>
#include <random>

#include "config.hpp"

using namespace batmat::linalg;
using batmat::index_t;
using batmat::real_t;

const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));

template <class Abi>
class GemmTest : public ::testing::Test {
  protected:
    using type          = real_t;
    using abi_t         = Abi;
    using types         = simd_view_types<type, abi_t>;
    using simd          = stdx::simd<type, abi_t>;
    using simd_stride_t = stdx::simd_size<type, abi_t>;
    using Mat           = batmat::linalg::matrix<type, abi_t>;
    using View          = batmat::linalg::view<const type, abi_t>;
    using MutView       = batmat::linalg::view<type, abi_t>;
    using func_t        = void(View, View, MutView);
    using naive_func_t  = std::function<void(View, View, MutView)>;

    void SetUp() override { rng.seed(12345); }

    std::mt19937 rng{12345};
    std::normal_distribution<type> nrml{0, 1};

    template <micro_kernels::gemm::KernelConfig Conf, bool InitZero>
    static void gemm_NN(view<const type, abi_t> A, view<const type, abi_t> B, view<type, abi_t> C) {
        batmat::linalg::detail::gemm<type, abi_t, Conf>(
            A, B, InitZero ? std::nullopt : std::make_optional(C.as_const()), C);
    }

    template <micro_kernels::gemm::KernelConfig Conf, bool InitZero>
    static void gemm_TN(view<const type, abi_t> A, view<const type, abi_t> B, view<type, abi_t> C) {
        batmat::linalg::detail::gemm<type, abi_t, Conf>(
            A.transposed(), B, InitZero ? std::nullopt : std::make_optional(C.as_const()), C);
    }

    template <micro_kernels::gemm::KernelConfig Conf, bool InitZero>
    static void gemm_NT(view<const type, abi_t> A, view<const type, abi_t> B, view<type, abi_t> C) {
        batmat::linalg::detail::gemm<type, abi_t, Conf>(
            A, B.transposed(), InitZero ? std::nullopt : std::make_optional(C.as_const()), C);
    }

    template <micro_kernels::gemm::KernelConfig Conf, bool InitZero>
    static void gemm_TT(view<const type, abi_t> A, view<const type, abi_t> B, view<type, abi_t> C) {
        batmat::linalg::detail::gemm<type, abi_t, Conf>(
            A.transposed(), B.transposed(),
            InitZero ? std::nullopt : std::make_optional(C.as_const()), C);
    }

    void RunTest(index_t n, func_t func, naive_func_t naive_func) {
        Mat A{{.depth = 15, .rows = n, .cols = n}};
        Mat B{{.depth = 15, .rows = n, .cols = n}};
        Mat C{{.depth = 15, .rows = n, .cols = n}};
        std::ranges::generate(A, [&] { return nrml(rng); });
        std::ranges::generate(B, [&] { return nrml(rng); });
        std::ranges::generate(C, [&] { return nrml(rng); });

        // Save the original C matrix for comparison
        Mat C_reference = C;

        // Perform the operation
        for (index_t l = 0; l < C.num_batches(); ++l)
            func(A.batch(l), B.batch(l), C.batch(l));

        // Perform the reference operation for comparison
        for (index_t l = 0; l < C.num_batches(); ++l)
            naive_func(A.batch(l), B.batch(l), C_reference.batch(l));

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < C.depth(); ++i)
            for (index_t j = 0; j < C.rows(); ++j)
                for (index_t k = 0; k < C.cols(); ++k)
                    ASSERT_NEAR(C(i, j, k), C_reference(i, j, k), ε)
                        << "[" << n << "]: GEMM mismatch at (" << i << ", " << j << ", " << k
                        << ")";
    }

    void RunTestTrmm(index_t n, func_t func, naive_func_t naive_func) {
        Mat A{{.depth = 15, .rows = n, .cols = n + 13}};
        Mat B{{.depth = 15, .rows = n + 13, .cols = n + 3}};
        Mat C{{.depth = 15, .rows = n, .cols = n + 3}};
        std::ranges::generate(A, [&] { return nrml(rng); });
        std::ranges::generate(B, [&] { return nrml(rng); });
        std::ranges::generate(C, [&] { return nrml(rng); });
        Mat Bl = B;
        for (index_t c = 1; c < Bl.cols(); ++c) // Make Bl lower trapezoidal
            Bl.block(0, c, c, 1).set_constant(0);

        // Save the original C matrix for comparison
        Mat C_reference = C;

        // Perform the operation
        for (index_t l = 0; l < C.num_batches(); ++l)
            func(A.batch(l), B.batch(l), C.batch(l));

        // Perform the reference operation for comparison
        for (index_t l = 0; l < C.num_batches(); ++l)
            naive_func(A.batch(l), B.batch(l), C_reference.batch(l));

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < C.depth(); ++i)
            for (index_t j = 0; j < C.rows(); ++j)
                for (index_t k = 0; k < C.cols(); ++k)
                    ASSERT_NEAR(C(i, j, k), C_reference(i, j, k), ε)
                        << "[" << n << "]: TRMM mismatch at (" << i << ", " << j << ", " << k
                        << ")";
    }

    static void naive_gemm(bool transA, bool transB, real_t α, real_t β, View A, View B,
                           MutView C) {
        auto M = transA ? A.cols() : A.rows();
        auto N = transB ? B.rows() : B.cols();
        auto K = transA ? A.rows() : A.cols();
        for (index_t l = 0; l < static_cast<index_t>(A.depth()); ++l) {
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

TYPED_TEST_SUITE(GemmTest, batmat::tests::Abis);

TYPED_TEST(GemmTest, Gemm) {
    for (index_t i : batmat::tests::sizes)
        this->RunTest(i, TestFixture::template gemm_NN<{}, true>,
                      std::bind_front(TestFixture::naive_gemm, false, false, 1, 0));
}
TYPED_TEST(GemmTest, GemmNeg) {
    for (index_t i : batmat::tests::sizes)
        this->RunTest(i, TestFixture::template gemm_NN<{.negate = true}, true>,
                      std::bind_front(TestFixture::naive_gemm, false, false, -1, 0));
}
TYPED_TEST(GemmTest, GemmAdd) {
    for (index_t i : batmat::tests::sizes)
        this->RunTest(i, TestFixture::template gemm_NN<{}, false>,
                      std::bind_front(TestFixture::naive_gemm, false, false, 1, 1));
}
TYPED_TEST(GemmTest, GemmSub) {
    for (index_t i : batmat::tests::sizes)
        this->RunTest(i, TestFixture::template gemm_NN<{.negate = true}, false>,
                      std::bind_front(TestFixture::naive_gemm, false, false, -1, 1));
}
TYPED_TEST(GemmTest, GemmTN) {
    for (index_t i : batmat::tests::sizes)
        this->RunTest(i, TestFixture::template gemm_TN<{}, true>,
                      std::bind_front(TestFixture::naive_gemm, true, false, 1, 0));
}
TYPED_TEST(GemmTest, GemmTNSub) {
    for (index_t i : batmat::tests::sizes)
        this->RunTest(i, TestFixture::template gemm_TN<{.negate = true}, false>,
                      std::bind_front(TestFixture::naive_gemm, true, false, -1, 1));
}
// TYPED_TEST(GemmTest, Trmm) {
//         for (index_t i : batmat::tests::sizes)
//             this->RunTestTrmm(
//                 i, TestFixture::CompactBLAS_t::xtrmm_RLNN,
//                 std::bind_front(TestFixture::naive_gemm, false, false, 1, 0));
// }
// TYPED_TEST(GemmTest, TrmmNeg) {
//         for (index_t i : batmat::tests::sizes)
//             this->RunTestTrmm(
//                 i, TestFixture::CompactBLAS_t::xtrmm_RLNN_neg,
//                 std::bind_front(TestFixture::naive_gemm, false, false, -1, 0));
// }
