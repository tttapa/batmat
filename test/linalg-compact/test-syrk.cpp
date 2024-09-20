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
class SyrkTest : public ::testing::Test {
  protected:
    using abi_t         = Abi;
    using simd          = stdx::simd<real_t, abi_t>;
    using simd_stride_t = stdx::simd_size<real_t, abi_t>;
    using Mat           = BatchedMatrix<real_t, index_t, simd_stride_t>;
    using BMat          = BatchedMatrix<bool, index_t, simd_stride_t>;
    using View = BatchedMatrixView<const real_t, index_t, simd_stride_t,
                                   index_t, index_t>;
    using MutView =
        BatchedMatrixView<real_t, index_t, simd_stride_t, index_t, index_t>;
    using func_t        = void(View, MutView, PreferredBackend);
    using naive_func_t  = std::function<void(View, MutView)>;
    using CompactBLAS_t = CompactBLAS<abi_t>;

    void SetUp() override { rng.seed(12345); }

    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    void RunTest(PreferredBackend backend, index_t n, func_t func,
                 naive_func_t naive_func) {
        Mat A{{.depth = 15, .rows = n, .cols = n}};
        Mat C{{.depth = 15, .rows = n, .cols = n}};
        std::ranges::generate(A, [&] { return nrml(rng); });
        std::ranges::generate(C, [&] { return nrml(rng); });

        // Save the original C matrix for comparison
        Mat C_reference = C;

        // Perform the operation
        func(A, C, backend);

        // Perform the reference operation for comparison
        naive_func(A, C_reference);

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < C.depth(); ++i)
            for (index_t j = 0; j < C.rows(); ++j)
                for (index_t k = 0; k < C.cols(); ++k)
                    ASSERT_NEAR(C(i, j, k), C_reference(i, j, k), ε)
                        << enum_name(backend) << "[" << n
                        << "]: SYRK mismatch at (" << i << ", " << j << ", "
                        << k << ")";
    }

    void RunTestNonSquare(index_t m, index_t n, func_t func,
                          naive_func_t naive_func) {
        const auto backend = PreferredBackend::Reference;

        Mat A{{.depth = 15, .rows = m, .cols = n}};
        Mat C{{.depth = 15, .rows = m, .cols = n}};
        std::ranges::generate(A, [&] { return nrml(rng); });
        std::ranges::generate(C, [&] { return nrml(rng); });

        // Save the original C matrix for comparison
        Mat C_reference = C;

        // Perform the operation
        func(A, C, backend);

        // Perform the reference operation for comparison
        naive_func(A, C_reference);

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < C.depth(); ++i)
            for (index_t j = 0; j < C.rows(); ++j)
                for (index_t k = 0; k < C.cols(); ++k)
                    ASSERT_NEAR(C(i, j, k), C_reference(i, j, k), ε)
                        << enum_name(backend) << "[" << m << ", " << n
                        << "]: SYRK mismatch at (" << i << ", " << j << ", "
                        << k << ")";
    }

    static void naive_gemmt(bool transA, bool transB, real_t α, real_t β,
                            View A, View B, MutView C) {
        auto M = C.rows();
        auto N = C.cols();
        auto K = transA ? A.rows() : A.cols();
        for (index_t l = 0; l < A.depth(); ++l) {
            for (index_t r = 0; r < M; ++r) {
                for (index_t c = 0; c <= std::min(r, N - 1); ++c) {
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

    static void naive_syrk(bool trans, real_t α, real_t β, View A, MutView C) {
        naive_gemmt(trans, !trans, α, β, A, A, C);
    }

    static void naive_gemmt_diag(bool transA, bool transB, real_t α, real_t β,
                                 View A, View B, MutView C, View d) {
        auto M = C.rows();
        auto N = C.cols();
        auto K = transA ? A.rows() : A.cols();
        for (index_t l = 0; l < A.depth(); ++l) {
            for (index_t r = 0; r < M; ++r) {
                for (index_t c = 0; c <= std::min(r, N - 1); ++c) {
                    auto Crc = C(l, r, c);
                    Crc *= β;
                    for (index_t k = 0; k < K; ++k) {
                        auto Ark = transA ? A(l, k, r) : A(l, r, k);
                        auto Bkc = transB ? B(l, c, k) : B(l, k, c);
                        auto dk  = d(l, k, 0);
                        Crc += α * (Ark * dk) * Bkc;
                    }
                    C(l, r, c) = Crc;
                }
            }
        }
    }

    static void naive_syrk_diag(bool trans, real_t α, real_t β, View A,
                                MutView C, View d) {
        naive_gemmt_diag(trans, !trans, α, β, A, A, C, d);
    }
};

TYPED_TEST_SUITE(SyrkTest, koqkatoo::tests::Abis);

TYPED_TEST(SyrkTest, Syrk) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(
                backend, i, TestFixture::CompactBLAS_t::xsyrk,
                std::bind_front(TestFixture::naive_syrk, false, 1, 0));
}
TYPED_TEST(SyrkTest, SyrkSub) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(
                backend, i, TestFixture::CompactBLAS_t::xsyrk_sub,
                std::bind_front(TestFixture::naive_syrk, false, -1, 1));
}
TYPED_TEST(SyrkTest, SyrkTN) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(backend, i, TestFixture::CompactBLAS_t::xsyrk_T,
                          std::bind_front(TestFixture::naive_syrk, true, 1, 0));
}
TYPED_TEST(SyrkTest, SyrkSubNonSquare) {
    for (index_t i : koqkatoo::tests::sizes)
        for (index_t j : koqkatoo::tests::sizes)
            if (i >= j)
                this->RunTestNonSquare(
                    i, j, TestFixture::CompactBLAS_t::xsyrk_sub,
                    std::bind_front(TestFixture::naive_syrk, false, -1, 1));
}
TYPED_TEST(SyrkTest, SyrkTSchurCopy) {
    for (index_t n : koqkatoo::tests::sizes) {
        using Mat          = TestFixture::Mat;
        using BMat         = TestFixture::BMat;
        const index_t m    = n * 2 + 3;
        const auto backend = PreferredBackend::Reference;
        std::bernoulli_distribution bernoulli{0.5};
        Mat C{{.depth = 15, .rows = m, .cols = n}};
        Mat Σ{{.depth = 15, .rows = m, .cols = 1}};
        BMat J{{.depth = 15, .rows = m, .cols = 1}};
        Mat H{{.depth = 15, .rows = n, .cols = n}};
        Mat H_out{{.depth = 15, .rows = n, .cols = n}};
        Mat ΣJ{{.depth = 15, .rows = m, .cols = 1}};
        std::ranges::generate(C, [&] { return this->nrml(this->rng); });
        std::ranges::generate(Σ, [&] { return this->nrml(this->rng); });
        std::ranges::generate(J, [&] { return bernoulli(this->rng); });
        std::ranges::generate(H, [&] { return this->nrml(this->rng); });
        std::ranges::generate(H_out, [&] { return this->nrml(this->rng); });

        TestFixture::CompactBLAS_t::xsyrk_T_schur_copy(C, Σ, J, H, H_out,
                                                       backend);

        Mat H_out_ref = H;
        for (index_t l = 0; l < J.depth(); ++l)
            for (index_t i = 0; i < J.rows(); ++i)
                ΣJ(l, i, 0) = J(l, i, 0) ? Σ(l, i, 0) : 0;
        this->naive_gemmt_diag(true, false, 1, 1, C, C, H_out_ref, ΣJ);
        for (index_t l = 0; l < J.depth(); ++l)
            for (index_t i = 0; i < J.rows(); ++i)
                for (index_t k = i + 1; k < H_out.cols(); ++k)
                    H_out_ref(l, i, k) = H_out(l, i, k);

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < H_out.depth(); ++i)
            for (index_t j = 0; j < H_out.rows(); ++j)
                for (index_t k = 0; k < H_out.cols(); ++k)
                    ASSERT_NEAR(H_out(i, j, k), H_out_ref(i, j, k), ε)
                        << enum_name(backend) << "[" << n
                        << "]: SYRK mismatch at (" << i << ", " << j << ", "
                        << k << ")";
    }
}
