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
class PotrfTest : public ::testing::Test {
  protected:
    using abi_t         = Abi;
    using simd          = stdx::simd<real_t, abi_t>;
    using simd_stride_t = stdx::simd_size<real_t, abi_t>;
    using Mat           = BatchedMatrix<real_t, index_t, simd_stride_t>;
    using View    = BatchedMatrixView<const real_t, index_t, simd_stride_t>;
    using MutView = BatchedMatrixView<real_t, index_t, simd_stride_t>;
    using func_t  = void(MutView, PreferredBackend);
    using naive_func_t  = std::function<void(MutView)>;
    using CompactBLAS_t = CompactBLAS<abi_t>;

    void SetUp() override { rng.seed(12345); }

    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    void RunTest(PreferredBackend backend, index_t n, func_t func,
                 naive_func_t naive_func) {
        Mat A{{.depth = 15, .rows = n, .cols = n}};
        std::ranges::generate(A, [&] { return nrml(rng); });
        A.view.add_to_diagonal(100);

        // Save the original C matrix for comparison
        Mat A_reference = A;

        // Perform the operation
        func(A, backend);

        // Perform the reference operation for comparison
        naive_func(A_reference);

        // Verify that the results match the reference implementation
        for (index_t i = 0; i < A.depth(); ++i)
            for (index_t j = 0; j < A.rows(); ++j)
                for (index_t k = 0; k < A.cols(); ++k)
                    ASSERT_NEAR(A(i, j, k), A_reference(i, j, k), ε)
                        << enum_name(backend) << "[" << n
                        << "]: POTRF mismatch at (" << i << ", " << j << ", "
                        << k << ")";
    }

    static void naive_potrf(MutView A) {
        using std::sqrt;
        auto N = A.rows();
        for (index_t l = 0; l < A.depth(); ++l) {
            // Cholesky–Crout
            for (index_t j = 0; j < N; ++j) {
                for (index_t k = 0; k < j; ++k)
                    A(l, j, j) -= A(l, j, k) * A(l, j, k);
                A(l, j, j) = sqrt(A(l, j, j));
                for (index_t i = j + 1; i < N; ++i) {
                    for (index_t k = 0; k < j; ++k)
                        A(l, i, j) -= A(l, i, k) * A(l, j, k);
                    A(l, i, j) = A(l, i, j) / A(l, j, j);
                }
            }
        }
    }
};

TYPED_TEST_SUITE(PotrfTest, koqkatoo::tests::Abis);

TYPED_TEST(PotrfTest, Potrf) {
    for (auto backend : koqkatoo::tests::backends)
        for (index_t i : koqkatoo::tests::sizes)
            this->RunTest(backend, i, TestFixture::CompactBLAS_t::xpotrf,
                          TestFixture::naive_potrf);
}
