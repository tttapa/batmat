#include <gtest/gtest.h>

#include <batmat/linalg/copy.hpp>
#include <batmat/matrix/matrix.hpp>
#include <batmat/matrix/view.hpp>

#include "config.hpp"

#include <numeric>

namespace stdx = std::experimental;
using batmat::index_t;
using batmat::matrix::StorageOrder;

template <ptrdiff_t N, StorageOrder OA, StorageOrder OB>
struct CopyConfig {
    using type                            = batmat::real_t;
    using abi                             = stdx::simd_abi::deduce_t<type, N>;
    static constexpr StorageOrder order_A = OA;
    static constexpr StorageOrder order_B = OB;
};

template <class Conf>
class CopyTest : public ::testing::Test {
  protected:
    using abi  = typename Conf::abi;
    using type = typename Conf::type;
    using MatA = batmat::linalg::matrix<type, typename Conf::abi, Conf::order_A>;
    using MatB = batmat::linalg::matrix<type, typename Conf::abi, Conf::order_B>;

    void test_copy(index_t rows, index_t cols) {
        const index_t padding = 128;
        MatA pad_A{{.rows = padding + rows + padding, .cols = padding + cols + padding}};
        MatB pad_B{{.rows = padding + rows + padding, .cols = padding + cols + padding}};
        std::ranges::iota(pad_A, index_t{});
        std::ranges::fill(pad_B, -42);
        auto A = pad_A.block(padding, padding, rows, cols).as_const();
        auto B = pad_B.block(padding, padding, rows, cols);
        batmat::linalg::copy<type, abi>(A.batch(0), B.batch(0));
        for (index_t l = 0; l < pad_B.depth(); ++l)
            for (index_t r = 0; r < pad_B.rows(); ++r)
                for (index_t c = 0; c < pad_B.cols(); ++c)
                    if (r < padding || r >= padding + rows || c < padding || c >= padding + cols)
                        ASSERT_EQ(pad_B(l, r, c), -42) << l << ", " << r << ", " << c;
                    else
                        ASSERT_EQ(pad_B(l, r, c), pad_A(l, r, c)) << l << ", " << r << ", " << c;
    }
};

TYPED_TEST_SUITE_P(CopyTest);

TYPED_TEST_P(CopyTest, copy) {
    for (index_t r : batmat::tests::sizes)
        for (index_t c : batmat::tests::sizes)
            this->test_copy(r, c);
}

REGISTER_TYPED_TEST_SUITE_P(CopyTest, copy);

using enum StorageOrder;
using TestConfigs =
    ::testing::Types<CopyConfig<1, ColMajor, ColMajor>, CopyConfig<1, RowMajor, ColMajor>,
                     CopyConfig<1, ColMajor, RowMajor>, CopyConfig<1, RowMajor, RowMajor>,
                     CopyConfig<4, ColMajor, ColMajor>, CopyConfig<4, RowMajor, ColMajor>,
                     CopyConfig<4, ColMajor, RowMajor>, CopyConfig<4, RowMajor, RowMajor>>;

INSTANTIATE_TYPED_TEST_SUITE_P(linalg, CopyTest, TestConfigs);
