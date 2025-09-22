#include <gtest/gtest.h>

#include <batmat/linalg/copy.hpp>
#include <batmat/matrix/matrix.hpp>
#include <batmat/matrix/view.hpp>
#include <batmat/simd.hpp>

#include "config.hpp"

#include <numeric>

using batmat::index_t;
using batmat::linalg::MatrixStructure;
using batmat::matrix::StorageOrder;

template <ptrdiff_t N, StorageOrder OA, StorageOrder OB, MatrixStructure S, bool G = false>
struct CopyConfig {
    using type                             = batmat::real_t;
    using abi                              = batmat::datapar::deduced_abi<type, N>;
    static constexpr StorageOrder order_A  = OA;
    static constexpr StorageOrder order_B  = OB;
    static constexpr MatrixStructure struc = S;
    static constexpr bool Guanaqo          = G;
    static_assert(!G || N == 1);
};

template <class Conf>
class CopyTest : public ::testing::Test {
  protected:
    using abi  = typename Conf::abi;
    using type = typename Conf::type;
    using MatA = batmat::linalg::matrix<type, typename Conf::abi, Conf::order_A>;
    using MatB = batmat::linalg::matrix<type, typename Conf::abi, Conf::order_B>;

    void test_copy(index_t rows, index_t cols) {
        using enum MatrixStructure;
        const index_t padding = 128;
        const type pad_value{-42};
        MatA pad_A{{.rows = padding + rows + padding, .cols = padding + cols + padding}};
        MatB pad_B{{.rows = padding + rows + padding, .cols = padding + cols + padding}};
        std::ranges::iota(pad_A, index_t{});
        std::ranges::fill(pad_B, pad_value);
        auto A = pad_A.block(padding, padding, rows, cols).as_const();
        auto B = pad_B.block(padding, padding, rows, cols);

        if constexpr (Conf::Guanaqo)
            batmat::linalg::copy(batmat::linalg::make_structured<Conf::struc>(A(0)),
                                 batmat::linalg::make_structured<Conf::struc>(B(0)));
        else
            batmat::linalg::copy(batmat::linalg::make_structured<Conf::struc>(A.batch(0)),
                                 batmat::linalg::make_structured<Conf::struc>(B.batch(0)));

        const index_t JI_adif = std::max<index_t>(0, cols - rows);
        const index_t IJ_adif = std::max<index_t>(0, rows - cols);
        for (index_t l = 0; l < pad_B.depth(); ++l)
            for (index_t r = 0; r < pad_B.rows(); ++r)
                for (index_t c = 0; c < pad_B.cols(); ++c) {
                    const bool in_block = (r >= padding && r < padding + rows && //
                                           c >= padding && c < padding + cols);
                    bool should_copy    = true;
                    if (in_block) {
                        const index_t i = r - padding;
                        const index_t j = c - padding;
                        if constexpr (Conf::struc == LowerTriangular)
                            should_copy = (i >= j - JI_adif);
                        else if constexpr (Conf::struc == UpperTriangular)
                            should_copy = (j >= i - IJ_adif);
                        else
                            static_assert(Conf::struc == General);
                    }
                    if (!in_block || !should_copy)
                        ASSERT_EQ(pad_B(l, r, c), pad_value) << l << ", " << r << ", " << c;
                    else
                        ASSERT_EQ(pad_B(l, r, c), pad_A(l, r, c)) << l << ", " << r << ", " << c;
                }
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
using enum MatrixStructure;
// clang-format off
using TestConfigs = ::testing::Types<
    // Rectangular
    CopyConfig<1, ColMajor, ColMajor, General, true>, CopyConfig<1, RowMajor, ColMajor, General, true>,
    CopyConfig<1, ColMajor, RowMajor, General, true>, CopyConfig<1, RowMajor, RowMajor, General, true>,
    CopyConfig<1, ColMajor, ColMajor, General>, CopyConfig<1, RowMajor, ColMajor, General>,
    CopyConfig<1, ColMajor, RowMajor, General>, CopyConfig<1, RowMajor, RowMajor, General>,
    CopyConfig<4, ColMajor, ColMajor, General>, CopyConfig<4, RowMajor, ColMajor, General>,
    CopyConfig<4, ColMajor, RowMajor, General>, CopyConfig<4, RowMajor, RowMajor, General>,
    // Lower trapezoidal
    CopyConfig<1, ColMajor, ColMajor, LowerTriangular, true>, CopyConfig<1, RowMajor, ColMajor, LowerTriangular, true>,
    CopyConfig<1, ColMajor, RowMajor, LowerTriangular, true>, CopyConfig<1, RowMajor, RowMajor, LowerTriangular, true>,
    CopyConfig<1, ColMajor, ColMajor, LowerTriangular>, CopyConfig<1, RowMajor, ColMajor, LowerTriangular>,
    CopyConfig<1, ColMajor, RowMajor, LowerTriangular>, CopyConfig<1, RowMajor, RowMajor, LowerTriangular>,
    CopyConfig<4, ColMajor, ColMajor, LowerTriangular>, CopyConfig<4, RowMajor, ColMajor, LowerTriangular>,
    CopyConfig<4, ColMajor, RowMajor, LowerTriangular>, CopyConfig<4, RowMajor, RowMajor, LowerTriangular>,
    // Upper trapezoidal
    CopyConfig<1, ColMajor, ColMajor, UpperTriangular, true>, CopyConfig<1, RowMajor, ColMajor, UpperTriangular, true>,
    CopyConfig<1, ColMajor, RowMajor, UpperTriangular, true>, CopyConfig<1, RowMajor, RowMajor, UpperTriangular, true>,
    CopyConfig<1, ColMajor, ColMajor, UpperTriangular>, CopyConfig<1, RowMajor, ColMajor, UpperTriangular>,
    CopyConfig<1, ColMajor, RowMajor, UpperTriangular>, CopyConfig<1, RowMajor, RowMajor, UpperTriangular>,
    CopyConfig<4, ColMajor, ColMajor, UpperTriangular>, CopyConfig<4, RowMajor, ColMajor, UpperTriangular>,
    CopyConfig<4, ColMajor, RowMajor, UpperTriangular>, CopyConfig<4, RowMajor, RowMajor, UpperTriangular>
>;
// clang-format on

INSTANTIATE_TYPED_TEST_SUITE_P(linalg, CopyTest, TestConfigs);
