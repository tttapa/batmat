#include <batmat/matrix/view.hpp>
#include <gtest/gtest.h>
#include <iomanip>
#include <numeric>

using batmat::matrix::DefaultStride;

template <ptrdiff_t N, bool R>
struct ViewTypeConfig {
    using size                         = std::integral_constant<ptrdiff_t, N>;
    static constexpr bool is_row_major = R;
};

template <typename Config>
class ViewParamTest : public ::testing::Test {
  protected:
    using T                             = float;
    using I                             = ptrdiff_t;
    using S                             = typename Config::size;
    static constexpr auto storage_order = Config::is_row_major
                                              ? batmat::matrix::StorageOrder::RowMajor
                                              : batmat::matrix::StorageOrder::ColMajor;

    static constexpr I depth      = S() * 6;
    static constexpr I rows       = 5;
    static constexpr I cols       = 7;
    static constexpr I batch_size = S::value;

    using View = batmat::matrix::View<T, I, S, I, DefaultStride, storage_order>;

    std::vector<T> data;
    View view;

    void SetUp() override {
        const auto stride = std::max(rows, cols) + 5;
        data.resize(depth * stride * stride);
        std::ranges::iota(data, 0);
        view.reassign(View{{.data         = data.data(),
                            .depth        = depth,
                            .rows         = rows,
                            .cols         = cols,
                            .outer_stride = stride,
                            .layer_stride = stride * stride}});
    }
};

TYPED_TEST_SUITE_P(ViewParamTest);

TYPED_TEST_P(ViewParamTest, TransposeMatchesOriginal) {
    constexpr ptrdiff_t D = TestFixture::depth;
    constexpr ptrdiff_t R = TestFixture::rows;
    constexpr ptrdiff_t C = TestFixture::cols;

    auto trans = this->view.transposed();
    for (ptrdiff_t l = 0; l < D; ++l) {
        for (ptrdiff_t r = 0; r < R; ++r) {
            for (ptrdiff_t c = 0; c < C; ++c) {
                std::cout << std::setw(5) << this->view(l, r, c);
                EXPECT_EQ(trans(l, c, r), this->view(l, r, c));
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
}

TYPED_TEST_P(ViewParamTest, TransposeMatchesOriginalBlock) {
    const auto bs    = this->batch_size;
    const auto depth = bs * 3;
    auto trans       = this->view.block(1, 2, 3, 4).middle_layers(bs, depth).transposed();
    for (ptrdiff_t l = 0; l < depth; ++l) {
        for (ptrdiff_t r = 0; r < 3; ++r) {
            for (ptrdiff_t c = 0; c < 4; ++c) {
                std::cout << std::setw(5) << trans(l, c, r);
                EXPECT_EQ(trans(l, c, r), this->view(l + bs, r + 1, c + 2));
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
}

REGISTER_TYPED_TEST_SUITE_P(ViewParamTest, TransposeMatchesOriginal, TransposeMatchesOriginalBlock);

using TestConfigs = ::testing::Types<ViewTypeConfig<1, true>, ViewTypeConfig<1, false>,
                                     ViewTypeConfig<4, true>, ViewTypeConfig<4, false>>;

INSTANTIATE_TYPED_TEST_SUITE_P(ViewTest, ViewParamTest, TestConfigs);
