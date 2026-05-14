#include <batmat/linalg/geqrf.hpp>
#include <Eigen/QR>
#include <gtest/gtest.h>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

using batmat::matrix::StorageOrder;
using enum Eigen::UpLoType;

template <class Config>
struct QRTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(QRTest);

TYPED_TEST_P(QRTest, geqrf) {
    using batmat::linalg::geqrf;
    using batmat::linalg::triu;
    using EMat = Eigen::MatrixX<typename TypeParam::value_type>;
    for (auto n : batmat::tests::sizes)
        for (auto m : batmat::tests::sizes) {
            if (m < n)
                continue; // Tall matrices only
            const auto A0 = this->template get_matrix<0>(m, n);
            auto A        = A0;
            A.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
            geqrf(triu(A0), triu(A));
            this->check([&](auto &&Al) { return tri<Upper>(Al.householderQr().matrixQR()); },
                        [&](auto l, auto &&res, auto &&ref, auto &&) {
                            EMat resU       = tri<Upper>(res);
                            const auto refU = tri<Upper>(ref);
                            for (index_t i = 0; i < resU.cols(); ++i)
                                if (resU(i, i) * refU(i, i) < 0)
                                    resU.row(i) *= -1; // Adjust sign ambiguity
                            EXPECT_THAT(resU, EigenAlmostEqualRel(refU, this->tolerance_n(n))) << l;
                        },
                        A, A0);
        }
}

REGISTER_TYPED_TEST_SUITE_P(QRTest, geqrf);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, QRTest, TestConfigs<OrderConfigs1>);
