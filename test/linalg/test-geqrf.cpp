#include <batmat/linalg/copy.hpp>
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
    using EMat = Eigen::MatrixX<typename TypeParam::value_type>;
    for (auto n : batmat::tests::sizes)
        for (auto m : batmat::tests::sizes) {
            if (m < n)
                continue; // Tall matrices only
            const auto A0 = this->template get_matrix<0>(m, n);
            auto w        = this->template get_matrix<StorageOrder::ColMajor>(n, 1);
            auto A        = A0;
            A.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
            geqrf(A0, A, w);
            this->check([&](auto &&Al) -> EMat { return Al.householderQr().matrixQR(); },
                        [&](auto l, auto &&res, auto &&ref, auto &&) {
                            EMat resU       = tri<Upper>(res);
                            const auto refU = tri<Upper>(ref);
                            for (index_t i = 0; i < resU.cols(); ++i)
                                if (resU(i, i) * refU(i, i) < 0)
                                    resU.row(i) *= -1; // Adjust sign ambiguity
                            EXPECT_THAT(resU, EigenAlmostEqualRel(refU, this->tolerance_n(n))) << l;
                            EMat resL       = tri<StrictlyLower>(res);
                            const auto refL = tri<StrictlyLower>(ref);
                            for (index_t i = 0; i < resL.cols(); ++i)
                                if (resU(i, i) * refU(i, i) < 0)
                                    resL.col(i) *= -1; // Adjust sign ambiguity
                            EXPECT_THAT(resL, EigenAlmostEqualRel(refL, this->tolerance_n(n))) << l;
                        },
                        A, A0);
            if (n > 1)
                this->check([&](auto &&Al) -> EMat { return Al.householderQr().hCoeffs(); },
                            [&](auto l, auto &&res, auto &&ref, auto &&) {
                                // last coefficient is zero for Eigen, and two for ours
                                // (either way, it represents Q = I)
                                EXPECT_THAT(
                                    res.topRows(n - 1),
                                    EigenAlmostEqualRel(ref.topRows(n - 1), this->tolerance_n(n)))
                                    << l;
                            },
                            w, A0);
        }
}

REGISTER_TYPED_TEST_SUITE_P(QRTest, geqrf);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, QRTest, TestConfigs<OrderConfigs1>);

template <class Config>
struct HouseholderApplyTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(HouseholderApplyTest);

TYPED_TEST_P(HouseholderApplyTest, geqrfApplyQR) {
    using batmat::linalg::copy;
    using batmat::linalg::geqrf;
    using batmat::linalg::geqrf_apply;
    using batmat::linalg::geqrf_size_W;
    using batmat::linalg::triu;
    using EMat = Eigen::MatrixX<typename TypeParam::value_type>;
    for (auto n : batmat::tests::sizes)
        for (auto m : batmat::tests::sizes) {
            if (m < n)
                continue; // Tall matrices only
            const auto A0 = this->template get_matrix<0>(m, n);
            auto [rw, cw] = geqrf_size_W(A0);
            auto W        = this->template get_matrix<StorageOrder::ColMajor>(rw, cw);
            auto A        = A0;
            A.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
            geqrf(A0, A, W);
            auto R   = this->template get_matrix<0>(m, n);
            auto QR  = this->template get_matrix<1>(m, n);
            auto QᵀA = this->template get_matrix<1>(m, n);
            R.set_constant(0);
            copy(triu(A.top_rows(n)), triu(R.top_rows(n)));
            geqrf_apply(R, QR, A, W, false);
            geqrf_apply(A0, QᵀA, A, W, true);

            this->check([&](auto &&A0l) -> EMat { return A0l; },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance_n(n))) << l;
                        },
                        QR, A0);
            this->check([&](auto &&Rl) -> EMat { return Rl; },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance_n(n))) << l;
                        },
                        QᵀA, R);

            auto I = this->template get_matrix<0>(m, m);
            auto Q = this->template get_matrix<1>(m, m);
            I.set_constant(0);
            I.set_diagonal(1);
            geqrf_apply(I, Q, A, W, false);
            auto QᵀQ = this->template get_matrix<1>(m, m);
            geqrf_apply(Q, QᵀQ, A, W, true);

            this->check([&] -> EMat { return EMat::Identity(m, m); },
                        [&](auto l, auto &&res, auto &&ref) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance_n(n))) << l;
                        },
                        QᵀQ);
        }
}

REGISTER_TYPED_TEST_SUITE_P(HouseholderApplyTest, geqrfApplyQR);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, HouseholderApplyTest, TestConfigs<OrderConfigs2>);
