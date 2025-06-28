#include <batmat/linalg/trtri.hpp>
#include <gtest/gtest.h>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

using enum Eigen::UpLoType;

template <class Config>
struct TrtriTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(TrtriTest);

TYPED_TEST_P(TrtriTest, trtriL) {
    using batmat::linalg::tril;
    using batmat::linalg::trtri;
    for (auto m : batmat::tests::sizes) {
        const auto A = [&] {
            auto A = this->template get_matrix<0>(m, m);
            A.view().add_to_diagonal(static_cast<TestFixture::value_type>(100 * m));
            return A;
        }();
        const auto D0 = this->template get_matrix<1>(m, m);
        auto D        = D0;
        trtri(tril(A), tril(D));
        const auto I = Eigen::MatrixX<typename TestFixture::value_type>::Identity(m, m);
        this->check([&](auto &&Al, auto &&) { return triv<Lower>(Al).solve(I).eval(); },
                    [&](auto l, auto &&res, auto &&ref, auto &&, auto &&D0) {
                        const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                        EXPECT_THAT(resL, EigenAlmostEqual(ref, this->tolerance)) << l;
                        EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(D0), this->tolerance))
                            << l;
                    },
                    D, A, D0);
    }
}

TYPED_TEST_P(TrtriTest, trtriU) {
    using batmat::linalg::triu;
    using batmat::linalg::trtri;
    for (auto m : batmat::tests::sizes) {
        const auto A = [&] {
            auto A = this->template get_matrix<0>(m, m);
            A.view().add_to_diagonal(static_cast<TestFixture::value_type>(100 * m));
            return A;
        }();
        const auto D0 = this->template get_matrix<1>(m, m);
        auto D        = D0;
        trtri(triu(A), triu(D));
        const auto I = Eigen::MatrixX<typename TestFixture::value_type>::Identity(m, m);
        this->check([&](auto &&Al, auto &&) { return triv<Upper>(Al).solve(I).eval(); },
                    [&](auto l, auto &&res, auto &&ref, auto &&, auto &&D0) {
                        const auto resU = tri<Upper>(res), resL = tri<StrictlyLower>(res);
                        EXPECT_THAT(resU, EigenAlmostEqual(ref, this->tolerance)) << l;
                        EXPECT_THAT(resL, EigenAlmostEqual(tri<StrictlyLower>(D0), this->tolerance))
                            << l;
                    },
                    D, A, D0);
    }
}

template <class Config>
struct TrtriInplaceTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(TrtriInplaceTest);

TYPED_TEST_P(TrtriInplaceTest, trtriL) {
    using batmat::linalg::tril;
    using batmat::linalg::trtri;
    for (auto m : batmat::tests::sizes) {
        const auto D0 = [&] {
            auto D = this->template get_matrix<0>(m, m);
            D.view().add_to_diagonal(static_cast<TestFixture::value_type>(100 * m));
            return D;
        }();
        auto D = D0;
        trtri(tril(D));
        const auto I = Eigen::MatrixX<typename TestFixture::value_type>::Identity(m, m);
        this->check([&](auto &&Dl) { return triv<Lower>(Dl).solve(I).eval(); },
                    [&](auto l, auto &&res, auto &&ref, auto &&D0) {
                        const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                        EXPECT_THAT(resL, EigenAlmostEqual(ref, this->tolerance)) << l;
                        EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(D0), this->tolerance))
                            << l;
                    },
                    D, D0);
    }
}

TYPED_TEST_P(TrtriInplaceTest, trtriU) {
    using batmat::linalg::triu;
    using batmat::linalg::trtri;
    for (auto m : batmat::tests::sizes) {
        const auto D0 = [&] {
            auto D = this->template get_matrix<0>(m, m);
            D.view().add_to_diagonal(static_cast<TestFixture::value_type>(100 * m));
            return D;
        }();
        auto D = D0;
        trtri(triu(D));
        const auto I = Eigen::MatrixX<typename TestFixture::value_type>::Identity(m, m);
        this->check([&](auto &&Dl) { return triv<Upper>(Dl).solve(I).eval(); },
                    [&](auto l, auto &&res, auto &&ref, auto &&D0) {
                        const auto resU = tri<Upper>(res), resL = tri<StrictlyLower>(res);
                        EXPECT_THAT(resU, EigenAlmostEqual(ref, this->tolerance)) << l;
                        EXPECT_THAT(resL, EigenAlmostEqual(tri<StrictlyLower>(D0), this->tolerance))
                            << l;
                    },
                    D, D0);
    }
}

REGISTER_TYPED_TEST_SUITE_P(TrtriTest, trtriL, trtriU);
REGISTER_TYPED_TEST_SUITE_P(TrtriInplaceTest, trtriL, trtriU);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, TrtriTest, TestConfigs<OrderConfigs2>);
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, TrtriInplaceTest, TestConfigs<OrderConfigs1>);
