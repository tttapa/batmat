#include <batmat/linalg/potrf.hpp>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

using enum Eigen::UpLoType;

template <class Config>
struct PotrfTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(PotrfTest);

TYPED_TEST_P(PotrfTest, potrfLinplace) {
    using batmat::linalg::potrf;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes) {
        const auto D0 = [&] {
            auto D = this->template get_matrix<0>(m, m);
            D.view().add_to_diagonal(static_cast<TypeParam::value_type>(100 * m));
            return D;
        }();
        auto D = D0;
        potrf(tril(D));
        this->check(
            [&](auto &&Dl) {
                return Dl.template selfadjointView<Lower>().llt().matrixL().toDenseMatrix();
            },
            [&](auto l, auto &&res, auto &&ref, auto &&D0) {
                const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                EXPECT_THAT(resL, EigenAlmostEqual(ref, this->tolerance)) << l;
                EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(D0), this->tolerance)) << l;
            },
            D, D0);
    }
}

template <class Config>
struct SyrkPotrfTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(SyrkPotrfTest);

TYPED_TEST_P(SyrkPotrfTest, syrkPotrfLinplace) {
    using batmat::linalg::syrk_add_potrf;
    using batmat::linalg::tril;
    using EMat = Eigen::MatrixX<typename TypeParam::value_type>;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto D0 = [&] {
                auto D = this->template get_matrix<1>(m, m);
                D.view().add_to_diagonal(static_cast<TypeParam::value_type>(10 * m));
                return D;
            }();
            const auto A = this->template get_matrix<0>(m, n);
            auto D       = D0;
            syrk_add_potrf(A, tril(D), tril(D));
            this->check(
                [&](auto &&A, EMat Dl) {
                    auto Dll = Dl.template selfadjointView<Lower>();
                    if (A.cols() > 0) // TODO: Eigen crashes if this is zero
                        Dll.rankUpdate(A);
                    return Dll.llt().matrixL().toDenseMatrix();
                },
                [&](auto l, auto &&res, auto &&ref, auto &&, auto &&D0) {
                    const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                    EXPECT_THAT(resL, EigenAlmostEqual(ref, this->tolerance)) << l;
                    EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(D0), this->tolerance))
                        << l;
                },
                D, A, D0);
        }
}

REGISTER_TYPED_TEST_SUITE_P(PotrfTest, potrfLinplace);
REGISTER_TYPED_TEST_SUITE_P(SyrkPotrfTest, syrkPotrfLinplace);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, PotrfTest, TestConfigs<OrderConfigs1>);
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, SyrkPotrfTest, TestConfigs<OrderConfigs2>);
