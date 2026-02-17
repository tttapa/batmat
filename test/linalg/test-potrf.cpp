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

TYPED_TEST_P(PotrfTest, potrfLinplaceTall) {
    using batmat::linalg::potrf;
    using batmat::linalg::tril;
    for (auto n : batmat::tests::sizes)
        for (auto m : batmat::tests::sizes) {
            const auto D0 = [&] {
                auto D = this->template get_matrix<0>(m + n, m);
                D.view().add_to_diagonal(static_cast<TypeParam::value_type>(100 * m));
                return D;
            }();
            auto D = D0;
            potrf(tril(D));
            this->check(
                [&](auto &&Dl) {
                    auto Dtop = Dl.topRows(m);
                    auto Dbot = Dl.bottomRows(n);
                    Eigen::MatrixX<typename TypeParam::value_type> R(m + n, m);
                    R.topRows(m) = Dtop.template selfadjointView<Lower>().llt().matrixL();
                    if (n > 0)
                        R.bottomRows(n) =
                            triv<Eigen::Lower>(R.topRows(m)).solve(Dbot.transpose()).transpose();
                    return R;
                },
                [&](auto l, auto &&res, auto &&ref, auto &&D0) {
                    const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                    EXPECT_THAT(resL, EigenAlmostEqual(ref, this->tolerance)) << l;
                    EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(D0), this->tolerance))
                        << l;
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

TYPED_TEST_P(SyrkPotrfTest, syrkDiagPotrfLinplace) {
    using batmat::linalg::syrk_diag_add_potrf;
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
            const auto d = this->get_vector(n);
            auto D       = D0;
            syrk_diag_add_potrf(A, tril(D), tril(D), d);
            this->check(
                [&](auto &&A, EMat Dl, EMat dl) {
                    EMat Ad = A * dl.asDiagonal();
                    if (A.cols() > 0) // TODO: Eigen crashes if this is zero
                        Dl += Ad * A.transpose();
                    auto Dll = Dl.template selfadjointView<Lower>();
                    return Dll.llt().matrixL().toDenseMatrix();
                },
                [&](auto l, auto &&res, auto &&ref, auto &&, auto &&D0, auto &&) {
                    const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                    EXPECT_THAT(resL, EigenAlmostEqual(ref, this->tolerance)) << l;
                    EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(D0), this->tolerance))
                        << l;
                },
                D, A, D0, d);
        }
}

REGISTER_TYPED_TEST_SUITE_P(PotrfTest, potrfLinplace, potrfLinplaceTall);
REGISTER_TYPED_TEST_SUITE_P(SyrkPotrfTest, syrkPotrfLinplace, syrkDiagPotrfLinplace);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, PotrfTest, TestConfigs<OrderConfigs1>);
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, SyrkPotrfTest, TestConfigs<OrderConfigs2>);
