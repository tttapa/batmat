#include <batmat/linalg/small-potrf.hpp>
#include <gtest/gtest.h>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

BATMAT_PRAGMA_GCC_OPTIMIZE_O3_BEGIN
#include <Eigen/Cholesky>
BATMAT_PRAGMA_GCC_OPTIMIZE_O3_END

using enum Eigen::UpLoType;

template <class Config>
struct SmallPotrfTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(SmallPotrfTest);

TYPED_TEST_P(SmallPotrfTest, potrfL) {
    using batmat::linalg::small_potrf;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes) {
        const auto A0 = [&] {
            auto A = this->template get_matrix<0>(m, m);
            A.view().add_to_diagonal(static_cast<TypeParam::value_type>(100 * m));
            return A;
        }();
        const auto D0 = this->template get_matrix<0>(m, m);
        auto D        = D0;
        small_potrf(tril(A0), tril(D));
        this->check(
            [&](auto &&Al, auto &&) {
                return Al.template selfadjointView<Lower>().llt().matrixL().toDenseMatrix();
            },
            [&](auto l, auto &&res, auto &&ref, auto &&, auto &&D0) {
                const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                EXPECT_THAT(resL, EigenAlmostEqual(ref, this->tolerance)) << l;
                EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(D0), this->tolerance)) << l;
            },
            D, A0, D0);
    }
}

TYPED_TEST_P(SmallPotrfTest, potrfLinplace) {
    using batmat::linalg::small_potrf;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes) {
        const auto D0 = [&] {
            auto D = this->template get_matrix<0>(m, m);
            D.view().add_to_diagonal(static_cast<TypeParam::value_type>(100 * m));
            return D;
        }();
        auto D = D0;
        small_potrf(tril(D));
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

TYPED_TEST_P(SmallPotrfTest, potrfLinplaceTall) {
    using batmat::linalg::small_potrf;
    using batmat::linalg::tril;
    for (auto n : batmat::tests::sizes)
        for (auto m : batmat::tests::sizes) {
            const auto D0 = [&] {
                auto D = this->template get_matrix<0>(m + n, m);
                D.view().add_to_diagonal(static_cast<TypeParam::value_type>(100 * m));
                return D;
            }();
            auto D = D0;
            small_potrf(tril(D));
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

TYPED_TEST_P(SmallPotrfTest, potrfLLeft) {
    using batmat::linalg::small_potrf_left;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes) {
        const auto A0 = [&] {
            auto A = this->template get_matrix<0>(m, m);
            A.view().add_to_diagonal(static_cast<TypeParam::value_type>(100 * m));
            return A;
        }();
        const auto D0 = this->template get_matrix<0>(m, m);
        auto D        = D0;
        small_potrf_left(tril(A0), tril(D));
        this->check(
            [&](auto &&Al, auto &&) {
                return Al.template selfadjointView<Lower>().llt().matrixL().toDenseMatrix();
            },
            [&](auto l, auto &&res, auto &&ref, auto &&, auto &&D0) {
                const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                EXPECT_THAT(resL, EigenAlmostEqual(ref, this->tolerance)) << l;
                EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(D0), this->tolerance)) << l;
            },
            D, A0, D0);
    }
}

TYPED_TEST_P(SmallPotrfTest, potrfLinplaceLeft) {
    using batmat::linalg::small_potrf_left;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes) {
        const auto D0 = [&] {
            auto D = this->template get_matrix<0>(m, m);
            D.view().add_to_diagonal(static_cast<TypeParam::value_type>(100 * m));
            return D;
        }();
        auto D = D0;
        small_potrf_left(tril(D));
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

TYPED_TEST_P(SmallPotrfTest, potrfLinplaceTallLeft) {
    using batmat::linalg::small_potrf_left;
    using batmat::linalg::tril;
    for (auto n : batmat::tests::sizes)
        for (auto m : batmat::tests::sizes) {
            const auto D0 = [&] {
                auto D = this->template get_matrix<0>(m + n, m);
                D.view().add_to_diagonal(static_cast<TypeParam::value_type>(100 * m));
                return D;
            }();
            auto D = D0;
            small_potrf_left(tril(D));
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

REGISTER_TYPED_TEST_SUITE_P(SmallPotrfTest, potrfL, potrfLinplace, potrfLinplaceTall, potrfLLeft,
                            potrfLinplaceLeft, potrfLinplaceTallLeft);

using namespace batmat::tests;
template <class T>
using ConfigScalarColMaj = batmat::types::Types<TestConfig<T, 1, ColMajor>>;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, SmallPotrfTest, TestConfigsDTypes<ConfigScalarColMaj>);
