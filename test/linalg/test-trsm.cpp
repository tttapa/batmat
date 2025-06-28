#include <batmat/linalg/trsm.hpp>
#include <gtest/gtest.h>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

using enum Eigen::UpLoType;

template <class Config>
struct TrsmTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(TrsmTest);

TYPED_TEST_P(TrsmTest, trsmLL) {
    using batmat::linalg::tril;
    using batmat::linalg::trsm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->template get_matrix<0>(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto C = this->template get_matrix<1>(m, n);
            auto D       = this->template get_matrix<2>(m, n);
            trsm(tril(A), C, D);
            this->check([&](auto &&Al, auto &&Cl) { return triv<Lower>(Al).solve(Cl).eval(); },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                        },
                        D, A, C);
        }
}

TYPED_TEST_P(TrsmTest, trsmLU) {
    using batmat::linalg::triu;
    using batmat::linalg::trsm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->template get_matrix<0>(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto C = this->template get_matrix<1>(m, n);
            auto D       = this->template get_matrix<2>(m, n);
            trsm(triu(A), C, D);
            this->check([&](auto &&Al, auto &&Cl) { return triv<Upper>(Al).solve(Cl).eval(); },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                        },
                        D, A, C);
        }
}

TYPED_TEST_P(TrsmTest, trsmRU) {
    using batmat::linalg::triu;
    using batmat::linalg::trsm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->template get_matrix<0>(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto C = this->template get_matrix<1>(n, m);
            auto D       = this->template get_matrix<2>(n, m);
            trsm(C, triu(A), D);
            this->check(
                [&](auto &&Al, auto &&Cl) {
                    return triv<Upper>(Al).transpose().solve(Cl.transpose()).transpose().eval();
                },
                [&](auto l, auto &&res, auto &&ref, auto &&...) {
                    EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                },
                D, A, C);
        }
}

TYPED_TEST_P(TrsmTest, trsmRL) {
    using batmat::linalg::tril;
    using batmat::linalg::trsm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->template get_matrix<0>(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto C = this->template get_matrix<1>(n, m);
            auto D       = this->template get_matrix<2>(n, m);
            trsm(C, tril(A), D);
            this->check(
                [&](auto &&Al, auto &&Cl) {
                    return triv<Lower>(Al).transpose().solve(Cl.transpose()).transpose().eval();
                },
                [&](auto l, auto &&res, auto &&ref, auto &&...) {
                    EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                },
                D, A, C);
        }
}

template <class Config>
struct TrsmInPlaceTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(TrsmInPlaceTest);

TYPED_TEST_P(TrsmInPlaceTest, trsmLL) {
    using batmat::linalg::tril;
    using batmat::linalg::trsm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->template get_matrix<0>(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto D0 = this->template get_matrix<1>(m, n);
            auto D        = D0;
            trsm(tril(A), D);
            this->check([&](auto &&Al, auto &&Dl) { return triv<Lower>(Al).solve(Dl).eval(); },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                        },
                        D, A, D0);
        }
}

TYPED_TEST_P(TrsmInPlaceTest, trsmLU) {
    using batmat::linalg::triu;
    using batmat::linalg::trsm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->template get_matrix<0>(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto D0 = this->template get_matrix<1>(m, n);
            auto D        = D0;
            trsm(triu(A), D);
            this->check([&](auto &&Al, auto &&Dl) { return triv<Upper>(Al).solve(Dl).eval(); },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                        },
                        D, A, D0);
        }
}

TYPED_TEST_P(TrsmInPlaceTest, trsmRU) {
    using batmat::linalg::triu;
    using batmat::linalg::trsm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->template get_matrix<0>(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto D0 = this->template get_matrix<1>(n, m);
            auto D        = D0;
            trsm(D, triu(A));
            this->check(
                [&](auto &&Al, auto &&Dl) {
                    return triv<Upper>(Al).transpose().solve(Dl.transpose()).transpose().eval();
                },
                [&](auto l, auto &&res, auto &&ref, auto &&...) {
                    EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                },
                D, A, D0);
        }
}

TYPED_TEST_P(TrsmInPlaceTest, trsmRL) {
    using batmat::linalg::tril;
    using batmat::linalg::trsm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->template get_matrix<0>(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto D0 = this->template get_matrix<1>(n, m);
            auto D        = D0;
            trsm(D, tril(A));
            this->check(
                [&](auto &&Al, auto &&Dl) {
                    return triv<Lower>(Al).transpose().solve(Dl.transpose()).transpose().eval();
                },
                [&](auto l, auto &&res, auto &&ref, auto &&...) {
                    EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                },
                D, A, D0);
        }
}

REGISTER_TYPED_TEST_SUITE_P(TrsmTest, trsmLL, trsmLU, trsmRU, trsmRL);
REGISTER_TYPED_TEST_SUITE_P(TrsmInPlaceTest, trsmLL, trsmLU, trsmRU, trsmRL);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, TrsmTest, TestConfigs<OrderConfigs3>);
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, TrsmInPlaceTest, TestConfigs<OrderConfigs2>);
