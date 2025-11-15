#include <gtest/gtest.h>

#include <batmat/linalg/symv.hpp>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

using batmat::index_t;

template <class Config>
struct SymvTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(SymvTest);

TYPED_TEST_P(SymvTest, symv) {
    using batmat::linalg::symv;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes) {
        const auto A = this->template get_matrix<0>(m, m);
        const auto B = this->get_vector(m);
        auto C       = this->get_vector(m);
        symv(tril(A), B, C);
        this->check(
            [&](auto &&Al, auto &&Bl) { return Al.template selfadjointView<Eigen::Lower>() * Bl; },
            [&](auto l, auto &&res, auto &&ref, auto &&...) {
                EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance))
                    << l << "    (" << m << "×" << m << ")";
            },
            C, A, B);
    }
}

TYPED_TEST_P(SymvTest, symvSub) {
    using batmat::linalg::symv_sub;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes) {
        const auto A = this->template get_matrix<0>(m, m);
        const auto B = this->get_vector(m);
        const auto C = this->get_vector(m);
        auto D       = this->get_vector(m);
        symv_sub(tril(A), B, C, D);
        this->check(
            [&](auto &&Al, auto &&Bl, auto &&Cl) {
                return Cl - Al.template selfadjointView<Eigen::Lower>() * Bl;
            },
            [&](auto l, auto &&res, auto &&ref, auto &&...) {
                EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance))
                    << l << "    (" << m << "×" << m << ")";
            },
            D, A, B, C);
    }
}

#if BATMAT_EXTENSIVE_TESTS
TYPED_TEST_P(SymvTest, symvNeg) {
    using batmat::linalg::symv_neg;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes) {
        const auto A = this->template get_matrix<0>(m, m);
        const auto B = this->get_vector(m);
        auto C       = this->get_vector(m);
        symv_neg(tril(A), B, C);
        this->check([&](auto &&Al,
                        auto &&Bl) { return (-Al).template selfadjointView<Eigen::Lower>() * Bl; },
                    [&](auto l, auto &&res, auto &&ref, auto &&...) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance))
                            << l << "    (" << m << "×" << m << ")";
                    },
                    C, A, B);
    }
}

TYPED_TEST_P(SymvTest, symvAdd) {
    using batmat::linalg::symv_add;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes) {
        const auto A = this->template get_matrix<0>(m, m);
        const auto B = this->get_vector(m);
        const auto C = this->get_vector(m);
        auto D       = this->get_vector(m);
        symv_add(tril(A), B, C, D);
        this->check(
            [&](auto &&Al, auto &&Bl, auto &&Cl) {
                return Cl + Al.template selfadjointView<Eigen::Lower>() * Bl;
            },
            [&](auto l, auto &&res, auto &&ref, auto &&...) {
                EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance))
                    << l << "    (" << m << "×" << m << ")";
            },
            D, A, B, C);
    }
}
#else
TYPED_TEST_P(SymvTest, symvNeg) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(SymvTest, symvAdd) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
#endif

REGISTER_TYPED_TEST_SUITE_P(SymvTest, symv, symvNeg, symvAdd, symvSub);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, SymvTest, TestConfigs<OrderConfigs1>);
