#include <gtest/gtest.h>

#include <batmat/linalg/gemv.hpp>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

using batmat::index_t;

template <class Config>
struct GemvTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(GemvTest);

TYPED_TEST_P(GemvTest, gemv) {
    using batmat::linalg::gemv;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            const auto A = this->template get_matrix<0>(m, k);
            const auto B = this->get_vector(k);
            auto C       = this->get_vector(m);
            gemv(A, B, C);
            this->check([&](auto &&Al, auto &&Bl) { return Al * Bl; },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance))
                                << l << "    (" << m << "×" << k << ")";
                        },
                        C, A, B);
        }
}

TYPED_TEST_P(GemvTest, gemvSub) {
    using batmat::linalg::gemv_sub;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            const auto A = this->template get_matrix<0>(m, k);
            const auto B = this->get_vector(k);
            const auto C = this->get_vector(m);
            auto D       = this->get_vector(m);
            gemv_sub(A, B, C, D);
            this->check([&](auto &&Al, auto &&Bl, auto &&Cl) { return Cl - Al * Bl; },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance))
                                << l << "    (" << m << "×" << k << ")";
                        },
                        D, A, B, C);
        }
}

#if BATMAT_EXTENSIVE_TESTS
TYPED_TEST_P(GemvTest, gemvNeg) {
    using batmat::linalg::gemv_neg;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            const auto A = this->template get_matrix<0>(m, k);
            const auto B = this->get_vector(k);
            auto C       = this->get_vector(m);
            gemv_neg(A, B, C);
            this->check([&](auto &&Al, auto &&Bl) { return -Al * Bl; },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance))
                                << l << "    (" << m << "×" << k << ")";
                        },
                        C, A, B);
        }
}

TYPED_TEST_P(GemvTest, gemvAdd) {
    using batmat::linalg::gemv_add;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            const auto A = this->template get_matrix<0>(m, k);
            const auto B = this->get_vector(k);
            const auto C = this->get_vector(m);
            auto D       = this->get_vector(m);
            gemv_add(A, B, C, D);
            this->check([&](auto &&Al, auto &&Bl, auto &&Cl) { return Cl + Al * Bl; },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance))
                                << l << "    (" << m << "×" << k << ")";
                        },
                        D, A, B, C);
        }
}

TYPED_TEST_P(GemvTest, gemvSubShiftA) {
    using batmat::linalg::gemv_sub;
    const auto ε = this->tolerance;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            auto A       = this->template get_matrix<0>(m, k);
            const auto B = this->get_vector(k);
            const auto C = this->get_vector(m);
            auto D       = this->get_vector(m);
            gemv_sub(A, B, C, D, batmat::linalg::with_shift_A<-1>);
            EXPECT_THAT(as_eigen(D(0)), EigenAlmostEqual(as_eigen(C(0)), ε))
                << 0 << "    (" << m << "×" << k << ")";
            for (index_t l = 1; l < A.depth(); ++l) {
                auto Cl_ref = as_eigen(C(l)) - as_eigen(A(l - 1)) * as_eigen(B(l));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Cl_ref, ε))
                    << l << "    (" << m << "×" << k << ")";
            }
        }
}

TYPED_TEST_P(GemvTest, gemvShiftCD) {
    using batmat::linalg::gemv;
    const auto ε = this->tolerance;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            auto A        = this->template get_matrix<0>(m, k);
            const auto B  = this->get_vector(k);
            const auto D0 = this->get_vector(m);
            auto D        = D0;
            gemv(A, B, D, batmat::linalg::with_rotate_C<1>, batmat::linalg::with_rotate_D<1>,
                 batmat::linalg::with_mask_D<1>);
            EXPECT_THAT(as_eigen(D(0)), EigenAlmostEqual(as_eigen(D0(0)), ε))
                << 0 << "    (" << m << "×" << k << ")";
            for (index_t l = 1; l < A.depth(); ++l) {
                auto Cl_ref = as_eigen(A(l - 1)) * as_eigen(B(l - 1));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Cl_ref, ε))
                    << l << "    (" << m << "×" << k << ")";
            }
        }
}

TYPED_TEST_P(GemvTest, gemvSubShiftCD) {
    using batmat::linalg::gemv_sub;
    const auto ε = this->tolerance;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            auto A        = this->template get_matrix<0>(m, k);
            const auto B  = this->get_vector(k);
            const auto C  = this->get_vector(m);
            const auto D0 = this->get_vector(m);
            auto D        = D0;
            gemv_sub(A, B, C, D, batmat::linalg::with_rotate_C<1>, batmat::linalg::with_rotate_D<1>,
                     batmat::linalg::with_mask_D<1>);
            EXPECT_THAT(as_eigen(D(0)), EigenAlmostEqual(as_eigen(D0(0)), ε))
                << 0 << "    (" << m << "×" << k << ")";
            for (index_t l = 1; l < A.depth(); ++l) {
                auto Cl_ref = as_eigen(C(l)) - as_eigen(A(l - 1)) * as_eigen(B(l - 1));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Cl_ref, ε))
                    << l << "    (" << m << "×" << k << ")";
            }
        }
}

TYPED_TEST_P(GemvTest, gemvShiftCDNeg) {
    using batmat::linalg::gemv;
    const auto ε = this->tolerance;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            auto A        = this->template get_matrix<0>(m, k);
            const auto B  = this->get_vector(k);
            const auto D0 = this->get_vector(m);
            auto D        = D0;
            gemv(A, B, D, batmat::linalg::with_rotate_C<-1>, batmat::linalg::with_rotate_D<-1>,
                 batmat::linalg::with_mask_D<0>);
            const auto N = A.depth();
            for (index_t l = 0; l < N; ++l) {
                auto l_next = (l + 1) % N;
                auto Cl_ref = as_eigen(A(l_next)) * as_eigen(B(l_next));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Cl_ref, ε))
                    << l << "    (" << m << "×" << k << ")";
            }
        }
}

TYPED_TEST_P(GemvTest, gemvSubShiftCDNeg) {
    using batmat::linalg::gemv_sub;
    const auto ε = this->tolerance;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            auto A       = this->template get_matrix<0>(m, k);
            const auto B = this->get_vector(k);
            const auto C = this->get_vector(m);
            auto D       = this->get_vector(m);
            gemv_sub(A, B, C, D, batmat::linalg::with_rotate_C<-1>,
                     batmat::linalg::with_rotate_D<-1>, batmat::linalg::with_mask_D<0>);
            const auto N = A.depth();
            for (index_t l = 0; l < N; ++l) {
                auto l_next = (l + 1) % N;
                auto Cl_ref = as_eigen(C(l)) - as_eigen(A(l_next)) * as_eigen(B(l_next));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Cl_ref, ε))
                    << l << "    (" << m << "×" << k << ")";
            }
        }
}

#else
TYPED_TEST_P(GemvTest, gemvNeg) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(GemvTest, gemvAdd) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(GemvTest, gemvSubShiftA) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(GemvTest, gemvShiftCD) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(GemvTest, gemvSubShiftCD) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(GemvTest, gemvShiftCDNeg) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(GemvTest, gemvSubShiftCDNeg) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
#endif

REGISTER_TYPED_TEST_SUITE_P(GemvTest, gemv, gemvNeg, gemvAdd, gemvSub, gemvSubShiftA, gemvShiftCD,
                            gemvSubShiftCD, gemvShiftCDNeg, gemvSubShiftCDNeg);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, GemvTest, TestConfigs<OrderConfigs1>);
