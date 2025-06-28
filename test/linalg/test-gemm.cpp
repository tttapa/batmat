#include <gtest/gtest.h>

#include <batmat/linalg/gemm.hpp>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

using batmat::index_t;
using enum Eigen::UpLoType;

template <class Config>
struct GemmTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(GemmTest);

TYPED_TEST_P(GemmTest, gemm) {
    using batmat::linalg::gemm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->template get_matrix<0>(m, k);
                const auto B = this->template get_matrix<1>(k, n);
                auto C       = this->template get_matrix<2>(m, n);
                gemm(A, B, C);
                this->check([&](auto &&Al, auto &&Bl) { return Al * Bl; },
                            [&](auto l, auto &&res, auto &&ref, auto &&...) {
                                EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                            },
                            C, A, B);
            }
}

TYPED_TEST_P(GemmTest, gemmSub) {
    using batmat::linalg::gemm_sub;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->template get_matrix<0>(m, k);
                const auto B = this->template get_matrix<1>(k, n);
                const auto C = this->template get_matrix<2>(m, n);
                auto D       = this->template get_matrix<2>(m, n);
                gemm_sub(A, B, C, D);
                this->check([&](auto &&Al, auto &&Bl, auto &&Cl) { return Cl - Al * Bl; },
                            [&](auto l, auto &&res, auto &&ref, auto &&...) {
                                EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                            },
                            D, A, B, C);
            }
}

#if BATMAT_EXTENSIVE_TESTS
TYPED_TEST_P(GemmTest, gemmNeg) {
    using batmat::linalg::gemm_neg;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->template get_matrix<0>(m, k);
                const auto B = this->template get_matrix<1>(k, n);
                auto C       = this->template get_matrix<2>(m, n);
                gemm_neg(A, B, C);
                this->check([&](auto &&Al, auto &&Bl) { return -Al * Bl; },
                            [&](auto l, auto &&res, auto &&ref, auto &&...) {
                                EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                            },
                            C, A, B);
            }
}

TYPED_TEST_P(GemmTest, gemmAdd) {
    using batmat::linalg::gemm_add;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->template get_matrix<0>(m, k);
                const auto B = this->template get_matrix<1>(k, n);
                const auto C = this->template get_matrix<2>(m, n);
                auto D       = this->template get_matrix<2>(m, n);
                gemm_add(A, B, C, D);
                this->check([&](auto &&Al, auto &&Bl, auto &&Cl) { return Cl + Al * Bl; },
                            [&](auto l, auto &&res, auto &&ref, auto &&...) {
                                EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                            },
                            D, A, B, C);
            }
}

TYPED_TEST_P(GemmTest, gemmSubShiftA) {
    using batmat::linalg::gemm_sub;
    const auto ε = this->tolerance;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                auto A       = this->template get_matrix<0>(m, k);
                const auto B = this->template get_matrix<1>(k, n);
                const auto C = this->template get_matrix<2>(m, n);
                auto D       = this->template get_matrix<2>(m, n);
                gemm_sub(A, B, C, D, {}, batmat::linalg::with_shift_A<-1>);
                EXPECT_THAT(as_eigen(D(0)), EigenAlmostEqual(as_eigen(C(0)), ε)) << 0;
                for (index_t l = 1; l < A.depth(); ++l) {
                    auto Cl_ref = as_eigen(C(l)) - as_eigen(A(l - 1)) * as_eigen(B(l));
                    EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Cl_ref, ε)) << l;
                }
            }
}

TYPED_TEST_P(GemmTest, gemmSubShiftCD) {
    using batmat::linalg::gemm_sub;
    const auto ε = this->tolerance;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                auto A       = this->template get_matrix<0>(m, k);
                const auto B = this->template get_matrix<1>(k, n);
                const auto C = this->template get_matrix<2>(m, n);
                auto D       = this->template get_matrix<2>(m, n);
                auto D0      = D;
                gemm_sub(A, B, C, D, {}, batmat::linalg::with_rotate_C<1>,
                         batmat::linalg::with_rotate_D<1>, batmat::linalg::with_mask_D<1>);
                EXPECT_THAT(as_eigen(D(0)), EigenAlmostEqual(as_eigen(D0(0)), ε)) << 0;
                for (index_t l = 1; l < A.depth(); ++l) {
                    auto Cl_ref = as_eigen(C(l)) - as_eigen(A(l - 1)) * as_eigen(B(l - 1));
                    EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Cl_ref, ε)) << l;
                }
            }
}

TYPED_TEST_P(GemmTest, gemmSubShiftCDNeg) {
    using batmat::linalg::gemm_sub;
    const auto ε = this->tolerance;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                auto A       = this->template get_matrix<0>(m, k);
                const auto B = this->template get_matrix<1>(k, n);
                const auto C = this->template get_matrix<2>(m, n);
                auto D       = this->template get_matrix<2>(m, n);
                gemm_sub(A, B, C, D, {}, batmat::linalg::with_rotate_C<-1>,
                         batmat::linalg::with_rotate_D<-1>, batmat::linalg::with_mask_D<0>);
                const auto N = A.depth();
                for (index_t l = 0; l < N; ++l) {
                    auto l_next = (l + 1) % N;
                    auto Cl_ref = as_eigen(C(l)) - as_eigen(A(l_next)) * as_eigen(B(l_next));
                    EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Cl_ref, ε)) << l;
                }
            }
}

#else
TYPED_TEST_P(GemmTest, gemmNeg) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(GemmTest, gemmAdd) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(GemmTest, gemmSubShiftA) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(GemmTest, gemmSubShiftCD) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
TYPED_TEST_P(GemmTest, gemmSubShiftCDNeg) { GTEST_SKIP() << "BATMAT_EXTENSIVE_TESTS=0"; }
#endif

template <class Config>
struct TrmmInplaceTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(TrmmInplaceTest);

TYPED_TEST_P(TrmmInplaceTest, trmmLGinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::trmm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A  = this->template get_matrix<0>(m, m);
            const auto D0 = this->template get_matrix<1>(m, n);
            auto D        = D0;
            trmm(tril(A), D, D);
            this->check([&](auto &&Al, auto &&Dl) { return triv<Lower>(Al) * Dl; },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                        },
                        D, A, D0);
        }
}

TYPED_TEST_P(TrmmInplaceTest, trmmUGinplace) {
    using batmat::linalg::triu;
    using batmat::linalg::trmm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A  = this->template get_matrix<0>(m, m);
            const auto D0 = this->template get_matrix<1>(m, n);
            auto D        = D0;
            trmm(triu(A), D, D);
            this->check([&](auto &&Al, auto &&Dl) { return triv<Upper>(Al) * Dl; },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                        },
                        D, A, D0);
        }
}

TYPED_TEST_P(TrmmInplaceTest, trmmGLinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::trmm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A  = this->template get_matrix<0>(m, m);
            const auto D0 = this->template get_matrix<1>(n, m);
            auto D        = D0;
            trmm(D, tril(A), D);
            this->check([&](auto &&Al, auto &&Dl) { return Dl * triv<Lower>(Al); },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                        },
                        D, A, D0);
        }
}

TYPED_TEST_P(TrmmInplaceTest, trmmGUinplace) {
    using batmat::linalg::triu;
    using batmat::linalg::trmm;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A  = this->template get_matrix<0>(m, m);
            const auto D0 = this->template get_matrix<1>(n, m);
            auto D        = D0;
            trmm(D, triu(A), D);
            this->check([&](auto &&Al, auto &&Dl) { return Dl * triv<Upper>(Al); },
                        [&](auto l, auto &&res, auto &&ref, auto &&...) {
                            EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                        },
                        D, A, D0);
        }
}

template <class Config>
struct TrtrsyrkInplaceTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(TrtrsyrkInplaceTest);

TYPED_TEST_P(TrtrsyrkInplaceTest, trmmULLinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::triu;
    using batmat::linalg::trmm;
    for (auto m : batmat::tests::sizes) {
        const auto A0 = this->template get_matrix<0>(m, m);
        auto A        = A0;
        trmm(triu(A), tril(A.transposed()), tril(A));
        this->check(
            [&](auto &&Al) {
                auto Alu = tri<Upper>(Al);
                return tri<Lower>(Alu * Alu.transpose());
            },
            [&](auto l, auto &&res, auto &&ref, auto &&A0) {
                const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                EXPECT_THAT(resL, EigenAlmostEqual(ref, this->tolerance)) << l;
                EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(A0), this->tolerance)) << l;
            },
            A, A0);
    }
}

TYPED_TEST_P(TrtrsyrkInplaceTest, trmmLUUinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::triu;
    using batmat::linalg::trmm;
    for (auto m : batmat::tests::sizes) {
        const auto A0 = this->template get_matrix<0>(m, m);
        auto A        = A0;
        trmm(tril(A), triu(A.transposed()), triu(A));
        this->check(
            [&](auto &&Al) {
                auto All = tri<Lower>(Al);
                return tri<Upper>(All * All.transpose());
            },
            [&](auto l, auto &&res, auto &&ref, auto &&A0) {
                const auto resL = tri<StrictlyLower>(res), resU = tri<Upper>(res);
                EXPECT_THAT(resU, EigenAlmostEqual(ref, this->tolerance)) << l;
                EXPECT_THAT(resL, EigenAlmostEqual(tri<StrictlyLower>(A0), this->tolerance)) << l;
            },
            A, A0);
    }
}

#if 0 // TODO
TYPED_TEST_P(GemmTest, trmmLLL) {
    using batmat::linalg::tril;
    using batmat::linalg::trmm;
    for (auto m : batmat::tests::sizes) {
        const auto A = this->template get_matrix<0>(m, m);
        auto D       = this->template get_matrix<0>(m, m);
        trmm(tril(A), tril(A), tril(D));
        for (index_t l = 0; l < A.depth(); ++l) {
            auto Al     = as_eigen(A(l)).template triangularView<Lower>().toDenseMatrix();
            auto Dl_ref = Al * Al;
            EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Dl_ref, ε));
        }
    }
}

TYPED_TEST_P(GemmTest, trmmLLLinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::trmm;
    for (auto m : batmat::tests::sizes) {
        const auto A0 = this->template get_matrix<0>(m, m);
        auto A        = A0;
        trmm(tril(A), tril(A), tril(A));
        for (index_t l = 0; l < A.depth(); ++l) {
            auto Al     = as_eigen(A0(l)).template triangularView<Lower>().toDenseMatrix();
            auto Dl_ref = Al * Al;
            EXPECT_THAT(as_eigen(A(l)), EigenAlmostEqual(Dl_ref, ε));
        }
    }
}
#endif

REGISTER_TYPED_TEST_SUITE_P(GemmTest, gemm, gemmNeg, gemmAdd, gemmSub, gemmSubShiftA,
                            gemmSubShiftCD, gemmSubShiftCDNeg);
REGISTER_TYPED_TEST_SUITE_P(TrmmInplaceTest, trmmLGinplace, trmmUGinplace, trmmGLinplace,
                            trmmGUinplace);
REGISTER_TYPED_TEST_SUITE_P(TrtrsyrkInplaceTest, trmmULLinplace, trmmLUUinplace);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, GemmTest, TestConfigs<OrderConfigs3>);
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, TrmmInplaceTest, TestConfigs<OrderConfigs2>);
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, TrtrsyrkInplaceTest, TestConfigs<OrderConfigs1>);
