#include <gtest/gtest.h>

#include <batmat/linalg/gemm-diag.hpp>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

using enum Eigen::UpLoType;

template <class Config>
struct GemmDiagTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(GemmDiagTest);

TYPED_TEST_P(GemmDiagTest, gemm) {
    using batmat::linalg::gemm_diag;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->template get_matrix<0>(m, k);
                const auto B = this->template get_matrix<1>(k, n);
                auto C       = this->template get_matrix<2>(m, n);
                const auto d = this->get_sparse_vector(k);
                gemm_diag(A, B, C, d);
                this->check(
                    [&](auto &&Al, auto &&Bl, auto &&dl) { return Al * dl.asDiagonal() * Bl; },
                    [&](auto l, auto &&res, auto &&ref, auto &&...) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                    },
                    C, A, B, d);
            }
}

TYPED_TEST_P(GemmDiagTest, gemmTrackZeros) {
    using batmat::linalg::gemm_diag;
    using batmat::linalg::track_zeros;
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->template get_matrix<0>(m, k);
                const auto B = this->template get_matrix<1>(k, n);
                auto C       = this->template get_matrix<2>(m, n);
                const auto d = this->get_sparse_vector(k);
                gemm_diag(A, B, C, d, track_zeros<>);
                this->check(
                    [&](auto &&Al, auto &&Bl, auto &&dl) { return Al * dl.asDiagonal() * Bl; },
                    [&](auto l, auto &&res, auto &&ref, auto &&...) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                    },
                    C, A, B, d);
            }
}

template <class Config>
struct SyrkDiagTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(SyrkDiagTest);

TYPED_TEST_P(SyrkDiagTest, syrkAdd) {
    using batmat::linalg::syrk_diag_add;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            const auto A  = this->template get_matrix<0>(m, k);
            const auto C0 = this->template get_matrix<1>(m, m);
            const auto d  = this->get_sparse_vector(k);
            auto C        = C0;
            syrk_diag_add(A, tril(C), d);
            this->check(
                [&](auto &&C0l, auto &&Al, auto &&dl) {
                    return C0l + Al * dl.asDiagonal() * Al.transpose();
                },
                [&](auto l, auto &&res, auto &&ref, auto &&C0l, auto &&...) {
                    const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                    EXPECT_THAT(resL, EigenAlmostEqual(tri<Lower>(ref), this->tolerance)) << l;
                    EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(C0l), this->tolerance))
                        << l;
                },
                C, C0, A, d);
        }
}

TYPED_TEST_P(SyrkDiagTest, syrkAddTrackZeros) {
    using batmat::linalg::syrk_diag_add;
    using batmat::linalg::track_zeros;
    using batmat::linalg::tril;
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            const auto A  = this->template get_matrix<0>(m, k);
            const auto C0 = this->template get_matrix<1>(m, m);
            const auto d  = this->get_sparse_vector(k);
            auto C        = C0;
            syrk_diag_add(A, tril(C), d);
            this->check(
                [&](auto &&C0l, auto &&Al, auto &&dl) {
                    return C0l + Al * dl.asDiagonal() * Al.transpose();
                },
                [&](auto l, auto &&res, auto &&ref, auto &&C0l, auto &&...) {
                    const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                    EXPECT_THAT(resL, EigenAlmostEqual(tri<Lower>(ref), this->tolerance)) << l;
                    EXPECT_THAT(resU, EigenAlmostEqual(tri<StrictlyUpper>(C0l), this->tolerance))
                        << l;
                },
                C, C0, A, d);
        }
}

REGISTER_TYPED_TEST_SUITE_P(GemmDiagTest, gemm, gemmTrackZeros);
REGISTER_TYPED_TEST_SUITE_P(SyrkDiagTest, syrkAdd, syrkAddTrackZeros);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, GemmDiagTest, TestConfigs<OrderConfigs3>);
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, SyrkDiagTest, TestConfigs<OrderConfigs2>);
