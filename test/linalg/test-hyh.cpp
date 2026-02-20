#include <batmat/linalg/gemm.hpp>
#include <batmat/linalg/hyhound.hpp>
#include <Eigen/Cholesky>
#include <gtest/gtest.h>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

using batmat::matrix::StorageOrder;
using enum Eigen::UpLoType;

template <class Config>
struct HyhTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(HyhTest);

TYPED_TEST_P(HyhTest, hyhoundL) {
    using batmat::linalg::hyhound_diag;
    using batmat::linalg::syrk;
    using batmat::linalg::tril;
    using EMat = Eigen::MatrixX<typename TypeParam::value_type>;
    for (auto n : batmat::tests::sizes)
        for (auto m : batmat::tests::sizes) {
            const auto L0 = [&] {
                auto L = this->template get_matrix<0>(n, n);
                L.view().add_to_diagonal(static_cast<TypeParam::value_type>(100 * n));
                return L;
            }();
            const auto A0 = this->template get_matrix<1>(n, m);
            const auto d  = this->get_sparse_vector(m, 0.5);
            auto L        = L0;
            auto A        = A0;
            hyhound_diag(tril(L), A, d);
            syrk(tril(L)); // Compute the backward error

            this->check(
                [&](auto &&Al, auto &&dl, auto &&Ll) {
                    EMat LLᵀ = tri<Lower>(Ll) * tri<Lower>(Ll).transpose();
                    LLᵀ += Al * dl.asDiagonal() * Al.transpose();
                    return tri<Lower>(LLᵀ);
                },
                [&](auto l, auto &&res, auto &&ref, auto &&, auto &&, auto &&L0) {
                    const auto resL = tri<Lower>(res), resU = tri<StrictlyUpper>(res);
                    EXPECT_THAT(resL, EigenAlmostEqualRel(ref, this->tolerance_n(n))) << l;
                    EXPECT_THAT(resU, EigenEqual(tri<StrictlyUpper>(L0))) << l;
                },
                L, A0, d, L0);
        }
}

TYPED_TEST_P(HyhTest, hyhoundLapply) {
    using batmat::linalg::hyhound_diag;
    using batmat::linalg::hyhound_diag_apply;
    using batmat::linalg::hyhound_size_W;
    using batmat::linalg::tril;
    using EMat = Eigen::MatrixX<typename TypeParam::value_type>;
    for (auto n : batmat::tests::sizes)
        for (batmat::index_t m : {0, 1, 2, 3, 51})
            for (auto k : batmat::tests::sizes) {
                const auto L0 = [&] {
                    auto L = this->template get_matrix<0>(n, n);
                    L.view().add_to_diagonal(static_cast<TypeParam::value_type>(10 * n));
                    return L;
                }();
                const auto Ls0      = this->template get_matrix<0>(k, n);
                const auto A0       = this->template get_matrix<1>(n, m);
                const auto As0      = this->template get_matrix<1>(k, m);
                const auto d        = this->get_sparse_vector(m, 0.5);
                const auto [rW, cW] = hyhound_size_W(tril(L0));
                auto W              = this->template get_matrix<StorageOrder::ColMajor>(rW, cW);
                auto L              = L0;
                auto A              = A0;
                auto Ls             = Ls0;
                auto As             = this->template get_matrix<1>(k, m);
                hyhound_diag(tril(L), A, d, W);
                hyhound_diag_apply(Ls, As0, As, A, d, W);

                this->check(
                    [&](auto &&Al, auto &&dl, auto &&Ll, auto &&Lsl, auto &&Asl) -> EMat {
                        EMat LLᵀ(n + k, n + k);
                        LLᵀ.topLeftCorner(n, n)     = tri<Lower>(Ll) * tri<Lower>(Ll).transpose();
                        LLᵀ.bottomLeftCorner(k, n)  = Lsl * tri<Lower>(Ll).transpose();
                        LLᵀ.bottomRightCorner(k, k) = 1e6 * EMat::Identity(k, k);
                        LLᵀ.topLeftCorner(n, n) += Al * dl.asDiagonal() * Al.transpose();
                        LLᵀ.bottomLeftCorner(k, n) += Asl * dl.asDiagonal() * Al.transpose();
                        return LLᵀ.template selfadjointView<Lower>()
                            .llt()
                            .matrixL()
                            .toDenseMatrix()
                            .bottomLeftCorner(k, n);
                    },
                    [&](auto l, auto &&res, auto &&ref, auto &&...) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                    },
                    Ls, A0, d, L0, Ls0, As0);
            }
}

TYPED_TEST_P(HyhTest, hyhoundLapplyOffset) {
    using batmat::linalg::hyhound_diag;
    using batmat::linalg::hyhound_diag_apply;
    using batmat::linalg::hyhound_size_W;
    using batmat::linalg::tril;
    using EMat = Eigen::MatrixX<typename TypeParam::value_type>;
    for (auto n : batmat::tests::sizes)
        for (batmat::index_t m : {0, 1, 4, 8, 16, 51})
            for (auto k : batmat::tests::sizes) {
                const auto L0 = [&] {
                    auto L = this->template get_matrix<0>(n, n);
                    L.view().add_to_diagonal(static_cast<TypeParam::value_type>(10 * n));
                    return L;
                }();
                const auto Ls0      = this->template get_matrix<0>(k, n);
                const auto A0       = this->template get_matrix<1>(n, m);
                const auto As0      = this->template get_matrix<1>(k, m - m / 4);
                const auto d        = this->get_sparse_vector(m, 0.5);
                const auto [rW, cW] = hyhound_size_W(tril(L0));
                auto W              = this->template get_matrix<StorageOrder::ColMajor>(rW, cW);
                auto L              = L0;
                auto A              = A0;
                auto Ls             = Ls0;
                auto As             = this->template get_matrix<1>(k, m);
                hyhound_diag(tril(L), A, d, W);
                hyhound_diag_apply(Ls, As0, As, A, d, W, m / 8);

                this->check(
                    [&](auto &&Al, auto &&dl, auto &&Ll, auto &&Lsl, auto &&Asl) -> EMat {
                        EMat LLᵀ(n + k, n + k);
                        LLᵀ.topLeftCorner(n, n)     = tri<Lower>(Ll) * tri<Lower>(Ll).transpose();
                        LLᵀ.bottomLeftCorner(k, n)  = Lsl * tri<Lower>(Ll).transpose();
                        LLᵀ.bottomRightCorner(k, k) = 1e6 * EMat::Identity(k, k);
                        LLᵀ.topLeftCorner(n, n) += Al * dl.asDiagonal() * Al.transpose();
                        EMat dl0 = dl.middleRows(m / 8, m - m / 4);
                        EMat Al0 = Al.middleCols(m / 8, m - m / 4);
                        LLᵀ.bottomLeftCorner(k, n) += Asl * dl0.asDiagonal() * Al0.transpose();
                        return LLᵀ.template selfadjointView<Lower>()
                            .llt()
                            .matrixL()
                            .toDenseMatrix()
                            .bottomLeftCorner(k, n);
                    },
                    [&](auto l, auto &&res, auto &&ref, auto &&...) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance)) << l;
                    },
                    Ls, A0, d, L0, Ls0, As0);
            }
}

REGISTER_TYPED_TEST_SUITE_P(HyhTest, hyhoundL, hyhoundLapply, hyhoundLapplyOffset);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, HyhTest, TestConfigs<OrderConfigs2>);
