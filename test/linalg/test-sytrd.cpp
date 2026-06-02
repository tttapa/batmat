#include <batmat/linalg/sytrd.hpp>
#include <gtest/gtest.h>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

using batmat::matrix::StorageOrder;
using enum Eigen::UpLoType;

template <class Config>
struct TridiagTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(TridiagTest);

using batmat::index_t;
using batmat::linalg::sytrd;
using batmat::linalg::sytrd_apply;
using batmat::linalg::sytrd_size_W;
using batmat::linalg::sytrd_size_Y;
using batmat::linalg::tril;

TYPED_TEST_P(TridiagTest, sytrdRandom) {
    using EMat = Eigen::MatrixX<typename TypeParam::value_type>;
    for (auto m : batmat::tests::sizes) {
        const auto A0 = this->template get_matrix<0>(m, m);
        auto [rw, cw] = sytrd_size_W(A0);
        auto W        = this->template get_matrix<StorageOrder::ColMajor>(rw, cw);
        auto [ry, cy] = sytrd_size_Y(A0);
        auto Y        = this->template get_matrix<StorageOrder::ColMajor>(ry, cy);
        auto A        = A0;
        W.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
        Y.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());

        // Tridiagonalize A in-place
        sytrd(tril(A), W, Y);

        // Extract the tridiagonal part of A into T
        auto T = this->template get_matrix<1>(m, m);
        T.set_constant(0);
        for (index_t i = 0; i < m; ++i) {
            for (index_t l = 0; l < A.depth(); ++l) {
                T(l, i, i) = A(l, i, i);
                if (i + 1 < m)
                    T(l, i, i + 1) = T(l, i + 1, i) = A(l, i + 1, i);
            }
        }

        // Reconstruct the original matrix
        auto QT = this->template get_matrix<2>(m, m);
        sytrd_apply(T, QT, tril(A), W, false);
        auto QTQᵀ = this->template get_matrix<2>(m, m);
        sytrd_apply(QT.transposed(), QTQᵀ.transposed(), tril(A), W, false); // QTQᵀ = (Q(QT)ᵀ)ᵀ

        this->check([&](auto &&Al) -> EMat { return Al.template selfadjointView<Eigen::Lower>(); },
                    [&](auto l, auto &&res, auto &&ref, auto &&) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance_n(m))) << l;
                    },
                    QTQᵀ, A0);

        // Check the orthogonality of Q
        auto I = this->template get_matrix<1>(m, m);
        I.set_constant(0);
        I.set_diagonal(1);
        auto Q = this->template get_matrix<2>(m, m);
        sytrd_apply(I, Q, tril(A), W, false);
        auto QᵀQ = this->template get_matrix<2>(m, m);
        sytrd_apply(Q, QᵀQ, tril(A), W, true);

        this->check([&] -> EMat { return EMat::Identity(m, m); },
                    [&](auto l, auto &&res, auto &&ref) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance_n(m))) << l;
                    },
                    QᵀQ);

        // Check that application of Qᵀ correctly results in the transpose of Q
        auto Qᵀ = this->template get_matrix<2>(m, m);
        sytrd_apply(I, Qᵀ, tril(A), W, true);

        this->check([&](auto &&Q) -> EMat { return Q.transpose(); },
                    [&](auto l, auto &&res, auto &&ref, auto &&) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance_n(m))) << l;
                    },
                    Qᵀ, Q);
    }
}

TYPED_TEST_P(TridiagTest, sytrdZeroOffDiag) {
    using EMat = Eigen::MatrixX<typename TypeParam::value_type>;
    for (auto m : batmat::tests::sizes) {
        const auto A0 = [&] {
            auto A0 = this->template get_matrix<0>(m, m);
            if (m > 1)
                batmat::linalg::fill(0, tril(A0.block(1, 0, m - 1, m / 2)));
            return A0;
        }();
        auto [rw, cw] = sytrd_size_W(A0);
        auto W        = this->template get_matrix<StorageOrder::ColMajor>(rw, cw);
        auto [ry, cy] = sytrd_size_Y(A0);
        auto Y        = this->template get_matrix<StorageOrder::ColMajor>(ry, cy);
        auto A        = A0;
        W.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
        Y.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());

        // Tridiagonalize A in-place
        sytrd(tril(A), W, Y);

        // Extract the tridiagonal part of A into T
        auto T = this->template get_matrix<1>(m, m);
        T.set_constant(0);
        for (index_t i = 0; i < m; ++i) {
            for (index_t l = 0; l < A.depth(); ++l) {
                T(l, i, i) = A(l, i, i);
                if (i + 1 < m)
                    T(l, i, i + 1) = T(l, i + 1, i) = A(l, i + 1, i);
            }
        }

        // Reconstruct the original matrix
        auto QT = this->template get_matrix<2>(m, m);
        sytrd_apply(T, QT, tril(A), W, false);
        auto QTQᵀ = this->template get_matrix<2>(m, m);
        sytrd_apply(QT.transposed(), QTQᵀ.transposed(), tril(A), W, false); // QTQᵀ = (Q(QT)ᵀ)ᵀ

        this->check([&](auto &&Al) -> EMat { return Al.template selfadjointView<Eigen::Lower>(); },
                    [&](auto l, auto &&res, auto &&ref, auto &&) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance_n(m))) << l;
                    },
                    QTQᵀ, A0);

        // Check the orthogonality of Q
        auto I = this->template get_matrix<1>(m, m);
        I.set_constant(0);
        I.set_diagonal(1);
        auto Q = this->template get_matrix<2>(m, m);
        sytrd_apply(I, Q, tril(A), W, false);
        auto QᵀQ = this->template get_matrix<2>(m, m);
        sytrd_apply(Q, QᵀQ, tril(A), W, true);

        this->check([&] -> EMat { return EMat::Identity(m, m); },
                    [&](auto l, auto &&res, auto &&ref) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance_n(m))) << l;
                    },
                    QᵀQ);

        // Check that application of Qᵀ correctly results in the transpose of Q
        auto Qᵀ = this->template get_matrix<2>(m, m);
        sytrd_apply(I, Qᵀ, tril(A), W, true);

        this->check([&](auto &&Q) -> EMat { return Q.transpose(); },
                    [&](auto l, auto &&res, auto &&ref, auto &&) {
                        EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance_n(m))) << l;
                    },
                    Qᵀ, Q);
    }
}

REGISTER_TYPED_TEST_SUITE_P(TridiagTest, sytrdRandom, sytrdZeroOffDiag);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, TridiagTest, TestConfigs<OrderConfigs3>);
