#include <batmat/linalg/sterf.hpp>
#include <batmat/linalg/sytrd.hpp>
#include <gtest/gtest.h>
#include <print>

#include "config.hpp"
#include "eigen-matchers.hpp"
#include "fixtures.hpp"

BATMAT_PRAGMA_GCC_OPTIMIZE_O3_BEGIN
#include <Eigen/Eigenvalues>
BATMAT_PRAGMA_GCC_OPTIMIZE_O3_END

using batmat::matrix::StorageOrder;
using enum Eigen::UpLoType;

template <class Config>
struct EigvalsTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(EigvalsTest);

using batmat::index_t;
using batmat::linalg::extract_bidiag;
using batmat::linalg::sterf;
using batmat::linalg::sytrd;
using batmat::linalg::sytrd_apply;
using batmat::linalg::sytrd_size_W;
using batmat::linalg::sytrd_size_Y;
using batmat::linalg::tril;

TYPED_TEST_P(EigvalsTest, sterfRandom) {
    using EVec = Eigen::VectorX<typename TypeParam::value_type>;
    for (auto m : batmat::tests::sizes) {
        if (m == 0)
            continue;
        const auto A0 = this->template get_matrix<0>(m, m);
        auto [rw, cw] = sytrd_size_W(A0);
        auto W        = this->template get_matrix<StorageOrder::ColMajor>(rw, cw);
        auto [ry, cy] = sytrd_size_Y(A0);
        auto Y        = this->template get_matrix<StorageOrder::ColMajor>(ry, cy);
        auto A        = A0;
        W.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
        Y.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
        auto d = this->get_vector(m);
        auto e = this->get_vector(m - 1);
        batmat::linalg::SterfOptions options{.max_iterations_per_eigenvalue = 5};

        // Tridiagonalize A in-place
        sytrd(tril(A), W, Y);
        extract_bidiag(tril(A), d, e);
        auto num_iter = sterf(d, e, options);
        ASSERT_TRUE(num_iter) << "Convergence failure (" << num_iter.error() << " iterations)";
        EXPECT_LE(*num_iter, options.max_iterations_per_eigenvalue * m)
            << "Too many iterations (" << *num_iter << ")";
        std::println("Tridiagonal eigvals of size {} computed in {} iterations", m, *num_iter);

        this->check(
            [&](auto &&Al) -> EVec {
                return Al.template selfadjointView<Eigen::Lower>().eigenvalues();
            },
            [&](auto l, EVec res, EVec ref, auto &&) {
                std::sort(res.begin(), res.end());
                std::sort(ref.begin(), ref.end());
                EXPECT_THAT(res, EigenAlmostEqualRel(ref, this->tolerance_n(m))) << l;
            },
            d, A0);
    }
}

TYPED_TEST_P(EigvalsTest, sterfDiag) {
    using EVec = Eigen::VectorX<typename TypeParam::value_type>;
    for (auto m : batmat::tests::sizes) {
        if (m == 0)
            continue;
        const auto A0 = [&] {
            auto A0 = this->template get_matrix<0>(m, m);
            for (index_t i = 0; i < m; ++i)
                for (index_t j = 0; j < m; ++j)
                    if (i != j)
                        A0(0, i, j) = 0;
            return A0;
        }();
        auto [rw, cw] = sytrd_size_W(A0);
        auto W        = this->template get_matrix<StorageOrder::ColMajor>(rw, cw);
        auto [ry, cy] = sytrd_size_Y(A0);
        auto Y        = this->template get_matrix<StorageOrder::ColMajor>(ry, cy);
        auto A        = A0;
        W.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
        Y.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
        auto d = this->get_vector(m);
        auto e = this->get_vector(m - 1);
        batmat::linalg::SterfOptions options{.max_iterations_per_eigenvalue = 5};

        // Tridiagonalize A in-place
        sytrd(tril(A), W, Y);
        extract_bidiag(tril(A), d, e);
        auto num_iter = sterf(d, e, options);
        ASSERT_TRUE(num_iter) << "Convergence failure (" << num_iter.error() << " iterations)";
        EXPECT_LE(*num_iter, options.max_iterations_per_eigenvalue * m)
            << "Too many iterations (" << *num_iter << ")";
        std::println("Tridiagonal eigvals of size {} computed in {} iterations", m, *num_iter);

        this->check(
            [&](auto &&Al) -> EVec {
                return Al.template selfadjointView<Eigen::Lower>().eigenvalues();
            },
            [&](auto l, EVec res, EVec ref, auto &&) {
                std::sort(res.begin(), res.end());
                std::sort(ref.begin(), ref.end());
                EXPECT_THAT(res, EigenAlmostEqualRel(ref, this->tolerance_n(m))) << l;
            },
            d, A0);
    }
}

TYPED_TEST_P(EigvalsTest, sterfZero) {
    using EVec = Eigen::VectorX<typename TypeParam::value_type>;
    for (auto m : batmat::tests::sizes) {
        if (m == 0)
            continue;
        const auto A0 = [&] {
            auto A0 = this->template get_matrix<0>(m, m);
            for (index_t i = 0; i < m; ++i)
                for (index_t j = 0; j < m; ++j)
                    A0(0, i, j) = 0;
            return A0;
        }();
        auto [rw, cw] = sytrd_size_W(A0);
        auto W        = this->template get_matrix<StorageOrder::ColMajor>(rw, cw);
        auto [ry, cy] = sytrd_size_Y(A0);
        auto Y        = this->template get_matrix<StorageOrder::ColMajor>(ry, cy);
        auto A        = A0;
        W.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
        Y.set_constant(std::numeric_limits<typename TypeParam::value_type>::quiet_NaN());
        auto d = this->get_vector(m);
        auto e = this->get_vector(m - 1);
        batmat::linalg::SterfOptions options{.max_iterations_per_eigenvalue = 5};

        // Tridiagonalize A in-place
        sytrd(tril(A), W, Y);
        extract_bidiag(tril(A), d, e);
        auto num_iter = sterf(d, e, options);
        ASSERT_TRUE(num_iter) << "Convergence failure (" << num_iter.error() << " iterations)";
        EXPECT_LE(*num_iter, options.max_iterations_per_eigenvalue * m)
            << "Too many iterations (" << *num_iter << ")";
        std::println("Tridiagonal eigvals of size {} computed in {} iterations", m, *num_iter);

        this->check(
            [&](auto &&Al) -> EVec {
                return Al.template selfadjointView<Eigen::Lower>().eigenvalues();
            },
            [&](auto l, EVec res, EVec ref, auto &&) {
                std::sort(res.begin(), res.end());
                std::sort(ref.begin(), ref.end());
                EXPECT_THAT(res, EigenAlmostEqual(ref, this->tolerance_n(m))) << l;
            },
            d, A0);
    }
}

REGISTER_TYPED_TEST_SUITE_P(EigvalsTest, sterfRandom, sterfDiag, sterfZero);

using namespace batmat::tests;
INSTANTIATE_TYPED_TEST_SUITE_P(linalg, EigvalsTest, TestConfigs<OrderConfigs1>);
