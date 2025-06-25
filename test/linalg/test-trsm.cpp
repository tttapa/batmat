#include <batmat/linalg/trsm.hpp>
#include <batmat/matrix/matrix.hpp>
#include <batmat/matrix/view.hpp>
#include <gtest/gtest.h>
#include <guanaqo/eigen/view.hpp>
#include <algorithm>
#include <limits>
#include <random>

#include "config.hpp"
#include "eigen-matchers.hpp"

using batmat::index_t;
using batmat::matrix::DefaultStride;
using batmat::matrix::StorageOrder;
using batmat::tests::CatTypes;

template <class T, index_t N, StorageOrder OA, StorageOrder OB>
struct TestConfig {
    using value_type                      = T;
    using batch_size                      = std::integral_constant<index_t, N>;
    static constexpr StorageOrder order_A = OA, order_B = OB;
};

template <class Config>
class TrsmTest : public ::testing::Test {
  protected:
    using value_type = typename Config::value_type;
    using batch_size = typename Config::batch_size;

    template <StorageOrder O>
    using Matrix = batmat::matrix::Matrix<value_type, index_t, batch_size, batch_size, O>;

    std::mt19937 rng{12345};
    std::normal_distribution<value_type> nrml{0, 1};
    void SetUp() override { rng.seed(12345); }

    template <StorageOrder O>
    auto get_matrix(index_t r, index_t c) {
        Matrix<O> a{{.rows = r, .cols = c}};
        std::ranges::generate(a, [this] { return nrml(rng); });
        return a;
    }

    auto get_A(index_t r, index_t c) { return get_matrix<Config::order_A>(r, c); }
    auto get_B(index_t r, index_t c) { return get_matrix<Config::order_B>(r, c); }
};

TYPED_TEST_SUITE_P(TrsmTest);

TYPED_TEST_P(TrsmTest, trsmLLinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::trsm;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->get_A(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto D0 = this->get_B(m, n);
            auto D        = D0;
            trsm(tril(A), D, D);
            for (index_t l = 0; l < A.depth(); ++l) {
                EMat Dl_ref =
                    as_eigen(A(l)).template triangularView<Eigen::Lower>().solve(as_eigen(D0(l)));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Dl_ref, ε));
            }
        }
}

TYPED_TEST_P(TrsmTest, trsmLUinplace) {
    using batmat::linalg::triu;
    using batmat::linalg::trsm;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->get_A(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto D0 = this->get_B(m, n);
            auto D        = D0;
            trsm(triu(A), D, D);
            for (index_t l = 0; l < A.depth(); ++l) {
                EMat Dl_ref =
                    as_eigen(A(l)).template triangularView<Eigen::Upper>().solve(as_eigen(D0(l)));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Dl_ref, ε));
            }
        }
}

TYPED_TEST_P(TrsmTest, trsmRUinplace) {
    using batmat::linalg::triu;
    using batmat::linalg::trsm;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->get_A(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto D0 = this->get_B(n, m);
            auto D        = D0;
            trsm(D, triu(A), D);
            for (index_t l = 0; l < A.depth(); ++l) {
                EMat Dl_ref = as_eigen(A(l))
                                  .template triangularView<Eigen::Upper>()
                                  .transpose()
                                  .solve(as_eigen(D0(l)).transpose())
                                  .transpose();
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Dl_ref, ε));
            }
        }
}

TYPED_TEST_P(TrsmTest, trsmRLinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::trsm;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A = [&] {
                auto A = this->get_A(m, m);
                A.view().add_to_diagonal(10);
                return A;
            }();
            const auto D0 = this->get_B(n, m);
            auto D        = D0;
            trsm(D, tril(A), D);
            for (index_t l = 0; l < A.depth(); ++l) {
                EMat Dl_ref = as_eigen(A(l))
                                  .template triangularView<Eigen::Lower>()
                                  .transpose()
                                  .solve(as_eigen(D0(l)).transpose())
                                  .transpose();
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Dl_ref, ε));
            }
        }
}

REGISTER_TYPED_TEST_SUITE_P(TrsmTest, trsmLLinplace, trsmLUinplace, trsmRUinplace, trsmRLinplace);

using enum batmat::matrix::StorageOrder;
template <class T, index_t N>
using TestConfigs =
    ::testing::Types<TestConfig<T, N, ColMajor, ColMajor>,
#if BATMAT_EXTENSIVE_TESTS
                     TestConfig<T, N, ColMajor, ColMajor>, TestConfig<T, N, ColMajor, RowMajor>,
                     TestConfig<T, N, RowMajor, ColMajor>,
#endif
                     TestConfig<T, N, RowMajor, RowMajor>>;
using AllTestConfigs = typename CatTypes<TestConfigs<double, 1>, TestConfigs<double, 4>,
                                         TestConfigs<float, 1>, TestConfigs<float, 8>>::type;

INSTANTIATE_TYPED_TEST_SUITE_P(linalg, TrsmTest, AllTestConfigs);
