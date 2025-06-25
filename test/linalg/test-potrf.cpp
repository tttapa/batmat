#include <batmat/linalg/potrf.hpp>
#include <batmat/matrix/matrix.hpp>
#include <batmat/matrix/view.hpp>
#include <Eigen/Cholesky>
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
class PotrfTest : public ::testing::Test {
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

TYPED_TEST_SUITE_P(PotrfTest);

template <Eigen::UpLoType UpLo, class T>
auto tri(T &&t) -> Eigen::MatrixX<typename T::Scalar> {
    return std::forward<T>(t).template triangularView<UpLo>().toDenseMatrix();
}

TYPED_TEST_P(PotrfTest, potrfLinplace) {
    using batmat::linalg::potrf;
    using batmat::linalg::tril;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));
    for (auto m : batmat::tests::sizes) {
        const auto D0 = [&] {
            auto D = this->get_B(m, m);
            D.view().add_to_diagonal(static_cast<real_t>(100 * m));
            return D;
        }();
        auto D = D0;
        potrf(tril(D));
        for (index_t l = 0; l < D.depth(); ++l) {
            EMat Cl     = as_eigen(D0(l));
            auto Cll    = Cl.template selfadjointView<Eigen::Lower>();
            EMat Dl_ref = Cll.llt().matrixL();
            EXPECT_THAT(tri<Eigen::Lower>(as_eigen(D(l))), EigenAlmostEqual(Dl_ref, ε));
            EXPECT_THAT(tri<Eigen::StrictlyUpper>(as_eigen(D(l))),
                        EigenAlmostEqual(tri<Eigen::StrictlyUpper>(as_eigen(D0(l))), ε));
        }
    }
}

TYPED_TEST_P(PotrfTest, syrkPotrfLinplace) {
    using batmat::linalg::syrk_add_potrf;
    using batmat::linalg::tril;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto D0 = [&] {
                auto D = this->get_B(m, m);
                D.view().add_to_diagonal(static_cast<real_t>(10 * m));
                return D;
            }();
            const auto A = this->get_A(m, n);
            auto D       = D0;
            syrk_add_potrf(A, tril(D), tril(D));
            for (index_t l = 0; l < A.depth(); ++l) {
                EMat Cl  = as_eigen(D0(l));
                auto Cll = Cl.template selfadjointView<Eigen::Lower>();
                Cll.rankUpdate(as_eigen(A(l)));
                EMat Dl_ref = Cll.llt().matrixL();
                EXPECT_THAT(tri<Eigen::Lower>(as_eigen(D(l))), EigenAlmostEqual(Dl_ref, ε));
                EXPECT_THAT(tri<Eigen::StrictlyUpper>(as_eigen(D(l))),
                            EigenAlmostEqual(tri<Eigen::StrictlyUpper>(as_eigen(D0(l))), ε));
            }
        }
}

REGISTER_TYPED_TEST_SUITE_P(PotrfTest, potrfLinplace, syrkPotrfLinplace);

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

INSTANTIATE_TYPED_TEST_SUITE_P(linalg, PotrfTest, AllTestConfigs);
