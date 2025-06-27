#include <batmat/linalg/gemm-diag.hpp>
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

template <class T, index_t N, StorageOrder OA, StorageOrder OB, StorageOrder OC>
struct TestConfig {
    using value_type                      = T;
    using batch_size                      = std::integral_constant<index_t, N>;
    static constexpr StorageOrder order_A = OA, order_B = OB, order_C = OC;
};

template <class Config>
class GemmDiagTest : public ::testing::Test {
  protected:
    using value_type = typename Config::value_type;
    using batch_size = typename Config::batch_size;

    template <StorageOrder O>
    using Matrix = batmat::matrix::Matrix<value_type, index_t, batch_size, batch_size, O>;

    std::mt19937 rng{12345};
    std::normal_distribution<value_type> nrml{0, 1};
    std::bernoulli_distribution brnl{0.5};
    void SetUp() override { rng.seed(12345); }

    template <StorageOrder O>
    auto get_matrix(index_t r, index_t c) {
        Matrix<O> a{{.rows = r, .cols = c}};
        std::ranges::generate(a, [this] { return nrml(rng); });
        return a;
    }

    auto get_A(index_t r, index_t c) { return get_matrix<Config::order_A>(r, c); }
    auto get_B(index_t r, index_t c) { return get_matrix<Config::order_B>(r, c); }
    auto get_C(index_t r, index_t c) { return get_matrix<Config::order_C>(r, c); }
    auto get_d(index_t r) {
        auto d = get_matrix<StorageOrder::ColMajor>(r, 1);
        for (auto &di : d)
            if (brnl(rng))
                di = 0;
        return d;
    }
};

TYPED_TEST_SUITE_P(GemmDiagTest);

template <Eigen::UpLoType UpLo, class T>
auto tri(T &&t) -> Eigen::MatrixX<typename std::remove_cvref_t<T>::Scalar> {
    return std::forward<T>(t).template triangularView<UpLo>().toDenseMatrix();
}

TYPED_TEST_P(GemmDiagTest, gemm) {
    using batmat::linalg::gemm_diag;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = 5000 * std::numeric_limits<real_t>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->get_A(m, k);
                const auto B = this->get_B(k, n);
                auto C       = this->get_C(m, n);
                const auto d = this->get_d(k);
                gemm_diag(A, B, C, d);
                for (index_t l = 0; l < A.depth(); ++l) {
                    EMat Cl_ref = as_eigen(A(l)) * as_eigen(d(l)).asDiagonal() * as_eigen(B(l));
                    EXPECT_THAT(as_eigen(C(l)), EigenAlmostEqual(Cl_ref, ε));
                }
            }
}

TYPED_TEST_P(GemmDiagTest, gemmTrackZeros) {
    using batmat::linalg::gemm_diag;
    using batmat::linalg::track_zeros;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = 5000 * std::numeric_limits<real_t>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->get_A(m, k);
                const auto B = this->get_B(k, n);
                auto C       = this->get_C(m, n);
                const auto d = this->get_d(k);
                gemm_diag(A, B, C, d, track_zeros<>);
                for (index_t l = 0; l < A.depth(); ++l) {
                    EMat Cl_ref = as_eigen(A(l)) * as_eigen(d(l)).asDiagonal() * as_eigen(B(l));
                    EXPECT_THAT(as_eigen(C(l)), EigenAlmostEqual(Cl_ref, ε));
                }
            }
}

TYPED_TEST_P(GemmDiagTest, syrkAdd) {
    using batmat::linalg::syrk_diag_add;
    using batmat::linalg::tril;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = 5000 * std::numeric_limits<real_t>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            const auto A  = this->get_A(m, k);
            const auto C0 = this->get_C(m, m);
            const auto d  = this->get_d(k);
            auto C        = C0;
            syrk_diag_add(A, tril(C), d);
            for (index_t l = 0; l < A.depth(); ++l) {
                const auto Al = as_eigen(A(l));
                EMat Cl_ref   = as_eigen(C0(l));
                Cl_ref += Al * as_eigen(d(l)).asDiagonal() * Al.transpose();
                EXPECT_THAT(tri<Eigen::Lower>(as_eigen(C(l))),
                            EigenAlmostEqual(tri<Eigen::Lower>(Cl_ref), ε));
            }
        }
}

TYPED_TEST_P(GemmDiagTest, syrkAddTrackZeros) {
    using batmat::linalg::syrk_diag_add;
    using batmat::linalg::track_zeros;
    using batmat::linalg::tril;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = 5000 * std::numeric_limits<real_t>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto k : batmat::tests::sizes) {
            const auto A  = this->get_A(m, k);
            const auto C0 = this->get_C(m, m);
            const auto d  = this->get_d(k);
            auto C        = C0;
            syrk_diag_add(A, tril(C), d, track_zeros<>);
            for (index_t l = 0; l < A.depth(); ++l) {
                const auto Al = as_eigen(A(l));
                EMat Cl_ref   = as_eigen(C0(l));
                Cl_ref += Al * as_eigen(d(l)).asDiagonal() * Al.transpose();
                EXPECT_THAT(tri<Eigen::Lower>(as_eigen(C(l))),
                            EigenAlmostEqual(tri<Eigen::Lower>(Cl_ref), ε));
            }
        }
}

REGISTER_TYPED_TEST_SUITE_P(GemmDiagTest, gemm, gemmTrackZeros, syrkAdd, syrkAddTrackZeros);

using enum batmat::matrix::StorageOrder;
template <class T, index_t N>
using TestConfigs = ::testing::Types<
    TestConfig<T, N, ColMajor, ColMajor, ColMajor>,
#if BATMAT_EXTENSIVE_TESTS
    TestConfig<T, N, ColMajor, ColMajor, RowMajor>, TestConfig<T, N, ColMajor, RowMajor, ColMajor>,
    TestConfig<T, N, ColMajor, RowMajor, RowMajor>, TestConfig<T, N, RowMajor, ColMajor, ColMajor>,
    TestConfig<T, N, RowMajor, ColMajor, RowMajor>, TestConfig<T, N, RowMajor, RowMajor, ColMajor>,
#endif
    TestConfig<T, N, RowMajor, RowMajor, RowMajor>>;
using AllTestConfigs = typename CatTypes<TestConfigs<double, 1>, TestConfigs<double, 4>,
                                         TestConfigs<float, 1>, TestConfigs<float, 8>>::type;

INSTANTIATE_TYPED_TEST_SUITE_P(linalg, GemmDiagTest, AllTestConfigs);
