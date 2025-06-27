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
#include "fixtures.hpp"

using batmat::index_t;
using batmat::matrix::DefaultStride;
using batmat::matrix::StorageOrder;
using batmat::tests::CatTypes;

template <class Config>
struct GemmDiagTest : batmat::tests::LinalgTest<Config> {};
TYPED_TEST_SUITE_P(GemmDiagTest);

TYPED_TEST_P(GemmDiagTest, gemm) {
    using batmat::linalg::gemm_diag;
    using real_t = typename TestFixture::value_type;
    using EMat   = Eigen::MatrixX<real_t>;
    const auto ε = 5000 * std::numeric_limits<real_t>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->template get_matrix<0>(m, k);
                const auto B = this->template get_matrix<1>(k, n);
                auto C       = this->template get_matrix<2>(m, n);
                const auto d = this->get_sparse_vector(k);
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
