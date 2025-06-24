#include <batmat/linalg/gemm.hpp>
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
class GemmTest : public ::testing::Test {
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
    auto get_C(index_t r, index_t c) { return get_matrix<Config::order_C>(r, c); }
};

TYPED_TEST_SUITE_P(GemmTest);

TYPED_TEST_P(GemmTest, gemm) {
    using batmat::linalg::gemm;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->get_A(m, k);
                const auto B = this->get_B(k, n);
                auto C       = this->get_C(m, n);
                gemm(A, B, C);
                for (index_t l = 0; l < A.depth(); ++l) {
                    auto Cl_ref = as_eigen(A(l)) * as_eigen(B(l));
                    EXPECT_THAT(as_eigen(C(l)), EigenAlmostEqual(Cl_ref, ε));
                }
            }
}

TYPED_TEST_P(GemmTest, gemmSub) {
    using batmat::linalg::gemm_sub;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->get_A(m, k);
                const auto B = this->get_B(k, n);
                const auto C = this->get_C(m, n);
                auto D       = this->get_C(m, n);
                gemm_sub(A, B, C, D);
                for (index_t l = 0; l < A.depth(); ++l) {
                    auto Cl_ref = as_eigen(C(l)) - as_eigen(A(l)) * as_eigen(B(l));
                    EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Cl_ref, ε));
                }
            }
}

#if BATMAT_EXTENSIVE_TESTS
TYPED_TEST_P(GemmTest, gemmNeg) {
    using batmat::linalg::gemm_neg;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->get_A(m, k);
                const auto B = this->get_B(k, n);
                auto C       = this->get_C(m, n);
                gemm_neg(A, B, C);
                for (index_t l = 0; l < A.depth(); ++l) {
                    auto Cl_ref = -as_eigen(A(l)) * as_eigen(B(l));
                    EXPECT_THAT(as_eigen(C(l)), EigenAlmostEqual(Cl_ref, ε));
                }
            }
}

TYPED_TEST_P(GemmTest, gemmAdd) {
    using batmat::linalg::gemm_add;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                const auto A = this->get_A(m, k);
                const auto B = this->get_B(k, n);
                const auto C = this->get_C(m, n);
                auto D       = this->get_C(m, n);
                gemm_add(A, B, C, D);
                for (index_t l = 0; l < A.depth(); ++l) {
                    auto Cl_ref = as_eigen(C(l)) + as_eigen(A(l)) * as_eigen(B(l));
                    EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Cl_ref, ε));
                }
            }
}

TYPED_TEST_P(GemmTest, gemmSubShiftA) {
    using batmat::linalg::gemm_sub;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                auto A       = this->get_A(m, k);
                const auto B = this->get_B(k, n);
                const auto C = this->get_C(m, n);
                auto D       = this->get_C(m, n);
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
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                auto A       = this->get_A(m, k);
                const auto B = this->get_B(k, n);
                const auto C = this->get_C(m, n);
                auto D       = this->get_C(m, n);
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
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes)
            for (auto k : batmat::tests::sizes) {
                auto A       = this->get_A(m, k);
                const auto B = this->get_B(k, n);
                const auto C = this->get_C(m, n);
                auto D       = this->get_C(m, n);
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

template <Eigen::UpLoType UpLo, class T>
auto tri(T &&t) -> Eigen::MatrixX<typename T::Scalar> {
    return std::forward<T>(t).template triangularView<UpLo>().toDenseMatrix();
}

TYPED_TEST_P(GemmTest, trmmLGinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::trmm;
    using EMat   = Eigen::MatrixX<typename TestFixture::value_type>;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A  = this->get_A(m, m);
            const auto D0 = this->get_B(m, n);
            auto D        = D0;
            trmm(tril(A), D, D);
            for (index_t l = 0; l < A.depth(); ++l) {
                EMat Dl_ref = tri<Eigen::Lower>(as_eigen(A(l))) * as_eigen(D0(l));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Dl_ref, ε));
            }
        }
}

TYPED_TEST_P(GemmTest, trmmUGinplace) {
    using batmat::linalg::triu;
    using batmat::linalg::trmm;
    using EMat   = Eigen::MatrixX<typename TestFixture::value_type>;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A  = this->get_A(m, m);
            const auto D0 = this->get_B(m, n);
            auto D        = D0;
            trmm(triu(A), D, D);
            for (index_t l = 0; l < A.depth(); ++l) {
                EMat Dl_ref = tri<Eigen::Upper>(as_eigen(A(l))) * as_eigen(D0(l));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Dl_ref, ε));
            }
        }
}

TYPED_TEST_P(GemmTest, trmmGLinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::trmm;
    using EMat   = Eigen::MatrixX<typename TestFixture::value_type>;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A  = this->get_A(m, m);
            const auto D0 = this->get_B(n, m);
            auto D        = D0;
            trmm(D, tril(A), D);
            for (index_t l = 0; l < A.depth(); ++l) {
                EMat Dl_ref = as_eigen(D0(l)) * tri<Eigen::Lower>(as_eigen(A(l)));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Dl_ref, ε));
            }
        }
}

TYPED_TEST_P(GemmTest, trmmGUinplace) {
    using batmat::linalg::triu;
    using batmat::linalg::trmm;
    using EMat   = Eigen::MatrixX<typename TestFixture::value_type>;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes)
        for (auto n : batmat::tests::sizes) {
            const auto A  = this->get_A(m, m);
            const auto D0 = this->get_B(n, m);
            auto D        = D0;
            trmm(D, triu(A), D);
            for (index_t l = 0; l < A.depth(); ++l) {
                EMat Dl_ref = as_eigen(D0(l)) * tri<Eigen::Upper>(as_eigen(A(l)));
                EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Dl_ref, ε));
            }
        }
}

TYPED_TEST_P(GemmTest, trmmULLinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::triu;
    using batmat::linalg::trmm;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes) {
        const auto A0 = this->get_A(m, m);
        auto A        = A0;
        trmm(triu(A), tril(A.transposed()), tril(A));
        for (index_t l = 0; l < A.depth(); ++l) {
            auto Al     = tri<Eigen::Upper>(as_eigen(A0(l)));
            auto Dl_ref = tri<Eigen::Lower>(Al * Al.transpose());
            EXPECT_THAT(tri<Eigen::Lower>(as_eigen(A(l))), EigenAlmostEqual(Dl_ref, ε));
            EXPECT_THAT(tri<Eigen::StrictlyUpper>(as_eigen(A(l))),
                        EigenAlmostEqual(tri<Eigen::StrictlyUpper>(as_eigen(A0(l))), ε));
        }
    }
}

#if 0 // TODO
TYPED_TEST_P(GemmTest, trmmLLL) {
    using batmat::linalg::tril;
    using batmat::linalg::trmm;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes) {
        const auto A = this->get_A(m, m);
        auto D       = this->get_A(m, m);
        trmm(tril(A), tril(A), tril(D));
        for (index_t l = 0; l < A.depth(); ++l) {
            auto Al     = as_eigen(A(l)).template triangularView<Eigen::Lower>().toDenseMatrix();
            auto Dl_ref = Al * Al;
            EXPECT_THAT(as_eigen(D(l)), EigenAlmostEqual(Dl_ref, ε));
        }
    }
}

TYPED_TEST_P(GemmTest, trmmLLLinplace) {
    using batmat::linalg::tril;
    using batmat::linalg::trmm;
    const auto ε = 1000 * std::numeric_limits<typename TestFixture::value_type>::epsilon();
    for (auto m : batmat::tests::sizes) {
        const auto A0 = this->get_A(m, m);
        auto A        = A0;
        trmm(tril(A), tril(A), tril(A));
        for (index_t l = 0; l < A.depth(); ++l) {
            auto Al     = as_eigen(A0(l)).template triangularView<Eigen::Lower>().toDenseMatrix();
            auto Dl_ref = Al * Al;
            EXPECT_THAT(as_eigen(A(l)), EigenAlmostEqual(Dl_ref, ε));
        }
    }
}
#endif

REGISTER_TYPED_TEST_SUITE_P(GemmTest, gemm, gemmNeg, gemmAdd, gemmSub, gemmSubShiftA,
                            gemmSubShiftCD, gemmSubShiftCDNeg, trmmLGinplace, trmmUGinplace,
                            trmmGLinplace, trmmGUinplace, trmmULLinplace);

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

INSTANTIATE_TYPED_TEST_SUITE_P(linalg, GemmTest, AllTestConfigs);
