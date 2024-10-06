#include <gtest/gtest.h>

#if KOQKATOO_WITH_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <limits>
#include <random>

#include <koqkatoo/cholundate/householder-downdate.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
#include <guanaqo/eigen/view.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace koqkatoo {
namespace {

struct ProblemMatrices {
    Eigen::MatrixXd K̃, K, L, A;
    Eigen::VectorXd S;
};

ProblemMatrices generate_problem(index_t m, index_t n) {
#if KOQKATOO_WITH_OPENMP
    int old_num_threads = omp_get_max_threads();
    omp_set_num_threads(std::thread::hardware_concurrency());
#endif

    std::mt19937 rng{12345};
    std::uniform_real_distribution<> dist(0.0, 1.0);
    std::bernoulli_distribution brnl(0.5);
    ProblemMatrices mat;
    mat.K̃.resize(n, n), mat.K.resize(n, n), mat.L.resize(n, n);
    mat.A.resize(n, m), mat.S.resize(m);
    std::ranges::generate(mat.K.reshaped(), [&] { return dist(rng); });
    std::ranges::generate(mat.A.reshaped(), [&] { return dist(rng); });
    const auto ldK = static_cast<index_t>(mat.K.outerStride());
    Eigen::MatrixXd Ad(n, m), Au(n, m);
    Eigen::Index ju = 0, jd = 0;
    for (index_t i = 0; i < m; ++i) {
        if (brnl(rng)) {
            Au.col(ju++) = mat.A.col(i);
            mat.S(i)     = +real_t(0);
        } else {
            Ad.col(jd++) = mat.A.col(i);
            mat.S(i)     = -real_t(0);
        }
    }
    Au = Au.leftCols(ju).eval();
    Ad = Ad.leftCols(jd).eval();
    // K̃ ← KᵀK
    linalg::xsyrk<real_t, index_t>(CblasColMajor, CblasLower, CblasTrans, n, n,
                                   1, mat.K.data(), ldK, 0, mat.K̃.data(), ldK);
    // K ← K̃
    mat.K = mat.K̃;
    // K += A₋A₋ᵀ
    linalg::xsyrk<real_t, index_t>(
        CblasColMajor, CblasLower, CblasNoTrans, n,
        static_cast<index_t>(Ad.cols()), 1, Ad.data(),
        static_cast<index_t>(Ad.outerStride()), 1, mat.K.data(),
        static_cast<index_t>(mat.K.outerStride()));
    // K̃ += A₊A₊ᵀ
    linalg::xsyrk<real_t, index_t>(
        CblasColMajor, CblasLower, CblasNoTrans, n,
        static_cast<index_t>(Au.cols()), 1, Au.data(),
        static_cast<index_t>(Au.outerStride()), 1, mat.K̃.data(),
        static_cast<index_t>(mat.K̃.outerStride()));
    // L = chol(K)
    mat.L        = mat.K;
    index_t info = 0;
    linalg::xpotrf<real_t, index_t>(
        "L", n, mat.L.data(), static_cast<index_t>(mat.L.outerStride()), &info);
    mat.L.triangularView<Eigen::StrictlyUpper>().setZero();
    mat.K̃.triangularView<Eigen::StrictlyUpper>() =
        mat.K̃.triangularView<Eigen::StrictlyLower>().transpose();
    mat.K.triangularView<Eigen::StrictlyUpper>() =
        mat.K.triangularView<Eigen::StrictlyLower>().transpose();

#if KOQKATOO_WITH_OPENMP
    omp_set_num_threads(old_num_threads);
#endif

    return mat;
}

real_t calculate_error(const ProblemMatrices &matrices,
                       const Eigen::Ref<const Eigen::MatrixX<real_t>> &L̃) {
    Eigen::MatrixXd E = matrices.K̃;
    const auto n      = static_cast<index_t>(L̃.rows()),
               ldL̃    = static_cast<index_t>(L̃.outerStride()),
               ldE    = static_cast<index_t>(E.outerStride());
#if KOQKATOO_WITH_OPENMP
    int old_num_threads = omp_get_max_threads();
    omp_set_num_threads(std::thread::hardware_concurrency());
#endif
    linalg::xsyrk<real_t, index_t>(CblasColMajor, CblasLower, CblasNoTrans, n,
                                   n, -1, L̃.data(), ldL̃, 1, E.data(), ldE);
#if KOQKATOO_WITH_OPENMP
    omp_set_num_threads(old_num_threads);
#endif
    E.triangularView<Eigen::StrictlyUpper>().setZero();
    return E.lpNorm<Eigen::Infinity>();
}

} // namespace
} // namespace koqkatoo

using koqkatoo::index_t;
using koqkatoo::real_t;

const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), 0.5);

struct SHHUpDown : testing::TestWithParam<index_t> {};

TEST_P(SHHUpDown, VariousSizes) {
    index_t n = GetParam();
    for (index_t m : {1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 17, 31, 32}) {
        auto matrices     = koqkatoo::generate_problem(m, n);
        Eigen::MatrixXd L̃ = matrices.L;
        Eigen::MatrixXd Ã = matrices.A;
        Eigen::VectorXd S̃ = matrices.S;
        koqkatoo::cholundate::householder::updowndate_blocked<{
            .block_size_r = 8, .block_size_s = 24, .enable_packing = false}>(
            as_view(L̃, guanaqo::with_index_type<index_t>),
            as_view(Ã, guanaqo::with_index_type<index_t>),
            as_view(S̃, guanaqo::with_index_type<index_t>));
        real_t residual = koqkatoo::calculate_error(matrices, L̃);
        EXPECT_LE(residual, ε) << "m=" << m;
    }
}

#if KOQKATOO_WITH_LIBFORK && 0 // TODO
TEST_P(SHHUpDown, VariousSizesLibFork) {
    index_t n = GetParam();
    for (index_t m : {1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 17, 31, 32}) {
        auto matrices     = koqkatoo::generate_problem(m, n);
        Eigen::MatrixXd L̃ = matrices.L;
        Eigen::MatrixXd Ã = matrices.A;
        koqkatoo::cholundate::householder::parallel::downdate_blocked<{8, 24}>(
            as_view(L̃, guanaqo::with_index_type<index_t>),
            as_view(Ã, guanaqo::with_index_type<index_t>));
        Eigen::LLT<Eigen::MatrixXd> llt(matrices.K̃);
        real_t residual = koqkatoo::calculate_error(matrices, L̃);
        EXPECT_LE(residual, ε) << "m=" << m;
    }
}
#endif

INSTANTIATE_TEST_SUITE_P(Cholundate, SHHUpDown,
                         testing::Range<index_t>(1, 256));
