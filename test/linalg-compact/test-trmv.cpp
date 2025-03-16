#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/xtrsm.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <random>

#include <ocp/eigen-matchers.hpp>

#include <guanaqo/eigen/view.hpp>

using namespace koqkatoo::linalg::compact;
using koqkatoo::index_t;
using koqkatoo::real_t;

const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));

TEST(Trmv, trmv) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    using compact_blas = CompactBLAS<stdx::simd_abi::scalar>;
    using matrix       = typename compact_blas::matrix;
    const index_t n    = 13;
    matrix L{{.depth = 1, .rows = n, .cols = n}},
        x{{.depth = 1, .rows = n, .cols = 1}};
    std::ranges::generate(L, [&] { return nrml(rng); });
    std::ranges::generate(x, [&] { return nrml(rng); });
    matrix y = x;
    compact_blas::xtrmv_ref(L.batch(0), y.batch(0));
    auto Le = guanaqo::as_eigen(L(0)).template triangularView<Eigen::Lower>();
    auto xe = guanaqo::as_eigen(x(0));
    EXPECT_THAT(guanaqo::as_eigen(y(0)), EigenAlmostEqual(Le * xe, ε));
}

TEST(Trmv, trmvT) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    using compact_blas = CompactBLAS<stdx::simd_abi::scalar>;
    using matrix       = typename compact_blas::matrix;
    const index_t n    = 13;
    matrix L{{.depth = 1, .rows = n, .cols = n}},
        x{{.depth = 1, .rows = n, .cols = 1}};
    std::ranges::generate(L, [&] { return nrml(rng); });
    std::ranges::generate(x, [&] { return nrml(rng); });
    matrix y = x;
    compact_blas::xtrmv_T_ref(L.batch(0), y.batch(0));
    auto Le = guanaqo::as_eigen(L(0)).template triangularView<Eigen::Lower>();
    auto xe = guanaqo::as_eigen(x(0));
    EXPECT_THAT(guanaqo::as_eigen(y(0)),
                EigenAlmostEqual(Le.transpose() * xe, ε));
}

TEST(Trsm, trsmLLNN) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};

    using abi            = stdx::simd_abi::scalar;
    using compact_blas   = CompactBLAS<abi>;
    using matrix         = typename compact_blas::matrix;
    constexpr index_t NR = 5, NC = 3;
    matrix L{{.depth = 1, .rows = 15, .cols = NR}},
        b{{.depth = 1, .rows = 15, .cols = NC}};
    std::ranges::generate(L, [&] { return nrml(rng); });
    std::ranges::generate(b, [&] { return nrml(rng); });
    matrix x = b;
    micro_kernels::trsm::xtrsm_llnn_microkernel<abi, NR, NC>(
        L.batch(0), x.batch(0), L.rows());
    auto Le  = guanaqo::as_eigen(L(0));
    auto L11 = Le.topRows(NR).template triangularView<Eigen::Lower>();
    auto L21 = Le.bottomRows(10);
    auto be  = guanaqo::as_eigen(b(0));
    Eigen::MatrixX<real_t> xe = be;
    auto x1 = xe.topRows(NR), x2 = xe.bottomRows(15 - NR);
    L11.solveInPlace(x1);
    x2 -= L21 * x1;
    EXPECT_THAT(guanaqo::as_eigen(x(0)), EigenAlmostEqual(xe, ε));
}
