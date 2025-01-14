#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/compact/micro-kernels/xtrtri.hpp>
#include <koqkatoo/linalg-compact/preferred-backend.hpp>
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <ocp/eigen-matchers.hpp>
#include <random>

namespace stdx = std::experimental;
using namespace koqkatoo;

TEST(xtrtri, xtrtritrmm) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};
    using EMat = Eigen::MatrixX<real_t>;
    EMat LB(14, 4);
    auto L = LB.topRows(4), B = LB.bottomRows(10);
    L << 1.10, 0, 0, 0,      //
        .21, 2.20, 0, 0,     //
        .31, .32, 3.30, 0,   //
        .41, .42, .43, 4.40; //
    std::ranges::generate(B.reshaped(), [&] { return nrml(rng); });

    EMat Linv  = L.triangularView<Eigen::Lower>().solve(EMat::Identity(4, 4));
    EMat BLinv = -B * Linv;
    linalg::compact::micro_kernels::trtri::xtrtri_trmm_microkernel<
        stdx::simd_abi::scalar, 4>(
        {LB.data(), static_cast<index_t>(L.outerStride())}, LB.rows());
    EXPECT_THAT(Linv, EigenAlmostEqual(L, 1e-8));
    EXPECT_THAT(BLinv, EigenAlmostEqual(B, 1e-8));
}

TEST(xtrtri, xtrmm) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};
    using EMat = Eigen::MatrixX<real_t>;
    EMat L(10, 4);
    EMat B(10, 4);
    std::ranges::generate(L.reshaped(), [&] { return nrml(rng); });
    std::ranges::generate(B.reshaped(), [&] { return nrml(rng); });
    L.triangularView<Eigen::StrictlyUpper>().setZero();

    EMat B1 = B;
    B1.topRows(4).setZero();
    EMat LB = B1 + L * B.topRows(4);
    linalg::compact::micro_kernels::trtri::xtrmm_microkernel<
        stdx::simd_abi::scalar, 4, 4>(
        {L.data(), static_cast<index_t>(L.outerStride())},
        {B.data(), static_cast<index_t>(B.outerStride())}, B.rows());
    EXPECT_THAT(LB, EigenAlmostEqual(B, 1e-8));
}

TEST(xtrtri, xtrtri) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};
    using EMat = Eigen::MatrixX<real_t>;
    index_t n  = 13;
    EMat L(n, n);
    std::ranges::generate(L.reshaped(), [&] { return nrml(rng); });
    L.triangularView<Eigen::StrictlyUpper>().setZero();
    EMat Linv = L.triangularView<Eigen::Lower>().solve(EMat::Identity(n, n));
    linalg::compact::CompactBLAS<stdx::simd_abi::scalar>::xtrtri_ref({{
        .data         = L.data(),
        .rows         = static_cast<index_t>(L.rows()),
        .cols         = static_cast<index_t>(L.cols()),
        .outer_stride = static_cast<index_t>(L.outerStride()),
    }});
    EXPECT_THAT(Linv, EigenAlmostEqual(L, 1e-8));
}

TEST(xtrtri, xtrtriRect) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};
    using EMat = Eigen::MatrixX<real_t>;
    index_t n = 13, m = 7;
    EMat L(n, n);
    std::ranges::generate(L.reshaped(), [&] { return nrml(rng); });
    L.triangularView<Eigen::StrictlyUpper>().setZero();
    EMat Linv = L.triangularView<Eigen::Lower>().solve(EMat::Identity(n, m));
    linalg::compact::CompactBLAS<stdx::simd_abi::scalar>::xtrtri_ref({{
        .data         = L.data(),
        .rows         = static_cast<index_t>(L.rows()),
        .cols         = m,
        .outer_stride = static_cast<index_t>(L.outerStride()),
    }});
    linalg::compact::CompactBLAS<stdx::simd_abi::scalar>::xtrsm_LLNN_ref(
        {{
            .data         = &L(m, m),
            .rows         = n - m,
            .cols         = n - m,
            .outer_stride = static_cast<index_t>(L.outerStride()),
        }},
        {{
            .data         = &L(m, 0),
            .rows         = n - m,
            .cols         = m,
            .outer_stride = static_cast<index_t>(L.outerStride()),
        }});
    EXPECT_THAT(Linv, EigenAlmostEqual(L.leftCols(m), 1e-8));
}

TEST(xtrtri, xtrtriCopyRect) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};
    using EMat = Eigen::MatrixX<real_t>;
    index_t n = 13, m = 7;
    EMat L(n, n);
    EMat Linv(n, m);
    std::ranges::generate(L.reshaped(), [&] { return nrml(rng); });
    L.triangularView<Eigen::StrictlyUpper>().setZero();
    EMat Linv_ref =
        L.triangularView<Eigen::Lower>().solve(EMat::Identity(n, m));
    linalg::compact::CompactBLAS<stdx::simd_abi::scalar>::xtrtri_copy_ref(
        {{
            .data         = L.data(),
            .rows         = static_cast<index_t>(L.rows()),
            .cols         = m,
            .outer_stride = static_cast<index_t>(L.outerStride()),
        }},
        {{
            .data         = Linv.data(),
            .rows         = static_cast<index_t>(Linv.rows()),
            .cols         = static_cast<index_t>(Linv.cols()),
            .outer_stride = static_cast<index_t>(Linv.outerStride()),
        }});
    linalg::compact::CompactBLAS<stdx::simd_abi::scalar>::xtrsm_LLNN_ref(
        {{
            .data         = &L(m, m),
            .rows         = n - m,
            .cols         = n - m,
            .outer_stride = static_cast<index_t>(L.outerStride()),
        }},
        {{
            .data         = &Linv(m, 0),
            .rows         = n - m,
            .cols         = m,
            .outer_stride = static_cast<index_t>(Linv.outerStride()),
        }});
    EXPECT_THAT(Linv_ref, EigenAlmostEqual(Linv.leftCols(m), 1e-8));
}

TEST(xtrtri, xtrtriRectBLAS) {
    std::mt19937 rng{12345};
    std::normal_distribution<real_t> nrml{0, 1};
    using EMat = Eigen::MatrixX<real_t>;
    index_t n = 13, m = 7;
    EMat L(n, n);
    std::ranges::generate(L.reshaped(), [&] { return nrml(rng); });
    L.triangularView<Eigen::StrictlyUpper>().setZero();
    EMat Linv = L.triangularView<Eigen::Lower>().solve(EMat::Identity(n, m));
    using compact = linalg::compact::CompactBLAS<stdx::simd_abi::scalar>;
    compact::xtrtri(compact::mut_single_batch_view{{
                        .data         = L.data(),
                        .rows         = static_cast<index_t>(L.rows()),
                        .cols         = m,
                        .outer_stride = static_cast<index_t>(L.outerStride()),
                    }},
                    linalg::compact::PreferredBackend::BLASScalar);
    compact::xtrsm_LLNN_ref(
        {{
            .data         = &L(m, m),
            .rows         = n - m,
            .cols         = n - m,
            .outer_stride = static_cast<index_t>(L.outerStride()),
        }},
        {{
            .data         = &L(m, 0),
            .rows         = n - m,
            .cols         = m,
            .outer_stride = static_cast<index_t>(L.outerStride()),
        }});
    EXPECT_THAT(Linv, EigenAlmostEqual(L.leftCols(m), 1e-8));
}
