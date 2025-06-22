#include <batmat/linalg/micro-kernels/gemm.hpp>
#include <gtest/gtest.h>
#include <eigen-matchers.hpp>

#include "config.hpp"

#include <Eigen/Core>
#include <optional>
#include <random>

namespace stdx = std::experimental;
using batmat::index_t;
using batmat::real_t;
using batmat::linalg::view;

TEST(SyrkTest, TrTrSyrk) {
    for (index_t n : batmat::tests::sizes) {
        std::mt19937 rng{12345};
        std::normal_distribution<real_t> nrml{0, 1};
        using EMat = Eigen::MatrixX<real_t>;
        EMat A(n, n), C(n, n);
        std::ranges::generate(A.reshaped(), [&] { return nrml(rng); });
        // A.triangularView<Eigen::StrictlyLower>().setZero(); // (implicit)
        std::ranges::generate(C.reshaped(), [&] { return nrml(rng); });
        EMat C_ref = C;

        using abi = stdx::simd_abi::scalar;
        view<const real_t, abi> vA{{
            .data         = std::as_const(A).data(),
            .rows         = static_cast<index_t>(A.rows()),
            .cols         = static_cast<index_t>(A.cols()),
            .outer_stride = static_cast<index_t>(A.outerStride()),
        }};
        view<real_t, abi> vD{{
            .data         = C.data(),
            .rows         = static_cast<index_t>(C.rows()),
            .cols         = static_cast<index_t>(C.cols()),
            .outer_stride = static_cast<index_t>(C.outerStride()),
        }};
        using enum batmat::linalg::micro_kernels::gemm::MatrixStructure;
        constexpr batmat::linalg::micro_kernels::gemm::KernelConfig Conf{
            .struc_A = UpperTriangular, .struc_B = LowerTriangular, .struc_C = LowerTriangular};
        batmat::linalg::micro_kernels::gemm::gemm_copy_register<real_t, abi, Conf>(
            vA, vA.transposed(), std::optional<view<const real_t, abi>>(), vD);
        auto AU                              = A.triangularView<Eigen::Upper>();
        C_ref.triangularView<Eigen::Lower>() = AU.toDenseMatrix() * AU.transpose();
        EXPECT_THAT(C, EigenAlmostEqual(C_ref, 1e-8));
    }
}

TEST(SyrkTest, TrTrSyrkInPlace) {
    for (index_t n : batmat::tests::sizes) {
        std::mt19937 rng{12345};
        std::normal_distribution<real_t> nrml{0, 1};
        using EMat = Eigen::MatrixX<real_t>;
        EMat A(n, n), C_ref(n, n);
        std::ranges::generate(A.reshaped(), [&] { return nrml(rng); });
        C_ref.setZero();
        EMat AU = A.triangularView<Eigen::Upper>();
        C_ref.selfadjointView<Eigen::Lower>().rankUpdate(AU);

        using abi = stdx::simd_abi::scalar;
        view<real_t, abi> vA{{
            .data         = A.data(),
            .rows         = static_cast<index_t>(A.rows()),
            .cols         = static_cast<index_t>(A.cols()),
            .outer_stride = static_cast<index_t>(A.outerStride()),
        }};
        using enum batmat::linalg::micro_kernels::gemm::MatrixStructure;
        constexpr batmat::linalg::micro_kernels::gemm::KernelConfig Conf{
            .struc_A = UpperTriangular, .struc_B = LowerTriangular, .struc_C = LowerTriangular};
        std::optional<view<const real_t, abi>> null;
        batmat::linalg::micro_kernels::gemm::gemm_copy_register<real_t, abi, Conf>(
            vA.as_const(), vA.as_const().transposed(), null, vA);
        A.triangularView<Eigen::StrictlyUpper>().setZero();
        EXPECT_THAT(A, EigenAlmostEqual(C_ref, 1e-8));
    }
}
