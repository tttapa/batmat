#include <koqkatoo/linalg-compact/compact.hpp>
#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/linalg-compact/preferred-backend.hpp>
#include <koqkatoo/loop.hpp>
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <ocp/eigen-matchers.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <koqkatoo/linalg-compact/compact/micro-kernels/xshh.hpp> // TODO

namespace stdx = std::experimental;
using namespace koqkatoo;

const auto ε = std::pow(std::numeric_limits<real_t>::epsilon(), real_t(0.6));

TEST(OCP, update) {
    using std::exp2;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};
    std::bernoulli_distribution brnl{0.5};
    constexpr auto be           = linalg::compact::PreferredBackend::Reference;
    using abi                   = stdx::simd_abi::deduce_t<real_t, 8>;
    using scalar_abi            = stdx::simd_abi::scalar;
    using compact_blas          = linalg::compact::CompactBLAS<abi>;
    using scalar_blas           = linalg::compact::CompactBLAS<scalar_abi>;
    using EMat [[maybe_unused]] = Eigen::MatrixX<real_t>;
    using EVec [[maybe_unused]] = Eigen::VectorX<real_t>;
    using Mat  = linalg::compact::BatchedMatrix<real_t, index_t,
                                                compact_blas::simd_stride_t>;
    using BMat = linalg::compact::BatchedMatrix<bool, index_t,
                                                compact_blas::simd_stride_t>;
    using SMat = linalg::compact::BatchedMatrix<real_t, index_t,
                                                scalar_blas::simd_stride_t>;
    using SBMat [[maybe_unused]] =
        linalg::compact::BatchedMatrix<bool, index_t,
                                       scalar_blas::simd_stride_t>;
    constexpr index_t N = 5, nx = 7, nu = 1, ny = 11;
    Mat AB{{.depth = N, .rows = nx, .cols = nx + nu}};
    Mat CD{{.depth = N + 1, .rows = ny, .cols = nx + nu}};
    Mat H{{.depth = N + 1, .rows = nx + nu, .cols = nx + nu}};
    std::ranges::generate(AB, [&] { return uni(rng); });
    std::ranges::generate(CD, [&] { return uni(rng); });
    std::ranges::generate(H, [&] { return uni(rng); });
    H.view.add_to_diagonal(10);
    H(N).bottom_rows(nu).set_constant(0);
    H(N).bottom_right(nu, nu).set_diagonal(1);
    CD(N).right_cols(nu).set_constant(0);

    Mat LH̃V{{.depth = N + 1, .rows = nx + nu + nx, .cols = nx + nu}};
    auto LH̃ = LH̃V.view.top_rows(nx + nu);
    auto V  = LH̃V.view.bottom_rows(nx).first_layers(N);

    // Initialize and factorize H
    LH̃ = H; // TODO: initial Σ?
    V  = AB;
    compact_blas::xpotrf(LH̃V, be);
    // Prepare Ψ
    Mat Wᵀ{{.depth = N + 1, .rows = nx + nu, .cols = nx}};
    Wᵀ.view = LH̃.left_cols(nx); // TODO: remove .view
    compact_blas::xtrsm_LLNN(LH̃.bottom_right(nu, nu), Wᵀ.view.bottom_rows(nu),
                             be);
    compact_blas::xtrtri(Wᵀ, be);
    Mat WWᵀ{{.depth = N + 1, .rows = nx, .cols = nx}};
    Mat VVᵀ{{.depth = N, .rows = nx, .cols = nx}};
    Mat nVWᵀ{{.depth = N, .rows = nx, .cols = nx}};
    compact_blas::xsyrk(V, VVᵀ, be);
    compact_blas::xtrmm_RLNN_neg(V, Wᵀ.view.first_layers(N), nVWᵀ, be);
    // TODO: implement xsyrk_T for triangular input
    for (index_t c = 1; c < Wᵀ.cols(); ++c) // Make Wᵀ lower trapezoidal
        Wᵀ.view.block(0, c, c, 1).set_constant(0);
    compact_blas::xsyrk_T(Wᵀ, WWᵀ, be);

    // Build Ψ and chol(Ψ)
    SMat LΨD{{.depth = N + 1, .rows = nx, .cols = nx}};
    SMat LΨS{{.depth = N, .rows = nx, .cols = nx}};
    LΨD(0) = WWᵀ(0);
    scalar_blas::xpotrf(LΨD.batch(0), be);
    for (index_t k = 0; k < N; ++k) {
        LΨS(k) = nVWᵀ(k);
        scalar_blas::xtrsm_RLTN(LΨD.batch(k), LΨS.batch(k), be);
        LΨD(k + 1) = VVᵀ(k);
        LΨD(k + 1) += WWᵀ(k + 1);
        scalar_blas::xsyrk_sub(LΨS.batch(k), LΨD.batch(k + 1), be);
        scalar_blas::xpotrf(LΨD.batch(k + 1), be);
    }

    // Compare against reference implementation
    SMat H_full{{
        .depth = 1,
        .rows  = N * (nx + nu) + nx,
        .cols  = N * (nx + nu) + nx,
    }};
    SMat M_full{{
        .depth = 1,
        .rows  = (N + 1) * nx,
        .cols  = N * (nx + nu) + nx,
    }};
    SMat G_full{{
        .depth = 1,
        .rows  = (N + 1) * ny,
        .cols  = N * (nx + nu) + nx,
    }};
    H_full.set_constant(0);
    for (index_t k = 0; k < N; ++k)
        H_full(0).block(k * (nx + nu), k * (nx + nu), nx + nu, nx + nu) = H(k);
    H_full(0).bottom_right(nx, nx) = H(N).top_left(nx, nx);
    SMat LH_full                   = H_full;
    scalar_blas::xpotrf(LH_full, be);
    M_full.set_constant(0);
    M_full(0).top_left(nx, nx).set_diagonal(1);
    for (index_t k = 0; k < N; ++k) {
        index_t r = (k + 1) * nx, c = k * (nx + nu);
        M_full(0).block(r, c, nx, nx + nu) = AB(k);
        M_full.view.block(r, c, nx, nx + nu).negate();
        M_full(0).block(r, c + nx + nu, nx, nx).set_diagonal(1);
    }
    G_full.set_constant(0);
    for (index_t k = 0; k < N; ++k)
        G_full(0).block(k * ny, k * (nx + nu), ny, nx + nu) = CD(k);
    G_full(0).bottom_right(ny, nx) = CD(N).left_cols(nx);
    SMat MLH_full                  = M_full;
    scalar_blas::xtrsm_RLTN(LH_full, MLH_full, be);
    SMat Ψ_full{{.depth = 1, .rows = (N + 1) * nx, .cols = (N + 1) * nx}};
    scalar_blas::xsyrk(MLH_full, Ψ_full, be);
    SMat LΨ_full = Ψ_full;
    scalar_blas::xpotrf(LΨ_full, be);
    // Reconstruct solution
    SMat LΨ_rec{{.depth = 1, .rows = (N + 1) * nx, .cols = (N + 1) * nx}};
    LΨ_rec(0).top_left(nx, nx) = LΨD(0);
    for (index_t k = 0; k < N; ++k) {
        LΨ_rec(0).block(k * nx + nx, k * nx, nx, nx)      = LΨS(k);
        LΨ_rec(0).block(k * nx + nx, k * nx + nx, nx, nx) = LΨD(k + 1);
    }
    guanaqo::print_python(std::cout << "LΨ ref:\n", LΨ_full(0));
    guanaqo::print_python(std::cout << "LΨ rec:\n", LΨ_rec(0));
    SMat LΨ_err{{.depth = 1, .rows = (N + 1) * nx, .cols = (N + 1) * nx}};
    scalar_blas::xsub_copy(LΨ_err, LΨ_rec, LΨ_full);
    EXPECT_LE(scalar_blas::xnrminf(LΨ_err), ε);

    // Prepare factorization update
    BMat J{{.depth = N + 1, .rows = ny, .cols = 1}};
    Mat Σ{{.depth = N + 1, .rows = ny, .cols = 1}};
    Mat S{{.depth = N + 1, .rows = ny, .cols = 1}};
    std::ranges::generate(J, [&] { return brnl(rng); });
    std::ranges::generate(Σ, [&] { return exp2(uni(rng)); });
    S.set_constant(+real_t(0)); // only entering constraints
    // Copy entering constraints to Ge (and transpose them)
    Mat Geᵀ{{.depth = N + 1, .rows = nx + nu, .cols = ny}};
    Mat Σe{{.depth = N + 1, .rows = ny, .cols = 1}};
    Mat Se{{.depth = N + 1, .rows = ny, .cols = 1}};
    std::vector<index_t> rjs;
    Σe.set_constant(1);
    Se.set_constant(+real_t(0));
    index_t rj_max = 0;
    rjs.reserve(N + 1);
    for (index_t k = 0; k <= N; ++k) {
        index_t rj = 0;
        for (index_t r = 0; r < ny; ++r) {
            if (J(k, r, 0)) {
                Σe(k, rj, 0) = Σ(k, r, 0);
                Se(k, rj, 0) = S(k, r, 0);
                for (index_t c = 0; c < nx + nu; ++c)
                    Geᵀ(k, c, rj) = CD(k, r, c);
                ++rj;
            }
        }
        rjs.push_back(rj);
        rj_max = std::max(rj_max, rj);
    }
    auto Gejᵀ = Geᵀ.view.left_cols(rj_max);
    auto Σj   = Σe.view.top_rows(rj_max);
    auto Sj   = Se.view.top_rows(rj_max);
    Mat Z{{.depth = N + 1, .rows = nx + nu, .cols = ny}};
    auto Zj = Z.view.left_cols(rj_max);
    Zj      = Gejᵀ;
    compact_blas::xtrsm_LLNN(LH̃, Zj, be);
    Mat ΣeZ{{.depth = N + 1, .rows = ny, .cols = ny}};
    auto ΣjZ = ΣeZ.view.top_left(rj_max, rj_max);
    compact_blas::xsyrk_T(Zj, ΣjZ, be);
    for (index_t k = 0; k <= N; ++k)
        for (index_t r = 0; r < rj_max; ++r)
            ΣjZ(k, r, r) += 1 / Σj(k, r, 0); // TODO: negate leaving
    compact_blas::xpntrf(ΣjZ, Sj);
    Mat Ye{{.depth = N + 1, .rows = nx + nu, .cols = ny}};
    auto Yj = Ye.view.left_cols(rj_max);
    Yj      = Zj;
    compact_blas::xtrsm_RLTN(ΣjZ, Yj, be);
    Mat W̃e{{.depth = N + 1, .rows = nx, .cols = ny}};
    auto W̃j = W̃e.view.left_cols(rj_max);
    compact_blas::xgemm_TN(Wᵀ, Yj, W̃j, be);
    Mat Ṽe{{.depth = N, .rows = nx, .cols = ny}};
    auto Ṽj = Ṽe.view.left_cols(rj_max);
    compact_blas::xgemm_neg(V, Yj.first_layers(N), Ṽj, be);

    // Check low-rank downdate of Ψ
    Mat H̃e{{.depth = N + 1, .rows = nx + nu, .cols = nx + nu}};
    compact_blas::xsyrk_T_schur_copy(CD, Σ, J, H, H̃e, be);

    SMat H̃e_full{{
        .depth = 1,
        .rows  = N * (nx + nu) + nx,
        .cols  = N * (nx + nu) + nx,
    }};
    H̃e_full.set_constant(0);
    for (index_t k = 0; k < N; ++k)
        H̃e_full(0).block(k * (nx + nu), k * (nx + nu), nx + nu, nx + nu) =
            H̃e(k);
    H̃e_full(0).bottom_right(nx, nx) = H̃e(N).top_left(nx, nx);
    SMat LH̃e_full                   = H̃e_full;
    scalar_blas::xpotrf(LH̃e_full, be);

    SMat MLH̃e_full = M_full;
    scalar_blas::xtrsm_RLTN(LH̃e_full, MLH̃e_full, be);
    SMat Ψ̃e_full{{.depth = 1, .rows = (N + 1) * nx, .cols = (N + 1) * nx}};
    scalar_blas::xsyrk(MLH̃e_full, Ψ̃e_full, be);
    SMat LΨ̃e_full = Ψ̃e_full;
    scalar_blas::xpotrf(LΨ̃e_full, be);

    // TODO
    SMat LΨ̃e_rec = LΨ_rec;
    SMat W̃_rec{{.depth = 1, .rows = (N + 1) * nx, .cols = (N + 1) * ny}};
    W̃_rec(0).top_left(nx, ny) = W̃e(0);
    for (index_t k = 0; k < N; ++k) {
        W̃_rec(0).block(k * nx + nx, k * ny, nx, ny)      = Ṽe(k);
        W̃_rec(0).block(k * nx + nx, k * ny + ny, nx, ny) = W̃e(k + 1);
    }
    guanaqo::print_python(std::cout << "W̃ rec:\n", W̃_rec(0));
    scalar_blas::xshh(LΨ̃e_rec, W̃_rec, be);
    guanaqo::print_python(std::cout << "HH:\n", W̃_rec(0));

    guanaqo::print_python(std::cout << "LΨ ref:\n", LΨ̃e_full(0));
    guanaqo::print_python(std::cout << "LΨ rec:\n", LΨ̃e_rec(0));
    SMat LLΨ̃e_full{{.depth = 1, .rows = (N + 1) * nx, .cols = (N + 1) * nx}};
    SMat LLΨ̃e_rec{{.depth = 1, .rows = (N + 1) * nx, .cols = (N + 1) * nx}};
    scalar_blas::xsyrk(LΨ̃e_full, LLΨ̃e_full, be);
    scalar_blas::xsyrk(LΨ̃e_rec, LLΨ̃e_rec, be);
    SMat LLΨ̃e_err{{.depth = 1, .rows = (N + 1) * nx, .cols = (N + 1) * nx}};
    scalar_blas::xsub_copy(LLΨ̃e_err, LLΨ̃e_rec, LLΨ̃e_full);
    EXPECT_LE(scalar_blas::xnrminf(LLΨ̃e_err), ε);

    // Block-wise downdate
    SMat LΨ̃D      = LΨD;
    SMat LΨ̃S      = LΨS;
    index_t colsA = 0;
    SMat A{{.depth = 2, .rows = nx, .cols = (N + 1) * ny}};
    for (index_t k = 0; k < N; ++k) {
        using namespace linalg::compact::micro_kernels;
        static constexpr index_t R = shh::SizeR, S = shh::SizeS;
        colsA += rjs[k];
        auto Ad = A.batch(k % 2).left_cols(colsA),
             As = A.batch((k + 1) % 2).left_cols(colsA);
        auto Ld = LΨ̃D.batch(k), Ls = LΨ̃S.batch(k);
        Ad(0).right_cols(rjs[k]) = W̃j(k).left_cols(rjs[k]);
        As(0).right_cols(rjs[k]) = Ṽj(k).left_cols(rjs[k]);

        guanaqo::print_python(std::cout << "A(" << k << "):\n", Ad(0));

        // Process all diagonal blocks (in multiples of R, except the last).
        using abi = scalar_abi;
        foreach_chunked(
            0, Ld.cols(), R,
            [&](index_t k) {
                // Part of A corresponding to this diagonal block
                auto Add = Ad.middle_rows(k, R);
                auto Ldd = Ld.block(k, k, R, R);
                // Process the diagonal block itself
                using W_t = shh::triangular_accessor<abi, real_t, R>;
                alignas(W_t::alignment()) real_t W[W_t::size()];
                shh::xshh_diag_microkernel<abi, R>(colsA, W, Ldd, Add);
                // Process all rows below the diagonal block (in multiples of S).
                foreach_chunked_merged(
                    k + R, Ld.rows(), S, [&](index_t i, auto rem_i) {
                        auto Ads = Ad.middle_rows(i, rem_i);
                        auto Lds = Ld.block(i, k, rem_i, R);
                        shh::microkernel_tail_lut<abi>[rem_i - 1](colsA, W, Lds,
                                                                  Ads, Add);
                    }); // TODO: decide on order
                // Process all rows below the diagonal block (in multiples of S).
                foreach_chunked_merged(
                    0, Ls.rows(), S, [&](index_t i, auto rem_i) {
                        auto Ass = As.middle_rows(i, rem_i);
                        auto Lss = Ls.block(i, k, rem_i, R);
                        shh::microkernel_tail_lut<abi>[rem_i - 1](colsA, W, Lss,
                                                                  Ass, Add);
                    }); // TODO: decide on order
            },
            [&](index_t k, index_t rem_k) {
                auto Add = Ad.middle_rows(k, rem_k);
                auto Ldd = Ld.block(k, k, rem_k, rem_k);
                // Process the diagonal block itself
                using W_t = shh::triangular_accessor<abi, real_t, R>;
                alignas(W_t::alignment()) real_t W[W_t::size()];
                shh::microkernel_diag_lut<abi>[rem_k - 1](colsA, W, Ldd, Add);
                // Process all rows below the diagonal block (in multiples of S).
                foreach_chunked_merged(
                    0, Ls.rows(), S, [&](index_t i, auto rem_i) {
                        auto Ass = As.middle_rows(i, rem_i);
                        auto Lss = Ls.block(i, k, rem_i, rem_k);
                        shh::microkernel_tail_lut_2<abi>[rem_k - 1][rem_i - 1](
                            colsA, W, Lss, Ass, Add);
                    }); // TODO: decide on order
            });
        guanaqo::print_python(std::cout << "HH(" << k << "):\n", Ad(0));
        Ad(0).set_constant(0);
    }
    {
        using namespace linalg::compact::micro_kernels;
        static constexpr index_t R = shh::SizeR, S = shh::SizeS;
        colsA += rjs[N];
        auto Ad                  = A.batch(N % 2).left_cols(colsA);
        auto Ld                  = LΨ̃D.batch(N);
        Ad(0).right_cols(rjs[N]) = W̃j(N).left_cols(rjs[N]);

        // Process all diagonal blocks (in multiples of R, except the last).
        using abi = scalar_abi;
        foreach_chunked(
            0, Ld.cols(), R,
            [&](index_t k) {
                // Part of A corresponding to this diagonal block
                auto Add = Ad.middle_rows(k, R);
                auto Ldd = Ld.block(k, k, R, R);
                // Process the diagonal block itself
                using W_t = shh::triangular_accessor<abi, real_t, R>;
                alignas(W_t::alignment()) real_t W[W_t::size()];
                shh::xshh_diag_microkernel<abi, R>(colsA, W, Ldd, Add);
                // Process all rows below the diagonal block (in multiples of S).
                foreach_chunked_merged(
                    k + R, Ld.rows(), S, [&](index_t i, auto rem_i) {
                        auto Ads = Ad.middle_rows(i, rem_i);
                        auto Lds = Ld.block(i, k, rem_i, R);
                        shh::microkernel_tail_lut<abi>[rem_i - 1](colsA, W, Lds,
                                                                  Ads, Add);
                    }); // TODO: decide on order
            },
            [&](index_t k, index_t rem_k) {
                auto Add = Ad.middle_rows(k, rem_k);
                auto Ldd = Ld.block(k, k, rem_k, rem_k);
                // Process the diagonal block itself
                shh::microkernel_full_lut<abi>[rem_k - 1](colsA, Ldd, Add);
            });
    }

    // Reconstruct solution
    SMat LΨ̃e_rec2{{.depth = 1, .rows = (N + 1) * nx, .cols = (N + 1) * nx}};
    LΨ̃e_rec2(0).top_left(nx, nx) = LΨ̃D(0);
    for (index_t k = 0; k < N; ++k) {
        LΨ̃e_rec2(0).block(k * nx + nx, k * nx, nx, nx)      = LΨ̃S(k);
        LΨ̃e_rec2(0).block(k * nx + nx, k * nx + nx, nx, nx) = LΨ̃D(k + 1);
    }
    guanaqo::print_python(std::cout << "LΨ ref:\n", LΨ̃e_full(0));
    guanaqo::print_python(std::cout << "LΨ rec:\n", LΨ̃e_rec2(0));
    SMat LΨ̃e_err2{{.depth = 1, .rows = (N + 1) * nx, .cols = (N + 1) * nx}};
    scalar_blas::xsub_copy(LΨ̃e_err2, LΨ̃e_rec2, LΨ̃e_full);
    EXPECT_LE(scalar_blas::xnrminf(LΨ̃e_err2), ε);
}
