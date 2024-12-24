#pragma once

#include <koqkatoo/loop.hpp>
#include <koqkatoo/ocp/solver/solve.hpp>
#include <koqkatoo/openmp.h>
#include <optional>

#if KOQKATOO_WITH_MKL
#include <koqkatoo/linalg/blas.hpp>
#endif

namespace koqkatoo::ocp {

template <simd_abi_tag Abi>
void Solver<Abi>::schur_complement_H(real_view Σ, bool_view J) {
    // Note: xsyrk_T_schur_copy has no MKL equivalent, so we always just use a
    //       (parallel) for loop.
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < CD().num_batches(); ++i)
        schur_complement_Hi(i, Σ, J);
}

template <simd_abi_tag Abi>
void Solver<Abi>::schur_complement_Hi(index_t i, real_view Σ, bool_view J) {
    compact_blas::xsyrk_T_schur_copy(CD().batch(i), Σ.batch(i), J.batch(i),
                                     H().batch(i), LH().batch(i));
    if (i < AB().num_batches())
        compact_blas::xcopy(AB().batch(i), V().batch(i));
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_H() {
    compact_blas::xpotrf(LHV(), settings.preferred_backend);
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_Hi(index_t i) {
    compact_blas::xpotrf(LHV().batch(i), settings.preferred_backend);
#if !defined(NDEBUG)
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    index_t j0                 = i * simd_stride;
    using std::isfinite;
    for (index_t j = j0; j < std::min(j0 + simd_stride, N + 1); ++j)
        if (!isfinite(LHV()(j, nx - 1, nx - 1)))
            throw std::runtime_error("Factorization of H(" + std::to_string(j) +
                                     ") failed");
#endif
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve_H(mut_real_view x) {
    if (settings.prefer_single_loop) {
        KOQKATOO_OMP(parallel for)
        for (index_t i = 0; i < LH().num_batches(); ++i) {
            compact_blas::xtrsm_LLNN(LH().batch(i), x.batch(i),
                                     settings.preferred_backend);
            compact_blas::xtrsm_LLTN(LH().batch(i), x.batch(i),
                                     settings.preferred_backend);
        }
    } else {
        compact_blas::xtrsm_LLNN(LH(), x, settings.preferred_backend);
        compact_blas::xtrsm_LLTN(LH(), x, settings.preferred_backend);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::prepare_Ψi(index_t i) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    auto LHi = LH().batch(i), Wi = Wᵀ().batch(i), Vi = V().batch(i);
    // Solve W = LH⁻¹ [I 0]ᵀ
    compact_blas::xcopy(LHi.top_left(nx + nu, nx), Wi);
    compact_blas::xtrtri(Wi, settings.preferred_backend);
    compact_blas::xtrsm_LLNN(LHi.bottom_right(nu, nu), Wi.bottom_rows(nu),
                             settings.preferred_backend);
    compact_blas::xsyrk_T(Wi, LΨd().batch(i), settings.preferred_backend);
    if (i < AB().num_batches()) {
        // Store V(i) = VVᵀ
        compact_blas::xsyrk(Vi, VV().batch(i), settings.preferred_backend);
        // Store LΨ(i+1,i) = -VW
        compact_blas::xgemm_neg(Vi, Wi, LΨs().batch(i),
                                settings.preferred_backend);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::prepare_Ψ() {
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < LH().num_batches(); ++i)
        prepare_Ψi(i);
}

template <simd_abi_tag Abi>
void Solver<Abi>::prepare_all(real_t S, real_view Σ, bool_view J) {
    using std::isfinite;
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < LH().num_batches(); ++i) {
        schur_complement_Hi(i, Σ, J);
        if (isfinite(S))
            LH().batch(i).add_to_diagonal(1 / S);
        cholesky_Hi(i);
        prepare_Ψi(i);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::prepare_all(real_t S, real_view Σ, bool_view J, Timings &t) {
    using std::isfinite;
    constexpr auto now = [] {
        return std::chrono::steady_clock::now().time_since_epoch();
    };
    int64_t timer_schur_complement{0}, timer_cholesky_H{0}, timer_prepare_Ψ{0};
    using timer_t = decltype(Timings::type::wall_time);
    KOQKATOO_OMP(parallel) {
        KOQKATOO_OMP(for reduction(+:timer_schur_complement,timer_cholesky_H,timer_prepare_Ψ))
        for (index_t i = 0; i < LH().num_batches(); ++i) {
            const auto t0 = now();
            schur_complement_Hi(i, Σ, J);
            if (isfinite(S))
                LH().batch(i).add_to_diagonal(1 / S);
            const auto t1 = now();
            timer_schur_complement += duration_cast<timer_t>(t1 - t0).count();
            cholesky_Hi(i);
            const auto t2 = now();
            timer_cholesky_H += duration_cast<timer_t>(t2 - t1).count();
            prepare_Ψi(i);
            const auto t3 = now();
            timer_prepare_Ψ += duration_cast<timer_t>(t3 - t2).count();
        }
        KOQKATOO_OMP(single) {
            const int n = KOQKATOO_OMP_IF_ELSE(omp_get_num_threads(), 1);
            timer_schur_complement /= n;
            timer_cholesky_H /= n;
            timer_prepare_Ψ /= n;
        }
    }
    t.schur_complement.wall_time += timer_t{timer_schur_complement};
    t.cholesky_H.wall_time += timer_t{timer_cholesky_H};
    t.prepare_Ψ.wall_time += timer_t{timer_prepare_Ψ};
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_Ψ() {
    const auto N = storage.dim.N_horiz;
    auto wLΨd = storage.work_LΨd(), wLΨs = storage.work_LΨs(),
         wVV = storage.work_VV();
    foreach_chunked_merged(0, N, simd_stride, [&](index_t i, auto ni) {
#if 1
        index_t b = i / simd_stride, nd = std::min(ni + 1, simd_stride);
        compact_blas::unpack_L(LΨd().batch(b), wLΨd.first_layers(nd));
        if (i > 0)
            wLΨd(0) += wVV(simd_stride - 1);
        compact_blas::unpack(LΨs().batch(b), wLΨs.first_layers(ni));
        compact_blas::unpack_L(VV().batch(b), wVV.first_layers(ni));
#elif 1
        auto do_copy_L = [](auto &&A, auto &&B) {
            assert(A.rows() == B.rows());
            assert(A.cols() == B.cols());
            assert(A.depth() == simd_stride);
            assert(B.depth() == simd_stride);
            for (index_t c = 0; c < A.cols(); ++c)
                for (index_t r = c; r < A.rows(); ++r)
                    for (index_t d = 0; d < simd_stride; ++d)
                        B(d, r, c) = A(d, r, c);
        };
        auto do_copy = [](auto &&A, auto &&B) {
            assert(A.rows() == B.rows());
            assert(A.cols() == B.cols());
            assert(A.depth() == simd_stride);
            assert(B.depth() == simd_stride);
            for (index_t c = 0; c < A.cols(); ++c)
                for (index_t r = 0; r < A.rows(); ++r)
                    for (index_t d = 0; d < simd_stride; ++d)
                        B(d, r, c) = A(d, r, c);
        };
        do_copy_L(LΨd().batch(i / simd_stride), wLΨd);
        if (i > 0)
            wLΨd(0) += wVV(simd_stride - 1);
        do_copy(LΨs().batch(i / simd_stride), wLΨs);
        do_copy_L(VV().batch(i / simd_stride), wVV);
#elif KOQKATOO_WITH_MKL
        auto LΨdi = LΨd().batch(i / simd_stride);
        assert(LΨdi.depth() == simd_stride);
        assert(wLΨd.depth() == simd_stride);
        assert(wLΨd.rows() == LΨdi.rows());
        mkl_domatcopy_batch_strided(                     //
            'C', 'T', (size_t)simd_stride,               // rows A
            (size_t)LΨdi.rows(),                         // cols A
            real_t{1},                                   // alpha
            LΨdi.data,                                   // A
            (size_t)simd_stride,                         // ldA
            (size_t)(simd_stride * LΨdi.outer_stride()), // stride A
            wLΨd.data,                                   // B
            (size_t)(wLΨd.cols() * wLΨd.outer_stride()), // ldB
            (size_t)wLΨd.outer_stride(),                 // stride B
            (size_t)wLΨd.cols()                          // batch_size
        );
        if (i > 0)
            wLΨd(0) += wVV(simd_stride - 1);
        auto LΨsi = LΨs().batch(i / simd_stride);
        assert(LΨsi.depth() == simd_stride);
        assert(wLΨs.depth() == simd_stride);
        assert(wLΨs.rows() == LΨsi.rows());
        mkl_domatcopy_batch_strided(                     //
            'C', 'T', (size_t)simd_stride,               // rows A
            (size_t)LΨsi.rows(),                         // cols A
            real_t{1},                                   // alpha
            LΨsi.data,                                   // A
            (size_t)simd_stride,                         // ldA
            (size_t)(simd_stride * LΨsi.outer_stride()), // stride A
            wLΨs.data,                                   // B
            (size_t)(wLΨs.cols() * wLΨs.outer_stride()), // ldB
            (size_t)wLΨs.outer_stride(),                 // stride B
            (size_t)wLΨs.cols()                          // batch_size
        );
        auto VVi = VV().batch(i / simd_stride);
        assert(VVi.depth() == simd_stride);
        assert(wVV.depth() == simd_stride);
        assert(wVV.rows() == VVi.rows());
        mkl_domatcopy_batch_strided(                    //
            'C', 'T', (size_t)simd_stride,              // rows A
            (size_t)VVi.rows(),                         // cols A
            real_t{1},                                  // alpha
            VVi.data,                                   // A
            (size_t)simd_stride,                        // ldA
            (size_t)(simd_stride * VVi.outer_stride()), // stride A
            wVV.data,                                   // B
            (size_t)(wVV.cols() * wVV.outer_stride()),  // ldB
            (size_t)wVV.outer_stride(),                 // stride B
            (size_t)wVV.cols()                          // batch_size
        );
#else
        // If the last batch is an incomplete one, already add Ld(N)
        for (index_t j = 0; j < std::min(ni + 1, simd_stride); ++j)
            wLΨd(j) = LΨd()(i + j);
        if (i > 0)
            wLΨd(0) += wVV(simd_stride - 1);
        for (index_t j = 0; j < ni; ++j)
            wLΨs(j) = LΨs()(i + j);
        for (index_t j = 0; j < ni; ++j)
            wVV(j) = VV()(i + j);
#endif
        for (index_t j = 0; j < ni; ++j) {
            scalar_blas::xpotrf(wLΨd.batch(j), settings.preferred_backend);
            scalar_blas::xtrsm_RLTN(wLΨd.batch(j), wLΨs.batch(j),
                                    settings.preferred_backend);
            scalar_blas::xsyrk_sub(wLΨs.batch(j), wVV.batch(j),
                                   settings.preferred_backend);
            if (j + 1 < simd_stride)
                wLΨd(j + 1) += wVV(j);
        }
        for (index_t j = 0; j < ni; ++j)
            storage.LΨd_scalar()(i + j) = wLΨd(j);
        for (index_t j = 0; j < ni; ++j)
            storage.LΨs_scalar()(i + j) = wLΨs(j);
    });
    index_t last_j = N % simd_stride;
    if (last_j == 0) {
        // If the previous batch was complete, the term VV - LsLs is in VV.
        // We load and add WW to it, factor it and store it.
        wVV(simd_stride - 1) += LΨd()(N);
        scalar_blas::xpotrf(wVV.batch(simd_stride - 1),
                            settings.preferred_backend);
        storage.LΨd_scalar()(N) = wVV(simd_stride - 1);
    } else {
        // If the previous batch was not complete, Ld has already been loaded
        // and updated by VV - LsLs.
        scalar_blas::xpotrf(wLΨd.batch(last_j), settings.preferred_backend);
        storage.LΨd_scalar()(N) = wLΨd(last_j);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::cholesky_Ψ(Timings &t) {
    std::optional<guanaqo::Timed<typename Timings::type>> timer;
    const auto N = storage.dim.N_horiz;
    auto wLΨd = storage.work_LΨd(), wLΨs = storage.work_LΨs(),
         wVV = storage.work_VV();
    foreach_chunked_merged(0, N, simd_stride, [&](index_t i, auto ni) {
#if 1
        timer.emplace(t.chol_Ψ_copy_1);
        index_t b = i / simd_stride, nd = std::min(ni + 1, simd_stride);
        compact_blas::unpack_L(LΨd().batch(b), wLΨd.first_layers(nd));
        if (i > 0)
            wLΨd(0) += wVV(simd_stride - 1);
        compact_blas::unpack(LΨs().batch(b), wLΨs.first_layers(ni));
        compact_blas::unpack_L(VV().batch(b), wVV.first_layers(ni));
#elif 1
        timer.emplace(t.chol_Ψ_copy_1);
        auto do_copy_L = [](auto &&A, auto &&B) {
            assert(A.rows() == B.rows());
            assert(A.cols() == B.cols());
            assert(A.depth() == simd_stride);
            assert(B.depth() == simd_stride);
            for (index_t c = 0; c < A.cols(); ++c)
                for (index_t r = c; r < A.rows(); ++r)
                    for (index_t d = 0; d < simd_stride; ++d)
                        B(d, r, c) = A(d, r, c);
        };
        auto do_copy = [](auto &&A, auto &&B) {
            assert(A.rows() == B.rows());
            assert(A.cols() == B.cols());
            assert(A.depth() == simd_stride);
            assert(B.depth() == simd_stride);
            for (index_t c = 0; c < A.cols(); ++c)
                for (index_t r = 0; r < A.rows(); ++r)
                    for (index_t d = 0; d < simd_stride; ++d)
                        B(d, r, c) = A(d, r, c);
        };
        do_copy_L(LΨd().batch(i / simd_stride), wLΨd);
        if (i > 0)
            wLΨd(0) += wVV(simd_stride - 1);
        do_copy(LΨs().batch(i / simd_stride), wLΨs);
        do_copy_L(VV().batch(i / simd_stride), wVV);
#elif KOQKATOO_WITH_MKL
        timer.emplace(t.chol_Ψ_copy_1);
        auto LΨdi = LΨd().batch(i / simd_stride);
        assert(LΨdi.depth() == simd_stride);
        assert(wLΨd.depth() == simd_stride);
        assert(wLΨd.rows() == LΨdi.rows());
        mkl_domatcopy_batch_strided(                     //
            'C', 'T', (size_t)simd_stride,               // rows A
            (size_t)LΨdi.rows(),                         // cols A
            real_t{1},                                   // alpha
            LΨdi.data,                                   // A
            (size_t)simd_stride,                         // ldA
            (size_t)(simd_stride * LΨdi.outer_stride()), // stride A
            wLΨd.data,                                   // B
            (size_t)(wLΨd.cols() * wLΨd.outer_stride()), // ldB
            (size_t)wLΨd.outer_stride(),                 // stride B
            (size_t)wLΨd.cols()                          // batch_size
        );
        if (i > 0)
            wLΨd(0) += wVV(simd_stride - 1);
        auto LΨsi = LΨs().batch(i / simd_stride);
        assert(LΨsi.depth() == simd_stride);
        assert(wLΨs.depth() == simd_stride);
        assert(wLΨs.rows() == LΨsi.rows());
        mkl_domatcopy_batch_strided(                     //
            'C', 'T', (size_t)simd_stride,               // rows A
            (size_t)LΨsi.rows(),                         // cols A
            real_t{1},                                   // alpha
            LΨsi.data,                                   // A
            (size_t)simd_stride,                         // ldA
            (size_t)(simd_stride * LΨsi.outer_stride()), // stride A
            wLΨs.data,                                   // B
            (size_t)(wLΨs.cols() * wLΨs.outer_stride()), // ldB
            (size_t)wLΨs.outer_stride(),                 // stride B
            (size_t)wLΨs.cols()                          // batch_size
        );
        auto VVi = VV().batch(i / simd_stride);
        assert(VVi.depth() == simd_stride);
        assert(wVV.depth() == simd_stride);
        assert(wVV.rows() == VVi.rows());
        mkl_domatcopy_batch_strided(                    //
            'C', 'T', (size_t)simd_stride,              // rows A
            (size_t)VVi.rows(),                         // cols A
            real_t{1},                                  // alpha
            VVi.data,                                   // A
            (size_t)simd_stride,                        // ldA
            (size_t)(simd_stride * VVi.outer_stride()), // stride A
            wVV.data,                                   // B
            (size_t)(wVV.cols() * wVV.outer_stride()),  // ldB
            (size_t)wVV.outer_stride(),                 // stride B
            (size_t)wVV.cols()                          // batch_size
        );
#else
        // If the last batch is an incomplete one, already add Ld(N)
        timer.emplace(t.chol_Ψ_copy_1);
        for (index_t j = 0; j < std::min(ni + 1, simd_stride); ++j)
            wLΨd(j) = LΨd()(i + j);
        if (i > 0)
            wLΨd(0) += wVV(simd_stride - 1);
        for (index_t j = 0; j < ni; ++j)
            wLΨs(j) = LΨs()(i + j);
        for (index_t j = 0; j < ni; ++j)
            wVV(j) = VV()(i + j);
#endif
        for (index_t j = 0; j < ni; ++j) {
            timer.emplace(t.chol_Ψ_potrf);
            scalar_blas::xpotrf(wLΨd.batch(j), settings.preferred_backend);
            timer.emplace(t.chol_Ψ_trsm);
            scalar_blas::xtrsm_RLTN(wLΨd.batch(j), wLΨs.batch(j),
                                    settings.preferred_backend);
            timer.emplace(t.chol_Ψ_syrk);
            scalar_blas::xsyrk_sub(wLΨs.batch(j), wVV.batch(j),
                                   settings.preferred_backend);
            if (j + 1 < simd_stride)
                wLΨd(j + 1) += wVV(j);
        }
        timer.emplace(t.chol_Ψ_copy_2);
        for (index_t j = 0; j < ni; ++j)
            storage.LΨd_scalar()(i + j) = wLΨd(j);
        for (index_t j = 0; j < ni; ++j)
            storage.LΨs_scalar()(i + j) = wLΨs(j);
    });
    index_t last_j = N % simd_stride;
    if (last_j == 0) {
        // If the previous batch was complete, the term VV - LsLs is in VV.
        // We load and add WW to it, factor it and store it.
        wVV(simd_stride - 1) += LΨd()(N);
        timer.emplace(t.chol_Ψ_potrf);
        scalar_blas::xpotrf(wVV.batch(simd_stride - 1),
                            settings.preferred_backend);
        timer.emplace(t.chol_Ψ_copy_2);
        storage.LΨd_scalar()(N) = wVV(simd_stride - 1);
    } else {
        // If the previous batch was not complete, Ld has already been loaded
        // and updated by VV - LsLs.
        timer.emplace(t.chol_Ψ_potrf);
        scalar_blas::xpotrf(wLΨd.batch(last_j), settings.preferred_backend);
        timer.emplace(t.chol_Ψ_copy_2);
        storage.LΨd_scalar()(N) = wLΨd(last_j);
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve_Ψ_scalar(std::span<real_t> λ) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    scalar_mut_real_view λ_{{.data = λ.data(), .depth = N + 1, .rows = nx}};
    for (index_t i = 0; i < N + 1; ++i) {
        // λ[i] = L[i,i]⁻¹ b[i]
        scalar_blas::xtrsm_LLNN(storage.LΨd_scalar().batch(i), λ_.batch(i),
                                settings.preferred_backend);
        if (i < N) {
            // b[i+1] -= L[i+1,i] λ[i]
            scalar_blas::xgemm_sub(storage.LΨs_scalar().batch(i), λ_.batch(i),
                                   λ_.batch(i + 1), settings.preferred_backend);
        }
    }
    for (index_t i = N + 1; i-- > 0;) {
        // λ[i] = L[i,i]⁻ᵀ b[i]
        scalar_blas::xtrsm_LLTN(storage.LΨd_scalar().batch(i), λ_.batch(i),
                                settings.preferred_backend);
        if (i > 0) {
            // b[i-1] -= L[i,i-1]ᵀ λ[i]
            scalar_blas::xgemm_TN_sub(storage.LΨs_scalar().batch(i - 1),
                                      λ_.batch(i), λ_.batch(i - 1),
                                      settings.preferred_backend);
        }
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve_Ψ_scalar(std::span<real_t> λ, Timings &t) {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    scalar_mut_real_view λ_{{.data = λ.data(), .depth = N + 1, .rows = nx}};
    std::optional<guanaqo::Timed<typename Timings::type>> timer;
    for (index_t i = 0; i < N + 1; ++i) {
        timer.emplace(t.solve_Ψ_solve);
        // λ[i] = L[i,i]⁻¹ b[i]
        scalar_blas::xtrsm_LLNN(storage.LΨd_scalar().batch(i), λ_.batch(i),
                                settings.preferred_backend);
        if (i < N) {
            timer.emplace(t.solve_Ψ_gemm);
            // b[i+1] -= L[i+1,i] λ[i]
            scalar_blas::xgemm_sub(storage.LΨs_scalar().batch(i), λ_.batch(i),
                                   λ_.batch(i + 1), settings.preferred_backend);
        }
    }
    for (index_t i = N + 1; i-- > 0;) {
        timer.emplace(t.solve_Ψ_solve_tp);
        // λ[i] = L[i,i]⁻ᵀ b[i]
        scalar_blas::xtrsm_LLTN(storage.LΨd_scalar().batch(i), λ_.batch(i),
                                settings.preferred_backend);
        if (i > 0) {
            timer.emplace(t.solve_Ψ_gemm_tp);
            // b[i-1] -= L[i,i-1]ᵀ λ[i]
            scalar_blas::xgemm_TN_sub(storage.LΨs_scalar().batch(i - 1),
                                      λ_.batch(i), λ_.batch(i - 1),
                                      settings.preferred_backend);
        }
    }
}

template <simd_abi_tag Abi>
void Solver<Abi>::factor(real_t S, real_view Σ, bool_view J) {
    prepare_all(S, Σ, J);
    cholesky_Ψ();
}

template <simd_abi_tag Abi>
void Solver<Abi>::factor(real_t S, real_view Σ, bool_view J, Timings &t) {
    timed(t.prepare_all, [&] { prepare_all(S, Σ, J, t); });
    timed(t.cholesky_Ψ, [&] { cholesky_Ψ(t); });
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve(real_view grad, real_view Mᵀλ, real_view Aᵀŷ,
                        real_view Mxb, mut_real_view d, mut_real_view Δλ,
                        mut_real_view MᵀΔλ) {
#if 0
    // d ← ∇f̃(x) + Mᵀλ + Aᵀŷ (= v)
    compact_blas::xadd_copy(d, grad, Mᵀλ, Aᵀŷ);
    // d ← H⁻¹ d
    solve_H(d);
    // Δλ ← Md - (Mx - b)
    residual_dynamics_constr(d, Mxb, Δλ);
    // Δλ ← (M H⁻¹ Mᵀ)⁻¹ Δλ
    storage.restore_dynamics_constraints(Δλ, storage.Δλ_scalar);
    solve_Ψ_scalar(storage.Δλ_scalar);
    storage.copy_dynamics_constraints(storage.Δλ_scalar, Δλ);
    // MᵀΔλ ← Mᵀ Δλ
    mat_vec_transpose_dynamics_constr(Δλ, MᵀΔλ);
    // d ← MᵀΔλ - ∇f̃(x) - Mᵀλ - Aᵀŷ
    compact_blas::xsub_copy(d, MᵀΔλ, Mᵀλ, grad, Aᵀŷ);
    // d ← H⁻¹ d
    solve_H(d);
#else

    auto [N, nx, nu, ny, ny_N] = storage.dim;
    const auto be              = settings.preferred_backend;
    scalar_mut_real_view Δλ_scal{{
        .data  = storage.Δλ_scalar.data(),
        .depth = N + 1,
        .rows  = nx,
        .cols  = 1,
    }};
    assert(storage.Δλ_scalar.size() == static_cast<size_t>(Δλ_scal.size()));
    auto &Δλ1 = storage.λ1;
    auto LΨd = storage.LΨd_scalar(), LΨs = storage.LΨs_scalar();

    // Parallel solve Hv = g
    // ---------------------
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < LH().num_batches(); ++i) {
        // Initialize rhs: g = ∇ϕ + Mᵀλ = ∇f̃ + Aᵀŷ + Mᵀλ                 (d ← g)
        for (index_t j = 0; j < nx + nu; ++j)
            for (index_t ii = i * simd_stride; ii < (i + 1) * simd_stride; ++ii)
                d(ii, j, 0) = grad(ii, j, 0) + Mᵀλ(ii, j, 0) + Aᵀŷ(ii, j, 0);
        // Solve Lᴴ vʹ = g                                              (d ← vʹ)
        compact_blas::xtrsm_LLNN(LH().batch(i), d.batch(i), be);
        // Solve Lᴴ⁻ᵀ v = vʹ                                             (d ← v)
        compact_blas::xtrsm_LLTN(LH().batch(i), d.batch(i), be);
        // Compute f = (A B) v                                          (Δλ ← f)
        if (i < AB().num_batches())
            compact_blas::xgemm(AB().batch(i), d.batch(i), Δλ.batch(i), be);
    }

    // Forward substitution Ψ
    // ----------------------
    // Initialize rhs r - v = Mx - b - v                         (Δλ_scal ← ...)
    for (index_t j = 0; j < nx; ++j)
        Δλ_scal(0, j, 0) = Mxb(0, j, 0) - d(0, j, 0);
    // Solve L(d) Δλʹ = r - v                                    (Δλ_scal ← Δλʹ)
    scalar_blas::xtrsm_LLNN(LΨd.batch(0), Δλ_scal.batch(0), be);
    for (index_t i = 1; i <= N; ++i) {
        // Initialize rhs r + f - v = Mx - b + (A B) v - v       (Δλ_scal ← ...)
        for (index_t j = 0; j < nx; ++j)
            Δλ_scal(i, j, 0) = Mxb(i, j, 0) - d(i, j, 0) + Δλ(i - 1, j, 0);
        // Subtract L(s) Δλʹ(i - 1)                              (Δλ_scal ← ...)
        scalar_blas::xgemm_sub(LΨs.batch(i - 1), Δλ_scal.batch(i - 1),
                               Δλ_scal.batch(i), be);
        // Solve L(d) Δλʹ = r + f - v - L(s) Δλʹ(i - 1)          (Δλ_scal ← ...)
        scalar_blas::xtrsm_LLNN(LΨd.batch(i), Δλ_scal.batch(i), be);
    }

    // Backward substitution Ψ
    // -----------------------
    scalar_blas::xtrsm_LLTN(LΨd.batch(N), Δλ_scal.batch(N), be);
    for (index_t j = 0; j < nx; ++j)
        Δλ1(N - 1, j, 0) = Δλ(N, j, 0) = Δλ_scal(N, j, 0);
    for (index_t i = N; i-- > 0;) {
        scalar_blas::xgemm_TN_sub(LΨs.batch(i), Δλ_scal.batch(i + 1),
                                  Δλ_scal.batch(i), be);
        scalar_blas::xtrsm_LLTN(LΨd.batch(i), Δλ_scal.batch(i), be);
        if (i > 0)
            for (index_t j = 0; j < nx; ++j)
                Δλ1(i - 1, j, 0) = Δλ(i, j, 0) = Δλ_scal(i, j, 0);
    }
    for (index_t j = 0; j < nx; ++j)
        Δλ(0, j, 0) = Δλ_scal(0, j, 0);

    // Parallel solve Hd = -g - MᵀΔλ
    // -----------------------------
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < LH().num_batches(); ++i) {
        MᵀΔλ.batch(i).top_rows(nx) = Δλ.batch(i);
        MᵀΔλ.batch(i).bottom_rows(nu).set_constant(0);
        if (i < AB().num_batches())
            compact_blas::xgemm_TN_sub(AB().batch(i), Δλ1.batch(i),
                                       MᵀΔλ.batch(i), be);
        for (index_t j = 0; j < nx + nu; ++j)
            for (index_t ii = i * simd_stride; ii < (i + 1) * simd_stride; ++ii)
                d(ii, j, 0) = -grad(ii, j, 0) - Mᵀλ(ii, j, 0) - Aᵀŷ(ii, j, 0) -
                              MᵀΔλ(ii, j, 0);
        compact_blas::xtrsm_LLNN(LH().batch(i), d.batch(i), be);
        compact_blas::xtrsm_LLTN(LH().batch(i), d.batch(i), be);
    }

#endif
}

template <simd_abi_tag Abi>
void Solver<Abi>::solve(real_view grad, real_view Mᵀλ, real_view Aᵀŷ,
                        real_view Mxb, mut_real_view d, mut_real_view Δλ,
                        mut_real_view MᵀΔλ, Timings &t) {
    // d ← ∇f̃(x) + Mᵀλ + Aᵀŷ (= v)
    timed(t.solve_add_rhs_1,
          [&] { compact_blas::xadd_copy(d, grad, Mᵀλ, Aᵀŷ); });
    // d ← H⁻¹ d
    timed(t.solve_H_1, &Solver::solve_H, this, d);
    // Δλ ← Md - (Mx - b)
    timed(t.solve_mat_vec, &Solver::residual_dynamics_constr, this, d, Mxb, Δλ);
    // Δλ ← (M H⁻¹ Mᵀ)⁻¹ Δλ
    timed(t.solve_unshuffle, &storage_t::restore_dynamics_constraints, storage,
          Δλ, storage.Δλ_scalar);
    timed(t.solve_Ψ, [&] { solve_Ψ_scalar(storage.Δλ_scalar, t); });
    timed(t.solve_shuffle, &storage_t::copy_dynamics_constraints, storage,
          storage.Δλ_scalar, Δλ);
    // MᵀΔλ ← Mᵀ Δλ
    timed(t.solve_mat_vec_tp, &Solver::mat_vec_transpose_dynamics_constr, this,
          Δλ, MᵀΔλ);
    // d ← MᵀΔλ - ∇f̃(x) - Mᵀλ - Aᵀŷ
    timed(t.solve_add_rhs_2,
          [&] { compact_blas::xsub_copy(d, MᵀΔλ, Mᵀλ, grad, Aᵀŷ); });
    // d ← H⁻¹ d
    timed(t.solve_H_2, &Solver::solve_H, this, d);
}

template <simd_abi_tag Abi>
Solver<Abi>::Solver(const LinearOCPStorage &ocp) : storage{.dim = ocp.dim} {
    auto [N, nx, nu, ny, ny_N] = storage.dim;
    auto H = this->H(), CD = this->CD(), AB = this->AB();
    KOQKATOO_OMP(parallel for)
    for (index_t i = 0; i < N; ++i) {
        H(i)  = ocp.H(i);
        CD(i) = ocp.CD(i);
        AB(i) = ocp.AB(i);
    }
    H(N).set_constant(0);
    H(N).set_diagonal(1);
    H(N).top_left(nx, nx) = ocp.H(N);
    CD(N).set_constant(0);
    CD(N).top_left(ny_N, nx) = ocp.CD(N);
    for (index_t i = N + 1; i < H.ceil_depth(); ++i) {
        H(i).set_constant(0);
        H(i).set_diagonal(1);
        CD(i).set_constant(0);
    }
    for (index_t i = N; i < AB.ceil_depth(); ++i)
        AB(i).set_constant(0);
}

} // namespace koqkatoo::ocp
