#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/ops/gather.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

template <class Abi, index_t N = 8>
index_t compress_masks(real_view<Abi> A_in, real_view<Abi> S_in, mut_real_view<Abi> A_out,
                       mut_real_view<Abi> S_out) {
    GUANAQO_TRACE("compress_masks", 0, (A_in.rows() + 1) * A_in.cols() * A_in.depth());
    using batmat::ops::gather;
    assert(A_in.rows() == A_out.rows());
    assert(A_in.cols() == A_out.cols());
    assert(A_in.depth() == A_out.depth());
    assert(A_in.depth() == S_in.depth());
    assert(A_in.depth() == S_out.depth());
    assert(A_in.cols() == S_in.rows());
    assert(A_out.cols() == S_out.rows());
    const auto C = A_in.cols();
    const auto R = A_in.rows();
    if (C == 0)
        return 0;
    BATMAT_ASSUME(R > 0);
    using types              = simd_view_types<real_t, Abi>;
    using simd               = typename types::simd;
    using isimd              = typename types::isimd;
    static constexpr auto VL = simd::size();

    isimd hist[N]{};
    index_t j = 0;
    static const isimd iota{[](auto i) { return i; }};

    const auto commit_no_shift = [&](auto h_commit) {
        h_commit -= 1;
        {
            const auto bs       = static_cast<index_t>(S_in.batch_size());
            const isimd offsets = h_commit * bs + iota;
            auto gather_S       = gather<real_t, VL>(&S_in(0, 0, 0), offsets);
            types::aligned_store(gather_S, &S_out(0, j, 0));
        }
        for (index_t r = 0; r < R; ++r) {
            const auto bs       = static_cast<index_t>(A_in.batch_size());
            const isimd offsets = h_commit * bs * A_in.outer_stride() + iota;
            auto gather_A       = gather<real_t, VL>(&A_in(0, r, 0), offsets);
            types::aligned_store(gather_A, &A_out(0, r, j));
        }
        ++j;
    };
    const auto commit = [&] [[gnu::always_inline]] () {
        const isimd h = hist[0];
        BATMAT_FULLY_UNROLLED_FOR (index_t k = 1; k < N; ++k)
            hist[k - 1] = hist[k];
        hist[N - 1] = 0;
        commit_no_shift(h);
    };

    isimd c1_simd{0};
    for (index_t c = 0; c < C; ++c) {
        c1_simd += 1; // current column index + 1
        const simd Sc = types::aligned_load(&S_in(0, c, 0));
        auto Sc_msk   = Sc != 0;
        BATMAT_FULLY_UNROLLED_FOR (auto &h : hist) {
#if BATMAT_WITH_GSI_HPC_SIMD
            const auto msk = (h == 0) && Sc_msk;
            h              = isimd{[&](int i) { return msk[i] ? c1_simd[i] : h[i]; }}; // TODO
            Sc_msk         = Sc_msk && (!msk);
#else
            const auto msk = (h == 0) && Sc_msk.__cvt();
            where(msk, h)  = c1_simd;
            Sc_msk         = Sc_msk && (!msk).__cvt();
#endif
        }
        // Masks of all ones can already be written to memory
        while (none_of(hist[0] == 0) || (c + 1 == C && any_of(hist[0] != 0)))
            commit();
        // If there are still bits set in the mask.
        if (any_of(Sc_msk)) {
            // Check if there's an empty slot (always at the end)
            if (any_of(hist[N - 1] != 0))
                // If not, commit the first slot to make room.
                commit();
            // Find the first empty slot, and add the remaining bits.
            BATMAT_FULLY_UNROLLED_FOR (auto &h : hist)
                if (all_of(h == 0))
#if BATMAT_WITH_GSI_HPC_SIMD
                    h = isimd{[&](int i) { return Sc_msk[i] ? c1_simd[i] : h[i]; }}; // TODO
#else
                    where(Sc_msk.__cvt(), h) = c1_simd;
#endif
        }
        // Invariant: first registers in the buffer contain fewest zeros
        BATMAT_FULLY_UNROLLED_FOR (index_t i = 1; i < N; ++i)
#if BATMAT_WITH_GSI_HPC_SIMD
            assert(reduce_count(hist[i] != 0) <= reduce_count(hist[i - 1] != 0));
#else
            assert(popcount(hist[i] != 0) <= popcount(hist[i - 1] != 0));
#endif
    }
    if (any_of(hist[0] != 0))
        commit_no_shift(hist[0]);
    return j;
}

template <class Abi, index_t N = 8>
index_t compress_masks_count(real_view<Abi> S_in) {
    GUANAQO_TRACE("compress_masks_count", 0, S_in.rows() * S_in.depth());
    const auto C = S_in.rows();
    if (C == 0)
        return 0;
    using types = simd_view_types<real_t, Abi>;
    using simd  = typename types::simd;
    using isimd = typename types::isimd;
    isimd hist[N]{};
    index_t j = 0;

    const auto commit = [&] [[gnu::always_inline]] () {
        BATMAT_FULLY_UNROLLED_FOR (index_t k = 1; k < N; ++k)
            hist[k - 1] = hist[k];
        hist[N - 1] = 0;
        ++j;
    };

    for (index_t c = 0; c < C; ++c) {
        const simd Sc = types::aligned_load(&S_in(0, c, 0));
        auto Sc_msk   = Sc != 0;
        BATMAT_FULLY_UNROLLED_FOR (auto &h : hist) {
#if BATMAT_WITH_GSI_HPC_SIMD
            const auto msk = (h == 0) && Sc_msk;
            h              = isimd{[&](int i) { return msk[i] ? 1 : h[i]; }}; // TODO
            Sc_msk         = Sc_msk && (!msk);
#else
            const auto msk = (h == 0) && Sc_msk.__cvt();
            where(msk, h)  = 1;
            Sc_msk         = Sc_msk && (!msk).__cvt();
#endif
        }
        // Masks of all ones can already be written to memory
        while (none_of(hist[0] == 0) || (c + 1 == C && any_of(hist[0] != 0)))
            commit();
        // If there are still bits set in the mask.
        if (any_of(Sc_msk)) {
            // Check if there's an empty slot (always at the end)
            if (any_of(hist[N - 1] != 0))
                // If not, commit the first slot to make room.
                commit();
            // Find the first empty slot, and add the remaining bits.
            BATMAT_FULLY_UNROLLED_FOR (auto &h : hist)
                if (all_of(h == 0))
#if BATMAT_WITH_GSI_HPC_SIMD
                    h = isimd{[&](int i) { return Sc_msk[i] ? 1 : h[i]; }}; // TODO
#else
                    where(Sc_msk.__cvt(), h) = 1;
#endif
        }
        // Invariant: first registers in the buffer contain fewest zeros
        BATMAT_FULLY_UNROLLED_FOR (index_t i = 1; i < N; ++i)
#if BATMAT_WITH_GSI_HPC_SIMD
            assert(reduce_count(hist[i] != 0) <= reduce_count(hist[i - 1] != 0));
#else
            assert(popcount(hist[i] != 0) <= popcount(hist[i - 1] != 0));
#endif
    }
    if (any_of(hist[0] != 0))
        ++j;
    return j;
}

} // namespace batmat::linalg
