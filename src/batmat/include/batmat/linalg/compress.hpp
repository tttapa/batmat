#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/ops/gather.hpp>
#include <guanaqo/trace.hpp>

namespace batmat::linalg {

namespace detail {

template <class T, class Abi, index_t N = 8, StorageOrder OAi>
[[gnu::always_inline, gnu::flatten]] inline index_t
compress_masks_impl(view<const T, Abi, OAi> A_in, view<const T, Abi> S_in, auto writeS,
                    auto writeA) noexcept {
    using batmat::ops::gather;
    BATMAT_ASSUME(A_in.depth() == S_in.depth());
    BATMAT_ASSUME(A_in.cols() == S_in.rows());
    BATMAT_ASSUME(S_in.cols() == 1);
    const auto C = A_in.cols();
    const auto R = A_in.rows();
    if (C == 0)
        return 0;
    BATMAT_ASSUME(R > 0);
    using types              = simd_view_types<T, Abi>;
    using simd               = typename types::simd;
    using isimd              = typename types::isimd;
    static constexpr auto VL = simd::size();

    isimd hist[N]{};
    simd Shist[N]{};
    index_t j = 0;
    static const isimd iota{[](auto i) { return i; }};

    const auto commit_no_shift = [&](isimd h_commit, simd S_commit) {
        writeS(S_commit, j);
        const auto bs       = static_cast<index_t>(A_in.batch_size());
        const auto stride   = (OAi == StorageOrder::ColMajor ? bs * A_in.outer_stride() : bs);
        const isimd offsets = h_commit * stride + iota;
        for (index_t r = 0; r < R; ++r) {
            auto gather_A                  = gather<T, VL>(&A_in(0, r, 0), offsets);
            where(S_commit == 0, gather_A) = simd{}; // TODO: masked gather
            writeA(gather_A, S_commit, r, j);
        }
        ++j;
    };
    const auto commit = [&] [[gnu::always_inline]] () {
        const isimd h = hist[0];
        const simd S  = Shist[0];
        BATMAT_FULLY_UNROLLED_FOR (index_t k = 1; k < N; ++k) {
            hist[k - 1]  = hist[k];
            Shist[k - 1] = Shist[k];
        }
        hist[N - 1]  = {};
        Shist[N - 1] = {};
        commit_no_shift(h, S);
    };

    isimd c_simd{0};
    for (index_t c = 0; c < C; ++c) {
        simd Sc = types::aligned_load(&S_in(0, c, 0));
        BATMAT_FULLY_UNROLLED_FOR (index_t i = 0; i < N; ++i) {
            auto &h               = hist[i];
            auto &Sh              = Shist[i];
            const auto msk        = (Sh == 0) && !(Sc == 0);
            where(msk, Sh)        = Sc;
            where(msk.__cvt(), h) = c_simd;
            where(msk, Sc)        = simd{};
        }
        // Masks of all ones can already be written to memory
        if (none_of(Shist[0] == 0) || !all_of(Sc == 0)) {
            commit();
            assert(any_of(Shist[0] == 0)); // at most one commit per iteration is possible
            if (!all_of(Sc == 0)) {
                // We now have room for any remaining bits
                auto &h               = hist[N - 1];
                auto &Sh              = Shist[N - 1];
                const auto msk        = !(Sc == 0);
                where(msk, Sh)        = Sc;
                where(msk.__cvt(), h) = c_simd;
            }
        }
        // Invariant: first registers in the buffer contain fewest zeros
        BATMAT_FULLY_UNROLLED_FOR (index_t i = 1; i < N; ++i)
#if BATMAT_WITH_GSI_HPC_SIMD
            assert(reduce_count(hist[i] != 0) <= reduce_count(hist[i - 1] != 0));
#else
            assert(popcount(hist[i] != 0) <= popcount(hist[i - 1] != 0));
#endif
        c_simd += isimd{1};
    }
    BATMAT_FULLY_UNROLLED_FOR (index_t i = 0; i < N; ++i) {
        auto &h = hist[i];
        auto &S = Shist[i];
        if (!all_of(S != 0))
            commit_no_shift(h, S);
    }
    return j;
}

template <class T, class Abi, index_t N = 8>
index_t compress_masks_count(view<const T, Abi> S_in) {
    GUANAQO_TRACE("compress_masks_count", 0, S_in.rows() * S_in.depth());
    const auto C = S_in.rows();
    if (C == 0)
        return 0;
    using types = simd_view_types<T, Abi>;
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
        auto Sc_msk   = !(Sc == 0);
        BATMAT_FULLY_UNROLLED_FOR (auto &h : hist) {
#if BATMAT_WITH_GSI_HPC_SIMD // TODO
            auto h_ = h;
            h       = isimd{[&](int i) -> index_t { return h[i] == 0 && Sc_msk[i] ? 1 : h[i]; }};
            Sc_msk  = decltype(Sc_msk){[&](int i) -> bool { return Sc_msk[i] && h_[i] != 0; }};
#else
            const auto msk = (h == 0) && Sc_msk.__cvt();
            where(msk, h)  = 1;
            Sc_msk         = Sc_msk && (!msk).__cvt();
#endif
        }
        // Masks of all ones can already be written to memory
        if (none_of(hist[0] == 0))
            commit();
        assert(any_of(hist[0] == 0)); // at most one commit per iteration is possible
        // If there are still bits set in the mask.
        if (any_of(Sc_msk)) {
            // Check if there's an empty slot (always at the end)
            auto &h = hist[N - 1];
            if (any_of(h != 0))
                commit(); // If not, commit the first slot to make room.
            assert(all_of(h == 0));
#if BATMAT_WITH_GSI_HPC_SIMD // TODO
            h = isimd{[&](int i) -> index_t { return Sc_msk[i] ? 1 : h[i]; }};
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
    BATMAT_FULLY_UNROLLED_FOR (auto &h : hist)
        if (any_of(h != 0))
            ++j;
    return j;
}

template <class T, class Abi, index_t N = 4, StorageOrder OAi, StorageOrder OAo>
index_t compress_masks(view<const T, Abi, OAi> A_in, view<const T, Abi> S_in,
                       view<T, Abi, OAo> A_out, view<T, Abi> S_out) noexcept {
    GUANAQO_TRACE("compress_masks", 0, (A_in.rows() + 1) * A_in.cols() * A_in.depth());
    BATMAT_ASSERT(A_in.rows() == A_out.rows());
    BATMAT_ASSERT(A_in.cols() == A_out.cols());
    BATMAT_ASSERT(A_in.depth() == A_out.depth());
    BATMAT_ASSERT(S_in.rows() == S_out.rows());
    BATMAT_ASSERT(S_in.cols() == S_out.cols());
    BATMAT_ASSERT(S_in.depth() == S_out.depth());
    auto writeS = [S_out] [[gnu::always_inline]] (auto gather_S, index_t j) {
        datapar::aligned_store(gather_S, &S_out(0, j, 0));
    };
    auto writeA = [A_out] [[gnu::always_inline]] (auto gather_A, auto /*gather_S*/, index_t r,
                                                  index_t j) {
        datapar::aligned_store(gather_A, &A_out(0, r, j));
    };
    return compress_masks_impl<T, Abi, N, OAi>(A_in, S_in, writeS, writeA);
}

template <class T, class Abi, index_t N = 4, StorageOrder OAi, StorageOrder OAo>
index_t compress_masks_sqrt(view<const T, Abi, OAi> A_in, view<const T, Abi> S_in,
                            view<T, Abi, OAo> A_out, view<T, Abi> S_sign_out = {}) {
    GUANAQO_TRACE("compress_masks_sqrt", 0, (A_in.rows() + 1) * A_in.cols() * A_in.depth());
    BATMAT_ASSERT(A_in.rows() == A_out.rows());
    BATMAT_ASSERT(A_in.cols() == A_out.cols());
    BATMAT_ASSERT(A_in.depth() == A_out.depth());
    BATMAT_ASSERT(S_sign_out.rows() == 0 || A_in.depth() == S_sign_out.depth());
    BATMAT_ASSERT(S_sign_out.rows() == 0 || A_out.cols() == S_sign_out.rows());
    BATMAT_ASSERT(S_sign_out.rows() == 0 || S_sign_out.cols() == 1);
    using std::abs;
    using std::copysign;
    using std::sqrt;
    auto writeS = [S_sign_out] [[gnu::always_inline]] (auto gather_S, index_t j) {
        if (S_sign_out.rows() > 0)
            datapar::aligned_store(copysign(decltype(gather_S){0}, gather_S), &S_sign_out(0, j, 0));
    };
    auto writeA = [A_out] [[gnu::always_inline]] (auto gather_A, auto gather_S, index_t r,
                                                  index_t j) {
        datapar::aligned_store(sqrt(abs(gather_S)) * gather_A, &A_out(0, r, j));
    };
    return compress_masks_impl<T, Abi, N, OAi>(A_in, S_in, writeS, writeA);
}

} // namespace detail

template <index_t N = 8, simdifiable VA, simdifiable VS, simdifiable VAo, simdifiable VSo>
index_t compress_masks(VA &&Ain, VS &&Sin, VAo &&Aout, VSo &&Sout) {
    return detail::compress_masks<simdified_value_t<VA>, simdified_abi_t<VA>, N>(
        simdify(Ain).as_const(), simdify(Sin).as_const(), simdify(Aout), simdify(Sout));
}

template <index_t N = 8, simdifiable VS>
index_t compress_masks_count(VS &&Sin) {
    return detail::compress_masks_count<simdified_value_t<VS>, simdified_abi_t<VS>, N>(
        simdify(Sin).as_const());
}

template <index_t N = 8, simdifiable VA, simdifiable VS, simdifiable VAo>
index_t compress_masks_sqrt(VA &&Ain, VS &&Sin, VAo &&Aout) {
    return detail::compress_masks_sqrt<simdified_value_t<VA>, simdified_abi_t<VA>, N>(
        simdify(Ain).as_const(), simdify(Sin).as_const(), simdify(Aout));
}

template <index_t N = 8, simdifiable VA, simdifiable VS, simdifiable VAo, simdifiable VSo>
index_t compress_masks_sqrt(VA &&Ain, VS &&Sin, VAo &&Aout, VSo &&Sout) {
    return detail::compress_masks_sqrt<simdified_value_t<VA>, simdified_abi_t<VA>, N>(
        simdify(Ain).as_const(), simdify(Sin).as_const(), simdify(Aout), simdify(Sout));
}

} // namespace batmat::linalg
