#pragma once

#include <batmat/config.hpp>
#include <batmat/openmp.h>
#if !BATMAT_WITH_OPENMP
#include <batmat/thread-pool.hpp>
#endif

namespace batmat {

/// @ingroup topic-utils
enum class LoopDir {
    Forward,
    Backward,
};

/// Iterate over the range `[i_begin, i_end)` in chunks of size @p chunk_size, calling @p func_chunk
/// for each full chunk and @p func_rem for the remaining elements (if any).
/// @ingroup topic-utils
[[gnu::always_inline]] inline void foreach_chunked(index_t i_begin, index_t i_end, auto chunk_size,
                                                   auto func_chunk, auto func_rem,
                                                   LoopDir dir = LoopDir::Forward) {
    if (dir == LoopDir::Forward) {
        index_t i;
        for (i = i_begin; i + chunk_size <= i_end; i += chunk_size)
            func_chunk(i);
        index_t rem_i = i_end - i;
        if (rem_i > 0)
            func_rem(i, rem_i);
    } else {
        index_t rem_i = (i_end - i_begin) % chunk_size;
        index_t i     = i_end - rem_i;
        if (rem_i > 0)
            func_rem(i, rem_i);
        for (i -= chunk_size; i >= i_begin; i -= chunk_size)
            func_chunk(i);
    }
}

/// Iterate over the range `[i_begin, i_end)` in chunks of size @p chunk_size, calling @p func_chunk
/// for each chunk (including the last chunk, which may be smaller than @p chunk_size).
/// @ingroup topic-utils
[[gnu::always_inline]] inline void foreach_chunked_merged(index_t i_begin, index_t i_end,
                                                          auto chunk_size, auto func_chunk,
                                                          LoopDir dir = LoopDir::Forward) {
    if (dir == LoopDir::Forward) {
        index_t i;
        for (i = i_begin; i + chunk_size <= i_end; i += chunk_size)
            func_chunk(i, chunk_size);
        index_t rem_i = i_end - i;
        if (rem_i > 0)
            func_chunk(i, rem_i);
    } else {
        index_t rem_i = (i_end - i_begin) % chunk_size;
        index_t i     = i_end - rem_i;
        if (rem_i > 0)
            func_chunk(i, rem_i);
        for (i -= chunk_size; i >= i_begin; i -= chunk_size)
            func_chunk(i, chunk_size);
    }
}

/// @deprecated
[[deprecated, gnu::always_inline]] inline void
foreach_chunked_merged_parallel(index_t i_begin, index_t i_end, auto chunk_size, auto func_chunk,
                                LoopDir dir = LoopDir::Forward) {
    const index_t rem_i = (i_end - i_begin) % chunk_size;
    if (dir == LoopDir::Forward) {
        BATMAT_OMP(parallel) {
            BATMAT_OMP(for nowait)
            for (index_t i = i_begin; i <= i_end - chunk_size; i += chunk_size)
                func_chunk(i, chunk_size);
            BATMAT_OMP(single) {
                if (rem_i > 0)
                    func_chunk(i_end - rem_i, rem_i);
            }
        }
    } else {
        const index_t i_last = i_end - rem_i;
        BATMAT_OMP(parallel) {
            BATMAT_OMP(single nowait) {
                if (rem_i > 0)
                    func_chunk(i_last, rem_i);
            }
            BATMAT_OMP(for)
            for (index_t i = i_last - chunk_size; i >= i_begin; i -= chunk_size)
                func_chunk(i, chunk_size);
        }
    }
}

/// @deprecated
[[deprecated, gnu::always_inline]] inline void foreach_thread(auto &&func) {
#if BATMAT_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        func(index_t{0}, index_t{1});
    } else {
        BATMAT_OMP(parallel) {
            auto ni = static_cast<index_t>(omp_get_num_threads());
            BATMAT_OMP(for schedule(static))
            for (index_t i = 0; i < ni; ++i)
                func(i, ni);
        }
    }
#else
    pool_sync_run_all<index_t>(func);
#endif
}

/// @deprecated
[[deprecated, gnu::always_inline]] inline void foreach_thread(index_t num_threads, auto &&func) {
#if BATMAT_WITH_OPENMP
    if (num_threads == 1) {
        func(index_t{0}, index_t{1});
    } else {
        BATMAT_OMP(parallel num_threads(num_threads)) {
            auto ni = static_cast<index_t>(omp_get_num_threads());
            BATMAT_OMP(for schedule(static))
            for (index_t i = 0; i < ni; ++i)
                func(i, ni);
        }
    }
#else
    pool_sync_run_n<index_t>(num_threads, func);
#endif
}

} // namespace batmat
