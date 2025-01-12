#pragma once

#include <koqkatoo/config.hpp>
#include <koqkatoo/openmp.h>
#include <koqkatoo/thread-pool.hpp>

namespace koqkatoo {

enum class LoopDir {
    Forward,
    Backward,
};

[[gnu::always_inline]] inline void
foreach_chunked(index_t i_begin, index_t i_end, auto chunk_size,
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

[[gnu::always_inline]] inline void
foreach_chunked_merged(index_t i_begin, index_t i_end, auto chunk_size,
                       auto func_chunk, LoopDir dir = LoopDir::Forward) {
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

[[gnu::always_inline]] inline void
foreach_chunked_merged_parallel(index_t i_begin, index_t i_end, auto chunk_size,
                                auto func_chunk,
                                LoopDir dir = LoopDir::Forward) {
    const index_t rem_i = (i_end - i_begin) % chunk_size;
    if (dir == LoopDir::Forward) {
        KOQKATOO_OMP(parallel) {
            KOQKATOO_OMP(for nowait)
            for (index_t i = i_begin; i <= i_end - chunk_size; i += chunk_size)
                func_chunk(i, chunk_size);
            KOQKATOO_OMP(single) {
                if (rem_i > 0)
                    func_chunk(i_end - rem_i, rem_i);
            }
        }
    } else {
        const index_t i_last = i_end - rem_i;
        KOQKATOO_OMP(parallel) {
            KOQKATOO_OMP(single nowait) {
                if (rem_i > 0)
                    func_chunk(i_last, rem_i);
            }
            KOQKATOO_OMP(for)
            for (index_t i = i_last - chunk_size; i >= i_begin; i -= chunk_size)
                func_chunk(i, chunk_size);
        }
    }
}

[[gnu::always_inline]] inline void foreach_thread(auto &&func) {
#if KOQKATOO_WITH_OPENMP
    if (omp_get_max_threads() == 1) {
        func(index_t{0}, index_t{1});
    } else {
        KOQKATOO_OMP(parallel) {
            auto ni = static_cast<index_t>(omp_get_num_threads());
            KOQKATOO_OMP(for schedule(static))
            for (index_t i = 0; i < ni; ++i)
                func(i, ni);
        }
    }
#else
    pool->sync_run_all<index_t>(func);
#endif
}

} // namespace koqkatoo
