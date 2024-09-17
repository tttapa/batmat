#pragma once

#include <koqkatoo/config.hpp>

namespace koqkatoo {

enum class LoopDir {
    Forward,
    Backward,
};

[[gnu::always_inline]] void foreach_chunked(index_t i_begin, index_t i_end,
                                            auto chunk_size, auto func_chunk,
                                            auto func_rem,
                                            LoopDir dir = LoopDir::Forward) {
    if (dir == LoopDir::Forward) {
        index_t i;
        for (i = i_begin; i + chunk_size <= i_end; i += chunk_size)
            func_chunk(i);
        index_t rem_i = i_end - i;
        if (rem_i > 0)
            func_rem(i, rem_i);
    } else {
        index_t rem_i = i_end % chunk_size;
        index_t i     = i_end - rem_i;
        if (rem_i > 0)
            func_rem(i, rem_i);
        for (i -= chunk_size; i >= i_begin; i -= chunk_size)
            func_chunk(i);
    }
}

} // namespace koqkatoo
