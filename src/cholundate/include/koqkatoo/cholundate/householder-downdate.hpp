#pragma once

#include <koqkatoo/matrix-view.hpp>

namespace koqkatoo::cholundate::householder {

struct Config {
    /// Block size of the block column of L to process in the micro-kernels.
    index_t block_size_r;
    /// Block size of the block row of L to process in the micro-kernels.
    index_t block_size_s;
    /// Number of block columns per cache block.
    index_t num_blocks_r = 1;
    /// Number of block rows per cache block.
    index_t num_blocks_s = 1;
    /// Enable cache blocking by copying the current block row of A to a
    /// temporary buffer.
    bool enable_packing = true;
};

inline namespace serial {
template <Config Conf>
void downdate_blocked(MutableRealMatrixView L, MutableRealMatrixView A);
}

#if KOQKATOO_WITH_LIBFORK
namespace parallel {
template <Config Conf>
void downdate_blocked(MutableRealMatrixView L, MutableRealMatrixView A);
}
#endif

} // namespace koqkatoo::cholundate::householder
