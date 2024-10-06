#pragma once

#include <koqkatoo/cholundate/updown.hpp>
#include <koqkatoo/matrix-view.hpp>

namespace koqkatoo::cholundate::householder {

#ifdef __AVX512F__
// AVX512 has 32 vector registers, TODO:
static constexpr index_t DefaultSizeR = 8;
static constexpr index_t DefaultSizeS = 24;
#elif defined(__ARM_NEON)
// NEON has 32 vector registers, TODO:
static constexpr index_t DefaultSizeR = 8;
static constexpr index_t DefaultSizeS = 24;
#else
// AVX2 has 16 vector registers, TODO:
static constexpr index_t DefaultSizeR = 4;
static constexpr index_t DefaultSizeS = 12;
#endif

struct Config {
    /// Block size of the block column of L to process in the micro-kernels.
    index_t block_size_r = DefaultSizeR;
    /// Block size of the block row of L to process in the micro-kernels.
    index_t block_size_s = DefaultSizeS;
    /// Number of block columns per cache block.
    index_t num_blocks_r = 1;
    /// Number of block rows per cache block.
    index_t num_blocks_s = 1;
    /// Column prefetch distance for the matrix A.
    index_t prefetch_dist_col_a = 4;
    /// Enable cache blocking by copying the current block row of A to a
    /// temporary buffer.
    bool enable_packing = true;
};

inline namespace serial {
template <Config Conf, class UpDown>
void updowndate_blocked(MutableRealMatrixView L, MutableRealMatrixView A,
                        UpDown signs);
} // namespace serial

namespace naive {
template <Config Conf, class UpDown>
void updowndate_blocked(MutableRealMatrixView L, MutableRealMatrixView A,
                        UpDown signs);
} // namespace naive

#if KOQKATOO_WITH_LIBFORK
namespace parallel {
template <Config Conf, class UpDown>
void updowndate_blocked(MutableRealMatrixView L, MutableRealMatrixView A,
                        UpDown signs);
}
#endif

namespace parallel_static {
template <Config Conf, class UpDown>
void updowndate_blocked(MutableRealMatrixView L, MutableRealMatrixView A,
                        UpDown signs);
}

} // namespace koqkatoo::cholundate::householder
