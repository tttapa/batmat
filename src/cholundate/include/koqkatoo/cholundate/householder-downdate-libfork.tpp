#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/lut.hpp>
#include <atomic>
#include <cstdint>
#include <memory_resource>
#include <new>
#include <print>
#include <vector>

#include <guanaqo/print.hpp>
#include <libfork/core.hpp>
#include <libfork/schedule.hpp>

#include <koqkatoo/cholundate/householder-downdate-common.tpp>
#include <koqkatoo/cholundate/householder-downdate.hpp>

namespace koqkatoo::cholundate::householder::parallel {

namespace detail {

template <index_t Bs>
struct PackedBlockRowMatrix {
    const index_t n, m;
    const index_t ceil_n = (n + Bs - 1) / Bs * Bs;
    std::pmr::vector<real_t> storage;

    PackedBlockRowMatrix(index_t n, index_t m, std::pmr::memory_resource *mr)
        : n{n}, m{m}, storage(static_cast<size_t>(n + Bs - 1) / Bs * Bs *
                                  static_cast<size_t>(m),
                              mr) {}

    MutableRealMatrixView operator[](index_t block, index_t num_rows = Bs) {
        index_t offset = block * Bs * m;
        return {{
            .data         = &storage[offset],
            .rows         = num_rows,
            .cols         = m,
            .outer_stride = Bs,
        }};
    }
};

using uConfig = micro_kernels::householder::Config;

template <uConfig uConf>
constinit static auto full_microkernel_lut = make_1d_lut<uConf.block_size_r>(
    []<index_t N>(index_constant<N>) { return downdate_full<N + 1>; });

template <Config Conf>
struct Context {
    static constexpr index_t MaxConcurrentCols = 64;
    static constexpr index_t NumColBarriers    = 16;
    static constexpr index_t R                 = Conf.block_size_r;
    static constexpr index_t S                 = std::max(R, Conf.block_size_s);
    static constexpr index_t ColsPerBlock      = S / R;
    static constexpr uConfig uConf{.block_size_r = Conf.block_size_r,
                                   .block_size_s = Conf.block_size_s};
    static_assert(S % R == 0);

    MutableRealMatrixView L, A;
    const index_t n  = L.rows;
    const index_t ns = (n + S - 1) / S;
    const index_t nr = (n + R - 1) / R;

#if __clang__
    static constexpr size_t false_sharing_size = 64;
#else
    static constexpr size_t false_sharing_size =
        std::hardware_destructive_interference_size;
#endif

    struct alignas(false_sharing_size) ColBarrier {
        std::atomic<index_t> counter{};
        index_t target{};
    };
    using ColBarrierArray = std::array<ColBarrier, NumColBarriers + 1>;
    std::array<ColBarrierArray, MaxConcurrentCols> col_barriers;

    bool column_arrive(index_t br, index_t bc) {
        static constexpr auto mo = std::memory_order_acq_rel;
        index_t br_diag          = bc / ColsPerBlock;
        index_t i                = br - br_diag;
        auto &bar   = col_barriers[bc % MaxConcurrentCols][i % NumColBarriers];
        auto &final = col_barriers[bc % MaxConcurrentCols][NumColBarriers];
        auto value  = bar.counter.fetch_add(1, mo) + 1;
        if (value == bar.target) {
            auto final_value = final.counter.fetch_add(value, mo) + value;
            if (final_value == final.target)
                return true;
        }
        return false;
    }

    void reset_column_barrier(index_t bc) {
        index_t br_diag = bc / ColsPerBlock;
        index_t total   = ns - br_diag;
        auto &barriers  = col_barriers[bc % MaxConcurrentCols];
        for (index_t i = 0; i < NumColBarriers; ++i) {
            barriers[i].counter.store(0, std::memory_order_relaxed);
            barriers[i].target =
                total / NumColBarriers +
                static_cast<index_t>(total % NumColBarriers > i);
        }
        auto &final_barrier = barriers[NumColBarriers];
        final_barrier.counter.store(0, std::memory_order_relaxed);
        final_barrier.target = total;
    }

    alignas(std::max_align_t) std::byte buffer[64 * 1024];
    std::pmr::monotonic_buffer_resource mbr{buffer, sizeof(buffer)};
    using block_counter_t       = std::atomic<uint8_t>;
    using col_counter_t         = std::atomic<index_t>;
    const index_t column_stride = std::min(MaxConcurrentCols, nr);
    std::pmr::vector<block_counter_t> blocks_ready{
        static_cast<size_t>(ns) * column_stride,
        &mbr,
    };
    PackedBlockRowMatrix<S> packed_A{Conf.enable_packing ? n : 0, A.cols, &mbr};
    [[nodiscard]] block_counter_t &block_ready(index_t br, index_t bc) {
        assert(br < ns);
        assert(bc < nr);
        [[maybe_unused]] index_t i = br, j = bc / ColsPerBlock,
                                 k = bc % ColsPerBlock;
        assert(j <= i);
        return blocks_ready[(bc % MaxConcurrentCols) + br * column_stride];
    }
    [[nodiscard]] bool mark_block_ready(index_t br, index_t bc) {
        auto r = block_ready(br, bc).fetch_add(1, std::memory_order_acq_rel);
        assert(r < 2);
        return r == 1;
    }
    void clear_blocks_ready(index_t bc) {
        assert(bc < nr);
        index_t br_diag = bc / ColsPerBlock;
        for (index_t br = br_diag; br < ns; ++br)
            blocks_ready[(bc % MaxConcurrentCols) + br * column_stride].store(
                0, std::memory_order_relaxed);
    }

    // Workspace storage for T (upper triangular Householder representation)
    micro_kernels::householder::matrix_W_storage<R> Ws[MaxConcurrentCols]{};

    bool compute_diag(index_t bc) {
        /*
            ┌────── S ──────┐ = R × ColsPerBlock
                ┌ R ┐

            i0  i1  i2      i3
            │   │   │       │
            ┌───┬───┬───┬───┐  ─ i0      ─┐
            │   │   │   │   │             │
            ├   ┼───┼   ┼   ┤  ─ i1   ─┐  │  = R × br
            │   │ D │   │   │          r  │
        L = ├   ┼   ┼───┤   ┤  ─ i2   ─┤  S
            │   │ T │   │   │          │  │
            ├   ┼   ┼   ┼───┤          s  │
            │   │ T │   │   │          │  │
            └───┴───┴───┴───┘  ─ i3   ─┘ ─┘
        */
        const index_t i0 = bc / ColsPerBlock * S;
        const index_t i1 = bc * R;
        const index_t i2 = std::min<index_t>(i1 + R, n);
        const index_t i3 = std::min<index_t>(i0 + S, n);
        const index_t r = i2 - i1, s = i3 - i2;
        const index_t work_id = bc % MaxConcurrentCols;
        static_assert(R <= Conf.block_size_r);
        KOQKATOO_ASSUME(r > 0);
        KOQKATOO_ASSUME(r <= Conf.block_size_r);
        KOQKATOO_ASSUME(s >= 0);
        auto packed_Ad = [&] {
            if constexpr (Conf.enable_packing) {
                auto Adt_p = packed_A[i0 / S].top_rows(i3 - i0);
                if (bc == 0)
                    Adt_p = A.middle_rows(i0, i3 - i0);
                return Adt_p.middle_rows(i1 - i0, r);
            }
            return A.middle_rows(i1, r);
        }();
        auto Ld = L.block(i1, i1, r, r);
        if (r == Conf.block_size_r) [[likely]] // Most diagonal blocks
            downdate_diag<Conf.block_size_r>(A.cols, Ws[work_id], Ld,
                                             packed_Ad);
        else // Last diagonal block
            full_microkernel_lut<uConf>[r - 1](A.cols, Ld, packed_Ad);
        return s > 0;
    }

    void compute_diag_tail(index_t bc) {
        const index_t i0 = bc / ColsPerBlock * S;
        const index_t i1 = bc * R;
        const index_t i2 = std::min<index_t>(i1 + R, n);
        const index_t i3 = std::min<index_t>(i0 + S, n);
        const index_t r = i2 - i1, s = i3 - i2;
        const index_t work_id = bc % MaxConcurrentCols;
        static_assert(R <= Conf.block_size_r);
        KOQKATOO_ASSUME(r > 0);
        KOQKATOO_ASSUME(r <= Conf.block_size_r);
        KOQKATOO_ASSUME(s >= 0);
        auto [packed_Ad, packed_At] = [&] {
            if constexpr (Conf.enable_packing) {
                auto Adt_p = packed_A[i0 / S].top_rows(i3 - i0);
                return std::tuple{Adt_p.middle_rows(i1 - i0, r),
                                  Adt_p.middle_rows(i2 - i0, s)};
            }
            return std::tuple{A.middle_rows(i1, r), A.middle_rows(i2, s)};
        }();
        if (s > 0) { // TODO: could be handled without forking
            auto Ls = L.block(i2, i1, s, r);
            tile_tail<uConf>(s, A.cols, Ws[work_id], Ls, packed_Ad, packed_At);
        }
    }

    void compute_tail(index_t br, index_t bc) {
        /*
            ┌────── S ──────┐ = R × ColsPerBlock
                ┌ r ┐

            i0  i1  i2
            │   │   │
            ┌───┬───┬───┬───┐  ─ i0      ─┐
            │   │   │   │   │             │
            ├   ┼───┼   ┼   ┤  ─ i1   ─┐  │  = R × br
            │   │ D │   │   │          r  │
        L = ├   ┼   ┼───┤   ┤  ─ i2   ─┘  S
            │   │   │   │   │             │
            ├   ┼   ┼   ┼───┤             │
            │   │   │   │   │             │
            ├───┼───┼───┼───┤            ─┘
             ··· ··· ··· ···
            ├───┼───┼───┼───┤  ─ i4   ─┐
            │   │   │   │   │          │
            ├   ┼   ┼   ┼   ┤          │
            │   │   │   │   │          │
            ├   ┼   ┼   ┼   ┤          s
            │   │   │   │   │          │
            ├   ┼   ┼   ┼   ┤          │
            │   │   │   │   │          │
            └───┴───┴───┴───┘  ─ i5   ─┘
        */
        const index_t i0 = bc / ColsPerBlock * S;
        const index_t i1 = bc * R;
        const index_t i2 = std::min<index_t>(i1 + R, n);
        const index_t i4 = br * S;
        const index_t i5 = std::min<index_t>(i4 + S, n);
        const index_t r = i2 - i1, s = i5 - i4;
        const index_t work_id = bc % MaxConcurrentCols;
        KOQKATOO_ASSUME(r == Conf.block_size_r);
        KOQKATOO_ASSUME(s > 0);
        auto [packed_Ad, packed_At] = [&] {
            if constexpr (Conf.enable_packing) {
                auto Ad_p = packed_A[i0 / S].middle_rows(i1 - i0, r);
                auto At_p = packed_A[i4 / S].top_rows(s);
                if (bc == 0)
                    At_p = A.middle_rows(i4, s);
                return std::tuple{Ad_p, At_p};
            }
            return std::tuple{A.middle_rows(i1, r), A.middle_rows(i4, s)};
        }();
        auto Ls = L.block(i4, i1, s, r);
        if (s == Conf.block_size_s) [[likely]] // Most block rows
            downdate_tail<uConf>(A.cols, Ws[work_id], Ls, packed_Ad, packed_At);
        else // Last block row
            tile_tail<uConf>(s, A.cols, Ws[work_id], Ls, packed_Ad, packed_At);
    }
};

} // namespace detail

template <Config Conf>
void downdate_blocked(MutableRealMatrixView L, MutableRealMatrixView A) {
    assert(L.rows == L.cols);
    assert(L.rows == A.rows);

    using Context = detail::Context<Conf>;
    Context context{L, A};

    static constexpr auto process_block =
        [](auto process_block, Context &ctx, index_t br, index_t bc,
           bool tail = false) -> lf::task<void> {
        const auto br_diag         = bc / ctx.ColsPerBlock;
        const bool is_diag         = br == br_diag;
        const bool is_diag_no_tail = is_diag && !tail;
        bool have_diag_tail_block  = false;
        if (is_diag_no_tail)
            // Update the diagonal block
            have_diag_tail_block = ctx.compute_diag(bc);
        else if (is_diag)
            // Update the small piece of tail in the diagonal block
            ctx.compute_diag_tail(bc);
        else
            // Update the sub-diagonal block
            ctx.compute_tail(br, bc);

        // Last column?
        if (bc + ctx.R >= ctx.n)
            co_return;

        // More columns than workspaces? Need to synchronize columns
        if (ctx.nr > ctx.MaxConcurrentCols) {
            bool col_done = !is_diag_no_tail || !have_diag_tail_block;
            auto bc_next  = bc + ctx.MaxConcurrentCols - 1;
            // Kick off another column
            if (col_done && bc_next < ctx.nr && ctx.column_arrive(br, bc)) {
                auto br_next = bc_next / ctx.ColsPerBlock;
                ctx.reset_column_barrier(bc_next);
                ctx.clear_blocks_ready(bc);
                if (ctx.mark_block_ready(br_next, bc_next))
                    co_await lf::fork[process_block](ctx, br_next, bc_next);
            }
        }

        // Unlock the next blocks
        if (is_diag_no_tail) {
            // Kick off all sub-diagonal blocks
            if (have_diag_tail_block)
                co_await lf::fork[process_block](ctx, br_diag, bc, true);
            for (index_t br = br_diag + 1; br < ctx.ns; ++br)
                if (ctx.mark_block_ready(br, bc))
                    co_await lf::fork[process_block](ctx, br, bc);
        } else if (is_diag) {
            // Kick off the diagonal block to our right (if it exists)
            const index_t br_next_diag = (bc + 1) / ctx.ColsPerBlock;
            bool has_next_diag_block   = br_next_diag == br && bc + 1 < ctx.nr;
            if (has_next_diag_block && ctx.mark_block_ready(br, bc + 1))
                co_await lf::fork[process_block](ctx, br, bc + 1, false);
        } else {
            // Kick off the block to our right
            if (bc + 1 < ctx.nr && ctx.mark_block_ready(br, bc + 1))
                co_await lf::call[process_block](ctx, br, bc + 1);
        }
        // Note: this only waits for the blocks launched by us!
        co_await lf::join;
    };

    // Boundary conditions: indicate which blocks can be started
    for (index_t br = 0; br < context.ns; ++br)
        context.block_ready(br, 0).store(1, std::memory_order_relaxed);
    index_t n_cols_init = std::min(context.MaxConcurrentCols - 1, context.nr);
    if (context.nr == context.MaxConcurrentCols)
        n_cols_init = context.MaxConcurrentCols;
    for (index_t bc = 0; bc < n_cols_init; ++bc) {
        context.reset_column_barrier(bc);
        context.block_ready(bc / context.ColsPerBlock, bc)
            .fetch_add(1, std::memory_order_relaxed);
    }
    static lf::lazy_pool pool{8}; // TODO: make configurable/argument
    lf::sync_wait(pool, process_block, context, 0, 0);
}

} // namespace koqkatoo::cholundate::householder::parallel
