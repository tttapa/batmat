#pragma once

#include <koqkatoo/assume.hpp>
#include <koqkatoo/lut.hpp>
#include <barrier>
#include <memory_resource>
#include <thread>
#include <vector>

#include <koqkatoo/cholundate/householder-downdate-common.tpp>
#include <koqkatoo/cholundate/householder-downdate.hpp>

namespace koqkatoo::cholundate::householder::parallel_static {

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
    MutableRealMatrixView L, A;
    const index_t p                       = 8;
    const index_t max_concurrent_columns  = 2 * p; // TODO: check OBOE!
    static constexpr index_t R            = Conf.block_size_r;
    static constexpr index_t S            = std::max(R, Conf.block_size_s);
    static constexpr index_t ColsPerBlock = S / R;
    static constexpr uConfig uConf{.block_size_r = Conf.block_size_r,
                                   .block_size_s = Conf.block_size_s};
    static_assert(S % R == 0);

    const index_t n  = L.rows;
    const index_t ns = (n + S - 1) / S;
    const index_t nr = (n + R - 1) / R;

    alignas(std::max_align_t) std::byte buffer[64 * 1024];
    std::pmr::monotonic_buffer_resource mbr{buffer, sizeof(buffer)};
    PackedBlockRowMatrix<S> packed_A{Conf.enable_packing ? n : 0, A.cols, &mbr};

    // Workspace storage for T (upper triangular Householder representation)
    std::vector<micro_kernels::householder::matrix_W_storage<R>> Ws =
        std::vector<micro_kernels::householder::matrix_W_storage<R>>(
            max_concurrent_columns);

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
        assert(bc < nr);
        const index_t i0 = bc / ColsPerBlock * S;
        const index_t i1 = bc * R;
        const index_t i2 = std::min<index_t>(i1 + R, n);
        const index_t i3 = std::min<index_t>(i0 + S, n);
        const index_t r = i2 - i1, s = i3 - i2;
        const index_t work_id = bc % max_concurrent_columns;
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
        assert(bc < nr);
        const index_t i0 = bc / ColsPerBlock * S;
        const index_t i1 = bc * R;
        const index_t i2 = std::min<index_t>(i1 + R, n);
        const index_t i3 = std::min<index_t>(i0 + S, n);
        const index_t r = i2 - i1, s = i3 - i2;
        const index_t work_id = bc % max_concurrent_columns;
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
        assert(bc < nr);
        assert(br < ns);
        const index_t i0 = bc / ColsPerBlock * S;
        const index_t i1 = bc * R;
        const index_t i2 = std::min<index_t>(i1 + R, n);
        const index_t i4 = br * S;
        const index_t i5 = std::min<index_t>(i4 + S, n);
        const index_t r = i2 - i1, s = i5 - i4;
        const index_t work_id = bc % max_concurrent_columns;
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

inline auto create_schedule(const index_t n, const index_t p,
                            const bool min_mem = true) {

    index_t c_diag       = 1;
    index_t c            = 0;
    index_t r            = 1;
    const index_t nblock = n - p;
    index_t t            = 1;
    bool next_col        = false;
    std::vector<index_t> rem_rows(p);
    std::vector<std::vector<std::pair<index_t, index_t>>> work_threads(p);
    auto allocate = [&](index_t r, index_t c, index_t tid) {
        work_threads[tid].emplace_back(r, c);
    };
    while (c < nblock) {
        index_t threads_allocated = 0;
        // Advance the spares row to limit the required workspace.
        if (min_mem && rem_rows.back() + 2 * p < c_diag) { // TODO: ???
            for (index_t j = 0; j < p; ++j) {
                auto &rem_j = rem_rows[j];
                if (rem_j < nblock + j)
                    allocate(nblock + j, rem_j++, threads_allocated++);
            }
            for (index_t tid = threads_allocated; tid < p; ++tid)
                allocate(0, 0, tid);
            ++t;
            continue;
        }
        // To start a new column, we first need to compute the diagonal block.
        bool inc_c_diag = next_col;
        if (next_col) {
            allocate(c_diag, c_diag, threads_allocated++);
            next_col = false;
        }
        // If the diagonal block for this column is done, then we can start
        // computing the sub-diagonal blocks.
        while (c < c_diag) {
            while (r < nblock && threads_allocated < p) {
                // If this is the first sub-diagonal row in the column, we
                // should compute the block to the right (the diagonal block of
                // the next column) in the next iteration.
                next_col |= r == c + 1;
                allocate(r++, c, threads_allocated++);
            }
            // If all threads are busy without finishing the column, go to the
            // next iteration.
            if (r < nblock)
                break;
            // If we finished the column, try to schedule some blocks in the
            // next column already.
            r = ++c + 1;
        }
        // If we're in a column that doesn't yet have the diagonal block ready,
        // schedule some of the spare blocks at the bottom of the matrix.
        if (c == c_diag) {
            // Select the spare rows in such a way that a trapezoidal shape is
            // created, which will reduce load imbalance in the bottom right
            // block. This is checked by the trapezoidal_spare condition.
            // In the following graphic, (×) indicates an element that's already
            // there (represented by the rem_rows array), and (+) indicates an
            // element that preserves the trapezoidal structure of the spares.
            // [ × × × × × ]
            // [ × × × +   ]
            // [ × × ×     ]
            // [ × × +     ]
            // We also need to make sure that the diagonal element of the column
            // is done before scheduling a block (even in the spares, for when
            // n is small), and we don't want to end up in the upper triangular
            // part of the matrix.
            auto try_allocate_spare = [&](index_t j) {
                auto &rem_j            = rem_rows[j];
                bool trapezoidal_spare = rem_j < rem_rows.back() + p - j;
                bool lower_triangle    = rem_j < nblock + j;
                bool diag_ready        = rem_j < c;
                if (trapezoidal_spare && lower_triangle && diag_ready) {
                    allocate(nblock + j, rem_j++, threads_allocated++);
                    return true;
                }
                return false;
            };
            index_t first_j = -1;
            for (index_t j = 0; j < p && threads_allocated < p; ++j)
                if (try_allocate_spare(j) && first_j < 0)
                    first_j = j;
            // Now that we've possibly incremented the column index of the last
            // row, there may be room in the first spare rows (which were
            // previously unavailable because the trapezoidal_spare condition
            // was violated). So fill up those available blocks, but don't
            // allocate multiple blocks per row (that's why we only go up to
            // row first_j).
            for (index_t j = 0; j < first_j && threads_allocated < p; ++j)
                try_allocate_spare(j);
        }
        if (inc_c_diag)
            ++c_diag;
        for (index_t tid = threads_allocated; tid < p; ++tid)
            allocate(0, 0, tid);
        ++t;
    }
    // Element (0, 0) overlaps with spares row if n <= p.
    if (t == 2)
        ++rem_rows[0];
    // Finish the spares row.
    while (rem_rows.back() < n) {
        index_t threads_allocated = 0;
        index_t diag              = n;
        for (index_t j = 0; j < p; ++j) {
            auto &rem_j        = rem_rows[j];
            bool is_diag_block = rem_j == nblock + j;
            if (is_diag_block) {
                diag = rem_j;
                allocate(nblock + j, rem_j++, threads_allocated++);
            } else if (rem_j < nblock + j && rem_j < diag) {
                allocate(nblock + j, rem_j++, threads_allocated++);
            }
        }
        for (index_t tid = threads_allocated; tid < p; ++tid)
            allocate(0, 0, tid);
        ++t;
    }
    return work_threads;
}

} // namespace detail

template <Config Conf>
void downdate_blocked(MutableRealMatrixView L, MutableRealMatrixView A) {
    assert(L.rows == L.cols);
    assert(L.rows == A.rows);

    using Context = detail::Context<Conf>;
    Context context{.L = L, .A = A, .p = 8};
    std::barrier barrier(context.p);

    static constexpr auto process_block = [](Context &ctx, index_t r,
                                             index_t c) {
        if (r == c) {
            for (index_t i = 0; i < ctx.ColsPerBlock; ++i) {
                index_t bc = c * ctx.ColsPerBlock + i;
                if (bc >= ctx.nr)
                    break;
                // Update the diagonal block
                bool have_diag_tail_block = ctx.compute_diag(bc);
                // Update the small piece of tail in the diagonal block
                if (have_diag_tail_block)
                    ctx.compute_diag_tail(bc);
            }
        } else {
            for (index_t i = 0; i < ctx.ColsPerBlock; ++i) {
                index_t br = r, bc = c * ctx.ColsPerBlock + i;
                if (bc >= ctx.nr)
                    break;
                // Update the sub-diagonal block
                ctx.compute_tail(br, bc);
            }
        }
    };

    process_block(context, 0, 0);
    // TODO: static
    static auto schedule = detail::create_schedule(context.ns, context.p, true);
    std::vector<std::jthread> threads;
    threads.reserve(context.p);

    auto run_thread = [&](index_t tid) {
        for (auto [r, c] : schedule[tid]) {
            if (r != 0 || c != 0)
                process_block(context, r, c);
            barrier.arrive_and_wait();
        }
    };
    for (index_t tid = 0; tid < context.p; ++tid)
        threads.emplace_back(run_thread, tid);
}

} // namespace koqkatoo::cholundate::householder::parallel_static
