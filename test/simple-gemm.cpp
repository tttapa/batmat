/*
 * This is an isolated GEMM micro-kernel implementation that results in incorrect codegen on Intel
 * icx 2025.3 at optimization levels higher than -O1. The specific issue seems to be the
 * step that stores the accumulator to memory: for some reason, the compiler seems to skip all rows
 * except the last one.
 */

#pragma clang fp contract(fast)

#include <batmat/simd.hpp>
#include <batmat/unroll.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <print>
#include <random>
#include <type_traits>
#include <utility>

#define UNROLL_FOR(...) BATMAT_FULLY_UNROLLED_FOR (__VA_ARGS__)

enum class StorageOrder { RowMajor, ColMajor };

template <class T, class Abi, StorageOrder Order>
struct uview {
    using value_type = T;
    value_type *data;
    int outer_stride;

    using simd                              = batmat::datapar::simd<std::remove_cv_t<T>, Abi>;
    static constexpr ptrdiff_t inner_stride = simd::size();

    [[gnu::always_inline]] value_type *get(int r, int c) const noexcept {
        ptrdiff_t i0 = Order == StorageOrder::RowMajor ? c : r;
        ptrdiff_t i1 = Order == StorageOrder::RowMajor ? r : c;
        return data + inner_stride * (i0 + i1 * static_cast<ptrdiff_t>(outer_stride));
    }
    [[gnu::always_inline]] simd load(int r, int c) const noexcept {
        return batmat::datapar::aligned_load<simd>(get(r, c));
    }
    [[gnu::always_inline]] void store(simd x, int r, int c) const noexcept
        requires(!std::is_const_v<T>)
    {
        batmat::datapar::aligned_store(x, get(r, c));
    }

    [[gnu::always_inline]] uview(value_type *data, int outer_stride) noexcept
        : data{data}, outer_stride{outer_stride} {}
};

template <int Size, class T, class Abi, StorageOrder Order>
struct cached_uview {
    using value_type = T;
    const std::array<value_type *, Size> data;

    using simd                              = batmat::datapar::simd<std::remove_cv_t<T>, Abi>;
    static constexpr ptrdiff_t inner_stride = simd::size();

    [[gnu::always_inline]] value_type *get(int r, int c) const noexcept {
        ptrdiff_t i0 = Order == StorageOrder::RowMajor ? c : r;
        int i1       = Order == StorageOrder::RowMajor ? r : c;
        assert(i1 < Size);
        return data[i1] + i0 * inner_stride;
    }
    [[gnu::always_inline]] simd load(int r, int c) const noexcept {
        return batmat::datapar::aligned_load<simd>(get(r, c));
    }
    [[gnu::always_inline]] void store(simd x, int r, int c) const noexcept
        requires(!std::is_const_v<T>)
    {
        batmat::datapar::aligned_store(x, get(r, c));
    }

    template <int... Is>
    [[gnu::always_inline]] cached_uview(const uview<T, Abi, Order> &o,
                                        std::integer_sequence<int, Is...>) noexcept
        : data{(o.data + Is * o.inner_stride * static_cast<ptrdiff_t>(o.outer_stride))...} {}
    [[gnu::always_inline]] cached_uview(const uview<T, Abi, Order> &o) noexcept
        : cached_uview{o, std::make_integer_sequence<int, Size>()} {}
};

template <int Rows, int Cols, class T, class Abi, StorageOrder O>
    requires(Rows > 0 && Cols > 0)
[[gnu::always_inline]] inline cached_uview<O == StorageOrder::ColMajor ? Cols : Rows, T, Abi, O>
with_cached_access(const uview<T, Abi, O> &o) noexcept {
    return {o};
}

template <int Rows, int Cols, class T, class Abi>
    requires(Rows == 0 && Cols > 0)
[[gnu::always_inline]] inline cached_uview<Cols, T, Abi, StorageOrder::ColMajor>
with_cached_access(const uview<T, Abi, StorageOrder::ColMajor> &o) noexcept {
    return {o};
}

template <int Rows, int Cols, class T, class Abi>
    requires(Rows == 0 && Cols > 0)
[[gnu::always_inline]] inline uview<T, Abi, StorageOrder::RowMajor>
with_cached_access(const uview<T, Abi, StorageOrder::RowMajor> &o) noexcept {
    return o;
}

template <int Rows, int Cols, class T, class Abi>
    requires(Rows > 0 && Cols == 0)
[[gnu::always_inline]] inline cached_uview<Rows, T, Abi, StorageOrder::RowMajor>
with_cached_access(const uview<T, Abi, StorageOrder::RowMajor> &o) noexcept {
    return {o};
}

template <int Rows, int Cols, class T, class Abi>
    requires(Rows > 0 && Cols == 0)
[[gnu::always_inline]] inline uview<T, Abi, StorageOrder::ColMajor>
with_cached_access(const uview<T, Abi, StorageOrder::ColMajor> &o) noexcept {
    return o;
}

/// Batched generalized matrix multiplication D⁽ᵀ⁾ = C⁽ᵀ⁾ + A⁽ᵀ⁾ B⁽ᵀ⁾. Single register block.
template <class T, class Abi, int RowsReg, int ColsReg, StorageOrder OA, StorageOrder OB,
          StorageOrder OC, StorageOrder OD>
[[gnu::hot, gnu::flatten, gnu::noinline]] void
batch_gemm_microkernel(const uview<const T, Abi, OA> A, const uview<const T, Abi, OB> B,
                       const std::optional<uview<const T, Abi, OC>> C, const uview<T, Abi, OD> D,
                       const int k) noexcept {
    using simd = batmat::datapar::simd<T, Abi>;
    [[assume(k > 0)]]; // Eliminate unnecessary branch for k == 0.

    // Load accumulator into registers
    simd C_reg[RowsReg][ColsReg];
    if (C) [[likely]] {
        UNROLL_FOR (int ii = 0; ii < RowsReg; ++ii)
            UNROLL_FOR (int jj = 0; jj < ColsReg; ++jj)
                C_reg[ii][jj] = C->load(ii, jj);
    } else {
        UNROLL_FOR (int ii = 0; ii < RowsReg; ++ii)
            UNROLL_FOR (int jj = 0; jj < ColsReg; ++jj)
                C_reg[ii][jj] = simd{0};
    }

    // Rectangular matrix multiplication kernel
    const auto A_cached = with_cached_access<RowsReg, 0>(A);
    const auto B_cached = with_cached_access<0, ColsReg>(B);
    for (int l = 0; l < k; ++l) {
        UNROLL_FOR (int ii = 0; ii < RowsReg; ++ii) {
            simd Ail = A_cached.load(ii, l);
            UNROLL_FOR (int jj = 0; jj < ColsReg; ++jj) {
                simd &Cij = C_reg[ii][jj];
                simd Blj  = B_cached.load(l, jj);
                Cij += Ail * Blj;
            }
        }
    }

    // Store accumulator to memory again
#if BROKEN
    const auto D_cached = with_cached_access<RowsReg, ColsReg>(D);
#else
    const auto D_cached = D;
#endif
    UNROLL_FOR (int ii = 0; ii < RowsReg; ++ii)
        UNROLL_FOR (int jj = 0; jj < ColsReg; ++jj)
            D_cached.store(C_reg[ii][jj], ii, jj);

    // Sanity checks
    std::println("D stride = {:#x}", D.outer_stride * D.inner_stride * sizeof(T));
#if BROKEN
    std::print("D_cached = {{ ");
    for (auto *ptr : D_cached.data)
        std::print("{} ", (void *)ptr);
    std::println("}}");
#endif
    for (int i = 0; i < RowsReg; ++i)
        for (int j = 0; j < ColsReg; ++j)
            std::println("D_cached[{},{}] @ {},\tD[{},{}] @ {} \t{}", i, j,
                         (void *)D_cached.get(i, j), i, j, (void *)D.get(i, j),
                         D_cached.get(i, j) == D.get(i, j) ? "ok" : "MISMATCH !!!");
    std::println();
}

int main() {
    static constexpr int v = 4, m = 3, n = 3, k = 7;
    using enum StorageOrder;
    using wp  = double;
    using abi = batmat::datapar::deduced_abi<wp, v>;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<wp> dist{-1, 1};
    alignas(v * sizeof(wp)) wp A[m * k * v];
    alignas(v * sizeof(wp)) wp B[k * n * v];
    alignas(v * sizeof(wp)) wp D[m * n * v];
    std::ranges::generate(A, [&] { return dist(rng); });
    std::ranges::generate(B, [&] { return dist(rng); });
    std::ranges::fill(D, std::numeric_limits<wp>::quiet_NaN());

    batch_gemm_microkernel<wp, abi, m, n, ColMajor, ColMajor, RowMajor, RowMajor>(
        {A, m}, {B, k}, std::nullopt, {D, n}, k);

    const auto idx_cm = [=](int i, int j, int l, int stride) { return (j * stride + i) * v + l; };
    const auto idx_rm = [=](int i, int j, int l, int stride) { return (i * stride + j) * v + l; };

    for (int l = 0; l < v; ++l, std::println())
        for (int i = 0; i < m; ++i, std::println())
            for (int j = 0; j < n; ++j)
                std::print("{:+25.17e} ", D[idx_rm(i, j, l, n)]);

    for (int l = 0; l < v; ++l)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                wp Dij = 0;
                for (int p = 0; p < k; ++p)
                    Dij += A[idx_cm(i, p, l, m)] * B[idx_cm(p, j, l, k)];
                if (!(std::abs(D[idx_rm(i, j, l, n)] - Dij) < 1e-12)) {
                    std::println("Err: D[{},{};{}] = {:+.17e},\treference Dij = {:+.17e}", i, j, l,
                                 D[idx_rm(i, j, l, n)], Dij);
                    return 1;
                }
            }
}
