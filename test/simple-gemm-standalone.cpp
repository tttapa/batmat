///usr/bin/env icpx -march=x86-64-v3 -fp-model=precise -ffp-contract=on -O3 -DNDEBUG -std=c++23 -ggdb -Wall -Wextra "$0" -o simple-gemm-standalone -DBROKEN=1 && exec ./simple-gemm-standalone "$@" || exit $?

/*
This is an isolated GEMM micro-kernel implementation that results in incorrect codegen on Intel
icpx 2025.3 at optimization levels higher than -O1. The specific issue seems to be the step that
stores the accumulator to memory: for some reason, the compiler seems to skip all rows except the
last one. Example output:

D stride = 0x60
D_cached = { 0x7ffcde8d5480 0x7ffcde8d54e0 0x7ffcde8d5540 }
D_cached[0,0] @ 0x7ffcde8d5480, D[0,0] @ 0x7ffcde8d5480         ok
D_cached[0,1] @ 0x7ffcde8d54a0, D[0,1] @ 0x7ffcde8d54a0         ok
D_cached[0,2] @ 0x7ffcde8d54c0, D[0,2] @ 0x7ffcde8d54c0         ok
D_cached[1,0] @ 0x7ffcde8d54e0, D[1,0] @ 0x7ffcde8d54e0         ok
D_cached[1,1] @ 0x7ffcde8d5500, D[1,1] @ 0x7ffcde8d5500         ok
D_cached[1,2] @ 0x7ffcde8d5520, D[1,2] @ 0x7ffcde8d5520         ok
D_cached[2,0] @ 0x7ffcde8d5540, D[2,0] @ 0x7ffcde8d5540         ok
D_cached[2,1] @ 0x7ffcde8d5560, D[2,1] @ 0x7ffcde8d5560         ok
D_cached[2,2] @ 0x7ffcde8d5580, D[2,2] @ 0x7ffcde8d5580         ok

                     +nan                      +nan                      +nan
                     +nan                      +nan                      +nan
 +9.66700365005358775e-02  +2.74452518972885362e-01  +9.03725340181264769e-02

                     +nan                      +nan                      +nan
                     +nan                      +nan                      +nan
 -1.85409902038319280e-01  +1.24307454297255648e+00  +3.83556144420183598e-01

                     +nan                      +nan                      +nan
                     +nan                      +nan                      +nan
 -2.75242912113221105e-01  +4.35770299121508886e-01  -3.18958466705292154e-01

                     +nan                      +nan                      +nan
                     +nan                      +nan                      +nan
 -7.10312539049554048e-01  -1.37506873804276308e+00  -3.03838922048146609e-01

Err: D[0,0;0] = +nan,   reference Dij = -2.66755322463899158e-01

Compiling with -O1 works around the issue. Compiling with -fsanitize=address,undefined also works
without any problems. GCC and Clang both give the correct result at all optimization levels, only
icpx does not.
*/

#pragma clang fp contract(fast)

#include <immintrin.h>
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

#ifdef __clang__
#define FULLY_UNROLL_LOOP _Pragma("clang loop unroll(full)")
#define UNROLL_FOR(...) FULLY_UNROLL_LOOP for (__VA_ARGS__)
#else
#define FULLY_UNROLL_LOOP _Pragma("GCC unroll 99")
#define UNROLL_FOR(...) FULLY_UNROLL_LOOP for (__VA_ARGS__)
#endif

// Minimal std::datapar-like SIMD wrapper for double with AVX2
namespace datapar {

template <class T, int Size>
struct deduced_abi : std::integral_constant<int, Size> {};

template <class T, class Abi>
struct simd;

template <>
struct simd<double, deduced_abi<double, 4>> {
    using value_type = double;
    using abi_type   = deduced_abi<double, 4>;
    static constexpr ptrdiff_t size() noexcept { return abi_type::value; }
    __m256d data;
    [[gnu::always_inline]] simd() = default;
    [[gnu::always_inline]] explicit simd(__m256d v) : data{v} {}
    [[gnu::always_inline]] explicit simd(double v) : simd{_mm256_set1_pd(v)} {}
    [[gnu::always_inline]] simd &operator+=(const simd &o) noexcept {
        data += o.data;
        return *this;
    }
    [[gnu::always_inline, nodiscard]] friend simd operator*(simd a, const simd &b) noexcept {
        return simd{a.data * b.data};
    }
    [[gnu::always_inline, nodiscard]] static simd aligned_load(const double *p) noexcept {
        return simd{_mm256_load_pd(p)};
    }
    [[gnu::always_inline]] void aligned_store(double *p) const noexcept {
        _mm256_store_pd(p, data);
    }
};

template <class Simd>
[[gnu::always_inline, nodiscard]] inline Simd
aligned_load(const typename Simd::value_type *p) noexcept {
    return Simd::aligned_load(p);
}
template <class Simd>
[[gnu::always_inline]] inline void aligned_store(const Simd &v,
                                                 typename Simd::value_type *p) noexcept {
    v.aligned_store(p);
}

} // namespace datapar

enum class StorageOrder { RowMajor, ColMajor };
using enum StorageOrder;

template <class T, class Abi, StorageOrder Order>
struct uview {
    using value_type = T;
    value_type *data;
    int outer_stride;

    using simd                              = datapar::simd<std::remove_cv_t<T>, Abi>;
    static constexpr ptrdiff_t inner_stride = simd::size();

    [[gnu::always_inline, nodiscard]] value_type *get(int r, int c) const noexcept {
        ptrdiff_t i0 = Order == RowMajor ? c : r;
        ptrdiff_t i1 = Order == RowMajor ? r : c;
        return data + inner_stride * (i0 + i1 * static_cast<ptrdiff_t>(outer_stride));
    }
    [[gnu::always_inline, nodiscard]] simd load(int r, int c) const noexcept {
        return datapar::aligned_load<simd>(get(r, c));
    }
    [[gnu::always_inline]] void store(simd x, int r, int c) const noexcept
        requires(!std::is_const_v<T>)
    {
        datapar::aligned_store(x, get(r, c));
    }

    [[gnu::always_inline]] uview(value_type *data, int outer_stride) noexcept
        : data{data}, outer_stride{outer_stride} {}
};

template <int Size, class T, class Abi, StorageOrder Order>
struct cached_uview {
    using value_type = T;
    const std::array<value_type *, Size> data;

    using simd                              = datapar::simd<std::remove_cv_t<T>, Abi>;
    static constexpr ptrdiff_t inner_stride = simd::size();

    [[gnu::always_inline, nodiscard]] value_type *get(int r, int c) const noexcept {
        ptrdiff_t i0 = Order == RowMajor ? c : r;
        int i1       = Order == RowMajor ? r : c;
        assert(i1 < Size);
        return data[i1] + i0 * inner_stride;
    }
    [[gnu::always_inline, nodiscard]] simd load(int r, int c) const noexcept {
        return datapar::aligned_load<simd>(get(r, c));
    }
    [[gnu::always_inline]] void store(simd x, int r, int c) const noexcept
        requires(!std::is_const_v<T>)
    {
        datapar::aligned_store(x, get(r, c));
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
[[gnu::always_inline, nodiscard]] inline cached_uview<O == ColMajor ? Cols : Rows, T, Abi, O>
with_cached_access(const uview<T, Abi, O> &o) noexcept {
    return {o};
}

template <int Rows, int Cols, class T, class Abi>
    requires(Rows == 0 && Cols > 0)
[[gnu::always_inline, nodiscard]] inline cached_uview<Cols, T, Abi, ColMajor>
with_cached_access(const uview<T, Abi, ColMajor> &o) noexcept {
    return {o};
}

template <int Rows, int Cols, class T, class Abi>
    requires(Rows == 0 && Cols > 0)
[[gnu::always_inline, nodiscard]] inline uview<T, Abi, RowMajor>
with_cached_access(const uview<T, Abi, RowMajor> &o) noexcept {
    return o;
}

template <int Rows, int Cols, class T, class Abi>
    requires(Rows > 0 && Cols == 0)
[[gnu::always_inline, nodiscard]] inline cached_uview<Rows, T, Abi, RowMajor>
with_cached_access(const uview<T, Abi, RowMajor> &o) noexcept {
    return {o};
}

template <int Rows, int Cols, class T, class Abi>
    requires(Rows > 0 && Cols == 0)
[[gnu::always_inline, nodiscard]] inline uview<T, Abi, ColMajor>
with_cached_access(const uview<T, Abi, ColMajor> &o) noexcept {
    return o;
}

/// Batched generalized matrix multiplication D⁽ᵀ⁾ = C⁽ᵀ⁾ + A⁽ᵀ⁾ B⁽ᵀ⁾. Single register block.
template <class T, class Abi, int RowsReg, int ColsReg, StorageOrder OA, StorageOrder OB,
          StorageOrder OC, StorageOrder OD>
[[gnu::hot, gnu::flatten, gnu::noinline]] void
batch_gemm_microkernel(const uview<const T, Abi, OA> A, const uview<const T, Abi, OB> B,
                       const std::optional<uview<const T, Abi, OC>> C, const uview<T, Abi, OD> D,
                       const int k) noexcept {
    using simd = datapar::simd<T, Abi>;
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
    using wp  = double;
    using abi = datapar::deduced_abi<wp, v>;
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
