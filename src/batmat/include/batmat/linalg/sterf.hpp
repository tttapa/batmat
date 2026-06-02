#pragma once

#include <batmat/assume.hpp>
#include <batmat/linalg/elementwise.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/simd.hpp>

#include <cmath>
#include <expected>
#include <limits>

namespace batmat::linalg {

struct SterfOptions {
    /// If <= 0, machine epsilon is used.
    double relative_tolerance             = 0.0;
    index_t max_iterations_per_eigenvalue = 64;
};

namespace detail {

[[nodiscard]] constexpr auto default_tolerance(auto user_tol) noexcept {
    return user_tol > 0 ? user_tol : std::numeric_limits<decltype(user_tol)>::epsilon();
}

template <class T, class Abi>
[[nodiscard]] datapar::simd<T, Abi> safe_scaling_factor(datapar::simd<T, Abi> anorm) noexcept {
    using std::sqrt;
    const datapar::simd<T, Abi> zero{0}, one{1};
    static constexpr T safe_min = std::numeric_limits<T>::min();
    static constexpr T safe_max = std::numeric_limits<T>::max();
    static constexpr T ε        = std::numeric_limits<T>::epsilon();
    // Conservative safe range for intermediate squared/hypot-like quantities.
    static constexpr T small = sqrt(safe_min) / ε;
    static constexpr T large = sqrt(safe_max) * ε;

    auto factor = datapar::simd<T, Abi>{one};
    factor      = datapar::select(anorm > large, large / anorm, factor);
    factor      = datapar::select(anorm < small, small / anorm, factor);
    factor      = datapar::select(anorm == zero, one, factor);
    return factor;
}

template <class T, class Abi>
[[nodiscard]] bool all_zero(datapar::simd<T, Abi> x) noexcept {
    using std::all_of;
    return all_of(x == T{0});
}

template <class T, class Abi>
[[nodiscard]] bool negligible_squared_e(datapar::simd<T, Abi> e0_sq, datapar::simd<T, Abi> d0,
                                        datapar::simd<T, Abi> d1, T ε_sq) noexcept {
    using std::all_of;
    using std::fabs;
    return all_of(fabs(e0_sq) <= ε_sq * fabs(d0 * d1));
}

/// Eigenvalues of [a b; b c].
template <class T, class Abi>
[[nodiscard]] std::pair<datapar::simd<T, Abi>, datapar::simd<T, Abi>>
stable_2x2_eigenvalues(datapar::simd<T, Abi> a, datapar::simd<T, Abi> b,
                       datapar::simd<T, Abi> c) noexcept {
    // The hypot form avoids spurious overflow in sqrt((a-c)^2 + 4b^2) for reasonably scaled blocks.
    using std::hypot;
    using std::swap;
    const T half{0.5};
    const auto half_trace = (a + c) * half;
    const auto half_diff  = (a - c) * half;
    const auto radius     = hypot(half_diff, b);
    auto lambda1          = half_trace - radius;
    auto lambda2          = half_trace + radius;
    return {lambda1, lambda2};
}

template <class T, class Abi>
void solve_2x2_squared_e_inplace(uview<T, Abi, StorageOrder::ColMajor> d,
                                 uview<T, Abi, StorageOrder::ColMajor> e, index_t l) noexcept {
    using std::fabs;
    using std::sqrt;
    const auto b = sqrt(fabs(e.load(l, 0))); // TODO: can we avoid the square root here?
    const auto a = d.load(l, 0), c = d.load(l + 1, 0);
    const auto [λ1, λ2] = stable_2x2_eigenvalues(a, b, c);
    d.store(λ1, l, 0), d.store(λ2, l + 1, 0), e.store(T{0}, l, 0);
}

template <class T, class Abi>
void scale_diag_only(uview<T, Abi, StorageOrder::ColMajor> d, index_t l, index_t m,
                     datapar::simd<T, Abi> factor) noexcept {
    for (index_t i = l; i <= m; ++i)
        d.store(d.load(i, 0) * factor, i, 0);
}

template <class T, class Abi>
void scale_squared_e(uview<T, Abi, StorageOrder::ColMajor> d,
                     uview<T, Abi, StorageOrder::ColMajor> e, index_t l, index_t m,
                     datapar::simd<T, Abi> factor) noexcept {
    scale_diag_only(d, l, m, factor);
    const auto factor_sq = factor * factor;
    for (index_t i = l; i < m; ++i)
        e.store(e.load(i, 0) * factor_sq, i, 0);
}

template <class T, class Abi>
void sterf_ql_sweep_squared_e_inplace(uview<T, Abi, StorageOrder::ColMajor> d,
                                      uview<T, Abi, StorageOrder::ColMajor> e, index_t l,
                                      index_t m) noexcept {
    using simd = datapar::simd<T, Abi>;
    using std::fabs;
    using std::hypot;
    using std::sqrt;

    const simd zero{T{0}}, one{T{1}}, two{T{2}};

    const auto p0     = d.load(l, 0);
    const auto e0     = sqrt(fabs(e.load(l, 0)));
    auto σ            = (d.load(l + 1, 0) - p0) / (two * e0);
    const auto rshift = hypot(σ, one);
    σ                 = p0 - e0 / (σ + copysign(rshift, σ));

    auto c = one;
    auto s = zero;
    auto γ = d.load(m, 0) - σ;
    auto p = γ * γ;

    for (index_t i = m; i-- > l;) {
        const auto bb = e.load(i, 0);
        const auto r  = p + bb;
        if (i != m - 1)
            e.store(s * r, i + 1, 0);
        const auto old_c = c;
        c                = p / r;
        s                = bb / r;
        const auto old_γ = γ;
        const auto α     = d.load(i, 0);
        γ                = c * (α - σ) - s * old_γ;
        d.store(old_γ + (α - γ), i + 1, 0);
        p = datapar::select(c != zero, (γ * γ) / c, old_c * bb);
    }
    e.store(s * p, l, 0);
    d.store(σ + γ, l, 0);
}

template <class T, class Abi>
void sterf_qr_sweep_squared_e_inplace(uview<T, Abi, StorageOrder::ColMajor> d,
                                      uview<T, Abi, StorageOrder::ColMajor> e, index_t l,
                                      index_t m) noexcept {
    using simd = datapar::simd<T, Abi>;
    using std::fabs;
    using std::hypot;
    using std::sqrt;

    const simd zero{T{0}}, one{T{1}}, two{T{2}};

    const auto p0     = d.load(m, 0);
    const auto e0     = sqrt(fabs(e.load(m - 1, 0)));
    auto σ            = (d.load(m - 1, 0) - p0) / (two * e0);
    const auto rshift = hypot(σ, one);
    σ                 = p0 - e0 / (σ + copysign(rshift, σ));

    auto c = one;
    auto s = zero;
    auto γ = d.load(l, 0) - σ;
    auto p = γ * γ;

    for (index_t i = l; i < m; ++i) {
        const auto bb = e.load(i, 0);
        const auto r  = p + bb;
        if (i != l)
            e.store(s * r, i - 1, 0);
        const auto old_c = c;
        c                = p / r;
        s                = bb / r;
        const auto old_γ = γ;
        const auto α     = d.load(i + 1, 0);
        γ                = c * (α - σ) - s * old_γ;
        d.store(old_γ + (α - γ), i, 0);
        p = datapar::select(c != zero, (γ * γ) / c, old_c * bb);
    }
    e.store(s * p, m - 1, 0);
    d.store(σ + γ, m, 0);
}

template <class T, class Abi>
void sterf_dynamic_step_squared_e_inplace(uview<T, Abi, StorageOrder::ColMajor> d,
                                          uview<T, Abi, StorageOrder::ColMajor> e, index_t l,
                                          index_t m) noexcept {
    using std::fabs;
    static constexpr index_t half_v = datapar::simd_size<T, Abi>::value / 2;
    const bool use_qr = datapar::reduce_count(fabs(d.load(m, 0)) < fabs(d.load(l, 0))) > half_v;
    if (use_qr)
        sterf_qr_sweep_squared_e_inplace<T, Abi>(d, e, l, m);
    else
        sterf_ql_sweep_squared_e_inplace<T, Abi>(d, e, l, m);
}

template <class T, class Abi>
[[nodiscard]] datapar::simd<T, Abi>
squared_block_norm_estimate_from_squared_e(uview<T, Abi, StorageOrder::ColMajor> d,
                                           uview<T, Abi, StorageOrder::ColMajor> e_sq, index_t l,
                                           index_t m) noexcept {
    using simd = datapar::simd<T, Abi>;
    using std::fabs;
    using std::max;

    simd anorm_sq{T{0}};
    for (index_t i = l; i <= m; ++i) {
        const auto di = d.load(i, 0);
        anorm_sq      = max(anorm_sq, di * di);
    }
    for (index_t i = l; i < m; ++i) {
        const auto ei_sq = fabs(e_sq.load(i, 0)); // may be negative due to rounding
        anorm_sq         = max(anorm_sq, ei_sq);
    }
    return anorm_sq;
}

template <class T, class Abi>
[[nodiscard]] datapar::simd<T, Abi>
block_norm_estimate_from_squared_e(uview<T, Abi, StorageOrder::ColMajor> d,
                                   uview<T, Abi, StorageOrder::ColMajor> e_sq, index_t l,
                                   index_t m) noexcept {
    using std::sqrt;
    return sqrt(squared_block_norm_estimate_from_squared_e<T, Abi>(d, e_sq, l, m));
}

template <class T, class Abi>
std::expected<index_t, index_t> sterf(view<T, Abi, StorageOrder::ColMajor> diag,
                                      view<T, Abi, StorageOrder::ColMajor> subdiag,
                                      SterfOptions options) {
    static_assert(!std::is_const_v<T>);
    BATMAT_ASSERT(diag.cols() == 1);
    BATMAT_ASSERT(subdiag.cols() == 1);
    const index_t n = diag.rows();
    if (n > 0)
        BATMAT_ASSERT(subdiag.rows() == n - 1);
    else
        BATMAT_ASSERT(subdiag.rows() == 0);
    if (n <= 1)
        return 0;

    using simd = datapar::simd<T, Abi>;
    using std::any_of;

    const T ε    = default_tolerance(static_cast<T>(options.relative_tolerance));
    const T ε_sq = ε * ε;
    const simd zero{T{0}}, one{T{1}};
    const index_t max_total_iterations = options.max_iterations_per_eigenvalue * n;
    index_t total_iterations           = 0;

    const uview<T, Abi, StorageOrder::ColMajor> d{diag};
    const uview<T, Abi, StorageOrder::ColMajor> e{subdiag};

    // Square all offdiagonals globally and apply the equivalent squared LAPACK split test:
    //     |e_i| <= eps * sqrt(|d_i|) * sqrt(|d_{i+1}|)
    // becomes
    //     e_i^2 <= eps^2 * |d_i * d_{i+1}|.
    for (index_t i = 0; i + 1 < n; ++i) {
        const auto ei    = e.load(i, 0);
        const auto ei_sq = ei * ei;
        const auto di = d.load(i, 0), di_next = d.load(i + 1, 0);
        const bool split = negligible_squared_e(ei_sq, di, di_next, ε_sq);
        e.store(split ? zero : ei_sq, i, 0);
    }

    bool found_unreduced_block;
    do {
        found_unreduced_block = false;

        index_t l = 0;
        while (l < n) {
            // Skip over converged 1x1 blocks.
            while (l + 1 < n && all_zero(e.load(l, 0)))
                ++l;
            if (l + 1 >= n)
                break;
            // Find the end of the current unreduced block.
            index_t m;
            for (m = l; m + 1 < n; ++m) {
                const auto em = e.load(m, 0);
                if (all_zero(em))
                    break;
                if (negligible_squared_e(em, d.load(m, 0), d.load(m + 1, 0), ε_sq)) {
                    e.store(zero, m, 0);
                    break;
                }
            }
            // Active unreduced block is d[l..m]. Reduce it.
            if (m > l) {
                found_unreduced_block = true;
                const auto anorm      = block_norm_estimate_from_squared_e(d, e, l, m);
                const auto factor     = safe_scaling_factor(anorm);
                const bool scaled     = any_of(factor != one);
                if (scaled)
                    scale_squared_e(d, e, l, m, factor);

                if (m == l + 1) // Solve the 2×2 block directly rather than using QR sweeps
                    solve_2x2_squared_e_inplace(d, e, l);
                else if (++total_iterations < max_total_iterations)
                    sterf_dynamic_step_squared_e_inplace(d, e, l, m);

                if (scaled)
                    scale_squared_e(d, e, l, m, T{1} / factor);
                if (total_iterations >= max_total_iterations)
                    return std::unexpected(total_iterations);
            }

            l = m + 1; // Beginning of next block
        }
    } while (found_unreduced_block);
    return total_iterations;
}

} // namespace detail

/// Eigenvalues of a symmetric tridiagonal matrix given by `diag` and `subdiag`, computed in-place
/// using the Pal-Walker-Kahan variant of the implicit QR/QL method with Wilkinson shifts. Based on
/// LAPACK 3.12.1's `STERF`:
/// https://netlib.org/lapack//explore-html/d4/d9d/group__sterf_gad293bb81da1c7785b42796d1e197f08c.html
/// @return Number of QR steps taken. An unexpected value signals failure to converge within the
///         maximum number of iterations specified in `options`.
template <simdifiable Vd, simdifiable Vs>
    requires simdify_compatible<Vd, Vs>
std::expected<index_t, index_t> sterf(Vd &&diag, Vs &&subdiag, SterfOptions options = {}) {
    static_assert(decltype(simdify(diag))::storage_order == StorageOrder::ColMajor);
    static_assert(decltype(simdify(subdiag))::storage_order == StorageOrder::ColMajor);
    return detail::sterf<simdified_value_t<Vd>, simdified_abi_t<Vd>>(simdify(diag),
                                                                     simdify(subdiag), options);
}

template <simdifiable VA, simdifiable Vd, simdifiable Vs, MatrixStructure SA>
    requires simdify_compatible<VA, Vd, Vs>
void extract_bidiag(Structured<VA, SA> A, Vd &&diag, Vs &&offdiag) {
    BATMAT_ASSERT(rows(A.value) == cols(A.value));
    BATMAT_ASSERT(rows(A.value) == rows(diag));
    BATMAT_ASSERT(cols(diag) == 1);
    copy_diag(A.value, diag);
    if (const index_t n = rows(A.value); n > 1) {
        BATMAT_ASSERT(rows(offdiag) == n - 1);
        BATMAT_ASSERT(cols(offdiag) == 1);
        if constexpr (SA == MatrixStructure::UpperTriangular)
            copy_diag(A.value.top_right(n - 1, n - 1), offdiag);
        else
            copy_diag(A.value.bottom_left(n - 1, n - 1), offdiag);
    }
}

} // namespace batmat::linalg
