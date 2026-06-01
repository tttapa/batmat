#pragma once

#include <cmath>
#include <limits>
#include <utility>

#include <batmat/assume.hpp>
#include <batmat/linalg/elementwise.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/simd.hpp>

namespace batmat::datapar {

auto two_mult_fma(auto a, auto b) {
    using std::fma;
    auto p = a * b;
    return std::make_pair(p, fma(a, b, -p));
}

auto weak_hypot(auto f, auto g) {
    using datapar::select;
    using std::abs;
    using std::sqrt;
    auto af = abs(f), ag = abs(g);
    auto smol = select(af < ag, af, ag);
    auto larg = select(af < ag, ag, af);
    auto r    = smol / larg;
    return select(larg == 0, 0, larg * sqrt(1 + r * r));
}

/// https://arxiv.org/pdf/2406.02750v1
[[gnu::flatten]]
auto compensated_givens(auto f, auto g) {
    using std::fma;
    auto r = weak_hypot(f, g);
    auto c̅ = f / r, s̅ = g / r;
    auto [c1, c2] = two_mult_fma(c̅, c̅);
    auto [s1, s2] = two_mult_fma(s̅, s̅);
    auto ε_norm   = select(c1 > s1, 1 - c1 - s1 - c2 - s2, 1 - s1 - c1 - s2 - c2) / 2;
    auto [p, pp]  = two_mult_fma(c̅, g);
    auto ε_orth   = (fma(-s̅, f, p) + pp) / r;
    auto δc       = c̅ * ε_norm - s̅ * ε_orth;
    auto δs       = s̅ * ε_norm + c̅ * ε_orth;
    auto c        = select(r == 0, 1, c̅ + δc);
    auto s        = select(r == 0, 0, s̅ + δs);
    return std::make_pair(c, s);
}

[[gnu::flatten]]
auto weak_givens(auto f, auto g) {
    auto r = weak_hypot(f, g);
    auto c = select(r == 0, 1, f / r);
    auto s = select(r == 0, 0, g / r);
    return std::make_pair(c, s);
}

} // namespace batmat::datapar

namespace batmat::linalg {

struct TridiagonalQrOptions {
    /// If <= 0, machine epsilon is used.
    double relative_tolerance            = 0.0;
    size_t max_iterations_per_eigenvalue = 64;
};

namespace detail {

[[nodiscard]] constexpr auto default_tolerance(auto user_tol) noexcept {
    return user_tol > 0 ? user_tol : std::numeric_limits<decltype(user_tol)>::epsilon();
}

template <class T, class Abi>
[[nodiscard]] datapar::simd<T, Abi>
block_norm_estimate(uview<const T, Abi, StorageOrder::ColMajor> d,
                    uview<const T, Abi, StorageOrder::ColMajor> e, index_t l, index_t m) noexcept {
    using std::abs;
    using std::max;
    datapar::simd<T, Abi> anorm{};
    for (index_t i = l; i <= m; ++i)
        anorm = max(anorm, abs(d.load(i, 0)));
    for (index_t i = l; i < m; ++i)
        anorm = max(anorm, abs(e.load(i, 0)));
    return anorm;
}

template <class T, class Abi>
void scale_block(uview<T, Abi, StorageOrder::ColMajor> d, uview<T, Abi, StorageOrder::ColMajor> e,
                 index_t l, index_t m, datapar::simd<T, Abi> factor) noexcept {
    for (index_t i = l; i <= m; ++i)
        d.store(d.load(i, 0) * factor, i, 0);
    for (index_t i = l; i < m; ++i)
        e.store(e.load(i, 0) * factor, i, 0);
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

    datapar::simd<T, Abi> factor = one;
    factor                       = datapar::select(anorm > large, large / anorm, factor);
    factor                       = datapar::select(anorm < small, small / anorm, factor);
    factor                       = datapar::select(anorm == zero, one, factor);
    return factor;
}

template <class T, class Abi>
[[nodiscard]] bool negligible_subdiagonal_pwk(datapar::simd<T, Abi> subdiag,
                                              datapar::simd<T, Abi> diag_left,
                                              datapar::simd<T, Abi> diag_right,
                                              datapar::simd<T, Abi> rel_tol) noexcept {
    // Pal-Walker-Kahan/LAPACK-style squared test:
    //     e_i^2 <= eps^2 * |d_i| * |d_{i+1}| + safe_min
    // This avoids taking square roots and behaves better when adjacent
    // diagonals vary substantially in scale.
    using std::abs;
    using std::all_of;
    const auto eps2  = rel_tol * rel_tol;
    const T safe_min = std::numeric_limits<T>::min();

    const auto lhs = abs(subdiag) * abs(subdiag);
    const auto rhs = eps2 * abs(diag_left) * abs(diag_right) + safe_min;
    return all_of(lhs <= rhs);
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
void solve_2x2_block_inplace(uview<T, Abi, StorageOrder::ColMajor> d,
                             uview<T, Abi, StorageOrder::ColMajor> e, index_t l) noexcept {
    const auto [lambda1, lambda2] =
        stable_2x2_eigenvalues(d.load(l, 0), e.load(l, 0), d.load(l + 1, 0));
    d.store(lambda1, l, 0);
    e.store(datapar::simd<T, Abi>{}, l, 0);
    d.store(lambda2, l + 1, 0);
}

template <class T, class Abi>
void reverse_simd(uview<T, Abi, StorageOrder::ColMajor> v, index_t n) noexcept {
    index_t head = 0, tail = n;
    if (head != tail) {
        --tail;
        while (head < tail) {
            const auto t = v.load(head, 0);
            v.store(v.load(tail, 0), head, 0);
            v.store(t, tail, 0);
            ++head;
            --tail;
        }
    }
}

template <class T, class Abi>
void reverse_active_block(uview<T, Abi, StorageOrder::ColMajor> d,
                          uview<T, Abi, StorageOrder::ColMajor> e, index_t l, index_t m) noexcept {
    reverse_simd<T, Abi>(d.middle_rows(l), m - l + 1);
    if (m > l)
        reverse_simd<T, Abi>(e.middle_rows(l), m - l);
}

template <class T, class Abi>
[[gnu::flatten]]
void implicit_wilkinson_ql_step_inplace(uview<T, Abi, StorageOrder::ColMajor> d,
                                        uview<T, Abi, StorageOrder::ColMajor> e, index_t l,
                                        index_t m, index_t n) {
    using simd = datapar::simd<T, Abi>;
    using std::abs;
    using std::copysign;
    using std::hypot;
    const T zero{0}, one{1}, two{2};

    // Wilkinson shift in QL orientation.
    auto g = (d.load(l + 1, 0) - d.load(l, 0)) / (two * e.load(l, 0));
    auto r = hypot(g, one); // TODO: do we need a fully robust hypot here?

    g = d.load(m, 0) - d.load(l, 0) + e.load(l, 0) / (g + copysign(r, g));

    simd s{one};
    simd c{one};
    simd p{zero};

    for (index_t i = m; i-- > l;) {
        const auto ei = e.load(i, 0);
        const auto f = s * ei, b = c * ei;
        std::tie(c, s) = datapar::compensated_givens(g, f);
        if (i + 1 < n - 1)
            e.store(c * g + s * f, i + 1, 0);

        g            = d.load(i + 1, 0) - p;
        const auto r = (d.load(i, 0) - g) * s + c * b + b * c;
        p            = s * r;

        d.store(g + p, i + 1, 0);
        g = c * r - b;
    }

    d.store(d.load(l, 0) - p, l, 0);
    e.store(g, l, 0);
    if (m < n - 1)
        e.store(zero, m, 0); // e[m] is boundary after active block. Ignore if m == n - 1.
}

template <class T, class Abi>
void implicit_wilkinson_qr_step_inplace(uview<T, Abi, StorageOrder::ColMajor> d,
                                        uview<T, Abi, StorageOrder::ColMajor> e, index_t l,
                                        index_t m, index_t n) {
    // TODO: avoid actually reversing the storage, just templatize the QR/QL step
    reverse_active_block<T, Abi>(d, e, l, m);
    implicit_wilkinson_ql_step_inplace<T, Abi>(d, e, l, m, n);
    reverse_active_block<T, Abi>(d, e, l, m);
}

template <class T, class Abi>
void implicit_wilkinson_step_inplace(uview<T, Abi, StorageOrder::ColMajor> d,
                                     uview<T, Abi, StorageOrder::ColMajor> e, index_t l, index_t m,
                                     index_t n) {
    // Temporarily scale the active block to safeguard against overflow/underflow in the QR/QL step.
    using std::abs;
    using std::any_of;

    const auto anorm  = block_norm_estimate<T, Abi>(d, e, l, m);
    const auto factor = safe_scaling_factor(anorm);

    if (any_of(factor != T{1}))
        scale_block(d, e, l, m, factor);

    // Direction choice:
    // If the lower-right endpoint is smaller in magnitude, QR orientation
    // tends to deflate from that end more favorably. Otherwise use QL.
    static constexpr index_t half_v = datapar::simd_size<T, Abi>::value / 2;
    if (datapar::reduce_count(abs(d.load(m, 0)) < abs(d.load(l, 0))) > half_v)
        implicit_wilkinson_qr_step_inplace<T, Abi>(d, e, l, m, n);
    else
        implicit_wilkinson_ql_step_inplace<T, Abi>(d, e, l, m, n);

    if (any_of(factor != T{1}))
        scale_block(d, e, l, m, T{1} / factor);
}

template <class T, class Abi>
bool all_zero(datapar::simd<T, Abi> v) noexcept {
    using std::all_of;
    return all_of(v == T{0});
}

template <class T, class Abi>
index_t symmetric_tridiagonal_eigenvalues_inplace(view<T, Abi, StorageOrder::ColMajor> diag,
                                                  view<T, Abi, StorageOrder::ColMajor> subdiag,
                                                  TridiagonalQrOptions options) {
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

    const T rtol = default_tolerance(static_cast<T>(options.relative_tolerance));
    const index_t max_total_iterations = options.max_iterations_per_eigenvalue * n;
    index_t total_iterations           = 0;

    const uview<T, Abi, StorageOrder::ColMajor> d{diag};
    const uview<T, Abi, StorageOrder::ColMajor> e{subdiag};

    // Repeatedly split the matrix at negligible subdiagonals and process
    // each unreduced block.
    const auto reduce_block = [&](index_t l, index_t m) {
        implicit_wilkinson_step_inplace(d, e, l, m, n);
        // Clean up any newly negligible subdiagonal entries in the active block.
        for (index_t i = l; i < m; ++i)
            if (negligible_subdiagonal_pwk<T, Abi>(e.load(i, 0), d.load(i, 0), d.load(i + 1, 0),
                                                   rtol))
                e.store(T{0}, i, 0);
    };
    bool found_unreduced_block;
    do {
        found_unreduced_block = false;
        index_t l             = 0;
        while (l < n) {
            // Skip over converged 1x1 blocks.
            while (l + 1 < n && all_zero(e.load(l, 0)))
                ++l;
            if (l + 1 >= n)
                break;
            // Find the end of the active unreduced block.
            index_t m;
            for (m = l; m + 1 < n; ++m) {
                auto em = e.load(m, 0);
                if (all_zero(em))
                    break;
                if (negligible_subdiagonal_pwk<T, Abi>(em, d.load(m, 0), d.load(m + 1, 0), rtol)) {
                    e.store(T{0}, m, 0);
                    break;
                }
            }
            // Active unreduced block is d[l..m]. Reduce it.
            if (m > l) {
                found_unreduced_block = true;
                if (m == l + 1)
                    solve_2x2_block_inplace(d, e, l);
                else if (++total_iterations > max_total_iterations)
                    return total_iterations;
                else
                    reduce_block(l, m);
            }
            l = m + 1;
        }
    } while (found_unreduced_block);
    return total_iterations;
}

} // namespace detail

template <simdifiable Vd, simdifiable Vs>
    requires simdify_compatible<Vd, Vs>
index_t eigvalsh_trd(Vd &&diag, Vs &&subdiag, TridiagonalQrOptions options = {}) {
    static_assert(decltype(simdify(diag))::storage_order == StorageOrder::ColMajor);
    static_assert(decltype(simdify(subdiag))::storage_order == StorageOrder::ColMajor);
    return detail::symmetric_tridiagonal_eigenvalues_inplace<simdified_value_t<Vd>,
                                                             simdified_abi_t<Vd>>(
        simdify(diag), simdify(subdiag), options);
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
