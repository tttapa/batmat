#pragma once

#include <batmat/linalg/elementwise.hpp>
#include <batmat/linalg/norms.hpp>

namespace batmat::linalg {

/// @cond DETAIL

namespace detail {

template <class T, class Abi, StorageOrder O0, class Tinit, class F, class... Args>
auto vreduce(Tinit init, F fun, view<const T, Abi, O0> x0, const Args &...xs) {
    BATMAT_ASSERT(((x0.rows() == xs.rows()) && ...));
    BATMAT_ASSERT(((x0.cols() == xs.cols()) && ...));
    BATMAT_ASSERT(((x0.depth() == xs.depth()) && ...));
    BATMAT_ASSERT(((x0.batch_size() == xs.batch_size()) && ...));
    iter_elems<T, Abi, O0>([&](auto... args) { init = fun(init, args...); }, x0, xs...);
    return init;
}

template <class T, class Abi, StorageOrder O0, class Tinit, class F, class R, class... Args>
auto reduce(Tinit init, F fun, R reduce_fn, view<const T, Abi, O0> x0, const Args &...xs) {
    return reduce_fn(vreduce(init, fun, x0, xs...));
}

template <class T, class Abi, StorageOrder OA>
[[gnu::flatten]] auto vnorms_all(view<const T, Abi, OA> A) {
    using simd  = datapar::simd<T, Abi>;
    using norms = linalg::norms<T, simd>;
    return vreduce<T, Abi>(norms::zero_simd(), norms(), A);
}

template <class T, class Abi, StorageOrder OA>
[[gnu::flatten]] linalg::norms<T>::result norms_all(view<const T, Abi, OA> A) {
    using simd = datapar::simd<T, Abi>;
    return linalg::norms<T, simd>{}(vnorms_all<T, Abi>(A));
}

/// Dot product (lane-wise).
template <class T, class Abi, StorageOrder OA, StorageOrder OB>
[[gnu::flatten]] auto vdot(view<const T, Abi, OA> a, view<const T, Abi, OB> b) {
    using simd = datapar::simd<T, Abi>;
    auto fma   = [](auto accum, auto ai, auto bi) { return ai * bi + accum; };
    return vreduce<T, Abi>(simd{0}, fma, a, b);
}

/// Dot product.
template <class T, class Abi, StorageOrder OA, StorageOrder OB>
[[gnu::flatten]] T dot(view<const T, Abi, OA> a, view<const T, Abi, OB> b) {
    return reduce(vdot<T, Abi>(a, b));
}

/// Squared 2-norm (lane-wise).
template <class T, class Abi, StorageOrder OA>
[[gnu::flatten]] auto vnorm_2_squared(view<const T, Abi, OA> a) {
    using simd = datapar::simd<T, Abi>;
    auto fma   = [](auto accum, auto ai) { return ai * ai + accum; };
    return vreduce<T, Abi>(simd{0}, fma, a);
}

/// Squared 2-norm.
template <class T, class Abi, StorageOrder OA>
[[gnu::flatten]] T norm_2_sq(view<const T, Abi, OA> a) {
    return reduce(vnorm_2_squared<T, Abi>(a));
}

/// ∑ wᵢ aᵢ² (lane-wise).
template <class T, class Abi, StorageOrder OW, StorageOrder OA>
[[gnu::flatten]] auto weighted_vnorm_sq(view<const T, Abi, OW> w, view<const T, Abi, OA> a) {
    using simd = datapar::simd<T, Abi>;
    auto wnd   = [](auto accum, auto wi, auto ai) { return wi * (ai * ai) + accum; };
    return vreduce<T, Abi>(simd{0}, wnd, w, a);
}

/// ∑ wᵢ aᵢ².
template <class T, class Abi, StorageOrder OW, StorageOrder OA>
[[gnu::flatten]] T weighted_norm_sq(view<const T, Abi, OW> w, view<const T, Abi, OA> a) {
    return reduce(weighted_vnorm_sq<T, Abi>(w, a));
}

/// ∑ wᵢ(aᵢ - bᵢ)² (lane-wise).
template <class T, class Abi, StorageOrder OW, StorageOrder OA, StorageOrder OB>
[[gnu::flatten]] auto weighted_vnorm_sq_difference(view<const T, Abi, OW> w,
                                                   view<const T, Abi, OA> a,
                                                   view<const T, Abi, OB> b) {
    using simd = datapar::simd<T, Abi>;
    auto wnd   = [](auto accum, auto wi, auto ai, auto bi) {
        auto ei = ai - bi;
        return wi * (ei * ei) + accum;
    };
    return vreduce<T, Abi>(simd{0}, wnd, w, a, b);
}

/// ∑ wᵢ(aᵢ - bᵢ)².
template <class T, class Abi, StorageOrder OW, StorageOrder OA, StorageOrder OB>
[[gnu::flatten]] T weighted_norm_sq_difference(view<const T, Abi, OW> w, view<const T, Abi, OA> a,
                                               view<const T, Abi, OB> b) {
    return reduce(weighted_vnorm_sq_difference<T, Abi>(w, a, b));
}

} // namespace detail

/// @endcond

/// @addtogroup topic-linalg
/// @{

/// @name Single-batch reduction operations
/// @{

/// Compute the lane-wise norms (max, 1-norm, and 2-norm) of a batch of vectors.
template <simdifiable Vx>
norms<simdified_value_t<Vx>, simdified_simd_t<Vx>>::result_simd vnorms_all(Vx &&x) {
    GUANAQO_TRACE_LINALG("vnorms_all", 3 * detail::num_elem(simdify(x))); // fma, add, max
    return detail::vnorms_all<simdified_value_t<Vx>, simdified_abi_t<Vx>>(simdify(x).as_const());
}

/// Compute the norms (max, 1-norm, and 2-norm) of a vector.
template <simdifiable Vx>
norms<simdified_value_t<Vx>>::result norms_all(Vx &&x) {
    GUANAQO_TRACE_LINALG("norms_all", 3 * detail::num_elem(simdify(x))); // fma, add, max
    return detail::norms_all<simdified_value_t<Vx>, simdified_abi_t<Vx>>(simdify(x).as_const());
}

/// Compute the lane-wise infinity norms of a batch of vectors.
template <simdifiable Vx>
simdified_simd_t<Vx> vnorm_inf(Vx &&x) {
    return vnorms_all(std::forward<Vx>(x)).amax;
}

/// Compute the infinity norm of a vector.
template <simdifiable Vx>
simdified_value_t<Vx> norm_inf(Vx &&x) {
    return norms_all(std::forward<Vx>(x)).norm_inf();
}

/// Compute the lane-wise 1-norms of a batch of vectors.
template <simdifiable Vx>
simdified_simd_t<Vx> vnorm_1(Vx &&x) {
    return vnorms_all(std::forward<Vx>(x)).asum;
}

/// Compute the 1-norm of a vector.
template <simdifiable Vx>
simdified_value_t<Vx> norm_1(Vx &&x) {
    return norms_all(std::forward<Vx>(x)).norm_1();
}

/// Compute the lane-wise squared 2-norms of a batch of vectors.
template <simdifiable Vx>
simdified_simd_t<Vx> vnorm_2_squared(Vx &&x) {
    GUANAQO_TRACE_LINALG("vnorm_2_squared", detail::num_elem(simdify(x)));
    return detail::vnorm_2_squared<simdified_value_t<Vx>, simdified_abi_t<Vx>>(
        simdify(x).as_const());
}

/// Compute the squared 2-norm of a vector.
template <simdifiable Vx>
simdified_value_t<Vx> norm_2_squared(Vx &&x) {
    GUANAQO_TRACE_LINALG("norm_2_squared", detail::num_elem(simdify(x)));
    return detail::norm_2_sq<simdified_value_t<Vx>, simdified_abi_t<Vx>>(simdify(x).as_const());
}

/// Compute the lane-wise 2-norms of a batch of vectors.
template <simdifiable Vx>
simdified_simd_t<Vx> vnorm_2(Vx &&x) {
    using std::sqrt;
    return sqrt(vnorm_2_squared(std::forward<Vx>(x)));
}

/// Compute the 2-norm of a vector.
template <simdifiable Vx>
simdified_value_t<Vx> norm_2(Vx &&x) {
    using std::sqrt;
    return sqrt(norm_2_squared(std::forward<Vx>(x)));
}

/// Compute the lane-wise dot products of two batches of vectors.
template <simdifiable Vx, simdifiable Vy>
    requires simdify_compatible<Vx, Vy>
simdified_simd_t<Vx> vdot(Vx &&x, Vy &&y) {
    GUANAQO_TRACE_LINALG("vdot", detail::num_elem(simdify(x)));
    return detail::vdot<simdified_value_t<Vx>, simdified_abi_t<Vx>>(simdify(x).as_const(),
                                                                    simdify(y).as_const());
}

/// Compute the dot product of two vectors.
template <simdifiable Vx, simdifiable Vy>
    requires simdify_compatible<Vx, Vy>
simdified_value_t<Vx> dot(Vx &&x, Vy &&y) {
    GUANAQO_TRACE_LINALG("dot", detail::num_elem(simdify(x)));
    return detail::dot<simdified_value_t<Vx>, simdified_abi_t<Vx>>(simdify(x).as_const(),
                                                                   simdify(y).as_const());
}

/// ∑ wᵢ aᵢ² (lane-wise).
template <simdifiable Vw, simdifiable Va>
    requires simdify_compatible<Vw, Va>
simdified_simd_t<Vw> weighted_vnorm_sq(Vw &&w, Va &&a) {
    GUANAQO_TRACE_LINALG("weighted_vnorm_sq", 2 * detail::num_elem(simdify(w)));
    return detail::weighted_vnorm_sq<simdified_value_t<Vw>, simdified_abi_t<Vw>>(
        simdify(w).as_const(), simdify(a).as_const());
}

/// ∑ wᵢ aᵢ².
template <simdifiable Vw, simdifiable Va>
    requires simdify_compatible<Vw, Va>
simdified_value_t<Vw> weighted_norm_sq(Vw &&w, Va &&a) {
    GUANAQO_TRACE_LINALG("weighted_norm_sq", 2 * detail::num_elem(simdify(w)));
    return detail::weighted_norm_sq<simdified_value_t<Vw>, simdified_abi_t<Vw>>(
        simdify(w).as_const(), simdify(a).as_const());
}

/// ∑ wᵢ(aᵢ - bᵢ)² (lane-wise).
template <simdifiable Vw, simdifiable Va, simdifiable Vb>
    requires simdify_compatible<Vw, Va, Vb>
simdified_simd_t<Vw> weighted_vnorm_sq_diff(Vw &&w, Va &&a, Vb &&b) {
    GUANAQO_TRACE_LINALG("weighted_vnorm_sq_diff", 3 * detail::num_elem(simdify(w)));
    return detail::weighted_vnorm_sq_difference<simdified_value_t<Vw>, simdified_abi_t<Vw>>(
        simdify(w).as_const(), simdify(a).as_const(), simdify(b).as_const());
}

/// ∑ wᵢ(aᵢ - bᵢ)².
template <simdifiable Vw, simdifiable Va, simdifiable Vb>
    requires simdify_compatible<Vw, Va, Vb>
simdified_value_t<Vw> weighted_norm_sq_diff(Vw &&w, Va &&a, Vb &&b) {
    GUANAQO_TRACE_LINALG("weighted_norm_sq_difference", 3 * detail::num_elem(simdify(w)));
    return detail::weighted_norm_sq_difference<simdified_value_t<Vw>, simdified_abi_t<Vw>>(
        simdify(w).as_const(), simdify(a).as_const(), simdify(b).as_const());
}

/// @}

/// @}

// TODO: doxygen gets confused because the template parameters are the same as the single-batch
// versions, so put in a separate namespace
inline namespace multi {

/// @addtogroup topic-linalg
/// @{

/// @name Multi-batch reduction operations
/// @{

/// Compute the norms (max, 1-norm, and 2-norm) of a vector.
template <simdifiable_multi Vx>
norms<simdified_value_t<Vx>>::result norms_all(Vx &&x) {
    using norms = linalg::norms<simdified_value_t<Vx>>;
    typename norms::result result{};
    for (index_t b = 0; b < x.num_batches(); ++b)
        result = norms{}(result, linalg::norms_all(x.batch(b)));
    return result;
}

/// Compute the infinity norm of a vector.
template <simdifiable_multi Vx>
simdified_value_t<Vx> norm_inf(Vx &&x) {
    return norms_all(std::forward<Vx>(x)).norm_inf();
}

/// Compute the 1-norm of a vector.
template <simdifiable_multi Vx>
simdified_value_t<Vx> norm_1(Vx &&x) {
    return norms_all(std::forward<Vx>(x)).norm_1();
}

/// Compute the squared 2-norm of a vector.
template <simdifiable_multi Vx>
simdified_value_t<Vx> norm_2_squared(Vx &&x) {
    simdified_value_t<Vx> sumsq{};
    for (index_t b = 0; b < x.num_batches(); ++b)
        sumsq += linalg::norm_2_squared(x.batch(b));
    return sumsq;
}

/// Compute the 2-norm of a vector.
template <simdifiable_multi Vx>
simdified_value_t<Vx> norm_2(Vx &&x) {
    using std::sqrt;
    return sqrt(norm_2_squared(std::forward<Vx>(x)));
}

/// Compute the dot product of two vectors.
template <simdifiable_multi Vx, simdifiable_multi Vy>
    requires simdify_compatible<Vx, Vy>
simdified_value_t<Vx> dot(Vx &&x, Vy &&y) {
    BATMAT_ASSERT(x.num_batches() == y.num_batches());
    simdified_value_t<Vx> result{};
    for (index_t b = 0; b < x.num_batches(); ++b)
        result += linalg::dot(x.batch(b), y.batch(b));
    return result;
}

/// ∑ wᵢ xᵢ².
template <simdifiable_multi Vw, simdifiable_multi Vx>
    requires simdify_compatible<Vw, Vx>
simdified_value_t<Vw> weighted_norm_sq(Vw &&w, Vx &&x) {
    BATMAT_ASSERT(w.num_batches() == x.num_batches());
    simdified_value_t<Vw> result{};
    for (index_t b = 0; b < w.num_batches(); ++b)
        result += linalg::weighted_norm_sq(w.batch(b), x.batch(b));
    return result;
}

/// ∑ wᵢ(xᵢ - yᵢ)².
template <simdifiable_multi Vw, simdifiable_multi Vx, simdifiable_multi Vy>
    requires simdify_compatible<Vw, Vx, Vy>
simdified_value_t<Vw> weighted_norm_sq_difference(Vw &&w, Vx &&x, Vy &&y) {
    BATMAT_ASSERT(w.num_batches() == x.num_batches());
    BATMAT_ASSERT(w.num_batches() == y.num_batches());
    simdified_value_t<Vw> result{};
    for (index_t b = 0; b < w.num_batches(); ++b)
        result += linalg::weighted_norm_sq_diff(w.batch(b), x.batch(b), y.batch(b));
    return result;
}

/// Compute the lane-wise norms (max, 1-norm, and 2-norm) of a batch of vectors.
template <simdifiable_multi Vx>
norms<simdified_value_t<Vx>, simdified_simd_t<Vx>>::result_simd vnorms_all(Vx &&x) {
    using std::max;
    using norms = linalg::norms<simdified_value_t<Vx>, simdified_simd_t<Vx>>;
    typename norms::result_simd result{};
    for (index_t b = 0; b < x.num_batches(); ++b)
        result = norms{}(result, linalg::vnorms_all(x.batch(b)));
    return result;
}

/// Compute the lane-wise infinity norms of a batch of vectors.
template <simdifiable_multi Vx>
simdified_simd_t<Vx> vnorm_inf(Vx &&x) {
    return vnorms_all(std::forward<Vx>(x)).norm_inf();
}

/// Compute the lane-wise 1-norms of a batch of vectors.
template <simdifiable_multi Vx>
simdified_simd_t<Vx> vnorm_1(Vx &&x) {
    return vnorms_all(std::forward<Vx>(x)).norm_1();
}

/// Compute the lane-wise squared 2-norms of a batch of vectors.
template <simdifiable_multi Vx>
simdified_simd_t<Vx> vnorm_2_squared(Vx &&x) {
    simdified_simd_t<Vx> result{};
    for (index_t b = 0; b < x.num_batches(); ++b)
        result += linalg::vnorm_2_squared(x.batch(b));
    return result;
}

/// Compute the lane-wise 2-norms of a batch of vectors.
template <simdifiable_multi Vx>
simdified_simd_t<Vx> vnorm_2(Vx &&x) {
    using std::sqrt;
    return sqrt(vnorm_2_squared(std::forward<Vx>(x)));
}

/// Compute the lane-wise dot products of two batches of vectors.
template <simdifiable_multi Vx, simdifiable_multi Vy>
    requires simdify_compatible<Vx, Vy>
simdified_simd_t<Vx> vdot(Vx &&x, Vy &&y) {
    BATMAT_ASSERT(x.num_batches() == y.num_batches());
    simdified_simd_t<Vx> result{};
    for (index_t b = 0; b < x.num_batches(); ++b)
        result += linalg::vdot(x.batch(b), y.batch(b));
    return result;
}

/// ∑ wᵢ xᵢ² (lane-wise).
template <simdifiable_multi Vw, simdifiable_multi Vx>
    requires simdify_compatible<Vw, Vx>
simdified_simd_t<Vw> weighted_vnorm_sq(Vw &&w, Vx &&x) {
    BATMAT_ASSERT(w.num_batches() == x.num_batches());
    simdified_simd_t<Vw> result{};
    for (index_t b = 0; b < w.num_batches(); ++b)
        result += linalg::weighted_vnorm_sq(w.batch(b), x.batch(b));
    return result;
}

/// ∑ wᵢ(xᵢ - yᵢ)² (lane-wise).
template <simdifiable_multi Vw, simdifiable_multi Vx, simdifiable_multi Vy>
    requires simdify_compatible<Vw, Vx, Vy>
simdified_simd_t<Vw> weighted_vnorm_sq_diff(Vw &&w, Vx &&x, Vy &&y) {
    BATMAT_ASSERT(w.num_batches() == x.num_batches());
    BATMAT_ASSERT(w.num_batches() == y.num_batches());
    simdified_simd_t<Vw> result{};
    for (index_t b = 0; b < w.num_batches(); ++b)
        result += linalg::weighted_vnorm_sq_diff(w.batch(b), x.batch(b), y.batch(b));
    return result;
}

/// @}

} // namespace multi

} // namespace batmat::linalg
