#pragma once

/// @file
/// Vector reductions.
/// @ingroup topic-utilities

#include <batmat/config.hpp>
#include <batmat/simd.hpp>
#include <cmath>

namespace batmat::linalg {

/// @addtogroup topic-utilities
/// @{

/// Utilities for computing vector norms.
/// @tparam T Scalar type.
/// @tparam simd SIMD type. Void for scalar-only.
template <class T, class simd = void>
struct norms;

template <class T>
struct norms<T>;

template <class T, class simd>
struct norms : norms<T> {
    /// Accumulator.
    using result = typename norms<T>::result;
    /// Lane-wise accumulators.
    struct result_simd {
        simd amax;
        simd asum;
        simd sumsq;

        /// ℓ₁ norm.
        [[nodiscard]] simd norm_1() const { return asum; }
        /// ℓ₂ norm.
        [[nodiscard]] simd norm_2() const {
            using std::sqrt;
            return sqrt(sumsq);
        }
        /// max-norm.
        [[nodiscard]] simd norm_inf() const {
            using std::isfinite;
            return datapar::select(isfinite(asum), amax, asum);
        }
    };

    using norms<T>::operator();

    /// Update the accumulator with a new value.
    result_simd operator()(result_simd accum, simd t) const {
        using std::abs;
        using std::max;
        auto at = abs(t);
        return {.amax = max(at, accum.amax), .asum = at + accum.asum, .sumsq = t * t + accum.sumsq};
    }

    /// Combine two accumulators.
    result_simd operator()(result_simd a, result_simd b) const {
        using std::max;
        return {.amax = max(a.amax, b.amax), .asum = a.asum + b.asum, .sumsq = a.sumsq + b.sumsq};
    }

    /// Reduce the SIMD accumulator to a scalar result.
    result operator()(result_simd accum) const {
        using batmat::datapar::hmax;
        return {hmax(accum.amax), reduce(accum.asum), reduce(accum.sumsq)};
    }

    using norms<T>::zero;
    static result_simd zero_simd() { return {}; }
};

template <class T>
struct norms<T, void> {
    /// Accumulator.
    struct result {
        T amax; ///< Maximum absolute value (ignoring NaNs).
        T asum;
        T sumsq;

        /// ℓ₁ norm.
        [[nodiscard]] T norm_1() const { return asum; }
        /// ℓ₂ norm.
        [[nodiscard]] T norm_2() const {
            using std::sqrt;
            return sqrt(sumsq);
        }
        /// max-norm.
        [[nodiscard]] T norm_inf() const {
            using std::isfinite;
            return isfinite(asum) ? amax : asum;
        }
    };

    /// Update the accumulator with a new value.
    result operator()(result accum, T t) const {
        using std::abs;
        using std::max;
        auto at = abs(t);
        return {.amax = max(at, accum.amax), .asum = at + accum.asum, .sumsq = t * t + accum.sumsq};
    }

    /// Combine two accumulators.
    result operator()(result accum, result t) const {
        using std::max;
        return {max(accum.amax, t.amax), accum.asum + t.asum, accum.sumsq + t.sumsq};
    }

    /// Identity element for the reduction.
    static result zero() { return {}; }
};

/// @}

} // namespace batmat::linalg
