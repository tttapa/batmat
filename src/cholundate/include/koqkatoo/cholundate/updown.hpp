#pragma once

#include <koqkatoo/config.hpp>
#include <span>

namespace koqkatoo::cholundate {

/// Perform a factorization update, i.e. given the factorization of @f$ K @f$,
/// compute the factorization of @f$ K + A A^\top @f$.
struct Update {};
/// Perform a factorization downdate, i.e. given the factorization of @f$ K @f$,
/// compute the factorization of @f$ K - A A^\top @f$.
struct Downdate {};
/// Perform a factorization update or downdate, depending on the given signs,
/// i.e. given the factorization of @f$ K @f$, compute the factorization of
/// @f$ K + A S A^\top @f$, where @f$ S @f$ is a diagonal matrix with
/// @f$ S_{jj} = 1 @f$ if @p signs[j] is `+0.0`, and @f$ S_{jj} = -1 @f$ if
/// @p signs[j] is `-0.0`. Other values for @p signs are not allowed.
struct UpDowndate {
    std::span<const real_t> signs;
};
/// Perform a factorization downdate or update, depending on the given signs,
/// i.e. given the factorization of @f$ K @f$, compute the factorization of
/// @f$ K - A S A^\top @f$, where @f$ S @f$ is a diagonal matrix with
/// @f$ S_{jj} = 1 @f$ if @p signs[j] is `+0.0`, and @f$ S_{jj} = -1 @f$ if
/// @p signs[j] is `-0.0`. Other values for @p signs are not allowed.
struct DownUpdate {
    std::span<const real_t> signs;
};

} // namespace koqkatoo::cholundate
