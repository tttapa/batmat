#pragma once

#include <batmat/config.hpp>
#include <guanaqo/assume.hpp>

/// @def BATMAT_ASSUME(x)
/// Invokes undefined behavior if the expression @p x does not evaluate to true.
/// @throws std::logic_error in debug mode (when `NDEBUG` is not defined).

#if defined(NDEBUG) && !BATMAT_VERIFY_ASSUMPTIONS
#define BATMAT_ASSUME(x) GUANAQO_ASSUME(x)
#endif // defined(NDEBUG) && !BATMAT_VERIFY_ASSUMPTIONS

#define BATMAT_ASSERT(x) GUANAQO_ASSERT(x)

#ifndef BATMAT_ASSUME
#define BATMAT_ASSUME(x) BATMAT_ASSERT(x)
#endif
