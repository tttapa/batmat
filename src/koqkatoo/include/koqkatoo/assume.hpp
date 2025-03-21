#pragma once

#include <koqkatoo/config.hpp>
#include <guanaqo/assume.hpp>

/// @def KOQKATOO_ASSUME(x)
/// Invokes undefined behavior if the expression @p x does not evaluate to true.
/// @throws std::logic_error in debug mode (when `NDEBUG` is not defined).

#if defined(NDEBUG) && !KOQKATOO_VERIFY_ASSUMPTIONS
#define KOQKATOO_ASSUME(x) GUANAQO_ASSUME(x)
#endif // defined(NDEBUG) && !KOQKATOO_VERIFY_ASSUMPTIONS

#define KOQKATOO_ASSERT(x) GUANAQO_ASSERT(x)

#ifndef KOQKATOO_ASSUME
#define KOQKATOO_ASSUME(x) KOQKATOO_ASSERT(x)
#endif
