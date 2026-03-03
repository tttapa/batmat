#pragma once

#include <batmat/config.hpp>

namespace batmat::literals {
constexpr real_t operator""_r(long double x) { return static_cast<real_t>(x); }
} // namespace batmat::literals
