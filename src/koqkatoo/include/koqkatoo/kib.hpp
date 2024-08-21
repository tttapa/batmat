#pragma once

#include <koqkatoo/config.hpp>

namespace koqkatoo {

constexpr index_t operator""_KiB(unsigned long long i) {
    return static_cast<index_t>(i) * 1024;
}

constexpr index_t operator""_MiB(unsigned long long i) {
    return static_cast<index_t>(i) * 1024_KiB;
}

} // namespace koqkatoo
