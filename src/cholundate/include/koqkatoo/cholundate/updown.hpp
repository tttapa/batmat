#pragma once

#include <koqkatoo/config.hpp>
#include <span>

namespace koqkatoo::cholundate {

struct Update {};
struct Downdate {};
struct UpDowndate {
    std::span<const real_t> signs;
};
struct DownUpdate {
    std::span<const real_t> signs;
};

} // namespace koqkatoo::cholundate
