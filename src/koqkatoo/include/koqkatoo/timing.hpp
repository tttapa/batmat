#pragma once

#include <koqkatoo/config.hpp>
#include <koqkatoo/export.h>
#include <guanaqo/timed.hpp>
#if KOQKATOO_WITH_CPU_TIME
#include <guanaqo/timed-cpu.hpp>
#else
#include <chrono>
#endif

namespace koqkatoo {

#if KOQKATOO_WITH_CPU_TIME
using DefaultTimings = guanaqo::TimingsCPU;
#else
/// Measures the number of invocations of a specific piece of code and its
/// run time.
/// @todo   Move to guanaqo.
struct KOQKATOO_EXPORT DefaultTimings {
    int64_t num_invocations{};
    std::chrono::nanoseconds wall_time{};
};

KOQKATOO_EXPORT std::ostream &operator<<(std::ostream &, DefaultTimings);

#endif

using guanaqo::timed;

} // namespace koqkatoo

#if !KOQKATOO_WITH_CPU_TIME

namespace guanaqo {

/// RAII class for measuring wall and CPU time.
template <>
struct KOQKATOO_EXPORT Timed<koqkatoo::DefaultTimings> {
    Timed(koqkatoo::DefaultTimings &time);
    ~Timed();
    Timed(const Timed &)            = delete;
    Timed &operator=(const Timed &) = delete;

  private:
    using clock = std::chrono::steady_clock;
    koqkatoo::DefaultTimings &time;
    clock::time_point wall_start_time;
};

} // namespace guanaqo

#endif
