#pragma once

#include <batmat/config.hpp>
#include <batmat/export.h>
#include <guanaqo/timed.hpp>
#if BATMAT_WITH_CPU_TIME
#include <guanaqo/timed-cpu.hpp>
#else
#include <chrono>
#endif

namespace batmat {

#if BATMAT_WITH_CPU_TIME // TODO
using DefaultTimings = guanaqo::TimingsCPU;
#else
/// Measures the number of invocations of a specific piece of code and its
/// run time.
/// @todo   Move to guanaqo.
struct BATMAT_EXPORT DefaultTimings {
    int64_t num_invocations{};
    std::chrono::nanoseconds wall_time{};
};

BATMAT_EXPORT std::ostream &operator<<(std::ostream &, DefaultTimings);

#endif

using guanaqo::timed;

} // namespace batmat

#if !BATMAT_WITH_CPU_TIME

namespace guanaqo {

/// RAII class for measuring wall time.
template <>
struct BATMAT_EXPORT Timed<batmat::DefaultTimings> {
    Timed(batmat::DefaultTimings &time) : time(time) { wall_start_time = clock::now(); }
    ~Timed() {
        auto wall_end_time = clock::now();
        ++time.num_invocations;
        time.wall_time += wall_end_time - wall_start_time;
    }
    Timed(const Timed &)            = delete;
    Timed &operator=(const Timed &) = delete;

  private:
    using clock = std::chrono::steady_clock;
    batmat::DefaultTimings &time;
    clock::time_point wall_start_time;
};

} // namespace guanaqo

#endif
