#include <koqkatoo/timing.hpp>
#include <iomanip>

#if !KOQKATOO_WITH_CPU_TIME

namespace koqkatoo {

std::ostream &operator<<(std::ostream &os, DefaultTimings t) {
    using millis_f64 = std::chrono::duration<double, std::milli>;
    auto wall_ms     = millis_f64(t.wall_time).count();
    auto prec        = os.precision(6);
    os << std::setw(8) << wall_ms << " ms (wall) ─ "            //
       << std::setw(8) << "?" << " ms (CPU) ─ "                 //
       << std::setprecision(5) << std::setw(7) << "?" << "% ─ " //
       << std::setw(8) << t.num_invocations << " calls";        //
    os.precision(prec);
    return os;
}

} // namespace koqkatoo

namespace guanaqo {

Timed<koqkatoo::DefaultTimings>::Timed(koqkatoo::DefaultTimings &time)
    : time(time) {
    wall_start_time = clock::now();
}

Timed<koqkatoo::DefaultTimings>::~Timed() {
    auto wall_end_time = clock::now();
    ++time.num_invocations;
    time.wall_time += wall_end_time - wall_start_time;
}

} // namespace guanaqo

#endif
