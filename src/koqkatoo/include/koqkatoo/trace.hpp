#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <thread>
#include <utility>

#include <koqkatoo/config.hpp>
#include <koqkatoo/stringify.h>

namespace koqkatoo {

struct TraceLogger {
    struct Log {
        const char *name = "";
        std::int64_t instance{0};
        std::chrono::nanoseconds start_time{0};
        std::chrono::nanoseconds duration{0};
        std::size_t thread_id{0};

        friend std::ostream &operator<<(std::ostream &os, const Log &log) {
            return os << '(' << std::quoted(log.name) << ", " << log.instance
                      << ", " << log.start_time.count() << ", "
                      << log.duration.count() << ", " << log.thread_id << ')';
        }
    };

    using clock = std::chrono::steady_clock;

    clock::time_point t0 = clock::now();
    std::vector<Log> logs;
    std::atomic_size_t count{0};

    struct ScopedLog {
        Log *log = nullptr;
        clock::time_point start_time_point;

        ScopedLog() = default;
        ScopedLog(Log *log, clock::time_point start_time_point)
            : log{log}, start_time_point{start_time_point} {}
        ScopedLog(const ScopedLog &)            = delete;
        ScopedLog &operator=(const ScopedLog &) = delete;
        ScopedLog(ScopedLog &&other) noexcept
            : log{std::exchange(other.log, nullptr)},
              start_time_point{other.start_time_point} {}
        ScopedLog &operator=(ScopedLog &&) = delete;
        ~ScopedLog() {
            if (log)
                log->duration = clock::now() - start_time_point;
        }
    };

    TraceLogger(size_t capacity) { logs.resize(capacity); }

    ScopedLog trace(const char *name, std::int64_t instance) {
        size_t index = count.fetch_add(1, std::memory_order_relaxed);
        if (index >= logs.size())
            return ScopedLog{nullptr, {}};
        static constexpr std::hash<std::thread::id> hasher;
        auto &log      = logs[index];
        auto t1        = clock::now();
        log.name       = name;
        log.instance   = instance;
        log.start_time = t1 - t0;
        log.thread_id  = hasher(std::this_thread::get_id());
        return ScopedLog{&log, t1};
    }

    [[nodiscard]] std::span<const Log> get_logs() const {
        auto n = std::min(logs.size(), count.load(std::memory_order_relaxed));
        return std::span{logs}.first(n);
    }

    void reset() { count.store(0, std::memory_order_relaxed); }
};

#if KOQKATOO_WITH_TRACING
extern TraceLogger trace_logger;
#define KOQKATOO_TRACE(name, instance)                                         \
    const auto KOQKATOO_CAT(trace_log_, __COUNTER__) =                         \
        ::koqkatoo::trace_logger.trace(name, instance)
#else
#define KOQKATOO_TRACE(...)                                                    \
    do {                                                                       \
    } while (0)
#endif

} // namespace koqkatoo
