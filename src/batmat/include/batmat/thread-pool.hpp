#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace batmat {

/// @ingroup topic-utils
class thread_pool {
  private:
    struct State {
        std::mutex mtx;
        std::condition_variable_any cv;
        std::function<void()> func;
        std::exception_ptr exception;

        void run(std::stop_token stop);
    };
    std::vector<State> states;
    std::vector<std::jthread> threads; // must be destroyed first

  public:
    explicit thread_pool(size_t num_threads = std::thread::hardware_concurrency())
        : states(num_threads) {
        threads.reserve(num_threads);
        for (auto &state : states)
            threads.emplace_back([&state](std::stop_token stop) { state.run(std::move(stop)); });
    }

    thread_pool(const thread_pool &)            = delete;
    thread_pool &operator=(const thread_pool &) = delete;
    thread_pool(thread_pool &&)                 = default;
    thread_pool &operator=(thread_pool &&)      = default;

    void schedule(size_t i, std::function<void()> func) {
        auto &state = states[i];
        std::unique_lock lck{state.mtx};
        state.func = std::move(func);
        lck.unlock();
        state.cv.notify_all();
    }

    void wait(size_t i) {
        auto &state = states[i];
        std::unique_lock lck{state.mtx};
        state.cv.wait(lck, [&] { return !state.func; });
        if (auto &e = state.exception)
            std::rethrow_exception(std::exchange(e, nullptr));
    }

    void wait_all() {
        for (size_t i = 0; i < size(); ++i)
            wait(i);
    }

    template <class I = size_t, class F>
    void sync_run_all(F &&f) {
        const auto n = size();
        for (size_t i = 0; i < n; ++i)
            schedule(i, [&f, i, n] { f(static_cast<I>(i), static_cast<I>(n)); });
        wait_all();
    }

    template <class I = size_t, class F>
    void sync_run_n(I n, F &&f) {
        if (static_cast<size_t>(n) > size())
            throw std::invalid_argument("Not enough threads in pool");
        for (size_t i = 0; i < static_cast<size_t>(n); ++i)
            schedule(i, [&f, i, n] { f(static_cast<I>(i), n); });
        for (size_t i = 0; i < static_cast<size_t>(n); ++i)
            wait(i);
    }

    [[nodiscard]] size_t size() const { return threads.size(); }
};

inline void thread_pool::State::run(std::stop_token stop) {
    while (true) {
        std::unique_lock lck{mtx};
        cv.wait(lck, stop, [&] { return static_cast<bool>(func); });
        if (stop.stop_requested()) {
            break;
        } else {
            try {
                func();
            } catch (...) {
                exception = std::current_exception();
            }
            func = nullptr;
            lck.unlock();
            cv.notify_all();
        }
    }
}

namespace detail {
extern std::mutex pool_mtx;
extern std::optional<thread_pool> pool;
} // namespace detail

/// Set the number of threads in the global thread pool.
/// @ingroup topic-utils
/// @deprecated
[[deprecated]] void pool_set_num_threads(size_t num_threads);

/// Run a function on all threads in the global thread pool, synchronously waiting for all threads.
/// @ingroup topic-utils
/// @deprecated
template <class I = size_t, class F>
[[deprecated]] void pool_sync_run_all(F &&f) {
    std::lock_guard<std::mutex> lck(detail::pool_mtx);
    if (!detail::pool)
        return;
    detail::pool->sync_run_all(std::forward<F>(f));
}

/// Run a function on the first @p n threads in the global thread pool, synchronously waiting for
/// those threads. If @p n is greater than the number of threads in the pool, the pool is expanded.
/// @ingroup topic-utils
/// @deprecated
template <class I = size_t, class F>
[[deprecated]] void pool_sync_run_n(I n, F &&f) {
    std::lock_guard<std::mutex> lck(detail::pool_mtx);
    if (!detail::pool || detail::pool->size() < static_cast<size_t>(n))
        detail::pool.emplace(static_cast<size_t>(n));
    detail::pool->sync_run_n(n, std::forward<F>(f));
}

} // namespace batmat
