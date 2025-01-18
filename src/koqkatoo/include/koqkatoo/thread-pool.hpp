#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace koqkatoo {

class thread_pool {
  private:
    struct Signals {
        std::mutex mtx;
        std::condition_variable_any cv;
    };
    std::vector<Signals> signals;
    std::vector<std::function<void()>> funcs;
    std::vector<std::exception_ptr> exceptions;
    std::vector<std::jthread> threads; // must be destroyed first

    void work(std::stop_token stop, size_t i) {
        auto &sig = signals[i];
        while (true) {
            std::unique_lock lck{sig.mtx};
            sig.cv.wait(lck, stop, [&] { return static_cast<bool>(funcs[i]); });
            if (stop.stop_requested()) {
                break;
            } else {
                try {
                    funcs[i]();
                } catch (...) {
                    exceptions[i] = std::current_exception();
                }
                funcs[i] = nullptr;
                lck.unlock();
                sig.cv.notify_all();
            }
        }
    }

  public:
    explicit thread_pool(
        size_t num_threads = std::thread::hardware_concurrency())
        : signals(num_threads), funcs(num_threads), exceptions(num_threads) {
        threads.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i)
            threads.emplace_back(&thread_pool::work, this, i);
    }

    thread_pool(const thread_pool &)            = delete;
    thread_pool &operator=(const thread_pool &) = delete;
    thread_pool(thread_pool &&)                 = default;
    thread_pool &operator=(thread_pool &&)      = default;

    void schedule(size_t i, std::function<void()> func) {
        auto &sig = signals[i];
        std::unique_lock lck{sig.mtx};
        funcs[i] = std::move(func);
        lck.unlock();
        sig.cv.notify_all();
    }

    void wait(size_t i) {
        auto &sig = signals[i];
        std::unique_lock lck{sig.mtx};
        sig.cv.wait(lck, [&] { return !funcs[i]; });
        if (auto &e = exceptions[i])
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
            schedule(i,
                     [&f, i, n] { f(static_cast<I>(i), static_cast<I>(n)); });
        wait_all();
    }

    [[nodiscard]] size_t size() const { return threads.size(); }
};

extern std::optional<thread_pool> pool;

void pool_set_num_threads(size_t num_threads);

} // namespace koqkatoo
