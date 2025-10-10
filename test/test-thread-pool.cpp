#include <batmat/thread-pool.hpp>
#include <gtest/gtest.h>
#include <atomic>

TEST(ThreadPool, syncN) {
    std::atomic<int> counter{};
    batmat::pool_sync_run_n(4, [&](int i, int n) {
        ASSERT_EQ(n, 4);
        counter.fetch_add(1 + i, std::memory_order_relaxed);
    });
    batmat::pool_sync_run_n(2, [&](int i, int n) {
        ASSERT_EQ(n, 2);
        counter.fetch_add(1 + i, std::memory_order_relaxed);
    });
    ASSERT_EQ(counter.load(std::memory_order_relaxed), 1 + 2 + 3 + 4 + 1 + 2);
}

TEST(ThreadPool, syncNlarge) {
    std::atomic<int> counter{};
    batmat::pool_sync_run_n(100, [&](int i, int n) {
        ASSERT_EQ(n, 100);
        counter.fetch_add(1 + i, std::memory_order_relaxed);
    });
    batmat::pool_sync_run_n(2, [&](int i, int n) {
        ASSERT_EQ(n, 2);
        counter.fetch_add(1 + i, std::memory_order_relaxed);
    });
    ASSERT_EQ(counter.load(std::memory_order_relaxed), 5050 + 1 + 2);
}
