#include <koqkatoo/thread-pool.hpp>

namespace koqkatoo {

std::optional<thread_pool> pool{std::in_place};

void pool_set_num_threads(size_t num_threads) {
    if (pool->size() != num_threads)
        pool.emplace(num_threads);
}

} // namespace koqkatoo
