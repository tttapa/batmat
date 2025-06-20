#include <batmat/openmp.h>
#include <batmat/thread-pool.hpp>

namespace batmat {

std::optional<thread_pool> pool{
    BATMAT_OMP_IF_ELSE(std::nullopt, std::in_place),
};

void pool_set_num_threads(size_t num_threads) {
    if (!pool)
        return;
    if (pool->size() != num_threads)
        pool.emplace(num_threads);
}

} // namespace batmat
