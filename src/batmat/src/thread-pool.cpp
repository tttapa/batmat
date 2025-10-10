#include <batmat/openmp.h>
#include <batmat/thread-pool.hpp>

namespace batmat {

std::mutex detail::pool_mtx;
std::optional<thread_pool> detail::pool{
    BATMAT_OMP_IF_ELSE(std::nullopt, std::in_place),
};

void pool_set_num_threads(size_t num_threads) {
    std::lock_guard<std::mutex> lck(detail::pool_mtx);
    if (!detail::pool || detail::pool->size() != num_threads)
        detail::pool.emplace(num_threads);
}

} // namespace batmat
