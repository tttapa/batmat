#include <koqkatoo/fork-pool.hpp>

namespace koqkatoo {

std::optional<lf::lazy_pool> fork_pool{std::in_place};

void fork_set_num_threads(size_t num_threads) {
    fork_pool.emplace(num_threads);
}

} // namespace koqkatoo
