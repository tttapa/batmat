#pragma once

#include <libfork/schedule/lazy_pool.hpp>
#include <optional>

namespace koqkatoo {

extern std::optional<lf::lazy_pool> fork_pool;

void fork_set_num_threads(size_t num_threads);

} // namespace koqkatoo
