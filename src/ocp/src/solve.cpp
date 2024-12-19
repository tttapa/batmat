#include <koqkatoo/ocp/solver/solve.hpp>

#include <koqkatoo/ocp/solver/mat-vec.tpp>
#include <koqkatoo/ocp/solver/solve.tpp>
#include <koqkatoo/ocp/solver/storage.tpp>
#include <koqkatoo/ocp/solver/updowndate.tpp>
#if KOQKATOO_WITH_TBB
#include <koqkatoo/ocp/solver/solve-tbb-coarse.tpp>
// #include <koqkatoo/ocp/solver/solve-tbb.tpp>
#endif
#if KOQKATOO_WITH_LIBFORK
#include <koqkatoo/ocp/solver/solve-fork.tpp>
#endif
#include <koqkatoo/ocp/solver/solve-omp.tpp>

namespace koqkatoo::ocp {

#define INSTANTIATE(...)                                                       \
    template struct SolverStorage<__VA_ARGS__>;                                \
    template struct Solver<__VA_ARGS__>

INSTANTIATE(stdx::simd_abi::scalar);
INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 16>);
INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 8>);
INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 4>);
INSTANTIATE(stdx::simd_abi::deduce_t<real_t, 2>);

} // namespace koqkatoo::ocp
