#include <koqkatoo/ocp/cyclic-solver/cyclic-solver.hpp>

#include <koqkatoo/ocp/cyclic-solver/factor.tpp>
#include <koqkatoo/ocp/cyclic-solver/mat-vec.tpp>
#include <koqkatoo/ocp/cyclic-solver/packing.tpp>
#include <koqkatoo/ocp/cyclic-solver/pcg.tpp>

namespace koqkatoo::ocp {

template struct CyclicOCPSolver<stdx::simd_abi::scalar>;
template struct CyclicOCPSolver<stdx::simd_abi::deduce_t<real_t, 16>>;
template struct CyclicOCPSolver<stdx::simd_abi::deduce_t<real_t, 8>>;
template struct CyclicOCPSolver<stdx::simd_abi::deduce_t<real_t, 4>>;
template struct CyclicOCPSolver<stdx::simd_abi::deduce_t<real_t, 2>>;

} // namespace koqkatoo::ocp
