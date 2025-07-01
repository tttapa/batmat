#include <cyclocp/cyclocp.hpp>
#include <cyclocp/implementation/data.tpp>
#include <cyclocp/implementation/factor.tpp>
#include <cyclocp/implementation/indexing.tpp>
#include <cyclocp/implementation/mat-vec.tpp>
#include <cyclocp/implementation/pcg.tpp>
#include <cyclocp/implementation/solve.tpp>
#include <cyclocp/implementation/sparse.tpp>
#include <cyclocp/implementation/update.tpp>

namespace cyclocp::ocp::cyclocp {

template class CyclicOCPSolver<1>;
template class CyclicOCPSolver<4>;
template class CyclicOCPSolver<8>;

} // namespace cyclocp::ocp::cyclocp
