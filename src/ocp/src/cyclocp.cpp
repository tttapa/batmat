#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/ocp/cyclocp/data.tpp>
#include <koqkatoo/ocp/cyclocp/factor.tpp>
#include <koqkatoo/ocp/cyclocp/indexing.tpp>
// #include <koqkatoo/ocp/cyclocp/mat-vec.tpp> // TODO
#include <koqkatoo/ocp/cyclocp/pcg.tpp>
#include <koqkatoo/ocp/cyclocp/solve.tpp>
#include <koqkatoo/ocp/cyclocp/sparse.tpp>
#include <koqkatoo/ocp/cyclocp/update.tpp>

namespace koqkatoo::ocp::cyclocp {

template struct CyclicOCPSolver<16>;
template struct CyclicOCPSolver<8>;
template struct CyclicOCPSolver<4>;
template struct CyclicOCPSolver<2>;
template struct CyclicOCPSolver<1>;

} // namespace koqkatoo::ocp::cyclocp
