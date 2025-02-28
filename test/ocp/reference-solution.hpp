#pragma once

#include <koqkatoo/ocp/ocp.hpp>
#include <Eigen/SparseLU>

namespace koqkatoo::ocp::testing {

using SpMat = Eigen::SparseMatrix<real_t, 0, index_t>;

struct KKTMatrices {
    SpMat Q, G, M, K;
};

KKTMatrices reference_qp(const LinearOCPStorage &ocp, real_t S,
                         Eigen::Ref<const Eigen::VectorX<real_t>> Σ,
                         Eigen::Ref<const Eigen::VectorX<bool>> J);

real_t cond_sparse_sym(Eigen::Ref<const SpMat> K,
                       const Eigen::SparseLU<SpMat> *luK = nullptr);

} // namespace koqkatoo::ocp::testing