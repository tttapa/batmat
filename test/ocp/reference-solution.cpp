#include "reference-solution.hpp"

#include <koqkatoo/ocp/conversion.hpp>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <guanaqo/eigen/span.hpp>

namespace koqkatoo::ocp::testing {

KKTMatrices reference_qp(const LinearOCPStorage &ocp, real_t S,
                         Eigen::Ref<const Eigen::VectorX<real_t>> Σ,
                         Eigen::Ref<const Eigen::VectorX<bool>> J) {
    index_t n_constr = ocp.num_constraints(),
            n_dyn    = ocp.num_dynamics_constraints();
    // Build quadratic program and standard KKT system for the OCP.
    auto qp   = LinearOCPSparseQP::build(ocp);
    auto kkt  = qp.build_kkt(S, guanaqo::as_span(Σ), guanaqo::as_span(J));
    auto &qpA = qp.A_sparsity, &qpQ = qp.Q_sparsity, &qpK = kkt.sparsity;
    Eigen::Map<const SpMat> Q_vw(qpQ.rows, qpQ.cols, qpQ.nnz(),
                                 qpQ.outer_ptr.data(), qpQ.inner_idx.data(),
                                 qp.Q_values.data(), nullptr);
    KKTMatrices r{.Q = Q_vw,
                  .G{n_constr, qpA.cols},
                  .M{n_dyn, qpA.cols},
                  .K{qpK.rows, qpK.cols}};
    std::vector<Eigen::Triplet<real_t>> triplets_G, triplets_M, triplets_K;
    for (index_t c = 0; c < qpA.cols; ++c)
        for (index_t i = qpA.outer_ptr[c]; i < qpA.outer_ptr[c + 1]; ++i)
            if (index_t r = qpA.inner_idx[i]; r < n_dyn) // top rows
                triplets_M.emplace_back(r, c, qp.A_values[i]);
            else // bottom rows
                triplets_G.emplace_back(r - n_dyn, c, qp.A_values[i]);
    for (index_t c = 0; c < qpK.cols; ++c)
        for (index_t i = qpK.outer_ptr[c]; i < qpK.outer_ptr[c + 1]; ++i) {
            index_t r = qpK.inner_idx[i];
            if (r >= c)
                triplets_K.emplace_back(r, c, kkt.values[i]);
            if (r > c)
                triplets_K.emplace_back(c, r, kkt.values[i]);
        }
    r.G.setFromTriplets(triplets_G.begin(), triplets_G.end());
    r.M.setFromTriplets(triplets_M.begin(), triplets_M.end());
    r.K.setFromTriplets(triplets_K.begin(), triplets_K.end());
    return r;
}

real_t cond_sparse_sym(Eigen::Ref<const SpMat> K,
                       const Eigen::SparseLU<SpMat> *luK) {
    Spectra::SparseSymMatProd<real_t, Eigen::Lower, 0, index_t> op(K);
    Spectra::SymEigsSolver eigs{op, 1, 10};
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn);
    if (eigs.info() != Spectra::CompInfo::Successful)
        throw std::runtime_error("Largest eigenvalue failed to converge");
    real_t λ_min, λ_max = eigs.eigenvalues()[0];
    if (luK) {
        struct InvKOp {
            std::remove_pointer_t<decltype(luK)> &lu;
            using Scalar = real_t;
            [[nodiscard]] index_t rows() const { return lu.rows(); }
            [[nodiscard]] index_t cols() const { return lu.cols(); }
            void perform_op(const Scalar *x_in, Scalar *x_out) const {
                Eigen::Map<const Eigen::VectorX<real_t>> xi{x_in, cols()};
                Eigen::Map<Eigen::VectorX<real_t>> xo{x_out, rows()};
                xo = lu.solve(xi);
            }
        };
        InvKOp op{*luK};
        Spectra::SymEigsSolver eigs{op, 1, 10};
        eigs.init();
        eigs.compute(Spectra::SortRule::LargestMagn);
        if (eigs.info() != Spectra::CompInfo::Successful)
            throw std::runtime_error("Smallest eigenvalue failed to converge");
        λ_min = 1 / eigs.eigenvalues()[0];
    } else {
        eigs.init();
        eigs.compute(Spectra::SortRule::SmallestMagn);
        if (eigs.info() != Spectra::CompInfo::Successful)
            throw std::runtime_error("Smallest eigenvalue failed to converge");
        λ_min = eigs.eigenvalues()[0];
    }
    using std::abs;
    return abs(λ_max) / abs(λ_min);
}

} // namespace koqkatoo::ocp::testing
