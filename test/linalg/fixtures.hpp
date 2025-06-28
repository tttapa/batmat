#pragma once

#include <gtest/gtest.h>

#include <batmat/config.hpp>
#include <batmat/matrix/matrix.hpp>
#include <guanaqo/eigen/view.hpp>

#include <Eigen/Core>
#include <random>

namespace batmat::tests {
using matrix::StorageOrder;

template <class Config>
class LinalgTest : public ::testing::Test {
  protected:
    using value_type = typename Config::value_type;
    using batch_size = typename Config::batch_size;

    template <StorageOrder O>
    using Matrix = batmat::matrix::Matrix<value_type, index_t, batch_size, batch_size, O>;
    using EMat   = Eigen::MatrixX<value_type>;

    std::mt19937 rng{12345};
    std::uniform_real_distribution<value_type> uni{-1, 1};
    void SetUp() override { rng.seed(12345); }

    template <StorageOrder O>
    auto get_matrix(index_t r, index_t c) {
        Matrix<O> a{{.rows = r, .cols = c}};
        std::ranges::generate(a, [this] { return uni(rng); });
        return a;
    }

    template <int O>
    auto get_matrix(index_t r, index_t c) {
        return get_matrix<Config::orders[O]>(r, c);
    }

    auto get_sparse_vector(index_t r, double sparsity = 0.5) {
        Matrix<StorageOrder::ColMajor> a{{.rows = r, .cols = 1}};
        std::bernoulli_distribution brnl{sparsity};
        std::ranges::generate(a, [this, &brnl] { return brnl(rng) ? value_type{} : uni(rng); });
        return a;
    }

    static void check(auto eval_ref, auto check_res, auto &&res, auto &&...args) {
        for (index_t l = 0; l < res.depth(); ++l) {
            EMat ref = eval_ref(as_eigen(args(l))...);
            check_res(l, as_eigen(res(l)), ref, as_eigen(args(l))...);
        }
    }

    const value_type tolerance =
        std::pow(std::numeric_limits<value_type>::epsilon(), value_type(0.55));
};

} // namespace batmat::tests
