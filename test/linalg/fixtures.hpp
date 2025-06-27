#pragma once

#include <gtest/gtest.h>

#include <batmat/config.hpp>
#include <batmat/matrix/matrix.hpp>

#include <random>

namespace batmat::tests {
using matrix::StorageOrder;

template <class T, index_t N, StorageOrder... Orders>
struct TestConfig {
    using value_type = T;
    using batch_size = std::integral_constant<index_t, N>;
    static constexpr StorageOrder orders[sizeof...(Orders)]{Orders...};
};

template <class Config>
class LinalgTest : public ::testing::Test {
  protected:
    using value_type = typename Config::value_type;
    using batch_size = typename Config::batch_size;

    template <StorageOrder O>
    using Matrix = batmat::matrix::Matrix<value_type, index_t, batch_size, batch_size, O>;

    std::mt19937 rng{12345};
    std::normal_distribution<value_type> nrml{0, 1};
    void SetUp() override { rng.seed(12345); }

    template <StorageOrder O>
    auto get_matrix(index_t r, index_t c) {
        Matrix<O> a{{.rows = r, .cols = c}};
        std::ranges::generate(a, [this] { return nrml(rng); });
        return a;
    }

    template <int O>
    auto get_matrix(index_t r, index_t c) {
        return get_matrix<Config::orders[O]>(r, c);
    }

    auto get_sparse_vector(index_t r, double sparsity = 0.5) {
        Matrix<StorageOrder::ColMajor> a{{.rows = r, .cols = 1}};
        std::bernoulli_distribution brnl{sparsity};
        std::ranges::generate(a, [this, &brnl] { return brnl(rng) ? value_type{} : nrml(rng); });
        return a;
    }
};

} // namespace batmat::tests
