#include <batmat/dtypes.hpp>
#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/gemm.hpp>
#include <batmat/linalg/potrf.hpp>
#include <batmat/matrix/matrix.hpp>
#include <guanaqo/demangled-typename.hpp>
#include <guanaqo/print.hpp>
#include <batmat-version.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <print>
#include <random>

using batmat::index_t;
using batmat::real_t;
namespace la = batmat::linalg;

int main() {
    std::println("{:+<79}", "");
    std::println("batmat {} ({}:{})", BATMAT_VERSION, batmat_commit_hash, batmat_build_time);
    std::println("{:+<79}", "");
    std::println("Index type: {}", guanaqo::demangled_typename(typeid(batmat::index_t)));
    std::println("Default dtype: {}", guanaqo::demangled_typename(typeid(batmat::real_t)));
    std::println("Supported dtypes and vector lengths:");
    batmat::types::foreach_dtype_vl([]<class T>(T) {
        std::println(" - {} [{}]", guanaqo::demangled_typename(typeid(typename T::dtype)), T::vl);
    });
    std::println("{:+<79}", "");
    // Select an appropriate vector length.
    constexpr auto v = batmat::types::vl_at_most<real_t, 4>;
    static_assert(v != 0, "No suitable vector length for real_t");
    using batch_size             = std::integral_constant<index_t, 4>;
    constexpr auto storage_order = batmat::matrix::StorageOrder::ColMajor;
    // Class representing a batch of four matrices.
    using Mat = batmat::matrix::Matrix<real_t, index_t, batch_size, batch_size, storage_order>;
    // Allocate some batches of matrices (initialized to zero).
    index_t n = 3, m = n + 5;
    Mat C{{.rows = n, .cols = n}}, A{{.rows = n, .cols = m}};
    // Fill A with random values.
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1.0, 1.0};
    std::ranges::generate(A, [&] { return uni(rng); });
    // Compute C = AAᵀ to make it symmetric positive definite (lower triangular part only).
    la::syrk(A, la::tril(C));
    // Allocate L for the Cholesky factors.
    Mat L{{.rows = n, .cols = n}, batmat::matrix::uninitialized};
    // Compute the Cholesky factors L of C (lower triangular).
    la::fill(0, la::triu(L));
    la::potrf(la::tril(C), la::tril(L));
    // Print the results.
    for (index_t l = 0; l < C.depth(); ++l) {
        guanaqo::print_python(std::cout << "C[" << l << "] =\n", C(l));
        guanaqo::print_python(std::cout << "L[" << l << "] =\n", L(l));
    }
    // Compute LLᵀ (in-place).
    la::syrk(la::tril(L));
    // Check that LLᵀ == C.
    int errors     = 0;
    const auto eps = std::numeric_limits<real_t>::epsilon();
    for (index_t l = 0; l < C.depth(); ++l)
        for (index_t c = 0; c < C.cols(); ++c)
            for (index_t r = c; r < C.rows(); ++r)
                errors += std::abs(C(l, r, c) - L(l, r, c)) < 10 * eps ? 0 : 1;
    return errors;
}
