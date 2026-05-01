#include <batmat/linalg/elementwise.hpp>
#include <batmat/matrix/matrix.hpp>
#include <batmat/simd.hpp>

#include <gtest/gtest.h>
#include <limits>
#include <random>

TEST(linalg, scale) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> b{{.rows = n, .cols = 1}};
    linalg::scale(3.14, a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(3.14 * a(j, i, 0), b(j, i, 0), eps) << i << ", " << j;
}

TEST(linalg, vscale) {
    using namespace batmat;

    const index_t n            = 43;
    using v_t                  = index_constant<4>;
    using abi                  = datapar::deduced_abi<real_t, v_t{}()>;
    constexpr size_t alignment = datapar::simd_align<real_t, abi>::value;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    alignas(alignment) real_t alpha_data[v_t()] = {3.14, 4.15, 5.16, 6.17};
    auto alpha = datapar::aligned_load<datapar::simd<real_t, abi>>(alpha_data);
    matrix::Matrix<real_t, index_t, v_t, v_t> b{{.rows = n, .cols = 1}};
    linalg::scale(alpha, a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(alpha_data[j] * a(j, i, 0), b(j, i, 0), eps) << i << ", " << j;
}

TEST(linalg, hadamard) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> b{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });
    std::ranges::generate(b, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> c{{.rows = n, .cols = 1}};
    linalg::hadamard(a, b, c);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(a(j, i, 0) * b(j, i, 0), c(j, i, 0), eps) << i << ", " << j;
}

TEST(linalg, hadamard_inplace) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> b{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });
    std::ranges::generate(b, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> a2 = a;
    linalg::hadamard(a2, b);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(a(j, i, 0) * b(j, i, 0), a2(j, i, 0), eps) << i << ", " << j;
}

TEST(linalg, clamp_scalar) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-4, 4};
    std::ranges::generate(a, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> z{{.rows = n, .cols = 1}};
    linalg::clamp(a, real_t{-1.0}, real_t{1.0}, z);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j) {
            real_t expect = std::max(real_t{-1.0}, std::min(a(j, i, 0), real_t{1.0}));
            EXPECT_NEAR(expect, z(j, i, 0), eps) << i << ", " << j;
        }
}

TEST(linalg, clamp_perlane_matrix) {
    using namespace batmat;

    const index_t n            = 43;
    using v_t                  = index_constant<4>;
    using abi                  = datapar::deduced_abi<real_t, v_t{}()>;
    constexpr size_t alignment = datapar::simd_align<real_t, abi>::value;

    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-4, 4};
    std::ranges::generate(a, [&] { return dist(rng); });

    alignas(alignment) real_t lo_data[v_t()] = {-0.5, -1.0, -2.0, -3.0};
    alignas(alignment) real_t hi_data[v_t()] = {0.5, 1.0, 2.0, 3.0};
    matrix::Matrix<real_t, index_t, v_t, v_t> lo_mat{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> hi_mat{{.rows = n, .cols = 1}};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j) {
            lo_mat(j, i, 0) = lo_data[j];
            hi_mat(j, i, 0) = hi_data[j];
        }
    matrix::Matrix<real_t, index_t, v_t, v_t> z2{{.rows = n, .cols = 1}};
    linalg::clamp(a, lo_mat, hi_mat, z2);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j) {
            real_t expect = std::max(lo_data[j], std::min(a(j, i, 0), hi_data[j]));
            EXPECT_NEAR(expect, z2(j, i, 0), eps) << i << ", " << j;
        }
}

TEST(linalg, clamp_resid) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> lo{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> hi{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-4, 4};
    std::ranges::generate(a, [&] { return dist(rng); });
    // Use controlled lo/hi so expectations are deterministic and within sensible ranges
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j) {
            lo(j, i, 0) = -1.0;
            hi(j, i, 0) = 1.0;
        }

    matrix::Matrix<real_t, index_t, v_t, v_t> z{{.rows = n, .cols = 1}};
    linalg::clamp_resid(a, lo, hi, z);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j) {
            real_t expect = a(j, i, 0) - std::max(lo(j, i, 0), std::min(a(j, i, 0), hi(j, i, 0)));
            EXPECT_NEAR(expect, z(j, i, 0), eps) << i << ", " << j;
        }
}

TEST(linalg, axpby_scalar) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> x{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> y{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(x, [&] { return dist(rng); });
    std::ranges::generate(y, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> z{{.rows = n, .cols = 1}};
    linalg::axpby(real_t{2.0}, x, real_t{-1.5}, y, z);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(2.0 * x(j, i, 0) - 1.5 * y(j, i, 0), z(j, i, 0), eps) << i << ", " << j;
}

TEST(linalg, axpby_vector) {
    using namespace batmat;

    const index_t n            = 43;
    using v_t                  = index_constant<4>;
    using abi                  = datapar::deduced_abi<real_t, v_t{}()>;
    constexpr size_t alignment = datapar::simd_align<real_t, abi>::value;

    matrix::Matrix<real_t, index_t, v_t, v_t> x{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> y{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(x, [&] { return dist(rng); });
    std::ranges::generate(y, [&] { return dist(rng); });

    alignas(alignment) real_t a_data[v_t()] = {1.1, 2.2, 3.3, 4.4};
    alignas(alignment) real_t b_data[v_t()] = {-0.5, -1.0, -1.5, -2.0};
    auto a_simd = datapar::aligned_load<datapar::simd<real_t, abi>>(a_data);
    auto b_simd = datapar::aligned_load<datapar::simd<real_t, abi>>(b_data);
    matrix::Matrix<real_t, index_t, v_t, v_t> z2{{.rows = n, .cols = 1}};
    linalg::axpby(a_simd, x, b_simd, y, z2);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(a_data[j] * x(j, i, 0) + b_data[j] * y(j, i, 0), z2(j, i, 0), eps)
                << i << ", " << j;
}

TEST(linalg, axpy_inplace) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> x{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> y{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(x, [&] { return dist(rng); });
    std::ranges::generate(y, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> y2 = y;
    linalg::axpy(real_t{0.5}, x, y2);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(0.5 * x(j, i, 0) + y(j, i, 0), y2(j, i, 0), eps) << i << ", " << j;
}

TEST(linalg, negate) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> A{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-3, 3};
    std::ranges::generate(A, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> C{{.rows = n, .cols = 1}};
    linalg::negate(A, C);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(-A(j, i, 0), C(j, i, 0), eps) << i << ", " << j;
}

TEST(linalg, add) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> A{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> B{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-3, 3};
    std::ranges::generate(A, [&] { return dist(rng); });
    std::ranges::generate(B, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> D{{.rows = n, .cols = 1}};
    linalg::add(A, B, D);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(A(j, i, 0) + B(j, i, 0), D(j, i, 0), eps) << i << ", " << j;
}

TEST(linalg, sub) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> A{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> B{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-3, 3};
    std::ranges::generate(A, [&] { return dist(rng); });
    std::ranges::generate(B, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> E{{.rows = n, .cols = 1}};
    linalg::sub(A, B, E);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(A(j, i, 0) - B(j, i, 0), E(j, i, 0), eps) << i << ", " << j;
}

TEST(linalg, transform_elementwise) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> A{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-3, 3};
    std::ranges::generate(A, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> Z{{.rows = n, .cols = 1}};
    linalg::transform_elementwise([](auto xi) { return xi * xi; }, Z, A);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            EXPECT_NEAR(A(j, i, 0) * A(j, i, 0), Z(j, i, 0), eps) << i << ", " << j;
}

TEST(linalg, transform2_elementwise) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> A{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> B{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-3, 3};
    std::ranges::generate(A, [&] { return dist(rng); });
    std::ranges::generate(B, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t, v_t> C1{{.rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t, v_t> E1{{.rows = n, .cols = 1}};
    linalg::transform2_elementwise(
        [](auto ai, auto bi) { return std::make_tuple(ai + bi, ai - bi); }, C1, E1, A, B);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j) {
            EXPECT_NEAR(A(j, i, 0) + B(j, i, 0), C1(j, i, 0), eps) << i << ", " << j;
            EXPECT_NEAR(A(j, i, 0) - B(j, i, 0), E1(j, i, 0), eps) << i << ", " << j;
        }
}

// Multi-batch elementwise tests

TEST(linalg, scale_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t> b{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    linalg::scale(real_t{2.5}, a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t d = 0; d < 3 * v_t(); ++d)
        for (index_t i = 0; i < n; ++i)
            EXPECT_NEAR(2.5 * a(d, i, 0), b(d, i, 0), eps) << d << ", " << i;
}

TEST(linalg, hadamard_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t> b{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });
    std::ranges::generate(b, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t> c{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    linalg::hadamard(a, b, c);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t d = 0; d < 3 * v_t(); ++d)
        for (index_t i = 0; i < n; ++i)
            EXPECT_NEAR(a(d, i, 0) * b(d, i, 0), c(d, i, 0), eps) << d << ", " << i;
}

TEST(linalg, clamp_scalar_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-4, 4};
    std::ranges::generate(a, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t> z{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    linalg::clamp(a, real_t{-1.0}, real_t{1.0}, z);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t d = 0; d < 3 * v_t(); ++d)
        for (index_t i = 0; i < n; ++i) {
            real_t expect = std::max(real_t{-1.0}, std::min(a(d, i, 0), real_t{1.0}));
            EXPECT_NEAR(expect, z(d, i, 0), eps) << d << ", " << i;
        }
}

TEST(linalg, axpby_scalar_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> x{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t> y{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(x, [&] { return dist(rng); });
    std::ranges::generate(y, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t> z{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    linalg::axpby(real_t{2.0}, x, real_t{-1.5}, y, z);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t d = 0; d < 3 * v_t(); ++d)
        for (index_t i = 0; i < n; ++i)
            EXPECT_NEAR(2.0 * x(d, i, 0) - 1.5 * y(d, i, 0), z(d, i, 0), eps) << d << ", " << i;
}

TEST(linalg, axpy_inplace_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> x{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t> y{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(x, [&] { return dist(rng); });
    std::ranges::generate(y, [&] { return dist(rng); });

    matrix::Matrix<real_t, index_t, v_t> y2 = y;
    linalg::axpy(real_t{0.5}, x, y2);
    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t d = 0; d < 3 * v_t(); ++d)
        for (index_t i = 0; i < n; ++i)
            EXPECT_NEAR(0.5 * x(d, i, 0) + y(d, i, 0), y2(d, i, 0), eps) << d << ", " << i;
}

TEST(linalg, scale_multi_simd) {
    using namespace batmat;

    const index_t n            = 43;
    using v_t                  = index_constant<4>;
    using abi                  = datapar::deduced_abi<real_t, v_t{}()>;
    constexpr size_t alignment = datapar::simd_align<real_t, abi>::value;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    alignas(alignment) real_t alpha_data[v_t()] = {3.14, 4.15, 5.16, 6.17};
    auto alpha = datapar::aligned_load<datapar::simd<real_t, abi>>(alpha_data);

    matrix::Matrix<real_t, index_t, v_t> b{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    linalg::scale(alpha, a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t d = 0; d < 3 * v_t(); ++d)
        for (index_t i = 0; i < n; ++i)
            EXPECT_NEAR(alpha_data[d % v_t()] * a(d, i, 0), b(d, i, 0), eps) << d << ", " << i;
}

TEST(linalg, clamp_scalar_multi_simd) {
    using namespace batmat;

    const index_t n            = 43;
    using v_t                  = index_constant<4>;
    using abi                  = datapar::deduced_abi<real_t, v_t{}()>;
    constexpr size_t alignment = datapar::simd_align<real_t, abi>::value;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-4, 4};
    std::ranges::generate(a, [&] { return dist(rng); });

    alignas(alignment) real_t lo_data[v_t()] = {-1.5, -1.0, -0.5, -2.0};
    alignas(alignment) real_t hi_data[v_t()] = {1.5, 1.0, 0.5, 2.0};
    auto lo = datapar::aligned_load<datapar::simd<real_t, abi>>(lo_data);
    auto hi = datapar::aligned_load<datapar::simd<real_t, abi>>(hi_data);

    matrix::Matrix<real_t, index_t, v_t> z{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    linalg::clamp(a, lo, hi, z);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t d = 0; d < 3 * v_t(); ++d)
        for (index_t i = 0; i < n; ++i) {
            real_t expect = std::max(lo_data[d % v_t()], std::min(a(d, i, 0), hi_data[d % v_t()]));
            EXPECT_NEAR(expect, z(d, i, 0), eps) << d << ", " << i;
        }
}

TEST(linalg, axpby_scalar_multi_simd) {
    using namespace batmat;

    const index_t n            = 43;
    using v_t                  = index_constant<4>;
    using abi                  = datapar::deduced_abi<real_t, v_t{}()>;
    constexpr size_t alignment = datapar::simd_align<real_t, abi>::value;
    matrix::Matrix<real_t, index_t, v_t> x{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t> y{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(x, [&] { return dist(rng); });
    std::ranges::generate(y, [&] { return dist(rng); });

    alignas(alignment) real_t alpha_data[v_t()] = {2.5, 2.0, 1.5, 3.0};
    alignas(alignment) real_t beta_data[v_t()]  = {-1.0, -1.5, -0.5, -2.0};
    auto alpha = datapar::aligned_load<datapar::simd<real_t, abi>>(alpha_data);
    auto beta  = datapar::aligned_load<datapar::simd<real_t, abi>>(beta_data);

    matrix::Matrix<real_t, index_t, v_t> z{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    linalg::axpby(alpha, x, beta, y, z);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t d = 0; d < 3 * v_t(); ++d)
        for (index_t i = 0; i < n; ++i)
            EXPECT_NEAR(alpha_data[d % v_t()] * x(d, i, 0) + beta_data[d % v_t()] * y(d, i, 0),
                        z(d, i, 0), eps)
                << d << ", " << i;
}

TEST(linalg, axpy_inplace_multi_simd) {
    using namespace batmat;

    const index_t n            = 43;
    using v_t                  = index_constant<4>;
    using abi                  = datapar::deduced_abi<real_t, v_t{}()>;
    constexpr size_t alignment = datapar::simd_align<real_t, abi>::value;
    matrix::Matrix<real_t, index_t, v_t> x{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t> y{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(x, [&] { return dist(rng); });
    std::ranges::generate(y, [&] { return dist(rng); });

    alignas(alignment) real_t alpha_data[v_t()] = {0.5, 1.5, 2.0, -0.5};
    auto alpha = datapar::aligned_load<datapar::simd<real_t, abi>>(alpha_data);

    matrix::Matrix<real_t, index_t, v_t> y2 = y;
    linalg::axpy(alpha, x, y2);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t d = 0; d < 3 * v_t(); ++d)
        for (index_t i = 0; i < n; ++i)
            EXPECT_NEAR(alpha_data[d % v_t()] * x(d, i, 0) + y(d, i, 0), y2(d, i, 0), eps)
                << d << ", " << i;
}
