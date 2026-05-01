#include <batmat/linalg/reduce.hpp>
#include <batmat/matrix/matrix.hpp>

#include <gtest/gtest.h>
#include <limits>
#include <random>

TEST(linalg, vdot) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}}, b{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });
    std::ranges::generate(b, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            expected[j] += a(j, i, 0) * b(j, i, 0);
    auto result = linalg::vdot(a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, dot) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}}, b{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });
    std::ranges::generate(b, [&] { return dist(rng); });

    real_t expected{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            expected += a(j, i, 0) * b(j, i, 0);
    auto result = linalg::dot(a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    EXPECT_NEAR(expected, result, eps);
}

TEST(linalg, vnorms_all) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected_amax[v_t()]{}, expected_asum[v_t()]{}, expected_sumsq[v_t()]{};
    for (index_t i = 0; i < n; ++i) {
        for (index_t j = 0; j < v_t(); ++j) {
            real_t val       = std::abs(a(j, i, 0));
            expected_amax[j] = std::max(expected_amax[j], val);
            expected_asum[j] += val;
            expected_sumsq[j] += a(j, i, 0) * a(j, i, 0);
        }
    }

    auto result = linalg::vnorms_all(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t j = 0; j < v_t(); ++j) {
        EXPECT_NEAR(expected_amax[j], result.amax[j], eps) << "amax at lane " << j;
        EXPECT_NEAR(expected_asum[j], result.asum[j], eps) << "asum at lane " << j;
        EXPECT_NEAR(expected_sumsq[j], result.sumsq[j], eps) << "sumsq at lane " << j;
    }
}

TEST(linalg, norms_all) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected_amax{}, expected_asum{}, expected_sumsq{};
    for (index_t i = 0; i < n; ++i) {
        for (index_t j = 0; j < v_t(); ++j) {
            real_t val    = std::abs(a(j, i, 0));
            expected_amax = std::max(expected_amax, val);
            expected_asum += val;
            expected_sumsq += a(j, i, 0) * a(j, i, 0);
        }
    }

    auto result = linalg::norms_all(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    EXPECT_NEAR(expected_amax, result.amax, eps) << "amax";
    EXPECT_NEAR(expected_asum, result.asum, eps) << "asum";
    EXPECT_NEAR(expected_sumsq, result.sumsq, eps) << "sumsq";
}

TEST(linalg, vnorm_inf) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            expected[j] = std::max(expected[j], std::abs(a(j, i, 0)));

    auto result = linalg::vnorm_inf(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, norm_inf) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            expected = std::max(expected, std::abs(a(j, i, 0)));

    auto result = linalg::norm_inf(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    EXPECT_NEAR(expected, result, eps);
}

TEST(linalg, vnorm_1) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            expected[j] += std::abs(a(j, i, 0));

    auto result = linalg::vnorm_1(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, norm_1) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            expected += std::abs(a(j, i, 0));

    auto result = linalg::norm_1(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    EXPECT_NEAR(expected, result, eps);
}

TEST(linalg, vnorm_2_squared) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            expected[j] += a(j, i, 0) * a(j, i, 0);

    auto result = linalg::vnorm_2_squared(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, norm_2_squared) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            expected += a(j, i, 0) * a(j, i, 0);

    auto result = linalg::norm_2_squared(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    EXPECT_NEAR(expected, result, eps);
}

TEST(linalg, vnorm_2) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t sumsq[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            sumsq[j] += a(j, i, 0) * a(j, i, 0);

    real_t expected[v_t()];
    for (index_t j = 0; j < v_t(); ++j)
        expected[j] = std::sqrt(sumsq[j]);

    auto result = linalg::vnorm_2(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, norm_2) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t sumsq{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            sumsq += a(j, i, 0) * a(j, i, 0);

    real_t expected = std::sqrt(sumsq);
    auto result     = linalg::norm_2(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    EXPECT_NEAR(expected, result, eps);
}

TEST(linalg, weighted_vnorm_sq) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> w{{.rows = n, .cols = 1}}, a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(w, [&] { return dist(rng); });
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            expected[j] += w(j, i, 0) * (a(j, i, 0) * a(j, i, 0));

    auto result = linalg::weighted_vnorm_sq(w, a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, weighted_norm_sq) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> w{{.rows = n, .cols = 1}}, a{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(w, [&] { return dist(rng); });
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < v_t(); ++j)
            expected += w(j, i, 0) * (a(j, i, 0) * a(j, i, 0));

    auto result = linalg::weighted_norm_sq(w, a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    EXPECT_NEAR(expected, result, eps);
}

TEST(linalg, weighted_vnorm_sq_diff) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> w{{.rows = n, .cols = 1}}, a{{.rows = n, .cols = 1}},
        b{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(w, [&] { return dist(rng); });
    std::ranges::generate(a, [&] { return dist(rng); });
    std::ranges::generate(b, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i) {
        for (index_t j = 0; j < v_t(); ++j) {
            real_t diff = a(j, i, 0) - b(j, i, 0);
            expected[j] += w(j, i, 0) * (diff * diff);
        }
    }

    auto result = linalg::weighted_vnorm_sq_diff(w, a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, weighted_norm_sq_diff) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t, v_t> w{{.rows = n, .cols = 1}}, a{{.rows = n, .cols = 1}},
        b{{.rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(w, [&] { return dist(rng); });
    std::ranges::generate(a, [&] { return dist(rng); });
    std::ranges::generate(b, [&] { return dist(rng); });

    real_t expected{};
    for (index_t i = 0; i < n; ++i) {
        for (index_t j = 0; j < v_t(); ++j) {
            real_t diff = a(j, i, 0) - b(j, i, 0);
            expected += w(j, i, 0) * (diff * diff);
        }
    }

    auto result = linalg::weighted_norm_sq_diff(w, a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    EXPECT_NEAR(expected, result, eps);
}

TEST(linalg, vdot_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t> b{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });
    std::ranges::generate(b, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < 3 * v_t(); ++j)
            expected[j % v_t()] += a(j, i, 0) * b(j, i, 0);
    auto result = linalg::vdot(a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, dot_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    matrix::Matrix<real_t, index_t, v_t> b{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });
    std::ranges::generate(b, [&] { return dist(rng); });

    real_t expected{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < 3 * v_t(); ++j)
            expected += a(j, i, 0) * b(j, i, 0);
    auto result = linalg::dot(a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 100;
    EXPECT_NEAR(expected, result, eps);
}

TEST(linalg, vnorms_all_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected_amax[v_t()]{}, expected_asum[v_t()]{}, expected_sumsq[v_t()]{};
    for (index_t i = 0; i < n; ++i) {
        for (index_t j = 0; j < 3 * v_t(); ++j) {
            real_t val               = std::abs(a(j, i, 0));
            expected_amax[j % v_t()] = std::max(expected_amax[j % v_t()], val);
            expected_asum[j % v_t()] += val;
            expected_sumsq[j % v_t()] += a(j, i, 0) * a(j, i, 0);
        }
    }

    auto result = linalg::vnorms_all(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t j = 0; j < v_t(); ++j) {
        EXPECT_NEAR(expected_amax[j], result.amax[j], eps) << "amax at lane " << j;
        EXPECT_NEAR(expected_asum[j], result.asum[j], eps) << "asum at lane " << j;
        EXPECT_NEAR(expected_sumsq[j], result.sumsq[j], eps) << "sumsq at lane " << j;
    }
}

TEST(linalg, vnorm_inf_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < 3 * v_t(); ++j)
            expected[j % v_t()] = std::max(expected[j % v_t()], std::abs(a(j, i, 0)));

    auto result = linalg::vnorm_inf(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, vnorm_1_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < 3 * v_t(); ++j)
            expected[j % v_t()] += std::abs(a(j, i, 0));

    auto result = linalg::vnorm_1(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, vnorm_2_squared_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < 3 * v_t(); ++j)
            expected[j % v_t()] += a(j, i, 0) * a(j, i, 0);

    auto result = linalg::vnorm_2_squared(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, vnorm_2_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t sumsq[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < 3 * v_t(); ++j)
            sumsq[j % v_t()] += a(j, i, 0) * a(j, i, 0);

    real_t expected[v_t()];
    for (index_t j = 0; j < v_t(); ++j)
        expected[j] = std::sqrt(sumsq[j]);

    auto result = linalg::vnorm_2(a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, weighted_vnorm_sq_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> w{{.depth = 3 * v_t(), .rows = n, .cols = 1}},
        a{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(w, [&] { return dist(rng); });
    std::ranges::generate(a, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < 3 * v_t(); ++j)
            expected[j % v_t()] += w(j, i, 0) * (a(j, i, 0) * a(j, i, 0));

    auto result = linalg::weighted_vnorm_sq(w, a);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}

TEST(linalg, weighted_vnorm_sq_diff_multi) {
    using namespace batmat;

    const index_t n = 43;
    using v_t       = index_constant<4>;
    matrix::Matrix<real_t, index_t, v_t> w{{.depth = 3 * v_t(), .rows = n, .cols = 1}},
        a{{.depth = 3 * v_t(), .rows = n, .cols = 1}},
        b{{.depth = 3 * v_t(), .rows = n, .cols = 1}};
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> dist{-2, 2};
    std::ranges::generate(w, [&] { return dist(rng); });
    std::ranges::generate(a, [&] { return dist(rng); });
    std::ranges::generate(b, [&] { return dist(rng); });

    real_t expected[v_t()]{};
    for (index_t i = 0; i < n; ++i) {
        for (index_t j = 0; j < 3 * v_t(); ++j) {
            real_t diff = a(j, i, 0) - b(j, i, 0);
            expected[j % v_t()] += w(j, i, 0) * (diff * diff);
        }
    }

    auto result = linalg::weighted_vnorm_sq_diff(w, a, b);

    constexpr real_t eps = std::numeric_limits<real_t>::epsilon() * 1000;
    for (index_t j = 0; j < v_t(); ++j)
        EXPECT_NEAR(expected[j], result[j], eps) << j;
}
