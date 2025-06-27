#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <guanaqo/eigen/view.hpp>
#include <guanaqo/print.hpp>

#include <Eigen/Core>

/// @file
/// @see https://google.github.io/googletest/reference/matchers.html#defining-matchers

namespace guanaqo_test {
template <class T>
void print(std::ostream &os, const T &arg) {
    using Scalar = typename T::Scalar;
    Eigen::MatrixX<Scalar> M{arg};
    guanaqo::print_python(os, guanaqo::as_view(M));
}
} // namespace guanaqo_test

MATCHER_P(EigenEqual, expect, "") {
    auto diff     = arg - expect;
    auto diffnorm = diff.template lpNorm<Eigen::Infinity>();
    if (auto *os = result_listener->stream()) {
        if (std::max(diff.rows(), diff.cols()) < 100) {
            *os << "\nactual = ...\n";
            guanaqo_test::print(*os, arg);
            *os << "and expected = ...\n";
            guanaqo_test::print(*os, expect);
            *os << "with difference = ...\n";
            guanaqo_test::print(*os, diff);
        }
        *os << "which has infinity norm " << guanaqo::float_to_str(diffnorm);
    }
    return diffnorm == 0 && diff.allFinite();
}

MATCHER_P2(EigenAlmostEqual, expect, atol, "") {
    auto diff     = arg - expect;
    auto diffnorm = diff.template lpNorm<Eigen::Infinity>();
    if (auto *os = result_listener->stream()) {
        if (std::max(diff.rows(), diff.cols()) < 100) {
            *os << "\nactual = ...\n";
            guanaqo_test::print(*os, arg);
            *os << "and expected = ...\n";
            guanaqo_test::print(*os, expect);
            *os << "with difference = ...\n";
            guanaqo_test::print(*os, diff);
        }
        *os << "which has infinity norm                      " << guanaqo::float_to_str(diffnorm);
        *os << ",\nwhich is greater than the absolute tolerance " << guanaqo::float_to_str(atol);
    }
    return diffnorm <= atol && diff.allFinite();
}

MATCHER_P2(EigenAlmostEqualRel, expect, rtol, "") {
    auto diff     = arg - expect;
    auto diffnorm = diff.cwiseQuotient(expect).template lpNorm<Eigen::Infinity>();
    if (auto *os = result_listener->stream()) {
        if (std::max(diff.rows(), diff.cols()) < 100) {
            *os << "\nactual = ...\n";
            guanaqo_test::print(*os, arg);
            *os << "and expected = ...\n";
            guanaqo_test::print(*os, expect);
            *os << "with difference = ...\n";
            guanaqo_test::print(*os, diff);
        }
        *os << "which has relative infinity norm             " << guanaqo::float_to_str(diffnorm);
        *os << ",\nwhich is greater than the relative tolerance " << guanaqo::float_to_str(rtol);
    }
    return diffnorm <= rtol && diff.allFinite();
}

template <Eigen::UpLoType UpLo, class T>
auto tri(T &&t) -> Eigen::MatrixX<typename std::remove_cvref_t<T>::Scalar> {
    return std::forward<T>(t).template triangularView<UpLo>().toDenseMatrix();
}
