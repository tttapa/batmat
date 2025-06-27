#pragma once

#include <batmat/linalg/structure.hpp>
#include <utility>

namespace batmat::linalg {

/// Light-weight wrapper class used for overload resolution of triangular and symmetric matrices.
template <class M, MatrixStructure S = MatrixStructure::General>
struct Structured {
    static constexpr MatrixStructure structure = S;
    Structured(M &&m) : value{std::forward<M>(m)} {}
    Structured(const Structured &) = default;
    Structured(Structured &&)      = default;
    M value;
    [[nodiscard]] constexpr auto transposed() const &
        requires requires { value.transposed(); }
    {
        return Structured<decltype(value.transposed()), transpose(S)>{value.transposed()};
    }
    [[nodiscard]] constexpr auto transposed() &&
        requires requires { std::forward<M>(value).transposed(); }
    {
        return Structured<decltype(std::forward<M>(value).transposed()), transpose(S)>{
            std::forward<M>(value).transposed()};
    }
};

template <class M>
Structured(M &&) -> Structured<M>;

template <class M>
[[nodiscard]] constexpr Structured<M, MatrixStructure::LowerTriangular> tril(M &&m) {
    return {std::forward<M>(m)};
}

template <class M>
[[nodiscard]] constexpr Structured<M, MatrixStructure::UpperTriangular> triu(M &&m) {
    return {std::forward<M>(m)};
}

} // namespace batmat::linalg
