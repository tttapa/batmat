#pragma once

#include <batmat/linalg/structure.hpp>

namespace batmat::linalg {

template <class M, MatrixStructure S = MatrixStructure::General>
struct Structured {
    static constexpr MatrixStructure structure = S;
    Structured(M &&m) : value{std::forward<M>(m)} {}
    Structured(const Structured &) = default;
    Structured(Structured &&)      = default;
    M value;
};

template <class M>
Structured(M &&) -> Structured<M>;
template <class M, MatrixStructure S>
Structured(const Structured<M, S> &) -> Structured<M, S>;
template <class M, MatrixStructure S>
Structured(Structured<M, S> &&) -> Structured<M, S>;

template <class M>
Structured<M, MatrixStructure::LowerTriangular> tril(M &&m) {
    return {std::forward<M>(m)};
}

template <class M>
Structured<M, MatrixStructure::UpperTriangular> triu(M &&m) {
    return {std::forward<M>(m)};
}

} // namespace batmat::linalg
