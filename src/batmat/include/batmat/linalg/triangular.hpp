#pragma once

#include <batmat/linalg/structure.hpp>
#include <type_traits>
#include <utility>

namespace batmat::linalg {

/// Light-weight wrapper class used for overload resolution of triangular and symmetric matrices.
template <class M, MatrixStructure S = MatrixStructure::General>
struct Structured {
    static constexpr MatrixStructure structure = S;
    explicit(S != MatrixStructure::General) Structured(M &&m) : value{std::forward<M>(m)} {}
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
    [[nodiscard]] constexpr auto ref() { return Structured<M &, S>{value}; } // TODO: const version
};

template <class M>
Structured(M &&) -> Structured<M>;

template <class M>
[[nodiscard]] constexpr auto tril(M &&m) {
    return Structured<M, MatrixStructure::LowerTriangular>{std::forward<M>(m)};
}

template <class M>
[[nodiscard]] constexpr auto triu(M &&m) {
    return Structured<M, MatrixStructure::UpperTriangular>{std::forward<M>(m)};
}

template <MatrixStructure S, class M>
[[nodiscard]] constexpr auto make_structured(M &&m) {
    return Structured<M, S>{std::forward<M>(m)};
}

template <class M, MatrixStructure S>
void simdify(const Structured<M, S> &) = delete;

} // namespace batmat::linalg
