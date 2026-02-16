#pragma once

#include <cstdint>

namespace batmat::linalg {

/// @ingroup topic-linalg
enum class MatrixStructure : int8_t { General, LowerTriangular, UpperTriangular };

/// @ingroup topic-linalg
constexpr MatrixStructure transpose(MatrixStructure s) {
    using enum MatrixStructure;
    return s == General ? General : s == LowerTriangular ? UpperTriangular : LowerTriangular;
}

} // namespace batmat::linalg
