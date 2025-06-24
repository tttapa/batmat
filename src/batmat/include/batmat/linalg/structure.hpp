#pragma once

#include <cstdint>

namespace batmat::linalg {

enum class MatrixStructure : int8_t { General, LowerTriangular, UpperTriangular };

constexpr MatrixStructure transpose(MatrixStructure s) {
    using enum MatrixStructure;
    return s == General ? General : s == LowerTriangular ? UpperTriangular : LowerTriangular;
}

} // namespace batmat::linalg
