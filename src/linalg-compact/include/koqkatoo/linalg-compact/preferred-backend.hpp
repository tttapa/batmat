#pragma once

#include <stdexcept>
#include <utility>

namespace koqkatoo::linalg::compact {

enum class PreferredBackend {
    Reference        = 0,
    BLASScalar       = 1,
    MKLCompact       = 2,
    MKLBatched       = 4,
    MKLScalarBatched = BLASScalar | MKLBatched,
    MKLAll           = BLASScalar | MKLCompact | MKLBatched,
};

inline bool use_blas_scalar(PreferredBackend b) {
    return (std::to_underlying(b) &
            std::to_underlying(PreferredBackend::BLASScalar)) != 0;
}

inline bool use_mkl_compact(PreferredBackend b) {
    return (std::to_underlying(b) &
            std::to_underlying(PreferredBackend::MKLCompact)) != 0;
}

inline bool use_mkl_batched(PreferredBackend b) {
    return (std::to_underlying(b) &
            std::to_underlying(PreferredBackend::MKLBatched)) != 0;
}

constexpr const char *enum_name(PreferredBackend b) {
    switch (b) {
        case PreferredBackend::Reference: return "Reference";
        case PreferredBackend::BLASScalar: return "BLASScalar";
        case PreferredBackend::MKLCompact: return "MKLCompact";
        case PreferredBackend::MKLBatched: return "MKLBatched";
        case PreferredBackend::MKLScalarBatched: return "MKLScalarBatched";
        case PreferredBackend::MKLAll: return "MKLAll";
        default:;
    }
    throw std::out_of_range(
        "invalid value for koqkatoo::linalg::compact::PreferredBackend");
}

} // namespace koqkatoo::linalg::compact
