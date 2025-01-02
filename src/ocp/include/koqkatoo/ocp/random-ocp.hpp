#pragma once

#include <koqkatoo/ocp/ocp.hpp>
#include <random>

namespace koqkatoo::ocp {

inline LinearOCPStorage generate_random_ocp(OCPDim dim,
                                            uint_fast32_t seed = 0) {
    LinearOCPStorage ocp{dim};
    auto [N, nx, nu, ny, ny_N] = ocp.dim;

    std::mt19937 rng{seed};
    std::normal_distribution<real_t> nrml{0, 1};
    std::bernoulli_distribution bernoulli{0.5};

    // Dynamics and constraint matrices
    for (index_t i = 0; i < N; ++i) {
        auto Ai = ocp.A(i), Bi = ocp.B(i), Ci = ocp.C(i), Di = ocp.D(i);
        auto Qi = ocp.Q(i), Ri = ocp.R(i);
        Ai.generate([&] { return nrml(rng) / 2; });
        Bi.generate([&] { return nrml(rng); });
        Ci.generate([&] { return nrml(rng); });
        Di.generate([&] { return nrml(rng); });
        Qi.generate([&] { return nrml(rng); });
        Ri.generate([&] { return nrml(rng); });
        for (index_t j = 0; j < nx; ++j)
            Qi(j, j) += 10;
        for (index_t j = 0; j < nu; ++j)
            Ri(j, j) += 10;
    }
    auto Ci = ocp.C(N), Qi = ocp.Q(N);
    Ci.generate([&] { return nrml(rng); });
    Qi.generate([&] { return nrml(rng); });
    for (index_t j = 0; j < nx; ++j)
        Qi(j, j) += 25;
    return ocp;
}

} // namespace koqkatoo::ocp
