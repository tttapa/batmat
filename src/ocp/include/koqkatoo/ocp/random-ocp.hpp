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
        auto Qi = ocp.Q(i), Ri = ocp.R(i), Si = ocp.S(i), Hi = ocp.H(i);
        Ai.generate([&] { return nrml(rng) * 0.9; });
        Bi.generate([&] { return nrml(rng); });
        Ci.generate([&] { return nrml(rng); });
        Di.generate([&] { return nrml(rng); });
        Qi.generate([&] { return nrml(rng); });
        Ri.generate([&] { return nrml(rng); });
        Si.generate([&] { return nrml(rng); });
        for (index_t j = 0; j < nx; ++j)
            Qi(j, j) += 2 * static_cast<real_t>(nx + nu);
        for (index_t j = 0; j < nu; ++j)
            Ri(j, j) += 2 * static_cast<real_t>(nx + nu);
        for (index_t c = 0; c < nx + nu; ++c)
            for (index_t r = c + 1; r < nx + nu; ++r)
                Hi(c, r) = Hi(r, c);
    }
    auto Ci = ocp.C(N), Qi = ocp.Q(N);
    Ci.generate([&] { return nrml(rng); });
    Qi.generate([&] { return nrml(rng); });
    for (index_t j = 0; j < nx; ++j)
        Qi(j, j) += 2 * static_cast<real_t>(nx);
    for (index_t c = 0; c < nx; ++c)
        for (index_t r = c + 1; r < nx; ++r)
            Qi(c, r) = Qi(r, c);
    return ocp;
}

} // namespace koqkatoo::ocp
