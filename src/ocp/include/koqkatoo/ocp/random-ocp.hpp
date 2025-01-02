#pragma once

#include <koqkatoo/linalg-compact/matrix-batch.hpp>
#include <koqkatoo/linalg/blas-interface.hpp>
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
    std::vector<real_t> work_storage((nx + nu) * (nx + nu));
    guanaqo::MatrixView<real_t, index_t> work{{
        .data = work_storage.data(),
        .rows = nx + nu,
        .cols = nx + nu,
    }};

    // Dynamics and constraint matrices
    for (index_t i = 0; i < N; ++i) {
        auto Ai = ocp.A(i), Bi = ocp.B(i), Ci = ocp.C(i), Di = ocp.D(i);
        auto Hi = ocp.H(i);
        Ai.generate([&] { return nrml(rng) / 2; });
        Bi.generate([&] { return nrml(rng); });
        Ci.generate([&] { return nrml(rng); });
        Di.generate([&] { return nrml(rng); });
        work.generate([&] { return nrml(rng); });
        linalg::xsyrk(CblasColMajor, CblasLower, CblasNoTrans, nx + nu, nx + nu,
                      real_t{1}, work.data, work.outer_stride, real_t{0},
                      Hi.data, Hi.outer_stride);
    }
    auto Ci = ocp.C(N), Qi = ocp.Q(N);
    Ci.generate([&] { return nrml(rng); });
    work.top_left(nx, nx).generate([&] { return nrml(rng); });
    linalg::xsyrk(CblasColMajor, CblasLower, CblasNoTrans, nx, nx, real_t{1},
                  work.data, work.outer_stride, real_t{0}, Qi.data,
                  Qi.outer_stride);
    return ocp;
}

} // namespace koqkatoo::ocp
