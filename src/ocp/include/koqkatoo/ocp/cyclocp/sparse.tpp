#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/assume.hpp>
#include <guanaqo/blas/hl-blas-interface.hpp>

namespace koqkatoo::ocp::cyclocp {

template <index_t VL>
auto CyclicOCPSolver<VL>::build_sparse(
    const koqkatoo::ocp::LinearOCPStorage &ocp, std::span<const real_t> Σ) const
    -> std::vector<std::tuple<index_t, index_t, real_t>> {
    using std::sqrt;
    std::vector<std::tuple<index_t, index_t, real_t>> tuples;

    auto [N, nx, nu, ny, ny_N] = ocp.dim;
    const index_t nux = nu + nx, nuxx = nux + nx;
    const index_t vstride    = N >> lvl;
    const index_t num_stages = N >> lP; // number of stages per thread
    const index_t num_proc   = 1 << (lP - lvl);
    const index_t sλ         = N * nuxx - (nx << lP);

    linalg::compact::BatchedMatrix<real_t, index_t> RSQ_DC{{
        .depth = N,
        .rows  = nux,
        .cols  = nux,
    }};
    linalg::compact::BatchedMatrix<real_t, index_t> DC{{
        .depth = N,
        .rows  = std::max(ny, ny_N),
        .cols  = nux,
    }};
    auto R  = RSQ_DC.top_left(nu, nu);
    auto Q  = RSQ_DC.bottom_right(nx, nx);
    auto Sᵀ = RSQ_DC.bottom_left(nx, nu);
    for (index_t k = 0; k < N; ++k) {
        R(k) = ocp.R(k);
        Q(k) = ocp.Q(k == 0 ? N : k);
        if (k > 0) {
            Sᵀ(k)                   = ocp.S_trans(k);
            DC.top_left(ny, nu)(k)  = ocp.D(k);
            DC.top_right(ny, nx)(k) = ocp.C(k);
            for (index_t j = 0; j < ny; ++j)
                for (index_t i = 0; i < nux; ++i)
                    DC(k, j, i) *= sqrt(Σ[k * ny + j]);
            guanaqo::blas::xsyrk_LT(real_t{1}, DC(k), real_t{1}, RSQ_DC(k));
        } else {
            auto D0 = DC.top_left(ny, nu)(k);
            auto CN = DC.bottom_right(ny_N, nx)(k);
            D0      = ocp.D(0);
            CN      = ocp.C(N);
            DC.bottom_left(ny_N, nu)(k).set_constant(0); // TODO
            for (index_t j = 0; j < ny; ++j)
                for (index_t i = 0; i < nu; ++i)
                    D0(j, i) *= sqrt(Σ[k * ny + j]);
            for (index_t j = 0; j < ny_N; ++j)
                for (index_t i = 0; i < nx; ++i)
                    CN(j, i) *= sqrt(Σ[N * ny + j]);
            guanaqo::blas::xsyrk_LT(real_t{1}, DC(k), real_t{1}, RSQ_DC(k));
        }
    }
    for (index_t vi = 0; vi < vl; ++vi) {
        const index_t sv = vi * num_proc * (nuxx * num_stages - nx);
        for (index_t ti = 0; ti < num_proc; ++ti) {
            const index_t k0 = ti * num_stages + vi * vstride;
            const auto biA   = ti + vi * num_proc;
            const auto biI   = sub_wrap_P(biA, 1);
            const auto sλA   = sλ + nx * get_linear_batch_offset(biA);
            const auto sλI   = sλ + nx * get_linear_batch_offset(biI);
            // TODO: handle case if lev > or >= lP - lvl
            for (index_t i = 0; i < num_stages; ++i) {
                const index_t k = sub_wrap_N(k0, i);
                index_t s       = sv + ti * (nuxx * num_stages - nx) + nuxx * i;
                for (index_t c = 0; c < nu; ++c) {
                    for (index_t r = c; r < nu; ++r)
                        tuples.emplace_back(s + r, s + c, R(k)(r, c));
                    if (k > 0)
                        for (index_t r = 0; r < nx; ++r)
                            tuples.emplace_back(s + r + nu, s + c, Sᵀ(k)(r, c));
                    if (i == 0) {
                        for (index_t r = 0; r < nx; ++r)
                            tuples.emplace_back(sλA + r, s + c,
                                                ocp.B(k0)(r, c));
                    }
                }
                for (index_t c = 0; c < nx; ++c) {
                    for (index_t r = c; r < nx; ++r)
                        tuples.emplace_back(s + r + nu, s + c + nu, Q(k)(r, c));
                    if (i + 1 < num_stages)
                        tuples.emplace_back(s + c + nux, s + c + nu, -1);
                    else
                        tuples.emplace_back(sλI + c, s + c + nu, -1);
                    if (i == 0 && k > 0) {
                        for (index_t r = 0; r < nx; ++r)
                            tuples.emplace_back(sλA + r, s + c + nu,
                                                ocp.A(k0)(r, c));
                    }
                }
                if (i > 0) {
                    for (index_t c = 0; c < nx; ++c) {
                        for (index_t r = 0; r < nu; ++r)
                            tuples.emplace_back(s + r, s - nx + c,
                                                ocp.B(k)(c, r));
                        for (index_t r = 0; r < nx; ++r)
                            tuples.emplace_back(s + nu + r, s - nx + c,
                                                ocp.A(k)(c, r));
                    }
                }
            }
        }
    }
    return tuples;
}

template <index_t VL>
auto CyclicOCPSolver<VL>::build_rhs(matrix_view ux, matrix_view λ) const
    -> std::vector<real_t> {
    auto [N, nx, nu, ny, ny_N] = dim;
    const index_t nux = nu + nx, nuxx = nux + nx;
    std::vector<real_t> tuples(nuxx * N);
    std::ranges::fill(tuples, std::numeric_limits<real_t>::quiet_NaN());
    const index_t num_stages = N >> lP; // number of stages per thread
    const index_t num_proc   = 1 << (lP - lvl);
    const index_t sλ         = N * nuxx - (nx << lP);

    for (index_t vi = 0; vi < vl; ++vi) {
        const index_t sv = vi * num_proc * (nuxx * num_stages - nx);
        for (index_t ti = 0; ti < num_proc; ++ti) {
            const index_t di0 = ti * num_stages;
            for (index_t i = 0; i < num_stages; ++i) {
                const index_t di = di0 + i;
                index_t s = sv + ti * (nuxx * num_stages - nx) + nuxx * i;
                if (i > 0)
                    for (index_t c = 0; c < nx; ++c)
                        tuples[s - nx + c] = λ.batch(di)(vi)(c, 0);
                for (index_t c = 0; c < nux; ++c)
                    tuples[s + c] = ux.batch(di)(vi)(c, 0);
            }
        }
    }
    index_t s               = sλ;
    const auto cyclic_block = [&](index_t i) {
        const index_t bi = i % (1 << (lP - lvl));
        const index_t vi = i / (1 << (lP - lvl));
        const index_t di = bi * num_stages;
        for (index_t c = 0; c < nx; ++c)
            tuples[s + c] = λ.batch(di)(vi)(c, 0);
        s += nx;
    };
    if (lP != lvl) {
        for (index_t i = 0; i < (1 << (lP - 1)); ++i)
            cyclic_block(2 * i + 1);
        for (index_t l = 1; l < lP - lvl; ++l) {
            index_t offset = 1 << l;
            index_t stride = offset << 1;
            for (index_t i = offset; i < (1 << lP); i += stride)
                cyclic_block(i);
        }
    }
    for (index_t i = 0; i < (1 << lP); i += (1 << (lP - lvl))) {
        cyclic_block(i);
    }
    return tuples;
}

template <index_t VL>
auto CyclicOCPSolver<VL>::build_sparse_factor() const
    -> std::vector<std::tuple<index_t, index_t, real_t>> {
    std::vector<std::tuple<index_t, index_t, real_t>> tuples;
    auto [N, nx, nu, ny, ny_N] = dim;
    const index_t nux = nu + nx, nuxx = nux + nx;
    const index_t vstride    = N >> lvl;
    const index_t num_stages = N >> lP; // number of stages per thread
    const index_t num_proc   = 1 << (lP - lvl);
    const index_t sλ         = N * nuxx - (nx << lP);
    matrix AinvQᵀ{{
        .depth = 1 << lP,
        .rows  = dim.nx,
        .cols  = (dim.N_horiz >> lP) * dim.nx,
    }};
    matrix invQᵀ{{
        .depth = 1 << lP,
        .rows  = dim.nx,
        .cols  = (dim.N_horiz >> lP) * dim.nx,
    }};
    matrix LBA{{
        .depth = 1 << lP,
        .rows  = dim.nu + dim.nx,
        .cols  = ((dim.N_horiz >> lP) - 1) * dim.nx,
    }};
    for (index_t ti = 0; ti < num_proc; ++ti) {
        const index_t di0 = ti * num_stages; // data batch index
        for (index_t i = 0; i < num_stages; ++i) {
            const auto di = di0 + i;
            auto RSQ      = riccati_R̂ŜQ̂.batch(ti);
            auto RSQi     = RSQ.middle_cols(i * nux, nux);
            auto Qi       = RSQi.bottom_right(nx, nx);
            auto Qi_inv   = invQᵀ.batch(ti).middle_cols(i * nx, nx);
            auto Â        = riccati_ÂB̂.batch(ti).left_cols(num_stages * nx);
            auto Âi       = Â.middle_cols(i * nx, nx);
            auto AiQᵀ     = AinvQᵀ.batch(ti);
            auto AiQiᵀ    = AiQᵀ.middle_cols(i * nx, nx);
            auto BAᵀ      = riccati_BAᵀ.batch(ti);
            auto LBAt     = LBA.batch(ti);

            compact_blas::xcopy(Âi, AiQiᵀ);
            if (i + 1 < num_stages) // Final block already inverted
                compact_blas::xtrtri_T_copy_ref(Qi, Qi_inv);
            else
                compact_blas::xcopy(RSQi.block(nu - 1, nu, nx, nx), Qi_inv);
            if (i + 1 < num_stages) // Final block is already Â LQ⁻ᵀ
                compact_blas::xtrsm_RLTN(Qi, AiQiᵀ, backend);
            if (i > 0) {
                auto LBAi = LBAt.middle_cols((i - 1) * nx, nx);
                if (alt) {
                    auto RSQ_prev = RSQ.middle_cols((i - 1) * nux, nux);
                    auto Q_prev   = RSQ_prev.bottom_right(nx, nx);
                    auto BA       = data_BA.batch(di);
                    compact_blas::xcopy_T(BA, LBAi);
                    compact_blas::xtrmm_RLNN(LBAi, Q_prev, LBAi, backend);
                } else {
                    auto BAᵀi = BAᵀ.middle_cols((i - 1) * nx, nx);
                    compact_blas::xcopy(BAᵀi, LBAi);
                }
            }
        }
    }
    for (index_t vi = 0; vi < vl; ++vi) {
        const index_t sv = vi * num_proc * (nuxx * num_stages - nx);
        for (index_t ti = 0; ti < num_proc; ++ti) {
            const index_t k0 = ti * num_stages + vi * vstride;
            const auto biA   = ti + vi * num_proc;
            const auto biI   = sub_wrap_P(biA, 1);
            const auto sλA   = sλ + nx * get_linear_batch_offset(biA);
            const auto sλI   = sλ + nx * get_linear_batch_offset(biI);
            auto B̂           = riccati_ÂB̂.batch(ti).right_cols(num_stages * nu);
            auto R̂ŜQ̂         = riccati_R̂ŜQ̂.batch(ti);
            auto LBAt        = LBA.batch(ti);
            // TODO: handle case if lev > or >= lP - lvl
            for (index_t i = 0; i < num_stages; ++i) {
                [[maybe_unused]] const index_t k = sub_wrap_N(k0, i);
                index_t s  = sv + ti * (nuxx * num_stages - nx) + nuxx * i;
                auto B̂i    = B̂.middle_cols(i * nu, nu);
                auto R̂ŜQ̂i  = R̂ŜQ̂.middle_cols(i * nux, nux);
                auto RSi   = R̂ŜQ̂i(vi).left_cols(nu);
                auto Qi    = R̂ŜQ̂i(vi).bottom_right(nx, nx);
                auto iQiᵀ  = invQᵀ.batch(ti).middle_cols(i * nx, nx);
                auto AiQᵀ  = AinvQᵀ.batch(ti);
                auto AiQiᵀ = AiQᵀ.middle_cols(i * nx, nx);
                if (i > 0) {
                    auto LBAi      = LBAt.middle_cols((i - 1) * nx, nx);
                    auto R̂ŜQ̂i_prev = R̂ŜQ̂.middle_cols((i - 1) * nux, nux);
                    auto iQᵀprev   = R̂ŜQ̂i_prev(vi).block(nu - 1, nu, nx, nx);
                    auto AiQprevᵀ  = AiQᵀ.middle_cols((i - 1) * nx, nx);
                    for (index_t c = 0; c < nx; ++c) {
                        for (index_t r = 0; r <= c; ++r)
                            tuples.emplace_back(s - nx + r, s - nx + c,
                                                -iQᵀprev(r, c));
                        for (index_t r = 0; r < nux; ++r)
                            tuples.emplace_back(s + r, s - nx + c,
                                                LBAi(vi)(r, c));
                        for (index_t r = 0; r < nx; ++r)
                            tuples.emplace_back(sλA + r, s - nx + c,
                                                AiQprevᵀ(vi)(r, c));
                    }
                }
                for (index_t c = 0; c < nu; ++c) {
                    for (index_t r = c; r < nux; ++r)
                        tuples.emplace_back(s + r, s + c, RSi(r, c));
                    for (index_t r = 0; r < nx; ++r)
                        tuples.emplace_back(sλA + r, s + c, B̂i(vi)(r, c));
                }
                for (index_t c = 0; c < nx; ++c) {
                    for (index_t r = c; r < nx; ++r) {
                        tuples.emplace_back(s + r + nu, s + c + nu, Qi(r, c));
                    }
                    if (i + 1 < num_stages)
                        for (index_t r = 0; r <= c; ++r)
                            tuples.emplace_back(s + r + nux, s + c + nu,
                                                -iQiᵀ(vi)(r, c));
                    else
                        for (index_t r = 0; r <= c; ++r)
                            tuples.emplace_back(sλI + r, s + c + nu,
                                                -iQiᵀ(vi)(r, c));
                    for (index_t r = 0; r < nx; ++r)
                        tuples.emplace_back(sλA + r, s + c + nu,
                                            AiQiᵀ(vi)(r, c));
                }
            }
        }
    }
    index_t s               = sλ;
    const auto cyclic_block = [&](index_t i, index_t offset) {
        const index_t sY = sλ + nx * get_linear_batch_offset(i + offset);
        const index_t sU = sλ + nx * get_linear_batch_offset(i - offset);
        const index_t bi = i % (1 << (lP - lvl));
        const index_t vi = i / (1 << (lP - lvl));
        for (index_t c = 0; c < nx; ++c) {
            for (index_t r = c; r < nx; ++r)
                tuples.emplace_back(s + r, s + c,
                                    coupling_D.batch(bi)(vi)(r, c));
            if (i + offset < (1 << lP))
                for (index_t r = 0; r < nx; ++r)
                    tuples.emplace_back(sY + r, s + c,
                                        coupling_Y.batch(bi)(vi)(r, c));
            for (index_t r = 0; r < nx; ++r)
                tuples.emplace_back(sU + r, s + c,
                                    coupling_U.batch(bi)(vi)(r, c));
        }
        s += nx;
    };
    const auto cyclic_block_final = [&](index_t i, index_t offset) {
        const index_t sY = sλ + nx * get_linear_batch_offset(i + offset);
        const index_t bi = i % (1 << (lP - lvl));
        const index_t vi = i / (1 << (lP - lvl));
        for (index_t c = 0; c < nx; ++c) {
            for (index_t r = c; r < nx; ++r)
                tuples.emplace_back(s + r, s + c,
                                    coupling_D.batch(bi)(vi)(r, c));
            if (i + offset < (1 << lP))
                for (index_t r = 0; r < nx; ++r)
                    tuples.emplace_back(sY + r, s + c,
                                        coupling_Y.batch(bi)(vi)(r, c));
        }
        s += nx;
    };
    if (lP != lvl) {
        for (index_t i = 0; i < (1 << (lP - 1)); ++i)
            cyclic_block(2 * i + 1, 1);
        for (index_t l = 1; l < lP - lvl; ++l) {
            index_t offset = 1 << l;
            index_t stride = offset << 1;
            for (index_t i = offset; i < (1 << lP); i += stride)
                cyclic_block(i, offset);
        }
    }
    for (index_t i = 0; i < (1 << lP); i += (1 << (lP - lvl))) {
        cyclic_block_final(i, 1 << (lP - lvl));
    }
    return tuples;
}

template <index_t VL>
auto CyclicOCPSolver<VL>::build_sparse_diag() const
    -> std::vector<std::tuple<index_t, index_t, real_t>> {
    std::vector<std::tuple<index_t, index_t, real_t>> tuples;
    auto [N, nx, nu, ny, ny_N] = dim;
    const index_t nux = nu + nx, nuxx = nux + nx;
    const index_t num_stages = N >> lP; // number of stages per thread
    const index_t num_proc   = 1 << (lP - lvl);
    const index_t sλ         = N * nuxx - (nx << lP);
    for (index_t vi = 0; vi < vl; ++vi) {
        const index_t sv = vi * num_proc * (nuxx * num_stages - nx);
        for (index_t ti = 0; ti < num_proc; ++ti) {
            for (index_t i = 0; i < num_stages; ++i) {
                index_t s = sv + ti * (nuxx * num_stages - nx) + nuxx * i;
                if (i > 0)
                    for (index_t c = 0; c < nx; ++c)
                        tuples.emplace_back(s - nx + c, s - nx + c, -1);
                for (index_t c = 0; c < nu; ++c)
                    tuples.emplace_back(s + c, s + c, 1);
                for (index_t c = 0; c < nx; ++c)
                    tuples.emplace_back(s + c + nu, s + c + nu, 1);
            }
        }
    }
    for (index_t i = 0; i < 1 << lP; ++i)
        for (index_t r = 0; r < nx; ++r)
            tuples.emplace_back(sλ + nx * i + r, sλ + nx * i + r, -1);
    return tuples;
}

} // namespace koqkatoo::ocp::cyclocp
