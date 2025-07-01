#pragma once

#include <batmat/kib.hpp>
#include <batmat/linalg/copy.hpp>
#include <batmat/linalg/micro-kernels/gemm.hpp>
#include <batmat/linalg/shift.hpp>
#include <batmat/linalg/simdify.hpp>
#include <batmat/linalg/triangular.hpp>
#include <batmat/linalg/uview.hpp>
#include <batmat/loop.hpp>
#include <batmat/matrix/storage.hpp>
#include <guanaqo/trace.hpp>
#include <optional>

namespace batmat::linalg {

enum class PackingSelector : int8_t { Never, Always, Transpose };

struct TilingOptions {
    bool no_tiling         = false;
    PackingSelector pack_A = PackingSelector::Transpose;
    PackingSelector pack_B = PackingSelector::Always;
    index_t n_c = 0, k_c = 0, m_c = 0;
};

namespace detail {
template <class T, class Abi, micro_kernels::gemm::KernelConfig Conf = {}, StorageOrder OA,
          StorageOrder OB, StorageOrder OC, StorageOrder OD>
    requires(Conf.struc_A == MatrixStructure::General && Conf.struc_B == MatrixStructure::General &&
             Conf.struc_C == MatrixStructure::General)
void gemm(view<const T, Abi, OA> A, view<const T, Abi, OB> B,
          std::optional<view<const T, Abi, OC>> C, view<T, Abi, OD> D, TilingOptions packing = {}) {
    GUANAQO_TRACE("gemm", 0, A.rows() * A.cols() * B.cols() * A.depth());
    // Check dimensions
    BATMAT_ASSERT(!C || C->rows() == D.rows());
    BATMAT_ASSERT(!C || C->cols() == D.cols());
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == B.rows());
    BATMAT_ASSERT(B.cols() == D.cols());
    const index_t M = D.rows(), N = D.cols(), K = A.cols();
    static const index_t N_reg = micro_kernels::gemm::ColsReg<T, Abi>;
    static const index_t M_reg = micro_kernels::gemm::RowsReg<T, Abi>;

    // Degenerate case
    if (M == 0 || N == 0) [[unlikely]]
        return;
    if (K == 0) [[unlikely]] {
        // https://github.com/llvm/llvm-project/issues/146272
        constexpr detail::copy::CopyConfig rot{
            .rotate = Conf.rotate_C - Conf.rotate_D, .mask = Conf.mask_D, .struc = Conf.struc_C};
        constexpr detail::copy::FillConfig msk{.mask = Conf.mask_D, .struc = Conf.struc_C};
        if (C)
            detail::copy::copy<T, Abi, rot>(*C, D);
        else
            detail::copy::fill<T, Abi, msk>(T{}, D);
        return;
    }

    // Small matrices
    using micro_kernels::gemm::gemm_copy_lut;
    if (M <= M_reg && N <= N_reg) [[likely]]
        return gemm_copy_lut<T, Abi, Conf, OA, OB, OC, OD>[M - 1][N - 1](A, B, C, D, K);

    // Determine block sizes for cache tiling
    static const index_t simd_stride   = simd_view_types<T, Abi>::simd_stride;
    static const index_t L1_cache_size = 48_KiB; // TODO: determine dynamically
    static const index_t L2_cache_size = 512_KiB;
    static const index_t L3_cache_size = 16_MiB;
    static const index_t n_cores       = 8; // TODO: OMP
    // clang-format off
    static const index_t K_cache_default = L1_cache_size / sizeof(T) / simd_stride / N_reg;
    static const index_t M_cache_default = (L2_cache_size / sizeof(T) / simd_stride / K_cache_default / M_reg) * M_reg;
    static const index_t N_cache_default = std::max<index_t>(L3_cache_size / sizeof(T) / simd_stride / K_cache_default / n_cores / M_cache_default, 1) * M_cache_default;
    // clang-format on
    const index_t K_cache = packing.k_c ? packing.k_c : K_cache_default;
    const index_t M_cache = packing.m_c ? packing.m_c : M_cache_default;
    const index_t N_cache = packing.n_c ? packing.n_c : N_cache_default;

    // Medium size (no tiling)
    if ((M <= M_cache && N <= N_cache && K <= K_cache) || packing.no_tiling) [[likely]]
        return micro_kernels::gemm::gemm_copy_register<T, Abi, Conf>(A, B, C, D);

    // Determine sizes for packing tiles of A and B
    using simd_align_t        = typename simd_view_types<T, Abi>::simd_align_t;
    const index_t B_pack_size = B.ceil_depth() * K_cache * N_cache;
    const index_t A_pack_size = A.ceil_depth() * M_cache * K_cache;
    const index_t B_size      = B.ceil_depth() * K * N;
    const index_t A_size      = A.ceil_depth() * M * K;
    const bool select_pack_B =
        packing.pack_B == PackingSelector::Always ||
        (packing.pack_B == PackingSelector::Transpose && OB == StorageOrder::RowMajor);
    const bool select_pack_A =
        packing.pack_A == PackingSelector::Always ||
        (packing.pack_A == PackingSelector::Transpose && OA == StorageOrder::ColMajor);
    const bool pack_B = select_pack_B && B_size >= 2 * B_pack_size; // TODO: tune
    const bool pack_A = select_pack_A && A_size >= 2 * A_pack_size; // TODO: tune
    using batmat::matrix::uninitialized;
    auto B_pack = make_aligned_unique_ptr<T>(pack_B ? static_cast<size_t>(B_pack_size) : 0,
                                             simd_align_t(), uninitialized);
    auto A_pack = make_aligned_unique_ptr<T>(pack_A ? static_cast<size_t>(A_pack_size) : 0,
                                             simd_align_t(), uninitialized);
    view<T, Abi, StorageOrder::ColMajor> Bkj_pack;
    view<T, Abi, StorageOrder::RowMajor> Aik_pack;

    // Three outer loops for tiling, with optional packing of A and B
    using micro_kernels::gemm::gemm_copy_register;
    foreach_chunked_merged(0, N, N_cache, [&](index_t j_c, index_t n_c) {
        foreach_chunked_merged(0, K, K_cache, [&](index_t p_c, index_t k_c) {
            auto Bkj = B.block(p_c, j_c, k_c, n_c);
            if (pack_B) {
                Bkj_pack.reassign({{.data = B_pack.get(), .rows = k_c, .cols = n_c}});
                detail::copy::copy<T, Abi>(Bkj, Bkj_pack);
                foreach_chunked_merged(0, M, M_cache, [&](index_t i_c, index_t m_c) {
                    auto Cij = C ? std::make_optional(C->block(i_c, j_c, m_c, n_c)) : std::nullopt;
                    auto Dij = D.block(i_c, j_c, m_c, n_c);
                    auto Aik = A.block(i_c, p_c, m_c, k_c);
                    if (pack_A) {
                        Aik_pack.reassign({{.data = A_pack.get(), .rows = m_c, .cols = k_c}});
                        detail::copy::copy<T, Abi>(Aik, Aik_pack);
                        gemm_copy_register<T, Abi, Conf>(Aik_pack.as_const(), Bkj_pack.as_const(),
                                                         p_c == 0 ? Cij : Dij, Dij);
                    } else {
                        gemm_copy_register<T, Abi, Conf>(Aik, Bkj_pack.as_const(),
                                                         p_c == 0 ? Cij : Dij, Dij);
                    }
                });
            } else {
                foreach_chunked_merged(0, M, M_cache, [&](index_t i_c, index_t m_c) {
                    auto Cij = C ? std::make_optional(C->block(i_c, j_c, m_c, n_c)) : std::nullopt;
                    auto Dij = D.block(i_c, j_c, m_c, n_c);
                    auto Aik = A.block(i_c, p_c, m_c, k_c);
                    if (pack_A) {
                        Aik_pack.reassign({{.data = A_pack.get(), .rows = m_c, .cols = k_c}});
                        detail::copy::copy<T, Abi>(Aik, Aik_pack);
                        gemm_copy_register<T, Abi, Conf>(Aik_pack.as_const(), Bkj,
                                                         p_c == 0 ? Cij : Dij, Dij);
                    } else {
                        gemm_copy_register<T, Abi, Conf>(Aik, Bkj, p_c == 0 ? Cij : Dij, Dij);
                    }
                });
            }
        });
    });
}

template <class T, class Abi, micro_kernels::gemm::KernelConfig Conf = {}, StorageOrder OA,
          StorageOrder OB, StorageOrder OC, StorageOrder OD>
    requires(Conf.struc_A == MatrixStructure::General && Conf.struc_B == MatrixStructure::General &&
             Conf.struc_C != MatrixStructure::General)
void gemmt(view<const T, Abi, OA> A, view<const T, Abi, OB> B,
           std::optional<view<const T, Abi, OC>> C, view<T, Abi, OD> D) {
    GUANAQO_TRACE("gemmt", 0, A.rows() * A.cols() * (B.cols() + 1) * A.depth() / 2);
    BATMAT_ASSERT(D.rows() == D.cols()); // TODO: could be relaxed
    BATMAT_ASSERT(!C || C->rows() == D.rows());
    BATMAT_ASSERT(!C || C->cols() == D.cols());
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == B.rows());
    BATMAT_ASSERT(B.cols() == D.cols());
    const index_t M = D.rows(), N = D.cols(), K = A.cols();
    if (M == 0 || N == 0) [[unlikely]]
        return;
    if (K == 0) [[unlikely]] {
        // https://github.com/llvm/llvm-project/issues/146272
        constexpr detail::copy::CopyConfig rot{
            .rotate = Conf.rotate_C - Conf.rotate_D, .mask = Conf.mask_D, .struc = Conf.struc_C};
        constexpr detail::copy::FillConfig msk{.mask = Conf.mask_D, .struc = Conf.struc_C};
        if (C)
            detail::copy::copy<T, Abi, rot>(*C, D);
        else
            detail::copy::fill<T, Abi, msk>(T{}, D);
        return;
    }
    // TODO: cache blocking
    return micro_kernels::gemm::gemm_copy_register<T, Abi, Conf>(A, B, C, D);
}

template <class T, class Abi, micro_kernels::gemm::KernelConfig Conf = {}, StorageOrder OA,
          StorageOrder OB, StorageOrder OC, StorageOrder OD>
    requires(Conf.struc_A != MatrixStructure::General || Conf.struc_B != MatrixStructure::General)
void trmm(view<const T, Abi, OA> A, view<const T, Abi, OB> B,
          std::optional<view<const T, Abi, OC>> C, view<T, Abi, OD> D) {
    [[maybe_unused]] index_t flop_count = Conf.struc_A == Conf.struc_B ? 0 : 0; // TODO
    GUANAQO_TRACE("trmm", 0, flop_count * A.depth());
    static_assert(Conf.struc_A != MatrixStructure::General ||
                  Conf.struc_B != MatrixStructure::General);
    static_assert(Conf.struc_A != Conf.struc_B,
                  "lower times lower or upper times upper currently not supported"); // TODO
    if (Conf.struc_A != MatrixStructure::General)
        BATMAT_ASSERT(A.rows() == A.cols()); // TODO: could be relaxed
    if (Conf.struc_B != MatrixStructure::General)
        BATMAT_ASSERT(B.rows() == B.cols()); // TODO: could be relaxed
    BATMAT_ASSERT(!C || C->rows() == D.rows());
    BATMAT_ASSERT(!C || C->cols() == D.cols());
    BATMAT_ASSERT(A.rows() == D.rows());
    BATMAT_ASSERT(A.cols() == B.rows());
    BATMAT_ASSERT(B.cols() == D.cols());
    const index_t M = D.rows(), N = D.cols(), K = A.cols();
    if (M == 0 || N == 0) [[unlikely]]
        return;
    if (K == 0) [[unlikely]] {
        // https://github.com/llvm/llvm-project/issues/146272
        constexpr detail::copy::CopyConfig rot{
            .rotate = Conf.rotate_C - Conf.rotate_D, .mask = Conf.mask_D, .struc = Conf.struc_C};
        constexpr detail::copy::FillConfig msk{.mask = Conf.mask_D, .struc = Conf.struc_C};
        if (C)
            detail::copy::copy<T, Abi, rot>(*C, D);
        else
            detail::copy::fill<T, Abi, msk>(T{}, D);
        return;
    }
    // TODO: cache blocking
    return micro_kernels::gemm::gemm_copy_register<T, Abi, Conf>(A, B, C, D);
}

template <class... Opts>
constexpr micro_kernels::gemm::KernelConfig apply_options(micro_kernels::gemm::KernelConfig conf,
                                                          Opts...) {
    if (auto s = shift_A<Opts...>)
        conf.shift_A = *s;
    if (auto s = shift_B<Opts...>)
        conf.shift_B = *s;
    if (auto s = rotate_C<Opts...>)
        conf.rotate_C = *s;
    if (auto s = rotate_D<Opts...>)
        conf.rotate_D = *s;
    if (auto s = mask_D<Opts...>)
        conf.mask_D = *s;
    return conf;
}
} // namespace detail

/// D = A B
template <simdifiable VA, simdifiable VB, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VD>
void gemm(VA &&A, VB &&B, VD &&D, TilingOptions packing = {}, Opts... opts) {
    std::optional<decltype(simdify(D).as_const())> null;
    constexpr auto conf = detail::apply_options({.negate = false}, opts...);
    detail::gemm<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(B).as_const(), null, simdify(D), packing);
}

/// D = -A B
template <simdifiable VA, simdifiable VB, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VD>
void gemm_neg(VA &&A, VB &&B, VD &&D, TilingOptions packing = {}, Opts... opts) {
    std::optional<decltype(simdify(D).as_const())> null;
    constexpr auto conf = detail::apply_options({.negate = true}, opts...);
    detail::gemm<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(B).as_const(), null, simdify(D), packing);
}

/// D = C + A B
template <simdifiable VA, simdifiable VB, simdifiable VC, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VC, VD>
void gemm_add(VA &&A, VB &&B, VC &&C, VD &&D, TilingOptions packing = {}, Opts... opts) {
    constexpr auto conf = detail::apply_options({.negate = false}, opts...);
    detail::gemm<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(B).as_const(), std::make_optional(simdify(C).as_const()),
        simdify(D), packing);
}
/// D += A B
template <simdifiable VA, simdifiable VB, simdifiable VD, shift_opt... Opts>
void gemm_add(VA &&A, VB &&B, VD &&D, TilingOptions packing = {}, Opts... opts) {
    return gemm_add(A, B, D, D, packing, opts...);
}

/// D = C - A B
template <simdifiable VA, simdifiable VB, simdifiable VC, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VC, VD>
void gemm_sub(VA &&A, VB &&B, VC &&C, VD &&D, TilingOptions packing = {}, Opts... opts) {
    constexpr auto conf = detail::apply_options({.negate = true}, opts...);
    detail::gemm<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(B).as_const(), std::make_optional(simdify(C).as_const()),
        simdify(D), packing);
}
/// D -= A B
template <simdifiable VA, simdifiable VB, simdifiable VD, shift_opt... Opts>
void gemm_sub(VA &&A, VB &&B, VD &&D, TilingOptions packing = {}, Opts... opts) {
    return gemm_sub(A, B, D, D, packing, opts...);
}

/// D = A Aᵀ with D symmetric
template <MatrixStructure SD, simdifiable VA, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VD>
void syrk(VA &&A, Structured<VD, SD> D, Opts... opts) {
    using enum MatrixStructure;
    static_assert(SD != General);
    std::optional<decltype(simdify(D.value).as_const())> null;
    constexpr auto conf = detail::apply_options({.negate = false, .struc_C = SD}, opts...);
    detail::gemmt<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(A).as_const().transposed(), null, simdify(D.value));
}

/// D = -A Aᵀ with D symmetric
template <MatrixStructure SD, simdifiable VA, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VD>
void syrk_neg(VA &&A, Structured<VD, SD> D, Opts... opts) {
    using enum MatrixStructure;
    static_assert(SD != General);
    std::optional<decltype(simdify(D.value).as_const())> null;
    constexpr auto conf = detail::apply_options({.negate = true, .struc_C = SD}, opts...);
    detail::gemmt<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(A).as_const().transposed(), null, simdify(D.value));
}

/// D = C + A Aᵀ with C, D symmetric
template <MatrixStructure SD, simdifiable VA, simdifiable VC, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VC, VD>
void syrk_add(VA &&A, Structured<VC, SD> C, Structured<VD, SD> D, Opts... opts) {
    using enum MatrixStructure;
    static_assert(SD != General);
    constexpr auto conf = detail::apply_options({.negate = false, .struc_C = SD}, opts...);
    detail::gemmt<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(A).as_const().transposed(),
        std::make_optional(simdify(C.value).as_const()), simdify(D.value));
}
/// D += A Aᵀ with D symmetric
template <MatrixStructure SD, simdifiable VA, simdifiable VD, shift_opt... Opts>
void syrk_add(VA &&A, Structured<VD, SD> D, Opts... opts) {
    return syrk_add(A, D.ref(), D.ref(), opts...);
}

/// D = C - A Aᵀ with C, D symmetric
template <MatrixStructure SD, simdifiable VA, simdifiable VC, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VC, VD>
void syrk_sub(VA &&A, Structured<VC, SD> C, Structured<VD, SD> D, Opts... opts) {
    using enum MatrixStructure;
    static_assert(SD != General);
    constexpr auto conf = detail::apply_options({.negate = true, .struc_C = SD}, opts...);
    detail::gemmt<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A).as_const(), simdify(A).as_const().transposed(),
        std::make_optional(simdify(C.value).as_const()), simdify(D.value));
}
/// D -= A Aᵀ with D symmetric
template <MatrixStructure SD, simdifiable VA, simdifiable VD, shift_opt... Opts>
void syrk_sub(VA &&A, Structured<VD, SD> D, Opts... opts) {
    return syrk_sub(A, D.ref(), D.ref(), opts...);
}

/// D = A B with A and/or B triangular
template <MatrixStructure SA, MatrixStructure SB, MatrixStructure SD, simdifiable VA,
          simdifiable VB, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VD>
void trmm(Structured<VA, SA> A, Structured<VB, SB> B, Structured<VD, SD> D, Opts... opts) {
    std::optional<decltype(simdify(D.value).as_const())> null;
    constexpr auto conf = detail::apply_options(
        {.negate = false, .struc_A = SA, .struc_B = SB, .struc_C = SD}, opts...);
    detail::trmm<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B.value).as_const(), null, simdify(D.value));
}
/// D = A B with A and/or B triangular
template <class TA, class TB, class TD, class... Opts>
void trmm(TA &&A, TB &&B, TD &&D, Opts &&...opts) {
    return trmm(Structured{std::forward<TA>(A)}, Structured{std::forward<TB>(B)},
                Structured{std::forward<TD>(D)}, std::forward<Opts>(opts)...);
}
/// D = A D with A triangular
template <MatrixStructure SA, simdifiable VA, simdifiable VD, class... Opts>
void trmm(Structured<VA, SA> A, VD &&D, Opts &&...opts) {
    return trmm(A.ref(), Structured{D}, Structured{D}, std::forward<Opts>(opts)...);
}
/// D = D B with B triangular
template <MatrixStructure SB, simdifiable VB, simdifiable VD, class... Opts>
void trmm(VD &&D, Structured<VB, SB> B, Opts &&...opts) {
    return trmm(Structured{D}, B.ref(), Structured{D}, std::forward<Opts>(opts)...);
}

/// D = -A B with A and/or B triangular
template <MatrixStructure SA, MatrixStructure SB, MatrixStructure SD, simdifiable VA,
          simdifiable VB, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VD>
void trmm_neg(Structured<VA, SA> A, Structured<VB, SB> B, Structured<VD, SD> D, Opts... opts) {
    std::optional<decltype(simdify(D.value).as_const())> null;
    constexpr auto conf = detail::apply_options(
        {.negate = true, .struc_A = SA, .struc_B = SB, .struc_C = SD}, opts...);
    detail::trmm<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B.value).as_const(), null, simdify(D.value));
}
/// D = -A B with A and/or B triangular
template <class TA, class TB, class TD, class... Opts>
void trmm_neg(TA &&A, TB &&B, TD &&D, Opts &&...opts) {
    return trmm_neg(Structured{std::forward<TA>(A)}, Structured{std::forward<TB>(B)},
                    Structured{std::forward<TD>(D)}, std::forward<Opts>(opts)...);
}

/// D = C + A B with A and/or B triangular
template <MatrixStructure SA, MatrixStructure SB, MatrixStructure SD, simdifiable VA,
          simdifiable VB, simdifiable VC, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VD>
void trmm_add(Structured<VA, SA> A, Structured<VB, SB> B, Structured<VC, SD> C,
              Structured<VD, SD> D, Opts... opts) {
    constexpr auto conf = detail::apply_options(
        {.negate = false, .struc_A = SA, .struc_B = SB, .struc_C = SD}, opts...);
    detail::trmm<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B.value).as_const(),
        std::make_optional(simdify(C.value).as_const()), simdify(D.value));
}
/// D = C + A B with A and/or B triangular
template <class TA, class TB, class TC, class TD, class... Opts>
void trmm_add(TA &&A, TB &&B, TC &&C, TD &&D, Opts &&...opts) {
    return trmm_add(Structured{std::forward<TA>(A)}, Structured{std::forward<TB>(B)},
                    Structured{std::forward<TC>(C)}, Structured{std::forward<TD>(D)},
                    std::forward<Opts>(opts)...);
}

/// D = C - A B with A and/or B triangular
template <MatrixStructure SA, MatrixStructure SB, MatrixStructure SD, simdifiable VA,
          simdifiable VB, simdifiable VC, simdifiable VD, shift_opt... Opts>
    requires simdify_compatible<VA, VB, VD>
void trmm_sub(Structured<VA, SA> A, Structured<VB, SB> B, Structured<VC, SD> C,
              Structured<VD, SD> D, Opts... opts) {
    constexpr auto conf = detail::apply_options(
        {.negate = true, .struc_A = SA, .struc_B = SB, .struc_C = SD}, opts...);
    detail::trmm<simdified_value_t<VA>, simdified_abi_t<VA>, conf>(
        simdify(A.value).as_const(), simdify(B.value).as_const(),
        std::make_optional(simdify(C.value).as_const()), simdify(D.value));
}
/// D = C - A B with A and/or B triangular
template <class TA, class TB, class TC, class TD, class... Opts>
void trmm_sub(TA &&A, TB &&B, TC &&C, TD &&D, Opts &&...opts) {
    return trmm_sub(Structured{std::forward<TA>(A)}, Structured{std::forward<TB>(B)},
                    Structured{std::forward<TC>(C)}, Structured{std::forward<TD>(D)},
                    std::forward<Opts>(opts)...);
}

} // namespace batmat::linalg
