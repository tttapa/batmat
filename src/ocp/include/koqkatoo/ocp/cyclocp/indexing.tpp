#include <koqkatoo/ocp/cyclocp.hpp>

#include <koqkatoo/assume.hpp>

namespace koqkatoo::ocp::cyclocp {

template <index_t VL>
auto CyclicOCPSolver<VL>::add_wrap_N(index_t a, index_t b) const -> index_t {
    const index_t N = N_horiz;
    KOQKATOO_ASSUME(a >= 0);
    KOQKATOO_ASSUME(b >= 0);
    KOQKATOO_ASSUME(a < N);
    a += b;
    return a >= N ? a - N : a;
}
template <index_t VL>
auto CyclicOCPSolver<VL>::sub_wrap_N(index_t a, index_t b) const -> index_t {
    const index_t N = N_horiz;
    KOQKATOO_ASSUME(a >= 0);
    KOQKATOO_ASSUME(b >= 0);
    KOQKATOO_ASSUME(a < N);
    a -= b;
    return a < 0 ? a + N : a;
}
template <index_t VL>
auto CyclicOCPSolver<VL>::sub_wrap_PmV(index_t a, index_t b) const -> index_t {
    KOQKATOO_ASSUME(a >= 0);
    KOQKATOO_ASSUME(b >= 0);
    KOQKATOO_ASSUME(a < (1 << (lP - lvl)));
    a -= b;
    return a < 0 ? a + (1 << (lP - lvl)) : a;
}
template <index_t VL>
auto CyclicOCPSolver<VL>::add_wrap_PmV(index_t a, index_t b) const -> index_t {
    KOQKATOO_ASSUME(a >= 0);
    KOQKATOO_ASSUME(b >= 0);
    KOQKATOO_ASSUME(a < (1 << (lP - lvl)));
    a += b;
    return a >= (1 << (lP - lvl)) ? a - (1 << (lP - lvl)) : a;
}
template <index_t VL>
auto CyclicOCPSolver<VL>::sub_wrap_P(index_t a, index_t b) const -> index_t {
    KOQKATOO_ASSUME(a >= 0);
    KOQKATOO_ASSUME(b >= 0);
    KOQKATOO_ASSUME(a < (1 << lP));
    a -= b;
    return a < 0 ? a + (1 << lP) : a;
}
template <index_t VL>
auto CyclicOCPSolver<VL>::get_linear_batch_offset(index_t biA) const
    -> index_t {
    const auto levA = biA > 0 ? get_level(biA) : lP;
    const auto levP = lP - lvl;
    if (levA >= levP)
        return (((1 << levP) - 1) << (lP - levP)) + (biA >> levP);
    return (((1 << levA) - 1) << (lP - levA)) + get_index_in_level(biA);
}
template <index_t VL>
[[nodiscard]] bool CyclicOCPSolver<VL>::is_active(index_t l, index_t bi) const {
    const index_t lbi   = bi > 0 ? get_level(bi) : lP - lvl;
    const bool inactive = lbi < l;
    return ((bi >> l) & 1) == 1 && !inactive;
}
template <index_t VL>
[[nodiscard]] bool CyclicOCPSolver<VL>::is_U_below_Y(index_t l,
                                                     index_t bi) const {
    return ((bi >> l) & 3) == 1 && l + 1 != lP - lvl;
}

} // namespace koqkatoo::ocp::cyclocp
