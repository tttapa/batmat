#pragma once

#include <optional>
#include <type_traits>

namespace batmat::linalg {

template <int I>
struct with_shift_A_t : std::integral_constant<int, I> {};
template <int I>
struct with_shift_B_t : std::integral_constant<int, I> {};
template <int I>
struct with_rotate_C_t : std::integral_constant<int, I> {};
template <int I>
struct with_rotate_D_t : std::integral_constant<int, I> {};
template <int I>
struct with_mask_D_t : std::integral_constant<int, I> {};

template <int I>
inline constexpr with_shift_A_t<I> with_shift_A;
template <int I>
inline constexpr with_shift_B_t<I> with_shift_B;
template <int I>
inline constexpr with_rotate_C_t<I> with_rotate_C;
template <int I>
inline constexpr with_rotate_D_t<I> with_rotate_D;
template <int I>
inline constexpr with_mask_D_t<I> with_mask_D;

template <class...>
inline constexpr std::optional<int> shift_A = std::nullopt;
template <class T, class... Ts>
inline constexpr std::optional<int> shift_A<T, Ts...> = shift_A<Ts...>;
template <int I, class... Ts>
inline constexpr std::optional<int> shift_A<with_shift_A_t<I>, Ts...> = I;

template <class...>
inline constexpr std::optional<int> shift_B = std::nullopt;
template <class T, class... Ts>
inline constexpr std::optional<int> shift_B<T, Ts...> = shift_B<Ts...>;
template <int I, class... Ts>
inline constexpr std::optional<int> shift_B<with_shift_B_t<I>, Ts...> = I;

template <class...>
inline constexpr std::optional<int> rotate_C = std::nullopt;
template <class T, class... Ts>
inline constexpr std::optional<int> rotate_C<T, Ts...> = rotate_C<Ts...>;
template <int I, class... Ts>
inline constexpr std::optional<int> rotate_C<with_rotate_C_t<I>, Ts...> = I;

template <class...>
inline constexpr std::optional<int> rotate_D = std::nullopt;
template <class T, class... Ts>
inline constexpr std::optional<int> rotate_D<T, Ts...> = rotate_D<Ts...>;
template <int I, class... Ts>
inline constexpr std::optional<int> rotate_D<with_rotate_D_t<I>, Ts...> = I;

template <class...>
inline constexpr std::optional<int> mask_D = std::nullopt;
template <class T, class... Ts>
inline constexpr std::optional<int> mask_D<T, Ts...> = mask_D<Ts...>;
template <int I, class... Ts>
inline constexpr std::optional<int> mask_D<with_mask_D_t<I>, Ts...> = I;

template <class>
inline constexpr bool is_shift_opt = false;
template <int I>
inline constexpr bool is_shift_opt<with_shift_A_t<I>> = true;
template <int I>
inline constexpr bool is_shift_opt<with_shift_B_t<I>> = true;
template <int I>
inline constexpr bool is_shift_opt<with_rotate_C_t<I>> = true;
template <int I>
inline constexpr bool is_shift_opt<with_rotate_D_t<I>> = true;
template <int I>
inline constexpr bool is_shift_opt<with_mask_D_t<I>> = true;

template <class Opt>
concept shift_opt = is_shift_opt<Opt>;

template <int I>
struct with_rotate_t : std::integral_constant<int, I> {};
template <int I>
struct with_mask_t : std::integral_constant<int, I> {};

template <int I>
inline constexpr with_rotate_t<I> with_rotate;
template <int I>
inline constexpr with_mask_t<I> with_mask;

template <class...>
inline constexpr std::optional<int> get_rotate = std::nullopt;
template <class T, class... Ts>
inline constexpr std::optional<int> get_rotate<T, Ts...> = get_rotate<Ts...>;
template <int I, class... Ts>
inline constexpr std::optional<int> get_rotate<with_rotate_t<I>, Ts...> = I;

template <class...>
inline constexpr std::optional<int> get_mask = std::nullopt;
template <class T, class... Ts>
inline constexpr std::optional<int> get_mask<T, Ts...> = get_mask<Ts...>;
template <int I, class... Ts>
inline constexpr std::optional<int> get_mask<with_mask_t<I>, Ts...> = I;

template <class>
inline constexpr bool is_rotate_opt = false;
template <int I>
inline constexpr bool is_rotate_opt<with_rotate_t<I>> = true;
template <int I>
inline constexpr bool is_rotate_opt<with_mask_t<I>> = true;

template <class Opt>
concept rotate_opt = is_rotate_opt<Opt>;

} // namespace batmat::linalg
