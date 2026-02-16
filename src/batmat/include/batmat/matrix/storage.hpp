#pragma once

/// @file
/// Aligned allocation for matrix storage.
/// @ingroup topic-matrix-utils

#include <cstddef>
#include <memory>
#include <new>
#include <tuple>
#include <type_traits>

namespace batmat::matrix {

template <class A>
constexpr std::align_val_t as_align_val(A a) {
    if constexpr (std::is_integral_v<A>)
        return std::align_val_t{a};
    else
        return std::align_val_t{a()};
}

/// @addtogroup topic-matrix-utils
/// @{

struct uninitialized_t {
} inline constexpr uninitialized; ///< Tag type to indicate that memory should not be initialized.

/// Deleter for aligned memory allocated with `operator new(size, align_val)`.
template <class T, class A>
struct aligned_deleter {
    size_t size                   = 0;
    [[no_unique_address]] A align = {};
    void operator()(T *p) const {
        std::destroy_n(p, size);
        ::operator delete(p, size * sizeof(T), as_align_val(align));
    }
};

template <class A>
struct aligned_deleter<void, A> {
    size_t size                   = 0;
    [[no_unique_address]] A align = {};
    void operator()(void *p) const { ::operator delete(p, size, as_align_val(align)); }
};

/// @}

namespace detail {
template <class T, class A, bool Init>
auto make_aligned_unique_ptr(size_t size, A align) {
    if (size == 0)
        return std::unique_ptr<T[], aligned_deleter<T, A>>{};
    std::unique_ptr<void, aligned_deleter<void, A>> raw{
        ::operator new(size * sizeof(T), as_align_val(align)),
        {.size = size * sizeof(T), .align = align},
    };
    auto *uninitialized = std::launder(static_cast<T *>(raw.get()));
    if constexpr (Init)
        std::uninitialized_value_construct_n(uninitialized, size);
    else
        std::uninitialized_default_construct_n(uninitialized, size);
    std::ignore = raw.release();
    return std::unique_ptr<T[], aligned_deleter<T, A>>{
        uninitialized,
        {.size = size, .align = align},
    };
}
} // namespace detail

/// @addtogroup topic-matrix-utils
/// @{

/// Returns a smart pointer to an array of @p T that satisfies the given
/// alignment requirements.
template <class T, class A>
auto make_aligned_unique_ptr(size_t size, A align) {
    return detail::make_aligned_unique_ptr<T, A, true>(size, align);
}

/// Returns a smart pointer to an array of @p T that satisfies the given
/// alignment requirements.
template <class T, class A>
auto make_aligned_unique_ptr(size_t size, A align, uninitialized_t) {
    return detail::make_aligned_unique_ptr<T, A, false>(size, align);
}

/// @}

} // namespace batmat::matrix
