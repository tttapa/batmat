#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <tuple>

namespace koqkatoo::linalg::compact {

struct uninitialized_t {};
inline constexpr uninitialized_t uninitialized;

template <class T, size_t Align = 0>
struct aligned_deleter {
    size_t size = 0;
    void operator()(T *p) const {
        std::destroy_n(p, size);
        ::operator delete(p, std::align_val_t{Align});
    }
};

template <class T>
struct aligned_deleter<T, 0> {
    size_t size  = 0;
    size_t align = 0;
    void operator()(T *p) const {
        std::destroy_n(p, size);
        ::operator delete(p, std::align_val_t{align});
    }
};

template <size_t Align>
struct aligned_deleter<void, Align> {
    void operator()(void *p) const {
        ::operator delete(p, std::align_val_t{Align});
    }
};

template <>
struct aligned_deleter<void, 0> {
    size_t align = 0;
    void operator()(void *p) const {
        ::operator delete(p, std::align_val_t{align});
    }
};

/// Returns a smart pointer to an array of @p T that satisfies the given
/// alignment requirements.
template <class T, size_t Align>
auto make_aligned_unique_ptr(size_t size) {
    static_assert(Align > 0);
    if (size == 0)
        return std::unique_ptr<T[], aligned_deleter<T, Align>>{};
    std::unique_ptr<void, aligned_deleter<void, Align>> raw{
        ::operator new(size * sizeof(T), std::align_val_t{Align}),
    };
    auto *uninitialized = static_cast<T *>(raw.get());
    std::uninitialized_value_construct_n(uninitialized, size);
    std::ignore = raw.release();
    return std::unique_ptr<T[], aligned_deleter<T, Align>>{
        uninitialized,
        {.size = size},
    };
}

/// Returns a smart pointer to an array of @p T that satisfies the given
/// alignment requirements.
template <class T, size_t Align>
auto make_aligned_unique_ptr(size_t size, uninitialized_t) {
    static_assert(Align > 0);
    if (size == 0)
        return std::unique_ptr<T[], aligned_deleter<T, Align>>{};
    std::unique_ptr<void, aligned_deleter<void, Align>> raw{
        ::operator new(size * sizeof(T), std::align_val_t{Align}),
    };
    auto *uninitialized = static_cast<T *>(raw.get());
    std::uninitialized_default_construct_n(uninitialized, size);
    std::ignore = raw.release();
    return std::unique_ptr<T[], aligned_deleter<T, Align>>{
        uninitialized,
        {.size = size},
    };
}

template <class T>
auto make_aligned_unique_ptr(size_t size, size_t align) {
    if (size == 0)
        return std::unique_ptr<T[], aligned_deleter<T>>{};
    std::unique_ptr<void, aligned_deleter<void>> raw{
        ::operator new(size * sizeof(T), std::align_val_t{align}),
        {.align = align},
    };
    auto *uninitialized = static_cast<T *>(raw.get());
    std::uninitialized_value_construct_n(uninitialized, size);
    std::ignore = raw.release();
    return std::unique_ptr<T[], aligned_deleter<T>>{
        uninitialized,
        {.size = size, .align = align},
    };
}

template <class T, size_t Align>
class aligned_simd_storage {
  public:
    explicit aligned_simd_storage(size_t size = 0)
        : storage{make_aligned_unique_ptr<T, Align>(size)} {}
    explicit aligned_simd_storage(size_t size, uninitialized_t t)
        : storage{make_aligned_unique_ptr<T, Align>(size, t)} {}
    aligned_simd_storage(const aligned_simd_storage &other)
        : storage{make_aligned_unique_ptr<T, Align>(other.size())} {
        std::ranges::copy(other, begin());
    }
    aligned_simd_storage &operator=(const aligned_simd_storage &other) {
        storage = make_aligned_unique_ptr<T, Align>(other.size());
        std::ranges::copy(other, begin());
        return *this;
    }
    aligned_simd_storage(aligned_simd_storage &&) noexcept            = default;
    aligned_simd_storage &operator=(aligned_simd_storage &&) noexcept = default;

    [[nodiscard]] size_t size() const { return storage.get_deleter().size; }
    [[nodiscard]] T *data() { return storage.get(); }
    [[nodiscard]] T *begin() { return data(); }
    [[nodiscard]] T *end() { return begin() + size(); }
    [[nodiscard]] const T *data() const { return storage.get(); }
    [[nodiscard]] const T *begin() const { return data(); }
    [[nodiscard]] const T *end() const { return begin() + size(); }
    T &operator[](size_t pos) { return data()[pos]; }
    const T &operator[](size_t pos) const { return data()[pos]; }

    static constexpr size_t alignment = Align;

  private:
    using storage_t = decltype(make_aligned_unique_ptr<T, Align>(0));
    storage_t storage;
};

} // namespace koqkatoo::linalg::compact
