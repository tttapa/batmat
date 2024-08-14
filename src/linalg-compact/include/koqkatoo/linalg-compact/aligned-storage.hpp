#pragma once

#include <cstddef>
#include <memory>
#include <new>

namespace koqkatoo::linalg::compact {

/// Returns a smart pointer to an array of @p T that satisfies the given
/// alignment requirements.
template <class T, size_t Align>
auto make_aligned_unique_ptr(size_t size) {
    static constexpr auto raw_aligned_delete = [](void *p) {
        ::operator delete(p, std::align_val_t{Align});
    };
    struct aligned_delete {
        size_t size = 0;
        void operator()(T *p) const {
            std::destroy_n(p, size);
            raw_aligned_delete(p);
        }
    };
    if (size == 0)
        return std::unique_ptr<T[], aligned_delete>{};
    std::unique_ptr<void, decltype(raw_aligned_delete)> raw{
        ::operator new(size * sizeof(T), std::align_val_t{Align}),
        raw_aligned_delete,
    };
    auto *uninitialized = static_cast<T *>(raw.get());
    std::uninitialized_value_construct_n(uninitialized, size);
    raw.release();
    return std::unique_ptr<T[], aligned_delete>{
        uninitialized,
        {.size = size},
    };
}

template <class T>
struct aligned_deleter {
    size_t size  = 0;
    size_t align = 0;
    void operator()(T *p) const {
        std::destroy_n(p, size);
        ::operator delete(p, std::align_val_t{align});
    }
};

template <class T>
auto make_aligned_unique_ptr(size_t size, size_t align) {
    auto raw_aligned_delete = [align](void *p) {
        ::operator delete(p, std::align_val_t{align});
    };
    if (size == 0)
        return std::unique_ptr<T[], aligned_deleter<T>>{};
    std::unique_ptr<void, decltype(raw_aligned_delete)> raw{
        ::operator new(size * sizeof(T), std::align_val_t{align}),
        raw_aligned_delete,
    };
    auto *uninitialized = static_cast<T *>(raw.get());
    std::uninitialized_value_construct_n(uninitialized, size);
    raw.release();
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
    aligned_simd_storage(const aligned_simd_storage &other)
        : storage{make_aligned_unique_ptr<T, Align>(other.size())} {
        std::ranges::copy(other, begin());
    }
    aligned_simd_storage &operator=(const aligned_simd_storage &other) {
        storage = make_aligned_unique_ptr<T, Align>(other.size());
        std::ranges::copy(other, begin());
        return *this;
    }
    aligned_simd_storage(aligned_simd_storage &&)            = default;
    aligned_simd_storage &operator=(aligned_simd_storage &&) = default;

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
