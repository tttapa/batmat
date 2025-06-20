#pragma once

#include <batmat/config.hpp>
#include <guanaqo/mat-view.hpp>

namespace batmat::matrix {

using guanaqo::StorageOrder;

template <class T>
struct integral_value_type {
    using type = T;
};
template <class IntConst>
    requires requires { typename IntConst::value_type; }
struct integral_value_type<IntConst> {
    using type = typename IntConst::value_type;
};
template <class T>
using integral_value_type_t = typename integral_value_type<T>::type;

struct DefaultStride {
    DefaultStride() = default;
    DefaultStride(index_t) {} // TODO: this is error prone
};

template <class I = ptrdiff_t, class S = std::integral_constant<I, 1>, class D = I,
          class L = DefaultStride, StorageOrder O = StorageOrder::ColMajor>
struct Layout {
    using index_type                            = I;
    using batch_size_type                       = S;
    using depth_type                            = D;
    using layer_stride_type                     = L;
    static constexpr StorageOrder storage_order = O;
    static constexpr bool is_column_major       = O == StorageOrder::ColMajor;
    static constexpr bool is_row_major          = O == StorageOrder::RowMajor;

    using standard_stride_type = std::conditional_t<requires {
        S::value;
    }, std::integral_constant<index_t, S::value>, index_t>;

    [[no_unique_address]] depth_type depth;
    index_type rows;
    index_type cols;
    index_type outer_stride;
    [[no_unique_address]] batch_size_type batch_size;
    [[no_unique_address]] layer_stride_type layer_stride;

    struct PlainLayout {
        [[no_unique_address]] depth_type depth = guanaqo::default_stride<depth_type>::value;
        index_type rows                        = 0;
        index_type cols                        = 1;
        index_type outer_stride                = is_row_major ? cols : rows;
        [[no_unique_address]] batch_size_type batch_size =
            guanaqo::default_stride<batch_size_type>::value;
        [[no_unique_address]] layer_stride_type layer_stride =
            outer_stride * (is_row_major ? rows : cols);
    };

    [[nodiscard]] constexpr index_type outer_size() const { return is_row_major ? rows : cols; }
    [[nodiscard]] constexpr index_type inner_size() const { return is_row_major ? cols : rows; }
    [[nodiscard]] constexpr index_type num_batches() const {
        const auto bs = static_cast<I>(batch_size);
        const auto d  = static_cast<I>(depth);
        return (d + bs - 1) / bs;
    }
    /// Round up the given size @p n to a multiple of @ref batch_size.
    [[nodiscard]] constexpr index_type ceil_depth(index_type n) const {
        const auto bs = static_cast<I>(batch_size);
        return n + (bs - n % bs) % bs;
    }
    /// Round up the @ref depth to a multiple of @ref batch_size.
    [[nodiscard]] constexpr index_type ceil_depth() const {
        return ceil_depth(static_cast<I>(depth));
    }
    /// Round down the given size @p n to a multiple of @ref batch_size.
    [[nodiscard]] constexpr index_type floor_depth(index_type n) const {
        const auto bs = static_cast<I>(batch_size);
        return n - (n % bs);
    }
    /// Round down the @ref depth to a multiple of @ref batch_size.
    [[nodiscard]] constexpr index_type floor_depth() const {
        return floor_depth(static_cast<I>(depth));
    }
    [[nodiscard]] constexpr auto get_layer_stride() const {
        if constexpr (std::is_same_v<layer_stride_type, DefaultStride>)
            return outer_stride * outer_size();
        else
            return layer_stride;
    }
    [[nodiscard]] bool has_full_layer_stride() const {
        return static_cast<index_t>(get_layer_stride()) == outer_stride * outer_size();
    }
    [[nodiscard]] constexpr index_type layer_index(index_type l, index_type s) const {
        assert(0 <= l && l < ceil_depth());
        const auto bs     = static_cast<I>(batch_size);
        index_type offset = l % bs;
        return s * (l - offset) + offset;
    }
    [[nodiscard]] constexpr index_type layer_index(index_type l) const {
        return layer_index(l, get_layer_stride());
    }

    constexpr Layout(PlainLayout p = {})
        : depth{p.depth}, rows{p.rows}, cols{p.cols}, outer_stride{p.outer_stride},
          batch_size{p.batch_size}, layer_stride{p.layer_stride} {}

    static standard_stride_type convert_to_standard_stride(auto s) {
        if constexpr (requires { standard_stride_type::value; })
            return {};
        else
            return static_cast<standard_stride_type>(s);
    }

    template <class T>
        requires is_column_major
    [[nodiscard]] guanaqo::MatrixView<T, I, standard_stride_type> operator()(T *data,
                                                                             index_type l) const {
        return {{.data         = data + layer_index(l),
                 .rows         = rows,
                 .cols         = cols,
                 .inner_stride = convert_to_standard_stride(batch_size),
                 .outer_stride = outer_stride * static_cast<I>(batch_size)}};
    }
    template <class T>
    [[nodiscard]] T &operator()(T *data, index_type l, index_type r, index_type c) const {
        auto *const p = data + layer_index(l);
        const auto bs = static_cast<I>(batch_size);
        return *(is_row_major ? p + bs * (c + outer_stride * r) : p + bs * (r + outer_stride * c));
    }
    [[nodiscard]] index_type size() const { return static_cast<I>(depth) * rows * cols; }
    [[nodiscard]] index_type padded_size() const { return ceil_depth() * get_layer_stride(); }
};

} // namespace batmat::matrix
