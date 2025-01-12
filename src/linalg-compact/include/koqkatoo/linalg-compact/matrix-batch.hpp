/**
 * @file
 * Provides layouts, views and owning batches of matrices (interleaved rank-3
 * tensors).
 */

#pragma once

#include <koqkatoo/linalg-compact/aligned-storage.hpp>
#include <koqkatoo/matrix-view.hpp>
#include <concepts>
#include <iterator>
#include <type_traits>
#include <utility>

namespace koqkatoo::linalg::compact {

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
    DefaultStride(index_t) {}
};

template <class I = ptrdiff_t, class S = std::integral_constant<I, 1>,
          class D = I, class L = DefaultStride>
struct BatchedMatrixLayout {
    using index_type           = I;
    using batch_size_type      = S;
    using depth_type           = D;
    using layer_stride_type    = L;
    using standard_stride_type = std::conditional_t<requires {
        S::value;
    }, std::integral_constant<index_t, S::value>, index_t>;

    [[no_unique_address]] depth_type depth;
    index_type rows;
    index_type cols;
    index_type outer_stride;
    [[no_unique_address]] batch_size_type batch_size;
    [[no_unique_address]] layer_stride_type layer_stride;

    struct PlainBatchedMatrixLayout {
        [[no_unique_address]] depth_type depth =
            guanaqo::default_stride<depth_type>::value;
        index_type rows         = 0;
        index_type cols         = 1;
        index_type outer_stride = rows;
        [[no_unique_address]] batch_size_type batch_size =
            guanaqo::default_stride<batch_size_type>::value;
        [[no_unique_address]] layer_stride_type layer_stride =
            outer_stride * cols;
    };

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
            return outer_stride * cols;
        else
            return layer_stride;
    }
    [[nodiscard]] bool has_full_layer_stride() const {
        return static_cast<index_t>(get_layer_stride()) == outer_stride * cols;
    }
    [[nodiscard]] constexpr index_type layer_index(index_type l,
                                                   index_type s) const {
        assert(0 <= l && l < ceil_depth());
        const auto bs     = static_cast<I>(batch_size);
        index_type offset = l % bs;
        return s * (l - offset) + offset;
    }
    [[nodiscard]] constexpr index_type layer_index(index_type l) const {
        return layer_index(l, get_layer_stride());
    }

    constexpr BatchedMatrixLayout(PlainBatchedMatrixLayout p = {})
        : depth{p.depth}, rows{p.rows}, cols{p.cols},
          outer_stride{p.outer_stride}, batch_size{p.batch_size},
          layer_stride{p.layer_stride} {}

    static standard_stride_type convert_to_standard_stride(auto s) {
        if constexpr (requires { standard_stride_type::value; })
            return {};
        else
            return static_cast<standard_stride_type>(s);
    }

    template <class T>
    [[nodiscard]] guanaqo::MatrixView<T, I, standard_stride_type>
    operator()(T *data, index_type l) const {
        return {{.data         = data + layer_index(l),
                 .rows         = rows,
                 .cols         = cols,
                 .inner_stride = convert_to_standard_stride(batch_size),
                 .outer_stride = outer_stride * static_cast<I>(batch_size)}};
    }
    template <class T>
    [[nodiscard]] T &operator()(T *data, index_type l, index_type r,
                                index_type c) const {
        return operator()(data, l)(r, c);
    }
    [[nodiscard]] index_type size() const {
        return static_cast<I>(depth) * rows * cols;
    }
    [[nodiscard]] index_type padded_size() const {
        return ceil_depth() * get_layer_stride();
    }
};

template <class T, class I = ptrdiff_t, class S = std::integral_constant<I, 1>,
          class D = I, class L = DefaultStride>
struct BatchedMatrixViewShorterLastColumn {
    using layout_type       = BatchedMatrixLayout<I, S, D, L>;
    using value_type        = T;
    using index_type        = typename layout_type::index_type;
    using batch_size_type   = typename layout_type::batch_size_type;
    using layer_stride_type = typename layout_type::layer_stride_type;

    value_type *data;
    layout_type layout;
    index_type rows_last;

    struct linear_iterator {
        using value_type      = T;
        using reference       = T &;
        using difference_type = I;
        T *data;
        T *end;
        T *next_jump;
        T *short_jump;
        I padding_size;
        I batch_size;
        T *begin = data;

        linear_iterator &operator++() {
            ++data;
            if (data == next_jump && data != end) [[unlikely]] {
                if (data == short_jump)
                    next_jump += batch_size - 1;
                else
                    next_jump += batch_size;
                if (data > short_jump)
                    data += padding_size + 1;
                else
                    data += padding_size;
            }
            return *this;
        }
        linear_iterator operator++(int) {
            linear_iterator t = *this;
            ++*this;
            return t;
        }
        reference operator*() const { return *data; }
        bool operator==(std::default_sentinel_t) const { return data == end; }
    };

    [[nodiscard]] linear_iterator begin() const {
        assert(layout.rows == layout.outer_stride);
        assert(layout.has_full_layer_stride());
        // Edge case for empty views
        if (layout.rows == 0) {
            return {.data         = data,
                    .end          = data,
                    .next_jump    = nullptr,
                    .short_jump   = data,
                    .padding_size = 0,
                    .batch_size   = 0};
        }
        // Standard case without a shorter last row
        else if (layout.rows == rows_last) {
            // Number of elements in each layer
            const auto size = layout.rows * layout.cols;
            // How many layers are in batches that are completely full?
            const auto contig_layers = layout.floor_depth();
            // Remaining layers have padding we should skip over (in the last batch)
            const auto remaining_layers = layout.depth - contig_layers;
            // Index of the first padding element
            const auto first_jump = contig_layers * size + remaining_layers;
            // Index of last layer in our storage
            const auto padded_end     = layout.ceil_depth();
            const auto padding_layers = padded_end - layout.depth;
            const auto end            = padded_end * size - padding_layers;
            const auto batch_size     = static_cast<I>(layout.batch_size);
            return {.data      = data,
                    .end       = data + end,
                    .next_jump = remaining_layers ? data + first_jump : nullptr,
                    .short_jump   = data + end,
                    .padding_size = batch_size - remaining_layers,
                    .batch_size   = batch_size};
        }
        // Case without the last row (can be reduced to standard case)
        else if (rows_last == 0) {
            auto actual_layout = layout;
            --actual_layout.depth;
            BatchedMatrixViewShorterLastColumn actual_view{
                .data      = data,
                .layout    = actual_layout,
                .rows_last = actual_layout.rows,
            };
            return actual_view.begin();
        }
        // Number of elements in each layer
        const auto size = layout.rows * layout.cols;
        // Number of elements in the last layer
        const auto size_last = rows_last * layout.cols;
        // How many layers in total?
        const auto depth = static_cast<I>(layout.depth);
        // How many layers are in batches that are completely full?
        const auto contig_layers = layout.floor_depth(depth - 1);
        // Remaining layers have padding we should skip over (in the last batch)
        const auto remaining_layers = depth - contig_layers;
        // Index of the first padding element
        const auto first_jump = contig_layers * size + remaining_layers;
        // Index of last layer in our storage
        const auto padded_end     = layout.ceil_depth();
        const auto padding_layers = padded_end - depth;
        // Index past the end of the range
        const auto end        = padded_end * size - padding_layers - 1;
        const auto batch_size = static_cast<I>(layout.batch_size);
        // If the last layer is smaller, at some point we need to jump sooner
        const auto short_jump = first_jump + (size_last - 1) * batch_size;
        // TODO: optimize case where there's no padding layers
        // TODO: handle remaining_layers == 1 more elegantly
        return {.data       = data,
                .end        = data + (remaining_layers == 1 ? short_jump : end),
                .next_jump  = data + first_jump,
                .short_jump = data + short_jump,
                .padding_size = batch_size - remaining_layers,
                .batch_size   = batch_size};
    }
    [[nodiscard]] std::default_sentinel_t end() const { return {}; }
    [[nodiscard]] index_type size() const {
        return layout.size() - layout.rows * layout.cols +
               rows_last * layout.cols;
    }
};

template <class T, class I = ptrdiff_t, class S = std::integral_constant<I, 1>,
          class D = I, class L = DefaultStride>
struct BatchedMatrixView {
    using layout_type          = BatchedMatrixLayout<I, S, D, L>;
    using value_type           = T;
    using index_type           = typename layout_type::index_type;
    using batch_size_type      = typename layout_type::batch_size_type;
    using depth_type           = typename layout_type::depth_type;
    using layer_stride_type    = typename layout_type::layer_stride_type;
    using standard_stride_type = typename layout_type::standard_stride_type;
    static constexpr bool has_single_batch = requires {
        S::value;
        D::value;
    } && S{} == D{};
    using batch_view_type = BatchedMatrixView<T, I, S, S>;
    using col_slice_view_type =
        std::conditional_t<has_single_batch, BatchedMatrixView,
                           BatchedMatrixView<T, I, S, D, I>>;

    value_type *data;
    layout_type layout;

    struct PlainBatchedMatrixView {
        value_type *data = nullptr;
        [[no_unique_address]] depth_type depth =
            guanaqo::default_stride<depth_type>::value;
        index_type rows         = 0;
        index_type cols         = 1;
        index_type outer_stride = rows;
        [[no_unique_address]] batch_size_type batch_size =
            guanaqo::default_stride<batch_size_type>::value;
        [[no_unique_address]] layer_stride_type layer_stride =
            outer_stride * cols;
    };

    constexpr BatchedMatrixView(PlainBatchedMatrixView p = {})
        : data{p.data}, layout{{.depth        = p.depth,
                                .rows         = p.rows,
                                .cols         = p.cols,
                                .outer_stride = p.outer_stride,
                                .batch_size   = p.batch_size,
                                .layer_stride = p.layer_stride}} {}
    constexpr BatchedMatrixView(std::span<T> data, layout_type layout)
        : data{data.data()}, layout{layout} {
        assert(data.size() == layout.padded_size());
    }
    constexpr BatchedMatrixView(value_type *data, layout_type layout)
        : data{data}, layout{layout} {}

    operator BatchedMatrixView<const T, I, S, D, L>() const
        requires(!std::is_const_v<T>)
    {
        return {data, layout};
    }

    BatchedMatrixView<const T, I, S, D, L> as_const() const { return *this; }

    [[nodiscard]] guanaqo::MatrixView<T, I, standard_stride_type>
    operator()(index_type l) const {
        return layout(data, l);
    }
    [[nodiscard]] value_type &operator()(index_type l, index_type r,
                                         index_type c) const {
        return layout(data, l, r, c);
    }

    [[nodiscard]] batch_view_type batch(index_type b) const {
        const auto layer = b * static_cast<index_t>(batch_size());
        return {{.data         = data + layout.layer_index(layer),
                 .depth        = batch_size(),
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = batch_size()}};
    }

    [[nodiscard]] BatchedMatrixView batch_dyn(index_type b) const {
        const auto d     = static_cast<I>(depth());
        const auto layer = b * static_cast<index_t>(batch_size());
        const auto last  = b == d / batch_size();
        return {{.data         = data + layout.layer_index(layer),
                 .depth        = last ? d - layout.floor_depth() : batch_size(),
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = batch_size(),
                 .layer_stride = layout.layer_stride}};
    }

    [[nodiscard]] BatchedMatrixView first_layers(index_type b) const {
        assert(b <= depth());
        return {{.data         = data,
                 .depth        = b,
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = batch_size(),
                 .layer_stride = layout.layer_stride}};
    }

    template <class N>
    [[nodiscard]] BatchedMatrixView<T, I, S, N, L> middle_layers(index_type l,
                                                                 N n) const {
        assert(l + n < depth());
        return {{.data         = data + layout.layer_index(l),
                 .depth        = n,
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = batch_size(),
                 .layer_stride = layout.layer_stride}};
    }

    struct linear_iterator {
        using value_type      = T;
        using reference       = T &;
        using difference_type = I;
        T *data;
        T *end;
        T *next_jump;
        I padding_size;
        I batch_size;

        linear_iterator &operator++() {
            ++data;
            if (data == next_jump && data != end) {
                data += padding_size;
                next_jump += batch_size;
            }
            return *this;
        }
        linear_iterator operator++(int) {
            linear_iterator t = *this;
            ++*this;
            return t;
        }
        reference operator*() const { return *data; }
        bool operator==(std::default_sentinel_t) const { return data == end; }
    };

    [[nodiscard]] linear_iterator begin() const {
        assert(rows() == outer_stride());
        assert(layout.has_full_layer_stride());
        // Number of elements in each layer
        const auto size = layout.rows * layout.cols;
        // How many layers are in batches that are completely full?
        const auto contig_layers = layout.floor_depth();
        // How many layers in total?
        const auto depth = static_cast<I>(layout.depth);
        // Remaining layers have padding we should skip over (in the last batch)
        const auto remaining_layers = depth - contig_layers;
        // Index of the first padding element
        const auto first_jump = contig_layers * size + remaining_layers;
        // Index of last layer in our storage
        const auto padded_end     = layout.ceil_depth();
        const auto padding_layers = padded_end - depth;
        const auto end            = padded_end * size - padding_layers;
        const auto batch_size     = static_cast<I>(layout.batch_size);
        return {
            .data         = data,
            .end          = data + end,
            .next_jump    = remaining_layers ? data + first_jump : nullptr,
            .padding_size = batch_size - remaining_layers,
            .batch_size   = batch_size,
        };
    }
    [[nodiscard]] std::default_sentinel_t end() const { return {}; }
    [[nodiscard]] index_type size() const { return layout.size(); }
    [[nodiscard]] index_type padded_size() const {
        return layout.padded_size();
    }

    [[nodiscard]] BatchedMatrixViewShorterLastColumn<T, I, S>
    linear_view_with_short_last_layer(index_type rows_last) const {
        assert(cols() == 1 && "Only column vectors are supported");
        return {data, layout, rows_last};
    }

    [[nodiscard, gnu::always_inline]] depth_type depth() const {
        return layout.depth;
    }
    [[nodiscard, gnu::always_inline]] index_type ceil_depth() const {
        return layout.ceil_depth();
    }
    [[nodiscard, gnu::always_inline]] index_type num_batches() const {
        return layout.num_batches();
    }
    [[nodiscard, gnu::always_inline]] index_type rows() const {
        return layout.rows;
    }
    [[nodiscard, gnu::always_inline]] index_type cols() const {
        return layout.cols;
    }
    [[nodiscard, gnu::always_inline]] index_type outer_stride() const {
        return layout.outer_stride;
    }
    [[nodiscard, gnu::always_inline]] batch_size_type batch_size() const {
        return layout.batch_size;
    }
    [[nodiscard, gnu::always_inline]] index_type layer_stride() const {
        return layout.get_layer_stride();
    }
    [[nodiscard, gnu::always_inline]] bool has_full_layer_stride() const {
        return layout.has_full_layer_stride();
    }

    [[nodiscard]] BatchedMatrixView top_rows(index_type n) const {
        assert(0 <= n && n <= rows());
        return BatchedMatrixView{
            PlainBatchedMatrixView{.data         = data,
                                   .depth        = depth(),
                                   .rows         = n,
                                   .cols         = cols(),
                                   .outer_stride = outer_stride(),
                                   .batch_size   = batch_size(),
                                   .layer_stride = layout.layer_stride}};
    }
    [[nodiscard]] col_slice_view_type left_cols(index_type n) const {
        assert(0 <= n && n <= cols());
        return col_slice_view_type{
            typename col_slice_view_type::PlainBatchedMatrixView{
                .data         = data,
                .depth        = depth(),
                .rows         = rows(),
                .cols         = n,
                .outer_stride = outer_stride(),
                .batch_size   = batch_size(),
                .layer_stride = layer_stride()}};
    }
    [[nodiscard]] BatchedMatrixView bottom_rows(index_type n) const {
        assert(0 <= n && n <= rows());
        const auto bs     = static_cast<I>(batch_size());
        const auto offset = bs * (rows() - n);
        return BatchedMatrixView{
            PlainBatchedMatrixView{.data         = data + offset,
                                   .depth        = depth(),
                                   .rows         = n,
                                   .cols         = cols(),
                                   .outer_stride = outer_stride(),
                                   .batch_size   = batch_size(),
                                   .layer_stride = layout.layer_stride}};
    }
    [[nodiscard]] col_slice_view_type right_cols(index_type n) const {
        assert(0 <= n && n <= cols());
        const auto bs     = static_cast<I>(batch_size());
        const auto offset = bs * outer_stride() * (cols() - n);
        return col_slice_view_type{
            typename col_slice_view_type::PlainBatchedMatrixView{
                .data         = data + offset,
                .depth        = depth(),
                .rows         = rows(),
                .cols         = n,
                .outer_stride = outer_stride(),
                .batch_size   = batch_size(),
                .layer_stride = layer_stride()}};
    }
    [[nodiscard]] BatchedMatrixView middle_rows(index_type r,
                                                index_type n) const {
        return bottom_rows(rows() - r).top_rows(n);
    }
    [[nodiscard]] col_slice_view_type middle_cols(index_type c,
                                                  index_type n) const {
        return right_cols(cols() - c).left_cols(n);
    }
    [[nodiscard]] col_slice_view_type top_left(index_type nr,
                                               index_type nc) const {
        return top_rows(nr).left_cols(nc);
    }
    [[nodiscard]] col_slice_view_type top_right(index_type nr,
                                                index_type nc) const {
        return top_rows(nr).right_cols(nc);
    }
    [[nodiscard]] col_slice_view_type bottom_left(index_type nr,
                                                  index_type nc) const {
        return bottom_rows(nr).left_cols(nc);
    }
    [[nodiscard]] col_slice_view_type bottom_right(index_type nr,
                                                   index_type nc) const {
        return bottom_rows(nr).right_cols(nc);
    }
    [[nodiscard]] col_slice_view_type
    block(index_type r, index_type c, index_type nr, index_type nc) const {
        return middle_rows(r, nr).middle_cols(c, nc);
    }
    [[nodiscard]] static BatchedMatrixView as_column(std::span<T> v) {
        return {{
            .data = v.data(),
            .rows = static_cast<index_type>(v.size()),
            .cols = 1,
        }};
    }

    void add_to_diagonal(const value_type &t) {
        const auto bs = static_cast<I>(batch_size());
        const auto n  = std::min(rows(), cols());
        for (index_type b = 0; b < num_batches(); ++b) {
            auto *p = batch(b).data;
            for (index_type i = 0; i < n; ++i) {
                for (index_type r = 0; r < bs; ++r)
                    *p++ += t;
                p += bs * outer_stride();
            }
        }
    }

    void set_constant(value_type t) {
        const auto bs = static_cast<I>(batch_size());
        for (index_type b = 0; b < num_batches(); ++b) {
            auto *dst = this->batch(b).data;
            for (index_type c = 0; c < this->cols(); ++c) {
                auto *dst_         = dst;
                const index_type n = rows() * bs;
                for (index_type r = 0; r < n; ++r)
                    *dst_++ = t;
                dst += bs * this->outer_stride();
            }
        }
    }

    void negate() {
        const auto bs = static_cast<I>(batch_size());
        for (index_type b = 0; b < num_batches(); ++b) {
            auto *dst = this->batch(b).data;
            for (index_type c = 0; c < this->cols(); ++c) {
                auto *dst_         = dst;
                const index_type n = rows() * bs;
                for (index_type r = 0; r < n; ++r, ++dst_)
                    *dst_ = -*dst_;
                dst += bs * this->outer_stride();
            }
        }
    }

    void copy_values(auto &other) const {
        assert(other.rows() == this->rows());
        assert(other.cols() == this->cols());
        assert(other.batch_size() == this->batch_size());
        const auto bs = static_cast<I>(batch_size());
        for (index_type b = 0; b < num_batches(); ++b) {
            const auto *src = other.batch(b).data;
            auto *dst       = this->batch(b).data;
            for (index_type c = 0; c < this->cols(); ++c) {
                const auto *src_   = src;
                auto *dst_         = dst;
                const index_type n = rows() * bs;
                for (index_type r = 0; r < n; ++r)
                    *dst_++ = *src_++;
                src += bs * other.outer_stride();
                dst += bs * this->outer_stride();
            }
        }
    }
    BatchedMatrixView(const BatchedMatrixView &) = default;
    BatchedMatrixView &operator=(const BatchedMatrixView &other) {
        if (this != &other)
            copy_values(other);
        return *this;
    }
    template <class U, class J, class R, class E, class M>
        requires(!std::is_const_v<T> &&
                 std::convertible_to<U, std::remove_cv_t<T>> &&
                 std::equality_comparable_with<I, J>)
    BatchedMatrixView &operator=(BatchedMatrixView<U, J, R, E, M> other) {
        copy_values(other);
        return *this;
    }
    // TODO: abstract logic into generic function (and check performance)
    template <class U, class J, class R, class E, class M>
        requires(!std::is_const_v<T> &&
                 std::convertible_to<U, std::remove_cv_t<T>> &&
                 std::equality_comparable_with<I, J>)
    BatchedMatrixView &operator+=(BatchedMatrixView<U, J, R, E, M> other) {
        assert(other.rows() == this->rows());
        assert(other.cols() == this->cols());
        assert(other.batch_size() == this->batch_size());
        const auto bs = static_cast<I>(batch_size());
        for (index_type b = 0; b < num_batches(); ++b) {
            const auto *src = other.batch(b).data;
            auto *dst       = this->batch(b).data;
            for (index_type c = 0; c < this->cols(); ++c) {
                const auto *src_   = src;
                auto *dst_         = dst;
                const index_type n = rows() * bs;
                for (index_type r = 0; r < n; ++r)
                    *dst_++ += *src_++;
                src += bs * other.outer_stride();
                dst += bs * this->outer_stride();
            }
        }
        return *this;
    }

    BatchedMatrixView &reassign(BatchedMatrixView other) {
        this->data   = other.data;
        this->layout = other.layout;
        return *this;
    }

    operator BatchedMatrixView<T, I, S, integral_value_type_t<D>, L>() const
        requires(!std::same_as<integral_value_type_t<D>, D>)
    {
        const auto bs = static_cast<integral_value_type_t<D>>(batch_size());
        return {{
            .data         = data,
            .depth        = depth(),
            .rows         = rows(),
            .cols         = cols(),
            .outer_stride = outer_stride(),
            .batch_size   = bs,
            .layer_stride = layout.layer_stride,
        }};
    }
    operator BatchedMatrixView<T, I, S, D, I>() const
        requires(!std::same_as<I, L>)
    {
        return {{
            .data         = data,
            .depth        = depth(),
            .rows         = rows(),
            .cols         = cols(),
            .outer_stride = outer_stride(),
            .batch_size   = batch_size(),
            .layer_stride = layer_stride(),
        }};
    }
    operator BatchedMatrixView<const T, I, S, D, I>() const
        requires(!std::is_const_v<T> && !std::same_as<I, L>)
    {
        return {{
            .data         = data,
            .depth        = depth(),
            .rows         = rows(),
            .cols         = cols(),
            .outer_stride = outer_stride(),
            .batch_size   = batch_size(),
            .layer_stride = layer_stride(),
        }};
    }
};

template <class T, class I, class S, class D>
bool operator==(
    typename BatchedMatrixView<T, I, S, D>::std::default_sentinel_t s,
    typename BatchedMatrixView<T, I, S, D>::linear_iterator i) {
    return i == s;
}
template <class T, class I, class S, class D>
bool operator!=(
    typename BatchedMatrixView<T, I, S, D>::std::default_sentinel_t s,
    typename BatchedMatrixView<T, I, S, D>::linear_iterator i) {
    return !(i == s);
}
template <class T, class I, class S, class D>
bool operator!=(
    typename BatchedMatrixView<T, I, S, D>::linear_iterator i,
    typename BatchedMatrixView<T, I, S, D>::std::default_sentinel_t s) {
    return !(i == s);
}

namespace detail {

template <class, class I, class Stride>
struct default_alignment {
    using type = std::integral_constant<I, 0>;
};
template <class T, class I, class Stride>
    requires requires {
        { Stride::value } -> std::convertible_to<I>;
    }
struct default_alignment<T, I, Stride> {
    using type = std::integral_constant<I, alignof(T) * Stride::value>;
};
template <class T, class I, class Stride>
using default_alignment_t = typename default_alignment<T, I, Stride>::type;

} // namespace detail

template <class T, class I = ptrdiff_t, class S = std::integral_constant<I, 1>,
          class D = I, class A = detail::default_alignment_t<T, I, S>>
struct BatchedMatrix {
    static_assert(!std::is_const_v<T>);
    using view_type            = BatchedMatrixView<T, I, S, D>;
    using const_view_type      = BatchedMatrixView<const T, I, S, D>;
    using layout_type          = typename view_type::layout_type;
    using plain_layout_type    = typename layout_type::PlainBatchedMatrixLayout;
    using value_type           = T;
    using index_type           = typename layout_type::index_type;
    using batch_size_type      = typename layout_type::batch_size_type;
    using depth_type           = typename layout_type::depth_type;
    using standard_stride_type = typename layout_type::standard_stride_type;
    using alignment_type       = A;

    view_type view;

    static constexpr alignment_type default_alignment(layout_type layout) {
        if constexpr (alignment_type{} == 0)
            return alignof(T) * layout.batch_size;
        else
            return {};
    }

    layout_type layout() const { return view.layout; }

    void resize(layout_type new_layout) {
        if (new_layout.padded_size() != layout().padded_size()) {
            clear();
            view.data = allocate(new_layout).release();
        }
        view.layout = new_layout;
    }

    static auto allocate(layout_type layout) {
        const auto alignment = default_alignment(layout);
        return make_aligned_unique_ptr<T>(layout.padded_size(), alignment);
    }
    void clear() {
        const auto alignment = default_alignment(view.layout);
        if (auto d = std::exchange(view.data, nullptr))
            aligned_deleter<T>(view.layout.padded_size(), alignment)(d);
        view.layout.rows = 0;
    }

    void set_constant(value_type t) { view.set_constant(t); }

    BatchedMatrix() = default;
    BatchedMatrix(layout_type layout)
        : view{allocate(layout).release(), layout} {}
    BatchedMatrix(plain_layout_type p) : BatchedMatrix{layout_type{p}} {}

    BatchedMatrix(const BatchedMatrix &o) : BatchedMatrix{o.layout()} {
        this->view.copy_values(o.view); // TODO: exception safety
    }
    BatchedMatrix(BatchedMatrix &&o) noexcept : view{o.view} {
        o.view.reassign({});
    }
    BatchedMatrix &operator=(const BatchedMatrix &o) {
        if (&o != this) {
            clear();
            view.reassign({allocate(o.layout()).release(), o.layout()});
            // TODO: use allocate_for_overwrite or similar to avoid copy
            //       assignment
            this->view.copy_values(o.view); // TODO: exception safety
        }
        return *this;
    }
    BatchedMatrix &operator=(BatchedMatrix &&o) noexcept {
        using std::swap;
        if (&o != this) {
            swap(o.view.data, this->view.data);
            swap(o.view.layout, this->view.layout);
        }
        return *this;
    }
    ~BatchedMatrix() { clear(); }

    operator view_type() { return view; }
    operator const_view_type() const { return view; }
    operator BatchedMatrixView<T, I, S, D, I>() { return view; }
    operator BatchedMatrixView<const T, I, S, D, I>() const { return view; }

    [[nodiscard]] guanaqo::MatrixView<T, I, standard_stride_type>
    operator()(index_type l) {
        return view(l);
    }
    [[nodiscard]] value_type &operator()(index_type l, index_type r,
                                         index_type c) {
        return view(l, r, c);
    }
    [[nodiscard]] guanaqo::MatrixView<const T, I, standard_stride_type>
    operator()(index_type l) const {
        return view.as_const()(l);
    }
    [[nodiscard]] const value_type &operator()(index_type l, index_type r,
                                               index_type c) const {
        return view.as_const()(l, r, c);
    }

    [[nodiscard]] auto batch(index_type b) { return view.batch(b); }
    [[nodiscard]] auto batch(index_type b) const {
        return view.as_const().batch(b);
    }
    [[nodiscard]] auto batch_dyn(index_type b) { return view.batch_dyn(b); }
    [[nodiscard]] auto batch_dyn(index_type b) const {
        return view.as_const().batch_dyn(b);
    }

    [[nodiscard]] auto as_const() const { return view.as_const(); }
    [[nodiscard]] auto begin() { return view.begin(); }
    [[nodiscard]] auto begin() const { return view.as_const().begin(); }
    [[nodiscard]] auto end() { return view.end(); }
    [[nodiscard]] auto end() const { return view.as_const().end(); }
    [[nodiscard]] index_type size() const { return view.size(); }
    [[nodiscard]] index_type padded_size() const { return view.padded_size(); }

    [[gnu::always_inline]] value_type *data() { return view.data; }
    [[gnu::always_inline]] const value_type *data() const { return view.data; }
    [[gnu::always_inline]] depth_type depth() const { return view.depth(); }
    [[gnu::always_inline]] index_type ceil_depth() const {
        return view.ceil_depth();
    }
    [[gnu::always_inline]] index_type num_batches() const {
        return view.num_batches();
    }
    [[gnu::always_inline]] index_type rows() const { return view.rows(); }
    [[gnu::always_inline]] index_type cols() const { return view.cols(); }
    [[gnu::always_inline]] index_type outer_stride() const {
        return view.outer_stride();
    }
    [[gnu::always_inline]] batch_size_type batch_size() const {
        return view.batch_size();
    }
};

} // namespace koqkatoo::linalg::compact
