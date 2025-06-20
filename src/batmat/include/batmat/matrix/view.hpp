#pragma once

#include <batmat/config.hpp>
#include <batmat/matrix/layout.hpp>
#include <guanaqo/mat-view.hpp>

namespace batmat::matrix {

template <class T, class I = ptrdiff_t, class S = std::integral_constant<I, 1>, class D = I,
          class L = DefaultStride, StorageOrder O = StorageOrder::ColMajor>
struct View {
    using layout_type                           = Layout<I, S, D, L, O>;
    using value_type                            = T;
    using index_type                            = typename layout_type::index_type;
    using batch_size_type                       = typename layout_type::batch_size_type;
    using depth_type                            = typename layout_type::depth_type;
    using layer_stride_type                     = typename layout_type::layer_stride_type;
    using standard_stride_type                  = typename layout_type::standard_stride_type;
    using const_view_type                       = View<const T, I, S, D, L, O>;
    static constexpr StorageOrder storage_order = layout_type::storage_order;
    static constexpr bool is_column_major       = layout_type::is_column_major;
    static constexpr bool is_row_major          = layout_type::is_row_major;

    static constexpr bool has_single_batch = requires {
        S::value;
        D::value;
    } && S{} == D{};
    static constexpr bool has_single_layer_at_compile_time = requires { D::value; } && D{} == 1;
    using batch_view_type                                  = View<T, I, S, S, DefaultStride, O>;
    using general_slice_view_type =
        std::conditional_t<has_single_batch, View, View<T, I, S, D, I, O>>;
    using col_slice_view_type = std::conditional_t<is_row_major, View, general_slice_view_type>;
    using row_slice_view_type = std::conditional_t<is_column_major, View, general_slice_view_type>;

    value_type *data;
    layout_type layout;

    struct PlainBatchedMatrixView {
        value_type *data                       = nullptr;
        [[no_unique_address]] depth_type depth = guanaqo::default_stride<depth_type>::value;
        index_type rows                        = 0;
        index_type cols                        = 1;
        index_type outer_stride                = is_row_major ? cols : rows;
        [[no_unique_address]] batch_size_type batch_size =
            guanaqo::default_stride<batch_size_type>::value;
        [[no_unique_address]] layer_stride_type layer_stride =
            outer_stride * (is_row_major ? rows : cols);
    };

    constexpr View(PlainBatchedMatrixView p = {})
        : data{p.data}, layout{{.depth        = p.depth,
                                .rows         = p.rows,
                                .cols         = p.cols,
                                .outer_stride = p.outer_stride,
                                .batch_size   = p.batch_size,
                                .layer_stride = p.layer_stride}} {}
    constexpr View(std::span<T> data, layout_type layout) : data{data.data()}, layout{layout} {
        assert(data.size() == layout.padded_size());
    }
    constexpr View(value_type *data, layout_type layout) : data{data}, layout{layout} {}

    operator const_view_type() const
        requires(!std::is_const_v<T>)
    {
        return {data, layout};
    }

    operator guanaqo::MatrixView<T, I, standard_stride_type, O>() const
        requires has_single_layer_at_compile_time
    {
        return operator()(0);
    }

    const_view_type as_const() const { return *this; }

    [[nodiscard]] guanaqo::MatrixView<T, I, standard_stride_type, O>
    operator()(index_type l) const {
        return layout(data, l);
    }
    [[nodiscard]] value_type &operator()(index_type l, index_type r, index_type c) const {
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

    [[nodiscard]] View batch_dyn(index_type b) const {
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

    [[nodiscard]] View first_layers(index_type b) const {
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
    [[nodiscard]] View<T, I, S, N, L, O> middle_layers(index_type l, N n) const {
        assert(l + n <= depth());
        assert(l % static_cast<I>(batch_size()) == 0);
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
        assert(inner_size() == outer_stride());
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
    [[nodiscard]] index_type padded_size() const { return layout.padded_size(); }

    [[nodiscard, gnu::always_inline]] depth_type depth() const { return layout.depth; }
    [[nodiscard, gnu::always_inline]] index_type ceil_depth() const { return layout.ceil_depth(); }
    [[nodiscard, gnu::always_inline]] index_type num_batches() const {
        return layout.num_batches();
    }
    [[nodiscard, gnu::always_inline]] index_type rows() const { return layout.rows; }
    [[nodiscard, gnu::always_inline]] index_type cols() const { return layout.cols; }
    [[nodiscard, gnu::always_inline]] index_type outer_stride() const {
        return layout.outer_stride;
    }
    [[nodiscard, gnu::always_inline]] index_type outer_size() const { return layout.outer_size(); }
    [[nodiscard, gnu::always_inline]] index_type inner_size() const { return layout.inner_size(); }
    [[nodiscard, gnu::always_inline]] batch_size_type batch_size() const {
        return layout.batch_size;
    }
    [[nodiscard, gnu::always_inline]] index_type layer_stride() const {
        return layout.get_layer_stride();
    }
    [[nodiscard, gnu::always_inline]] bool has_full_layer_stride() const {
        return layout.has_full_layer_stride();
    }

    template <class V>
    constexpr auto get_layer_stride_for() const {
        if constexpr (std::is_same_v<typename V::layer_stride_type, layer_stride_type>)
            return layout.layer_stride;
        else
            return layer_stride();
    }

    [[nodiscard]] general_slice_view_type reshaped(index_type rows, index_type cols) const {
        assert(rows * cols == this->rows() * this->cols());
        assert(this->inner_size() == this->outer_stride() || this->inner_size() == 1);
        return general_slice_view_type{typename general_slice_view_type::PlainBatchedMatrixView{
            .data         = data,
            .depth        = depth(),
            .rows         = rows,
            .cols         = cols,
            .outer_stride = is_row_major ? cols : rows,
            .batch_size   = batch_size(),
            .layer_stride = this->get_layer_stride_for<general_slice_view_type>()}};
    }

    [[nodiscard]] row_slice_view_type top_rows(index_type n) const {
        assert(0 <= n && n <= rows());
        return row_slice_view_type{typename row_slice_view_type::PlainBatchedMatrixView{
            .data         = data,
            .depth        = depth(),
            .rows         = n,
            .cols         = cols(),
            .outer_stride = outer_stride(),
            .batch_size   = batch_size(),
            .layer_stride = this->get_layer_stride_for<row_slice_view_type>()}};
    }
    [[nodiscard]] col_slice_view_type left_cols(index_type n) const {
        assert(0 <= n && n <= cols());
        return col_slice_view_type{typename col_slice_view_type::PlainBatchedMatrixView{
            .data         = data,
            .depth        = depth(),
            .rows         = rows(),
            .cols         = n,
            .outer_stride = outer_stride(),
            .batch_size   = batch_size(),
            .layer_stride = this->get_layer_stride_for<col_slice_view_type>()}};
    }
    [[nodiscard]] row_slice_view_type bottom_rows(index_type n) const {
        assert(0 <= n && n <= rows());
        const auto bs     = static_cast<I>(batch_size());
        const auto offset = (is_row_major ? outer_stride() : 1) * bs * (rows() - n);
        return row_slice_view_type{typename row_slice_view_type::PlainBatchedMatrixView{
            .data         = data + offset,
            .depth        = depth(),
            .rows         = n,
            .cols         = cols(),
            .outer_stride = outer_stride(),
            .batch_size   = batch_size(),
            .layer_stride = this->get_layer_stride_for<row_slice_view_type>()}};
    }
    [[nodiscard]] col_slice_view_type right_cols(index_type n) const {
        assert(0 <= n && n <= cols());
        const auto bs     = static_cast<I>(batch_size());
        const auto offset = (is_row_major ? 1 : outer_stride()) * bs * (cols() - n);
        return col_slice_view_type{typename col_slice_view_type::PlainBatchedMatrixView{
            .data         = data + offset,
            .depth        = depth(),
            .rows         = rows(),
            .cols         = n,
            .outer_stride = outer_stride(),
            .batch_size   = batch_size(),
            .layer_stride = this->get_layer_stride_for<col_slice_view_type>()}};
    }
    [[nodiscard]] row_slice_view_type middle_rows(index_type r, index_type n) const {
        return bottom_rows(rows() - r).top_rows(n);
    }
    [[nodiscard]] col_slice_view_type middle_cols(index_type c, index_type n) const {
        return right_cols(cols() - c).left_cols(n);
    }
    [[nodiscard]] general_slice_view_type top_left(index_type nr, index_type nc) const {
        return top_rows(nr).left_cols(nc);
    }
    [[nodiscard]] general_slice_view_type top_right(index_type nr, index_type nc) const {
        return top_rows(nr).right_cols(nc);
    }
    [[nodiscard]] general_slice_view_type bottom_left(index_type nr, index_type nc) const {
        return bottom_rows(nr).left_cols(nc);
    }
    [[nodiscard]] general_slice_view_type bottom_right(index_type nr, index_type nc) const {
        return bottom_rows(nr).right_cols(nc);
    }
    [[nodiscard]] general_slice_view_type block(index_type r, index_type c, index_type nr,
                                                index_type nc) const {
        return middle_rows(r, nr).middle_cols(c, nc);
    }
    [[nodiscard]] static View as_column(std::span<T> v) {
        return {{
            .data = v.data(),
            .rows = static_cast<index_type>(v.size()),
            .cols = 1,
        }};
    }

    [[nodiscard]] auto transposed() const {
        using TpBm = View<T, I, S, D, L, transpose(O)>;
        return TpBm{typename TpBm::PlainBatchedMatrixView{.data         = data,
                                                          .depth        = depth(),
                                                          .rows         = cols(),
                                                          .cols         = rows(),
                                                          .outer_stride = outer_stride(),
                                                          .batch_size   = batch_size(),
                                                          .layer_stride = layout.layer_stride}};
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
            for (index_type c = 0; c < this->outer_size(); ++c) {
                auto *dst_         = dst;
                const index_type n = inner_size() * bs;
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
            for (index_type c = 0; c < this->outer_size(); ++c) {
                auto *dst_         = dst;
                const index_type n = inner_size() * bs;
                for (index_type r = 0; r < n; ++r, ++dst_)
                    *dst_ = -*dst_;
                dst += bs * this->outer_stride();
            }
        }
    }

    void copy_values(auto &other) const {
        static_assert(this->is_row_major == other.is_row_major);
        assert(other.rows() == this->rows());
        assert(other.cols() == this->cols());
        assert(other.batch_size() == this->batch_size());
        const auto bs = static_cast<I>(batch_size());
        for (index_type b = 0; b < num_batches(); ++b) {
            const auto *src = other.batch(b).data;
            auto *dst       = this->batch(b).data;
            for (index_type c = 0; c < this->outer_size(); ++c) {
                const auto *src_   = src;
                auto *dst_         = dst;
                const index_type n = inner_size() * bs;
                for (index_type r = 0; r < n; ++r)
                    *dst_++ = *src_++;
                src += bs * other.outer_stride();
                dst += bs * this->outer_stride();
            }
        }
    }
    View(const View &) = default;
    View &operator=(const View &other) {
        if (this != &other)
            copy_values(other);
        return *this;
    }
    template <class U, class J, class R, class E, class M>
        requires(!std::is_const_v<T> && std::convertible_to<U, std::remove_cv_t<T>> &&
                 std::equality_comparable_with<I, J>)
    View &operator=(View<U, J, R, E, M, O> other) {
        copy_values(other);
        return *this;
    }
    // TODO: abstract logic into generic function (and check performance)
    template <class U, class J, class R, class E, class M>
        requires(!std::is_const_v<T> && std::convertible_to<U, std::remove_cv_t<T>> &&
                 std::equality_comparable_with<I, J>)
    View &operator+=(View<U, J, R, E, M, O> other) {
        assert(other.rows() == this->rows());
        assert(other.cols() == this->cols());
        assert(other.batch_size() == this->batch_size());
        const auto bs = static_cast<I>(batch_size());
        for (index_type b = 0; b < num_batches(); ++b) {
            const auto *src = other.batch(b).data;
            auto *dst       = this->batch(b).data;
            for (index_type c = 0; c < this->outer_size(); ++c) {
                const auto *src_   = src;
                auto *dst_         = dst;
                const index_type n = inner_size() * bs;
                for (index_type r = 0; r < n; ++r)
                    *dst_++ += *src_++;
                src += bs * other.outer_stride();
                dst += bs * this->outer_stride();
            }
        }
        return *this;
    }

    View &reassign(View other) {
        this->data   = other.data;
        this->layout = other.layout;
        return *this;
    }

    operator View<T, I, S, integral_value_type_t<D>, L, O>() const
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
    operator View<T, I, S, D, I, O>() const
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
    operator View<const T, I, S, D, I, O>() const
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

template <class T, class I, class S, class D, class L, StorageOrder P>
bool operator==(std::default_sentinel_t s, typename View<T, I, S, D, L, P>::linear_iterator i) {
    return i == s;
}
template <class T, class I, class S, class D, class L, StorageOrder P>
bool operator!=(std::default_sentinel_t s, typename View<T, I, S, D, L, P>::linear_iterator i) {
    return !(i == s);
}
template <class T, class I, class S, class D, class L, StorageOrder P>
bool operator!=(typename View<T, I, S, D, L, P>::linear_iterator i, std::default_sentinel_t s) {
    return !(i == s);
}

} // namespace batmat::matrix
