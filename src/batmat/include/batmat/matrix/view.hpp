#pragma once

#include <batmat/assume.hpp>
#include <batmat/config.hpp>
#include <batmat/matrix/layout.hpp>
#include <guanaqo/mat-view.hpp>

namespace batmat::matrix {

/// @tparam T
///         Element value type (possibly const-qualified).
/// @tparam I
///         Index and size type. Usually `std::ptrdiff_t` or `int`.
/// @tparam S
///         Inner stride type (batch size). Usually `std::integral_constant<I, N>` for some `N`.
/// @tparam D
///         Batch depth type. Usually equal to @p S for a single batch, or @p I for a dynamic depth.
/// @tparam L
///         Layer stride type. Usually @ref DefaultStride (which implies that the layer stride is
///         equal to `outer_stride() * outer_size()`), or @p I for a dynamic layer stride.
///         Dynamic strides are used for subviews of views with a larger `outer_size()`.
/// @tparam O
///         Storage order (column or row major).
template <class T, class I = index_t, class S = std::integral_constant<I, 1>, class D = I,
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

    /// True if @ref batch_size() and @ref depth() are compile-time constants and are equal.
    /// @note Views with dynamic batch size and depth may still have a single batch at runtime,
    ///       but this cannot be statically asserted.
    static constexpr bool has_single_batch_at_compile_time = requires {
        S::value;
        D::value;
    } && S{} == D{};
    /// True if @ref depth() is a compile-time constant and is equal to one.
    static constexpr bool has_single_layer_at_compile_time = requires { D::value; } && D{} == 1;
    /// When extracing a single batch, the depth equals the batch size, and the layer stride is no
    /// longer relevant.
    using batch_view_type = View<T, I, S, S, DefaultStride, O>;
    /// When slicing along the outer dimension, the layer stride stays the same, but the outer size
    /// may be smaller, which means that even if the original view has a default layer stride, the
    /// sliced view may require a dynamic layer stride. For a single batch, the layer stride is not
    /// relevant, so it is preserved.
    using general_slice_view_type =
        std::conditional_t<has_single_batch_at_compile_time, View, View<T, I, S, D, I, O>>;
    /// View with the correct layer stride when slicing along the column dimension.
    /// For row-major storage, slicing along columns does not change the outer size, in which case
    /// the layer stride is still correct. For column-major storage, slicing along columns does
    /// change the outer size, in which case we need a dynamic layer stride.
    using col_slice_view_type = std::conditional_t<is_row_major, View, general_slice_view_type>;
    /// View with the correct layer stride when slicing along the row dimension.
    /// @see @ref col_slice_view_type
    using row_slice_view_type = std::conditional_t<is_column_major, View, general_slice_view_type>;

    /// Pointer to the first element of the first layer.
    value_type *data;
    /// Layout describing the dimensions and strides of the view.
    layout_type layout;

    /// POD helper struct to enable designated initializers during construction.
    struct PlainBatchedMatrixView {
        value_type *data                       = nullptr;
        [[no_unique_address]] depth_type depth = guanaqo::default_stride<depth_type>::value;
        index_type rows                        = 0;
        index_type cols                        = rows == 0 ? 0 : 1;
        index_type outer_stride                = is_row_major ? cols : rows;
        [[no_unique_address]] batch_size_type batch_size =
            guanaqo::default_stride<batch_size_type>::value;
        [[no_unique_address]] layer_stride_type layer_stride =
            outer_stride * (is_row_major ? rows : cols);
    };

    /// Create a new view.
    /// @note It is recommended to use designated initializers for the arguments to avoid mistakes.
    constexpr View(PlainBatchedMatrixView p = {})
        : data{p.data}, layout{{.depth        = p.depth,
                                .rows         = p.rows,
                                .cols         = p.cols,
                                .outer_stride = p.outer_stride,
                                .batch_size   = p.batch_size,
                                .layer_stride = p.layer_stride}} {}
    /// Create a new view with the given layout, using the given buffer.
    constexpr View(std::span<T> data, layout_type layout) : data{data.data()}, layout{layout} {
        BATMAT_ASSERT(data.size() == layout.padded_size());
    }
    /// Create a new view with the given layout, using the given buffer.
    constexpr View(value_type *buffer, layout_type layout) : data{buffer}, layout{layout} {}

    /// Copy a view. No data is copied.
    View(const View &) = default;

    /// Non-const views implicitly convert to const views.
    operator const_view_type() const
        requires(!std::is_const_v<T>)
    {
        return {data, layout};
    }
    /// Explicit conversion to a const view.
    [[nodiscard]] const_view_type as_const() const { return *this; }

    /// If we have a single layer at compile time, we can implicitly convert to a non-batched view.
    operator guanaqo::MatrixView<T, I, standard_stride_type, O>() const
        requires has_single_layer_at_compile_time
    {
        return operator()(0);
    }

    /// Access a single layer @p l as a non-batched view.
    /// @note   The inner stride of the returned view is equal to the batch size of this view, so it
    ///         cannot be used directly with functions that require unit inner stride (e.g. BLAS).
    [[nodiscard]] guanaqo::MatrixView<T, I, standard_stride_type, O>
    operator()(index_type l) const {
        return layout(data, l);
    }

    /// Access a single element at layer @p l, row @p r and column @p c.
    [[nodiscard]] value_type &operator()(index_type l, index_type r, index_type c) const {
        return layout(data, l, r, c);
    }

    /// @name Batch-wise slicing
    /// @{

    /// Access a batch of @ref batch_size() layers, starting at batch index @p b (i.e. starting at
    /// layer `b * batch_size()`).
    [[nodiscard]] batch_view_type batch(index_type b) const {
        const auto layer = b * static_cast<index_t>(batch_size());
        return {{.data         = data + layout.layer_index(layer),
                 .depth        = batch_size(),
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = batch_size()}};
    }

    /// Same as @ref batch(), but returns a view with a dynamic batch size. If the total depth is
    /// not a multiple of the batch size, the last batch will have a smaller size.
    [[nodiscard]] View<T, I, S, I, L, O> batch_dyn(index_type b) const {
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

    /// Get a view of the first @p n layers. Note that @p n can be a compile-time constant.
    template <class N>
    [[nodiscard]] View<T, I, S, N, L, O> first_layers(N n) const {
        BATMAT_ASSERT(n <= depth());
        return {{.data         = data,
                 .depth        = n,
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = batch_size(),
                 .layer_stride = layout.layer_stride}};
    }

    /// Get a view of @p n layers starting at layer @p l. Note that @p n can be a compile-time
    /// constant.
    /// @pre `l % batch_size() == 0` (i.e. the starting layer must be at the start of a batch).
    template <class N>
    [[nodiscard]] View<T, I, S, N, L, O> middle_layers(index_type l, N n) const {
        BATMAT_ASSERT(l + n <= depth());
        BATMAT_ASSERT(l % static_cast<I>(batch_size()) == 0);
        return {{.data         = data + layout.layer_index(l),
                 .depth        = n,
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = batch_size(),
                 .layer_stride = layout.layer_stride}};
    }

    /// @}

    /// @name Iterators and buffer access
    /// @{

    /// Iterator over all elements of a view.
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

    /// Iterate linearly (in storage order) over all elements of the view.
    /// @pre `has_full_outer_stride()` (i.e. no padding within layers).
    /// @pre `has_full_layer_stride()` (i.e. no padding between batches).
    [[nodiscard]] linear_iterator begin() const {
        BATMAT_ASSERT(has_full_outer_stride());
        BATMAT_ASSERT(has_full_layer_stride());
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
    /// Sentinel for @ref begin().
    [[nodiscard]] std::default_sentinel_t end() const { return {}; }

    /// @name Dimensions and strides
    /// @{

    /// Total number of elements in the view (excluding padding).
    [[nodiscard]] constexpr index_type size() const { return layout.size(); }
    /// Total number of elements in the view (including all padding).
    [[nodiscard]] constexpr index_type padded_size() const { return layout.padded_size(); }

    /// Number of layers in the view (i.e. depth).
    [[nodiscard, gnu::always_inline]] constexpr depth_type depth() const { return layout.depth; }
    /// The depth rounded up to a multiple of the batch size.
    [[nodiscard, gnu::always_inline]] constexpr index_type ceil_depth() const {
        return layout.ceil_depth();
    }
    /// Number of batches in the view, i.e. `ceil_depth() / batch_size()`.
    [[nodiscard, gnu::always_inline]] constexpr index_type num_batches() const {
        return layout.num_batches();
    }
    /// Number of rows of the matrices.
    [[nodiscard, gnu::always_inline]] constexpr index_type rows() const { return layout.rows; }
    /// Number of columns of the matrices.
    [[nodiscard, gnu::always_inline]] constexpr index_type cols() const { return layout.cols; }
    /// Outer stride of the matrices (leading dimension in BLAS parlance). Should be multiplied by
    /// the batch size to get the actual number of elements.
    [[nodiscard, gnu::always_inline]] constexpr index_type outer_stride() const {
        return layout.outer_stride;
    }
    /// The size of the outer dimension, i.e. the number of columns for column-major storage, or the
    /// number of rows for row-major storage.
    [[nodiscard, gnu::always_inline]] constexpr index_type outer_size() const {
        return layout.outer_size();
    }
    /// The size of the inner dimension, i.e. the number of rows for column-major storage, or the
    /// number of columns for row-major storage.
    [[nodiscard, gnu::always_inline]] constexpr index_type inner_size() const {
        return layout.inner_size();
    }
    /// The inner stride of the matrices. Should be multiplied by the batch size to get the actual
    /// number of elements.
    [[nodiscard, gnu::always_inline]] constexpr index_type inner_stride() const { return 1; }
    /// The batch size, i.e. the number of layers in each batch. Equals the inner stride.
    [[nodiscard, gnu::always_inline]] constexpr batch_size_type batch_size() const {
        return layout.batch_size;
    }
    /// The layer stride, i.e. the distance between the first layer of one batch and the first layer
    /// of the next batch. Should be multiplied by the batch size to get the actual number of
    /// elements.
    [[nodiscard, gnu::always_inline]] constexpr index_type layer_stride() const {
        return layout.get_layer_stride();
    }
    /// Whether the `layer_stride() == outer_stride() * outer_size()`.
    [[nodiscard, gnu::always_inline]] constexpr bool has_full_layer_stride() const {
        return layout.has_full_layer_stride();
    }
    /// Whether the `outer_stride() == inner_stride() * inner_size()`.
    [[nodiscard, gnu::always_inline]] constexpr bool has_full_outer_stride() const {
        return layout.has_full_outer_stride();
    }
    /// Whether the `inner_stride() == 1`. Always true.
    [[nodiscard, gnu::always_inline]] constexpr bool has_full_inner_stride() const {
        return layout.has_full_inner_stride();
    }

    /// @}

  private:
    template <class V>
    constexpr auto get_layer_stride_for() const {
        if constexpr (std::is_same_v<typename V::layer_stride_type, layer_stride_type>)
            return layout.layer_stride;
        else
            return layer_stride();
    }

  public:
    /// @name Reshaping and slicing
    /// @{

    /// Reshape the view to the given dimensions. The total size should not change.
    [[nodiscard]] general_slice_view_type reshaped(index_type rows, index_type cols) const {
        BATMAT_ASSERT(rows * cols == this->rows() * this->cols());
        BATMAT_ASSERT(has_full_outer_stride());
        return general_slice_view_type{typename general_slice_view_type::PlainBatchedMatrixView{
            .data         = data,
            .depth        = depth(),
            .rows         = rows,
            .cols         = cols,
            .outer_stride = is_row_major ? cols : rows,
            .batch_size   = batch_size(),
            .layer_stride = this->get_layer_stride_for<general_slice_view_type>()}};
    }

    /// Get a view of the first @p n rows.
    [[nodiscard]] row_slice_view_type top_rows(index_type n) const {
        BATMAT_ASSERT(0 <= n && n <= rows());
        return row_slice_view_type{typename row_slice_view_type::PlainBatchedMatrixView{
            .data         = data,
            .depth        = depth(),
            .rows         = n,
            .cols         = cols(),
            .outer_stride = outer_stride(),
            .batch_size   = batch_size(),
            .layer_stride = this->get_layer_stride_for<row_slice_view_type>()}};
    }

    /// Get a view of the first @p n columns.
    [[nodiscard]] col_slice_view_type left_cols(index_type n) const {
        BATMAT_ASSERT(0 <= n && n <= cols());
        return col_slice_view_type{typename col_slice_view_type::PlainBatchedMatrixView{
            .data         = data,
            .depth        = depth(),
            .rows         = rows(),
            .cols         = n,
            .outer_stride = outer_stride(),
            .batch_size   = batch_size(),
            .layer_stride = this->get_layer_stride_for<col_slice_view_type>()}};
    }

    /// Get a view of the last @p n rows.
    [[nodiscard]] row_slice_view_type bottom_rows(index_type n) const {
        BATMAT_ASSERT(0 <= n && n <= rows());
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

    /// Get a view of the last @p n columns.
    [[nodiscard]] col_slice_view_type right_cols(index_type n) const {
        BATMAT_ASSERT(0 <= n && n <= cols());
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

    /// Get a view of @p n rows starting at row @p r.
    [[nodiscard]] row_slice_view_type middle_rows(index_type r, index_type n) const {
        return bottom_rows(rows() - r).top_rows(n);
    }

    /// Get a view of @p n columns starting at column @p c.
    [[nodiscard]] col_slice_view_type middle_cols(index_type c, index_type n) const {
        return right_cols(cols() - c).left_cols(n);
    }

    /// Get a view of the top-left @p nr by @p nc block of the matrices.
    [[nodiscard]] general_slice_view_type top_left(index_type nr, index_type nc) const {
        return top_rows(nr).left_cols(nc);
    }

    /// Get a view of the top-right @p nr by @p nc block of the matrices.
    [[nodiscard]] general_slice_view_type top_right(index_type nr, index_type nc) const {
        return top_rows(nr).right_cols(nc);
    }

    /// Get a view of the bottom-left @p nr by @p nc block of the matrices.
    [[nodiscard]] general_slice_view_type bottom_left(index_type nr, index_type nc) const {
        return bottom_rows(nr).left_cols(nc);
    }

    /// Get a view of the bottom-right @p nr by @p nc block of the matrices.
    [[nodiscard]] general_slice_view_type bottom_right(index_type nr, index_type nc) const {
        return bottom_rows(nr).right_cols(nc);
    }

    /// Get a view of the @p nr by @p nc block of the matrices starting at row @p r and column @p c.
    [[nodiscard]] general_slice_view_type block(index_type r, index_type c, index_type nr,
                                                index_type nc) const {
        return middle_rows(r, nr).middle_cols(c, nc);
    }

    /// Get a view of the given span as a column vector.
    [[nodiscard]] static View as_column(std::span<T> v) {
        return {{.data = v.data(), .rows = static_cast<index_type>(v.size()), .cols = 1}};
    }

    /// Get a transposed view of the matrices. Note that the data itself is not modified, the
    /// returned view simply accesses the same data with rows and column indices swapped.
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

    /// @}

    /// @name Value manipulation
    /// @{

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

    template <class Other>
    void copy_values(const Other &other) const {
        static_assert(is_row_major == Other::is_row_major);
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

    /// Copy assignment copies the values from another view with the same layout to this view.
    View &operator=(const View &other) {
        if (this != &other)
            copy_values(other);
        return *this;
    }
    /// Copy values from another view with a compatible value type and the same layout to this view.
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

    /// @}

    /// Reassign the buffer and layout of this view to those of another view. No data is copied.
    View &reassign(View other) {
        this->data   = other.data;
        this->layout = other.layout;
        return *this;
    }

    /// Implicit conversion to a view with a dynamic depth.
    operator View<T, I, S, integral_value_type_t<D>, L, O>() const
        requires(!std::same_as<integral_value_type_t<D>, D>)
    {
        const auto bs = static_cast<integral_value_type_t<D>>(batch_size());
        return {{.data         = data,
                 .depth        = depth(),
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = bs,
                 .layer_stride = layout.layer_stride}};
    }
    /// Implicit conversion to a view with a dynamic depth, going from non-const to const.
    operator View<const T, I, S, integral_value_type_t<D>, L, O>() const
        requires(!std::is_const_v<T> && !std::same_as<integral_value_type_t<D>, D>)
    {
        const auto bs = static_cast<integral_value_type_t<D>>(batch_size());
        return {{.data         = data,
                 .depth        = depth(),
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = bs,
                 .layer_stride = layout.layer_stride}};
    }
    /// Implicit conversion to a view with a dynamic layer stride.
    operator View<T, I, S, D, I, O>() const
        requires(!std::same_as<I, L>)
    {
        return {{.data         = data,
                 .depth        = depth(),
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = batch_size(),
                 .layer_stride = layer_stride()}};
    }
    /// Implicit conversion to a view with a dynamic layer stride, going from non-const to const.
    operator View<const T, I, S, D, I, O>() const
        requires(!std::is_const_v<T> && !std::same_as<I, L>)
    {
        return {{.data         = data,
                 .depth        = depth(),
                 .rows         = rows(),
                 .cols         = cols(),
                 .outer_stride = outer_stride(),
                 .batch_size   = batch_size(),
                 .layer_stride = layer_stride()}};
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
