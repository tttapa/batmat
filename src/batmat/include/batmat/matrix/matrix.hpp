#pragma once

/// @file
/// Class for a batch of matrices that owns its storage.
/// @ingroup topic-matrix

#include <batmat/matrix/storage.hpp>
#include <batmat/matrix/view.hpp>

#include <type_traits>
#include <utility>

namespace batmat::matrix {

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

static_assert(default_alignment_t<double, int, std::integral_constant<int, 4>>::value == 8 * 4);
static_assert(default_alignment_t<double, int, int>::value == 0);

} // namespace detail

/// Class for a batch of matrices that owns its storage.
/// @tparam T
///         Element value type.
/// @tparam I
///         Index type.
/// @tparam S
///         Inner stride (batch size).
/// @tparam D
///         Depth type.
/// @tparam O
///         Storage order (column or row major).
/// @tparam A
///         Batch alignment type.
/// @ingroup topic-matrix
template <class T, class I = index_t, class S = std::integral_constant<I, 1>, class D = I,
          StorageOrder O = StorageOrder::ColMajor, class A = detail::default_alignment_t<T, I, S>>
struct Matrix {
    static_assert(!std::is_const_v<T>);
    using view_type                             = View<T, I, S, D, DefaultStride, O>;
    using const_view_type                       = typename view_type::const_view_type;
    using layout_type                           = typename view_type::layout_type;
    using plain_layout_type                     = typename layout_type::PlainLayout;
    using value_type                            = T;
    using index_type                            = typename layout_type::index_type;
    using batch_size_type                       = typename layout_type::batch_size_type;
    using depth_type                            = typename layout_type::depth_type;
    using standard_stride_type                  = typename layout_type::standard_stride_type;
    using alignment_type                        = A;
    static constexpr StorageOrder storage_order = view_type::storage_order;
    static constexpr bool is_column_major       = view_type::is_column_major;
    static constexpr bool is_row_major          = view_type::is_row_major;
    static constexpr bool has_single_batch_at_compile_time =
        view_type::has_single_batch_at_compile_time;
    static constexpr bool has_single_layer_at_compile_time =
        view_type::has_single_layer_at_compile_time;

  private:
    view_type view_;

    static constexpr auto default_alignment(layout_type layout) {
        if constexpr (std::is_integral_v<alignment_type>)
            return alignof(T) * static_cast<size_t>(layout.batch_size); // TODO
        if constexpr (alignment_type::value == 0)
            return alignof(T) * static_cast<size_t>(layout.batch_size);
        else
            return alignment_type{};
    }
    [[nodiscard]] static auto allocate(layout_type layout) {
        const auto alignment = default_alignment(layout);
        return make_aligned_unique_ptr<T>(layout.padded_size(), alignment);
    }
    [[nodiscard]] static auto allocate(layout_type layout, uninitialized_t init) {
        const auto alignment = default_alignment(layout);
        return make_aligned_unique_ptr<T>(layout.padded_size(), alignment, init);
    }
    void clear() {
        const auto alignment = default_alignment(view_.layout);
        if (auto d = std::exchange(view_.data_ptr, nullptr))
            aligned_deleter<T, decltype(alignment)>(view_.layout.padded_size(), alignment)(d);
        view_.layout.rows = 0;
    }

    template <class U, class J, class R, class E, class M>
        requires(std::convertible_to<U, T> && std::equality_comparable_with<index_type, J>)
    void assign_from_view(View<U, J, R, E, M, O> other) {
        layout_type new_layout{{.depth      = static_cast<depth_type>(other.depth()),
                                .rows       = other.rows(),
                                .cols       = other.cols(),
                                .batch_size = static_cast<batch_size_type>(other.batch_size())}};
        resize(new_layout);
        view().copy_values(other); // TODO: exception safety
    }

  public:
    /// @name Constructors, assignment and resizing
    /// @{

    // TODO: allowing the user to specify strides during construction is kind of pointless,
    //       because it introduces many padding elements, which can never be accessed.

    Matrix() = default;
    Matrix(layout_type layout) : view_{allocate(layout).release(), layout} {}
    Matrix(layout_type layout, uninitialized_t init)
        : view_{allocate(layout, init).release(), layout} {}
    Matrix(plain_layout_type p) : Matrix{layout_type{p}} {}
    Matrix(plain_layout_type p, uninitialized_t init) : Matrix{layout_type{p}, init} {}
    /// Copy the values from another matrix.
    Matrix(const Matrix &o) : Matrix{o.layout()} {
        this->view().copy_values(o.view()); // TODO: exception safety
    }
    Matrix(Matrix &&o) noexcept : view_{o.view()} { o.view_.reassign({}); }
    /// Cheap move assignment. No data is copied.
    Matrix &operator=(Matrix &&o) noexcept {
        using std::swap;
        if (&o != this) {
            swap(o.view_.data_ptr, this->view_.data_ptr);
            swap(o.view_.layout, this->view_.layout);
        }
        return *this;
    }
    ~Matrix() { clear(); }

    /// Resize the matrix to a new layout, reallocating if the padded size changes.
    void resize(layout_type new_layout) {
        if (new_layout.padded_size() != layout().padded_size()) {
            clear();
            view_.data_ptr = allocate(new_layout).release();
        }
        view_.layout = new_layout;
    }

    /// @}

    /// @name Element access
    /// @{

    /// Access a single element at layer @p l, row @p r and column @p c.
    [[nodiscard]] value_type &operator()(index_type l, index_type r, index_type c) {
        return view()(l, r, c);
    }
    /// Access a single element at layer @p l, row @p r and column @p c.
    [[nodiscard]] const value_type &operator()(index_type l, index_type r, index_type c) const {
        return view()(l, r, c);
    }

    /// @}

    /// @name Batch-wise slicing
    /// @{

    /// @copydoc View::batch()
    [[nodiscard]] auto batch(index_type b) { return view().batch(b); }
    /// @copydoc View::batch()
    [[nodiscard]] auto batch(index_type b) const { return view().batch(b); }
    /// @copydoc View::batch_dyn()
    [[nodiscard]] auto batch_dyn(index_type b) { return view().batch_dyn(b); }
    /// @copydoc View::batch_dyn()
    [[nodiscard]] auto batch_dyn(index_type b) const { return view().batch_dyn(b); }
    /// @copydoc View::middle_batches()
    [[nodiscard]] auto middle_batches(index_type b, index_type n, index_type stride = 1) {
        return view().middle_batches(b, n, stride);
    }
    /// @copydoc View::middle_batches()
    [[nodiscard]] auto middle_batches(index_type b, index_type n, index_type stride = 1) const {
        return view().middle_batches(b, n, stride);
    }

    /// @}

    /// @name Layer-wise slicing
    /// @{

    /// Access a single layer @p l as a non-batched view.
    [[nodiscard]] guanaqo::MatrixView<T, I, standard_stride_type, O> operator()(index_type l) {
        return view()(l);
    }
    /// Access a single layer @p l as a non-batched view.
    [[nodiscard]] guanaqo::MatrixView<const T, I, standard_stride_type, O>
    operator()(index_type l) const {
        return view()(l);
    }
    /// @copydoc View::first_layers()
    template <class N>
    [[nodiscard]] auto first_layers(N n) {
        return view().first_layers(n);
    }
    /// @copydoc View::first_layers()
    template <class N>
    [[nodiscard]] auto first_layers(N n) const {
        return view().first_layers(n);
    }
    /// @copydoc View::middle_layers()
    template <class N>
    [[nodiscard]] auto middle_layers(index_type l, N n) {
        return view().middle_layers(l, n);
    }
    /// @copydoc View::middle_layers()
    template <class N>
    [[nodiscard]] auto middle_layers(index_type l, N n) const {
        return view().middle_layers(l, n);
    }

    /// @}

    /// @name Iterators and buffer access
    /// @{

    /// @copydoc View::data()
    [[nodiscard]] value_type *data() { return view().data(); }
    /// @copydoc View::data()
    [[nodiscard]] const value_type *data() const { return view().data(); }
    /// @copydoc View::begin()
    [[nodiscard]] auto begin() { return view().begin(); }
    /// @copydoc View::begin()
    [[nodiscard]] auto begin() const { return view().begin(); }
    /// @copydoc View::end()
    [[nodiscard]] auto end() { return view().end(); }
    /// @copydoc View::end()
    [[nodiscard]] auto end() const { return view().end(); }

    /// @}

    /// @name Dimensions
    /// @{

    [[nodiscard]] layout_type layout() const { return view_.layout; }
    /// @copydoc View::size()
    [[nodiscard]] index_type size() const { return view().size(); }
    /// @copydoc View::padded_size()
    [[nodiscard]] index_type padded_size() const { return view().padded_size(); }
    /// @copydoc View::depth()
    [[nodiscard]] depth_type depth() const { return view().depth(); }
    /// @copydoc View::ceil_depth()
    [[nodiscard]] index_type ceil_depth() const { return view().ceil_depth(); }
    /// @copydoc View::num_batches()
    [[nodiscard]] index_type num_batches() const { return view().num_batches(); }
    /// @copydoc View::rows()
    [[nodiscard]] index_type rows() const { return view().rows(); }
    /// @copydoc View::cols()
    [[nodiscard]] index_type cols() const { return view().cols(); }
    /// @copydoc View::outer_size()
    [[nodiscard]] index_type outer_size() const { return view().outer_size(); }
    /// @copydoc View::inner_size()
    [[nodiscard]] index_type inner_size() const { return view().inner_size(); }

    /// @}

    /// @name Strides
    /// @{

    /// @copydoc View::outer_stride()
    [[nodiscard]] index_type outer_stride() const { return view().outer_stride(); }
    /// @copydoc View::inner_stride()
    [[nodiscard]] constexpr auto inner_stride() const { return view().inner_stride(); }
    /// @copydoc View::row_stride()
    [[nodiscard]] constexpr auto row_stride() const { return view().row_stride(); }
    /// @copydoc View::col_stride()
    [[nodiscard]] constexpr auto col_stride() const { return view().col_stride(); }
    /// @copydoc View::layer_stride()
    [[nodiscard]] index_type layer_stride() const { return view().layer_stride(); }
    /// @copydoc View::has_full_layer_stride()
    [[nodiscard]] bool has_full_layer_stride() const { return view().has_full_layer_stride(); }
    /// @copydoc View::has_full_outer_stride()
    [[nodiscard]] bool has_full_outer_stride() const { return view().has_full_outer_stride(); }
    /// @copydoc View::has_full_inner_stride()
    [[nodiscard]] bool has_full_inner_stride() const { return view().has_full_inner_stride(); }
    /// @copydoc View::batch_size()
    [[nodiscard]] batch_size_type batch_size() const { return view().batch_size(); }

    /// @}

    /// @name Reshaping and slicing
    /// @{

    /// @copydoc View::reshaped()
    [[nodiscard]] auto reshaped(index_type rows, index_type cols) {
        return view().reshaped(rows, cols);
    }
    /// @copydoc View::reshaped()
    [[nodiscard]] auto reshaped(index_type rows, index_type cols) const {
        return view().reshaped(rows, cols);
    }
    /// @copydoc View::top_rows()
    [[nodiscard]] auto top_rows(index_type n) { return view().top_rows(n); }
    /// @copydoc View::top_rows()
    [[nodiscard]] auto top_rows(index_type n) const { return view().top_rows(n); }
    /// @copydoc View::left_cols()
    [[nodiscard]] auto left_cols(index_type n) { return view().left_cols(n); }
    /// @copydoc View::left_cols()
    [[nodiscard]] auto left_cols(index_type n) const { return view().left_cols(n); }
    /// @copydoc View::bottom_rows()
    [[nodiscard]] auto bottom_rows(index_type n) { return view().bottom_rows(n); }
    /// @copydoc View::bottom_rows()
    [[nodiscard]] auto bottom_rows(index_type n) const { return view().bottom_rows(n); }
    /// @copydoc View::right_cols()
    [[nodiscard]] auto right_cols(index_type n) { return view().right_cols(n); }
    /// @copydoc View::right_cols()
    [[nodiscard]] auto right_cols(index_type n) const { return view().right_cols(n); }
    /// @copydoc View::middle_rows
    [[nodiscard]] auto middle_rows(index_type r, index_type n) { return view().middle_rows(r, n); }
    /// @copydoc View::middle_rows
    [[nodiscard]] auto middle_rows(index_type r, index_type n) const {
        return view().middle_rows(r, n);
    }
    /// @copydoc View::middle_rows
    [[nodiscard]] auto middle_rows(index_type r, index_type n, index_type stride)
        requires(view_type::is_row_major)
    {
        return view().middle_rows(r, n, stride);
    }
    /// @copydoc View::middle_rows
    [[nodiscard]] auto middle_rows(index_type r, index_type n, index_type stride) const
        requires(view_type::is_row_major)
    {
        return view().middle_rows(r, n, stride);
    }
    /// @copydoc View::middle_cols
    [[nodiscard]] auto middle_cols(index_type c, index_type n) { return view().middle_cols(c, n); }
    /// @copydoc View::middle_cols
    [[nodiscard]] auto middle_cols(index_type c, index_type n) const {
        return view().middle_cols(c, n);
    }
    /// @copydoc View::middle_cols
    [[nodiscard]] auto middle_cols(index_type c, index_type n, index_type stride)
        requires(view_type::is_column_major)
    {
        return view().middle_cols(c, n, stride);
    }
    /// @copydoc View::middle_cols
    [[nodiscard]] auto middle_cols(index_type c, index_type n, index_type stride) const
        requires(view_type::is_column_major)
    {
        return view().middle_cols(c, n, stride);
    }
    /// @copydoc View::top_left()
    [[nodiscard]] auto top_left(index_type nr, index_type nc) { return view().top_left(nr, nc); }
    /// @copydoc View::top_left()
    [[nodiscard]] auto top_left(index_type nr, index_type nc) const {
        return view().top_left(nr, nc);
    }
    /// @copydoc View::top_right()
    [[nodiscard]] auto top_right(index_type nr, index_type nc) { return view().top_right(nr, nc); }
    /// @copydoc View::top_right()
    [[nodiscard]] auto top_right(index_type nr, index_type nc) const {
        return view().top_right(nr, nc);
    }
    /// @copydoc View::bottom_left()
    [[nodiscard]] auto bottom_left(index_type nr, index_type nc) {
        return view().bottom_left(nr, nc);
    }
    /// @copydoc View::bottom_left()
    [[nodiscard]] auto bottom_left(index_type nr, index_type nc) const {
        return view().bottom_left(nr, nc);
    }
    /// @copydoc View::bottom_right()
    [[nodiscard]] auto bottom_right(index_type nr, index_type nc) {
        return view().bottom_right(nr, nc);
    }
    /// @copydoc View::bottom_right()
    [[nodiscard]] auto bottom_right(index_type nr, index_type nc) const {
        return view().bottom_right(nr, nc);
    }
    /// @copydoc View::block()
    [[nodiscard]] auto block(index_type r, index_type c, index_type nr, index_type nc) {
        return view().block(r, c, nr, nc);
    }
    /// @copydoc View::block()
    [[nodiscard]] auto block(index_type r, index_type c, index_type nr, index_type nc) const {
        return view().block(r, c, nr, nc);
    }
    /// @copydoc View::transposed()
    [[nodiscard]] auto transposed() { return view().transposed(); }
    /// @copydoc View::transposed()
    [[nodiscard]] auto transposed() const { return view().transposed(); }

    /// @}

    /// @name Value manipulation
    /// @{

    /// Copy the values of another matrix, resizing if necessary.
    Matrix &operator=(const Matrix &o) {
        if (&o != this) {
            clear();
            view_.reassign({allocate(o.layout()).release(), o.layout()});
            // TODO: use allocate_for_overwrite or similar to avoid copy
            //       assignment
            this->view_.copy_values(o.view_); // TODO: exception safety
        }
        return *this;
    }

    /// Copy the values from a compatible view, resizing if necessary.
    template <class U, class J, class R, class E, class M>
        requires(std::convertible_to<U, T> && std::equality_comparable_with<index_type, J>)
    Matrix &operator=(View<U, J, R, E, M, O> other) {
        assign_from_view(other);
        return *this;
    }

    /// @copydoc View::set_constant()
    void set_constant(value_type t) { view().set_constant(t); }
    /// @copydoc View::add_to_diagonal()
    void add_to_diagonal(const value_type &t) { view().add_to_diagonal(t); }
    /// @copydoc View::negate()
    void negate() { view().negate(); }
    /// @copydoc View::copy_values()
    template <class Other>
    void copy_values(const Other &other) {
        view().copy_values(other);
    }
    /// @copydoc View::operator+=()
    template <class U, class J, class R, class E, class M>
        requires(std::convertible_to<U, T> && std::equality_comparable_with<index_type, J>)
    Matrix &operator+=(View<U, J, R, E, M, O> other) {
        view() += other;
        return *this;
    }

    /// @}

    /// @name View conversions
    /// @{

    /// @copydoc View::view()
    [[nodiscard, gnu::always_inline]] view_type view() { return view_; }
    /// @copydoc View::view()
    [[nodiscard, gnu::always_inline]] const_view_type view() const { return view_.as_const(); }
    /// @copydoc View::as_const()
    [[nodiscard]] auto as_const() const { return view(); }

    operator view_type() { return view(); }
    operator const_view_type() const { return view(); }
    operator View<T, I, S, D, I, O>() { return view(); }
    operator View<const T, I, S, D, I, O>() const { return view(); }
    operator View<T, I, S, integral_value_type_t<D>, DefaultStride, O>()
        requires(!std::same_as<integral_value_type_t<D>, D>)
    {
        return view();
    }
    operator View<const T, I, S, integral_value_type_t<D>, DefaultStride, O>() const
        requires(!std::same_as<integral_value_type_t<D>, D>)
    {
        return view();
    }
    operator guanaqo::MatrixView<T, I, standard_stride_type, O>()
        requires has_single_layer_at_compile_time
    {
        return view();
    }
    operator guanaqo::MatrixView<const T, I, standard_stride_type, O>() const
        requires(has_single_layer_at_compile_time)
    {
        return view();
    }

    /// @}
};

template <class T, class I, class S, class D, class A, StorageOrder O>
constexpr auto data(Matrix<T, I, S, D, O, A> &v) {
    return v.data();
}
template <class T, class I, class S, class D, class A, StorageOrder O>
constexpr auto data(Matrix<T, I, S, D, O, A> &&v) = delete;
template <class T, class I, class S, class D, class A, StorageOrder O>
constexpr auto data(const Matrix<T, I, S, D, O, A> &v) {
    return v.data();
}
template <class T, class I, class S, class D, class A, StorageOrder O>
constexpr auto rows(const Matrix<T, I, S, D, O, A> &v) {
    return v.rows();
}
template <class T, class I, class S, class D, class A, StorageOrder O>
constexpr auto cols(const Matrix<T, I, S, D, O, A> &v) {
    return v.cols();
}
template <class T, class I, class S, class D, class A, StorageOrder O>
constexpr auto outer_stride(const Matrix<T, I, S, D, O, A> &v) {
    return v.outer_stride();
}
template <class T, class I, class S, class D, class A, StorageOrder O>
constexpr auto depth(const Matrix<T, I, S, D, O, A> &v) {
    return v.depth();
}

} // namespace batmat::matrix
