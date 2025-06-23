#pragma once

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

/// @tparam T
///         Element value type.
/// @tparam I
///         Index type.
/// @tparam S
///         Inner stride (batch size).
/// @tparam D
///         Depth type.
/// @tparam A
///         Batch alignment type.
/// @tparam O
///         Storage order (column or row major).
template <class T, class I = ptrdiff_t, class S = std::integral_constant<I, 1>, class D = I,
          StorageOrder O = StorageOrder::ColMajor, class A = detail::default_alignment_t<T, I, S>>
struct Matrix {
    static_assert(!std::is_const_v<T>);
    using view_type            = View<T, I, S, D, DefaultStride, O>;
    using const_view_type      = typename view_type::const_view_type;
    using layout_type          = typename view_type::layout_type;
    using plain_layout_type    = typename layout_type::PlainLayout;
    using value_type           = T;
    using index_type           = typename layout_type::index_type;
    using batch_size_type      = typename layout_type::batch_size_type;
    using depth_type           = typename layout_type::depth_type;
    using standard_stride_type = typename layout_type::standard_stride_type;
    using alignment_type       = A;

  private:
    view_type view_;

    static constexpr auto default_alignment(layout_type layout) {
        if constexpr (alignment_type{} == 0)
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
        if (auto d = std::exchange(view_.data, nullptr))
            aligned_deleter<T, decltype(alignment)>(view_.layout.padded_size(), alignment)(d);
        view_.layout.rows = 0;
    }

  public:
    [[nodiscard, gnu::always_inline]] view_type view() { return view_; }
    [[nodiscard, gnu::always_inline]] const_view_type view() const { return view_.as_const(); }

    [[nodiscard]] layout_type layout() const { return view_.layout; }

    void resize(layout_type new_layout) {
        if (new_layout.padded_size() != layout().padded_size()) {
            clear();
            view_.data = allocate(new_layout).release();
        }
        view_.layout = new_layout;
    }

    void set_constant(value_type t) { view().set_constant(t); }

    Matrix() = default;
    Matrix(layout_type layout) : view_{allocate(layout).release(), layout} {}
    Matrix(layout_type layout, uninitialized_t init)
        : view_{allocate(layout, init).release(), layout} {}
    Matrix(plain_layout_type p) : Matrix{layout_type{p}} {}
    Matrix(plain_layout_type p, uninitialized_t init) : Matrix{layout_type{p}, init} {}

    Matrix(const Matrix &o) : Matrix{o.layout()} {
        this->view().copy_values(o.view()); // TODO: exception safety
    }
    Matrix(Matrix &&o) noexcept : view_{o.view()} { o.view_.reassign({}); }
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
    Matrix &operator=(Matrix &&o) noexcept {
        using std::swap;
        if (&o != this) {
            swap(o.view_.data, this->view_.data);
            swap(o.view_.layout, this->view_.layout);
        }
        return *this;
    }
    ~Matrix() { clear(); }

    operator view_type() { return view(); }
    operator const_view_type() const { return view(); }
    operator View<T, I, S, D, I, O>() { return view(); }
    operator View<const T, I, S, D, I, O>() const { return view(); }

    [[nodiscard]] guanaqo::MatrixView<T, I, standard_stride_type, O> operator()(index_type l) {
        return view()(l);
    }
    [[nodiscard]] value_type &operator()(index_type l, index_type r, index_type c) {
        return view()(l, r, c);
    }
    [[nodiscard]] guanaqo::MatrixView<const T, I, standard_stride_type, O>
    operator()(index_type l) const {
        return view()(l);
    }
    [[nodiscard]] const value_type &operator()(index_type l, index_type r, index_type c) const {
        return view()(l, r, c);
    }

    [[nodiscard]] auto batch(index_type b) { return view().batch(b); }
    [[nodiscard]] auto batch(index_type b) const { return view().batch(b); }
    [[nodiscard]] auto batch_dyn(index_type b) { return view().batch_dyn(b); }
    [[nodiscard]] auto batch_dyn(index_type b) const { return view().batch_dyn(b); }

    [[nodiscard]] auto reshaped(index_type rows, index_type cols) {
        return view().reshaped(rows, cols);
    }
    [[nodiscard]] auto reshaped(index_type rows, index_type cols) const {
        return view().reshaped(rows, cols);
    }
    [[nodiscard]] auto top_rows(index_type n) { return view().top_rows(n); }
    [[nodiscard]] auto top_rows(index_type n) const { return view().top_rows(n); }
    [[nodiscard]] auto left_cols(index_type n) { return view().left_cols(n); }
    [[nodiscard]] auto left_cols(index_type n) const { return view().left_cols(n); }
    [[nodiscard]] auto bottom_rows(index_type n) { return view().bottom_rows(n); }
    [[nodiscard]] auto bottom_rows(index_type n) const { return view().bottom_rows(n); }
    [[nodiscard]] auto right_cols(index_type n) { return view().right_cols(n); }
    [[nodiscard]] auto right_cols(index_type n) const { return view().right_cols(n); }
    [[nodiscard]] auto middle_rows(index_type r, index_type n) { return view().middle_rows(r, n); }
    [[nodiscard]] auto middle_rows(index_type r, index_type n) const {
        return view().middle_rows(r, n);
    }
    [[nodiscard]] auto middle_cols(index_type c, index_type n) { return view().middle_cols(c, n); }
    [[nodiscard]] auto middle_cols(index_type c, index_type n) const {
        return view().middle_cols(c, n);
    }
    [[nodiscard]] auto top_left(index_type nr, index_type nc) { return view().top_left(nr, nc); }
    [[nodiscard]] auto top_left(index_type nr, index_type nc) const {
        return view().top_left(nr, nc);
    }
    [[nodiscard]] auto top_right(index_type nr, index_type nc) { return view().top_right(nr, nc); }
    [[nodiscard]] auto top_right(index_type nr, index_type nc) const {
        return view().top_right(nr, nc);
    }
    [[nodiscard]] auto bottom_left(index_type nr, index_type nc) {
        return view().bottom_left(nr, nc);
    }
    [[nodiscard]] auto bottom_left(index_type nr, index_type nc) const {
        return view().bottom_left(nr, nc);
    }
    [[nodiscard]] auto bottom_right(index_type nr, index_type nc) {
        return view().bottom_right(nr, nc);
    }
    [[nodiscard]] auto bottom_right(index_type nr, index_type nc) const {
        return view().bottom_right(nr, nc);
    }
    [[nodiscard]] auto block(index_type r, index_type c, index_type nr, index_type nc) {
        return view().block(r, c, nr, nc);
    }
    [[nodiscard]] auto block(index_type r, index_type c, index_type nr, index_type nc) const {
        return view().block(r, c, nr, nc);
    }
    [[nodiscard]] auto transposed() { return view().transposed(); }
    [[nodiscard]] auto transposed() const { return view().transposed(); }

    [[nodiscard]] auto as_const() const { return view(); }
    [[nodiscard]] auto begin() { return view().begin(); }
    [[nodiscard]] auto begin() const { return view().begin(); }
    [[nodiscard]] auto end() { return view().end(); }
    [[nodiscard]] auto end() const { return view().end(); }
    [[nodiscard]] index_type size() const { return view().size(); }
    [[nodiscard]] index_type padded_size() const { return view().padded_size(); }

    [[gnu::always_inline]] value_type *data() { return view().data; }
    [[gnu::always_inline]] const value_type *data() const { return view().data; }
    [[gnu::always_inline]] depth_type depth() const { return view().depth(); }
    [[gnu::always_inline]] index_type ceil_depth() const { return view().ceil_depth(); }
    [[gnu::always_inline]] index_type num_batches() const { return view().num_batches(); }
    [[gnu::always_inline]] index_type rows() const { return view().rows(); }
    [[gnu::always_inline]] index_type cols() const { return view().cols(); }
    [[gnu::always_inline]] index_type outer_stride() const { return view().outer_stride(); }
    [[gnu::always_inline]] batch_size_type batch_size() const { return view().batch_size(); }
};

} // namespace batmat::matrix
