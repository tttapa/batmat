#pragma once

#include <batmat/assume.hpp>
#include <batmat/config.hpp>
#include <guanaqo/mat-view.hpp>
#include <blasfeo.h>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

namespace batmat::blasfeo {

struct dmat {
    blasfeo_dmat mat{};
    dmat()                        = default;
    dmat(const dmat &)            = delete;
    dmat &operator=(const dmat &) = delete;
    dmat(dmat &&o) noexcept : mat{std::exchange(o.mat, {})} {}
    dmat &operator=(dmat &&o) noexcept {
        using std::swap;
        swap(o.mat, mat);
        return *this;
    }
    ~dmat() { reset(); }
    void reset() {
        if (mat.mem)
            blasfeo_free_dmat(&mat);
    }
    [[nodiscard]] blasfeo_dmat *get() { return &mat; }
    [[nodiscard]] const blasfeo_dmat *get() const { return &mat; }

    void allocate(index_t rows, index_t cols) {
        reset();
        blasfeo_allocate_dmat(static_cast<int>(rows), static_cast<int>(cols), &mat);
        if (!mat.mem)
            throw std::bad_alloc{};
    }

    dmat(guanaqo::MatrixView<double, index_t> view) {
        allocate(view.rows, view.cols);
        blasfeo_pack_dmat(static_cast<int>(view.rows), static_cast<int>(view.cols), view.data,
                          static_cast<int>(view.outer_stride), get(), 0, 0);
    }
    dmat &operator=(guanaqo::MatrixView<double, index_t> view) {
        BATMAT_ASSERT(static_cast<index_t>(mat.m) == view.rows);
        BATMAT_ASSERT(static_cast<index_t>(mat.n) == view.cols);
        blasfeo_pack_dmat(static_cast<int>(view.rows), static_cast<int>(view.cols), view.data,
                          static_cast<int>(view.outer_stride), get(), 0, 0);
        return *this;
    }

    void copy_to(guanaqo::MatrixView<double, index_t> view) const {
        BATMAT_ASSERT(static_cast<index_t>(mat.m) == view.rows);
        BATMAT_ASSERT(static_cast<index_t>(mat.n) == view.cols);
        blasfeo_unpack_dmat(static_cast<int>(view.rows), static_cast<int>(view.cols),
                            const_cast<blasfeo_dmat *>(get()), 0, 0, view.data,
                            static_cast<int>(view.outer_stride));
    }

    void copy_to(dmat &other) const {
        BATMAT_ASSERT(mat.m == other.mat.m);
        BATMAT_ASSERT(mat.n == other.mat.n);
        blasfeo_dgecp(mat.m, mat.n, const_cast<blasfeo_dmat *>(get()), 0, 0, other.get(), 0, 0);
    }

    template <class Rng>
    void init_random(index_t rows, index_t cols, Rng &rng) {
        allocate(rows, cols);
        std::vector<double> gen_buf(static_cast<size_t>(rows * cols));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::ranges::generate(gen_buf, [&] { return dist(rng); });
        blasfeo_pack_dmat(static_cast<int>(rows), static_cast<int>(cols), gen_buf.data(),
                          static_cast<int>(rows), get(), 0, 0);
    }

    template <class Rng>
    void init_random_pos_def(index_t rows, index_t cols, Rng &&rng) {
        using std::min;
        allocate(rows, cols);
        std::vector<double> gen_buf(static_cast<size_t>(rows * cols));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::ranges::generate(gen_buf, [&] { return dist(rng); });
        index_t m = min(rows, cols);
        for (index_t i = 0; i < m; ++i)
            gen_buf[static_cast<size_t>(i * rows + i)] += 10 * static_cast<double>(m);
        blasfeo_pack_dmat(static_cast<int>(rows), static_cast<int>(cols), gen_buf.data(),
                          static_cast<int>(rows), get(), 0, 0);
    }

    [[nodiscard]] index_t rows() const { return static_cast<index_t>(mat.m); }
    [[nodiscard]] index_t cols() const { return static_cast<index_t>(mat.n); }

    template <class Rng>
    [[nodiscard]] static std::vector<dmat> random_batch(index_t depth, index_t rows, index_t cols,
                                                        Rng &&rng) {
        std::vector<dmat> mats(static_cast<size_t>(depth));
        for (auto &m : mats)
            m.init_random(rows, cols, rng);
        return mats;
    }

    template <class Rng>
    [[nodiscard]] static std::vector<dmat> random_batch_pos_def(index_t depth, index_t rows,
                                                                index_t cols, Rng &&rng) {
        std::vector<dmat> mats(static_cast<size_t>(depth));
        for (auto &m : mats)
            m.init_random_pos_def(rows, cols, rng);
        return mats;
    }
};

} // namespace batmat::blasfeo
