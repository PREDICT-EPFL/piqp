// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_MODEL_HPP
#define PIQP_SPARSE_MODEL_HPP

#include <limits>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/typedefs.hpp"
#include "piqp/dense/model.hpp"
#include "piqp/utils/optional.hpp"

namespace piqp
{

namespace sparse
{

template<typename T_, typename I_>
struct Model
{
    using T = T_;
    using I = I_;

    SparseMat<T, I> P;
    SparseMat<T, I> A;
    SparseMat<T, I> G;

    Vec<T> c;
    Vec<T> b;
    Vec<T> h;
    Vec<T> x_lb;
    Vec<T> x_ub;

    Model(const SparseMat<T, I>& P,
          const CVecRef<T>& c,
          const optional<SparseMat<T, I>>& A,
          const optional<CVecRef<T>>& b,
          const optional<SparseMat<T, I>>& G,
          const optional<CVecRef<T>>& h,
          const optional<CVecRef<T>>& x_lb,
          const optional<CVecRef<T>>& x_ub) noexcept
      : P(P), c(c)
    {
        isize n = P.rows();
        isize p = A.has_value() ? A->rows() : 0;
        isize m = G.has_value() ? G->rows() : 0;

        if (P.rows() != n || P.cols() != n) { piqp_eprint("P must be square\n"); }
        if (A.has_value() && (A->rows() != p || A->cols() != n)) { piqp_eprint("A must have correct dimensions\n"); }
        if (G.has_value() && (G->rows() != m || G->cols() != n)) { piqp_eprint("G must have correct dimensions\n"); }
        if (c.size() != n) { piqp_eprint("c must have correct dimensions\n"); }
        if ((b.has_value() && b->size() != p) || (!b.has_value() && p > 0)) { piqp_eprint("b must have correct dimensions\n"); }
        if ((h.has_value() && h->size() != m) || (!h.has_value() && m > 0)) { piqp_eprint("h must have correct dimensions\n"); }
        if (x_lb.has_value() && x_lb->size() != n) { piqp_eprint("x_lb must have correct dimensions\n"); }
        if (x_ub.has_value() && x_ub->size() != n) { piqp_eprint("x_ub must have correct dimensions\n"); }

        this->A = A.value_or(SparseMat<T, I>(p, n));
        this->G = G.value_or(SparseMat<T, I>(m, n));
        this->b = b.value_or(Vec<T>(p));
        this->h = h.value_or(Vec<T>(m));
        this->x_lb = x_lb.value_or(Vec<T>::Constant(n, -std::numeric_limits<T>::infinity()));
        this->x_ub = x_ub.value_or(Vec<T>::Constant(n, std::numeric_limits<T>::infinity()));
    }

    dense::Model<T> dense_model()
    {
        return dense::Model<T>(Mat<T>(P), c, Mat<T>(A), b, Mat<T>(G), h, x_lb, x_ub);
    }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_MODEL_HPP
