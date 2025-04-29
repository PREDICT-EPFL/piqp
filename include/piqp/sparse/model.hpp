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
    Vec<T> h_l;
    Vec<T> h_u;
    Vec<T> x_l;
    Vec<T> x_u;

    Model(const SparseMat<T, I>& P,
          const CVecRef<T>& c,
          const optional<SparseMat<T, I>>& A = nullopt,
          const optional<CVecRef<T>>& b = nullopt,
          const optional<SparseMat<T, I>>& G = nullopt,
          const optional<CVecRef<T>>& h_l = nullopt,
          const optional<CVecRef<T>>& h_u = nullopt,
          const optional<CVecRef<T>>& x_l = nullopt,
          const optional<CVecRef<T>>& x_u = nullopt) noexcept
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
        if (h_l.has_value() && h_l->size() != m) { piqp_eprint("h_l must have correct dimensions\n"); }
        if (h_u.has_value() && h_u->size() != m) { piqp_eprint("h_u must have correct dimensions\n"); }
        if (!h_l.has_value() && !h_u.has_value() && m > 0) { piqp_eprint("h_l or h_u should be provided\n"); }
        if (x_l.has_value() && x_l->size() != n) { piqp_eprint("x_l must have correct dimensions\n"); }
        if (x_u.has_value() && x_u->size() != n) { piqp_eprint("x_u must have correct dimensions\n"); }

        this->A = A.value_or(SparseMat<T, I>(p, n));
        this->G = G.value_or(SparseMat<T, I>(m, n));
        this->b = b.value_or(Vec<T>(p));
        this->h_l = h_l.value_or(Vec<T>::Constant(m, -std::numeric_limits<T>::infinity()));
        this->h_u = h_u.value_or(Vec<T>::Constant(m, std::numeric_limits<T>::infinity()));
        this->x_l = x_l.value_or(Vec<T>::Constant(n, -std::numeric_limits<T>::infinity()));
        this->x_u = x_u.value_or(Vec<T>::Constant(n, std::numeric_limits<T>::infinity()));
    }

    dense::Model<T> dense_model()
    {
        return dense::Model<T>(Mat<T>(P), c, Mat<T>(A), b, Mat<T>(G), h_l, h_u, x_l, x_u);
    }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_MODEL_HPP
