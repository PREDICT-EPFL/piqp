// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_MODEL_HPP
#define PIQP_DENSE_MODEL_HPP

#include <limits>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/typedefs.hpp"
#include "piqp/utils/optional.hpp"

namespace piqp
{

namespace dense
{

template<typename T_>
struct Model
{
    using T = T_;

    Mat<T> P;
    Mat<T> A;
    Mat<T> G;

    Vec<T> c;
    Vec<T> b;
    Vec<T> h;
    Vec<T> x_lb;
    Vec<T> x_ub;

    Model(const CMatRef<T>& P,
          const CVecRef<T>& c,
          const optional<CMatRef<T>>& A = nullopt,
          const optional<CVecRef<T>>& b = nullopt,
          const optional<CMatRef<T>>& G = nullopt,
          const optional<CVecRef<T>>& h = nullopt,
          const optional<CVecRef<T>>& x_lb = nullopt,
          const optional<CVecRef<T>>& x_ub = nullopt) noexcept
      : P(P), c(c)
    {
        isize n = P.rows();
        isize p = A.has_value() ? A->rows() : 0;
        isize m = G.has_value() ? G->rows() : 0;

        if (P.rows() != n || P.cols() != n) { piqp_eprint("P must be square"); }
        if (A.has_value() && (A->rows() != p || A->cols() != n)) { piqp_eprint("A must have correct dimensions"); }
        if (G.has_value() && (G->rows() != m || G->cols() != n)) { piqp_eprint("G must have correct dimensions"); }
        if (c.size() != n) { piqp_eprint("c must have correct dimensions"); }
        if ((b.has_value() && b->size() != p) || (!b.has_value() && p > 0)) { piqp_eprint("b must have correct dimensions"); }
        if ((h.has_value() && h->size() != m) || (!h.has_value() && m > 0)) { piqp_eprint("h must have correct dimensions"); }
        if (x_lb.has_value() && x_lb->size() != n) { piqp_eprint("x_lb must have correct dimensions"); }
        if (x_ub.has_value() && x_ub->size() != n) { piqp_eprint("x_ub must have correct dimensions"); }

        this->A = A.value_or(Mat<T>(p, n));
        this->G = G.value_or(Mat<T>(m, n));
        this->b = b.value_or(Vec<T>(p));
        this->h = h.value_or(Vec<T>(m));
        this->x_lb = x_lb.value_or(Vec<T>::Constant(n, -std::numeric_limits<T>::infinity()));
        this->x_ub = x_ub.value_or(Vec<T>::Constant(n, std::numeric_limits<T>::infinity()));
    }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_DENSE_MODEL_HPP
