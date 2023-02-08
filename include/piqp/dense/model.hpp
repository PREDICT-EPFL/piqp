// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_MODEL_HPP
#define PIQP_DENSE_MODEL_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

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

    Model(const CMatRef<T>& P,
          const CMatRef<T>& A,
          const CMatRef<T>& G,
          const CVecRef<T>& c,
          const CVecRef<T>& b,
          const CVecRef<T>& h) noexcept
      : P(P), A(A), G(G), c(c), b(b), h(h) {}
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_DENSE_MODEL_HPP
