// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_MODEL_HPP
#define PIQP_MODEL_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace piqp
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

    Model(const SparseMat<T, I>& P,
          const CVecRef<T>& c,
          const SparseMat<T, I>& A,
          const CVecRef<T>& b,
          const SparseMat<T, I>& G,
          const CVecRef<T>& h) noexcept
      : P(P), A(A), G(G), c(c), b(b), h(h) {}
};

} // namespace piqp

#endif //PIQP_MODEL_HPP
