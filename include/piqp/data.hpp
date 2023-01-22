// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DATA_HPP
#define PIQP_DATA_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace piqp
{

template<typename T, typename I>
struct Data
{
    isize n; // number of variables
    isize p; // number of equality constraints
    isize m; // number of inequality constraints

    SparseMat<T, I> P_utri; // upper triangular part of H
    SparseMat<T, I> AT;     // A transpose
    SparseMat<T, I> GT;     // G transpose

    Vec<T> c;
    Vec<T> b;
    Vec<T> h;
};

} // namespace piqp

#endif //PIQP_DATA_HPP
