// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_DATA_HPP
#define PIQP_DENSE_DATA_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/typedefs.hpp"

namespace piqp
{

namespace dense
{

template<typename T>
struct Data
{
    isize n; // number of variables
    isize p; // number of equality constraints
    isize m; // number of inequality constraints

    Mat<T> P_utri; // upper triangular part of P
    Mat<T> AT;     // A transpose
    Mat<T> GT;     // G transpose

    Vec<T> c;
    Vec<T> b;
    Vec<T> h;

    Eigen::Index non_zeros_P_utri() { return P_utri.rows() * (P_utri.rows() - 1) / 2; }
    Eigen::Index non_zeros_A() { return AT.rows() * AT.cols(); }
    Eigen::Index non_zeros_G() { return GT.rows() * GT.cols(); }
};

} // namespace dense

} // namespace piqp

#endif //PIQP_DENSE_DATA_HPP
