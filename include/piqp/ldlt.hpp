// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2005-2022 by Timothy A. Davis.
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_LDLT_HPP
#define PIQP_LDLT_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace piqp
{

template<typename T, typename I>
struct LDLt
{
    Vec<I> etree;    // elimination tree
    // L in CSC
    Vec<I> L_cols;   // column starts [n+1]
    Vec<I> L_nnz;    // number of non-zeros per column [n]
    Vec<I> L_ind;    // row indices
    Vec<T> L_vals;   // values

    Vec<T> D;        // diagonal matrix D
    Vec<T> D_inv;    // inverse of D

    Vec<I> perm;     // permutation matrix
    Vec<I> perm_inv; // inverse permutation matrix

    // working variables used in numerical factorization
    struct {
        Vec<I> flag;
        Vec<I> pattern;
        Vec<T> y;
    } work;
};

} // namespace piqp

#endif //PIQP_LDLT_HPP
