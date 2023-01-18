// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/utils.hpp"

#include "gtest/gtest.h"

using namespace piqp;

using T = double;
using I = int;

TEST(LDLT, Symbolic)
{
    isize dim = 10;
    T sparsity_factor = 0.5;

    SparseMat<T, I> P = rand::sparse_positive_definite_upper_triangular_rand<T, I>(dim, sparsity_factor);

    LDLt<T, I> ldlt;
    ldlt.factorize_symbolic_upper_triangular(P);
}

TEST(LDLT, Numeric)
{
    isize dim = 10;
    T sparsity_factor = 0.5;

    SparseMat<T, I> P = rand::sparse_positive_definite_upper_triangular_rand<T, I>(dim, sparsity_factor);

    LDLt<T, I> ldlt;
    ldlt.factorize_symbolic_upper_triangular(P);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    isize n = ldlt.factorize_numeric_upper_triangular(P);
    PIQP_EIGEN_MALLOC_ALLOWED();

    EXPECT_EQ(dim, n);
}

TEST(LDLT, Solve)
{
    isize dim = 10;
    T sparsity_factor = 0.5;

    SparseMat<T, I> P = rand::sparse_positive_definite_upper_triangular_rand<T, I>(dim, sparsity_factor);

    LDLt<T, I> ldlt;
    ldlt.factorize_symbolic_upper_triangular(P);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    isize n = ldlt.factorize_numeric_upper_triangular(P);
    PIQP_EIGEN_MALLOC_ALLOWED();

    EXPECT_EQ(dim, n);

    Vec<T> b = rand::vector_rand<T>(dim);
    Vec<T> x = b;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    ldlt.lsolve(x);
    ldlt.dsolve(x);
    ldlt.ltsolve(x);
    PIQP_EIGEN_MALLOC_ALLOWED();

    SparseMat<T, I> PT = P.transpose();
    SparseMat<T, I> P_full = P + PT;
    P_full.diagonal().array() -= P.diagonal().array();

    EXPECT_TRUE(b.isApprox(P_full * x, 1e-8));
}
