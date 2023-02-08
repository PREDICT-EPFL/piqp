// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/dense/ldlt_no_pivot.hpp"
#include "piqp/utils/random_utils.hpp"

#include "gtest/gtest.h"

using namespace piqp;
using namespace piqp::dense;

using T = double;

TEST(DenseLDLT, SolveLower)
{
    isize dim = 50;

    Mat<T> P = rand::dense_positive_definite_upper_triangular_rand<T>(dim);
    P.transposeInPlace();

    LDLTNoPivot<Mat<T>, Eigen::Lower> ldlt;
    ldlt.compute(P);
    EXPECT_EQ(ldlt.info(), Eigen::Success);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    ldlt.compute(P);
    EXPECT_EQ(ldlt.info(), Eigen::Success);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> b = rand::vector_rand<T>(dim);
    Vec<T> x = b;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    ldlt.solveInPlace(x);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Mat<T> P_full = P + P.transpose();
    P_full.diagonal().array() -= P.diagonal().array();

    EXPECT_TRUE(b.isApprox(P_full * x, 1e-8));
}

TEST(DenseLDLT, SolveUpper)
{
    isize dim = 50;

    Mat<T> P = rand::dense_positive_definite_upper_triangular_rand<T>(dim);

    LDLTNoPivot<Mat<T>, Eigen::Upper> ldlt;
    ldlt.compute(P);
    EXPECT_EQ(ldlt.info(), Eigen::Success);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    ldlt.compute(P);
    EXPECT_EQ(ldlt.info(), Eigen::Success);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> b = rand::vector_rand<T>(dim);
    Vec<T> x = b;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    ldlt.solveInPlace(x);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Mat<T> P_full = P + P.transpose();
    P_full.diagonal().array() -= P.diagonal().array();

    EXPECT_TRUE(b.isApprox(P_full * x, 1e-8));
}
