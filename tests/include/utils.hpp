// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_TESTS_UTILS_HPP
#define PIQP_TESTS_UTILS_HPP

#include "piqp/piqp.hpp"
#include "gtest/gtest.h"

template<typename T, unsigned int Mode>
void assert_dense_triangular_equal(piqp::Mat<T>& A, piqp::Mat<T>& B)
{
    ASSERT_EQ(A.rows(), B.rows());
    ASSERT_EQ(A.cols(), B.cols());

    piqp::Mat<T> diff(A.rows(), A.cols());
    diff.setZero();

    diff = A.template triangularView<Mode>();
    diff -= B.template triangularView<Mode>();
    ASSERT_TRUE(diff.cwiseAbs().maxCoeff() == 0);
}

template<typename T, typename I>
void assert_sparse_matrices_equal(piqp::SparseMat<T, I>& A, piqp::SparseMat<T, I>& B)
{
    ASSERT_EQ(A.rows(), B.rows());
    ASSERT_EQ(A.cols(), B.cols());
    ASSERT_EQ(A.nonZeros(), B.nonZeros());
    ASSERT_EQ(Eigen::Map<piqp::Vec<I>>(A.outerIndexPtr(), A.outerSize() + 1),
              Eigen::Map<piqp::Vec<I>>(B.outerIndexPtr(), B.outerSize() + 1));
    ASSERT_EQ(Eigen::Map<piqp::Vec<I>>(A.innerIndexPtr(), A.nonZeros()),
              Eigen::Map<piqp::Vec<I>>(B.innerIndexPtr(), B.nonZeros()));
    ASSERT_EQ(Eigen::Map<piqp::Vec<T>>(A.valuePtr(), A.nonZeros()),
              Eigen::Map<piqp::Vec<T>>(B.valuePtr(), B.nonZeros()));
}

#endif //PIQP_TESTS_UTILS_HPP
