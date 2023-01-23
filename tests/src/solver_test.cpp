// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/utils/random_utils.hpp"

#include "gtest/gtest.h"

using namespace piqp;

using T = double;
using I = int;

/*
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 1, x1 = 2 x2
*/
TEST(Solver, SimpleQP)
{
    SparseMat<T, I> P(2, 2);
    P.insert(0, 0) = 6;
    P.insert(1, 1) = 4;
    P.makeCompressed();
    Vec<T> c(2); c << -1, -4;

    SparseMat<T, I> A(1, 2);
    A.insert(0, 0) = 1;
    A.insert(0, 1) = -2;
    A.makeCompressed();
    Vec<T> b(1); b << 0;

    SparseMat<T, I> G(4, 2);
    G.insert(0, 0) = 1;
    G.insert(1, 1) = 1;
    G.insert(2, 0) = -1;
    G.insert(3, 1) = -1;
    G.makeCompressed();
    Vec<T> h(4); h << 1, 1, 1, 1;

    Solver<T, I> solver;
    solver.settings().verbose = true;
    solver.setup(P, A, G, c, b, h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);

    ASSERT_NEAR(solver.result().x(0), 0.4285714, 1e-6);
    ASSERT_NEAR(solver.result().x(1), 0.2142857, 1e-6);
}

TEST(Solver, StronglyConvexWithEqualityAndInequality)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.2;

    Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    Solver<T, I> solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.A, qp_model.G, qp_model.c, qp_model.b, qp_model.h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

