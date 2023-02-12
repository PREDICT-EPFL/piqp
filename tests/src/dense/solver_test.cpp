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
 * first QP:
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 1, x1 = 2 x2
 *
 * second QP:
 * min 4 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 2, x1 = 3 x2
*/
TEST(DenseSolverTest, SimpleQPWithUpdate)
{
    Mat<T> P(2, 2); P << 6, 0, 0, 4;
    Vec<T> c(2); c << -1, -4;

    Mat<T> A(1, 2); A << 1, -2;
    Vec<T> b(1); b << 0;

    Mat<T> G(2, 2); G << 1, 0, -1, 0;
    Vec<T> h(2); h << 1, 1;

    Vec<T> x_lb(2); x_lb << -std::numeric_limits<T>::infinity(), -1;
    Vec<T> x_ub(2); x_ub << std::numeric_limits<T>::infinity(), 1;

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(P, c, A, b, G, h, x_lb, x_ub);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
    ASSERT_NEAR(solver.result().x(0), 0.4285714, 1e-6);
    ASSERT_NEAR(solver.result().x(1), 0.2142857, 1e-6);
    ASSERT_NEAR(solver.result().y(0), -1.5714286, 1e-6);
    ASSERT_NEAR(solver.result().z(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z(1), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_lb(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_lb(1), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_ub(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_ub(1), 0, 1e-6);

    P(0, 0) = 8;
    A(0, 1) = -3;
    h(0) = 2;
    x_ub(1) = 2;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    solver.update(P, c, A, b, nullopt, h, nullopt, x_ub);
    PIQP_EIGEN_MALLOC_ALLOWED();

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
    ASSERT_NEAR(solver.result().x(0), 0.2763157, 1e-6);
    ASSERT_NEAR(solver.result().x(1), 0.0921056, 1e-6);
    ASSERT_NEAR(solver.result().y(0), -1.2105263, 1e-6);
    ASSERT_NEAR(solver.result().z(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z(1), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_lb(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_lb(1), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_ub(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_ub(1), 0, 1e-6);
}

/*
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x1 <= 0, 1 <= x2 <= 2,  x1 = 2 x2
*/
TEST(DenseSolverTest, PrimalInfeasibleQP)
{
    Mat<T> P(2, 2); P << 6, 0, 0, 4;
    Vec<T> c(2); c << -1, -4;

    Mat<T> A(1, 2); A << 1, -2;
    Vec<T> b(1); b << 0;

    Mat<T> G(4, 2); G << 1, 0, 0, 1, -1, 0, 0, -1;
    Vec<T> h(4); h << 0, 2, 1, -1;

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(P, c, A, b, G, h, nullopt, nullopt);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_PRIMAL_INFEASIBLE);
}

/*
 * min -x1 - x2
 * s.t. 0 <= x
*/
TEST(DenseSolverTest, DualInfeasibleQP)
{
    Mat<T> P(2, 2);
    P.setZero();
    Vec<T> c(2); c << -1, -1;

    Mat<T> A(0, 2);
    Vec<T> b(0);

    Mat<T> G(2, 2); G << -1, 0, 0, -1;
    Vec<T> h(2); h << 0, 0;

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(P, c, A, b, G, h, nullopt, nullopt);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_DUAL_INFEASIBLE);
}

//TEST(DenseSolverTest, NonConvexQP)
//{
//    Mat<T> P(2, 2); P << 2, 5, 5, 1;
//    Vec<T> c(2); c << 3, 4;
//
//    Mat<T> A(0, 2);
//    Vec<T> b(0);
//
//    Mat<T> G(5, 2); G << -1, 0, 0, -1, -1, 3, 2, 5, 3, 4;
//    Vec<T> h(5); h << 0, 0, -15, 100, 80;
//
//    DenseSolver<T> solver;
//    solver.settings().verbose = true;
//    solver.setup(P, c, A, b, G, h, nullopt, nullopt);
//
//    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
//    Status status = solver.solve();
//    PIQP_EIGEN_MALLOC_ALLOWED();
//
//    ASSERT_EQ(status, Status::PIQP_NON_CONVEX);
//}

TEST(DenseSolverTest, StronglyConvexWithEqualityAndInequalities)
{
    isize dim = 20;
    isize n_eq = 10;
    isize n_ineq = 12;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h, qp_model.x_lb, qp_model.x_ub);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

TEST(DenseSolverTest, NonStronglyConvexWithEqualityAndInequalities)
{
    isize dim = 20;
    isize n_eq = 10;
    isize n_ineq = 12;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq, 0.5, 0.0);

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h, qp_model.x_lb, qp_model.x_ub);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

TEST(DenseSolverTest, StronglyConvexOnlyEqualities)
{
    isize dim = 64;
    isize n_eq = 10;
    isize n_ineq = 0;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq, 0.0);

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h, qp_model.x_lb, qp_model.x_ub);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

TEST(DenseSolverTest, StronglyConvexOnlyInequalities)
{
    isize dim = 20;
    isize n_eq = 0;
    isize n_ineq = 12;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h, qp_model.x_lb, qp_model.x_ub);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

TEST(DenseSolverTest, StronglyConvexNoConstraints)
{
    isize dim = 64;
    isize n_eq = 0;
    isize n_ineq = 0;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq, 0.0);

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h, qp_model.x_lb, qp_model.x_ub);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}
