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

using solver_types = testing::Types<SparseSolver<T, I, KKTMode::KKT_FULL>,
                                    SparseSolver<T, I, KKTMode::KKT_EQ_ELIMINATED>,
                                    SparseSolver<T, I, KKTMode::KKT_INEQ_ELIMINATED>,
                                    SparseSolver<T, I, KKTMode::KKT_ALL_ELIMINATED>>;
template <typename T>
class SparseSolverTest : public ::testing::Test {};
TYPED_TEST_SUITE(SparseSolverTest, solver_types);

/*
 * first QP:
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 1, x1 = 2 x2
 *
 * second QP:
 * min 4 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 2, x1 = 3 x2
*/
TYPED_TEST(SparseSolverTest, SimpleQPWithUpdate)
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

    TypeParam solver;
    solver.settings().verbose = true;
    solver.setup(P, c, A, b, G, h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
    ASSERT_NEAR(solver.result().x(0), 0.4285714, 1e-6);
    ASSERT_NEAR(solver.result().x(1), 0.2142857, 1e-6);

    P.coeffRef(0, 0) = 8;
    A.coeffRef(0, 1) = -3;
    h(0) = 2;
    h(1) = 2;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    solver.update(P, c, A, b, nullopt, h);
    PIQP_EIGEN_MALLOC_ALLOWED();

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
    ASSERT_NEAR(solver.result().x(0), 0.2763157, 1e-6);
    ASSERT_NEAR(solver.result().x(1), 0.0921056, 1e-6);
}

/*
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x1 <= 0, 1 <= x2 <= 2,  x1 = 2 x2
*/
TYPED_TEST(SparseSolverTest, PrimalInfeasibleQP)
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
    Vec<T> h(4); h << 0, 2, 1, -1;

    TypeParam solver;
    solver.settings().verbose = true;
    solver.setup(P, c, A, b, G, h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_PRIMAL_INFEASIBLE);
}

/*
 * min -x1 - x2
 * s.t. 0 <= x
*/
TYPED_TEST(SparseSolverTest, DualInfeasibleQP)
{
    SparseMat<T, I> P(2, 2);
    P.setZero();
    Vec<T> c(2); c << -1, -1;

    SparseMat<T, I> A(0, 2);
    Vec<T> b(0);

    SparseMat<T, I> G(2, 2);
    G.insert(0, 0) = -1;
    G.insert(1, 1) = -1;
    G.makeCompressed();
    Vec<T> h(2); h << 0, 0;

    TypeParam solver;
    solver.settings().verbose = true;
    solver.setup(P, c, A, b, G, h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_DUAL_INFEASIBLE);
}

//TYPED_TEST(SparseSolverTest, NonConvexQP)
//{
//    Mat<T> P(2, 2); P << 2, 5, 5, 1;
//    Vec<T> c(2); c << 3, 4;
//
//    SparseMat<T, I> A(0, 2);
//    Vec<T> b(0);
//
//    Mat<T> G(5, 2); G << -1, 0, 0, -1, -1, 3, 2, 5, 3, 4;
//    Vec<T> h(5); h << 0, 0, -15, 100, 80;
//
//    TypeParam solver;
//    solver.settings().verbose = true;
//    solver.setup(P.sparseView(), A, G.sparseView(), c, b, h);
//
//    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
//    Status status = solver.solve();
//    PIQP_EIGEN_MALLOC_ALLOWED();
//
//    ASSERT_EQ(status, Status::PIQP_NON_CONVEX);
//}

TYPED_TEST(SparseSolverTest, StronglyConvexWithEqualityAndInequalities)
{
    isize dim = 20;
    isize n_eq = 10;
    isize n_ineq = 12;
    T sparsity_factor = 0.2;

    sparse::Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    TypeParam solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

TYPED_TEST(SparseSolverTest, NonStronglyConvexWithEqualityAndInequalities)
{
    isize dim = 20;
    isize n_eq = 10;
    isize n_ineq = 12;
    T sparsity_factor = 0.2;

    sparse::Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor, 0.0);

    TypeParam solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

TYPED_TEST(SparseSolverTest, StronglyConvexOnlyEqualities)
{
    isize dim = 20;
    isize n_eq = 10;
    isize n_ineq = 0;
    T sparsity_factor = 0.2;

    sparse::Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    TypeParam solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

TYPED_TEST(SparseSolverTest, StronglyConvexOnlyInequalities)
{
    isize dim = 20;
    isize n_eq = 0;
    isize n_ineq = 12;
    T sparsity_factor = 0.2;

    sparse::Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    TypeParam solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

TYPED_TEST(SparseSolverTest, StronglyConvexNoConstraints)
{
    isize dim = 20;
    isize n_eq = 0;
    isize n_ineq = 0;
    T sparsity_factor = 0.2;

    sparse::Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    TypeParam solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}
