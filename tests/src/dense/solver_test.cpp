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

    // last constraint is redundant and just for testing
    Mat<T> G(3, 2); G << 1, 0, 1, 0, 1, 0;
    Vec<T> h_l(3); h_l << -1, -std::numeric_limits<T>::infinity(), -2;
    Vec<T> h_u(3); h_u << std::numeric_limits<T>::infinity(), 1, 2;
    // Mat<T> G(4, 2); G << 1, 0, 1, 0, -1, 0, -1, 0;
    // Vec<T> h_u(4); h_u << 1, 2, 1, 2;
    // Mat<T> G(5, 2); G << 1, 0, 1, 0, 1, 0, -1, 0, -1, 0;
    // Vec<T> h_u(5); h_u << 2, 1, 2, 1, 2;

    Vec<T> x_l(2); x_l << -std::numeric_limits<T>::infinity(), -1;
    Vec<T> x_u(2); x_u << std::numeric_limits<T>::infinity(), 1;

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
    // solver.setup(P, c, A, b, G, nullopt, h_u, x_l, x_u);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
    ASSERT_NEAR(solver.result().x(0), 0.4285714, 1e-6);
    ASSERT_NEAR(solver.result().x(1), 0.2142857, 1e-6);
    ASSERT_NEAR(solver.result().y(0), -1.5714286, 1e-6);
    ASSERT_NEAR(solver.result().z_l(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_l(1), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_l(2), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_u(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_u(1), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_u(2), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_bl(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_bl(1), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_bu(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_bu(1), 0, 1e-6);

    P(0, 0) = 8;
    A(0, 1) = -3;
    h_u(0) = 2;
    x_u(1) = 2;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    solver.update(P, c, A, b, nullopt, nullopt, h_u, nullopt, x_u);
    PIQP_EIGEN_MALLOC_ALLOWED();

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
    ASSERT_NEAR(solver.result().x(0), 0.2763157, 1e-6);
    ASSERT_NEAR(solver.result().x(1), 0.0921056, 1e-6);
    ASSERT_NEAR(solver.result().y(0), -1.2105263, 1e-6);
    ASSERT_NEAR(solver.result().z_l(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_l(1), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_l(2), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_u(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_u(1), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_u(2), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_bl(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_bl(1), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_bu(0), 0, 1e-6);
    ASSERT_NEAR(solver.result().z_bu(1), 0, 1e-6);
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
    solver.setup(P, c, A, b, G, nullopt, h, nullopt, nullopt);

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
    solver.setup(P, c, A, b, G, nullopt, h, nullopt, nullopt);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_DUAL_INFEASIBLE);
}

TEST(DenseSolverTest, IllConditionedSmall)
{
    T inf = std::numeric_limits<T>::infinity();

    Mat<T> P(6, 6);
    P.setZero();
    P.diagonal() << 61, 2e+09, 61, 2e+09, 1000, 100;
    Vec<T> c(6); c.setZero();

    Mat<T> A(2, 6);
    A << 1,    0,    1,    0,    1,    0,
         2.4,  0, -2.4,    0,    0,    1;
    Vec<T> b(2); b.setZero();

    Vec<T> x_l(6); x_l << -2e+04, -0.3491, -2e+04, -0.3491, -inf, -inf;
    Vec<T> x_u(6); x_u << 2e+04, 0.3491, 2e+04, 0.3491, inf, inf;

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(P, c, A, b, nullopt, nullopt, nullopt, x_l, x_u);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
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
//    solver.setup(P, c, A, b, G, nullopt, h, nullopt, nullopt);
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
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

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
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

TEST(DenseSolverTest, SameResultWithRuizPreconditioner)
{
    isize dim = 20;
    isize n_eq = 10;
    isize n_ineq = 12;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq, 0.5, 0.0);

    DenseSolver<T, dense::IdentityPreconditioner<T>> solver_no_precon;
    solver_no_precon.settings().eps_rel = 0;
    solver_no_precon.settings().verbose = true;
    solver_no_precon.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

    DenseSolver<T, dense::RuizEquilibration<T>> solver_ruiz;
    solver_ruiz.settings().eps_rel = 0;
    solver_ruiz.settings().verbose = true;
    solver_ruiz.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver_no_precon.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    status = solver_ruiz.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);

    ASSERT_LT((solver_no_precon.result().x - solver_ruiz.result().x).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().y - solver_ruiz.result().y).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().z - solver_ruiz.result().z).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().z_lb - solver_ruiz.result().z_lb).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().z_ub - solver_ruiz.result().z_ub).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().s - solver_ruiz.result().s).norm(), 1e-6);
//    // convert inf to something finite
//    ASSERT_LT((solver_no_precon.result().s_lb.cwiseMin(1e10) - solver_ruiz.result().s_lb.cwiseMin(1e10)).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().s_ub.cwiseMin(1e10) - solver_ruiz.result().s_ub.cwiseMin(1e10)).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().zeta - solver_ruiz.result().zeta).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().lambda - solver_ruiz.result().lambda).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().nu - solver_ruiz.result().nu).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().nu_lb - solver_ruiz.result().nu_lb).norm(), 1e-6);
//    ASSERT_LT((solver_no_precon.result().nu_ub - solver_ruiz.result().nu_ub).norm(), 1e-6);
}

TEST(DenseSolverTest, StronglyConvexOnlyEqualities)
{
    isize dim = 64;
    isize n_eq = 10;
    isize n_ineq = 0;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq, 0.0);

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

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
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

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
    solver.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
}

TEST(DenseSolverTest, InfinityBounds)
{
    Mat<T> P(4, 4); P << 1, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, 1, 0,
                         0, 0, 0, 1;
    Vec<T> c(4); c << 1, 1, 1, 1;

    Mat<T> G(6, 4); G << 1,  0, 0,  0,
                         1,  0, -1, 0,
                         -1, 0, -1, 0,
                         -1, 0, 0,  0,
                         -1, 0, 1,  0,
                         1,  0, 1,  0;
    T inf = std::numeric_limits<T>::infinity();
    Vec<T> h(6); h << 1, 1, 1, 1, inf, inf;

    DenseSolver<T> solver;
    solver.settings().verbose = true;
    solver.setup(P, c, piqp::nullopt, piqp::nullopt, G, piqp::nullopt, h);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status, Status::PIQP_SOLVED);
    ASSERT_NEAR(solver.result().x(0), -0.5, 1e-6);
    ASSERT_NEAR(solver.result().x(1), -1.0, 1e-6);
    ASSERT_NEAR(solver.result().x(2), -0.5, 1e-6);
    ASSERT_NEAR(solver.result().x(3), -1.0, 1e-6);
}

TEST(DenseSolverTest, CopyConstructor)
{
    isize dim = 20;
    isize n_eq = 10;
    isize n_ineq = 12;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);

    DenseSolver<T> solver1;
    solver1.settings().verbose = true;
    solver1.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

    DenseSolver<T> solver2(solver1);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status1 = solver1.solve();
    Status status2 = solver2.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status1, Status::PIQP_SOLVED);
    ASSERT_EQ(status2, Status::PIQP_SOLVED);
    ASSERT_EQ(solver1.result().x, solver2.result().x);
}

TEST(DenseSolverTest, MoveConstructor)
{
    isize dim = 20;
    isize n_eq = 10;
    isize n_ineq = 12;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);

    DenseSolver<T> solver1;
    solver1.settings().verbose = true;
    solver1.setup(qp_model.P, qp_model.c, qp_model.A, qp_model.b, qp_model.G, qp_model.h_l, qp_model.h_u, qp_model.x_l, qp_model.x_u);

    DenseSolver<T> solver2(std::move(solver1));

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    Status status2 = solver2.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(status2, Status::PIQP_SOLVED);
}
