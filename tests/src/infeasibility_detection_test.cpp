// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <tuple>

#include "piqp/piqp.hpp"
#include "piqp/utils/io_utils.hpp"

#include "gtest/gtest.h"
#include "utils.hpp"

using T = double;
using I = int;

template<typename Solver>
piqp::Status solve_model(Solver& solver, piqp::sparse::Model<T, I> model);

template<>
piqp::Status solve_model(piqp::DenseSolver<T>& solver, piqp::sparse::Model<T, I> sparse_model)
{
    piqp::dense::Model<T> model = sparse_model.dense_model();
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    return solver.solve();
}

template<>
piqp::Status solve_model(piqp::SparseSolver<T, I>& solver, piqp::sparse::Model<T, I> model)
{
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    return solver.solve();
}

/*
 * Feasible QP which was falsely reported as primal infeasible,
 * see https://github.com/PREDICT-EPFL/piqp/issues/37
 */
piqp::sparse::Model<T, I> issue37_model()
{
    T inf = std::numeric_limits<T>::infinity();

    piqp::Vec<T> P_diag(6); P_diag << 11, 1e7, 11, 1e7, 700, 1000;
    piqp::Mat<T> P = P_diag.asDiagonal();
    piqp::Vec<T> c(6); c << 876.8, 0, 876.8, 0, 0, 0;

    piqp::Mat<T> A(2, 6);
    A << 0.9973, -156.3, 0.9973, 154.2, 1, 0,
         -1.619, -1.084e4, 1.612, -1.084e4, 0, 1;
    piqp::Vec<T> b(2); b << 757.8, 1.255;

    piqp::Mat<T> G(0, 6);
    piqp::Vec<T> h_l(0);
    piqp::Vec<T> h_u(0);

    piqp::Vec<T> x_l(6); x_l << -2000, -0.3491, -2000, -0.3491, -inf, -inf;
    piqp::Vec<T> x_u(6); x_u << -0.0001, 0.3491, -0.0001, 0.3491, inf, inf;

    return piqp::sparse::Model<T, I>(P.sparseView(), c, A.sparseView(), b, G.sparseView(), h_l, h_u, x_l, x_u);
}

// (preconditioner_iter, preconditioner_scale_cost)
class Issue37Test : public testing::TestWithParam<std::tuple<int, bool>> {};

template<typename Solver>
void run_issue37(int preconditioner_iter, bool preconditioner_scale_cost)
{
    Solver solver;
    solver.settings().verbose = true;
    solver.settings().preconditioner_iter = preconditioner_iter;
    solver.settings().preconditioner_scale_cost = preconditioner_scale_cost;

    piqp::Status status = solve_model(solver, issue37_model());
    ASSERT_EQ(status, piqp::Status::PIQP_SOLVED);
    ASSERT_NEAR(solver.result().info.primal_obj, 1.488237e8, 1e-6 * 1.488237e8);
}

TEST_P(Issue37Test, DenseSolved)
{
    run_issue37<piqp::DenseSolver<T>>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

TEST_P(Issue37Test, SparseSolved)
{
    run_issue37<piqp::SparseSolver<T, I>>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    Settings,
    Issue37Test,
    testing::Combine(testing::Values(10, 0), testing::Bool()));

/*
 * Feasible, but unbounded QPs which were falsely reported as primal infeasible
 * or not detected, see https://github.com/PREDICT-EPFL/piqp/issues/49
 * The problems are stored in the original formulation from the issue (bounds),
 * and with fixed variables and equality rows as equality constraints (eq).
 */
// (problem name, preconditioner_iter)
class Issue49Test : public testing::TestWithParam<std::tuple<std::string, int>> {};

template<typename Solver>
void run_issue49(const std::string& name, int preconditioner_iter)
{
    piqp::sparse::Model<T, I> model = piqp::load_sparse_model<T, I>("data/issues/" + name + ".mat");

    Solver solver;
    solver.settings().verbose = true;
    solver.settings().preconditioner_iter = preconditioner_iter;

    piqp::Status status = solve_model(solver, model);
    ASSERT_EQ(status, piqp::Status::PIQP_DUAL_INFEASIBLE);
    assert_dual_infeasibility_certificate(model, solver.result(), solver.settings().eps_dual_inf);
}

TEST_P(Issue49Test, DenseDualInfeasible)
{
    run_issue49<piqp::DenseSolver<T>>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

TEST_P(Issue49Test, SparseDualInfeasible)
{
    run_issue49<piqp::SparseSolver<T, I>>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    Problems,
    Issue49Test,
    testing::Combine(testing::Values("issue49_unb42_bounds", "issue49_unb42_eq",
                                     "issue49_unb28_bounds", "issue49_unb28_eq"),
                     testing::Values(10, 0)),
    [](const testing::TestParamInfo<Issue49Test::ParamType>& info) {
        return std::get<0>(info.param) + "_prec" + std::to_string(std::get<1>(info.param));
    });

/*
 * min x1^2 + x2^2 + x1
 * s.t. x1 + x2 = 3, 1 <= x1 - x2, x1 + 2 x2 <= 10, 0 <= x <= 1
 * The certificate has to combine the equality, the inequalities, and the bounds.
 */
piqp::sparse::Model<T, I> primal_infeasible_model()
{
    T inf = std::numeric_limits<T>::infinity();

    piqp::Mat<T> P(2, 2); P << 2, 0, 0, 2;
    piqp::Vec<T> c(2); c << 1, 0;

    piqp::Mat<T> A(1, 2); A << 1, 1;
    piqp::Vec<T> b(1); b << 3;

    piqp::Mat<T> G(2, 2); G << 1, -1, 1, 2;
    piqp::Vec<T> h_l(2); h_l << 1, -inf;
    piqp::Vec<T> h_u(2); h_u << inf, 10;

    piqp::Vec<T> x_l(2); x_l << 0, 0;
    piqp::Vec<T> x_u(2); x_u << 1, 1;

    return piqp::sparse::Model<T, I>(P.sparseView(), c, A.sparseView(), b, G.sparseView(), h_l, h_u, x_l, x_u);
}

// preconditioner_iter
class PrimalInfeasibleTest : public testing::TestWithParam<int> {};

template<typename Solver>
void run_primal_infeasible(int preconditioner_iter)
{
    piqp::sparse::Model<T, I> model = primal_infeasible_model();

    Solver solver;
    solver.settings().verbose = true;
    solver.settings().preconditioner_iter = preconditioner_iter;

    piqp::Status status = solve_model(solver, model);
    ASSERT_EQ(status, piqp::Status::PIQP_PRIMAL_INFEASIBLE);
    assert_primal_infeasibility_certificate(model, solver.result(), solver.settings().eps_primal_inf);
}

TEST_P(PrimalInfeasibleTest, Dense)
{
    run_primal_infeasible<piqp::DenseSolver<T>>(GetParam());
}

TEST_P(PrimalInfeasibleTest, Sparse)
{
    run_primal_infeasible<piqp::SparseSolver<T, I>>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Settings,
    PrimalInfeasibleTest,
    testing::Values(10, 0));
