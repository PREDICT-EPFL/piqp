// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/utils/io_utils.hpp"
#include "piqp/utils/random_utils.hpp"
#include "piqp/sparse/multistage_kkt.hpp"
#include "piqp/utils/optional.hpp"

#include <iostream>

#include <fstream>
#include <iomanip>
#include "gtest/gtest.h"

using T = double;
using I = int;

using namespace piqp;
using namespace piqp::sparse;

class BlocksparseStageParallelKKTTest : public testing::TestWithParam<std::string> {};

template<typename KKT1, typename KKT2>
void test_solve_multiply(Data<T, I>& data, Settings<T> settings1, Settings<T> settings2, KKT1& kkt1, KKT2& kkt2)
{
    Variables<T> rhs;
    rhs.x = rand::vector_rand<T>(data.n);
    rhs.y = rand::vector_rand<T>(data.p);
    rhs.z_l = rand::vector_rand<T>(data.m);
    rhs.z_u = rand::vector_rand<T>(data.m);
    rhs.z_bl = rand::vector_rand<T>(data.n);
    rhs.z_bu = rand::vector_rand<T>(data.n);
    rhs.s_l = rand::vector_rand<T>(data.m);
    rhs.s_u = rand::vector_rand<T>(data.m);
    rhs.s_bl = rand::vector_rand<T>(data.n);
    rhs.s_bu = rand::vector_rand<T>(data.n);

    Variables<T> lhs_1;
    lhs_1.resize(data.n, data.p, data.m);

    Variables<T> lhs_2;
    lhs_2.resize(data.n, data.p, data.m);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt1.solve(data, settings1, rhs, lhs_1);
    kkt2.solve(data, settings2, rhs, lhs_2);
    PIQP_EIGEN_MALLOC_ALLOWED();


    ASSERT_TRUE(lhs_1.x.isApprox(lhs_2.x, 1e-8));
    ASSERT_TRUE(lhs_1.y.isApprox(lhs_2.y, 1e-8));
    ASSERT_TRUE(lhs_1.z_bl.head(data.n_x_l).isApprox(lhs_2.z_bl.head(data.n_x_l), 1e-8));
    ASSERT_TRUE(lhs_1.z_bu.head(data.n_x_u).isApprox(lhs_2.z_bu.head(data.n_x_u), 1e-8));
    ASSERT_TRUE(lhs_1.s_bl.head(data.n_x_l).isApprox(lhs_2.s_bl.head(data.n_x_l), 1e-8));
    ASSERT_TRUE(lhs_1.s_bu.head(data.n_x_u).isApprox(lhs_2.s_bu.head(data.n_x_u), 1e-8));
    for (isize i = 0; i < data.n_h_l; i++)
    {
        Eigen::Index idx = data.h_l_idx(i);
        ASSERT_NEAR(lhs_1.z_l(idx), lhs_2.z_l(idx), 1e-8);
        ASSERT_NEAR(lhs_1.s_l(idx), lhs_2.s_l(idx), 1e-8);
    }
    for (isize i = 0; i < data.n_h_u; i++)
    {
        Eigen::Index idx = data.h_u_idx(i);
        ASSERT_NEAR(lhs_1.z_u(idx), lhs_2.z_u(idx), 1e-8);
        ASSERT_NEAR(lhs_1.s_u(idx), lhs_2.s_u(idx), 1e-8);
    }

    Variables<T> rhs_sol_1;
    rhs_sol_1.resize(data.n, data.p, data.m);

    Variables<T> rhs_sol_2;
    rhs_sol_2.resize(data.n, data.p, data.m);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt1.mul(data, lhs_1, rhs_sol_1);
    kkt2.mul(data, lhs_2, rhs_sol_2);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(rhs_sol_1.x.isApprox(rhs_sol_2.x, 1e-8));
    ASSERT_TRUE(rhs_sol_1.y.isApprox(rhs_sol_2.y, 1e-8));
    ASSERT_TRUE(rhs_sol_1.z_bl.head(data.n_x_l).isApprox(rhs_sol_2.z_bl.head(data.n_x_l), 1e-8));
    ASSERT_TRUE(rhs_sol_1.z_bu.head(data.n_x_u).isApprox(rhs_sol_2.z_bu.head(data.n_x_u), 1e-8));
    ASSERT_TRUE(rhs_sol_1.s_bl.head(data.n_x_l).isApprox(rhs_sol_2.s_bl.head(data.n_x_l), 1e-8));
    ASSERT_TRUE(rhs_sol_1.s_bu.head(data.n_x_u).isApprox(rhs_sol_2.s_bu.head(data.n_x_u), 1e-8));
    for (isize i = 0; i < data.n_h_l; i++)
    {
        Eigen::Index idx = data.h_l_idx(i);
        ASSERT_NEAR(rhs_sol_1.z_l(idx), rhs_sol_2.z_l(idx), 1e-8);
        ASSERT_NEAR(rhs_sol_1.s_l(idx), rhs_sol_2.s_l(idx), 1e-8);
    }
    for (isize i = 0; i < data.n_h_u; i++)
    {
        Eigen::Index idx = data.h_u_idx(i);
        ASSERT_NEAR(rhs_sol_1.z_u(idx), rhs_sol_2.z_u(idx), 1e-8);
        ASSERT_NEAR(rhs_sol_1.s_u(idx), rhs_sol_2.s_u(idx), 1e-8);
    }
}

// The robot arm problems are highly degenerate with non-unique minimizers. Different KKT backends
// accumulate different rounding errors and converge to different minimizers with the same objective.
bool has_unique_solution(const std::string& name)
{
    return name.rfind("robot_arm_sqp", 0) != 0;
}

template<typename PIQPSolver1, typename PIQPSolver2>
void compare_results(const PIQPSolver1& solver1, const PIQPSolver2& solver2, bool unique_solution)
{
    ASSERT_EQ(solver1.result().info.status, solver2.result().info.status);
    if (!unique_solution) {
        if (solver1.result().info.status == PIQP_SOLVED) {
            ASSERT_NEAR(solver1.result().info.primal_obj, solver2.result().info.primal_obj, 1e-8);
        }
        return;
    }
    ASSERT_EQ(solver1.result().info.iter, solver2.result().info.iter);
    if (solver1.result().info.status == PIQP_SOLVED) {
        ASSERT_TRUE(solver1.result().x.isApprox(solver2.result().x, 1e-8));
        ASSERT_TRUE(solver1.result().y.isApprox(solver2.result().y, 1e-8));
        ASSERT_TRUE(solver1.result().z_l.isApprox(solver2.result().z_l, 1e-8));
        ASSERT_TRUE(solver1.result().z_u.isApprox(solver2.result().z_u, 1e-8));
    }
}

template<typename PIQPSolver1, typename PIQPSolver2>
void test_solve_qp(Model<T, I>& model, PIQPSolver1& solver1, PIQPSolver2& solver2, bool unique_solution = true)
{
    // setup and solve
    solver1.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    solver2.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    solver1.solve();
    solver2.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    compare_results(solver1, solver2, unique_solution);

    // update and solve
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    solver1.update(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    solver2.update(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    solver1.solve();
    solver2.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    compare_results(solver1, solver2, unique_solution);
}

TEST(BlocksparseStageParallelKKTTest, FactorizeSolveSQPBlocksize1)
{
    Eigen::Index N = 35;
    SparseMat<T, I> P(N, N); P.setIdentity();
    for (Eigen::Index i = 0; i < N; ++i) {
        T tmp = static_cast<T>(i+1) / static_cast<T>(N) / static_cast<T>(100);
        P.coeffRef(N-1, i) = tmp; // last row
        P.coeffRef(i, N-1) = tmp; // last column
    }
    Vec<T> c(N); c.setConstant(1.0);
    SparseMat<T, I> A(N-1, N);
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<size_t>(N) * 2);  // each row has at most 2 nonzeros
    for (int i = 0; i < N-1; ++i) {
        triplets.emplace_back(i, i, 1.0);       // diagonal
        triplets.emplace_back(i, i + 1, 1.0);   // superdiagonal
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    Vec<T> b(N-1); b.setConstant(1.0);
    Model<T, I> model(P, c, A, b, nullopt, nullopt, nullopt, nullopt);
    Data<T, I> data(model);

    Settings<T> settings_multistage;
    settings_multistage.kkt_solver = KKTSolver::sparse_multistage;

    Settings<T> settings_sparse;
    settings_sparse.kkt_solver = KKTSolver::sparse_ldlt;

    Settings<T> settings_multistage_parallel;
    settings_multistage_parallel.kkt_solver = KKTSolver::sparse_multistage_parallel;

    T rho = 0.9;
    T delta = 1.2;
    Variables<T> scaling; scaling.resize(data.n, data.p, data.m);
    scaling.s_l.setConstant(1);
    scaling.s_u.setConstant(1);
    scaling.s_bl.setConstant(1);
    scaling.s_bu.setConstant(1);
    scaling.z_l.setConstant(1);
    scaling.z_u.setConstant(1);
    scaling.z_bl.setConstant(1);
    scaling.z_bu.setConstant(1);

    KKTSystem<T, I, PIQP_SPARSE> kkt_multistage;
    kkt_multistage.init(data, settings_multistage);
    KKTSystem<T, I, PIQP_SPARSE> kkt_multistage_parallel;
    kkt_multistage_parallel.init(data, settings_multistage_parallel);
    KKTSystem<T, I, PIQP_SPARSE> kkt_sparse;
    kkt_sparse.init(data, settings_sparse);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_multistage.update_scalings_and_factor(data, settings_multistage, false, rho, delta, scaling);
    kkt_sparse.update_scalings_and_factor(data, settings_sparse, false, rho, delta, scaling);
    kkt_multistage_parallel.update_scalings_and_factor(data, settings_multistage_parallel, false, rho, delta, scaling);
    PIQP_EIGEN_MALLOC_ALLOWED();

    test_solve_multiply(data, settings_multistage, settings_sparse, kkt_multistage, kkt_sparse);
    test_solve_multiply(data, settings_multistage, settings_multistage_parallel, kkt_multistage, kkt_multistage_parallel);
}

TEST_P(BlocksparseStageParallelKKTTest, FactorizeSolveSQP)
{
    std::string path = "data/" + GetParam() + ".mat";
    Model<T, I> model = load_sparse_model<T, I>(path);
    Data<T, I> data(model);

    Settings<T> settings_sparse;
    settings_sparse.kkt_solver = KKTSolver::sparse_ldlt;

    Settings<T> settings_multistage_serial;
    settings_multistage_serial.kkt_solver = KKTSolver::sparse_multistage;

    Settings<T> settings_multistage_parallel;
    settings_multistage_parallel.kkt_solver = KKTSolver::sparse_multistage_parallel;

    T rho = 0.9;
    T delta = 1.2;

    Variables<T> scaling; scaling.resize(data.n, data.p, data.m);
    scaling.s_l.setConstant(1);
    scaling.s_u.setConstant(1);
    scaling.s_bl.setConstant(1);
    scaling.s_bu.setConstant(1);
    scaling.z_l.setConstant(1);
    scaling.z_u.setConstant(1);
    scaling.z_bl.setConstant(1);
    scaling.z_bu.setConstant(1);

    KKTSystem<T, I, PIQP_SPARSE> kkt_sparse;
    kkt_sparse.init(data, settings_sparse);
    KKTSystem<T, I, PIQP_SPARSE> kkt_multistage_serial;
    kkt_multistage_serial.init(data, settings_multistage_serial);
    KKTSystem<T, I, PIQP_SPARSE> kkt_multistage_parallel;
    kkt_multistage_parallel.init(data, settings_multistage_parallel);

    kkt_sparse.update_scalings_and_factor(data, settings_sparse, false, rho, delta, scaling);
    kkt_multistage_serial.update_scalings_and_factor(data, settings_multistage_serial, false, rho, delta, scaling);
    kkt_multistage_parallel.update_scalings_and_factor(data, settings_multistage_parallel, false, rho, delta, scaling);

    test_solve_multiply(data, settings_sparse, settings_multistage_parallel, kkt_sparse, kkt_multistage_parallel);
    test_solve_multiply(data, settings_multistage_serial, settings_multistage_parallel, kkt_multistage_serial, kkt_multistage_parallel);
}

TEST_P(BlocksparseStageParallelKKTTest, SolveQP)
{
    std::string path = "data/" + GetParam() + ".mat";
    Model<T, I> model = load_sparse_model<T, I>(path);

    SparseSolver<T, I> solver_multistage;
    solver_multistage.settings().kkt_solver = KKTSolver::sparse_multistage;
    SparseSolver<T, I> solver_multistage_parallel;
    solver_multistage_parallel.settings().kkt_solver = KKTSolver::sparse_multistage_parallel;

    test_solve_qp(model, solver_multistage, solver_multistage_parallel, has_unique_solution(GetParam()));
}

TEST_P(BlocksparseStageParallelKKTTest, SolveQPNumThreads)
{
    std::string path = "data/" + GetParam() + ".mat";
    Model<T, I> model = load_sparse_model<T, I>(path);

    for (isize num_threads : {1, 2, 3})
    {
        SCOPED_TRACE("num_threads = " + std::to_string(num_threads));

        SparseSolver<T, I> solver_multistage;
        solver_multistage.settings().kkt_solver = KKTSolver::sparse_multistage;
        solver_multistage.settings().num_threads = num_threads;
        SparseSolver<T, I> solver_multistage_parallel;
        solver_multistage_parallel.settings().kkt_solver = KKTSolver::sparse_multistage_parallel;
        solver_multistage_parallel.settings().num_threads = num_threads;

        test_solve_qp(model, solver_multistage, solver_multistage_parallel, has_unique_solution(GetParam()));
    }
}

TEST_P(BlocksparseStageParallelKKTTest, CopyConstructor)
{
    std::string path = "data/" + GetParam() + ".mat";
    Model<T, I> model = load_sparse_model<T, I>(path);

    SparseSolver<T, I> solver1;
    solver1.settings().kkt_solver = KKTSolver::sparse_multistage_parallel;
    solver1.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    solver1.solve();

    SparseSolver<T, I> solver2(solver1);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    solver1.solve();
    solver2.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_EQ(solver1.result().info.status, solver2.result().info.status);
    ASSERT_EQ(solver1.result().info.iter, solver2.result().info.iter);
    // the copy has to behave identically, also if both diverge to NaN
    const auto& x1 = solver1.result().x.array();
    const auto& x2 = solver2.result().x.array();
    ASSERT_TRUE((x1 == x2 || (x1.isNaN() && x2.isNaN())).all());
}

TEST(BlocksparseStageParallelKKTTest, SolveIllConditionedQP)
{
    for (const std::string name : {"robot_arm_sqp", "robot_arm_sqp_no_global"})
    {
        SCOPED_TRACE(name);
        Model<T, I> model = load_sparse_model<T, I>("data/" + name + ".mat");

        SparseSolver<T, I> solver;
        solver.settings().kkt_solver = KKTSolver::sparse_multistage_parallel;
        solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);

        ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    }
}

INSTANTIATE_TEST_SUITE_P(FromFolder, BlocksparseStageParallelKKTTest,
                         ::testing::Values("scenario_mpc", "chain_mass_sqp", "race_line", "robot_arm_sqp",
                                           "robot_arm_sqp_constr_perm", "robot_arm_sqp_no_global"));
