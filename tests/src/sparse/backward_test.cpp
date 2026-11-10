// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/utils/random_utils.hpp"

#include "gtest/gtest.h"
#include "backward_utils.hpp"

using namespace piqp;

using T = double;
using I = int;

class SparseBackwardTest : public testing::TestWithParam<KKTSolver>
{
protected:
    void SetUp() override
    {
#ifndef PIQP_HAS_BLASFEO
        if (GetParam() == KKTSolver::sparse_multistage || GetParam() == KKTSolver::sparse_multistage_parallel)
        {
            GTEST_SKIP() << "BLASFEO not available";
        }
#endif
#ifndef PIQP_HAS_OPENMP
        if (GetParam() == KKTSolver::sparse_multistage_parallel)
        {
            GTEST_SKIP() << "OpenMP not available";
        }
#endif
    }
};

namespace
{

std::pair<sparse::Model<T, I>, ActiveSet> make_qp(unsigned int seed)
{
    // same problem in every test, independent of the test order
    // (the distributions are reset as well, normal_distribution caches values)
    rand::gen.seed(seed);
    rand::uniform_dist.reset();
    rand::uniform_dist_pos.reset();
    rand::normal_dist.reset();
    return sparse_strongly_convex_qp_with_known_active_set<T, I>(12, 2, 8, 0.5);
}

sparse::Model<T, I> make_model()
{
    return make_qp(42).first;
}

template<typename Solver>
void configure(Solver& solver, KKTSolver kkt_solver)
{
    solver.settings().verbose = false;
    solver.settings().kkt_solver = kkt_solver;
    solver.settings().eps_abs = 1e-10;
    solver.settings().eps_rel = 1e-10;
    solver.settings().eps_duality_gap_abs = 1e-10;
    solver.settings().eps_duality_gap_rel = 1e-10;
    solver.settings().enable_backward = true;
}

} // namespace

// backward() requires enable_backward to be set before setup()
TEST_P(SparseBackwardTest, RequiresEnableBackward)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    solver.settings().enable_backward = false;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.backward(), Status::PIQP_INVALID_SETTINGS);

    // enabling it after setup() is not sufficient
    solver.settings().enable_backward = true;
    EXPECT_EQ(solver.backward(), Status::PIQP_INVALID_SETTINGS);

    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.backward(), Status::PIQP_SOLVED);
}

TEST_P(SparseBackwardTest, ProvidedResultInvalid)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    run_invalid_result_test(solver, model);
}

TEST_P(SparseBackwardTest, NoSideEffectsOnForward)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    SparseSolver<T, I> reference;
    configure(solver, GetParam());
    configure(reference, GetParam());
    run_no_side_effects_test(solver, reference, model);
}

TEST_P(SparseBackwardTest, FiniteDifferences)
{
    for (unsigned int seed : {1, 2, 3, 4, 5})
    {
        SCOPED_TRACE("seed " + std::to_string(seed));
        std::pair<sparse::Model<T, I>, ActiveSet> qp = make_qp(seed);
        const sparse::Model<T, I>& model = qp.first;
        SparseSolver<T, I> solver;
        configure(solver, GetParam());
        run_finite_difference_test(solver, model, qp.second);

        // the gradients have the sparsity pattern of the data, dP the one of the full symmetric P
        SparseMat<T, I> P_full = model.P.template selfadjointView<Eigen::Upper>();
        EXPECT_EQ(solver.gradients().P.nonZeros(), P_full.nonZeros());
        EXPECT_EQ(solver.gradients().A.nonZeros(), model.A.nonZeros());
        EXPECT_EQ(solver.gradients().G.nonZeros(), model.G.nonZeros());
    }
}

TEST_P(SparseBackwardTest, InvalidStates)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());

    EXPECT_EQ(solver.backward(), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.gradients().status, Status::PIQP_INVALID_SETTINGS);

    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    // not solved
    EXPECT_EQ(solver.backward(), Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.gradients().status, Status::PIQP_UNSOLVED);

    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.backward(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.gradients().status, Status::PIQP_SOLVED);

    // wrong dimensions, each adjoint is checked; a failed call invalidates the previous gradients
    Vec<T> wrong_n = Vec<T>::Zero(model.P.rows() + 1);
    Vec<T> wrong_p = Vec<T>::Zero(model.A.rows() + 1);
    Vec<T> wrong_m = Vec<T>::Zero(model.G.rows() + 1);
    EXPECT_EQ(solver.backward(wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.gradients().status, Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, wrong_p), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, wrong_m), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, wrong_m), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_m), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_m), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.gradients().status, Status::PIQP_INVALID_SETTINGS);

    EXPECT_EQ(solver.backward(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.gradients().status, Status::PIQP_SOLVED);

    // data changed since last solve, update() invalidates the gradients
    solver.update(nullopt, model.c);
    EXPECT_EQ(solver.gradients().status, Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.backward(), Status::PIQP_UNSOLVED);

    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.backward(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.gradients().status, Status::PIQP_SOLVED);

    // setup() invalidates the gradients
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    EXPECT_EQ(solver.gradients().status, Status::PIQP_UNSOLVED);
}

TEST_P(SparseBackwardTest, SmoothingAnalytic)
{
    SparseMat<T, I> P(1, 1);
    P.insert(0, 0) = T(1);
    P.makeCompressed();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    run_smoothing_analytic_test(solver, P);
}

INSTANTIATE_TEST_SUITE_P(
    KKTSolver,
    SparseBackwardTest,
    ::testing::Values(KKTSolver::sparse_ldlt,
                      KKTSolver::sparse_ldlt_eq_cond,
                      KKTSolver::sparse_ldlt_ineq_cond,
                      KKTSolver::sparse_ldlt_cond,
                      KKTSolver::sparse_multistage,
                      KKTSolver::sparse_multistage_parallel)
);
