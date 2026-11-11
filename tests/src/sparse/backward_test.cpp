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

void expect_same_pattern(const SparseMat<T, I>& a, SparseMat<T, I> b)
{
    b.makeCompressed();
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    ASSERT_EQ(a.nonZeros(), b.nonZeros());
    EXPECT_TRUE(std::equal(a.outerIndexPtr(), a.outerIndexPtr() + a.outerSize() + 1, b.outerIndexPtr()));
    EXPECT_TRUE(std::equal(a.innerIndexPtr(), a.innerIndexPtr() + a.nonZeros(), b.innerIndexPtr()));
}

} // namespace

// the P gradient follows the pattern of the P passed to setup(): upper triangular or full symmetric
TEST_P(SparseBackwardTest, GradientPatternFollowsInput)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.backward(solver.result().x), Status::PIQP_SOLVED);
    expect_same_pattern(solver.gradients().P, model.P);
    SparseMat<T, I> dP_utri = solver.gradients().P;

    SparseMat<T, I> P_full = model.P.template selfadjointView<Eigen::Upper>();
    solver.setup(P_full, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.backward(solver.result().x), Status::PIQP_SOLVED);
    const SparseMat<T, I>& dP_full = solver.gradients().P;
    expect_same_pattern(dP_full, P_full);

    // same values on the upper triangular part, mirrored to the lower one
    Mat<T> dP_full_dense = Mat<T>(dP_full);
    EXPECT_LT((dP_full_dense - dP_full_dense.transpose()).norm(), 1e-12);
    EXPECT_LT((Mat<T>(dP_full_dense.triangularView<Eigen::Upper>()) - Mat<T>(dP_utri)).norm(), 1e-8);
}

// setup() with an unchanged sparsity pattern keeps the storage of the gradients, such that views into it
// (e.g., in the Python interface) stay valid
TEST_P(SparseBackwardTest, SetupKeepsGradientStorage)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    const T* P_values = solver.gradients().P.valuePtr();
    const T* A_values = solver.gradients().A.valuePtr();
    const T* G_values = solver.gradients().G.valuePtr();

    // same pattern, different values
    sparse::Model<T, I> scaled = model;
    scaled.P *= T(2);
    scaled.A *= T(2);
    scaled.G *= T(2);
    solver.setup(scaled.P, scaled.c, scaled.A, scaled.b, scaled.G, scaled.h_l, scaled.h_u, scaled.x_l, scaled.x_u);
    EXPECT_EQ(solver.gradients().P.valuePtr(), P_values);
    EXPECT_EQ(solver.gradients().A.valuePtr(), A_values);
    EXPECT_EQ(solver.gradients().G.valuePtr(), G_values);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.backward(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.gradients().P.valuePtr(), P_values);

    // same dimensions and number of nonzeros, but an entry of G moved to a different column
    sparse::Model<T, I> moved = model;
    {
        std::vector<Eigen::Triplet<T, I>> triplets;
        for (Eigen::Index j = 0; j < model.G.outerSize(); j++)
        {
            for (SparseMat<T, I>::InnerIterator it(model.G, j); it; ++it)
            {
                triplets.emplace_back(static_cast<I>(it.row()), static_cast<I>(it.col()), it.value());
            }
        }
        Eigen::Triplet<T, I>& t = triplets[0];
        I free_col = 0;
        while (model.G.coeff(t.row(), free_col) != T(0)) free_col++;
        ASSERT_LT(free_col, model.G.cols());
        t = Eigen::Triplet<T, I>(t.row(), free_col, t.value());
        moved.G.setFromTriplets(triplets.begin(), triplets.end());
        moved.G.makeCompressed();
        ASSERT_EQ(moved.G.nonZeros(), model.G.nonZeros());
    }
    solver.setup(moved.P, moved.c, moved.A, moved.b, moved.G, moved.h_l, moved.h_u, moved.x_l, moved.x_u);
    const SparseMat<T, I>& G_grad = solver.gradients().G;
    ASSERT_EQ(G_grad.nonZeros(), moved.G.nonZeros());
    EXPECT_TRUE(std::equal(G_grad.outerIndexPtr(), G_grad.outerIndexPtr() + G_grad.outerSize() + 1, moved.G.outerIndexPtr()));
    EXPECT_TRUE(std::equal(G_grad.innerIndexPtr(), G_grad.innerIndexPtr() + G_grad.nonZeros(), moved.G.innerIndexPtr()));
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.backward(), Status::PIQP_SOLVED);

    // changed dimensions: the gradients take over the new pattern
    solver.setup(model.P, model.c, model.A, model.b, nullopt, nullopt, nullopt, model.x_l, model.x_u);
    EXPECT_EQ(solver.gradients().G.rows(), 0);
    EXPECT_EQ(solver.gradients().G.nonZeros(), 0);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.backward(), Status::PIQP_SOLVED);
}

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

        // the gradients have the sparsity patterns of the matrices passed to setup()
        expect_same_pattern(solver.gradients().P, model.P);
        expect_same_pattern(solver.gradients().A, model.A);
        expect_same_pattern(solver.gradients().G, model.G);
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

TEST_P(SparseBackwardTest, SmoothingEqualities)
{
    SparseSolver<T, I> solver;
    SparseSolver<T, I> reference;
    configure(solver, GetParam());
    configure(reference, GetParam());
    run_smoothing_equality_test(solver, reference, make_model());
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
