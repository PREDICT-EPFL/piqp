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
#include "derivative_utils.hpp"

using namespace piqp;

using T = double;
using I = int;

class SparseDerivativeTest : public testing::TestWithParam<KKTSolver>
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

std::pair<sparse::Model<T, I>, rand::ActiveSet> make_qp(unsigned int seed)
{
    // same problem in every test, independent of the test order
    rand::seed(seed);
    return rand::sparse_strongly_convex_qp_with_known_active_set<T, I>(12, 2, 8, 0.5);
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
    solver.settings().enable_derivatives = true;
}

} // namespace

// the P gradient follows the pattern of the P passed to setup(): upper triangular or full symmetric
TEST_P(SparseDerivativeTest, GradientPatternFollowsInput)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.vjp(solver.result().x), Status::PIQP_SOLVED);
    expect_same_pattern(solver.vjp_result().P, model.P);
    SparseMat<T, I> dP_utri = solver.vjp_result().P;

    SparseMat<T, I> P_full = model.P.template selfadjointView<Eigen::Upper>();
    solver.setup(P_full, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.vjp(solver.result().x), Status::PIQP_SOLVED);
    const SparseMat<T, I>& dP_full = solver.vjp_result().P;
    expect_same_pattern(dP_full, P_full);

    // same values on the upper triangular part, mirrored to the lower one
    Mat<T> dP_full_dense = Mat<T>(dP_full);
    EXPECT_LT((dP_full_dense - dP_full_dense.transpose()).norm(), 1e-12);
    EXPECT_LT((Mat<T>(dP_full_dense.triangularView<Eigen::Upper>()) - Mat<T>(dP_utri)).norm(), 1e-8);
}

// setup() with an unchanged sparsity pattern keeps the storage of the gradients, such that views into it
// (e.g., in the Python interface) stay valid
TEST_P(SparseDerivativeTest, SetupKeepsGradientStorage)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    const T* P_values = solver.vjp_result().P.valuePtr();
    const T* A_values = solver.vjp_result().A.valuePtr();
    const T* G_values = solver.vjp_result().G.valuePtr();

    // same pattern, different values
    sparse::Model<T, I> scaled = model;
    scaled.P *= T(2);
    scaled.A *= T(2);
    scaled.G *= T(2);
    solver.setup(scaled.P, scaled.c, scaled.A, scaled.b, scaled.G, scaled.h_l, scaled.h_u, scaled.x_l, scaled.x_u);
    EXPECT_EQ(solver.vjp_result().P.valuePtr(), P_values);
    EXPECT_EQ(solver.vjp_result().A.valuePtr(), A_values);
    EXPECT_EQ(solver.vjp_result().G.valuePtr(), G_values);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.vjp(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.vjp_result().P.valuePtr(), P_values);

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
    expect_same_pattern(solver.vjp_result().G, moved.G);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.vjp(), Status::PIQP_SOLVED);

    // changed dimensions: the gradients take over the new pattern
    solver.setup(model.P, model.c, model.A, model.b, nullopt, nullopt, nullopt, model.x_l, model.x_u);
    EXPECT_EQ(solver.vjp_result().G.rows(), 0);
    EXPECT_EQ(solver.vjp_result().G.nonZeros(), 0);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.vjp(), Status::PIQP_SOLVED);
}

TEST_P(SparseDerivativeTest, ProvidedResultInvalid)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    run_invalid_result_test(solver, model);
}

TEST_P(SparseDerivativeTest, DefaultMatchesProvidedResult)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    run_default_matches_provided_test(solver, model);
}

TEST_P(SparseDerivativeTest, NoSideEffectsOnSolve)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    SparseSolver<T, I> reference;
    configure(solver, GetParam());
    configure(reference, GetParam());
    run_no_side_effects_test(solver, reference, model);
}

TEST_P(SparseDerivativeTest, FiniteDifferences)
{
    for (unsigned int seed : {1, 2, 3, 4, 5})
    {
        SCOPED_TRACE("seed " + std::to_string(seed));
        std::pair<sparse::Model<T, I>, rand::ActiveSet> qp = make_qp(seed);
        const sparse::Model<T, I>& model = qp.first;
        SparseSolver<T, I> solver;
        configure(solver, GetParam());
        run_finite_difference_test(solver, model, qp.second);

        // the gradients have the sparsity patterns of the matrices passed to setup()
        expect_same_pattern(solver.vjp_result().P, model.P);
        expect_same_pattern(solver.vjp_result().A, model.A);
        expect_same_pattern(solver.vjp_result().G, model.G);
    }
}

TEST_P(SparseDerivativeTest, AdjointIdentity)
{
    for (unsigned int seed : {1, 2, 3})
    {
        SCOPED_TRACE("seed " + std::to_string(seed));
        SparseSolver<T, I> solver;
        configure(solver, GetParam());
        run_adjoint_identity_test(solver, make_qp(seed).first);
    }
}

TEST_P(SparseDerivativeTest, LinearizationReuse)
{
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    run_linearization_reuse_test(solver, make_model());
}

TEST_P(SparseDerivativeTest, JVPIgnoredInputs)
{
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    run_jvp_ignored_inputs_test(solver, make_model());
}

TEST_P(SparseDerivativeTest, InvalidStates)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    run_invalid_states_test(solver, model);
}

TEST_P(SparseDerivativeTest, NoConstraints)
{
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    run_no_constraints_test(solver, rand::sparse_positive_definite_upper_triangular_rand<T, I>(4, T(0.5), T(1)));
}

TEST_P(SparseDerivativeTest, SmoothingAnalytic)
{
    SparseMat<T, I> P(1, 1);
    P.insert(0, 0) = T(1);
    P.makeCompressed();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    run_smoothing_analytic_test(solver, P);
}

TEST_P(SparseDerivativeTest, SmoothingEqualities)
{
    SparseSolver<T, I> solver;
    SparseSolver<T, I> reference;
    configure(solver, GetParam());
    configure(reference, GetParam());
    run_smoothing_equality_test(solver, reference, make_model());
}

TEST_P(SparseDerivativeTest, CopyMove)
{
    sparse::Model<T, I> model = make_model();
    SparseSolver<T, I> solver;
    configure(solver, GetParam());
    run_copy_move_test(solver, model);
}

INSTANTIATE_TEST_SUITE_P(
    KKTSolver,
    SparseDerivativeTest,
    ::testing::Values(KKTSolver::sparse_ldlt,
                      KKTSolver::sparse_ldlt_eq_cond,
                      KKTSolver::sparse_ldlt_ineq_cond,
                      KKTSolver::sparse_ldlt_cond,
                      KKTSolver::sparse_multistage,
                      KKTSolver::sparse_multistage_parallel)
);
