#define PIQP_EIGEN_CHECK_MALLOC
#include "piqp/fwd.hpp"
#include "piqp/constrained/solver.hpp"
#include "gtest/gtest.h"
#include <cstdlib>
#include <new>

namespace
{
thread_local bool count_allocations = false;
thread_local std::size_t allocation_count = 0;

struct AllocationScope
{
    AllocationScope()
    {
        allocation_count = 0;
        count_allocations = true;
        PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    }
    ~AllocationScope()
    {
        PIQP_EIGEN_MALLOC_ALLOWED();
        count_allocations = false;
    }
};
}

void* operator new(std::size_t size)
{
    if (count_allocations) ++allocation_count;
    if (void* pointer = std::malloc(size ? size : 1)) return pointer;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

using namespace piqp;

namespace
{

sparse::Model<double, int> sparse_model(const dense::Model<double>& model)
{
    SparseMat<double, int> P = model.P.sparseView(), A = model.A.sparseView(), G = model.G.sparseView();
    sparse::Model<double, int> out(P, model.c, A, model.b, G, model.h_l, model.h_u, model.x_l, model.x_u);
    for (const auto& q : model.quadratic_constraints)
    {
        SparseMat<double, int> Q = q.Q.sparseView();
        out.quadratic_constraints.emplace_back(Q, q.q, q.upper, q.indices);
    }
    for (const auto& cone : model.cone_constraints)
    {
        SparseMat<double, int> F = cone.F.sparseView();
        out.cone_constraints.emplace_back(F, cone.f, cone.type, cone.indices);
    }
    return out;
}

void original_kkt(const dense::Model<double>& model, const constrained::Result<double>& result, double tolerance)
{
    ASSERT_TRUE(result.x.allFinite());
    ASSERT_TRUE(result.y.allFinite());
    Vec<double> residual = model.P * result.x + model.c + model.A.transpose() * result.y;
    EXPECT_LE((model.A * result.x - model.b).lpNorm<Eigen::Infinity>(), tolerance);
    double complementarity = 0;
    for (std::size_t i = 0; i < model.quadratic_constraints.size(); ++i)
    {
        const auto& q = model.quadratic_constraints[i];
        const double value = 0.5 * result.x.dot(q.Q * result.x) + q.q.dot(result.x) - q.upper;
        EXPECT_LE(value, tolerance);
        EXPECT_NEAR(value + result.quadratic_slack(static_cast<isize>(i)), 0, tolerance);
        EXPECT_GE(result.quadratic_slack(static_cast<isize>(i)), 0);
        EXPECT_GE(result.quadratic_dual(static_cast<isize>(i)), 0);
        residual += result.quadratic_dual(static_cast<isize>(i)) * (q.Q * result.x + q.q);
        complementarity += result.quadratic_slack(static_cast<isize>(i)) * result.quadratic_dual(static_cast<isize>(i));
    }
    for (std::size_t i = 0; i < model.cone_constraints.size(); ++i)
    {
        const auto& cone = model.cone_constraints[i];
        const Vec<double> value = cone.F * result.x + cone.f;
        const Vec<double> slack = result.cone_slack.segment(result.cone_offsets[i], cone.f.size());
        const Vec<double> dual = result.cone_dual.segment(result.cone_offsets[i], cone.f.size());
        EXPECT_LE(value.tail(value.size() - 1).stableNorm() - value(0), tolerance);
        EXPECT_LE((value - slack).lpNorm<Eigen::Infinity>(), tolerance);
        EXPECT_LE(slack.tail(slack.size() - 1).stableNorm() - slack(0), tolerance);
        EXPECT_LE(dual.tail(dual.size() - 1).stableNorm() - dual(0), tolerance);
        residual.noalias() -= cone.F.transpose() * dual;
        complementarity += slack.dot(dual);
    }
    EXPECT_LE(residual.lpNorm<Eigen::Infinity>(), tolerance);
    EXPECT_NEAR(result.dual_residual, residual.lpNorm<Eigen::Infinity>(), tolerance);
    EXPECT_GE(complementarity, -tolerance);
    EXPECT_LE(complementarity, tolerance);
    EXPECT_NEAR(result.complementarity, complementarity, tolerance);
}

template<typename Function>
void backends(const dense::Model<double>& model, Function function)
{
    ConstrainedDenseSolver<double> dense;
    {
        SCOPED_TRACE("dense");
        function(dense, model);
    }
    auto sparse = sparse_model(model);
    ConstrainedSparseSolver<double> general;
    {
        SCOPED_TRACE("sparse");
        function(general, sparse);
    }
#ifdef PIQP_HAS_BLASFEO
    ConstrainedSparseSolver<double> multistage;
    multistage.settings().kkt_solver = KKTSolver::sparse_multistage;
    {
        SCOPED_TRACE("multistage");
        function(multistage, sparse);
    }
#endif
}

void solved(const dense::Model<double>& model, const Vec<double>* expected = nullptr, double tolerance = 2e-7)
{
    backends(model, [&](auto& solver, const auto& source) {
        solver.settings().eps_abs = solver.settings().eps_duality_gap_abs = 1e-10;
        solver.settings().eps_rel = solver.settings().eps_duality_gap_rel = 0;
        solver.setup(source);
        ASSERT_EQ(solver.solve(), PIQP_SOLVED) << "iteration " << solver.result().iterations
            << " primal " << solver.result().primal_residual << " dual " << solver.result().dual_residual
            << " complementarity " << solver.result().complementarity;
        original_kkt(model, solver.result(), tolerance);
        if (expected) EXPECT_LE((solver.result().x - *expected).norm(), tolerance);
    });
}

void not_solved(const dense::Model<double>& model)
{
    backends(model, [&](auto& solver, const auto& source) {
        solver.settings().max_iter = 60;
        solver.setup(source);
        const Status status = solver.solve();
        EXPECT_TRUE(status == PIQP_NUMERICS || status == PIQP_MAX_ITER_REACHED) << status_to_string(status);
        EXPECT_LE(solver.result().iterations, 60);
        EXPECT_TRUE(solver.result().x.allFinite());
    });
}

dense::Model<double> sphere_model(const Mat<double>& P, const Vec<double>& c, double radius)
{
    const isize n = c.size();
    dense::Model<double> model(P, c);
    model.quadratic_constraints.emplace_back(Mat<double>::Identity(n, n), Vec<double>::Zero(n), 0.5 * radius * radius);
    Mat<double> F = Mat<double>::Zero(n + 1, n); F.bottomRows(n).setIdentity();
    Vec<double> f = Vec<double>::Zero(n + 1); f(0) = radius;
    model.cone_constraints.emplace_back(F, f);
    return model;
}

}

TEST(ConstrainedSolverEdgeTest, RedundantAndNearlyDependentEqualities)
{
    for (double perturbation : {0.0, 1e-6})
    {
        SCOPED_TRACE(perturbation);
        Vec<double> expected(3); expected << 0.2, -0.1, 0.3;
        Mat<double> P = Mat<double>::Identity(3, 3), A(3, 3);
        A << 1, 1, 0, 2, 2, 0, 1, 1 + perturbation, perturbation;
        Vec<double> multiplier(3); multiplier << 0.1, 0.2, -0.1;
        Vec<double> c = -1.4 * expected - 0.6 * expected / expected.norm() - A.transpose() * multiplier;
        auto model = sphere_model(P, c, expected.norm());
        model.A = A; model.b = A * expected;
        solved(model, &expected);
    }
}

TEST(ConstrainedSolverEdgeTest, ZeroAndSingularObjective)
{
    Mat<double> P = Mat<double>::Zero(2, 2);
    auto feasibility = sphere_model(P, Vec<double>::Zero(2), 1);
    feasibility.A = Mat<double>::Zero(1, 2); feasibility.A(0, 0) = 1;
    feasibility.b = Vec<double>::Constant(1, 0.25);
    solved(feasibility);
    P(0, 0) = 1;
    Vec<double> c(2); c << -2, 0;
    auto singular = sphere_model(P, c, 1);
    Vec<double> expected(2); expected << 1, 0;
    solved(singular, &expected);
}

TEST(ConstrainedSolverEdgeTest, SmallFeasibleMargin)
{
    Mat<double> P = Mat<double>::Identity(1, 1);
    auto model = sphere_model(P, Vec<double>::Constant(1, -1), 1e-4);
    Vec<double> expected = Vec<double>::Constant(1, 1e-4);
    solved(model, &expected, 2e-8);
}

TEST(ConstrainedSolverEdgeTest, OppositeNearlyBoundaryCones)
{
    Mat<double> P = Mat<double>::Identity(1, 1), F(2, 1);
    dense::Model<double> model(P, Vec<double>::Constant(1, -1));
    F << 1, 0;
    Vec<double> f(2); f << 1, 1;
    model.cone_constraints.emplace_back(F, f);
    F(0, 0) = -1; f(0) += 1e-6;
    model.cone_constraints.emplace_back(F, f);
    Vec<double> expected = Vec<double>::Constant(1, 1e-6);
    solved(model, &expected, 2e-8);
}

TEST(ConstrainedSolverEdgeTest, ConstantInteriorAndApexCones)
{
    Mat<double> P = Mat<double>::Identity(2, 2), F = Mat<double>::Zero(3, 2);
    Vec<double> c(2); c << -0.2, 0.3;
    const Vec<double> expected = -c;
    for (bool apex : {false, true})
    {
        SCOPED_TRACE(apex);
        Vec<double> f = Vec<double>::Zero(3);
        if (!apex) f << 2, 1, 0;
        dense::Model<double> model(P, c);
        model.cone_constraints.emplace_back(F, f);
        solved(model, &expected);
    }
}

TEST(ConstrainedSolverEdgeTest, ConstantInfeasibleCone)
{
    Mat<double> P = Mat<double>::Identity(2, 2), F = Mat<double>::Zero(3, 2);
    dense::Model<double> model(P, Vec<double>::Zero(2));
    Vec<double> f(3); f << 0, 1, 0;
    model.cone_constraints.emplace_back(F, f);
    not_solved(model);
}

TEST(ConstrainedSolverEdgeTest, WeaklyInfeasibleAffineCone)
{
    Mat<double> P = Mat<double>::Zero(1, 1), F(3, 1);
    F << 1, 1, 0;
    Vec<double> f(3); f << 0, 0, 1;
    dense::Model<double> model(P, Vec<double>::Ones(1));
    model.cone_constraints.emplace_back(F, f);
    not_solved(model);
}

TEST(ConstrainedSolverEdgeTest, SettingsAfterSetup)
{
    Mat<double> P(3, 3); P << 1.2, 0.13, 0.17, 0.13, 1.7, 0.29, 0.17, 0.29, 1.3;
    Vec<double> c(3); c << -1.13, 0.37, -2.19;
    auto model = sphere_model(P, c, 1);
    backends(model, [&](auto& solver, const auto& source) {
        const KKTSolver backend = solver.settings().kkt_solver;
        solver.setup(source);
        ASSERT_EQ(solver.solve(), PIQP_SOLVED);
        solver.settings().kkt_solver = backend == KKTSolver::dense_cholesky ? KKTSolver::sparse_ldlt : KKTSolver::dense_cholesky;
        EXPECT_EQ(solver.solve(), PIQP_INVALID_SETTINGS);
        solver.settings().kkt_solver = backend;
        solver.settings().iterative_refinement_eps_abs = 1e-30;
        solver.settings().iterative_refinement_eps_rel = 0;
        solver.settings().iterative_refinement_max_iter = 0;
        solver.settings().max_factor_retires = 1;
        EXPECT_EQ(solver.solve(), PIQP_NUMERICS);
        solver.settings().iterative_refinement_eps_abs = solver.settings().iterative_refinement_eps_rel = 1e-12;
        solver.settings().iterative_refinement_max_iter = solver.settings().max_factor_retires = 10;
        ASSERT_EQ(solver.solve(), PIQP_SOLVED);
        original_kkt(model, solver.result(), 2e-7);
    });
}

TEST(ConstrainedSolverEdgeTest, FailedSetupLeavesSafeUnsolvedState)
{
    auto model = sphere_model(Mat<double>::Identity(2, 2), Vec<double>::Constant(2, -2), 1);
    ConstrainedDenseSolver<double> solver;
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        SCOPED_TRACE(attempt);
        solver.settings().kkt_solver = KKTSolver::sparse_ldlt;
        EXPECT_THROW(solver.setup(model), std::invalid_argument);
        solver.settings().kkt_solver = KKTSolver::dense_cholesky;
        EXPECT_EQ(solver.solve(), PIQP_UNSOLVED);
        EXPECT_EQ(solver.result().status, PIQP_UNSOLVED);
        EXPECT_THROW(solver.update(model), std::invalid_argument);
        EXPECT_THROW(solver.update_quadratic(0, 1), std::invalid_argument);
        EXPECT_THROW(solver.update_cone(0, model.cone_constraints[0].f), std::invalid_argument);
        solver.setup(model);
        ASSERT_EQ(solver.solve(), PIQP_SOLVED);
        original_kkt(model, solver.result(), 2e-7);
    }
}

TEST(ConstrainedSolverEdgeTest, RepeatedUpdateAndSolveAllocateNeitherEigenNorCppStorage)
{
    auto model = sphere_model(Mat<double>::Identity(2, 2), Vec<double>::Constant(2, -2), 1);
    backends(model, [&](auto& solver, const auto& source) {
        solver.setup(source);
        Status statuses[3];
        {
            AllocationScope guard;
            for (int i = 0; i < 3; ++i)
            {
                solver.update(source);
                statuses[i] = solver.solve();
            }
        }
        EXPECT_EQ(allocation_count, 0u);
        for (Status status : statuses) EXPECT_EQ(status, PIQP_SOLVED);
        original_kkt(model, solver.result(), 2e-7);
    });
}
