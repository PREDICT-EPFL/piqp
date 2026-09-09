#define PIQP_EIGEN_CHECK_MALLOC
#include "piqp/fwd.hpp"
#include "piqp/constrained/solver.hpp"
#include "gtest/gtest.h"

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
    for (const auto& c : model.cone_constraints)
    {
        SparseMat<double, int> F = c.F.sparseView();
        out.cone_constraints.emplace_back(F, c.f, c.type, c.indices);
    }
    return out;
}

void check_kkt(const dense::Model<double>& model, const constrained::Result<double>& result, double tolerance)
{
    Vec<double> residual = model.P.selfadjointView<Eigen::Upper>() * result.x + model.c;
    residual.noalias() += model.A.transpose() * result.y;
    residual.noalias() += model.G.transpose() * (result.z_u - result.z_l);
    residual += result.z_bu - result.z_bl;
    EXPECT_LE((model.A * result.x - model.b).lpNorm<Eigen::Infinity>(), tolerance);
    double complementarity = 0;
    for (isize i = 0; i < model.G.rows(); ++i)
    {
        const double value = model.G.row(i).dot(result.x);
        if (std::isfinite(model.h_l(i)))
        {
            EXPECT_LE(std::abs(value - model.h_l(i) - result.s_l(i)), tolerance);
            EXPECT_GE(result.z_l(i), 0);
            complementarity += result.s_l(i) * result.z_l(i);
        }
        if (std::isfinite(model.h_u(i)))
        {
            EXPECT_LE(std::abs(value - model.h_u(i) + result.s_u(i)), tolerance);
            EXPECT_GE(result.z_u(i), 0);
            complementarity += result.s_u(i) * result.z_u(i);
        }
    }
    for (isize i = 0; i < model.c.size(); ++i)
    {
        if (std::isfinite(model.x_l(i)))
        {
            EXPECT_LE(std::abs(result.x(i) - model.x_l(i) - result.s_bl(i)), tolerance);
            complementarity += result.s_bl(i) * result.z_bl(i);
        }
        if (std::isfinite(model.x_u(i)))
        {
            EXPECT_LE(std::abs(result.x(i) - model.x_u(i) + result.s_bu(i)), tolerance);
            complementarity += result.s_bu(i) * result.z_bu(i);
        }
    }
    for (isize i = 0; i < static_cast<isize>(model.quadratic_constraints.size()); ++i)
    {
        const auto& q = model.quadratic_constraints[static_cast<usize>(i)];
        Vec<double> local(q.q.size());
        for (isize j = 0; j < local.size(); ++j) local(j) = result.x(q.indices.empty() ? j : q.indices[static_cast<usize>(j)]);
        Vec<double> gradient = q.Q.selfadjointView<Eigen::Upper>() * local;
        const double value = 0.5 * local.dot(gradient) + q.q.dot(local) - q.upper;
        gradient += q.q;
        EXPECT_LE(value, tolerance);
        EXPECT_LE(std::abs(value + result.quadratic_slack(i)), tolerance);
        EXPECT_GE(result.quadratic_dual(i), 0);
        for (isize j = 0; j < local.size(); ++j)
            residual(q.indices.empty() ? j : q.indices[static_cast<usize>(j)]) += result.quadratic_dual(i) * gradient(j);
        complementarity += result.quadratic_slack(i) * result.quadratic_dual(i);
    }
    for (isize i = 0; i < static_cast<isize>(model.cone_constraints.size()); ++i)
    {
        const auto& c = model.cone_constraints[static_cast<usize>(i)];
        Vec<double> local(c.F.cols());
        for (isize j = 0; j < local.size(); ++j) local(j) = result.x(c.indices.empty() ? j : c.indices[static_cast<usize>(j)]);
        Vec<double> slack = result.cone_slack.segment(result.cone_offsets[static_cast<usize>(i)], c.f.size());
        Vec<double> dual = result.cone_dual.segment(result.cone_offsets[static_cast<usize>(i)], c.f.size());
        EXPECT_LE((c.F * local + c.f - slack).lpNorm<Eigen::Infinity>(), tolerance);
        Vec<double> contribution = c.F.transpose() * dual;
        for (isize j = 0; j < local.size(); ++j) residual(c.indices.empty() ? j : c.indices[static_cast<usize>(j)]) -= contribution(j);
        complementarity += slack.dot(dual);
        if (c.type == ConeType::second_order)
        {
            EXPECT_LE(slack.tail(slack.size() - 1).norm() - slack(0), tolerance);
            EXPECT_LE(dual.tail(dual.size() - 1).norm() - dual(0), tolerance);
        }
        else
        {
            EXPECT_GE(slack(0), -tolerance); EXPECT_GE(slack(1), -tolerance);
            EXPECT_GE(dual(0), -tolerance); EXPECT_GE(dual(1), -tolerance);
            EXPECT_LE(slack.tail(slack.size() - 2).squaredNorm() - 2 * slack(0) * slack(1), tolerance);
            EXPECT_LE(dual.tail(dual.size() - 2).squaredNorm() - 2 * dual(0) * dual(1), tolerance);
        }
    }
    EXPECT_LE(residual.lpNorm<Eigen::Infinity>(), tolerance);
    EXPECT_NEAR(result.dual_residual, residual.lpNorm<Eigen::Infinity>(), tolerance);
    EXPECT_NEAR(result.complementarity, complementarity, tolerance);
    EXPECT_LE(complementarity, tolerance);
}

template<typename Solver, typename Model>
void check_solve(Solver& solver, const Model& model, const dense::Model<double>& original,
                 const Vec<double>& expected, double tolerance)
{
    SCOPED_TRACE(static_cast<int>(solver.settings().kkt_solver));
    SCOPED_TRACE(model.cone_constraints.empty() ? "QCQP" : "SOC");
    solver.settings().eps_abs = solver.settings().eps_rel = 1e-9;
    solver.settings().eps_duality_gap_abs = solver.settings().eps_duality_gap_rel = 1e-9;
    solver.setup(model);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    const Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();
    ASSERT_EQ(status, PIQP_SOLVED) << "iterations " << solver.result().iterations
        << " primal " << solver.result().primal_residual << " dual " << solver.result().dual_residual
        << " complementarity " << solver.result().complementarity;
    EXPECT_LE((solver.result().x - expected).template lpNorm<Eigen::Infinity>(), tolerance);
    check_kkt(original, solver.result(), tolerance);
}

void check_backends(const dense::Model<double>& model, const Vec<double>& expected, double tolerance = 1e-6)
{
    ConstrainedDenseSolver<double> dense_solver;
    check_solve(dense_solver, model, model, expected, tolerance);
    auto sparse = sparse_model(model);
    ConstrainedSparseSolver<double> sparse_solver;
    check_solve(sparse_solver, sparse, model, expected, tolerance);
#ifdef PIQP_HAS_BLASFEO
    ConstrainedSparseSolver<double> stage_solver;
    stage_solver.settings().kkt_solver = KKTSolver::sparse_multistage;
    check_solve(stage_solver, sparse, model, expected, tolerance);
#endif
}

}

TEST(ConstrainedSolverMathTest, AnalyticLorentzProjection)
{
    Vec<double> a(3); a << -0.5, 2, -1;
    Mat<double> P = Mat<double>::Identity(3, 3);
    dense::Model<double> model(P, Vec<double>(-a));
    model.cone_constraints.emplace_back(P, Vec<double>::Zero(3));
    const double t = (a(0) + a.tail(2).norm()) / 2;
    Vec<double> expected(3); expected << t, t * a(1) / a.tail(2).norm(), t * a(2) / a.tail(2).norm();
    check_backends(model, expected);
}

TEST(ConstrainedSolverMathTest, NativeBallAndMultiplier)
{
    Vec<double> a(2); a << 3, 4;
    Mat<double> P = Mat<double>::Identity(2, 2);
    dense::Model<double> model(P, Vec<double>(-a));
    model.quadratic_constraints.emplace_back(Mat<double>(2 * P), Vec<double>::Zero(2), 1);
    check_backends(model, Vec<double>(a / 5));
    ConstrainedDenseSolver<double> solver;
    solver.setup(model);
    ASSERT_EQ(solver.solve(), PIQP_SOLVED);
    EXPECT_NEAR(solver.result().quadratic_dual(0), 2, 1e-5);
}

TEST(ConstrainedSolverMathTest, MixedLocalConstraints)
{
    Mat<double> P = Mat<double>::Identity(3, 3), A = Mat<double>::Zero(1, 3);
    A(0, 2) = 1;
    Vec<double> a(3); a << 2, 2, 2;
    Vec<double> b(1); b << 1;
    dense::Model<double> model(P, Vec<double>(-a), A, b);
    Mat<double> Q(1, 1); Q << 2;
    model.quadratic_constraints.emplace_back(Q, Vec<double>::Zero(1), 0.25, std::vector<isize>{0});
    Mat<double> F = Mat<double>::Zero(3, 2); F.bottomRows(2).setIdentity();
    Vec<double> f(3); f << 1, 0, 0;
    model.cone_constraints.emplace_back(F, f, ConeType::second_order, std::vector<isize>{0, 1});
    model.x_l(1) = 0;
    model.G = Mat<double>::Zero(1, 3); model.G(0, 0) = model.G(0, 1) = 1;
    model.h_l = Vec<double>::Zero(1); model.h_u = Vec<double>::Constant(1, 1.5);
    Vec<double> expected(3); expected << 0.5, std::sqrt(0.75), 1;
    check_backends(model, expected);
}

TEST(ConstrainedSolverMathTest, RotatedDualTransformation)
{
    Mat<double> P(1, 1); P << 1;
    dense::Model<double> model(P, Vec<double>::Zero(1));
    Mat<double> F = Mat<double>::Zero(3, 1); F(0, 0) = 1;
    Vec<double> f(3); f << 0, 1, 2;
    model.cone_constraints.emplace_back(F, f, ConeType::rotated_second_order);
    check_backends(model, Vec<double>::Constant(1, 2));
    ConstrainedDenseSolver<double> solver;
    solver.settings().eps_abs = solver.settings().eps_rel = 1e-10;
    solver.settings().eps_duality_gap_abs = solver.settings().eps_duality_gap_rel = 1e-10;
    solver.setup(model);
    ASSERT_EQ(solver.solve(), PIQP_SOLVED);
    Vec<double> expected(3); expected << 2, 4, -4;
    EXPECT_LT((solver.result().cone_dual - expected).norm(), 1e-5);
}

TEST(ConstrainedSolverMathTest, ApexProjection)
{
    Mat<double> P = Mat<double>::Identity(3, 3);
    Vec<double> c(3); c << 2, -0.5, 0.25;
    dense::Model<double> model(P, c);
    model.cone_constraints.emplace_back(P, Vec<double>::Zero(3));
    check_backends(model, Vec<double>::Zero(3));
}

TEST(ConstrainedSolverMathTest, CoordinateAndConstraintScales)
{
    for (double scale : {1e-4, 1.0, 1e4})
    {
        SCOPED_TRACE(scale);
        Mat<double> E = Mat<double>::Zero(2, 2); E(0, 0) = 0.01; E(1, 1) = 100;
        Vec<double> a(2); a << 3, 4;
        Mat<double> P = E * E;
        dense::Model<double> model(P, Vec<double>(-E * a));
        model.quadratic_constraints.emplace_back(Mat<double>(2 * scale * P), Vec<double>::Zero(2), scale);
        Vec<double> expected(2); expected << 60, 0.008;
        check_backends(model, expected, 2e-5);
        model.quadratic_constraints.clear();
        Mat<double> F = Mat<double>::Zero(3, 2); F.bottomRows(2) = scale * E;
        Vec<double> f = Vec<double>::Zero(3); f(0) = scale;
        model.cone_constraints.emplace_back(F, f);
        check_backends(model, expected, 2e-5);
    }
}

TEST(ConstrainedSolverMathTest, PrescribedKktPointWithSingularLocalCurvature)
{
    Mat<double> P = Mat<double>::Identity(3, 3), Q(2, 2);
    Q << 2, -2, -2, 2;
    Vec<double> q(2); q << 0.2, -0.4;
    Vec<double> expected(3); expected << 0.5, -0.25, 0.75;
    Vec<double> local(2); local << expected(2), expected(0);
    Vec<double> gradient = Q * local + q;
    Vec<double> c = -expected;
    c(2) -= 0.7 * gradient(0); c(0) -= 0.7 * gradient(1);
    dense::Model<double> model(P, c);
    model.quadratic_constraints.emplace_back(Q, q, 0.5 * local.dot(Q * local) + q.dot(local), std::vector<isize>{2, 0});
    check_backends(model, expected);
}

TEST(ConstrainedSolverMathTest, InfeasibleQuadraticIsNotSolved)
{
    Mat<double> P = Mat<double>::Identity(2, 2);
    dense::Model<double> model(P, Vec<double>::Zero(2));
    model.quadratic_constraints.emplace_back(P, Vec<double>::Zero(2), -1);
    ConstrainedDenseSolver<double> solver;
    solver.setup(model);
    EXPECT_NE(solver.solve(), PIQP_SOLVED);
}

TEST(ConstrainedSolverMathTest, SingularObjectiveRotatedCone)
{
    Mat<double> P = Mat<double>::Zero(1, 1), F = Mat<double>::Zero(3, 1);
    F(0, 0) = 1;
    Vec<double> f(3); f << 0, 1, 2;
    dense::Model<double> model(P, Vec<double>::Ones(1));
    model.cone_constraints.emplace_back(F, f, ConeType::rotated_second_order);
    check_backends(model, Vec<double>::Constant(1, 2));
}

TEST(ConstrainedSolverMathTest, NumericalUpdatesAndOwnedModel)
{
    Mat<double> P = Mat<double>::Identity(2, 2);
    Vec<double> c(2); c << -3, -4;
    dense::Model<double> model(P, c);
    model.quadratic_constraints.emplace_back(Mat<double>(2 * P), Vec<double>::Zero(2), 1);
    ConstrainedDenseSolver<double> solver;
    solver.setup(model);
    model.quadratic_constraints[0].upper = 100;
    ASSERT_EQ(solver.solve(), PIQP_SOLVED);
    EXPECT_NEAR(solver.result().x.norm(), 1, 1e-6);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    solver.update_quadratic(0, 4);
    const Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();
    ASSERT_EQ(status, PIQP_SOLVED);
    EXPECT_NEAR(solver.result().x.norm(), 2, 1e-6);
    model.quadratic_constraints[0].upper = 4;
    ConstrainedDenseSolver<double> fresh;
    fresh.setup(model);
    ASSERT_EQ(fresh.solve(), PIQP_SOLVED);
    EXPECT_LT((fresh.result().x - solver.result().x).norm(), 1e-8);

    Mat<double> objective = Mat<double>::Identity(1, 1), F = Mat<double>::Zero(3, 1);
    F(0, 0) = 1;
    Vec<double> f(3); f << 0, 1, 2;
    dense::Model<double> rotated(objective, Vec<double>::Zero(1));
    rotated.cone_constraints.emplace_back(F, f, ConeType::rotated_second_order);
    solver.setup(rotated);
    f(2) = 3;
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    solver.update_cone(0, f);
    const Status updated_status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();
    ASSERT_EQ(updated_status, PIQP_SOLVED);
    EXPECT_NEAR(solver.result().x(0), 4.5, 1e-6);
    rotated.cone_constraints[0].f = f;
    fresh.setup(rotated);
    ASSERT_EQ(fresh.solve(), PIQP_SOLVED);
    EXPECT_NEAR(solver.result().x(0), fresh.result().x(0), 1e-8);
}

namespace
{

dense::Model<double> update_model(bool changed)
{
    Vec<double> x(3); x << (changed ? 0.5 : 0.6), (changed ? 0.75 : 0.8), (changed ? 1.2 : 1);
    Mat<double> P = Mat<double>::Identity(3, 3), A = Mat<double>::Zero(1, 3), G = Mat<double>::Zero(1, 3);
    A(0, 2) = 1; G(0, 0) = 1;
    if (changed) { P(0, 2) = P(2, 0) = 0.1; P(1, 1) = 1.2; A(0, 0) = 0.2; G(0, 1) = 0.2; }
    Mat<double> Q = 2 * Mat<double>::Identity(2, 2);
    Vec<double> q = Vec<double>::Zero(2);
    if (changed) { Q(0, 1) = Q(1, 0) = 0.25; Q(1, 1) = 3; q << 0.1, -0.2; }
    Mat<double> F = Mat<double>::Zero(3, 2); F.bottomRows(2).setIdentity();
    Vec<double> f = Vec<double>::Zero(3);
    if (changed) { F(0, 0) = 0.02; F(0, 1) = -0.01; F(1, 1) = 0.1; F(2, 1) = 1.2; f(1) = 0.05; f(2) = -0.03; }
    Vec<double> slack = F * x.head(2) + f;
    f(0) = slack.tail(2).norm() - slack(0);
    slack(0) += f(0);
    Vec<double> dual = -0.6 * slack / slack(0); dual(0) = 0.6;
    Vec<double> c = -P * x - A.transpose() * Vec<double>::Constant(1, 0.3);
    c.head(2) -= 0.4 * (Q * x.head(2) + q);
    c.head(2) += F.transpose() * dual;
    Vec<double> b = A * x, lower = Vec<double>::Constant(1, changed ? -2 : -1), upper = Vec<double>::Constant(1, changed ? 4 : 3);
    dense::Model<double> model(P, c, A, b, G, lower, upper);
    model.x_l(1) = changed ? 0.1 : 0;
    model.x_u(2) = changed ? 3 : 2;
    model.quadratic_constraints.emplace_back(Q, q, 0.5 * x.head(2).dot(Q * x.head(2)) + q.dot(x.head(2)), std::vector<isize>{0, 1});
    model.cone_constraints.emplace_back(F, f, ConeType::second_order, std::vector<isize>{0, 1});
    return model;
}

SparseMat<double, int> stored_zeros(const Mat<double>& matrix)
{
    std::vector<Eigen::Triplet<double, int>> entries;
    for (int j = 0; j < matrix.cols(); ++j)
        for (int i = 0; i < matrix.rows(); ++i) entries.emplace_back(i, j, matrix(i, j));
    SparseMat<double, int> sparse(matrix.rows(), matrix.cols());
    sparse.setFromTriplets(entries.begin(), entries.end());
    return sparse;
}

sparse::Model<double, int> full_sparse_model(const dense::Model<double>& model)
{
    auto out = sparse_model(model);
    out.P = stored_zeros(model.P); out.A = stored_zeros(model.A); out.G = stored_zeros(model.G);
    out.quadratic_constraints[0].Q = stored_zeros(model.quadratic_constraints[0].Q);
    out.cone_constraints[0].F = stored_zeros(model.cone_constraints[0].F);
    return out;
}

template<typename Solver, typename Model>
void check_full_update(Solver& solver, const Model& initial, const Model& updated)
{
    SCOPED_TRACE(static_cast<int>(solver.settings().kkt_solver));
    solver.settings().eps_abs = solver.settings().eps_rel = 1e-9;
    solver.settings().eps_duality_gap_abs = solver.settings().eps_duality_gap_rel = 1e-9;
    solver.setup(initial);
    ASSERT_EQ(solver.solve(), PIQP_SOLVED);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    solver.update(updated);
    const Status status = solver.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();
    ASSERT_EQ(status, PIQP_SOLVED);
    Vec<double> expected(3); expected << 0.5, 0.75, 1.2;
    EXPECT_LT((solver.result().x - expected).norm(), 2e-5);
    check_kkt(update_model(true), solver.result(), 1e-6);
    Solver fresh;
    fresh.settings() = solver.settings(); fresh.setup(updated);
    ASSERT_EQ(fresh.solve(), PIQP_SOLVED);
    EXPECT_LT((solver.result().x - fresh.result().x).norm(), 2e-5);
    Model invalid = updated;
    invalid.cone_constraints[0].indices = {1, 0};
    EXPECT_THROW(solver.update(invalid), std::invalid_argument);
    ASSERT_EQ(solver.solve(), PIQP_SOLVED);
    EXPECT_LT((solver.result().x - expected).norm(), 2e-5);
    invalid = updated;
    invalid.x_l(0) = 0;
    EXPECT_THROW(solver.update(invalid), std::invalid_argument);
    ASSERT_EQ(solver.solve(), PIQP_SOLVED);
    EXPECT_LT((solver.result().x - expected).norm(), 2e-5);
}

}

TEST(ConstrainedSolverMathTest, FullDenseNumericUpdate)
{
    ConstrainedDenseSolver<double> solver;
    check_full_update(solver, update_model(false), update_model(true));
}

TEST(ConstrainedSolverMathTest, FullSparseNumericUpdateWithStoredZeros)
{
    ConstrainedSparseSolver<double> solver;
    auto initial = full_sparse_model(update_model(false)), updated = full_sparse_model(update_model(true));
    check_full_update(solver, initial, updated);
    updated.P.prune(0.0);
    EXPECT_THROW(solver.update(updated), std::invalid_argument);
    ASSERT_EQ(solver.solve(), PIQP_SOLVED);
}

#ifdef PIQP_HAS_BLASFEO
TEST(ConstrainedSolverMathTest, FullMultistageNumericUpdateWithStoredZeros)
{
    ConstrainedSparseSolver<double> solver;
    solver.settings().kkt_solver = KKTSolver::sparse_multistage;
    check_full_update(solver, full_sparse_model(update_model(false)), full_sparse_model(update_model(true)));
}
#endif

TEST(ConstrainedSolverMathTest, WrongInfinityBoundsAndConstraintConversion)
{
    const double infinity = std::numeric_limits<double>::infinity();
    ConstrainedDenseSolver<double> solver;
    for (double value : {-infinity, infinity})
    {
        auto invalid = update_model(false);
        invalid.x_l(0) = invalid.x_u(0) = value;
        EXPECT_THROW(solver.setup(invalid), std::invalid_argument);
        invalid = update_model(false);
        invalid.h_l(0) = invalid.h_u(0) = value;
        EXPECT_THROW(solver.setup(invalid), std::invalid_argument);
    }
    auto model = update_model(true);
    auto sparse = full_sparse_model(model);
    auto dense = sparse.dense_model();
    ASSERT_EQ(dense.quadratic_constraints.size(), 1u);
    ASSERT_EQ(dense.cone_constraints.size(), 1u);
    EXPECT_EQ(dense.quadratic_constraints[0].indices, model.quadratic_constraints[0].indices);
    EXPECT_EQ(dense.cone_constraints[0].indices, model.cone_constraints[0].indices);
    EXPECT_EQ(dense.cone_constraints[0].type, model.cone_constraints[0].type);
    EXPECT_EQ((dense.quadratic_constraints[0].Q - model.quadratic_constraints[0].Q).norm(), 0);
    EXPECT_EQ((dense.cone_constraints[0].F - model.cone_constraints[0].F).norm(), 0);
    Vec<double> expected(3); expected << 0.5, 0.75, 1.2;
    check_solve(solver, dense, model, expected, 2e-5);
}
