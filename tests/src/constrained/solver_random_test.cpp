#define PIQP_EIGEN_CHECK_MALLOC
#include "piqp/fwd.hpp"
#include "piqp/constrained/solver.hpp"
#include "gtest/gtest.h"
#include <random>

using namespace piqp;

namespace
{

Vec<double> random_vector(std::mt19937& generator, isize size)
{
    Vec<double> result(size);
    for (isize i = 0; i < size; ++i) result(i) = 2. * double(generator()) / double(generator.max()) - 1.;
    return result;
}

Vec<double> local_values(const Vec<double>& x, const std::vector<isize>& indices)
{
    if (indices.empty()) return x;
    Vec<double> local(isize(indices.size()));
    for (isize i = 0; i < local.size(); ++i) local(i) = x(indices[usize(i)]);
    return local;
}

void accumulate(Vec<double>& out, const Vec<double>& local, const std::vector<isize>& indices, double scale)
{
    if (indices.empty()) out += scale * local;
    else for (isize i = 0; i < local.size(); ++i) out(indices[usize(i)]) += scale * local(i);
}

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

void check_original_kkt(const dense::Model<double>& model, const constrained::Result<double>& result, const Vec<double>& optimum)
{
    EXPECT_LT((result.x - optimum).norm(), 3e-5);
    const double optimum_value = .5 * optimum.dot(model.P * optimum) + model.c.dot(optimum);
    const double objective = .5 * result.x.dot(model.P * result.x) + model.c.dot(result.x);
    EXPECT_NEAR(result.objective, objective, 1e-12 * (1 + std::abs(objective)));
    EXPECT_NEAR(objective, optimum_value, 1e-6 * (1 + std::abs(optimum_value)));
    EXPECT_LT((model.A * result.x - model.b).norm(), 1e-7);
    Vec<double> stationarity = model.P * result.x + model.c + model.A.transpose() * result.y;
    double complementarity = 0.;
    for (isize i = 0; i < static_cast<isize>(model.quadratic_constraints.size()); ++i)
    {
        const auto& q = model.quadratic_constraints[usize(i)];
        Vec<double> x = local_values(result.x, q.indices), gradient = q.Q * x + q.q;
        const double value = .5 * x.dot(q.Q * x) + q.q.dot(x) - q.upper;
        EXPECT_LE(value, 1e-7);
        EXPECT_NEAR(value + result.quadratic_slack(i), 0., 1e-7);
        EXPECT_GE(result.quadratic_dual(i), 0.);
        accumulate(stationarity, gradient, q.indices, result.quadratic_dual(i));
        complementarity += result.quadratic_slack(i) * result.quadratic_dual(i);
    }
    for (isize i = 0; i < static_cast<isize>(model.cone_constraints.size()); ++i)
    {
        const auto& cone = model.cone_constraints[usize(i)];
        Vec<double> x = local_values(result.x, cone.indices), value = cone.F * x + cone.f;
        Vec<double> dual = result.cone_dual.segment(result.cone_offsets[usize(i)], value.size());
        Vec<double> slack = result.cone_slack.segment(result.cone_offsets[usize(i)], value.size());
        EXPECT_LT((value - slack).norm(), 1e-7);
        EXPECT_GE(value(0) - value.tail(value.size() - 1).norm(), -1e-7);
        EXPECT_GE(dual(0) - dual.tail(dual.size() - 1).norm(), -1e-7);
        accumulate(stationarity, Vec<double>(cone.F.transpose() * dual), cone.indices, -1.);
        complementarity += slack.dot(dual);
    }
    EXPECT_LT(stationarity.lpNorm<Eigen::Infinity>(), 5e-8 * (1 + model.c.lpNorm<Eigen::Infinity>()));
    EXPECT_GE(complementarity, -1e-12);
    EXPECT_LT(complementarity, 5e-8 * (1 + std::abs(objective)));
}

void check_backends(const dense::Model<double>& model, const Vec<double>& optimum)
{
    ConstrainedDenseSolver<double> dense;
    dense.setup(model);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    const auto dense_status = dense.solve();
    PIQP_EIGEN_MALLOC_ALLOWED();
    ASSERT_EQ(dense_status, PIQP_SOLVED) << dense.result().iterations;
    check_original_kkt(model, dense.result(), optimum);
    auto sparse = sparse_model(model);
    for (auto backend : {KKTSolver::sparse_ldlt, KKTSolver::sparse_ldlt_eq_cond, KKTSolver::sparse_ldlt_ineq_cond,
                         KKTSolver::sparse_ldlt_cond, KKTSolver::sparse_multistage})
    {
#ifndef PIQP_HAS_BLASFEO
        if (backend == KKTSolver::sparse_multistage) continue;
#endif
        SCOPED_TRACE(kkt_solver_to_string(backend));
        ConstrainedSparseSolver<double> solver;
        solver.settings().kkt_solver = backend;
        solver.setup(sparse);
        PIQP_EIGEN_MALLOC_NOT_ALLOWED();
        const auto status = solver.solve();
        PIQP_EIGEN_MALLOC_ALLOWED();
        ASSERT_EQ(status, PIQP_SOLVED) << solver.result().iterations;
        check_original_kkt(model, solver.result(), optimum);
    }
}

dense::Model<double> chain_model(int stages, Vec<double>& optimum, bool nonlocal)
{
    const isize n = 2 * stages + 1;
    optimum = Vec<double>::LinSpaced(n, -.2, .2);
    Mat<double> P = Mat<double>::Identity(n, n), A = Mat<double>::Zero(stages - 1, n);
    for (int k = 0; k < stages; ++k)
    {
        P(2 * k, n - 1) = P(n - 1, 2 * k) = .005;
        if (k + 1 < stages)
        {
            A(k, 2 * k) = -.8; A(k, 2 * k + 1) = -.2;
            A(k, 2 * k + 2) = 1.; A(k, n - 1) = -.02;
        }
    }
    Vec<double> b = A * optimum, c = -P * optimum;
    dense::Model<double> model(P, c, A, b);
    Vec<double> gradient(3); gradient << 0, 1, 0;
    for (int k = 0; k < stages; ++k)
    {
        std::vector<isize> indices = {2 * k, 2 * k + 1, n - 1};
        Vec<double> x = local_values(optimum, indices);
        Mat<double> Q = .1 * Mat<double>::Identity(3, 3);
        Vec<double> q = gradient - Q * x;
        model.quadratic_constraints.emplace_back(Q, q, .5 * x.dot(Q * x) + q.dot(x), indices);
        accumulate(c, gradient, indices, -.1);
        Mat<double> F(3, 3); F << 0, 0, 0, 0, 1, 0, .2, .1, .03;
        Vec<double> f(3); f << 1, 1, 0; f -= F * x;
        model.cone_constraints.emplace_back(F, f, ConeType::second_order, indices);
        Vec<double> dual(3); dual << .2, -.2, 0;
        accumulate(c, Vec<double>(F.transpose() * dual), indices, 1.);
    }
    if (nonlocal)
    {
        Mat<double> Q = .01 * Mat<double>::Identity(n, n);
        Vec<double> gradient = Vec<double>::Zero(n);
        for (int k = 0; k < stages; ++k) gradient(2 * k + 1) = 1.;
        Vec<double> q = gradient - Q * optimum;
        model.quadratic_constraints.emplace_back(Q, q, .5 * optimum.dot(Q * optimum) + q.dot(optimum));
        c -= .1 * gradient;
    }
    model.c = c;
    return model;
}

TEST(ConstrainedSolverRandom, FiftyMixedProblemsWithKnownKKTPoint)
{
    for (unsigned seed = 0; seed < 50; ++seed)
    {
        SCOPED_TRACE(seed);
        std::mt19937 generator(seed);
        const int n = 8;
        Mat<double> P = Mat<double>::Identity(n, n);
        Vec<double> optimum = random_vector(generator, n), c = -P * optimum;
        dense::Model<double> model(P, c);
        for (int k = 0; k < 5; ++k)
        {
            Mat<double> Q = (.1 + .1 * k) * Mat<double>::Identity(n, n);
            Vec<double> gradient = .1 * random_vector(generator, n); gradient(0) = 1.;
            Vec<double> q = gradient - Q * optimum;
            model.quadratic_constraints.emplace_back(Q, q, .5 * optimum.dot(Q * optimum) + q.dot(optimum));
            c -= (.2 + .1 * k) * gradient;
        }
        Mat<double> F(4, n);
        for (int j = 0; j < n; ++j) F.col(j) = .1 * random_vector(generator, 4);
        F.row(0).setZero();
        Vec<double> v = random_vector(generator, 3); v.normalize();
        Vec<double> f(4); f(0) = 1.; f.tail(3) = v - F.bottomRows(3) * optimum;
        model.cone_constraints.emplace_back(F, f);
        Vec<double> dual(4); dual(0) = .3; dual.tail(3) = -.3 * v;
        c += F.transpose() * dual; model.c = c;
        check_backends(model, optimum);
    }
}

TEST(ConstrainedSolverRandom, LocalChainWithGlobalVariable)
{
    for (int stages : {2, 8, 32})
    {
        SCOPED_TRACE(stages);
        Vec<double> optimum;
        auto model = chain_model(stages, optimum, false);
        check_backends(model, optimum);
    }
}

TEST(ConstrainedSolverRandom, NonlocalQuadraticKeepsAllCouplings)
{
    Vec<double> optimum;
    auto model = chain_model(6, optimum, true);
    check_backends(model, optimum);
}

TEST(ConstrainedSolverRandom, ConeRowsAcrossDistantStages)
{
    Vec<double> optimum;
    auto model = chain_model(6, optimum, false);
    Mat<double> F = Mat<double>::Zero(3, optimum.size());
    F(1, 1) = 1.; F(2, 11) = 1.;
    Vec<double> slack(3); slack << 1., std::sqrt(.5), std::sqrt(.5);
    Vec<double> f = slack - F * optimum;
    model.cone_constraints.emplace_back(F, f);
    Vec<double> dual = -.1 * slack; dual(0) = .1;
    model.c += F.transpose() * dual;
    check_backends(model, optimum);
}

} // namespace
