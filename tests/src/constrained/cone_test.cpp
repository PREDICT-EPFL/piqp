#define PIQP_EIGEN_CHECK_MALLOC
#include "piqp/fwd.hpp"
#include "piqp/constrained/cone.hpp"
#include "constrained_cone_reference.hpp"
#include "gtest/gtest.h"

using namespace piqp;
using namespace piqp::constrained;
using piqp::test::ConeScaling;

TEST(ConstrainedConeTest, JordanDivisionMatchesLinearSolve)
{
    for (int n : {2, 3, 8, 31})
    {
        for (int sample = 0; sample < 20; ++sample)
        {
            Vec<double> a = Vec<double>::Random(n), b = Vec<double>::Random(n);
            repair(a);
            Mat<double> L = Mat<double>::Identity(n, n) * a(0);
            L.row(0) = a.transpose();
            L.col(0) = a;
            const Vec<double> expected = L.fullPivLu().solve(b);
            EXPECT_LT((jordan_solve<double>(a, b) - expected).norm(), 1e-12);
            EXPECT_LT((jordan<double>(a, b) - L * b).norm(), 1e-12);
            Vec<double> work(n + 2);
            PIQP_EIGEN_MALLOC_NOT_ALLOWED();
            jordan<double>(a, b, work.segment(1, n));
            jordan_solve<double>(a, b, work.segment(1, n));
            repair<double>(work.segment(1, n));
            PIQP_EIGEN_MALLOC_ALLOWED();
            for (double scale : {1e-200, 1e200})
            {
                Vec<double> sa = scale * a, sb = scale * b;
                EXPECT_LT((jordan_solve<double>(sa, sb) - expected).norm(), 1e-12);
            }
        }
    }
}

TEST(ConstrainedConeTest, ScalingIdentitiesAndNoAllocation)
{
    for (int n : {2, 3, 8, 31})
    {
        ConeScaling<double> scaling(n);
        for (int sample = 0; sample < 20; ++sample)
        {
            Vec<double> s = Vec<double>::Random(n), z = Vec<double>::Random(n);
            repair(s);
            repair(z);
            PIQP_EIGEN_MALLOC_NOT_ALLOWED();
            bool valid = scaling.compute(s, z);
            PIQP_EIGEN_MALLOC_ALLOWED();
            ASSERT_TRUE(valid);
            EXPECT_LT((scaling.W - scaling.W.transpose()).norm(), 1e-13);
            EXPECT_LT((scaling.W * scaling.Winv - Mat<double>::Identity(n, n)).norm(), 1e-12);
            EXPECT_LT((scaling.lambda - scaling.Winv * s).norm(), 1e-12);
            EXPECT_NEAR(scaling.lambda.squaredNorm(), s.dot(z), 1e-11);
            EXPECT_GT(scaling.W.selfadjointView<Eigen::Lower>().eigenvalues().minCoeff(), 0);
            ASSERT_TRUE(scaling.compute(s, s));
            EXPECT_LT((scaling.W - Mat<double>::Identity(n, n)).norm(), 1e-12);
        }
    }
}

TEST(ConstrainedConeTest, ScalingExtremeMagnitudes)
{
    Vec<double> s(3), z(3);
    s << 2, 0.5, -0.8;
    z << 3, -0.3, 1.2;
    for (double ss : {1e-200, 1.0, 1e200})
    {
        for (double zs : {1e-200, 1.0, 1e200})
        {
            ConeScaling<double> scaling(3);
            Vec<double> a = ss * s, b = zs * z;
            ASSERT_TRUE(scaling.compute(a, b));
            Vec<double> mapped = scaling.Winv * a;
            EXPECT_LT((scaling.lambda - mapped).stableNorm() / scaling.lambda.stableNorm(), 1e-12);
            EXPECT_LT((scaling.W * scaling.Winv - Mat<double>::Identity(3, 3)).norm(), 1e-12);
        }
    }
    s << 1, 1 - 1e-9, 0;
    z << 1, 0, 1 - 1e-9;
    ConeScaling<double> scaling(3);
    ASSERT_TRUE(scaling.compute(s, z));
    EXPECT_LT((scaling.lambda - scaling.Winv * s).norm() / scaling.lambda.norm(), 1e-6);
    z(0) = 0;
    EXPECT_FALSE(scaling.compute(s, z));
}

TEST(ConstrainedConeTest, AnalyticBoundarySteps)
{
    Vec<double> v(3), d(3);
    v << 1, 0, 0;
    d << 0, 2, 0;
    EXPECT_DOUBLE_EQ(max_step<double>(v, d), 0.5);
    d << -2, 0, 0;
    EXPECT_DOUBLE_EQ(max_step<double>(v, d), 0.5);
    d << -2, 2, 0;
    EXPECT_DOUBLE_EQ(max_step<double>(v, d), 0.25);
    d << 2, 0, 0;
    EXPECT_DOUBLE_EQ(max_step<double>(v, d), 1);
    d.setZero();
    EXPECT_DOUBLE_EQ(max_step<double>(v, d), 1);
    v << 1, 1 - 1e-12, 0;
    d << 0, 0, 1;
    EXPECT_NEAR(max_step<double>(v, d), std::sqrt((v(0) - v(1)) * (v(0) + v(1))), 1e-15);
}

TEST(ConstrainedConeTest, RandomBoundaryAgainstBisection)
{
    for (int n : {2, 3, 8, 31})
    {
        for (int sample = 0; sample < 100; ++sample)
        {
            Vec<double> v = Vec<double>::Random(n), d = 10 * Vec<double>::Random(n);
            repair(v, 0.1);
            double lo = 0, hi = 1;
            for (int k = 0; k < 60; ++k)
            {
                const double mid = (lo + hi) / 2;
                Vec<double> point = v + mid * d;
                if (interior<double>(point)) lo = mid;
                else hi = mid;
            }
            EXPECT_NEAR(max_step<double>(v, d), lo, 2e-13);
            for (double scale : {1e-200, 1e200})
            {
                Vec<double> sv = scale * v, sd = scale * d;
                EXPECT_NEAR(max_step<double>(sv, sd), lo, 2e-13);
            }
        }
    }
}

TEST(ConstrainedConeTest, RepairAndCentrality)
{
    Vec<double> v(3), e = Vec<double>::Zero(3);
    e(0) = 1;
    v << -2, 3, 4;
    EXPECT_FALSE(interior<double>(v));
    repair(v);
    EXPECT_TRUE(interior<double>(v));
    EXPECT_EQ(v(0), 6);
    v << 0, 1e200, 0;
    repair(v);
    EXPECT_TRUE(interior<double>(v));
    v << 3, 1, -1;
    Vec<double> dual = 0.7 * jordan_solve<double>(v, e);
    EXPECT_LT((jordan<double>(v, dual) - 0.7 * e).norm(), 1e-14);
    EXPECT_NEAR(v.dot(dual), 0.7, 1e-14);
}

TEST(ConstrainedConeTest, SinglePrecisionScaling)
{
    Vec<float> s(3), z(3);
    s << 2, 0.5f, -0.8f;
    z << 3, -0.3f, 1.2f;
    for (float factor : {1e-20f, 1.0f, 1e20f})
    {
        Vec<float> a = factor * s, b = z / factor;
        ConeScaling<float> scaling(3);
        ASSERT_TRUE(scaling.compute(a, b));
        EXPECT_LT((scaling.lambda - scaling.Winv * a).norm() / scaling.lambda.norm(), 1e-5f);
    }
}
