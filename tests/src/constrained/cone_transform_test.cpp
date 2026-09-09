#define PIQP_EIGEN_CHECK_MALLOC
#include "piqp/constrained/cone_transform.hpp"
#include "piqp/constrained/newton.hpp"
#include "constrained_cone_reference.hpp"
#include "gtest/gtest.h"

using namespace piqp;
using namespace piqp::constrained;
using piqp::test::ConeScaling;

TEST(ConstrainedConeTransform, DenseReferenceAndAliasing)
{
    for (int dimension : {2, 3, 8, 31})
        for (double delta : {0., 1e-12, 0.1, 1e5})
            for (int trial = 0; trial < 10; ++trial)
            {
                Vec<double> s = Vec<double>::Random(dimension), z = Vec<double>::Random(dimension);
                s(0) = s.tail(dimension - 1).norm() + 0.3;
                z(0) = z.tail(dimension - 1).norm() + 0.7;
                ConeScaling<double> dense(dimension);
                ConeTransform<double> transform(dimension);
                ASSERT_TRUE(dense.compute(s, z));
                ASSERT_TRUE(transform.compute(s, z, delta));
                EXPECT_TRUE(transform.lambda.isApprox(dense.lambda, 1e-12));
                Mat<double> D = dense.W * dense.W;
                D.diagonal().array() += delta;
                Eigen::SelfAdjointEigenSolver<Mat<double>> eig(D);
                Mat<double> B = eig.eigenvectors() * eig.eigenvalues().array().sqrt().inverse().matrix().asDiagonal() * eig.eigenvectors().transpose();
                Vec<double> x = Vec<double>::Random(dimension), out(dimension), alias(dimension);
                PIQP_EIGEN_MALLOC_NOT_ALLOWED();
                transform.apply_W(x, out);
                alias = x; transform.apply_W(alias, alias);
                PIQP_EIGEN_MALLOC_ALLOWED();
                EXPECT_TRUE(out.isApprox(dense.W * x, 1e-12));
                EXPECT_TRUE(alias.isApprox(out, 1e-14));
                transform.apply_Winv(x, out);
                EXPECT_TRUE(out.isApprox(dense.Winv * x, 1e-12));
                alias = x; transform.apply_Winv(alias, alias);
                EXPECT_TRUE(alias.isApprox(out, 1e-14));
                transform.apply_W2(x, out);
                EXPECT_TRUE(out.isApprox(dense.W * dense.W * x, 1e-12));
                alias = x; transform.apply_W2(alias, alias);
                EXPECT_TRUE(alias.isApprox(out, 1e-14));
                transform.apply_inverse_sqrt_D(x, out);
                EXPECT_TRUE(out.isApprox(B * x, 1e-10));
                alias = x; transform.apply_inverse_sqrt_D(alias, alias);
                EXPECT_TRUE(alias.isApprox(out, 1e-14));
                transform.apply_inverse_sqrt_D(out, alias);
                EXPECT_TRUE((D * alias).isApprox(x, 1e-10));
                transform.apply_W_inverse_sqrt_D(x, out);
                EXPECT_TRUE(out.isApprox(B * dense.W * x, 1e-10));
                alias = x; transform.apply_W_inverse_sqrt_D(alias, alias);
                EXPECT_TRUE(alias.isApprox(out, 1e-14));
            }
}

TEST(ConstrainedConeTransform, LargeConeWithoutAllocation)
{
    constexpr int dimension = 10000;
    Vec<double> s = Vec<double>::Ones(dimension), z = -s, out(dimension);
    s(0) = 110; z(0) = 130;
    ConeTransform<double> transform(dimension);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    const bool ok = transform.compute(s, z, 1e-7);
    transform.apply_W(z, out);
    transform.apply_Winv(out, out);
    transform.apply_W2(out, out);
    transform.apply_inverse_sqrt_D(out, out);
    transform.apply_W_inverse_sqrt_D(out, out);
    PIQP_EIGEN_MALLOC_ALLOWED();
    ASSERT_TRUE(ok);
    EXPECT_TRUE(out.allFinite());
    transform.apply_W(z, out);
    transform.apply_W(out, out);
    EXPECT_TRUE(out.isApprox(s, 1e-12));
}

TEST(ConstrainedConeTransform, CentralAndExtremeScales)
{
    for (double scale : {1e-200, 1., 1e200})
    {
        Vec<double> s = Vec<double>::Zero(4), z = s, out(4);
        s(0) = 4 * scale; z(0) = scale;
        ConeTransform<double> transform(4);
        ASSERT_TRUE(transform.compute(s, z, 0.3));
        Vec<double> x = Vec<double>::Ones(4);
        transform.apply_W(x, out);
        EXPECT_TRUE(out.isApprox(2 * x, 1e-14));
        transform.apply_Winv(x, out);
        EXPECT_TRUE(out.isApprox(0.5 * x, 1e-14));
        transform.apply_inverse_sqrt_D(x, out);
        EXPECT_TRUE(out.isApprox(x / std::sqrt(4.3), 1e-14));
    }
    Vec<double> s = Vec<double>::Zero(3), z = s, out(3);
    s(0) = 1e200; z(0) = 1e-200;
    ConeTransform<double> transform(3);
    ASSERT_TRUE(transform.compute(s, z, 1.));
    Vec<double> x = Vec<double>::Ones(3);
    transform.apply_inverse_sqrt_D(x, out);
    EXPECT_NEAR(out(0) / 1e-200, 1., 1e-14);
    EXPECT_TRUE(out.allFinite());
}

TEST(ConstrainedConeTransform, NearlyCentralDirection)
{
    Vec<double> s = Vec<double>::Zero(3), z = s, x = Vec<double>::Ones(3), out(3);
    s(0) = z(0) = 1;
    s(1) = 1e-200;
    ConeTransform<double> transform(3);
    ASSERT_TRUE(transform.compute(s, z, 0.));
    transform.apply_W(x, out);
    EXPECT_TRUE(out.isApprox(x, 1e-14));
    x(0) = 1e200;
    transform.apply_W(x, out);
    EXPECT_NEAR(out(1), 1.5, 1e-14);
    transform.apply_Winv(x, out);
    EXPECT_NEAR(out(1), 0.5, 1e-14);
    transform.apply_inverse_sqrt_D(x, out);
    EXPECT_NEAR(out(1), 0.5, 1e-14);
}

TEST(ConstrainedConeTransform, SubnormalScaledCoefficients)
{
    for (double magnitude : {1e-308, std::numeric_limits<double>::min()})
    {
        Vec<double> s(3), z = Vec<double>::Zero(3), x(3), out(3);
        s << magnitude, .1 * magnitude, 0.;
        z(0) = 1. / magnitude;
        x << 1., 0., 1.;
        ConeTransform<double> transform(3);
        const double delta = 1e4;
        PIQP_EIGEN_MALLOC_NOT_ALLOWED();
        const bool valid = transform.compute(s, z, delta);
        transform.apply_W_inverse_sqrt_D(x, out);
        PIQP_EIGEN_MALLOC_ALLOWED();
        ASSERT_TRUE(valid);
        ASSERT_TRUE(out.allFinite());
        const double plus = std::sqrt(s(0) + s(1)) / std::sqrt(z(0));
        const double minus = std::sqrt(s(0) - s(1)) / std::sqrt(z(0));
        const double kp = plus / std::hypot(plus, std::sqrt(delta));
        const double km = minus / std::hypot(minus, std::sqrt(delta));
        EXPECT_NEAR(out(0) / (.5 * kp + .5 * km), 1., 1e-11);
        EXPECT_NEAR(out(1) / (.5 * kp - .5 * km), 1., 1e-11);
        Vec<double> alias = x;
        transform.apply_W_inverse_sqrt_D(alias, alias);
        EXPECT_EQ(out, alias);
    }
}

TEST(ConstrainedConeTransform, NearBoundaryNewtonResidual)
{
    for (double margin : {1e-3, 1e-6, 1e-9, 1e-12})
    {
        SCOPED_TRACE(margin);
        Vec<double> s(3), z(3), h(3), feasibility_rhs(3), rx(2);
        s << 1, 1 - margin, 0;
        z << 2, -2 + 3 * margin, 0;
        h << .1, .2, -.2;
        feasibility_rhs << .2, -.1, .4;
        rx << .3, -.4;
        const double rho = 1e-5, delta = 1e-4;
        ConeTransform<double> transform(3);
        ASSERT_TRUE(transform.compute(s, z, delta));
        Vec<double> scaled_z(3);
        transform.apply_W2(z, scaled_z);
        EXPECT_LT((scaled_z - s).norm(), 1e-9);
        Mat<double> J(3, 2);
        J << 1, .2, .3, -.4, .5, .6;
        Mat<double> whitened = J;
        for (int i = 0; i < 2; ++i) transform.apply_inverse_sqrt_D(whitened.col(i), whitened.col(i));
        Vec<double> t(3), rz(3), dz(3), ds(3), work(3), other(3), dx(2), dy(0), empty(0);
        transform.apply_W_inverse_sqrt_D(h, t);
        transform.apply_inverse_sqrt_D(feasibility_rhs, rz);
        rz -= t;
        dense::Data<double> data;
        data.resize(2, 0, 3);
        data.P_utri.setIdentity(); data.GT = whitened.transpose();
        Newton<double, int, PIQP_DENSE> newton;
        Settings<double> settings;
        ASSERT_TRUE(newton.setup(data, settings));
        Vec<double> regularization = Vec<double>::Ones(3);
        ASSERT_TRUE(newton.factor(rho, delta, regularization));
        ASSERT_TRUE(newton.solve(rx, empty, rz, dx, dy, dz));
        transform.apply_inverse_sqrt_D(dz, dz);
        ds = feasibility_rhs - J * dx + delta * dz;
        EXPECT_LT(((1 + rho) * dx + J.transpose() * dz - rx).norm(), 1e-9);
        EXPECT_LT((J * dx + ds - delta * dz - feasibility_rhs).norm(), 1e-12);
        transform.apply_Winv(ds, work);
        transform.apply_W(dz, other);
        EXPECT_LT((work + other - h).norm(), 1e-7);
        EXPECT_TRUE(dx.allFinite() && dz.allFinite() && ds.allFinite());
    }
}
