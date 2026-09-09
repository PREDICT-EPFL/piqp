#define PIQP_EIGEN_CHECK_MALLOC
#include "piqp/constrained/newton.hpp"
#include "gtest/gtest.h"

using namespace piqp;

namespace
{

void assign(dense::Data<double>&, const Mat<double>&, const Mat<double>&, const Mat<double>&);
void assign(sparse::Data<double, int>&, const Mat<double>&, const Mat<double>&, const Mat<double>&);

template<int MatrixType>
void check_newton(KKTSolver backend)
{
    constexpr int n = 13, p = 5, m = 6;
    Mat<double> H = Mat<double>::Identity(n, n);
    Mat<double> A = Mat<double>::Zero(p, n);
    Mat<double> J = Mat<double>::Zero(m, n);
    for (int i = 0; i < p; ++i)
    {
        A(i, 2 * i + 1) = -1;
        A(i, 2 * i + 2) = 1;
    }
    for (int i = 0; i < m; ++i)
    {
        J(i, 2 * i) = 0.5;
        J(i, 2 * i + 1) = -0.7;
        J(i, n - 1) = 0.1;
        H(2 * i, n - 1) = H(n - 1, 2 * i) = 0.02;
    }
    typename constrained::Newton<double, int, MatrixType>::DataType data;
    data.resize(n, p, m);
    assign(data, H, A, J);
    data.GT.coeffRef(0, 0) = 0;
    Settings<double> settings;
    settings.kkt_solver = backend;
    constrained::Newton<double, int, MatrixType> newton;
    ASSERT_TRUE(newton.setup(data, settings));
    Vec<double> z_reg = Vec<double>::LinSpaced(m, 0.1, 2);
    Vec<double> rhs = Vec<double>::LinSpaced(n + p + m, -1, 1);
    Vec<double> rx = rhs.head(n), ry = rhs.segment(n, p), rz = rhs.tail(m);
    Vec<double> dx(n), dy(p), dz(m);
    for (int pass = 0; pass < 3; ++pass)
    {
        const double rho = pass == 2 ? 1e-8 : 0.02;
        const double delta = pass == 2 ? 1e-7 : 0.04;
        H(0, 0) = 1 + pass;
        J(0, 0) = pass == 0 ? 0 : 0.25 * pass;
        newton.data.P_utri.coeffRef(0, 0) = H(0, 0);
        newton.data.GT.coeffRef(0, 0) = J(0, 0);
        Mat<double> K = Mat<double>::Zero(n + p + m, n + p + m);
        K.topLeftCorner(n, n) = H;
        K.topLeftCorner(n, n).diagonal().array() += rho;
        K.block(0, n, n, p) = A.transpose();
        K.block(n, 0, p, n) = A;
        K.block(n, n, p, p).diagonal().setConstant(-delta);
        K.block(0, n + p, n, m) = J.transpose();
        K.block(n + p, 0, m, n) = J;
        K.bottomRightCorner(m, m).diagonal() = -z_reg;
        Vec<double> expected = K.fullPivLu().solve(rhs);
        PIQP_EIGEN_MALLOC_NOT_ALLOWED();
        const bool factored = newton.factor(rho, delta, z_reg);
        const bool solved = factored && newton.solve(rx, ry, rz, dx, dy, dz);
        PIQP_EIGEN_MALLOC_ALLOWED();
        ASSERT_TRUE(factored);
        ASSERT_TRUE(solved) << newton.residual_norm;
        Vec<double> actual(n + p + m);
        actual << dx, dy, dz;
        EXPECT_LT((actual - expected).lpNorm<Eigen::Infinity>(), 1e-9);
        EXPECT_LT((K * actual - rhs).lpNorm<Eigen::Infinity>(), 5e-12);
    }
}

void assign(dense::Data<double>& data, const Mat<double>& H, const Mat<double>& A, const Mat<double>& J)
{
    data.P_utri = H.triangularView<Eigen::Upper>();
    data.AT = A.transpose();
    data.GT = J.transpose();
}

void assign(sparse::Data<double, int>& data, const Mat<double>& H, const Mat<double>& A, const Mat<double>& J)
{
    Mat<double> upper = H.triangularView<Eigen::Upper>();
    data.P_utri = upper.sparseView();
    data.AT = A.transpose().sparseView();
    data.GT = J.transpose().sparseView();
}

TEST(ConstrainedNewton, Dense)
{
    check_newton<PIQP_DENSE>(KKTSolver::dense_cholesky);
}

TEST(ConstrainedNewton, SparseModes)
{
    for (auto backend : {KKTSolver::sparse_ldlt, KKTSolver::sparse_ldlt_eq_cond,
                         KKTSolver::sparse_ldlt_ineq_cond, KKTSolver::sparse_ldlt_cond})
    {
        SCOPED_TRACE(kkt_solver_to_string(backend));
        check_newton<PIQP_SPARSE>(backend);
    }
}

TEST(ConstrainedNewton, Multistage)
{
#ifdef PIQP_HAS_BLASFEO
    check_newton<PIQP_SPARSE>(KKTSolver::sparse_multistage);
#else
    GTEST_SKIP() << "BLASFEO not available";
#endif
}

TEST(ConstrainedNewton, EmptyDualBlocks)
{
    dense::Data<double> data;
    data.resize(2, 0, 0);
    data.P_utri.setIdentity();
    constrained::Newton<double, int, PIQP_DENSE> newton;
    Settings<double> settings;
    ASSERT_TRUE(newton.setup(data, settings));
    Vec<double> rx = Vec<double>::Ones(2), empty(0), dx(2), dy(0), dz(0);
    ASSERT_TRUE(newton.factor(0.1, 0.1, empty));
    ASSERT_TRUE(newton.solve(rx, empty, empty, dx, dy, dz));
    EXPECT_NEAR(dx(0), 1 / 1.1, 1e-14);
}

} // namespace
