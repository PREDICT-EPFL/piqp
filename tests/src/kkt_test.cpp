// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/utils/random_utils.hpp"

#include "gtest/gtest.h"
#include "utils.hpp"

using namespace piqp;

using T = double;
using I = int;

using kkt_types = testing::Types<KKT<T, I, KKTMode::KKT_FULL>,
                                 KKT<T, I, KKTMode::KKT_EQ_ELIMINATED>,
                                 KKT<T, I, KKTMode::KKT_INEQ_ELIMINATED>,
                                 KKT<T, I, KKTMode::KKT_ALL_ELIMINATED>>;
template <typename T>
class KKTTest : public ::testing::Test {};
TYPED_TEST_SUITE(KKTTest, kkt_types);

TYPED_TEST(KKTTest, UpdateScalings)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.2;

    Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    Data<T, I> data;
    data.n = dim;
    data.p = n_eq;
    data.m = n_ineq;
    data.P_utri = qp_model.P.triangularView<Eigen::Upper>();
    data.AT = qp_model.A.transpose();
    data.GT = qp_model.G.transpose();
    data.c = qp_model.c;
    data.b = qp_model.b;
    data.h = qp_model.h;

    // make sure P_utri has not complete diagonal filled
    data.P_utri.coeffRef(1, 1) = 0;
    data.P_utri.prune(0.0);

    T rho = 0.9;
    T delta = 1.2;

    TypeParam kkt(data);
    kkt.init(rho, delta);

    rho = 0.8;
    delta = 0.2;
    Vec<T> s(n_ineq); s.setConstant(1);
    Vec<T> z(n_ineq); z.setConstant(1);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.update_scalings(rho, delta, s, z);
    PIQP_EIGEN_MALLOC_ALLOWED();

    // assert PKPt matrix is upper triangular
    SparseMat<T, I> PKPt_upper = kkt.PKPt.template triangularView<Eigen::Upper>();
    assert_sparse_matrices_equal(kkt.PKPt, PKPt_upper);

    TypeParam kkt2(data);
    kkt2.init(rho, delta);

    // assert update was correct, i.e. it's the same as a freshly initialized one
    EXPECT_TRUE(kkt.PKPt.isApprox(kkt2.PKPt, 1e-8));
}

TYPED_TEST(KKTTest, UpdateData)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.2;

    Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    Data<T, I> data;
    data.n = dim;
    data.p = n_eq;
    data.m = n_ineq;
    data.P_utri = qp_model.P.triangularView<Eigen::Upper>();
    data.AT = qp_model.A.transpose();
    data.GT = qp_model.G.transpose();
    data.c = qp_model.c;
    data.b = qp_model.b;
    data.h = qp_model.h;

    // make sure P_utri has not complete diagonal filled
    data.P_utri.coeffRef(1, 1) = 0;
    data.P_utri.prune(0.0);

    T rho = 0.9;
    T delta = 1.2;

    TypeParam kkt(data);
    kkt.init(rho, delta);

    // update data
    Eigen::Map<Vec<T>>(data.P_utri.valuePtr(), data.P_utri.nonZeros()) = rand::vector_rand<T>(data.P_utri.nonZeros());
    Eigen::Map<Vec<T>>(data.AT.valuePtr(), data.AT.nonZeros()) = rand::vector_rand<T>(data.AT.nonZeros());
    Eigen::Map<Vec<T>>(data.GT.valuePtr(), data.GT.nonZeros()) = rand::vector_rand<T>(data.GT.nonZeros());
    int update_options = KKTUpdateOptions::KKT_UPDATE_P | KKTUpdateOptions::KKT_UPDATE_A | KKTUpdateOptions::KKT_UPDATE_G;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.update_data(update_options);
    PIQP_EIGEN_MALLOC_ALLOWED();

    // assert PKPt matrix is upper triangular
    SparseMat<T, I> PKPt_upper = kkt.PKPt.template triangularView<Eigen::Upper>();
    assert_sparse_matrices_equal(kkt.PKPt, PKPt_upper);

    TypeParam kkt2(data);
    kkt2.init(rho, delta);

    // assert update was correct, i.e. it's the same as a freshly initialized one
    EXPECT_TRUE(kkt.PKPt.isApprox(kkt2.PKPt, 1e-8));
}

TYPED_TEST(KKTTest, FactorizeSolve)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.2;

    Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    Data<T, I> data;
    data.n = dim;
    data.p = n_eq;
    data.m = n_ineq;
    data.P_utri = qp_model.P.triangularView<Eigen::Upper>();
    data.AT = qp_model.A.transpose();
    data.GT = qp_model.G.transpose();
    data.c = qp_model.c;
    data.b = qp_model.b;
    data.h = qp_model.h;

    T rho = 0.9;
    T delta = 1.2;

    TypeParam kkt(data);
    kkt.init(rho, delta);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    ASSERT_TRUE(kkt.factorize());
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> rhs_x = rand::vector_rand<T>(dim);
    Vec<T> rhs_y = rand::vector_rand<T>(n_eq);
    Vec<T> rhs_z = rand::vector_rand<T>(n_ineq);
    Vec<T> rhs_s = rand::vector_rand<T>(n_ineq);

    Vec<T> delta_x(dim);
    Vec<T> delta_y(n_eq);
    Vec<T> delta_z(n_ineq);
    Vec<T> delta_s(n_ineq);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.solve(rhs_x, rhs_y, rhs_z, rhs_s, delta_x, delta_y, delta_z, delta_s);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> rhs_x_sol(dim);
    Vec<T> rhs_y_sol(n_eq);
    Vec<T> rhs_z_sol(n_ineq);
    Vec<T> rhs_s_sol(n_ineq);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.multiply(delta_x, delta_y, delta_z, delta_s, rhs_x_sol, rhs_y_sol, rhs_z_sol, rhs_s_sol);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(rhs_x.isApprox(rhs_x_sol, 1e-8));
    ASSERT_TRUE(rhs_y.isApprox(rhs_y_sol, 1e-8));
    ASSERT_TRUE(rhs_z.isApprox(rhs_z_sol, 1e-8));
    ASSERT_TRUE(rhs_s.isApprox(rhs_s_sol, 1e-8));
}
