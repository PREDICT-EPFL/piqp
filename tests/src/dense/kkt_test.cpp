// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/dense/kkt.hpp"
#include "piqp/kkt_system.hpp"
#include "piqp/utils/random_utils.hpp"

#include "gtest/gtest.h"
#include "utils.hpp"

using namespace piqp;
using namespace piqp::dense;

using T = double;

TEST(DenseKKTTest, UpdateData)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;

    Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);
    Data<T> data(qp_model);
    Settings<T> settings;

    // make sure P_utri has not complete diagonal filled
    data.P_utri(1, 1) = 0;

    T rho = 0.9;
    T delta = 1.2;
    Vec<T> x_reg(dim); x_reg.setConstant(rho);
    Vec<T> z_reg(n_ineq); z_reg.setConstant(1 + delta);

    KKT<T> kkt(data, settings);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.update_scalings_and_factor(delta, x_reg, z_reg);
    PIQP_EIGEN_MALLOC_ALLOWED();

    // update data
    Model<T> qp_model_new = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);
    data.P_utri = qp_model_new.P.triangularView<Eigen::Upper>();
    data.AT = qp_model_new.A.transpose();
    data.GT = qp_model_new.G.transpose();
    int update_options = KKTUpdateOptions::KKT_UPDATE_P | KKTUpdateOptions::KKT_UPDATE_A | KKTUpdateOptions::KKT_UPDATE_G;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.update_data(update_options);
    kkt.update_scalings_and_factor(delta, x_reg, z_reg);
    PIQP_EIGEN_MALLOC_ALLOWED();

    KKT<T> kkt2(data, settings);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt2.update_scalings_and_factor(delta, x_reg, z_reg);
    PIQP_EIGEN_MALLOC_ALLOWED();

    // assert update was correct, i.e. it's the same as a freshly initialized one
    assert_dense_triangular_equal<T, Eigen::Lower>(kkt.internal_kkt_mat(), kkt2.internal_kkt_mat());
}

TEST(DenseKKTTest, FactorizeSolve)
{
    isize dim = 20;
    isize n_eq = 8;
    isize n_ineq = 9;

    Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);
    Data<T> data(qp_model);
    Settings<T> settings;

    T rho = 0.9;
    T delta = 1.2;
    Variables<T> scaling; scaling.resize(dim, n_eq, n_ineq);
    scaling.s.setConstant(1);
    scaling.s_lb.setConstant(1);
    scaling.s_ub.setConstant(1);
    scaling.z.setConstant(1);
    scaling.z_lb.setConstant(1);
    scaling.z_ub.setConstant(1);

    KKTSystem<T, int, PIQP_DENSE> kkt(data, settings);
    kkt.init();
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.update_scalings_and_factor(false, rho, delta, scaling);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Variables<T> rhs;
    rhs.x = rand::vector_rand<T>(dim);
    rhs.y = rand::vector_rand<T>(n_eq);
    rhs.z = rand::vector_rand<T>(n_ineq);
    rhs.z_lb = rand::vector_rand<T>(dim);
    rhs.z_ub = rand::vector_rand<T>(dim);
    rhs.s = rand::vector_rand<T>(n_ineq);
    rhs.s_lb = rand::vector_rand<T>(dim);
    rhs.s_ub = rand::vector_rand<T>(dim);

    Variables<T> lhs;
    lhs.resize(dim, n_eq, n_ineq);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.solve(rhs, lhs);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Variables<T> rhs_sol;
    rhs_sol.resize(dim, n_eq, n_ineq);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.mul(lhs, rhs_sol);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(rhs.x.isApprox(rhs_sol.x, 1e-8));
    ASSERT_TRUE(rhs.y.isApprox(rhs_sol.y, 1e-8));
    ASSERT_TRUE(rhs.z.isApprox(rhs_sol.z, 1e-8));
    ASSERT_TRUE(rhs.z_lb.head(data.n_lb).isApprox(rhs_sol.z_lb.head(data.n_lb), 1e-8));
    ASSERT_TRUE(rhs.z_ub.head(data.n_ub).isApprox(rhs_sol.z_ub.head(data.n_ub), 1e-8));
    ASSERT_TRUE(rhs.s.isApprox(rhs_sol.s, 1e-8));
    ASSERT_TRUE(rhs.s_lb.head(data.n_lb).isApprox(rhs_sol.s_lb.head(data.n_lb), 1e-8));
    ASSERT_TRUE(rhs.s_ub.head(data.n_ub).isApprox(rhs_sol.s_ub.head(data.n_ub), 1e-8));
}
