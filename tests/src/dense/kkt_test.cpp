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
    Vec<T> s(n_ineq); s.setConstant(1);
    Vec<T> s_lb(dim); s_lb.setConstant(1);
    Vec<T> s_ub(dim); s_ub.setConstant(1);
    Vec<T> z(n_ineq); z.setConstant(1);
    Vec<T> z_lb(dim); z_lb.setConstant(1);
    Vec<T> z_ub(dim); z_ub.setConstant(1);

    KKTSystem<T, int, PIQP_DENSE> kkt(data, settings);
    kkt.init();
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.update_scalings_and_factor(false, rho, delta, s, s_lb, s_ub, z, z_lb, z_ub);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> rhs_x = rand::vector_rand<T>(dim);
    Vec<T> rhs_y = rand::vector_rand<T>(n_eq);
    Vec<T> rhs_z = rand::vector_rand<T>(n_ineq);
    Vec<T> rhs_z_lb = rand::vector_rand<T>(dim);
    Vec<T> rhs_z_ub = rand::vector_rand<T>(dim);
    Vec<T> rhs_s = rand::vector_rand<T>(n_ineq);
    Vec<T> rhs_s_lb = rand::vector_rand<T>(dim);
    Vec<T> rhs_s_ub = rand::vector_rand<T>(dim);

    Vec<T> delta_x(dim);
    Vec<T> delta_y(n_eq);
    Vec<T> delta_z(n_ineq);
    Vec<T> delta_z_lb(dim);
    Vec<T> delta_z_ub(dim);
    Vec<T> delta_s(n_ineq);
    Vec<T> delta_s_lb(dim);
    Vec<T> delta_s_ub(dim);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.solve(rhs_x, rhs_y, rhs_z, rhs_z_lb, rhs_z_ub, rhs_s, rhs_s_lb, rhs_s_ub,
              delta_x, delta_y, delta_z, delta_z_lb, delta_z_ub, delta_s, delta_s_lb, delta_s_ub);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> rhs_x_sol(dim);
    Vec<T> rhs_y_sol(n_eq);
    Vec<T> rhs_z_sol(n_ineq);
    Vec<T> rhs_z_lb_sol(dim);
    Vec<T> rhs_z_ub_sol(dim);
    Vec<T> rhs_s_sol(n_ineq);
    Vec<T> rhs_s_lb_sol(dim);
    Vec<T> rhs_s_ub_sol(dim);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.mul(delta_x, delta_y, delta_z, delta_z_lb, delta_z_ub, delta_s, delta_s_lb, delta_s_ub,
            rhs_x_sol, rhs_y_sol, rhs_z_sol, rhs_z_lb_sol, rhs_z_ub_sol, rhs_s_sol, rhs_s_lb_sol, rhs_s_ub_sol);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(rhs_x.isApprox(rhs_x_sol, 1e-8));
    ASSERT_TRUE(rhs_y.isApprox(rhs_y_sol, 1e-8));
    ASSERT_TRUE(rhs_z.isApprox(rhs_z_sol, 1e-8));
    ASSERT_TRUE(rhs_z_lb.head(data.n_lb).isApprox(rhs_z_lb_sol.head(data.n_lb), 1e-8));
    ASSERT_TRUE(rhs_z_ub.head(data.n_ub).isApprox(rhs_z_ub_sol.head(data.n_ub), 1e-8));
    ASSERT_TRUE(rhs_s.isApprox(rhs_s_sol, 1e-8));
    ASSERT_TRUE(rhs_s_lb.head(data.n_lb).isApprox(rhs_s_lb_sol.head(data.n_lb), 1e-8));
    ASSERT_TRUE(rhs_s_ub.head(data.n_ub).isApprox(rhs_s_ub_sol.head(data.n_ub), 1e-8));
}
