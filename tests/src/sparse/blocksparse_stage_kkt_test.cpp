// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/utils/io_utils.hpp"
#include "piqp/utils/random_utils.hpp"

#include "gtest/gtest.h"

using T = double;
using I = int;

using namespace piqp;
using namespace piqp::sparse;

class BlocksparseStageKKTTest : public testing::TestWithParam<std::string> {};

TEST_P(BlocksparseStageKKTTest, FactorizeSolve)
{
    std::string path = "data/" + GetParam() + ".mat";
    std::cout << path << std::endl;
    Model<T, I> model = load_sparse_model<T, I>(path);
    Data<T, I> data(model);
    Settings<T> settings;

    T rho = 0.9;
    T delta = 1.2;
    Vec<T> s(data.GT.cols()); s.setConstant(1);
    Vec<T> s_lb(data.n_lb); s_lb.setConstant(1);
    Vec<T> s_ub(data.n_ub); s_ub.setConstant(1);
    Vec<T> z(data.GT.cols()); z.setConstant(1);
    Vec<T> z_lb(data.n_lb); z_lb.setConstant(1);
    Vec<T> z_ub(data.n_ub); z_ub.setConstant(1);

    BlocksparseStageKKT<T, I> kkt_tridiag(data, settings);
    KKT<T, I> kkt_sparse(data, settings);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_tridiag.update_scalings_and_factor(false, rho, delta, s, s_lb, s_ub, z, z_lb, z_ub);
    kkt_sparse.update_scalings_and_factor(false, rho, delta, s, s_lb, s_ub, z, z_lb, z_ub);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> rhs_x = rand::vector_rand<T>(data.n);
    Vec<T> rhs_y = rand::vector_rand<T>(data.p);
    Vec<T> rhs_z = rand::vector_rand<T>(data.m);
    Vec<T> rhs_z_lb = rand::vector_rand<T>(data.n);
    Vec<T> rhs_z_ub = rand::vector_rand<T>(data.n);
    Vec<T> rhs_s = rand::vector_rand<T>(data.m);
    Vec<T> rhs_s_lb = rand::vector_rand<T>(data.n);
    Vec<T> rhs_s_ub = rand::vector_rand<T>(data.n);

    Vec<T> delta_x(data.n);
    Vec<T> delta_y(data.p);
    Vec<T> delta_z(data.m);
    Vec<T> delta_z_lb(data.n);
    Vec<T> delta_z_ub(data.n);
    Vec<T> delta_s(data.m);
    Vec<T> delta_s_lb(data.n);
    Vec<T> delta_s_ub(data.n);

    Vec<T> delta_x_sparse(data.n);
    Vec<T> delta_y_sparse(data.p);
    Vec<T> delta_z_sparse(data.m);
    Vec<T> delta_z_lb_sparse(data.n);
    Vec<T> delta_z_ub_sparse(data.n);
    Vec<T> delta_s_sparse(data.m);
    Vec<T> delta_s_lb_sparse(data.n);
    Vec<T> delta_s_ub_sparse(data.n);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_tridiag.solve(rhs_x, rhs_y, rhs_z, rhs_z_lb, rhs_z_ub, rhs_s, rhs_s_lb, rhs_s_ub,
                      delta_x, delta_y, delta_z, delta_z_lb, delta_z_ub, delta_s, delta_s_lb, delta_s_ub);
    kkt_sparse.solve(rhs_x, rhs_y, rhs_z, rhs_z_lb, rhs_z_ub, rhs_s, rhs_s_lb, rhs_s_ub,
                     delta_x_sparse, delta_y_sparse, delta_z_sparse, delta_z_lb_sparse, delta_z_ub_sparse,
                     delta_s_sparse, delta_s_lb_sparse, delta_s_ub_sparse);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(delta_x.isApprox(delta_x_sparse, 1e-8));
    ASSERT_TRUE(delta_y.isApprox(delta_y_sparse, 1e-8));
    ASSERT_TRUE(delta_z.isApprox(delta_z_sparse, 1e-8));
    ASSERT_TRUE(delta_z_lb.head(data.n_lb).isApprox(delta_z_lb_sparse.head(data.n_lb), 1e-8));
    ASSERT_TRUE(delta_z_ub.head(data.n_ub).isApprox(delta_z_ub_sparse.head(data.n_ub), 1e-8));
    ASSERT_TRUE(delta_s.isApprox(delta_s_sparse, 1e-8));
    ASSERT_TRUE(delta_s_lb.head(data.n_lb).isApprox(delta_s_lb_sparse.head(data.n_lb), 1e-8));
    ASSERT_TRUE(delta_s_ub.head(data.n_ub).isApprox(delta_s_ub_sparse.head(data.n_ub), 1e-8));

    Vec<T> rhs_x_sol(data.n);
    Vec<T> rhs_y_sol(data.p);
    Vec<T> rhs_z_sol(data.m);
    Vec<T> rhs_z_lb_sol(data.n);
    Vec<T> rhs_z_ub_sol(data.n);
    Vec<T> rhs_s_sol(data.m);
    Vec<T> rhs_s_lb_sol(data.n);
    Vec<T> rhs_s_ub_sol(data.n);

    Vec<T> rhs_x_sol_sparse(data.n);
    Vec<T> rhs_y_sol_sparse(data.p);
    Vec<T> rhs_z_sol_sparse(data.m);
    Vec<T> rhs_z_lb_sol_sparse(data.n);
    Vec<T> rhs_z_ub_sol_sparse(data.n);
    Vec<T> rhs_s_sol_sparse(data.m);
    Vec<T> rhs_s_lb_sol_sparse(data.n);
    Vec<T> rhs_s_ub_sol_sparse(data.n);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_tridiag.multiply(delta_x, delta_y, delta_z, delta_z_lb, delta_z_ub, delta_s, delta_s_lb, delta_s_ub,
                         rhs_x_sol, rhs_y_sol, rhs_z_sol, rhs_z_lb_sol, rhs_z_ub_sol, rhs_s_sol, rhs_s_lb_sol, rhs_s_ub_sol);
    kkt_sparse.multiply(delta_x, delta_y, delta_z, delta_z_lb, delta_z_ub, delta_s, delta_s_lb, delta_s_ub,
                        rhs_x_sol_sparse, rhs_y_sol_sparse, rhs_z_sol_sparse, rhs_z_lb_sol_sparse,
                        rhs_z_ub_sol_sparse, rhs_s_sol_sparse, rhs_s_lb_sol_sparse, rhs_s_ub_sol_sparse);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(rhs_x_sol.isApprox(rhs_x_sol_sparse, 1e-8));
    ASSERT_TRUE(rhs_y_sol.isApprox(rhs_y_sol_sparse, 1e-8));
    ASSERT_TRUE(rhs_z_sol.isApprox(rhs_z_sol_sparse, 1e-8));
    ASSERT_TRUE(rhs_z_lb_sol.head(data.n_lb).isApprox(rhs_z_lb_sol_sparse.head(data.n_lb), 1e-8));
    ASSERT_TRUE(rhs_z_ub_sol.head(data.n_ub).isApprox(rhs_z_ub_sol_sparse.head(data.n_ub), 1e-8));
    ASSERT_TRUE(rhs_s_sol.isApprox(rhs_s_sol_sparse, 1e-8));
    ASSERT_TRUE(rhs_s_lb_sol.head(data.n_lb).isApprox(rhs_s_lb_sol_sparse.head(data.n_lb), 1e-8));
    ASSERT_TRUE(rhs_s_ub_sol.head(data.n_ub).isApprox(rhs_s_ub_sol_sparse.head(data.n_ub), 1e-8));
}

INSTANTIATE_TEST_SUITE_P(FromFolder, BlocksparseStageKKTTest,
                         ::testing::Values("chain_mass_sqp", "robot_arm_sqp",
                                           "robot_arm_sqp_constr_perm"));