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

template<typename KKT1, typename KKT2>
void test_solve_multiply(Data<T, I>& data, Settings<T> settings1, Settings<T> settings2, KKT1& kkt1, KKT2& kkt2)
{
    Variables<T> rhs;
    rhs.x = rand::vector_rand<T>(data.n);
    rhs.y = rand::vector_rand<T>(data.p);
    rhs.z_l = rand::vector_rand<T>(data.m);
    rhs.z_u = rand::vector_rand<T>(data.m);
    rhs.z_bl = rand::vector_rand<T>(data.n);
    rhs.z_bu = rand::vector_rand<T>(data.n);
    rhs.s_l = rand::vector_rand<T>(data.m);
    rhs.s_u = rand::vector_rand<T>(data.m);
    rhs.s_bl = rand::vector_rand<T>(data.n);
    rhs.s_bu = rand::vector_rand<T>(data.n);

    Variables<T> lhs_1;
    lhs_1.resize(data.n, data.p, data.m);

    Variables<T> lhs_2;
    lhs_2.resize(data.n, data.p, data.m);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt1.solve(data, settings1, rhs, lhs_1);
    kkt2.solve(data, settings2, rhs, lhs_2);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(lhs_1.x.isApprox(lhs_2.x, 1e-8));
    ASSERT_TRUE(lhs_1.y.isApprox(lhs_2.y, 1e-8));
    ASSERT_TRUE(lhs_1.z_bl.head(data.n_x_l).isApprox(lhs_2.z_bl.head(data.n_x_l), 1e-8));
    ASSERT_TRUE(lhs_1.z_bu.head(data.n_x_u).isApprox(lhs_2.z_bu.head(data.n_x_u), 1e-8));
    ASSERT_TRUE(lhs_1.s_bl.head(data.n_x_l).isApprox(lhs_2.s_bl.head(data.n_x_l), 1e-8));
    ASSERT_TRUE(lhs_1.s_bu.head(data.n_x_u).isApprox(lhs_2.s_bu.head(data.n_x_u), 1e-8));
    for (isize i = 0; i < data.n_h_l; i++)
    {
        Eigen::Index idx = data.h_l_idx(i);
        ASSERT_NEAR(lhs_1.z_l(idx), lhs_2.z_l(idx), 1e-8);
        ASSERT_NEAR(lhs_1.s_l(idx), lhs_2.s_l(idx), 1e-8);
    }
    for (isize i = 0; i < data.n_h_u; i++)
    {
        Eigen::Index idx = data.h_u_idx(i);
        ASSERT_NEAR(lhs_1.z_u(idx), lhs_2.z_u(idx), 1e-8);
        ASSERT_NEAR(lhs_1.s_u(idx), lhs_2.s_u(idx), 1e-8);
    }

    Variables<T> rhs_sol_1;
    rhs_sol_1.resize(data.n, data.p, data.m);

    Variables<T> rhs_sol_2;
    rhs_sol_2.resize(data.n, data.p, data.m);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt1.mul(data, lhs_1, rhs_sol_1);
    kkt2.mul(data, lhs_2, rhs_sol_2);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(rhs_sol_1.x.isApprox(rhs_sol_2.x, 1e-8));
    ASSERT_TRUE(rhs_sol_1.y.isApprox(rhs_sol_2.y, 1e-8));
    ASSERT_TRUE(rhs_sol_1.z_bl.head(data.n_x_l).isApprox(rhs_sol_2.z_bl.head(data.n_x_l), 1e-8));
    ASSERT_TRUE(rhs_sol_1.z_bu.head(data.n_x_u).isApprox(rhs_sol_2.z_bu.head(data.n_x_u), 1e-8));
    ASSERT_TRUE(rhs_sol_1.s_bl.head(data.n_x_l).isApprox(rhs_sol_2.s_bl.head(data.n_x_l), 1e-8));
    ASSERT_TRUE(rhs_sol_1.s_bu.head(data.n_x_u).isApprox(rhs_sol_2.s_bu.head(data.n_x_u), 1e-8));
    for (isize i = 0; i < data.n_h_l; i++)
    {
        Eigen::Index idx = data.h_l_idx(i);
        ASSERT_NEAR(rhs_sol_1.z_l(idx), rhs_sol_2.z_l(idx), 1e-8);
        ASSERT_NEAR(rhs_sol_1.s_l(idx), rhs_sol_2.s_l(idx), 1e-8);
    }
    for (isize i = 0; i < data.n_h_u; i++)
    {
        Eigen::Index idx = data.h_u_idx(i);
        ASSERT_NEAR(rhs_sol_1.z_u(idx), rhs_sol_2.z_u(idx), 1e-8);
        ASSERT_NEAR(rhs_sol_1.s_u(idx), rhs_sol_2.s_u(idx), 1e-8);
    }
}

TEST(BlocksparseStageKKTTest, UpdateData)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.2;

    Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);
    qp_model.x_l.setConstant(-std::numeric_limits<T>::infinity());
    qp_model.x_u.setConstant(std::numeric_limits<T>::infinity());
    Data<T, I> data(qp_model);

    Settings<T> settings_multistage;
    settings_multistage.kkt_solver = KKTSolver::sparse_multistage;

    Settings<T> settings_sparse;
    settings_sparse.kkt_solver = KKTSolver::sparse_ldlt;

    T rho = 0.9;
    T delta = 1.2;
    Variables<T> scaling; scaling.resize(dim, n_eq, n_ineq);
    scaling.s_l.setConstant(1);
    scaling.s_u.setConstant(1);
    scaling.s_bl.setConstant(1);
    scaling.s_bu.setConstant(1);
    scaling.z_l.setConstant(1);
    scaling.z_u.setConstant(1);
    scaling.z_bl.setConstant(1);
    scaling.z_bu.setConstant(1);

    KKTSystem<T, I, PIQP_SPARSE> kkt_multistage;
    kkt_multistage.init(data, settings_multistage);
    KKTSystem<T, I, PIQP_SPARSE> kkt_sparse;
    kkt_sparse.init(data, settings_sparse);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_multistage.update_scalings_and_factor(data, settings_multistage, false, rho, delta, scaling);
    kkt_sparse.update_scalings_and_factor(data, settings_sparse, false, rho, delta, scaling);
    PIQP_EIGEN_MALLOC_ALLOWED();

    test_solve_multiply(data, settings_multistage, settings_sparse, kkt_multistage, kkt_sparse);

    // update data
    Eigen::Map<Vec<T>>(data.P_utri.valuePtr(), data.P_utri.nonZeros()) = rand::vector_rand<T>(data.P_utri.nonZeros());
    // ensure P_utri is positive semi-definite
    Mat<T> P_dense = data.P_utri.toDense();
    Vec<T> eig_h = P_dense.template selfadjointView<Eigen::Upper>().eigenvalues();
    T min_eig = eig_h.minCoeff();
    if (min_eig < 0) {
        for (int k = 0; k < dim; k++)
        {
            for (SparseMat<T, I>::InnerIterator it(data.P_utri, k); it; ++it)
            {
                if (it.row() == it.col()) {
                    it.valueRef() -= min_eig;
                }
            }
        }
    }
    Eigen::Map<Vec<T>>(data.AT.valuePtr(), data.AT.nonZeros()) = rand::vector_rand<T>(data.AT.nonZeros());
    Eigen::Map<Vec<T>>(data.GT.valuePtr(), data.GT.nonZeros()) = rand::vector_rand<T>(data.GT.nonZeros());
    int update_options = KKTUpdateOptions::KKT_UPDATE_P | KKTUpdateOptions::KKT_UPDATE_A | KKTUpdateOptions::KKT_UPDATE_G;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_multistage.update_data(data, update_options);
    kkt_multistage.update_scalings_and_factor(data, settings_multistage, false, rho, delta, scaling);
    kkt_sparse.update_data(data, update_options);
    kkt_sparse.update_scalings_and_factor(data, settings_sparse, false, rho, delta, scaling);
    PIQP_EIGEN_MALLOC_ALLOWED();

    test_solve_multiply(data, settings_multistage, settings_sparse, kkt_multistage, kkt_sparse);
}

TEST_P(BlocksparseStageKKTTest, FactorizeSolveSQP)
{
    std::string path = "data/" + GetParam() + ".mat";
    Model<T, I> model = load_sparse_model<T, I>(path);
    Data<T, I> data(model);

    Settings<T> settings_multistage;
    settings_multistage.kkt_solver = KKTSolver::sparse_multistage;

    Settings<T> settings_sparse;
    settings_sparse.kkt_solver = KKTSolver::sparse_ldlt;

    T rho = 0.9;
    T delta = 1.2;
    Variables<T> scaling; scaling.resize(data.n, data.p, data.m);
    scaling.s_l.setConstant(1);
    scaling.s_u.setConstant(1);
    scaling.s_bl.setConstant(1);
    scaling.s_bu.setConstant(1);
    scaling.z_l.setConstant(1);
    scaling.z_u.setConstant(1);
    scaling.z_bl.setConstant(1);
    scaling.z_bu.setConstant(1);

    KKTSystem<T, I, PIQP_SPARSE> kkt_multistage;
    kkt_multistage.init(data, settings_multistage);
    KKTSystem<T, I, PIQP_SPARSE> kkt_sparse;
    kkt_sparse.init(data, settings_sparse);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_multistage.update_scalings_and_factor(data, settings_multistage, false, rho, delta, scaling);
    kkt_sparse.update_scalings_and_factor(data, settings_sparse, false, rho, delta, scaling);
    PIQP_EIGEN_MALLOC_ALLOWED();

    test_solve_multiply(data, settings_multistage, settings_sparse, kkt_multistage, kkt_sparse);
}

INSTANTIATE_TEST_SUITE_P(FromFolder, BlocksparseStageKKTTest,
                         ::testing::Values("small_sparse_dual_inf", "small_dense", "scenario_mpc_small",
                                           "scenario_mpc", "chain_mass_sqp", "robot_arm_sqp",
                                           "robot_arm_sqp_constr_perm", "robot_arm_sqp_no_global"));
