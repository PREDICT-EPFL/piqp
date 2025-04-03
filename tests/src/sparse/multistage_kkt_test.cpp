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
void test_solve_multiply(Data<T, I>& data, KKT1& kkt1, KKT2& kkt2)
{
    Variables<T> rhs;
    rhs.x = rand::vector_rand<T>(data.n);
    rhs.y = rand::vector_rand<T>(data.p);
    rhs.z = rand::vector_rand<T>(data.m);
    rhs.z_lb = rand::vector_rand<T>(data.n);
    rhs.z_ub = rand::vector_rand<T>(data.n);
    rhs.s = rand::vector_rand<T>(data.m);
    rhs.s_lb = rand::vector_rand<T>(data.n);
    rhs.s_ub = rand::vector_rand<T>(data.n);

    Variables<T> lhs_1;
    lhs_1.resize(data.n, data.p, data.m);

    Variables<T> lhs_2;
    lhs_2.resize(data.n, data.p, data.m);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt1.solve(rhs, lhs_1);
    kkt2.solve(rhs, lhs_2);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(lhs_1.x.isApprox(lhs_2.x, 1e-8));
    ASSERT_TRUE(lhs_1.y.isApprox(lhs_2.y, 1e-8));
    ASSERT_TRUE(lhs_1.z.isApprox(lhs_2.z, 1e-8));
    ASSERT_TRUE(lhs_1.z_lb.head(data.n_lb).isApprox(lhs_2.z_lb.head(data.n_lb), 1e-8));
    ASSERT_TRUE(lhs_1.z_ub.head(data.n_ub).isApprox(lhs_2.z_ub.head(data.n_ub), 1e-8));
    ASSERT_TRUE(lhs_1.s.isApprox(lhs_2.s, 1e-8));
    ASSERT_TRUE(lhs_1.s_lb.head(data.n_lb).isApprox(lhs_2.s_lb.head(data.n_lb), 1e-8));
    ASSERT_TRUE(lhs_1.s_ub.head(data.n_ub).isApprox(lhs_2.s_ub.head(data.n_ub), 1e-8));

    Variables<T> rhs_sol_1;
    rhs_sol_1.resize(data.n, data.p, data.m);

    Variables<T> rhs_sol_2;
    rhs_sol_2.resize(data.n, data.p, data.m);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt1.mul(lhs_1, rhs_sol_1);
    kkt1.mul(lhs_2, rhs_sol_2);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(rhs_sol_1.x.isApprox(rhs_sol_2.x, 1e-8));
    ASSERT_TRUE(rhs_sol_1.y.isApprox(rhs_sol_2.y, 1e-8));
    ASSERT_TRUE(rhs_sol_1.z.isApprox(rhs_sol_2.z, 1e-8));
    ASSERT_TRUE(rhs_sol_1.z_lb.head(data.n_lb).isApprox(rhs_sol_2.z_lb.head(data.n_lb), 1e-8));
    ASSERT_TRUE(rhs_sol_1.z_ub.head(data.n_ub).isApprox(rhs_sol_2.z_ub.head(data.n_ub), 1e-8));
    ASSERT_TRUE(rhs_sol_1.s.isApprox(rhs_sol_2.s, 1e-8));
    ASSERT_TRUE(rhs_sol_1.s_lb.head(data.n_lb).isApprox(rhs_sol_2.s_lb.head(data.n_lb), 1e-8));
    ASSERT_TRUE(rhs_sol_1.s_ub.head(data.n_ub).isApprox(rhs_sol_2.s_ub.head(data.n_ub), 1e-8));
}

TEST(BlocksparseStageKKTTest, UpdateData)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.2;

    Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);
    qp_model.x_lb.setConstant(-std::numeric_limits<T>::infinity());
    qp_model.x_ub.setConstant(std::numeric_limits<T>::infinity());
    Data<T, I> data(qp_model);

    Settings<T> settings_multistage;
    settings_multistage.kkt_solver = KKTSolver::sparse_multistage;

    Settings<T> settings_sparse;
    settings_sparse.kkt_solver = KKTSolver::sparse_ldlt;

    T rho = 0.9;
    T delta = 1.2;
    Variables<T> scaling; scaling.resize(dim, n_eq, n_ineq);
    scaling.s.setConstant(1);
    scaling.s_lb.setConstant(1);
    scaling.s_ub.setConstant(1);
    scaling.z.setConstant(1);
    scaling.z_lb.setConstant(1);
    scaling.z_ub.setConstant(1);

    KKTSystem<T, I, PIQP_SPARSE> kkt_multistage(data, settings_multistage);
    kkt_multistage.init();
    KKTSystem<T, I, PIQP_SPARSE> kkt_sparse(data, settings_sparse);
    kkt_sparse.init();
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_multistage.update_scalings_and_factor(false, rho, delta, scaling);
    kkt_sparse.update_scalings_and_factor(false, rho, delta, scaling);
    PIQP_EIGEN_MALLOC_ALLOWED();

    test_solve_multiply(data, kkt_multistage, kkt_sparse);

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
    kkt_multistage.update_data(update_options);
    kkt_multistage.update_scalings_and_factor(false, rho, delta, scaling);
    kkt_sparse.update_data(update_options);
    kkt_sparse.update_scalings_and_factor(false, rho, delta, scaling);
    PIQP_EIGEN_MALLOC_ALLOWED();

    test_solve_multiply(data, kkt_multistage, kkt_sparse);
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
    scaling.s.setConstant(1);
    scaling.s_lb.setConstant(1);
    scaling.s_ub.setConstant(1);
    scaling.z.setConstant(1);
    scaling.z_lb.setConstant(1);
    scaling.z_ub.setConstant(1);

    KKTSystem<T, I, PIQP_SPARSE> kkt_multistage(data, settings_multistage);
    kkt_multistage.init();
    KKTSystem<T, I, PIQP_SPARSE> kkt_sparse(data, settings_sparse);
    kkt_sparse.init();
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_multistage.update_scalings_and_factor(false, rho, delta, scaling);
    kkt_sparse.update_scalings_and_factor(false, rho, delta, scaling);
    PIQP_EIGEN_MALLOC_ALLOWED();

    test_solve_multiply(data, kkt_multistage, kkt_sparse);
}

INSTANTIATE_TEST_SUITE_P(FromFolder, BlocksparseStageKKTTest,
                         ::testing::Values("small_sparse_dual_inf", "small_dense", "scenario_mpc_small",
                                           "scenario_mpc", "chain_mass_sqp", "robot_arm_sqp",
                                           "robot_arm_sqp_constr_perm", "robot_arm_sqp_no_global"));
