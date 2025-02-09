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
    Vec<T> rhs_x = rand::vector_rand<T>(data.n);
    Vec<T> rhs_y = rand::vector_rand<T>(data.p);
    Vec<T> rhs_z = rand::vector_rand<T>(data.m);
    Vec<T> rhs_z_lb = rand::vector_rand<T>(data.n);
    Vec<T> rhs_z_ub = rand::vector_rand<T>(data.n);
    Vec<T> rhs_s = rand::vector_rand<T>(data.m);
    Vec<T> rhs_s_lb = rand::vector_rand<T>(data.n);
    Vec<T> rhs_s_ub = rand::vector_rand<T>(data.n);

    Vec<T> delta_x_1(data.n);
    Vec<T> delta_y_1(data.p);
    Vec<T> delta_z_1(data.m);
    Vec<T> delta_z_lb_1(data.n);
    Vec<T> delta_z_ub_1(data.n);
    Vec<T> delta_s_1(data.m);
    Vec<T> delta_s_lb_1(data.n);
    Vec<T> delta_s_ub_1(data.n);

    Vec<T> delta_x_2(data.n);
    Vec<T> delta_y_2(data.p);
    Vec<T> delta_z_2(data.m);
    Vec<T> delta_z_lb_2(data.n);
    Vec<T> delta_z_ub_2(data.n);
    Vec<T> delta_s_2(data.m);
    Vec<T> delta_s_lb_2(data.n);
    Vec<T> delta_s_ub_2(data.n);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt1.solve(rhs_x, rhs_y, rhs_z, rhs_z_lb, rhs_z_ub, rhs_s, rhs_s_lb, rhs_s_ub,
               delta_x_1, delta_y_1, delta_z_1, delta_z_lb_1, delta_z_ub_1, delta_s_1, delta_s_lb_1, delta_s_ub_1);
    kkt2.solve(rhs_x, rhs_y, rhs_z, rhs_z_lb, rhs_z_ub, rhs_s, rhs_s_lb, rhs_s_ub,
               delta_x_2, delta_y_2, delta_z_2, delta_z_lb_2, delta_z_ub_2, delta_s_2, delta_s_lb_2, delta_s_ub_2);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(delta_x_1.isApprox(delta_x_2, 1e-5));
    ASSERT_TRUE(delta_y_1.isApprox(delta_y_2, 1e-5));
    ASSERT_TRUE(delta_z_1.isApprox(delta_z_2, 1e-5));
    ASSERT_TRUE(delta_z_lb_1.head(data.n_lb).isApprox(delta_z_lb_2.head(data.n_lb), 1e-5));
    ASSERT_TRUE(delta_z_ub_1.head(data.n_ub).isApprox(delta_z_ub_2.head(data.n_ub), 1e-5));
    ASSERT_TRUE(delta_s_1.isApprox(delta_s_2, 1e-5));
    ASSERT_TRUE(delta_s_lb_1.head(data.n_lb).isApprox(delta_s_lb_2.head(data.n_lb), 1e-5));
    ASSERT_TRUE(delta_s_ub_1.head(data.n_ub).isApprox(delta_s_ub_2.head(data.n_ub), 1e-5));

    Vec<T> rhs_x_sol_1(data.n);
    Vec<T> rhs_y_sol_1(data.p);
    Vec<T> rhs_z_sol_1(data.m);
    Vec<T> rhs_z_lb_sol_1(data.n);
    Vec<T> rhs_z_ub_sol_1(data.n);
    Vec<T> rhs_s_sol_1(data.m);
    Vec<T> rhs_s_lb_sol_1(data.n);
    Vec<T> rhs_s_ub_sol_1(data.n);

    Vec<T> rhs_x_sol_2(data.n);
    Vec<T> rhs_y_sol_2(data.p);
    Vec<T> rhs_z_sol_2(data.m);
    Vec<T> rhs_z_lb_sol_2(data.n);
    Vec<T> rhs_z_ub_sol_2(data.n);
    Vec<T> rhs_s_sol_2(data.m);
    Vec<T> rhs_s_lb_sol_2(data.n);
    Vec<T> rhs_s_ub_sol_2(data.n);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt1.multiply(delta_x_1, delta_y_1, delta_z_1, delta_z_lb_1, delta_z_ub_1, delta_s_1, delta_s_lb_1, delta_s_ub_1,
                  rhs_x_sol_1, rhs_y_sol_1, rhs_z_sol_1,
                  rhs_z_lb_sol_1, rhs_z_ub_sol_1, rhs_s_sol_1, rhs_s_lb_sol_1, rhs_s_ub_sol_1);
    kkt1.multiply(delta_x_1, delta_y_1, delta_z_1, delta_z_lb_1, delta_z_ub_1, delta_s_1, delta_s_lb_1, delta_s_ub_1,
                  rhs_x_sol_2, rhs_y_sol_2, rhs_z_sol_2,
                  rhs_z_lb_sol_2, rhs_z_ub_sol_2, rhs_s_sol_2, rhs_s_lb_sol_2, rhs_s_ub_sol_2);
    PIQP_EIGEN_MALLOC_ALLOWED();

    ASSERT_TRUE(rhs_x_sol_1.isApprox(rhs_x_sol_2, 1e-8));
    ASSERT_TRUE(rhs_y_sol_1.isApprox(rhs_y_sol_2, 1e-8));
    ASSERT_TRUE(rhs_z_sol_1.isApprox(rhs_z_sol_2, 1e-8));
    ASSERT_TRUE(rhs_z_lb_sol_1.head(data.n_lb).isApprox(rhs_z_lb_sol_2.head(data.n_lb), 1e-8));
    ASSERT_TRUE(rhs_z_ub_sol_1.head(data.n_ub).isApprox(rhs_z_ub_sol_2.head(data.n_ub), 1e-8));
    ASSERT_TRUE(rhs_s_sol_1.isApprox(rhs_s_sol_2, 1e-8));
    ASSERT_TRUE(rhs_s_lb_sol_1.head(data.n_lb).isApprox(rhs_s_lb_sol_2.head(data.n_lb), 1e-8));
    ASSERT_TRUE(rhs_s_ub_sol_1.head(data.n_ub).isApprox(rhs_s_ub_sol_2.head(data.n_ub), 1e-8));
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
    Settings<T> settings;

    T rho = 0.9;
    T delta = 1.2;
    Vec<T> s = rand::vector_rand_strictly_positive<T>(data.m);
    Vec<T> s_lb = rand::vector_rand_strictly_positive<T>(data.n_lb);
    Vec<T> s_ub = rand::vector_rand_strictly_positive<T>(data.n_ub);
    Vec<T> z = rand::vector_rand_strictly_positive<T>(data.m);
    Vec<T> z_lb = rand::vector_rand_strictly_positive<T>(data.n_lb);
    Vec<T> z_ub = rand::vector_rand_strictly_positive<T>(data.n_ub);

    BlocksparseStageKKT<T, I> kkt_tridiag(data, settings);
    KKT<T, I> kkt_sparse(data, settings);
//    KKT<T, I, KKTMode::KKT_ALL_ELIMINATED, NaturalOrdering<I>> kkt_sparse(data, settings);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_tridiag.update_scalings_and_factor(false, rho, delta, s, s_lb, s_ub, z, z_lb, z_ub);
    kkt_sparse.update_scalings_and_factor(false, rho, delta, s, s_lb, s_ub, z, z_lb, z_ub);
    PIQP_EIGEN_MALLOC_ALLOWED();

    test_solve_multiply(data, kkt_tridiag, kkt_sparse);

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
    kkt_tridiag.update_data(update_options);
    kkt_tridiag.update_scalings_and_factor(false, rho, delta, s, s_lb, s_ub, z, z_lb, z_ub);
    kkt_sparse.update_data(update_options);
    kkt_sparse.update_scalings_and_factor(false, rho, delta, s, s_lb, s_ub, z, z_lb, z_ub);
    PIQP_EIGEN_MALLOC_ALLOWED();

    test_solve_multiply(data, kkt_tridiag, kkt_sparse);
}

TEST_P(BlocksparseStageKKTTest, FactorizeSolveSQP)
{
    std::string path = "data/" + GetParam() + ".mat";
    Model<T, I> model = load_sparse_model<T, I>(path);
    Data<T, I> data(model);
    Settings<T> settings;

    T rho = 0.9;
    T delta = 1.2;
    Vec<T> s = rand::vector_rand_strictly_positive<T>(data.m);
    Vec<T> s_lb = rand::vector_rand_strictly_positive<T>(data.n_lb);
    Vec<T> s_ub = rand::vector_rand_strictly_positive<T>(data.n_ub);
    Vec<T> z = rand::vector_rand_strictly_positive<T>(data.m);
    Vec<T> z_lb = rand::vector_rand_strictly_positive<T>(data.n_lb);
    Vec<T> z_ub = rand::vector_rand_strictly_positive<T>(data.n_ub);

    BlocksparseStageKKT<T, I> kkt_tridiag(data, settings);
    KKT<T, I> kkt_sparse(data, settings);
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt_tridiag.update_scalings_and_factor(false, rho, delta, s, s_lb, s_ub, z, z_lb, z_ub);
    kkt_sparse.update_scalings_and_factor(false, rho, delta, s, s_lb, s_ub, z, z_lb, z_ub);
    PIQP_EIGEN_MALLOC_ALLOWED();

    test_solve_multiply(data, kkt_tridiag, kkt_sparse);
}

INSTANTIATE_TEST_SUITE_P(FromFolder, BlocksparseStageKKTTest,
                         ::testing::Values("small_dense", "chain_mass_sqp", "robot_arm_sqp",
                                           "robot_arm_sqp_constr_perm", "robot_arm_sqp_no_global"));
