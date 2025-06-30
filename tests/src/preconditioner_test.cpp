// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/sparse/preconditioner.hpp"
#include "piqp/utils/random_utils.hpp"

#include "gtest/gtest.h"

using namespace piqp;

using T = double;
using I = int;

TEST(RuizEquilibration, DenseScaleUnscale)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;

    dense::Model<T> qp_model = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);
    dense::Data<T> data_orig(qp_model);
    dense::Data<T> data(qp_model);

    // make sure P_utri has not complete diagonal filled
    data_orig.P_utri(1, 1) = 0;
    data.P_utri(1, 1) = 0;

    dense::RuizEquilibration<T> preconditioner;
    preconditioner.init(data);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    preconditioner.scale_data(data);
    preconditioner.unscale_data(data);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Mat<T> P_utri_orig(dim, dim); P_utri_orig.setZero();
    Mat<T> P_utri(dim, dim); P_utri.setZero();
    P_utri_orig.triangularView<Eigen::Upper>() = data_orig.P_utri.triangularView<Eigen::Upper>();
    P_utri.triangularView<Eigen::Upper>() = data.P_utri.triangularView<Eigen::Upper>();
    EXPECT_TRUE(P_utri.isApprox(P_utri_orig, 1e-8));
    EXPECT_TRUE(data.AT.isApprox(data_orig.AT, 1e-8));
    EXPECT_TRUE(data.GT.isApprox(data_orig.GT, 1e-8));
    EXPECT_TRUE(data.c.isApprox(data_orig.c, 1e-8));
    EXPECT_TRUE(data.b.isApprox(data_orig.b, 1e-8));
    EXPECT_TRUE(data.h_l.isApprox(data_orig.h_l, 1e-8));
    EXPECT_TRUE(data.h_u.isApprox(data_orig.h_u, 1e-8));
    EXPECT_TRUE(data.x_b_scaling.isApprox(data_orig.x_b_scaling, 1e-8));
    EXPECT_TRUE(data.x_l.head(data.n_x_l).isApprox(data_orig.x_l.head(data.n_x_l), 1e-8));
    EXPECT_TRUE(data.x_u.head(data.n_x_u).isApprox(data_orig.x_u.head(data.n_x_u), 1e-8));

    // scale, unscale data using previous scaling
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    preconditioner.scale_data(data, true);
    preconditioner.unscale_data(data);
    PIQP_EIGEN_MALLOC_ALLOWED();

    P_utri_orig.triangularView<Eigen::Upper>() = data_orig.P_utri.triangularView<Eigen::Upper>();
    P_utri.triangularView<Eigen::Upper>() = data.P_utri.triangularView<Eigen::Upper>();
    EXPECT_TRUE(P_utri.isApprox(P_utri_orig, 1e-8));
    EXPECT_TRUE(data.AT.isApprox(data_orig.AT, 1e-8));
    EXPECT_TRUE(data.GT.isApprox(data_orig.GT, 1e-8));
    EXPECT_TRUE(data.c.isApprox(data_orig.c, 1e-8));
    EXPECT_TRUE(data.b.isApprox(data_orig.b, 1e-8));
    EXPECT_TRUE(data.h_l.isApprox(data_orig.h_l, 1e-8));
    EXPECT_TRUE(data.h_u.isApprox(data_orig.h_u, 1e-8));
    EXPECT_TRUE(data.x_b_scaling.isApprox(data_orig.x_b_scaling, 1e-8));
    EXPECT_TRUE(data.x_l.head(data.n_x_l).isApprox(data_orig.x_l.head(data.n_x_l), 1e-8));
    EXPECT_TRUE(data.x_u.head(data.n_x_u).isApprox(data_orig.x_u.head(data.n_x_u), 1e-8));
}

TEST(RuizEquilibration, SparseScaleUnscale)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.2;

    sparse::Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);
    sparse::Data<T, I> data_orig(qp_model);
    sparse::Data<T, I> data(qp_model);

    // make sure P_utri has not complete diagonal filled
    data_orig.P_utri.coeffRef(1, 1) = 0;
    data_orig.P_utri.prune(0.0);
    data.P_utri.coeffRef(1, 1) = 0;
    data.P_utri.prune(0.0);

    sparse::RuizEquilibration<T, I> preconditioner;
    preconditioner.init(data);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    preconditioner.scale_data(data);
    preconditioner.unscale_data(data);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Mat<T> P_utri_orig(dim, dim); P_utri_orig.setZero();
    Mat<T> P_utri(dim, dim); P_utri.setZero();
    EXPECT_TRUE(P_utri.isApprox(P_utri_orig, 1e-8));
    EXPECT_TRUE(data.AT.isApprox(data_orig.AT, 1e-8));
    EXPECT_TRUE(data.GT.isApprox(data_orig.GT, 1e-8));
    EXPECT_TRUE(data.c.isApprox(data_orig.c, 1e-8));
    EXPECT_TRUE(data.b.isApprox(data_orig.b, 1e-8));
    EXPECT_TRUE(data.h_l.isApprox(data_orig.h_l, 1e-8));
    EXPECT_TRUE(data.h_u.isApprox(data_orig.h_u, 1e-8));
    EXPECT_TRUE(data.x_b_scaling.head(data.n_x_l).isApprox(data_orig.x_b_scaling.head(data.n_x_l), 1e-8));
    EXPECT_TRUE(data.x_l.head(data.n_x_l).isApprox(data_orig.x_l.head(data.n_x_l), 1e-8));
    EXPECT_TRUE(data.x_u.head(data.n_x_u).isApprox(data_orig.x_u.head(data.n_x_u), 1e-8));

    // scale, unscale data using previous scaling
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    preconditioner.scale_data(data, true);
    preconditioner.unscale_data(data);
    PIQP_EIGEN_MALLOC_ALLOWED();

    EXPECT_TRUE(P_utri.isApprox(P_utri_orig, 1e-8));
    EXPECT_TRUE(data.AT.isApprox(data_orig.AT, 1e-8));
    EXPECT_TRUE(data.GT.isApprox(data_orig.GT, 1e-8));
    EXPECT_TRUE(data.c.isApprox(data_orig.c, 1e-8));
    EXPECT_TRUE(data.b.isApprox(data_orig.b, 1e-8));
    EXPECT_TRUE(data.h_l.isApprox(data_orig.h_l, 1e-8));
    EXPECT_TRUE(data.h_u.isApprox(data_orig.h_u, 1e-8));
    EXPECT_TRUE(data.x_b_scaling.head(data.n_x_l).isApprox(data_orig.x_b_scaling.head(data.n_x_l), 1e-8));
    EXPECT_TRUE(data.x_l.head(data.n_x_l).isApprox(data_orig.x_l.head(data.n_x_l), 1e-8));
    EXPECT_TRUE(data.x_u.head(data.n_x_u).isApprox(data_orig.x_u.head(data.n_x_u), 1e-8));
}

// TEST(RuizEquilibration, DenseSameIneqScaling)
// {
//     isize dim = 20;
//     isize n_eq = 0;
//     isize n_ineq = 0;
//
//     dense::Model<T> qp_model_box = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);
//     dense::Data<T> data_box(qp_model_box);
//
//     Mat<T> G(dim, dim); G.setIdentity();
//     Vec<T> h(dim); h.setConstant(1);
//
//     dense::Model<T> qp_model_ineq(qp_model_box.P, qp_model_box.c,
//                                   qp_model_box.A, qp_model_box.b, G, h,
//                                   Vec<T>::Constant(dim, -std::numeric_limits<T>::infinity()),
//                                   Vec<T>::Constant(dim, std::numeric_limits<T>::infinity()));
//     dense::Data<T> data_ineq(qp_model_ineq);
//
//     dense::RuizEquilibration<T> preconditioner_box;
//     preconditioner_box.init(data_box);
//     dense::RuizEquilibration<T> preconditioner_ineq;
//     preconditioner_ineq.init(data_ineq);
//
//     PIQP_EIGEN_MALLOC_NOT_ALLOWED();
//     preconditioner_box.scale_data(data_box);
//     preconditioner_ineq.scale_data(data_ineq);
//     PIQP_EIGEN_MALLOC_ALLOWED();
//
//     Vec<T> z_b = rand::vector_rand<T>(dim);
//     Vec<T> z = z_b;
//
//     PIQP_EIGEN_MALLOC_NOT_ALLOWED();
//     z_b = preconditioner_box.unscale_dual_b(z_b);
//     z = preconditioner_ineq.scale_dual_ineq(z);
//     PIQP_EIGEN_MALLOC_ALLOWED();
//
//     EXPECT_TRUE(z_b.isApprox(z, 1e-8));
// }

// TEST(RuizEquilibration, SparseSameIneqScaling)
// {
//     isize dim = 20;
//     isize n_eq = 0;
//     isize n_ineq = 0;
//     T sparsity_factor = 0.2;
//
//     sparse::Model<T, I> qp_model_box = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);
//     sparse::Data<T, I> data_box(qp_model_box);
//
//     SparseMat<T, I> G(dim, dim); G.setIdentity();
//     Vec<T> h(dim); h.setConstant(1);
//
//     sparse::Model<T, I> qp_model_ineq(qp_model_box.P, qp_model_box.c,
//                                       qp_model_box.A, qp_model_box.b, G, h,
//                                       Vec<T>::Constant(dim, -std::numeric_limits<T>::infinity()),
//                                       Vec<T>::Constant(dim, std::numeric_limits<T>::infinity()));
//     sparse::Data<T, I> data_ineq(qp_model_ineq);
//
//     sparse::RuizEquilibration<T, I> preconditioner_box;
//     preconditioner_box.init(data_box);
//     sparse::RuizEquilibration<T, I> preconditioner_ineq;
//     preconditioner_ineq.init(data_ineq);
//
//     PIQP_EIGEN_MALLOC_NOT_ALLOWED();
//     preconditioner_box.scale_data(data_box);
//     preconditioner_ineq.scale_data(data_ineq);
//     PIQP_EIGEN_MALLOC_ALLOWED();
//
//     Vec<T> z_b = rand::vector_rand<T>(dim);
//     Vec<T> z = z_b;
//
//     PIQP_EIGEN_MALLOC_NOT_ALLOWED();
//     z_b = preconditioner_box.unscale_dual_b(z_b);
//     z = preconditioner_ineq.scale_dual_ineq(z);
//     PIQP_EIGEN_MALLOC_ALLOWED();
//
//     EXPECT_TRUE(z_b.isApprox(z, 1e-8));
// }

TEST(RuizEquilibration, DenseSparseCompare)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.2;

    sparse::Model<T, I> qp_model_sparse = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);
    dense::Model<T> qp_model_dense = qp_model_sparse.dense_model();
    sparse::Data<T, I> data_sparse(qp_model_sparse);
    dense::Data<T> data_dense(qp_model_dense);

    // make sure P_utri has not complete diagonal filled
    data_sparse.P_utri.coeffRef(1, 1) = 0;
    data_sparse.P_utri.prune(0.0);
    data_dense.P_utri(1, 1) = 0;

    sparse::RuizEquilibration<T, I> preconditioner_sparse;
    preconditioner_sparse.init(data_sparse);
    dense::RuizEquilibration<T> preconditioner_dense;
    preconditioner_dense.init(data_dense);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    preconditioner_sparse.scale_data(data_sparse);
    preconditioner_dense.scale_data(data_dense);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Mat<T> P_utri_dense(dim, dim); P_utri_dense.setZero();
    P_utri_dense.triangularView<Eigen::Upper>() = data_dense.P_utri.triangularView<Eigen::Upper>();
    EXPECT_TRUE(Mat<T>(data_sparse.P_utri).isApprox(P_utri_dense, 1e-8));
    EXPECT_TRUE(Mat<T>(data_sparse.AT).isApprox(data_dense.AT, 1e-8));
    EXPECT_TRUE(Mat<T>(data_sparse.GT).isApprox(data_dense.GT, 1e-8));
    EXPECT_TRUE(data_sparse.c.isApprox(data_dense.c, 1e-8));
    EXPECT_TRUE(data_sparse.b.isApprox(data_dense.b, 1e-8));
    EXPECT_TRUE(data_sparse.h_l.isApprox(data_dense.h_l, 1e-8));
    EXPECT_TRUE(data_sparse.h_u.isApprox(data_dense.h_u, 1e-8));
    EXPECT_TRUE(data_sparse.x_b_scaling.isApprox(data_dense.x_b_scaling, 1e-8));
    EXPECT_TRUE(data_sparse.x_l.head(data_sparse.n_x_l).isApprox(data_dense.x_l.head(data_dense.n_x_l), 1e-8));
    EXPECT_TRUE(data_sparse.x_u.head(data_sparse.n_x_u).isApprox(data_dense.x_u.head(data_dense.n_x_u), 1e-8));

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    preconditioner_sparse.unscale_data(data_sparse);
    preconditioner_sparse.scale_data(data_sparse, true);
    preconditioner_dense.unscale_data(data_dense);
    preconditioner_dense.scale_data(data_dense, true);
    PIQP_EIGEN_MALLOC_ALLOWED();

    P_utri_dense.triangularView<Eigen::Upper>() = data_dense.P_utri.triangularView<Eigen::Upper>();
    EXPECT_TRUE(Mat<T>(data_sparse.P_utri).isApprox(P_utri_dense, 1e-8));
    EXPECT_TRUE(Mat<T>(data_sparse.AT).isApprox(data_dense.AT, 1e-8));
    EXPECT_TRUE(Mat<T>(data_sparse.GT).isApprox(data_dense.GT, 1e-8));
    EXPECT_TRUE(data_sparse.c.isApprox(data_dense.c, 1e-8));
    EXPECT_TRUE(data_sparse.b.isApprox(data_dense.b, 1e-8));
    EXPECT_TRUE(data_sparse.h_l.isApprox(data_dense.h_l, 1e-8));
    EXPECT_TRUE(data_sparse.h_u.isApprox(data_dense.h_u, 1e-8));
    EXPECT_TRUE(data_sparse.x_b_scaling.isApprox(data_dense.x_b_scaling, 1e-8));
    EXPECT_TRUE(data_sparse.x_l.head(data_sparse.n_x_l).isApprox(data_dense.x_l.head(data_dense.n_x_l), 1e-8));
    EXPECT_TRUE(data_sparse.x_u.head(data_sparse.n_x_u).isApprox(data_dense.x_u.head(data_dense.n_x_u), 1e-8));
}
