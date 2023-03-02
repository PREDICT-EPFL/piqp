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
    EXPECT_TRUE(data.h.isApprox(data_orig.h, 1e-8));
    EXPECT_TRUE(data.x_lb_scaling.isApprox(data_orig.x_lb_scaling, 1e-8));
    EXPECT_TRUE(data.x_ub_scaling.isApprox(data_orig.x_ub_scaling, 1e-8));
    EXPECT_TRUE(data.x_lb_n.isApprox(data_orig.x_lb_n, 1e-8));
    EXPECT_TRUE(data.x_ub.isApprox(data_orig.x_ub, 1e-8));

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
    EXPECT_TRUE(data.h.isApprox(data_orig.h, 1e-8));
    EXPECT_TRUE(data.x_lb_scaling.isApprox(data_orig.x_lb_scaling, 1e-8));
    EXPECT_TRUE(data.x_ub_scaling.isApprox(data_orig.x_ub_scaling, 1e-8));
    EXPECT_TRUE(data.x_lb_n.isApprox(data_orig.x_lb_n, 1e-8));
    EXPECT_TRUE(data.x_ub.isApprox(data_orig.x_ub, 1e-8));
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
    EXPECT_TRUE(data.h.isApprox(data_orig.h, 1e-8));
    EXPECT_TRUE(data.x_lb_scaling.isApprox(data_orig.x_lb_scaling, 1e-8));
    EXPECT_TRUE(data.x_ub_scaling.isApprox(data_orig.x_ub_scaling, 1e-8));
    EXPECT_TRUE(data.x_lb_n.isApprox(data_orig.x_lb_n, 1e-8));
    EXPECT_TRUE(data.x_ub.isApprox(data_orig.x_ub, 1e-8));

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
    EXPECT_TRUE(data.h.isApprox(data_orig.h, 1e-8));
    EXPECT_TRUE(data.x_lb_scaling.isApprox(data_orig.x_lb_scaling, 1e-8));
    EXPECT_TRUE(data.x_ub_scaling.isApprox(data_orig.x_ub_scaling, 1e-8));
    EXPECT_TRUE(data.x_lb_n.isApprox(data_orig.x_lb_n, 1e-8));
    EXPECT_TRUE(data.x_ub.isApprox(data_orig.x_ub, 1e-8));
}

TEST(RuizEquilibration, DenseSameIneqScaling)
{
    isize dim = 20;
    isize n_eq = 0;
    isize n_ineq = 0;

    dense::Model<T> qp_model_box = rand::dense_strongly_convex_qp<T>(dim, n_eq, n_ineq);
    dense::Data<T> data_box(qp_model_box);

    Mat<T> G(data_box.n_lb + data_box.n_ub, dim); G.setZero();
    Vec<T> h(data_box.n_lb + data_box.n_ub);
    for (isize i = 0; i < data_box.n_lb; i++)
    {
        G(i, data_box.x_lb_idx(i)) = -1;
        h(i) = data_box.x_lb_n(i);
    }
    for (isize i = 0; i < data_box.n_ub; i++)
    {
        G(data_box.n_lb + i, data_box.x_ub_idx(i)) = 1;
        h(data_box.n_lb + i) = data_box.x_ub(i);
    }

    dense::Model<T> qp_model_ineq(qp_model_box.P, qp_model_box.A, G,
                                  qp_model_box.c, qp_model_box.b, h,
                                  Vec<T>::Constant(dim, -std::numeric_limits<T>::infinity()),
                                  Vec<T>::Constant(dim, std::numeric_limits<T>::infinity()));
    dense::Data<T> data_ineq(qp_model_ineq);

    dense::RuizEquilibration<T> preconditioner_box;
    preconditioner_box.init(data_box);
    dense::RuizEquilibration<T> preconditioner_ineq;
    preconditioner_ineq.init(data_ineq);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    preconditioner_box.scale_data(data_box);
    preconditioner_ineq.scale_data(data_ineq);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> z_lb = rand::vector_rand<T>(data_box.n_lb);
    Vec<T> z_ub = rand::vector_rand<T>(data_box.n_ub);
    Vec<T> z(data_box.n_lb + data_box.n_ub); z << z_lb, z_ub;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    z_lb = preconditioner_box.scale_dual_lb(z_lb);
    z_ub = preconditioner_box.scale_dual_ub(z_ub);
    z = preconditioner_ineq.scale_dual_ineq(z);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> z_comb(data_box.n_lb + data_box.n_ub); z_comb << z_lb, z_ub;
    EXPECT_TRUE(z_comb.isApprox(z, 1e-8));
}

TEST(RuizEquilibration, SparseSameIneqScaling)
{
    isize dim = 20;
    isize n_eq = 0;
    isize n_ineq = 0;
    T sparsity_factor = 0.2;

    sparse::Model<T, I> qp_model_box = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);
    sparse::Data<T, I> data_box(qp_model_box);

    SparseMat<T, I> G(data_box.n_lb + data_box.n_ub, dim);
    Vec<T> h(data_box.n_lb + data_box.n_ub);
    for (isize i = 0; i < data_box.n_lb; i++)
    {
        G.insert(i, data_box.x_lb_idx(i)) = -1;
        h(i) = data_box.x_lb_n(i);
    }
    for (isize i = 0; i < data_box.n_ub; i++)
    {
        G.insert(data_box.n_lb + i, data_box.x_ub_idx(i)) = 1;
        h(data_box.n_lb + i) = data_box.x_ub(i);
    }

    sparse::Model<T, I> qp_model_ineq(qp_model_box.P, qp_model_box.A, G,
                                      qp_model_box.c, qp_model_box.b, h,
                                      Vec<T>::Constant(dim, -std::numeric_limits<T>::infinity()),
                                      Vec<T>::Constant(dim, std::numeric_limits<T>::infinity()));
    sparse::Data<T, I> data_ineq(qp_model_ineq);

    sparse::RuizEquilibration<T, I> preconditioner_box;
    preconditioner_box.init(data_box);
    sparse::RuizEquilibration<T, I> preconditioner_ineq;
    preconditioner_ineq.init(data_ineq);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    preconditioner_box.scale_data(data_box);
    preconditioner_ineq.scale_data(data_ineq);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> z_lb = rand::vector_rand<T>(data_box.n_lb);
    Vec<T> z_ub = rand::vector_rand<T>(data_box.n_ub);
    Vec<T> z(data_box.n_lb + data_box.n_ub); z << z_lb, z_ub;

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    z_lb = preconditioner_box.scale_dual_lb(z_lb);
    z_ub = preconditioner_box.scale_dual_ub(z_ub);
    z = preconditioner_ineq.scale_dual_ineq(z);
    PIQP_EIGEN_MALLOC_ALLOWED();

    Vec<T> z_comb(data_box.n_lb + data_box.n_ub); z_comb << z_lb, z_ub;
    EXPECT_TRUE(z_comb.isApprox(z, 1e-8));
}

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
    EXPECT_TRUE(data_sparse.h.isApprox(data_dense.h, 1e-8));
    EXPECT_TRUE(data_sparse.x_lb_scaling.isApprox(data_dense.x_lb_scaling, 1e-8));
    EXPECT_TRUE(data_sparse.x_ub_scaling.isApprox(data_dense.x_ub_scaling, 1e-8));
    EXPECT_TRUE(data_sparse.x_lb_n.isApprox(data_dense.x_lb_n, 1e-8));
    EXPECT_TRUE(data_sparse.x_ub.isApprox(data_dense.x_ub, 1e-8));

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
    EXPECT_TRUE(data_sparse.h.isApprox(data_dense.h, 1e-8));
    EXPECT_TRUE(data_sparse.x_lb_scaling.isApprox(data_dense.x_lb_scaling, 1e-8));
    EXPECT_TRUE(data_sparse.x_ub_scaling.isApprox(data_dense.x_ub_scaling, 1e-8));
    EXPECT_TRUE(data_sparse.x_lb_n.isApprox(data_dense.x_lb_n, 1e-8));
    EXPECT_TRUE(data_sparse.x_ub.isApprox(data_dense.x_ub, 1e-8));
}
