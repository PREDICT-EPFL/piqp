// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/utils.hpp"

#include "gtest/gtest.h"

using namespace piqp;

using T = double;
using I = int;

TEST(KKTCondensed, Init)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.5;

    Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    Data<T, I> data;
    data.P_utri = qp_model.P.triangularView<Eigen::Upper>();
    data.AT = qp_model.A.transpose();
    data.GT = qp_model.G.transpose();
    data.c = qp_model.c;
    data.b = qp_model.b;
    data.h = qp_model.h;

    T rho = 0.9;
    T delta = 1.2;

    KKTCondensed<T, I> kkt(data);
    kkt.init_kkt(rho, delta);

    // assert KKT matrix is upper triangular
    EXPECT_TRUE(kkt.KKT.isApprox(kkt.KKT.triangularView<Eigen::Upper>(), 1e-8));
}

TEST(KKTCondensed, Update)
{
    isize dim = 10;
    isize n_eq = 8;
    isize n_ineq = 9;
    T sparsity_factor = 0.2;

    Model<T, I> qp_model = rand::sparse_strongly_convex_qp<T, I>(dim, n_eq, n_ineq, sparsity_factor);

    Data<T, I> data;
    data.P_utri = qp_model.P.triangularView<Eigen::Upper>();
    data.AT = qp_model.A.transpose();
    data.GT = qp_model.G.transpose();
    data.c = qp_model.c;
    data.b = qp_model.b;
    data.h = qp_model.h;

    T rho = 0.9;
    T delta = 1.2;

    KKTCondensed<T, I> kkt(data);
    kkt.init_kkt(rho, delta);

    rho = 0.8;
    delta = 0.2;
    Vec<T> s(n_ineq); s.setConstant(1);
    Vec<T> z(n_ineq); z.setConstant(1);

    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    kkt.update_kkt(rho, delta, s, z);
    PIQP_EIGEN_MALLOC_ALLOWED();

    // assert KKT matrix is upper triangular
    EXPECT_TRUE(kkt.KKT.isApprox(kkt.KKT.triangularView<Eigen::Upper>(), 1e-8));

    KKTCondensed<T, I> kkt2(data);
    kkt2.init_kkt(rho, delta);

    // assert update was correct, i.e. it's the same as a freshly initialized one
    EXPECT_TRUE(kkt.KKT.isApprox(kkt2.KKT, 1e-8));
}
