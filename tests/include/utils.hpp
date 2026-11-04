// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_TESTS_UTILS_HPP
#define PIQP_TESTS_UTILS_HPP

#include "piqp/piqp.hpp"
#include "gtest/gtest.h"

template<typename T, unsigned int Mode>
void assert_dense_triangular_equal(piqp::Mat<T>& A, piqp::Mat<T>& B)
{
    ASSERT_EQ(A.rows(), B.rows());
    ASSERT_EQ(A.cols(), B.cols());

    piqp::Mat<T> diff(A.rows(), A.cols());
    diff.setZero();

    diff = A.template triangularView<Mode>();
    diff -= B.template triangularView<Mode>();
    ASSERT_TRUE(diff.cwiseAbs().maxCoeff() == 0);
}

template<typename T, typename I>
void assert_sparse_matrices_equal(piqp::SparseMat<T, I>& A, piqp::SparseMat<T, I>& B)
{
    ASSERT_EQ(A.rows(), B.rows());
    ASSERT_EQ(A.cols(), B.cols());
    ASSERT_EQ(A.nonZeros(), B.nonZeros());
    ASSERT_EQ(Eigen::Map<piqp::Vec<I>>(A.outerIndexPtr(), A.outerSize() + 1),
              Eigen::Map<piqp::Vec<I>>(B.outerIndexPtr(), B.outerSize() + 1));
    ASSERT_EQ(Eigen::Map<piqp::Vec<I>>(A.innerIndexPtr(), A.nonZeros()),
              Eigen::Map<piqp::Vec<I>>(B.innerIndexPtr(), B.nonZeros()));
    ASSERT_EQ(Eigen::Map<piqp::Vec<T>>(A.valuePtr(), A.nonZeros()),
              Eigen::Map<piqp::Vec<T>>(B.valuePtr(), B.nonZeros()));
}

// checks that the result contains a normalized primal infeasibility certificate (y, z_l, z_u, z_bl, z_bu)
template<typename T, typename I>
void assert_primal_infeasibility_certificate(const piqp::sparse::Model<T, I>& model, const piqp::Result<T>& result, T tol)
{
    const piqp::Vec<T>& y = result.y;
    const piqp::Vec<T>& z_l = result.z_l;
    const piqp::Vec<T>& z_u = result.z_u;
    const piqp::Vec<T>& z_bl = result.z_bl;
    const piqp::Vec<T>& z_bu = result.z_bu;

    T norm = y.template lpNorm<Eigen::Infinity>();
    norm = (std::max)(norm, z_l.template lpNorm<Eigen::Infinity>());
    norm = (std::max)(norm, z_u.template lpNorm<Eigen::Infinity>());
    norm = (std::max)(norm, z_bl.template lpNorm<Eigen::Infinity>());
    norm = (std::max)(norm, z_bu.template lpNorm<Eigen::Infinity>());
    ASSERT_NEAR(norm, 1, tol);

    T farkas_obj = model.b.dot(y);
    for (Eigen::Index i = 0; i < model.h_l.rows(); i++)
    {
        ASSERT_GE(z_l(i), -tol);
        if (model.h_l(i) > -PIQP_INF) {
            farkas_obj -= model.h_l(i) * z_l(i);
        } else {
            ASSERT_EQ(z_l(i), 0);
        }
    }
    for (Eigen::Index i = 0; i < model.h_u.rows(); i++)
    {
        ASSERT_GE(z_u(i), -tol);
        if (model.h_u(i) < PIQP_INF) {
            farkas_obj += model.h_u(i) * z_u(i);
        } else {
            ASSERT_EQ(z_u(i), 0);
        }
    }
    for (Eigen::Index i = 0; i < model.x_l.rows(); i++)
    {
        ASSERT_GE(z_bl(i), -tol);
        if (model.x_l(i) > -PIQP_INF) {
            farkas_obj -= model.x_l(i) * z_bl(i);
        } else {
            ASSERT_EQ(z_bl(i), 0);
        }
    }
    for (Eigen::Index i = 0; i < model.x_u.rows(); i++)
    {
        ASSERT_GE(z_bu(i), -tol);
        if (model.x_u(i) < PIQP_INF) {
            farkas_obj += model.x_u(i) * z_bu(i);
        } else {
            ASSERT_EQ(z_bu(i), 0);
        }
    }
    ASSERT_LT(farkas_obj, -tol);

    piqp::Vec<T> stationarity = model.A.transpose() * y + model.G.transpose() * (z_u - z_l) - z_bl + z_bu;
    ASSERT_LE(stationarity.template lpNorm<Eigen::Infinity>(), tol);
}

// checks that the result contains a normalized dual infeasibility certificate x
template<typename T, typename I>
void assert_dual_infeasibility_certificate(const piqp::sparse::Model<T, I>& model, const piqp::Result<T>& result, T tol)
{
    const piqp::Vec<T>& x = result.x;

    ASSERT_NEAR(x.template lpNorm<Eigen::Infinity>(), 1, tol);
    ASSERT_LT(model.c.dot(x), -tol);

    piqp::Vec<T> Px = model.P.template selfadjointView<Eigen::Upper>() * x;
    ASSERT_LE(Px.template lpNorm<Eigen::Infinity>(), tol);
    piqp::Vec<T> Ax = model.A * x;
    ASSERT_LE(Ax.template lpNorm<Eigen::Infinity>(), tol);

    piqp::Vec<T> Gx = model.G * x;
    for (Eigen::Index i = 0; i < Gx.rows(); i++)
    {
        if (model.h_l(i) > -PIQP_INF) ASSERT_GE(Gx(i), -tol);
        if (model.h_u(i) < PIQP_INF) ASSERT_LE(Gx(i), tol);
    }
    for (Eigen::Index i = 0; i < x.rows(); i++)
    {
        if (model.x_l(i) > -PIQP_INF) ASSERT_GE(x(i), -tol);
        if (model.x_u(i) < PIQP_INF) ASSERT_LE(x(i), tol);
    }
}

#endif //PIQP_TESTS_UTILS_HPP
