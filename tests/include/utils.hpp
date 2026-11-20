// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_TESTS_UTILS_HPP
#define PIQP_TESTS_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "piqp/piqp.hpp"
#include "piqp/utils/random_utils.hpp"
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

// checks that A and B have the same sparsity pattern, ignoring the values
template<typename T, typename I>
void expect_same_pattern(const piqp::SparseMat<T, I>& A, piqp::SparseMat<T, I> B)
{
    B.makeCompressed();
    ASSERT_EQ(A.rows(), B.rows());
    ASSERT_EQ(A.cols(), B.cols());
    ASSERT_EQ(A.nonZeros(), B.nonZeros());
    EXPECT_TRUE(std::equal(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1, B.outerIndexPtr()));
    EXPECT_TRUE(std::equal(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros(), B.innerIndexPtr()));
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

// Exact (bitwise) comparison of two results, excluding the timings.
template<typename T>
void expect_results_identical(const piqp::Result<T>& r1, const piqp::Result<T>& r2)
{
    auto same = [](const piqp::Vec<T>& v1, const piqp::Vec<T>& v2)
    {
        return v1.size() == v2.size() && (v1.array() == v2.array()).all();
    };
    EXPECT_TRUE(same(r1.x, r2.x)) << "x";
    EXPECT_TRUE(same(r1.y, r2.y)) << "y";
    EXPECT_TRUE(same(r1.z_l, r2.z_l)) << "z_l";
    EXPECT_TRUE(same(r1.z_u, r2.z_u)) << "z_u";
    EXPECT_TRUE(same(r1.z_bl, r2.z_bl)) << "z_bl";
    EXPECT_TRUE(same(r1.z_bu, r2.z_bu)) << "z_bu";
    EXPECT_TRUE(same(r1.s_l, r2.s_l)) << "s_l";
    EXPECT_TRUE(same(r1.s_u, r2.s_u)) << "s_u";
    EXPECT_TRUE(same(r1.s_bl, r2.s_bl)) << "s_bl";
    EXPECT_TRUE(same(r1.s_bu, r2.s_bu)) << "s_bu";

    const piqp::Info<T>& i1 = r1.info;
    const piqp::Info<T>& i2 = r2.info;
    EXPECT_EQ(i1.status, i2.status);
    EXPECT_EQ(i1.iter, i2.iter);
    EXPECT_EQ(i1.rho, i2.rho);
    EXPECT_EQ(i1.delta, i2.delta);
    EXPECT_EQ(i1.mu, i2.mu);
    EXPECT_EQ(i1.sigma, i2.sigma);
    EXPECT_EQ(i1.primal_step, i2.primal_step);
    EXPECT_EQ(i1.dual_step, i2.dual_step);
    EXPECT_EQ(i1.primal_res, i2.primal_res);
    EXPECT_EQ(i1.primal_res_rel, i2.primal_res_rel);
    EXPECT_EQ(i1.dual_res, i2.dual_res);
    EXPECT_EQ(i1.dual_res_rel, i2.dual_res_rel);
    EXPECT_EQ(i1.primal_res_reg, i2.primal_res_reg);
    EXPECT_EQ(i1.primal_res_reg_rel, i2.primal_res_reg_rel);
    EXPECT_EQ(i1.dual_res_reg, i2.dual_res_reg);
    EXPECT_EQ(i1.dual_res_reg_rel, i2.dual_res_reg_rel);
    EXPECT_EQ(i1.primal_prox_inf, i2.primal_prox_inf);
    EXPECT_EQ(i1.dual_prox_inf, i2.dual_prox_inf);
    EXPECT_EQ(i1.prev_primal_res, i2.prev_primal_res);
    EXPECT_EQ(i1.prev_dual_res, i2.prev_dual_res);
    EXPECT_EQ(i1.primal_obj, i2.primal_obj);
    EXPECT_EQ(i1.dual_obj, i2.dual_obj);
    EXPECT_EQ(i1.duality_gap, i2.duality_gap);
    EXPECT_EQ(i1.duality_gap_rel, i2.duality_gap_rel);
    EXPECT_EQ(i1.factor_retires, i2.factor_retires);
    EXPECT_EQ(i1.reg_limit, i2.reg_limit);
    EXPECT_EQ(i1.no_primal_update, i2.no_primal_update);
    EXPECT_EQ(i1.no_dual_update, i2.no_dual_update);
}

// g == expected, for sparse matrices on the sparsity pattern of g
template<typename T>
void expect_on_pattern(const piqp::Mat<T>& g, const piqp::Mat<T>& expected, T tol)
{
    EXPECT_LT((g - expected).norm(), tol);
}

template<typename T, typename I>
void expect_on_pattern(const piqp::SparseMat<T, I>& g, const piqp::Mat<T>& expected, T tol)
{
    for (piqp::isize j = 0; j < g.outerSize(); j++)
    {
        for (typename piqp::SparseMat<T, I>::InnerIterator it(g, j); it; ++it)
        {
            EXPECT_NEAR(it.value(), expected(it.row(), it.col()), tol) << it.row() << ", " << it.col();
        }
    }
}

/*
 * checks that every finite bound in the active set has slack <= tol and dual >= tol,
 * and every other finite bound has slack >= tol and dual <= tol
 */
template<typename T>
void assert_active_set_match(const piqp::Result<T>& result, const piqp::rand::ActiveSet& active_set, T tol = T(1e-6))
{
    auto assert_match = [tol](const piqp::Vec<T>& s, const piqp::Vec<T>& z, const std::vector<piqp::isize>& active, const char* name)
    {
        for (piqp::isize i = 0; i < s.rows(); i++)
        {
            if (s(i) >= PIQP_INF) continue; // infinite bound
            if (std::find(active.begin(), active.end(), i) != active.end())
            {
                ASSERT_LE(s(i), tol) << name << " " << i << " should be active";
                ASSERT_GE(z(i), tol) << name << " " << i << " should be active";
            }
            else
            {
                ASSERT_GE(s(i), tol) << name << " " << i << " should be inactive";
                ASSERT_LE(z(i), tol) << name << " " << i << " should be inactive";
            }
        }
    };
    ASSERT_NO_FATAL_FAILURE(assert_match(result.s_l, result.z_l, active_set.h_l, "h_l"));
    ASSERT_NO_FATAL_FAILURE(assert_match(result.s_u, result.z_u, active_set.h_u, "h_u"));
    ASSERT_NO_FATAL_FAILURE(assert_match(result.s_bl, result.z_bl, active_set.x_l, "x_l"));
    ASSERT_NO_FATAL_FAILURE(assert_match(result.s_bu, result.z_bu, active_set.x_u, "x_u"));
}

template<typename T>
piqp::Mat<T> append_rows(const piqp::Mat<T>& A, const piqp::Mat<T>& R)
{
    piqp::Mat<T> out(A.rows() + R.rows(), R.cols());
    out << A, R;
    return out;
}

template<typename T, typename I>
piqp::SparseMat<T, I> append_rows(const piqp::SparseMat<T, I>& A, const piqp::Mat<T>& R)
{
    std::vector<Eigen::Triplet<T, I>> triplets;
    for (piqp::isize j = 0; j < A.outerSize(); j++)
    {
        for (typename piqp::SparseMat<T, I>::InnerIterator it(A, j); it; ++it)
        {
            triplets.emplace_back(I(it.row()), I(it.col()), it.value());
        }
    }
    for (piqp::isize i = 0; i < R.rows(); i++)
    {
        for (piqp::isize j = 0; j < R.cols(); j++)
        {
            if (R(i, j) != 0) triplets.emplace_back(I(A.rows() + i), I(j), R(i, j));
        }
    }
    piqp::SparseMat<T, I> out(A.rows() + R.rows(), R.cols());
    out.setFromTriplets(triplets.begin(), triplets.end());
    return out;
}

#endif //PIQP_TESTS_UTILS_HPP
