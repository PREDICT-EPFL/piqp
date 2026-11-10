// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_TESTS_BACKWARD_UTILS_HPP
#define PIQP_TESTS_BACKWARD_UTILS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "piqp/piqp.hpp"
#include "piqp/utils/random_utils.hpp"
#include "gtest/gtest.h"

// Random cotangents; slack cotangents are zeroed where the slack is infinite (inactive bound).
template<typename T>
piqp::Variables<T> random_variables(const piqp::Result<T>& r)
{
    piqp::Variables<T> w;
    w.x = piqp::rand::vector_rand<T>(r.x.rows());
    w.y = piqp::rand::vector_rand<T>(r.y.rows());
    w.z_l = piqp::rand::vector_rand<T>(r.z_l.rows());
    w.z_u = piqp::rand::vector_rand<T>(r.z_u.rows());
    w.z_bl = piqp::rand::vector_rand<T>(r.z_bl.rows());
    w.z_bu = piqp::rand::vector_rand<T>(r.z_bu.rows());
    w.s_l = piqp::rand::vector_rand<T>(r.s_l.rows());
    w.s_u = piqp::rand::vector_rand<T>(r.s_u.rows());
    w.s_bl = piqp::rand::vector_rand<T>(r.s_bl.rows());
    w.s_bu = piqp::rand::vector_rand<T>(r.s_bu.rows());
    for (piqp::isize i = 0; i < r.s_l.rows(); i++)
    {
        if (r.s_l(i) >= PIQP_INF)
        {
            w.s_l(i) = T(0);
        }
        if (r.s_u(i) >= PIQP_INF)
        {
            w.s_u(i) = T(0);
        }
    }
    for (piqp::isize i = 0; i < r.s_bl.rows(); i++)
    {
        if (r.s_bl(i) >= PIQP_INF)
        {
            w.s_bl(i) = T(0);
        }
        if (r.s_bu(i) >= PIQP_INF)
        {
            w.s_bu(i) = T(0);
        }
    }
    return w;
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

/*
 * backward() must not have side effects on the forward pass: the result of the last solve() and a
 * provided forward result stay unchanged, and a subsequent solve() gives exactly the same result as
 * on a solver that never called backward(). Both solvers have to be configured identically.
 */
template<typename Solver, typename Model>
void run_no_side_effects_test(Solver& solver, Solver& reference, Model& model)
{
    using T = typename Model::T;

    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    const piqp::Result<T> result_before = solver.result();
    piqp::optional<piqp::Result<T>> forward_result = solver.result();
    piqp::Variables<T> w = random_variables(result_before);

    for (bool smoothing : {false, true})
    {
        SCOPED_TRACE(smoothing ? "with smoothing" : "without smoothing");
        solver.settings().gradient_smoothing = smoothing;

        ASSERT_EQ(solver.backward(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu), piqp::Status::PIQP_SOLVED);
        expect_results_identical(solver.result(), result_before);

        ASSERT_EQ(solver.backward(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, forward_result), piqp::Status::PIQP_SOLVED);
        expect_results_identical(solver.result(), result_before);
        expect_results_identical(*forward_result, result_before);
    }
    solver.settings().gradient_smoothing = false;

    // the next solve is not affected by the backward passes
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    reference.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(reference.solve(), piqp::Status::PIQP_SOLVED);
    ASSERT_EQ(reference.solve(), piqp::Status::PIQP_SOLVED);
    expect_results_identical(solver.result(), reference.result());
}

// Invalid provided results are rejected: a forward result which is not solved with PIQP_UNSOLVED,
// a malformed one with PIQP_INVALID_SETTINGS.
template<typename Solver, typename Model>
void run_invalid_result_test(Solver& solver, Model& model)
{
    using T = typename Model::T;

    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    piqp::Variables<T> w = random_variables(solver.result());

    piqp::optional<piqp::Result<T>> r = solver.result();
    r->info.status = piqp::Status::PIQP_PRIMAL_INFEASIBLE;
    EXPECT_EQ(solver.backward(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_UNSOLVED);

    r = solver.result();
    r->x.resize(r->x.size() + 1);
    EXPECT_EQ(solver.backward(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_INVALID_SETTINGS);

    r = solver.result();
    r->info.rho = T(0);
    EXPECT_EQ(solver.backward(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_INVALID_SETTINGS);

    // non-positive dual on a finite bound
    piqp::isize i_finite = 0;
    while (i_finite < model.h_l.rows() && model.h_l(i_finite) <= -PIQP_INF) i_finite++;
    ASSERT_LT(i_finite, model.h_l.rows());
    r = solver.result();
    r->z_l(i_finite) = T(0);
    EXPECT_EQ(solver.backward(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_INVALID_SETTINGS);

    r = solver.result();
    EXPECT_EQ(solver.backward(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_SOLVED);
    EXPECT_EQ(solver.gradients().status, piqp::Status::PIQP_SOLVED);

    // a rejected forward result invalidates the previously computed gradients
    r->info.delta = T(-1);
    EXPECT_EQ(solver.backward(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.gradients().status, piqp::Status::PIQP_INVALID_SETTINGS);
}

/*
 * Indices of the active bounds in the user-space layout, i.e. h_l, h_u index rows of G
 * and x_l, x_u index variables.
 */
struct ActiveSet
{
    std::vector<piqp::isize> h_l;
    std::vector<piqp::isize> h_u;
    std::vector<piqp::isize> x_l;
    std::vector<piqp::isize> x_u;
};

// attempts of the known-active-set generators, each samples P, A, G and the active set
constexpr int MAX_ATTEMPTS = 500;
// maximum condition number of the stacked Jacobian of the equality and active constraints
constexpr double MAX_ACTIVE_JACOBIAN_COND = 50;

/*
 * Builds a QP backwards from a prescribed KKT point (x, y, z) with a randomly sampled active set.
 * Each G row is disabled (10%), has only a lower (30%), only an upper (30%) or both bounds (30%);
 * a fraction bounds_perc of the variables gets box bounds in the same way. Every finite bound is active
 * with probability 0.5, at most one bound per row/variable. Active bounds get duals in [0.5, 1.5],
 * inactive bounds slacks in [0.5, 1.5], such that the solution is strictly complementary. Active sets
 * violating LICQ (or with a stacked active Jacobian of condition number > MAX_ACTIVE_JACOBIAN_COND) are rejected
 * by returning nullopt, e.g., if A has an empty row.
 */
template<typename Model, typename MatP, typename MatA, typename MatG>
piqp::optional<std::pair<Model, ActiveSet>> qp_with_known_active_set(const MatP& P, const MatA& A, const MatG& G, typename Model::T bounds_perc)
{
    using T = typename Model::T;
    using piqp::isize;
    const T inf = std::numeric_limits<T>::infinity();
    isize n = P.rows();
    isize p = A.rows();
    isize m = G.rows();
    piqp::Mat<T> A_dense = piqp::Mat<T>(A);
    piqp::Mat<T> G_dense = piqp::Mat<T>(G);
    auto uniform = []() { return T(piqp::rand::uniform_dist(piqp::rand::gen)); };

    // finite bounds, and active bound: -1 lower, 1 upper, 0 none
    std::vector<bool> h_l_finite(m), h_u_finite(m), x_l_finite(n), x_u_finite(n);
    std::vector<int> G_active(m), x_active(n);
    auto sample_active = [&](bool lower_finite, bool upper_finite) {
        if (uniform() >= T(0.5)) return 0;
        if (lower_finite && (!upper_finite || uniform() < T(0.5))) return -1;
        return upper_finite ? 1 : 0;
    };
    for (isize i = 0; i < m; i++)
    {
        T u = uniform();
        h_l_finite[i] = (u >= T(0.1) && u < T(0.4)) || u >= T(0.7);
        h_u_finite[i] = u >= T(0.4);
        G_active[i] = sample_active(h_l_finite[i], h_u_finite[i]);
    }
    for (isize i = 0; i < n; i++)
    {
        T u = uniform();
        x_l_finite[i] = u < bounds_perc / 3 || (u >= 2 * bounds_perc / 3 && u < bounds_perc);
        x_u_finite[i] = u >= bounds_perc / 3 && u < bounds_perc;
        x_active[i] = sample_active(x_l_finite[i], x_u_finite[i]);
    }

    // LICQ: equality rows and active constraint gradients are linearly independent
    isize n_active = p;
    for (isize i = 0; i < m; i++) n_active += G_active[i] != 0;
    for (isize i = 0; i < n; i++) n_active += x_active[i] != 0;
    if (n_active > n) return piqp::nullopt;
    if (n_active > 0)
    {
        piqp::Mat<T> J = piqp::Mat<T>::Zero(n_active, n);
        J.topRows(p) = A_dense;
        isize k = p;
        for (isize i = 0; i < m; i++)
        {
            if (G_active[i] != 0) J.row(k++) = G_dense.row(i);
        }
        for (isize i = 0; i < n; i++)
        {
            if (x_active[i] != 0) J(k++, i) = T(1);
        }
        Eigen::JacobiSVD<piqp::Mat<T>> svd(J);
        const piqp::Vec<T>& sigma = svd.singularValues();
        if (!(sigma(n_active - 1) > T(0)) || sigma(0) / sigma(n_active - 1) > T(MAX_ACTIVE_JACOBIAN_COND)) return piqp::nullopt;
    }

    piqp::Vec<T> x = piqp::rand::vector_rand<T>(n);
    piqp::Vec<T> y = piqp::rand::vector_rand<T>(p);
    piqp::Vec<T> Gx = G_dense * x;
    auto margin = [&]() { return T(0.5) + uniform(); };

    ActiveSet active_set;
    piqp::Vec<T> h_l(m), h_u(m), z_l = piqp::Vec<T>::Zero(m), z_u = piqp::Vec<T>::Zero(m);
    for (isize i = 0; i < m; i++)
    {
        h_l(i) = h_l_finite[i] ? Gx(i) - margin() : -inf;
        h_u(i) = h_u_finite[i] ? Gx(i) + margin() : inf;
        if (G_active[i] < 0) { h_l(i) = Gx(i); z_l(i) = margin(); active_set.h_l.push_back(i); }
        if (G_active[i] > 0) { h_u(i) = Gx(i); z_u(i) = margin(); active_set.h_u.push_back(i); }
    }
    piqp::Vec<T> x_l(n), x_u(n), z_bl = piqp::Vec<T>::Zero(n), z_bu = piqp::Vec<T>::Zero(n);
    for (isize i = 0; i < n; i++)
    {
        x_l(i) = x_l_finite[i] ? x(i) - margin() : -inf;
        x_u(i) = x_u_finite[i] ? x(i) + margin() : inf;
        if (x_active[i] < 0) { x_l(i) = x(i); z_bl(i) = margin(); active_set.x_l.push_back(i); }
        if (x_active[i] > 0) { x_u(i) = x(i); z_bu(i) = margin(); active_set.x_u.push_back(i); }
    }

    // stationarity P x + c + A^T y + G^T (z_u - z_l) + z_bu - z_bl = 0 and primal feasibility A x = b
    piqp::Vec<T> c = -(P.template selfadjointView<Eigen::Upper>() * x + A_dense.transpose() * y
                       + G_dense.transpose() * (z_u - z_l) + z_bu - z_bl);
    piqp::Vec<T> b = A_dense * x;

    return std::make_pair(Model(P, c, A, b, G, h_l, h_u, x_l, x_u), active_set);
}

/*
 * Random strongly convex QPs with a known, strictly complementary solution satisfying LICQ (see qp_with_known_active_set).
 * Each of the MAX_ATTEMPTS attempts samples P, A, G and the active set. The result only depends on the state of
 * piqp::rand::gen.
 */
template<typename T>
std::pair<piqp::dense::Model<T>, ActiveSet>
dense_strongly_convex_qp_with_known_active_set(piqp::isize dim, piqp::isize n_eq, piqp::isize n_ineq,
                                               T bounds_perc = T(0.5), T strong_convexity_factor = T(1e-2))
{
    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
    {
        piqp::Mat<T> P = piqp::rand::dense_positive_definite_upper_triangular_rand<T>(dim, strong_convexity_factor);
        piqp::Mat<T> A = piqp::rand::dense_matrix_rand<T>(n_eq, dim);
        piqp::Mat<T> G = piqp::rand::dense_matrix_rand<T>(n_ineq, dim);
        auto qp = qp_with_known_active_set<piqp::dense::Model<T>>(P, A, G, bounds_perc);
        if (qp.has_value()) return *qp;
    }
    throw std::runtime_error("no QP with a known active set found");
}

template<typename T, typename I>
std::pair<piqp::sparse::Model<T, I>, ActiveSet>
sparse_strongly_convex_qp_with_known_active_set(piqp::isize dim, piqp::isize n_eq, piqp::isize n_ineq, T sparsity_factor,
                                                T bounds_perc = T(0.5), T strong_convexity_factor = T(1e-2))
{
    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
    {
        piqp::SparseMat<T, I> P = piqp::rand::sparse_positive_definite_upper_triangular_rand<T, I>(dim, sparsity_factor, strong_convexity_factor);
        piqp::SparseMat<T, I> A = piqp::rand::sparse_matrix_rand<T, I>(n_eq, dim, sparsity_factor);
        piqp::SparseMat<T, I> G = piqp::rand::sparse_matrix_rand<T, I>(n_ineq, dim, sparsity_factor);
        auto qp = qp_with_known_active_set<piqp::sparse::Model<T, I>>(P, A, G, bounds_perc);
        if (qp.has_value()) return *qp;
    }
    throw std::runtime_error("no QP with a known active set found");
}

/*
 * checks that every finite bound in the active set has slack <= tol and dual >= tol,
 * and every other finite bound has slack >= tol and dual <= tol
 */
template<typename T>
void assert_active_set_match(const piqp::Result<T>& result, const ActiveSet& active_set, T tol = T(1e-6))
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
void random_values(piqp::Mat<T>& M, bool nonzero)
{
    if (nonzero) M = piqp::rand::dense_matrix_rand<T>(M.rows(), M.cols());
    else M.setZero();
}

// keeps the sparsity pattern
template<typename T, typename I>
void random_values(piqp::SparseMat<T, I>& M, bool nonzero)
{
    for (piqp::isize k = 0; k < M.nonZeros(); k++)
    {
        M.valuePtr()[k] = nonzero ? T(piqp::rand::normal_dist(piqp::rand::gen)) : T(0);
    }
}

// zero where the bound is infinite
template<typename T>
void random_values(piqp::Vec<T>& d, const piqp::Vec<T>& bound, bool nonzero)
{
    for (piqp::isize i = 0; i < d.rows(); i++)
    {
        d(i) = nonzero && std::abs(bound(i)) < PIQP_INF ? T(piqp::rand::normal_dist(piqp::rand::gen)) : T(0);
    }
}

/*
 * Random unit direction in the data space, only in block (0: P, 1: c, 2: A, 3: b, 4: G, 5: h_l, 6: h_u, 7: x_l, 8: x_u)
 * or in all blocks (block = -1). It has the sparsity pattern of the model, is upper triangular in P
 * (the solver only reads the upper triangular part) and is zero at infinite bounds.
 */
template<typename Model>
Model random_direction(const Model& model, int block)
{
    Model d = model;
    random_values(d.P, block < 0 || block == 0);
    d.P = decltype(d.P)(d.P.template triangularView<Eigen::Upper>());
    random_values(d.c, model.c, block < 0 || block == 1);
    random_values(d.A, block < 0 || block == 2);
    random_values(d.b, model.b, block < 0 || block == 3);
    random_values(d.G, block < 0 || block == 4);
    random_values(d.h_l, model.h_l, block < 0 || block == 5);
    random_values(d.h_u, model.h_u, block < 0 || block == 6);
    random_values(d.x_l, model.x_l, block < 0 || block == 7);
    random_values(d.x_u, model.x_u, block < 0 || block == 8);

    // unit norm, such that the step size of the finite differences is independent of the block size
    using T = typename Model::T;
    T norm = std::sqrt(d.P.squaredNorm() + d.c.squaredNorm() + d.A.squaredNorm() + d.b.squaredNorm() + d.G.squaredNorm()
                       + d.h_l.squaredNorm() + d.h_u.squaredNorm() + d.x_l.squaredNorm() + d.x_u.squaredNorm());
    d.P *= T(1) / norm;
    d.c /= norm;
    d.A *= T(1) / norm;
    d.b /= norm;
    d.G *= T(1) / norm;
    d.h_l /= norm;
    d.h_u /= norm;
    d.x_l /= norm;
    d.x_u /= norm;
    return d;
}

template<typename Model>
Model perturbed_model(const Model& model, const Model& d, typename Model::T eps)
{
    Model r = model;
    r.P = model.P + eps * d.P;
    r.c += eps * d.c;
    r.A = model.A + eps * d.A;
    r.b += eps * d.b;
    r.G = model.G + eps * d.G;
    r.h_l += eps * d.h_l;
    r.h_u += eps * d.h_u;
    r.x_l += eps * d.x_l;
    r.x_u += eps * d.x_u;
    return r;
}

template<typename T>
T inner(const piqp::Mat<T>& g, const piqp::Mat<T>& d)
{
    return (g.array() * d.array()).sum();
}

template<typename T, typename I>
T inner(const piqp::SparseMat<T, I>& g, const piqp::SparseMat<T, I>& d)
{
    T sum = 0;
    for (piqp::isize j = 0; j < d.outerSize(); j++)
    {
        for (typename piqp::SparseMat<T, I>::InnerIterator it(d, j); it; ++it)
        {
            sum += g.coeff(it.row(), it.col()) * it.value();
        }
    }
    return sum;
}

// the gradient w.r.t. P is symmetric: dense matrices are symmetric, sparse ones (on the sparsity pattern
// of the P passed to setup()) have equal values on mirrored entries if both are stored
template<typename T>
void expect_symmetric_values(const piqp::Mat<T>& P)
{
    EXPECT_TRUE(P.isApprox(P.transpose(), 1e-12));
}

template<typename T, typename I>
void expect_symmetric_values(const piqp::SparseMat<T, I>& P)
{
    for (piqp::isize j = 0; j < P.outerSize(); j++)
    {
        for (typename piqp::SparseMat<T, I>::InnerIterator it(P, j); it; ++it)
        {
            for (typename piqp::SparseMat<T, I>::InnerIterator it_t(P, it.row()); it_t; ++it_t)
            {
                if (it_t.row() == it.col()) EXPECT_NEAR(it_t.value(), it.value(), 1e-12);
            }
        }
    }
}

// <gradients, d>; the upper triangular dP acts as the symmetric perturbation dP + dP^T - diag(dP)
template<typename Grads, typename Model>
typename Model::T directional_derivative(const Grads& g, const Model& d)
{
    return 2 * inner(g.P, d.P) - g.P.diagonal().dot(d.P.diagonal())
        + g.c.dot(d.c) + inner(g.A, d.A) + g.b.dot(d.b) + inner(g.G, d.G)
        + g.h_l.dot(d.h_l) + g.h_u.dot(d.h_u) + g.x_l.dot(d.x_l) + g.x_u.dot(d.x_u);
}

// all outputs of the solver, (x, y, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu)
template<typename T>
struct OutputField
{
    piqp::Vec<T> piqp::Variables<T>::* field;
    const char* name;
};

template<typename T>
std::vector<OutputField<T>> output_fields()
{
    using V = piqp::Variables<T>;
    return {{&V::x, "x"}, {&V::y, "y"}, {&V::z_l, "z_l"}, {&V::z_u, "z_u"}, {&V::z_bl, "z_bl"}, {&V::z_bu, "z_bu"},
            {&V::s_l, "s_l"}, {&V::s_u, "s_u"}, {&V::s_bl, "s_bl"}, {&V::s_bu, "s_bu"}};
}

/*
 * Central difference of the solution along d, i.e. the JVP J d. The solution map is only differentiable
 * within a fixed active set, which is asserted for both perturbed problems.
 */
template<typename Solver, typename Model>
void finite_difference_jvp(Solver& solver, const Model& model, const Model& d, const ActiveSet& active_set,
                           piqp::Variables<typename Model::T>& jvp)
{
    using T = typename Model::T;
    const T eps = T(1e-3);

    Model model_plus = perturbed_model(model, d, eps);
    solver.setup(model_plus.P, model_plus.c, model_plus.A, model_plus.b, model_plus.G, model_plus.h_l, model_plus.h_u, model_plus.x_l, model_plus.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    ASSERT_NO_FATAL_FAILURE(assert_active_set_match(solver.result(), active_set));
    jvp = solver.result();

    Model model_minus = perturbed_model(model, d, -eps);
    solver.setup(model_minus.P, model_minus.c, model_minus.A, model_minus.b, model_minus.G, model_minus.h_l, model_minus.h_u, model_minus.x_l, model_minus.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    ASSERT_NO_FATAL_FAILURE(assert_active_set_match(solver.result(), active_set));

    for (const OutputField<T>& f : output_fields<T>())
    {
        jvp.*f.field = (jvp.*f.field - solver.result().*f.field) / (2 * eps);
    }
}

/*
 * Compares J d from backward() with central finite differences of the solution along random unit
 * directions d, for each data block separately and for all blocks at once. backward() computes J^T w,
 * so the k-th entry of J d is <J^T e_k, d>, obtained from backward() with the unit cotangent e_k.
 * The model has to be generated with a known active set, which is checked at the solution.
 */
template<typename Solver, typename Model>
void run_finite_difference_test(Solver& solver, const Model& model, const ActiveSet& active_set)
{
    using T = typename Model::T;

    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    ASSERT_NO_FATAL_FAILURE(assert_active_set_match(solver.result(), active_set));
    const piqp::Result<T> result = solver.result();

    // random directions and their finite-difference JVPs, using a separate solver for the perturbed problems
    Solver fd_solver;
    fd_solver.settings() = solver.settings();
    std::vector<Model> directions;
    std::vector<int> blocks;
    std::vector<piqp::Variables<T>> jvps_fd;
    for (int block = -1; block < 9; block++)
    {
        for (int sample = 0; sample < 3; sample++)
        {
            directions.push_back(random_direction(model, block));
            blocks.push_back(block);
            jvps_fd.emplace_back();
            ASSERT_NO_FATAL_FAILURE(finite_difference_jvp(fd_solver, model, directions.back(), active_set, jvps_fd.back()));
        }
    }

    piqp::Variables<T> e;
    e.resize(model.P.rows(), model.A.rows(), model.G.rows());
    for (const OutputField<T>& f : output_fields<T>())
    {
        (e.*f.field).setZero();
    }
    e.x(0) = T(1);

    // gradients are preallocated in setup(), backward() must not allocate
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    ASSERT_EQ(solver.backward(e.x, e.y, e.z_l, e.z_u, e.z_bl, e.z_bu, e.s_l, e.s_u, e.s_bl, e.s_bu), piqp::Status::PIQP_SOLVED);
    ASSERT_EQ(solver.backward(e.x, e.y, e.z_l, e.z_u, e.z_bl, e.z_bu, e.s_l, e.s_u, e.s_bl, e.s_bu), piqp::Status::PIQP_SOLVED);
    PIQP_EIGEN_MALLOC_ALLOWED();
    e.x(0) = T(0);
    const auto& g = solver.gradients();

    expect_symmetric_values(g.P);

    // infinite bounds and disabled rows have zero gradients
    for (piqp::isize i = 0; i < model.G.rows(); i++)
    {
        if (model.h_l(i) <= -PIQP_INF) EXPECT_EQ(g.h_l(i), T(0)) << "h_l " << i;
        if (model.h_u(i) >= PIQP_INF) EXPECT_EQ(g.h_u(i), T(0)) << "h_u " << i;
        if (model.h_l(i) <= -PIQP_INF && model.h_u(i) >= PIQP_INF) EXPECT_EQ(g.G.row(i).norm(), T(0)) << "G " << i;
    }
    for (piqp::isize i = 0; i < model.P.rows(); i++)
    {
        if (model.x_l(i) <= -PIQP_INF) EXPECT_EQ(g.x_l(i), T(0)) << "x_l " << i;
        if (model.x_u(i) >= PIQP_INF) EXPECT_EQ(g.x_u(i), T(0)) << "x_u " << i;
    }

    for (const OutputField<T>& f : output_fields<T>())
    {
        piqp::Vec<T>& e_f = e.*f.field;
        for (piqp::isize k = 0; k < e_f.rows(); k++)
        {
            // infinite slacks are not differentiable
            if ((result.*f.field)(k) >= PIQP_INF) continue;

            e_f(k) = T(1);
            ASSERT_EQ(solver.backward(e.x, e.y, e.z_l, e.z_u, e.z_bl, e.z_bu, e.s_l, e.s_u, e.s_bl, e.s_bu), piqp::Status::PIQP_SOLVED);
            e_f(k) = T(0);

            // the directions have unit norm, i.e. |<g - g_exact, d>| <= ||g - g_exact||, so the tolerance is relative to ||g||
            T g_norm = std::sqrt(g.P.squaredNorm() + g.c.squaredNorm() + g.A.squaredNorm() + g.b.squaredNorm() + g.G.squaredNorm()
                                 + g.h_l.squaredNorm() + g.h_u.squaredNorm() + g.x_l.squaredNorm() + g.x_u.squaredNorm());
            T tol = T(1e-5) * (T(1) + g_norm);

            for (std::size_t j = 0; j < directions.size(); j++)
            {
                EXPECT_NEAR(directional_derivative(g, directions[j]), (jvps_fd[j].*f.field)(k), tol)
                    << f.name << " " << k << ", block " << blocks[j];
            }
        }
    }
}

/*
 * min 0.5 x^2 - a x  s.t.  x >= x_l = 0, with gradient smoothing the relaxed KKT conditions are
 *   x - a - z = 0,  x - x_l - s = 0,  s z = mu   =>   x = (a + sqrt(a^2 + 4 mu)) / 2,  s = x,  z = x - a,
 * such that with D = sqrt(a^2 + 4 mu) = s + z
 *   dx/da = 0.5 (1 + a / D),  dx/dx_l = z / (s + z),  ds/da = dx/da,  ds/dx_l = dx/dx_l - 1.
 * With the cotangents w_x on x and w_s on s_bl, i.e. L = w_x x + w_s s, this gives
 *   dL/dc = -(w_x + w_s) dx/da,  dL/dx_l = w_x dx/dx_l + w_s (dx/dx_l - 1),  dL/dP = x dL/dc,
 * since P and c only enter through P x + c. Checked with and without cost scaling, for a on the active side,
 * at the kink of the nonsmoothed solution and on the inactive side, and with and without slack cotangent, which
 * enters the adjoint right-hand side multiplied by the smoothed slack.
 * P is the 1x1 identity in the matrix type of the solver.
 */
template<typename Solver, typename MatP>
void run_smoothing_analytic_test(Solver& solver, const MatP& P)
{
    using T = typename MatP::Scalar;
    const T mu = 1e-3;
    const T w_x = 1;
    solver.settings().gradient_smoothing = true;
    solver.settings().gradient_smoothing_mu = mu;
    solver.settings().gradient_smoothing_tol = 1e-10;
    solver.settings().gradient_smoothing_max_iter = 20;

    for (bool scale_cost : {false, true})
    {
        for (T a : {T(-0.05), T(0), T(0.05)})
        {
            for (T w_s : {T(0), T(-0.6)})
            {
                SCOPED_TRACE("a = " + std::to_string(a) + ", scale_cost = " + std::to_string(scale_cost) + ", w_s = " + std::to_string(w_s));

                piqp::Vec<T> c = piqp::Vec<T>::Constant(1, -a);
                piqp::Vec<T> x_l = piqp::Vec<T>::Zero(1);
                solver.settings().preconditioner_scale_cost = scale_cost;
                solver.setup(P, c, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, x_l, piqp::nullopt);
                ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);

                // the forward solution is the true optimum
                EXPECT_NEAR(solver.result().x(0), (std::max)(a, T(0)), 1e-5);

                piqp::Vec<T> grad_x = piqp::Vec<T>::Constant(1, w_x);
                piqp::Vec<T> grad_s_bl = piqp::Vec<T>::Constant(1, w_s);
                ASSERT_EQ(solver.backward(grad_x, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt,
                                          piqp::nullopt, piqp::nullopt, grad_s_bl), piqp::Status::PIQP_SOLVED);
                ASSERT_EQ(solver.gradients().status, piqp::Status::PIQP_SOLVED);

                T D = std::sqrt(a * a + 4 * mu);
                T x = (a + D) / 2;
                T s = x;
                T z = x - a;
                T dx_da = T(0.5) * (1 + a / D);
                T dx_dxl = z / (s + z);
                T dL_dc = -(w_x + w_s) * dx_da;

                // backward() differentiates the KKT system regularized with the final delta of the forward solve,
                // which is larger at the kink (degenerate forward problem), so the error is O(delta)
                T tol = T(1e-7) + solver.result().info.delta;
                const auto& g = solver.gradients();
                EXPECT_NEAR(g.c(0), dL_dc, tol);
                EXPECT_NEAR(g.x_l(0), w_x * dx_dxl + w_s * (dx_dxl - 1), tol);
                EXPECT_NEAR(g.P.coeff(0, 0), x * dL_dc, tol);
            }
        }
    }
}

#endif //PIQP_TESTS_BACKWARD_UTILS_HPP
