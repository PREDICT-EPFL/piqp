// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_TESTS_DERIVATIVE_UTILS_HPP
#define PIQP_TESTS_DERIVATIVE_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "piqp/piqp.hpp"
#include "piqp/utils/random_utils.hpp"
#include "gtest/gtest.h"
#include "utils.hpp"

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
        M.valuePtr()[k] = nonzero ? T(piqp::rand::normal_dist()(piqp::rand::gen())) : T(0);
    }
}

// zero where the bound is infinite
template<typename T>
void random_values(piqp::Vec<T>& d, const piqp::Vec<T>& bound, bool nonzero)
{
    for (piqp::isize i = 0; i < d.rows(); i++)
    {
        d(i) = nonzero && std::abs(bound(i)) < PIQP_INF ? T(piqp::rand::normal_dist()(piqp::rand::gen())) : T(0);
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
 * vjp() and jvp() must not have side effects on solve(): the result of the last solve() and a
 * provided solve result stay unchanged, and a subsequent solve() gives exactly the same result as
 * on a solver that never called vjp() or jvp(). Both solvers have to be configured identically.
 */
template<typename Solver, typename Model>
void run_no_side_effects_test(Solver& solver, Solver& reference, Model& model)
{
    using T = typename Model::T;

    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    const piqp::Result<T> result_before = solver.result();
    piqp::optional<piqp::Result<T>> solve_result = solver.result();
    piqp::Variables<T> w = piqp::rand::variables_rand<T>(model.P.rows(), model.A.rows(), model.G.rows());
    Model d = random_direction(model, -1);

    for (bool smoothing : {false, true})
    {
        SCOPED_TRACE(smoothing ? "with smoothing" : "without smoothing");
        solver.settings().derivative_smoothing = smoothing;

        ASSERT_EQ(solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu), piqp::Status::PIQP_SOLVED);
        expect_results_identical(solver.result(), result_before);

        ASSERT_EQ(solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, solve_result), piqp::Status::PIQP_SOLVED);
        expect_results_identical(solver.result(), result_before);
        expect_results_identical(*solve_result, result_before);

        ASSERT_EQ(solver.jvp(d.P, d.c, d.A, d.b, d.G, d.h_l, d.h_u, d.x_l, d.x_u), piqp::Status::PIQP_SOLVED);
        expect_results_identical(solver.result(), result_before);

        ASSERT_EQ(solver.jvp(d.P, d.c, d.A, d.b, d.G, d.h_l, d.h_u, d.x_l, d.x_u, solve_result), piqp::Status::PIQP_SOLVED);
        expect_results_identical(solver.result(), result_before);
        expect_results_identical(*solve_result, result_before);
    }
    solver.settings().derivative_smoothing = false;

    // the next solve is not affected by the vjp() and jvp() calls
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    reference.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(reference.solve(), piqp::Status::PIQP_SOLVED);
    ASSERT_EQ(reference.solve(), piqp::Status::PIQP_SOLVED);
    expect_results_identical(solver.result(), reference.result());
}

/*
 * Status of vjp() and jvp() in all invalid states: not set up, not enabled before setup(), not solved, wrong
 * dimensions, data changed since the last solve(), smoothing not converged. The status is also stored in
 * vjp_result().status and jvp_result().status, and setup(), update() and solve() reset them.
 */
template<typename Solver, typename Model>
void run_invalid_states_test(Solver& solver, Model& model)
{
    using T = typename Model::T;
    using piqp::nullopt;
    using piqp::Status;
    auto setup = [&]() { solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u); };

    EXPECT_EQ(solver.vjp(), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp_result().status, Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp_result().status, Status::PIQP_INVALID_SETTINGS);

    // enable_derivatives has to be set before setup()
    solver.settings().enable_derivatives = false;
    setup();
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.vjp(), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(), Status::PIQP_INVALID_SETTINGS);
    solver.settings().enable_derivatives = true;
    EXPECT_EQ(solver.vjp(), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(), Status::PIQP_INVALID_SETTINGS);

    setup();
    EXPECT_EQ(solver.vjp_result().status, Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.vjp(), Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.vjp_result().status, Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.jvp_result().status, Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.jvp(), Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.jvp_result().status, Status::PIQP_UNSOLVED);

    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.vjp(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.vjp_result().status, Status::PIQP_SOLVED);
    EXPECT_EQ(solver.jvp(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.jvp_result().status, Status::PIQP_SOLVED);

    // wrong dimensions of each argument, a failed call invalidates the previous gradients
    piqp::Vec<T> wrong_n = piqp::Vec<T>::Zero(model.P.rows() + 1);
    piqp::Vec<T> wrong_p = piqp::Vec<T>::Zero(model.A.rows() + 1);
    piqp::Vec<T> wrong_m = piqp::Vec<T>::Zero(model.G.rows() + 1);
    EXPECT_EQ(solver.vjp(wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp_result().status, Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp(nullopt, wrong_p), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp(nullopt, nullopt, wrong_m), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp(nullopt, nullopt, nullopt, wrong_m), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp(nullopt, nullopt, nullopt, nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp(nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_m), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_m), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);

    EXPECT_EQ(solver.jvp(), Status::PIQP_SOLVED);
    Model wrong = model;
    wrong.P.resize(model.P.rows() + 1, model.P.cols() + 1);
    wrong.A.resize(model.A.rows() + 1, model.A.cols());
    wrong.G.resize(model.G.rows(), model.G.cols() + 1);
    EXPECT_EQ(solver.jvp(wrong.P), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp_result().status, Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(nullopt, nullopt, wrong.A), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(nullopt, nullopt, nullopt, wrong_p), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(nullopt, nullopt, nullopt, nullopt, wrong.G), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(nullopt, nullopt, nullopt, nullopt, nullopt, wrong_m), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_m), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.jvp(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), Status::PIQP_INVALID_SETTINGS);

    // smoothing not converged
    solver.settings().derivative_smoothing = true;
    solver.settings().derivative_smoothing_max_iter = 1;
    EXPECT_EQ(solver.vjp(), Status::PIQP_MAX_ITER_REACHED);
    EXPECT_EQ(solver.vjp_result().status, Status::PIQP_MAX_ITER_REACHED);
    EXPECT_EQ(solver.jvp(), Status::PIQP_MAX_ITER_REACHED);
    EXPECT_EQ(solver.jvp_result().status, Status::PIQP_MAX_ITER_REACHED);
    solver.settings().derivative_smoothing = false;

    // data changed since the last solve()
    EXPECT_EQ(solver.vjp(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.jvp(), Status::PIQP_SOLVED);
    solver.update(nullopt, model.c);
    EXPECT_EQ(solver.vjp_result().status, Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.jvp_result().status, Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.vjp(), Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.jvp(), Status::PIQP_UNSOLVED);

    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.vjp(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.jvp(), Status::PIQP_SOLVED);

    // a new solve() invalidates the derivatives of the previous solution
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.vjp_result().status, Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.jvp_result().status, Status::PIQP_UNSOLVED);

    EXPECT_EQ(solver.vjp(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.jvp(), Status::PIQP_SOLVED);
    setup();
    EXPECT_EQ(solver.vjp_result().status, Status::PIQP_UNSOLVED);
    EXPECT_EQ(solver.jvp_result().status, Status::PIQP_UNSOLVED);
}

/*
 * Without constraints, x = -P^{-1} c, such that for L = w^T x
 *   dL/dc = -P^{-1} w,  dL/dP = 0.5 (dL/dc x^T + x dL/dc^T),
 * and along a perturbation (dP, dc)
 *   dx = -P^{-1} (dP x + dc),
 * with and without derivative smoothing. All constraint gradients are empty or zero.
 */
template<typename Solver, typename MatP>
void run_no_constraints_test(Solver& solver, const MatP& P)
{
    using T = typename MatP::Scalar;
    piqp::Mat<T> P_full = piqp::Mat<T>(piqp::Mat<T>(P).template selfadjointView<Eigen::Upper>());
    piqp::isize n = P_full.rows();
    piqp::Vec<T> c = piqp::rand::vector_rand<T>(n);
    piqp::Vec<T> w = piqp::rand::vector_rand<T>(n);

    piqp::Vec<T> x = -P_full.ldlt().solve(c);
    piqp::Vec<T> dL_dc = -P_full.ldlt().solve(w);
    piqp::Mat<T> dL_dP = T(0.5) * (dL_dc * x.transpose() + x * dL_dc.transpose());

    // perturbation of P with the type and the (upper triangular) sparsity pattern of P
    MatP dP = P;
    random_values(dP, true);
    piqp::Mat<T> dP_full = piqp::Mat<T>(piqp::Mat<T>(dP).template selfadjointView<Eigen::Upper>());
    piqp::Vec<T> dc = piqp::rand::vector_rand<T>(n);
    piqp::Vec<T> dx = -P_full.ldlt().solve(dP_full * x + dc);

    for (bool smoothing : {false, true})
    {
        SCOPED_TRACE(smoothing ? "with smoothing" : "without smoothing");
        solver.settings().derivative_smoothing = smoothing;
        solver.setup(P, c);
        ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
        ASSERT_EQ(solver.vjp(w), piqp::Status::PIQP_SOLVED);

        // vjp() uses the regularized KKT matrix, i.e., the error is O(rho / lambda_min(P))
        const auto& g = solver.vjp_result();
        T tol = T(1e-5) * (T(1) + dL_dc.norm());
        EXPECT_LT((g.c - dL_dc).norm(), tol);
        expect_on_pattern(g.P, dL_dP, tol);
        EXPECT_EQ(g.A.rows(), 0);
        EXPECT_EQ(g.G.rows(), 0);
        EXPECT_EQ(g.b.rows(), 0);
        EXPECT_EQ(g.h_l.rows(), 0);
        EXPECT_EQ(g.h_u.rows(), 0);
        EXPECT_TRUE(g.x_l.isZero(0));
        EXPECT_TRUE(g.x_u.isZero(0));

        ASSERT_EQ(solver.jvp(dP, dc), piqp::Status::PIQP_SOLVED);
        const piqp::JVPResult<T>& t = solver.jvp_result();
        EXPECT_LT((t.x - dx).norm(), T(1e-5) * (T(1) + dx.norm()));
        EXPECT_EQ(t.y.rows(), 0);
        EXPECT_EQ(t.z_l.rows(), 0);
        EXPECT_TRUE(t.z_bl.isZero(0));
        EXPECT_TRUE(t.s_bu.isZero(0));
    }
}

// A copied or moved solver computes the same gradients, i.e., the differentiation state is copied as well.
template<typename Solver, typename Model>
void run_copy_move_test(Solver& solver, Model& model)
{
    using T = typename Model::T;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    piqp::Variables<T> w = piqp::rand::variables_rand<T>(model.P.rows(), model.A.rows(), model.G.rows());

    auto vjp = [&w](Solver& s) {
        ASSERT_EQ(s.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu), piqp::Status::PIQP_SOLVED);
    };
    Model d = random_direction(model, -1);
    auto jvp = [&d](Solver& s) {
        ASSERT_EQ(s.jvp(d.P, d.c, d.A, d.b, d.G, d.h_l, d.h_u, d.x_l, d.x_u), piqp::Status::PIQP_SOLVED);
    };
    auto expect_same_tangents = [](const Solver& s1, const Solver& s2) {
        const piqp::JVPResult<T>& t1 = s1.jvp_result();
        const piqp::JVPResult<T>& t2 = s2.jvp_result();
        EXPECT_EQ(t1.status, t2.status);
        for (const OutputField<T>& f : output_fields<T>())
        {
            EXPECT_EQ(t1.*f.field, t2.*f.field) << f.name;
        }
    };
    auto expect_same_gradients = [](const Solver& s1, const Solver& s2) {
        const auto& g1 = s1.vjp_result();
        const auto& g2 = s2.vjp_result();
        EXPECT_EQ(g1.status, g2.status);
        EXPECT_EQ(piqp::Mat<T>(g1.P), piqp::Mat<T>(g2.P));
        EXPECT_EQ(piqp::Mat<T>(g1.A), piqp::Mat<T>(g2.A));
        EXPECT_EQ(piqp::Mat<T>(g1.G), piqp::Mat<T>(g2.G));
        EXPECT_EQ(g1.c, g2.c);
        EXPECT_EQ(g1.b, g2.b);
        EXPECT_EQ(g1.h_l, g2.h_l);
        EXPECT_EQ(g1.h_u, g2.h_u);
        EXPECT_EQ(g1.x_l, g2.x_l);
        EXPECT_EQ(g1.x_u, g2.x_u);
    };

    Solver copy(solver);
    ASSERT_NO_FATAL_FAILURE(vjp(solver));
    ASSERT_NO_FATAL_FAILURE(vjp(copy));
    expect_same_gradients(solver, copy);
    ASSERT_NO_FATAL_FAILURE(jvp(solver));
    ASSERT_NO_FATAL_FAILURE(jvp(copy));
    expect_same_tangents(solver, copy);

    Solver moved(std::move(copy));
    ASSERT_NO_FATAL_FAILURE(vjp(moved));
    expect_same_gradients(solver, moved);
    ASSERT_NO_FATAL_FAILURE(jvp(moved));
    expect_same_tangents(solver, moved);

    // a copy after vjp() reuses the copied KKT factorization
    Solver copy_after_vjp(solver);
    ASSERT_NO_FATAL_FAILURE(vjp(copy_after_vjp));
    expect_same_gradients(solver, copy_after_vjp);
    ASSERT_NO_FATAL_FAILURE(jvp(copy_after_vjp));
    expect_same_tangents(solver, copy_after_vjp);
}

/*
 * The default vjp() and jvp() linearize at the result of the last solve(), i.e., they give the same derivatives as
 * passing that result explicitly. Repeated calls and mixing both paths reuse or refactor the KKT matrix correctly.
 */
template<typename Solver, typename Model>
void run_default_matches_provided_test(Solver& solver, Model& model)
{
    using T = typename Model::T;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    piqp::optional<piqp::Result<T>> solve_result = solver.result();
    piqp::Variables<T> w = piqp::rand::variables_rand<T>(model.P.rows(), model.A.rows(), model.G.rows());

    struct Grads { piqp::Mat<T> P, A, G; piqp::Vec<T> c, b, h_l, h_u, x_l, x_u; };
    auto vjp = [&](bool provided, Grads& out) {
        piqp::Status status = provided
            ? solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, solve_result)
            : solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu);
        ASSERT_EQ(status, piqp::Status::PIQP_SOLVED);
        const auto& g = solver.vjp_result();
        out = {piqp::Mat<T>(g.P), piqp::Mat<T>(g.A), piqp::Mat<T>(g.G), g.c, g.b, g.h_l, g.h_u, g.x_l, g.x_u};
    };
    auto expect_close = [](const Grads& g1, const Grads& g2) {
        auto close = [](const piqp::Mat<T>& a, const piqp::Mat<T>& b) {
            T scale = (std::max)(T(1), b.template lpNorm<Eigen::Infinity>());
            return (a - b).template lpNorm<Eigen::Infinity>() <= T(1e-9) * scale;
        };
        EXPECT_TRUE(close(g1.P, g2.P));
        EXPECT_TRUE(close(g1.A, g2.A));
        EXPECT_TRUE(close(g1.G, g2.G));
        EXPECT_TRUE(close(g1.c, g2.c));
        EXPECT_TRUE(close(g1.b, g2.b));
        EXPECT_TRUE(close(g1.h_l, g2.h_l));
        EXPECT_TRUE(close(g1.h_u, g2.h_u));
        EXPECT_TRUE(close(g1.x_l, g2.x_l));
        EXPECT_TRUE(close(g1.x_u, g2.x_u));
    };

    Grads g_default, g_cached, g_provided, g_after_provided;
    ASSERT_NO_FATAL_FAILURE(vjp(false, g_default));
    ASSERT_NO_FATAL_FAILURE(vjp(false, g_cached));
    ASSERT_NO_FATAL_FAILURE(vjp(true, g_provided));
    ASSERT_NO_FATAL_FAILURE(vjp(false, g_after_provided));
    expect_close(g_cached, g_default);
    expect_close(g_provided, g_default);
    expect_close(g_after_provided, g_default);

    Model d = random_direction(model, -1);
    auto jvp = [&](bool provided, piqp::Variables<T>& out) {
        piqp::Status status = provided
            ? solver.jvp(d.P, d.c, d.A, d.b, d.G, d.h_l, d.h_u, d.x_l, d.x_u, solve_result)
            : solver.jvp(d.P, d.c, d.A, d.b, d.G, d.h_l, d.h_u, d.x_l, d.x_u);
        ASSERT_EQ(status, piqp::Status::PIQP_SOLVED);
        out = solver.jvp_result();
    };
    auto expect_close_tangents = [](const piqp::Variables<T>& t1, const piqp::Variables<T>& t2) {
        for (const OutputField<T>& f : output_fields<T>())
        {
            T scale = (std::max)(T(1), (t2.*f.field).template lpNorm<Eigen::Infinity>());
            EXPECT_LE((t1.*f.field - t2.*f.field).template lpNorm<Eigen::Infinity>(), T(1e-9) * scale) << f.name;
        }
    };

    piqp::Variables<T> t_default, t_cached, t_provided, t_after_provided;
    ASSERT_NO_FATAL_FAILURE(jvp(false, t_default));
    ASSERT_NO_FATAL_FAILURE(jvp(false, t_cached));
    ASSERT_NO_FATAL_FAILURE(jvp(true, t_provided));
    ASSERT_NO_FATAL_FAILURE(jvp(false, t_after_provided));
    expect_close_tangents(t_cached, t_default);
    expect_close_tangents(t_provided, t_default);
    expect_close_tangents(t_after_provided, t_default);
}

// Invalid provided results are rejected: a solve result which is not solved with PIQP_UNSOLVED,
// a malformed one with PIQP_INVALID_SETTINGS.
template<typename Solver, typename Model>
void run_invalid_result_test(Solver& solver, Model& model)
{
    using T = typename Model::T;

    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    piqp::Variables<T> w = piqp::rand::variables_rand<T>(model.P.rows(), model.A.rows(), model.G.rows());

    piqp::optional<piqp::Result<T>> r = solver.result();
    r->info.status = piqp::Status::PIQP_PRIMAL_INFEASIBLE;
    EXPECT_EQ(solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_UNSOLVED);

    r = solver.result();
    r->x.resize(r->x.size() + 1);
    EXPECT_EQ(solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_INVALID_SETTINGS);

    r = solver.result();
    r->info.rho = T(0);
    EXPECT_EQ(solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_INVALID_SETTINGS);

    // non-positive dual on a finite bound
    piqp::isize i_finite = 0;
    while (i_finite < model.h_l.rows() && model.h_l(i_finite) <= -PIQP_INF) i_finite++;
    ASSERT_LT(i_finite, model.h_l.rows());
    r = solver.result();
    r->z_l(i_finite) = T(0);
    EXPECT_EQ(solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_INVALID_SETTINGS);

    r = solver.result();
    EXPECT_EQ(solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_SOLVED);
    EXPECT_EQ(solver.vjp_result().status, piqp::Status::PIQP_SOLVED);

    // a rejected solve result invalidates the previously computed gradients
    r->info.delta = T(-1);
    EXPECT_EQ(solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_INVALID_SETTINGS);
    EXPECT_EQ(solver.vjp_result().status, piqp::Status::PIQP_INVALID_SETTINGS);
}

/*
 * Central difference of the solution along d, i.e. the JVP J d. The solution map is only differentiable
 * within a fixed active set, which is asserted for both perturbed problems.
 */
template<typename Solver, typename Model>
void finite_difference_jvp(Solver& solver, const Model& model, const Model& d, const piqp::rand::ActiveSet& active_set,
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
 * Compares J d from vjp() and jvp() with central finite differences of the solution along random unit
 * directions d, for each data block separately and for all blocks at once. vjp() computes J^T w,
 * so the k-th entry of J d is <J^T e_k, d>, obtained from vjp() with the unit cotangent e_k.
 * jvp() computes J d directly. The model has to be generated with a known active set, which is
 * checked at the solution.
 */
template<typename Solver, typename Model>
void run_finite_difference_test(Solver& solver, const Model& model, const piqp::rand::ActiveSet& active_set)
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

    // gradients are preallocated in setup(), vjp() must not allocate
    PIQP_EIGEN_MALLOC_NOT_ALLOWED();
    ASSERT_EQ(solver.vjp(e.x, e.y, e.z_l, e.z_u, e.z_bl, e.z_bu, e.s_l, e.s_u, e.s_bl, e.s_bu), piqp::Status::PIQP_SOLVED);
    ASSERT_EQ(solver.vjp(e.x, e.y, e.z_l, e.z_u, e.z_bl, e.z_bu, e.s_l, e.s_u, e.s_bl, e.s_bu), piqp::Status::PIQP_SOLVED);
    PIQP_EIGEN_MALLOC_ALLOWED();
    e.x(0) = T(0);
    const auto& g = solver.vjp_result();

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
            ASSERT_EQ(solver.vjp(e.x, e.y, e.z_l, e.z_u, e.z_bl, e.z_bu, e.s_l, e.s_u, e.s_bl, e.s_bu), piqp::Status::PIQP_SOLVED);
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

    // jvp() along the same directions, compared with the finite differences
    for (std::size_t j = 0; j < directions.size(); j++)
    {
        SCOPED_TRACE("block " + std::to_string(blocks[j]));
        const Model& d = directions[j];

        // jvp() must not allocate
        PIQP_EIGEN_MALLOC_NOT_ALLOWED();
        ASSERT_EQ(solver.jvp(d.P, d.c, d.A, d.b, d.G, d.h_l, d.h_u, d.x_l, d.x_u), piqp::Status::PIQP_SOLVED);
        PIQP_EIGEN_MALLOC_ALLOWED();
        const piqp::JVPResult<T>& t = solver.jvp_result();
        EXPECT_EQ(t.status, piqp::Status::PIQP_SOLVED);

        T t_norm = 0;
        for (const OutputField<T>& f : output_fields<T>())
        {
            t_norm += (t.*f.field).squaredNorm();
        }
        T tol = T(1e-5) * (T(1) + std::sqrt(t_norm));

        for (const OutputField<T>& f : output_fields<T>())
        {
            for (piqp::isize k = 0; k < (t.*f.field).rows(); k++)
            {
                // infinite slacks are not differentiable, their tangents are zero
                if ((result.*f.field)(k) >= PIQP_INF)
                {
                    EXPECT_EQ((t.*f.field)(k), T(0)) << f.name << " " << k;
                    continue;
                }
                EXPECT_NEAR((t.*f.field)(k), (jvps_fd[j].*f.field)(k), tol) << f.name << " " << k;
            }
        }
    }
}

// copy of the gradients of vjp_result(), as dense matrices
template<typename T>
struct GradientsCopy
{
    piqp::Mat<T> P, A, G;
    piqp::Vec<T> c, b, h_l, h_u, x_l, x_u;

    template<typename VJPResult>
    explicit GradientsCopy(const VJPResult& g)
        : P(g.P), A(g.A), G(g.G), c(g.c), b(g.b), h_l(g.h_l), h_u(g.h_u), x_l(g.x_l), x_u(g.x_u) {}

    T max_diff(const GradientsCopy& o) const
    {
        T d = 0;
        d = (std::max)(d, (P - o.P).template lpNorm<Eigen::Infinity>());
        d = (std::max)(d, (A - o.A).template lpNorm<Eigen::Infinity>());
        d = (std::max)(d, (G - o.G).template lpNorm<Eigen::Infinity>());
        d = (std::max)(d, (c - o.c).template lpNorm<Eigen::Infinity>());
        d = (std::max)(d, (b - o.b).template lpNorm<Eigen::Infinity>());
        d = (std::max)(d, (h_l - o.h_l).template lpNorm<Eigen::Infinity>());
        d = (std::max)(d, (h_u - o.h_u).template lpNorm<Eigen::Infinity>());
        d = (std::max)(d, (x_l - o.x_l).template lpNorm<Eigen::Infinity>());
        d = (std::max)(d, (x_u - o.x_u).template lpNorm<Eigen::Infinity>());
        return d;
    }
};

/*
 * vjp() and jvp() without a solve result reuse the last linearization, i.e. the linearization point including the
 * derivative smoothing and the KKT factorization:
 *  - with smoothing, repeated calls do not smooth again, which is detected with derivative_smoothing_max_iter = 1,
 *    which is not enough to converge for this problem,
 *  - changing derivative_smoothing_mu or derivative_smoothing linearizes the result of the last solve() again,
 *  - a provided solve result is reused by the following calls without one, until the smoothing settings change,
 *  - a solve() or a rejected solve result starts over from the result of the last solve(),
 *  - without a differentiable solve(), only a provided solve result can be linearized.
 */
template<typename Solver, typename Model>
void run_linearization_reuse_test(Solver& solver, const Model& model)
{
    using T = typename Model::T;
    using piqp::Status;
    piqp::Variables<T> w = piqp::rand::variables_rand<T>(model.P.rows(), model.A.rows(), model.G.rows());
    Model d = random_direction(model, -1);
    auto vjp = [&](const piqp::optional<piqp::Result<T>>& r) {
        return solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r);
    };
    auto jvp = [&](const piqp::optional<piqp::Result<T>>& r) {
        return solver.jvp(d.P, d.c, d.A, d.b, d.G, d.h_l, d.h_u, d.x_l, d.x_u, r);
    };

    const piqp::Settings<T> settings = solver.settings();
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    const piqp::optional<piqp::Result<T>> last = solver.result();
    // a stored result at a different point than the result of the last solve()
    piqp::optional<piqp::Result<T>> stored = solver.result();
    stored->x += T(1e-3) * piqp::rand::vector_rand<T>(model.P.rows());

    // smoothing runs once
    solver.settings().derivative_smoothing = true;
    ASSERT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    const GradientsCopy<T> g_smoothed(solver.vjp_result());
    solver.settings().derivative_smoothing_max_iter = 1;
    ASSERT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    EXPECT_EQ(GradientsCopy<T>(solver.vjp_result()).max_diff(g_smoothed), T(0));
    ASSERT_EQ(jvp(piqp::nullopt), Status::PIQP_SOLVED);

    // a different mu smooths again, which does not converge with one iteration
    solver.settings().derivative_smoothing_mu = T(2) * settings.derivative_smoothing_mu;
    EXPECT_EQ(jvp(piqp::nullopt), Status::PIQP_MAX_ITER_REACHED);
    EXPECT_EQ(vjp(piqp::nullopt), Status::PIQP_MAX_ITER_REACHED);
    solver.settings().derivative_smoothing_max_iter = settings.derivative_smoothing_max_iter;
    ASSERT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    EXPECT_GT(GradientsCopy<T>(solver.vjp_result()).max_diff(g_smoothed), T(0));
    solver.settings().derivative_smoothing_mu = settings.derivative_smoothing_mu;
    ASSERT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    EXPECT_LE(GradientsCopy<T>(solver.vjp_result()).max_diff(g_smoothed), T(1e-12));

    // disabling smoothing linearizes the result of the last solve() again
    solver.settings().derivative_smoothing = false;
    ASSERT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    const GradientsCopy<T> g_last(solver.vjp_result());
    ASSERT_EQ(vjp(last), Status::PIQP_SOLVED);
    EXPECT_LE(GradientsCopy<T>(solver.vjp_result()).max_diff(g_last), T(1e-12));

    // a provided solve result is reused
    ASSERT_EQ(vjp(stored), Status::PIQP_SOLVED);
    const GradientsCopy<T> g_stored(solver.vjp_result());
    EXPECT_GT(g_stored.max_diff(g_last), T(0));
    ASSERT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    EXPECT_EQ(GradientsCopy<T>(solver.vjp_result()).max_diff(g_stored), T(0));

    // until the smoothing settings change, which falls back to the result of the last solve()
    solver.settings().derivative_smoothing = true;
    ASSERT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    EXPECT_LE(GradientsCopy<T>(solver.vjp_result()).max_diff(g_smoothed), T(1e-12));
    solver.settings().derivative_smoothing = false;
    ASSERT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    EXPECT_LE(GradientsCopy<T>(solver.vjp_result()).max_diff(g_last), T(1e-12));

    // a rejected solve result starts over from the result of the last solve()
    piqp::optional<piqp::Result<T>> rejected = stored;
    rejected->info.rho = T(0);
    EXPECT_EQ(vjp(rejected), Status::PIQP_INVALID_SETTINGS);
    ASSERT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    EXPECT_LE(GradientsCopy<T>(solver.vjp_result()).max_diff(g_last), T(1e-12));

    // so does a solve()
    ASSERT_EQ(vjp(stored), Status::PIQP_SOLVED);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    ASSERT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    EXPECT_LE(GradientsCopy<T>(solver.vjp_result()).max_diff(g_last), T(1e-12));

    // without a solve(), a provided solve result is reused until the smoothing settings change
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    EXPECT_EQ(vjp(piqp::nullopt), Status::PIQP_UNSOLVED);
    ASSERT_EQ(vjp(stored), Status::PIQP_SOLVED);
    ASSERT_EQ(jvp(piqp::nullopt), Status::PIQP_SOLVED);
    EXPECT_EQ(vjp(piqp::nullopt), Status::PIQP_SOLVED);
    EXPECT_EQ(GradientsCopy<T>(solver.vjp_result()).max_diff(g_stored), T(0));
    solver.settings().derivative_smoothing = true;
    EXPECT_EQ(vjp(piqp::nullopt), Status::PIQP_UNSOLVED);
    EXPECT_EQ(jvp(piqp::nullopt), Status::PIQP_UNSOLVED);
    solver.settings().derivative_smoothing = false;
}

/*
 * vjp() and jvp() solve the same linearized KKT system, i.e., <vjp(w), d> = <w, jvp(d)> for all cotangents w and
 * data perturbations d. At the default regularization floor, the KKT matrix is ill-conditioned and both are only
 * accurate to about machine precision / delta, i.e. about 1e-6. Hence, the identity is checked with a raised
 * regularization floor, where it has to hold up to round-off. Checked at the last solve() and at a provided solve
 * result, with and without derivative smoothing, and with cotangents on all outputs, including infinite bounds
 * and disabled rows, whose cotangents have no effect.
 */
template<typename Solver, typename Model>
void run_adjoint_identity_test(Solver& solver, const Model& model)
{
    using T = typename Model::T;
    solver.settings().reg_lower_limit = T(1e-4);
    solver.settings().reg_finetune_lower_limit = T(1e-4);
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    const piqp::optional<piqp::Result<T>> solve_result = solver.result();

    for (bool smoothing : {false, true})
    {
        solver.settings().derivative_smoothing = smoothing;
        for (bool provided : {false, true})
        {
            SCOPED_TRACE(std::string(smoothing ? "with" : "without") + " smoothing, " + (provided ? "provided" : "default") + " result");
            const piqp::optional<piqp::Result<T>>& r = provided ? solve_result : piqp::nullopt;
            for (int block = -1; block < 9; block++)
            {
                piqp::Variables<T> w = piqp::rand::variables_rand<T>(model.P.rows(), model.A.rows(), model.G.rows());
                Model d = random_direction(model, block);

                ASSERT_EQ(solver.vjp(w.x, w.y, w.z_l, w.z_u, w.z_bl, w.z_bu, w.s_l, w.s_u, w.s_bl, w.s_bu, r), piqp::Status::PIQP_SOLVED);
                T vjp_d = directional_derivative(solver.vjp_result(), d);

                ASSERT_EQ(solver.jvp(d.P, d.c, d.A, d.b, d.G, d.h_l, d.h_u, d.x_l, d.x_u, r), piqp::Status::PIQP_SOLVED);
                T w_jvp = 0;
                for (const OutputField<T>& f : output_fields<T>())
                {
                    w_jvp += (w.*f.field).dot(solver.jvp_result().*f.field);
                }

                EXPECT_NEAR(vjp_d, w_jvp, T(1e-10) * (T(1) + std::abs(vjp_d))) << "block " << block;
            }
        }
    }
    solver.settings().derivative_smoothing = false;
}

template<typename T>
void add_strictly_lower_values(piqp::Mat<T>& M)
{
    M += piqp::Mat<T>(piqp::rand::dense_matrix_rand<T>(M.rows(), M.cols()).template triangularView<Eigen::StrictlyLower>());
}

template<typename T, typename I>
void add_strictly_lower_values(piqp::SparseMat<T, I>& M)
{
    piqp::Mat<T> L = piqp::rand::dense_matrix_rand<T>(M.rows(), M.cols()).template triangularView<Eigen::StrictlyLower>();
    M = piqp::SparseMat<T, I>(M + L.sparseView());
    M.makeCompressed();
}

// sets the values of row i of M, keeping the sparsity pattern
template<typename T>
void set_row_values(piqp::Mat<T>& M, piqp::isize i, T value)
{
    M.row(i).setConstant(value);
}

template<typename T, typename I>
void set_row_values(piqp::SparseMat<T, I>& M, piqp::isize i, T value)
{
    for (piqp::isize j = 0; j < M.outerSize(); j++)
    {
        for (typename piqp::SparseMat<T, I>::InnerIterator it(M, j); it; ++it)
        {
            if (it.row() == i) it.valueRef() = value;
        }
    }
}

/*
 * jvp() only reads the upper triangular part of dP, as setup() for P, and ignores the entries of dh_l, dh_u, dx_l
 * and dx_u at infinite bounds and the rows of dG of constraints with both bounds infinite.
 */
template<typename Solver, typename Model>
void run_jvp_ignored_inputs_test(Solver& solver, const Model& model)
{
    using T = typename Model::T;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);

    Model d = random_direction(model, -1);
    ASSERT_EQ(solver.jvp(d.P, d.c, d.A, d.b, d.G, d.h_l, d.h_u, d.x_l, d.x_u), piqp::Status::PIQP_SOLVED);
    const piqp::Variables<T> reference = solver.jvp_result();

    Model e = d;
    add_strictly_lower_values(e.P);
    const T garbage = T(1e3);
    bool has_infinite_bound = false;
    bool has_disabled_row = false;
    for (piqp::isize i = 0; i < model.G.rows(); i++)
    {
        if (model.h_l(i) <= -PIQP_INF) { e.h_l(i) = garbage; has_infinite_bound = true; }
        if (model.h_u(i) >= PIQP_INF) { e.h_u(i) = garbage; has_infinite_bound = true; }
        if (model.h_l(i) <= -PIQP_INF && model.h_u(i) >= PIQP_INF) { set_row_values(e.G, i, garbage); has_disabled_row = true; }
    }
    for (piqp::isize i = 0; i < model.P.rows(); i++)
    {
        if (model.x_l(i) <= -PIQP_INF) { e.x_l(i) = garbage; has_infinite_bound = true; }
        if (model.x_u(i) >= PIQP_INF) { e.x_u(i) = garbage; has_infinite_bound = true; }
    }
    ASSERT_TRUE(has_infinite_bound);
    ASSERT_TRUE(has_disabled_row);

    ASSERT_EQ(solver.jvp(e.P, e.c, e.A, e.b, e.G, e.h_l, e.h_u, e.x_l, e.x_u), piqp::Status::PIQP_SOLVED);
    for (const OutputField<T>& f : output_fields<T>())
    {
        T scale = (std::max)(T(1), (reference.*f.field).template lpNorm<Eigen::Infinity>());
        EXPECT_LE((solver.jvp_result().*f.field - reference.*f.field).template lpNorm<Eigen::Infinity>(), T(1e-12) * scale) << f.name;
    }
}

/*
 * min 0.5 x^2 - a x  s.t.  x >= x_l = 0, with derivative smoothing the relaxed KKT conditions are
 *   x - a - z = 0,  x - x_l - s = 0,  s z = mu   =>   x = (a + sqrt(a^2 + 4 mu)) / 2,  s = x,  z = x - a,
 * such that with D = sqrt(a^2 + 4 mu) = s + z
 *   dx/da = 0.5 (1 + a / D),  dx/dx_l = z / (s + z),  ds/da = dx/da,  ds/dx_l = dx/dx_l - 1.
 * With the cotangents w_x on x and w_s on s_bl, i.e. L = w_x x + w_s s, this gives
 *   dL/dc = -(w_x + w_s) dx/da,  dL/dx_l = w_x dx/dx_l + w_s (dx/dx_l - 1),  dL/dP = x dL/dc,
 * since P and c only enter through P x + c. Checked with and without cost scaling, for a on the active side,
 * at the kink of the nonsmoothed solution and on the inactive side, and with and without slack cotangent, which
 * enters the adjoint right-hand side multiplied by the smoothed slack. The same derivatives are checked with jvp(),
 * e.g., along dc = -1, i.e. da = 1, the tangents are dx = ds = dx/da and dz = dx - da.
 * P is the 1x1 identity in the matrix type of the solver.
 */
template<typename Solver, typename MatP>
void run_smoothing_analytic_test(Solver& solver, const MatP& P)
{
    using T = typename MatP::Scalar;
    const T mu = 1e-3;
    const T w_x = 1;
    solver.settings().derivative_smoothing = true;
    solver.settings().derivative_smoothing_mu = mu;

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

                piqp::Vec<T> grad_x = piqp::Vec<T>::Constant(1, w_x);
                piqp::Vec<T> grad_s_bl = piqp::Vec<T>::Constant(1, w_s);
                ASSERT_EQ(solver.vjp(grad_x, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt,
                                          piqp::nullopt, piqp::nullopt, grad_s_bl), piqp::Status::PIQP_SOLVED);
                ASSERT_EQ(solver.vjp_result().status, piqp::Status::PIQP_SOLVED);

                T D = std::sqrt(a * a + 4 * mu);
                T x = (a + D) / 2;
                T s = x;
                T z = x - a;
                T dx_da = T(0.5) * (1 + a / D);
                T dx_dxl = z / (s + z);
                T dL_dc = -(w_x + w_s) * dx_da;

                // vjp() differentiates the KKT system regularized with the final delta of the forward solve,
                // which is larger at the kink (degenerate forward problem), so the error is O(delta)
                T tol = T(1e-7) + solver.result().info.delta;
                const auto& g = solver.vjp_result();
                EXPECT_NEAR(g.c(0), dL_dc, tol);
                EXPECT_NEAR(g.x_l(0), w_x * dx_dxl + w_s * (dx_dxl - 1), tol);
                EXPECT_NEAR(g.P.coeff(0, 0), x * dL_dc, tol);

                // along c: dc = -1, i.e. da = 1
                piqp::Vec<T> dc = piqp::Vec<T>::Constant(1, T(-1));
                ASSERT_EQ(solver.jvp(piqp::nullopt, dc), piqp::Status::PIQP_SOLVED);
                const piqp::JVPResult<T>& t_c = solver.jvp_result();
                EXPECT_NEAR(t_c.x(0), dx_da, tol);
                EXPECT_NEAR(t_c.s_bl(0), dx_da, tol);
                EXPECT_NEAR(t_c.z_bl(0), dx_da - 1, tol);

                // along x_l: dx_l = 1
                piqp::Vec<T> dx_l = piqp::Vec<T>::Constant(1, T(1));
                ASSERT_EQ(solver.jvp(piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt,
                                     piqp::nullopt, piqp::nullopt, dx_l), piqp::Status::PIQP_SOLVED);
                const piqp::JVPResult<T>& t_xl = solver.jvp_result();
                EXPECT_NEAR(t_xl.x(0), dx_dxl, tol);
                EXPECT_NEAR(t_xl.s_bl(0), dx_dxl - 1, tol);
                EXPECT_NEAR(t_xl.z_bl(0), dx_dxl, tol);
            }
        }
    }
}

/*
 * Derivative smoothing with an inequality row with h_l == h_u and a variable with x_l == x_u, which have no
 * interior and are not smoothed. The gradients have to match the ones of the same problem with both
 * constraints given as equality constraints, where only the sums dh_l + dh_u and dx_l + dx_u are defined.
 */
template<typename Solver, typename Model>
void run_smoothing_equality_test(Solver& solver, Solver& reference, const Model& model)
{
    using T = typename Model::T;

    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), piqp::Status::PIQP_SOLVED);
    piqp::Vec<T> x = solver.result().x;
    piqp::Mat<T> G_dense = piqp::Mat<T>(model.G);
    T g_x = G_dense.row(0).dot(x);

    Model ineq = model;
    ineq.h_l(0) = g_x;
    ineq.h_u(0) = g_x;
    ineq.x_l(0) = x(0);
    ineq.x_u(0) = x(0);

    Model eq = model;
    piqp::Mat<T> rows = piqp::Mat<T>::Zero(2, x.rows());
    rows.row(0) = G_dense.row(0);
    rows(1, 0) = T(1);
    eq.A = append_rows(model.A, rows);
    eq.b.conservativeResize(model.b.rows() + 2);
    eq.b.tail(2) << g_x, x(0);
    eq.h_l(0) = -std::numeric_limits<T>::infinity();
    eq.h_u(0) = std::numeric_limits<T>::infinity();
    eq.x_l(0) = -std::numeric_limits<T>::infinity();
    eq.x_u(0) = std::numeric_limits<T>::infinity();

    piqp::Vec<T> grad_x = piqp::Vec<T>::Ones(x.rows());
    for (Solver* s : {&solver, &reference})
    {
        const Model& m = s == &solver ? ineq : eq;
        s->settings().derivative_smoothing = true;
        s->setup(m.P, m.c, m.A, m.b, m.G, m.h_l, m.h_u, m.x_l, m.x_u);
        ASSERT_EQ(s->solve(), piqp::Status::PIQP_SOLVED);
        ASSERT_EQ(s->vjp(grad_x), piqp::Status::PIQP_SOLVED);
    }

    const auto& g_ineq = solver.vjp_result();
    const auto& g_eq = reference.vjp_result();
    piqp::isize p = model.b.rows();
    T tol = T(1e-5) * (std::max)(T(1), g_eq.c.template lpNorm<Eigen::Infinity>());
    EXPECT_LT((g_ineq.c - g_eq.c).template lpNorm<Eigen::Infinity>(), tol);
    EXPECT_NEAR(g_ineq.h_l(0) + g_ineq.h_u(0), g_eq.b(p), tol);
    EXPECT_NEAR(g_ineq.x_l(0) + g_ineq.x_u(0), g_eq.b(p + 1), tol);
}

#endif //PIQP_TESTS_DERIVATIVE_UTILS_HPP
