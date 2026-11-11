// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#define PIQP_EIGEN_CHECK_MALLOC

#include "piqp/piqp.hpp"
#include "piqp/utils/random_utils.hpp"

#include "gtest/gtest.h"
#include "derivative_utils.hpp"

using namespace piqp;

using T = double;
using I = int;

namespace
{

std::pair<dense::Model<T>, rand::ActiveSet> make_qp(unsigned int seed)
{
    // same problem in every test, independent of the test order
    rand::seed(seed);
    return rand::dense_strongly_convex_qp_with_known_active_set<T>(12, 2, 8);
}

dense::Model<T> make_model()
{
    return make_qp(42).first;
}

template<typename Solver>
void configure(Solver& solver, bool scale_cost = false)
{
    solver.settings().verbose = false;
    solver.settings().eps_abs = 1e-10;
    solver.settings().eps_rel = 1e-10;
    solver.settings().eps_duality_gap_abs = 1e-10;
    solver.settings().eps_duality_gap_rel = 1e-10;
    solver.settings().preconditioner_scale_cost = scale_cost;
    solver.settings().enable_derivatives = true;
}

} // namespace

TEST(DenseDerivativeTest, FiniteDifferences)
{
    for (unsigned int seed : {1, 2, 3, 4, 5})
    {
        SCOPED_TRACE("seed " + std::to_string(seed));
        std::pair<dense::Model<T>, rand::ActiveSet> qp = make_qp(seed);
        DenseSolver<T> solver;
        configure(solver);
        run_finite_difference_test(solver, qp.first, qp.second);
    }
}

TEST(DenseDerivativeTest, InvalidStates)
{
    dense::Model<T> model = make_model();
    DenseSolver<T> solver;
    configure(solver);
    run_invalid_states_test(solver, model);
}

TEST(DenseDerivativeTest, NoConstraints)
{
    DenseSolver<T> solver;
    configure(solver);
    run_no_constraints_test(solver, rand::dense_positive_definite_upper_triangular_rand<T>(4, T(1)));
}

TEST(DenseDerivativeTest, SmoothingAnalytic)
{
    Mat<T> P = Mat<T>::Identity(1, 1);
    DenseSolver<T> solver;
    configure(solver);
    run_smoothing_analytic_test(solver, P);
}

/*
 * 2D QP coupled through one inequality
 *   min 0.5 ||x||^2 - a^T x  s.t.  g^T x <= h
 * With derivative smoothing the relaxed KKT conditions are
 *   x = a - z g,  s = h - g^T x,  s z = mu,
 * i.e. with q = ||g||^2 and r = h - g^T a, z solves q z^2 + r z - mu = 0:
 *   z = (-r + sqrt(r^2 + 4 q mu)) / (2 q),  x = a - z g.
 * Implicit differentiation of the quadratic, with D = 2 q z + r = sqrt(r^2 + 4 q mu), gives
 *   dz/dh = -z / D,  dz/da = z g / D,  dz/dg = (z a - 2 z^2 g) / D.
 * For the loss L = w^T x + v z this results in
 *   dL/dc = -(w + (v - g^T w) dz/da),  dL/dh = (v - g^T w) dz/dh,  dL/dg = -z w + (v - g^T w) dz/dg,
 * and dL/dP = 0.5 (lam_x x^T + x lam_x^T) with lam_x = dL/dc, since P and c only enter through P x + c.
 */
TEST(DenseDerivativeTest, SmoothingAnalytic2D)
{
    T mu = 1e-3;
    Vec<T> g(2);
    g << T(1), T(2);
    Vec<T> a(2);
    a << T(0.3), T(-0.1);
    Vec<T> w(2);
    w << T(0.7), T(-1.3);
    T v = T(0.4);
    T q = g.squaredNorm();

    for (bool scale_cost : {false, true})
    {
        // r = h - g^T a: constraint active (r < 0), at the kink (r = 0), inactive (r > 0)
        for (T r : {T(-0.05), T(0), T(0.05)})
        {
            T h = g.dot(a) + r;

            Mat<T> P = Mat<T>::Identity(2, 2);
            Vec<T> c = -a;
            Mat<T> G = g.transpose();
            Vec<T> h_u = Vec<T>::Constant(1, h);

            DenseSolver<T> solver;
            configure(solver, scale_cost);
            solver.settings().derivative_smoothing = true;
            solver.settings().derivative_smoothing_mu = mu;
            solver.setup(P, c, nullopt, nullopt, G, nullopt, h_u, nullopt, nullopt);
            ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);

            Vec<T> grad_z_u = Vec<T>::Constant(1, v);
            ASSERT_EQ(solver.vjp(w, nullopt, nullopt, grad_z_u), Status::PIQP_SOLVED);
            const dense::VJPResult<T>& grads = solver.vjp_result();

            T z = (-r + std::sqrt(r * r + 4 * q * mu)) / (2 * q);
            Vec<T> x = a - z * g;
            T D = std::sqrt(r * r + 4 * q * mu);
            Vec<T> dz_da = z * g / D;
            T dz_dh = -z / D;
            Vec<T> dz_dg = (z * a - 2 * z * z * g) / D;
            T coupling = v - g.dot(w);

            Vec<T> dL_dc = -(w + coupling * dz_da);
            T dL_dh = coupling * dz_dh;
            Vec<T> dL_dg = -z * w + coupling * dz_dg;
            Mat<T> dL_dP = T(0.5) * (dL_dc * x.transpose() + x * dL_dc.transpose());

            EXPECT_LT((grads.c - dL_dc).norm(), 1e-6) << "r = " << r << ", scale_cost = " << scale_cost;
            EXPECT_NEAR(grads.h_u(0), dL_dh, 1e-6) << "r = " << r << ", scale_cost = " << scale_cost;
            EXPECT_EQ(grads.h_l(0), T(0)) << "r = " << r << ", scale_cost = " << scale_cost;
            EXPECT_LT((grads.G.row(0).transpose() - dL_dg).norm(), 1e-6) << "r = " << r << ", scale_cost = " << scale_cost;
            EXPECT_LT((grads.P - dL_dP).norm(), 1e-6) << "r = " << r << ", scale_cost = " << scale_cost;
        }
    }
}

TEST(DenseDerivativeTest, SmoothingEqualities)
{
    DenseSolver<T> solver;
    DenseSolver<T> reference;
    configure(solver);
    configure(reference);
    run_smoothing_equality_test(solver, reference, make_model());
}

TEST(DenseDerivativeTest, ProvidedResultInvalid)
{
    dense::Model<T> model = make_model();
    DenseSolver<T> solver;
    configure(solver);
    run_invalid_result_test(solver, model);
}

TEST(DenseDerivativeTest, DefaultMatchesProvidedResult)
{
    dense::Model<T> model = make_model();
    DenseSolver<T> solver;
    configure(solver);
    run_default_matches_provided_test(solver, model);
}

TEST(DenseDerivativeTest, NoSideEffectsOnSolve)
{
    dense::Model<T> model = make_model();
    DenseSolver<T> solver;
    DenseSolver<T> reference;
    configure(solver);
    configure(reference);
    run_no_side_effects_test(solver, reference, model);
}

TEST(DenseDerivativeTest, CopyMove)
{
    dense::Model<T> model = make_model();
    DenseSolver<T> solver;
    configure(solver);
    run_copy_move_test(solver, model);
}
