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
#include "backward_utils.hpp"

using namespace piqp;

using T = double;
using I = int;

namespace
{

std::pair<dense::Model<T>, ActiveSet> make_qp(unsigned int seed)
{
    // same problem in every test, independent of the test order
    // (the distributions are reset as well, normal_distribution caches values)
    rand::gen.seed(seed);
    rand::uniform_dist.reset();
    rand::uniform_dist_pos.reset();
    rand::normal_dist.reset();
    return dense_strongly_convex_qp_with_known_active_set<T>(12, 2, 8);
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
    solver.settings().enable_backward = true;
}

} // namespace

TEST(DenseBackwardTest, FiniteDifferences)
{
    for (unsigned int seed : {1, 2, 3, 4, 5})
    {
        SCOPED_TRACE("seed " + std::to_string(seed));
        std::pair<dense::Model<T>, ActiveSet> qp = make_qp(seed);
        DenseSolver<T> solver;
        configure(solver);
        run_finite_difference_test(solver, qp.first, qp.second);
    }
}

TEST(DenseBackwardTest, InvalidStates)
{
    dense::Model<T> model = make_model();
    DenseSolver<T> solver;
    configure(solver);

    // not setup
    EXPECT_EQ(solver.backward(), BackwardStatus::PIQP_BWD_INVALID_FORWARD_RESULT);
    EXPECT_FALSE(solver.gradients_valid());

    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    // not solved
    EXPECT_EQ(solver.backward(), BackwardStatus::PIQP_BWD_INVALID_FORWARD_RESULT);
    EXPECT_FALSE(solver.gradients_valid());

    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.backward(), BackwardStatus::PIQP_BWD_SOLVED);
    EXPECT_TRUE(solver.gradients_valid());

    // wrong dimensions, each adjoint is checked; a failed call invalidates the previous gradients
    Vec<T> wrong_n = Vec<T>::Zero(model.P.rows() + 1);
    Vec<T> wrong_p = Vec<T>::Zero(model.A.rows() + 1);
    Vec<T> wrong_m = Vec<T>::Zero(model.G.rows() + 1);
    EXPECT_EQ(solver.backward(wrong_n), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);
    EXPECT_FALSE(solver.gradients_valid());
    EXPECT_EQ(solver.backward(nullopt, wrong_p), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, wrong_m), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, wrong_m), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, wrong_n), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_m), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_m), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);
    EXPECT_EQ(solver.backward(nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, nullopt, wrong_n), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);
    EXPECT_FALSE(solver.gradients_valid());

    EXPECT_EQ(solver.backward(), BackwardStatus::PIQP_BWD_SOLVED);
    EXPECT_TRUE(solver.gradients_valid());

    // data changed since last solve, update() invalidates the gradients
    solver.update(nullopt, model.c);
    EXPECT_FALSE(solver.gradients_valid());
    EXPECT_EQ(solver.backward(), BackwardStatus::PIQP_BWD_INVALID_FORWARD_RESULT);

    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.backward(), BackwardStatus::PIQP_BWD_SOLVED);
    EXPECT_TRUE(solver.gradients_valid());

    // setup() invalidates the gradients
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    EXPECT_FALSE(solver.gradients_valid());
}

TEST(DenseBackwardTest, SmoothingAnalytic)
{
    Mat<T> P = Mat<T>::Identity(1, 1);
    DenseSolver<T> solver;
    configure(solver);
    run_smoothing_analytic_test(solver, P);
}

/*
 * 2D QP coupled through one inequality
 *   min 0.5 ||x||^2 - a^T x  s.t.  g^T x <= h
 * With gradient smoothing the relaxed KKT conditions are
 *   x = a - z g,  s = h - g^T x,  s z = mu,
 * i.e. with q = ||g||^2 and r = h - g^T a, z solves q z^2 + r z - mu = 0:
 *   z = (-r + sqrt(r^2 + 4 q mu)) / (2 q),  x = a - z g.
 * Implicit differentiation of the quadratic, with D = 2 q z + r = sqrt(r^2 + 4 q mu), gives
 *   dz/dh = -z / D,  dz/da = z g / D,  dz/dg = (z a - 2 z^2 g) / D.
 * For the loss L = w^T x + v z this results in
 *   dL/dc = -(w + (v - g^T w) dz/da),  dL/dh = (v - g^T w) dz/dh,  dL/dg = -z w + (v - g^T w) dz/dg,
 * and dL/dP = 0.5 (lam_x x^T + x lam_x^T) with lam_x = dL/dc, since P and c only enter through P x + c.
 */
TEST(DenseBackwardTest, SmoothingAnalytic2D)
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
            solver.settings().gradient_smoothing = true;
            solver.settings().gradient_smoothing_mu = mu;
            solver.settings().gradient_smoothing_tol = 1e-10;
            solver.settings().gradient_smoothing_max_iter = 20;
            solver.setup(P, c, nullopt, nullopt, G, nullopt, h_u, nullopt, nullopt);
            ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);

            // the forward solution is the true optimum, i.e. the projection onto the half space
            Vec<T> x_opt = a;
            if (r < 0)
            {
                x_opt -= g * (-r / q);
            }
            EXPECT_LT((solver.result().x - x_opt).norm(), 1e-5) << "r = " << r << ", scale_cost = " << scale_cost;

            Vec<T> grad_z_u = Vec<T>::Constant(1, v);
            ASSERT_EQ(solver.backward(w, nullopt, nullopt, grad_z_u), BackwardStatus::PIQP_BWD_SOLVED);
            const dense::Gradients<T>& grads = solver.gradients();

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

// backward() requires enable_backward to be set before setup()
TEST(DenseBackwardTest, RequiresEnableBackward)
{
    dense::Model<T> model = make_model();
    DenseSolver<T> solver;
    configure(solver);
    solver.settings().enable_backward = false;
    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.backward(), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);

    // enabling it after setup() is not sufficient
    solver.settings().enable_backward = true;
    EXPECT_EQ(solver.backward(), BackwardStatus::PIQP_BWD_INVALID_SETTINGS);

    solver.setup(model.P, model.c, model.A, model.b, model.G, model.h_l, model.h_u, model.x_l, model.x_u);
    ASSERT_EQ(solver.solve(), Status::PIQP_SOLVED);
    EXPECT_EQ(solver.backward(), BackwardStatus::PIQP_BWD_SOLVED);
}

TEST(DenseBackwardTest, ProvidedResultInvalid)
{
    dense::Model<T> model = make_model();
    DenseSolver<T> solver;
    configure(solver);
    run_invalid_result_test(solver, model);
}

TEST(DenseBackwardTest, NoSideEffectsOnForward)
{
    dense::Model<T> model = make_model();
    DenseSolver<T> solver;
    DenseSolver<T> reference;
    configure(solver);
    configure(reference);
    run_no_side_effects_test(solver, reference, model);
}
