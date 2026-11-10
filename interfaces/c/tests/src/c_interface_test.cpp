// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <cstdlib>

#include "piqp.h"

#include "gtest/gtest.h"

/*
 * first QP:
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 1, x1 = 2 x2
 *
 * second QP:
 * min 4 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 2, x1 = 3 x2
*/
TEST(CInterfaceTest, SimpleDenseQPWithUpdate)
{
    piqp_int n = 2;
    piqp_int p = 1;
    piqp_int m = 3;

    piqp_float P[4] = {6, 0, 0, 4};
    piqp_float c[2] = {-1, -4};

    piqp_float A[2] = {1, -2};
    piqp_float b[1] = {0};

    piqp_float G[6] = {1, 0, 1, 0, 1, 0};
    piqp_float h_l[3] = {-1, -PIQP_INF, -2};
    piqp_float h_u[3] = {PIQP_INF, 1, 2};

    piqp_float x_l[2] = {-PIQP_INF, -1};
    piqp_float x_u[2] = {PIQP_INF, 1};

    piqp_workspace* work;
    piqp_settings* settings = (piqp_settings*) malloc(sizeof(piqp_settings));
    piqp_data_dense* data = (piqp_data_dense*) malloc(sizeof(piqp_data_dense));

    piqp_set_default_settings_dense(settings);
    settings->verbose = 1;

    data->n = n;
    data->p = p;
    data->m = m;
    data->P = P;
    data->c = c;
    data->A = A;
    data->b = b;
    data->G = G;
    data->h_l = h_l;
    data->h_u = h_u;
    data->x_l = x_l;
    data->x_u = x_u;

    piqp_setup_dense(&work, data, settings);
    piqp_status status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->result->x[0], 0.4285714, 1e-6);
    ASSERT_NEAR(work->result->x[1], 0.2142857, 1e-6);
    ASSERT_NEAR(work->result->y[0], -1.5714286, 1e-6);
    ASSERT_NEAR(work->result->z_l[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[1], 0, 1e-6);

    P[0] = 8;
    A[1] = -3;
    h_u[0] = 2;
    x_u[1] = 2;

    piqp_update_dense(work, data->P, NULL, data->A, NULL, NULL, NULL, data->h_u, NULL, data->x_u);
    status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->result->x[0], 0.2763157, 1e-6);
    ASSERT_NEAR(work->result->x[1], 0.0921056, 1e-6);
    ASSERT_NEAR(work->result->y[0], -1.2105263, 1e-6);
    ASSERT_NEAR(work->result->z_l[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[1], 0, 1e-6);

    piqp_cleanup(work);
    if (settings) free(settings);
    if (data) free(data);
}

/*
 * first QP:
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 1, x1 = 2 x2
 *
 * second QP:
 * min 4 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. -1 <= x <= 2, x1 = 3 x2
*/
TEST(CInterfaceTest, SimpleSparseQPWithUpdate)
{
    piqp_int n = 2;
    piqp_int p = 1;
    piqp_int m = 3;

    piqp_float P_x[2] = {6, 4};
    piqp_int P_nnz = 2;
    piqp_int P_p[3] = {0, 1, 2};
    piqp_int P_i[2] = {0, 1};
    piqp_float c[2] = {-1, -4};

    piqp_float A_x[2] = {1, -2};
    piqp_int A_nnz = 2;
    piqp_int A_p[3] = {0, 1, 2};
    piqp_int A_i[2] = {0, 0};
    piqp_float b[1] = {0};

    piqp_float G_x[3] = {1, 1, 1};
    piqp_int G_nnz = 3;
    piqp_int G_p[3] = {0, 3, 3};
    piqp_int G_i[3] = {0, 1, 2};
    piqp_float h_l[3] = {-1, -PIQP_INF, -2};
    piqp_float h_u[3] = {PIQP_INF, 1, 2};

    piqp_float x_l[2] = {-PIQP_INF, -1};
    piqp_float x_u[2] = {PIQP_INF, 1};

    piqp_workspace* work;
    piqp_settings* settings = (piqp_settings*) malloc(sizeof(piqp_settings));
    piqp_data_sparse* data = (piqp_data_sparse*) malloc(sizeof(piqp_data_sparse));

    piqp_set_default_settings_sparse(settings);
    settings->verbose = 1;

    data->n = n;
    data->p = p;
    data->m = m;
    data->P = piqp_csc_matrix(data->n, data->n, P_nnz, P_p, P_i, P_x);
    data->c = c;
    data->A = piqp_csc_matrix(data->p, data->n, A_nnz, A_p, A_i, A_x);
    data->b = b;
    data->G = piqp_csc_matrix(data->m, data->n, G_nnz, G_p, G_i, G_x);
    data->h_l = h_l;
    data->h_u = h_u;
    data->x_l = x_l;
    data->x_u = x_u;

    piqp_setup_sparse(&work, data, settings);
    piqp_status status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->result->x[0], 0.4285714, 1e-6);
    ASSERT_NEAR(work->result->x[1], 0.2142857, 1e-6);
    ASSERT_NEAR(work->result->y[0], -1.5714286, 1e-6);
    ASSERT_NEAR(work->result->z_l[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[1], 0, 1e-6);

    P_x[0] = 8;
    A_x[1] = -3;
    h_u[0] = 2;
    x_u[1] = 2;

    piqp_update_sparse(work, data->P, NULL, data->A, NULL, NULL, NULL, data->h_u, NULL, data->x_u);
    status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->result->x[0], 0.2763157, 1e-6);
    ASSERT_NEAR(work->result->x[1], 0.0921056, 1e-6);
    ASSERT_NEAR(work->result->y[0], -1.2105263, 1e-6);
    ASSERT_NEAR(work->result->z_l[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_l[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_u[2], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bl[1], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[0], 0, 1e-6);
    ASSERT_NEAR(work->result->z_bu[1], 0, 1e-6);

    piqp_cleanup(work);
    if (settings) free(settings);
    if (data)
    {
        if (data->P) free(data->P);
        if (data->A) free(data->A);
        if (data->G) free(data->G);
        free(data);
    }
}

/*
 * QP for the backward tests:
 * min 3 x1^2 + 2 x2^2 - x1 - 4 x2
 * s.t. x1 = 2 x2, x1 >= -1, x2 <= 0.1 (active)
 *
 * loss L = x1 + 2 x2 + y
*/
struct BackwardQP
{
    piqp_float P[4] = {6, 0, 0, 4};
    piqp_float c[2] = {-1, -4};
    piqp_float A[2] = {1, -2};
    piqp_float b[1] = {0};
    piqp_float G[2] = {1, 0};
    piqp_float h_l[1] = {-1};
    piqp_float h_u[1] = {PIQP_INF};
    piqp_float x_l[2] = {-PIQP_INF, -PIQP_INF};
    piqp_float x_u[2] = {PIQP_INF, 0.1};
};

static const piqp_float backward_grad_x[2] = {1, 2};
static const piqp_float backward_grad_y[1] = {1};

static piqp_workspace* setup_backward_dense(BackwardQP& qp)
{
    piqp_settings settings;
    piqp_set_default_settings_dense(&settings);
    settings.eps_abs = 1e-12;
    settings.eps_rel = 1e-12;
    settings.enable_backward = 1;

    piqp_data_dense data;
    data.n = 2;
    data.p = 1;
    data.m = 1;
    data.P = qp.P;
    data.c = qp.c;
    data.A = qp.A;
    data.b = qp.b;
    data.G = qp.G;
    data.h_l = qp.h_l;
    data.h_u = qp.h_u;
    data.x_l = qp.x_l;
    data.x_u = qp.x_u;

    piqp_workspace* work;
    piqp_setup_dense(&work, &data, &settings);
    return work;
}

static piqp_float backward_loss(BackwardQP qp)
{
    piqp_workspace* work = setup_backward_dense(qp);
    EXPECT_EQ(piqp_solve(work), PIQP_SOLVED);
    piqp_float loss = work->result->x[0] + 2 * work->result->x[1] + work->result->y[0];
    piqp_cleanup(work);
    return loss;
}

template<typename Field>
static piqp_float backward_finite_difference(Field field)
{
    const piqp_float eps = 1e-6;
    BackwardQP qp_p, qp_m;
    *field(qp_p) += eps;
    *field(qp_m) -= eps;
    return (backward_loss(qp_p) - backward_loss(qp_m)) / (2 * eps);
}

TEST(CInterfaceTest, DenseBackward)
{
    BackwardQP qp;
    piqp_workspace* work = setup_backward_dense(qp);
    ASSERT_NE(work->gradients, nullptr);
    EXPECT_EQ(work->gradients->P_csc, nullptr);

    EXPECT_EQ(piqp_backward(work, backward_grad_x, backward_grad_y, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
              PIQP_BWD_INVALID_FORWARD_RESULT);
    EXPECT_EQ(piqp_gradients_valid(work), 0);

    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);
    ASSERT_EQ(piqp_backward(work, backward_grad_x, backward_grad_y, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
              PIQP_BWD_SOLVED);
    EXPECT_EQ(piqp_gradients_valid(work), 1);

    const piqp_gradients* g = work->gradients;
    const double tol = 1e-5;
    EXPECT_NEAR(g->c[0], backward_finite_difference([](BackwardQP& q) { return &q.c[0]; }), tol);
    EXPECT_NEAR(g->c[1], backward_finite_difference([](BackwardQP& q) { return &q.c[1]; }), tol);
    EXPECT_NEAR(g->b[0], backward_finite_difference([](BackwardQP& q) { return &q.b[0]; }), tol);
    EXPECT_NEAR(g->x_u[1], backward_finite_difference([](BackwardQP& q) { return &q.x_u[1]; }), tol);
    EXPECT_NEAR(g->h_l[0], 0, tol); // inactive
    EXPECT_EQ(g->h_u[0], 0);        // infinite bound
    EXPECT_EQ(g->x_u[0], 0);        // infinite bound
    // A and G are row-major, A has a single row
    EXPECT_NEAR(g->A[0], backward_finite_difference([](BackwardQP& q) { return &q.A[0]; }), tol);
    EXPECT_NEAR(g->A[1], backward_finite_difference([](BackwardQP& q) { return &q.A[1]; }), tol);
    // diagonal of P (only the upper triangular part of P is used)
    EXPECT_NEAR(g->P[0], backward_finite_difference([](BackwardQP& q) { return &q.P[0]; }), tol);
    EXPECT_NEAR(g->P[3], backward_finite_difference([](BackwardQP& q) { return &q.P[3]; }), tol);

    // the gradients at a provided forward result match the ones at the stored solution; the default call reuses
    // the factorization of the forward solve (second-to-last iterate), the forward result is refactorized
    piqp_float c0 = g->c[0];
    ASSERT_EQ(piqp_backward(work, backward_grad_x, backward_grad_y, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, work->result),
              PIQP_BWD_SOLVED);
    EXPECT_NEAR(g->c[0], c0, 1e-6);

    // a failed call invalidates the gradients
    piqp_result invalid = *work->result;
    invalid.info.rho = 0;
    EXPECT_EQ(piqp_backward(work, backward_grad_x, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &invalid),
              PIQP_BWD_INVALID_FORWARD_RESULT);
    EXPECT_EQ(piqp_gradients_valid(work), 0);

    piqp_update_dense(work, NULL, qp.c, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    EXPECT_EQ(piqp_backward(work, backward_grad_x, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
              PIQP_BWD_INVALID_FORWARD_RESULT);

    piqp_cleanup(work);
}

TEST(CInterfaceTest, SparseBackwardMatchesDense)
{
    BackwardQP qp;
    piqp_workspace* work_dense = setup_backward_dense(qp);
    ASSERT_EQ(piqp_solve(work_dense), PIQP_SOLVED);
    ASSERT_EQ(piqp_backward(work_dense, backward_grad_x, backward_grad_y, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
              PIQP_BWD_SOLVED);

    piqp_float P_x[2] = {6, 4};
    piqp_int P_p[3] = {0, 1, 2};
    piqp_int P_i[2] = {0, 1};
    piqp_float A_x[2] = {1, -2};
    piqp_int A_p[3] = {0, 1, 2};
    piqp_int A_i[2] = {0, 0};
    piqp_float G_x[1] = {1};
    piqp_int G_p[3] = {0, 1, 1};
    piqp_int G_i[1] = {0};

    piqp_settings settings;
    piqp_set_default_settings_sparse(&settings);
    settings.eps_abs = 1e-12;
    settings.eps_rel = 1e-12;
    settings.enable_backward = 1;

    piqp_data_sparse data;
    data.n = 2;
    data.p = 1;
    data.m = 1;
    data.P = piqp_csc_matrix(2, 2, 2, P_p, P_i, P_x);
    data.c = qp.c;
    data.A = piqp_csc_matrix(1, 2, 2, A_p, A_i, A_x);
    data.b = qp.b;
    data.G = piqp_csc_matrix(1, 2, 1, G_p, G_i, G_x);
    data.h_l = qp.h_l;
    data.h_u = qp.h_u;
    data.x_l = qp.x_l;
    data.x_u = qp.x_u;

    piqp_workspace* work;
    piqp_setup_sparse(&work, &data, &settings);
    ASSERT_NE(work->gradients, nullptr);
    EXPECT_EQ(work->gradients->P, nullptr);
    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);
    ASSERT_EQ(piqp_backward(work, backward_grad_x, backward_grad_y, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
              PIQP_BWD_SOLVED);

    const piqp_gradients* gs = work->gradients;
    const piqp_gradients* gd = work_dense->gradients;
    // both solvers stop at slightly different final iterates
    const double tol = 1e-5;
    for (int i = 0; i < 2; i++)
    {
        EXPECT_NEAR(gs->c[i], gd->c[i], tol);
        EXPECT_NEAR(gs->x_u[i], gd->x_u[i], tol);
    }
    EXPECT_NEAR(gs->b[0], gd->b[0], tol);
    // sparse matrices are views on the sparsity pattern of the data, dense ones are row-major
    for (piqp_int j = 0; j < 2; j++)
    {
        for (piqp_int k = gs->P_csc->p[j]; k < gs->P_csc->p[j + 1]; k++)
        {
            EXPECT_NEAR(gs->P_csc->x[k], gd->P[gs->P_csc->i[k] * 2 + j], tol);
        }
        for (piqp_int k = gs->A_csc->p[j]; k < gs->A_csc->p[j + 1]; k++)
        {
            EXPECT_NEAR(gs->A_csc->x[k], gd->A[gs->A_csc->i[k] * 2 + j], tol);
        }
        for (piqp_int k = gs->G_csc->p[j]; k < gs->G_csc->p[j + 1]; k++)
        {
            EXPECT_NEAR(gs->G_csc->x[k], gd->G[gs->G_csc->i[k] * 2 + j], tol);
        }
    }

    piqp_cleanup(work);
    piqp_cleanup(work_dense);
    free(data.P);
    free(data.A);
    free(data.G);
}

TEST(CInterfaceTest, BackwardNotEnabled)
{
    BackwardQP qp;
    piqp_settings settings;
    piqp_set_default_settings_dense(&settings);
    EXPECT_EQ(settings.enable_backward, 0);

    piqp_data_dense data;
    data.n = 2;
    data.p = 1;
    data.m = 1;
    data.P = qp.P;
    data.c = qp.c;
    data.A = qp.A;
    data.b = qp.b;
    data.G = qp.G;
    data.h_l = qp.h_l;
    data.h_u = qp.h_u;
    data.x_l = qp.x_l;
    data.x_u = qp.x_u;

    piqp_workspace* work;
    piqp_setup_dense(&work, &data, &settings);
    EXPECT_EQ(work->gradients, nullptr);
    ASSERT_EQ(piqp_solve(work), PIQP_SOLVED);
    EXPECT_EQ(piqp_backward(work, backward_grad_x, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
              PIQP_BWD_INVALID_SETTINGS);
    piqp_cleanup(work);
}
