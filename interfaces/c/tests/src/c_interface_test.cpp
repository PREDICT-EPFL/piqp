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
    piqp_int m = 2;

    piqp_float P[4] = {6, 0, 0, 4};
    piqp_float c[2] = {-1, -4};

    piqp_float A[2] = {1, -2};
    piqp_float b[1] = {0};

    piqp_float G[4] = {1, 0, -1, 0};
    piqp_float h[2] = {1, 1};

    piqp_float x_lb[2] = {-PIQP_INF, -1};
    piqp_float x_ub[2] = {PIQP_INF, 1};

    piqp_workspace* work;
    piqp_settings* settings = (piqp_settings*) malloc(sizeof(piqp_settings));
    piqp_data_dense* data = (piqp_data_dense*) malloc(sizeof(piqp_data_dense));

    piqp_set_default_settings(settings);
    settings->verbose = 1;

    data->n = n;
    data->p = p;
    data->m = m;
    data->P = P;
    data->c = c;
    data->A = A;
    data->b = b;
    data->G = G;
    data->h = h;
    data->x_lb = x_lb;
    data->x_ub = x_ub;

    piqp_setup_dense(&work, data, settings);
    piqp_status status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->results->x[0], 0.4285714, 1e-6);
    ASSERT_NEAR(work->results->x[1], 0.2142857, 1e-6);
    ASSERT_NEAR(work->results->y[0], -1.5714286, 1e-6);
    ASSERT_NEAR(work->results->z[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z[1], 0, 1e-6);
    ASSERT_NEAR(work->results->z_lb[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z_lb[1], 0, 1e-6);
    ASSERT_NEAR(work->results->z_ub[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z_ub[1], 0, 1e-6);

    P[0] = 8;
    A[1] = -3;
    h[0] = 2;
    x_ub[1] = 2;

    piqp_update_dense(work, data->P, NULL, data->A, NULL, NULL, data->h, NULL, data->x_ub);
    status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->results->x[0], 0.2763157, 1e-6);
    ASSERT_NEAR(work->results->x[1], 0.0921056, 1e-6);
    ASSERT_NEAR(work->results->y[0], -1.2105263, 1e-6);
    ASSERT_NEAR(work->results->z[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z[1], 0, 1e-6);
    ASSERT_NEAR(work->results->z_lb[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z_lb[1], 0, 1e-6);
    ASSERT_NEAR(work->results->z_ub[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z_ub[1], 0, 1e-6);
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
    piqp_int m = 2;

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

    piqp_float G_x[2] = {1, -1};
    piqp_int G_nnz = 2;
    piqp_int G_p[3] = {0, 2, 2};
    piqp_int G_i[2] = {0, 1};
    piqp_float h[2] = {1, 1};

    piqp_float x_lb[2] = {-PIQP_INF, -1};
    piqp_float x_ub[2] = {PIQP_INF, 1};

    piqp_workspace* work;
    piqp_settings* settings = (piqp_settings*) malloc(sizeof(piqp_settings));
    piqp_data_sparse* data = (piqp_data_sparse*) malloc(sizeof(piqp_data_sparse));

    piqp_set_default_settings(settings);
    settings->verbose = 1;

    data->n = n;
    data->p = p;
    data->m = m;
    data->P = (piqp_csc*) malloc(sizeof(piqp_csc));
    data->P->m = data->n;
    data->P->n = data->n;
    data->P->nnz = P_nnz;
    data->P->p = P_p;
    data->P->i = P_i;
    data->P->x = P_x;
    data->c = c;
    data->A = (piqp_csc*) malloc(sizeof(piqp_csc));
    data->A->m = data->p;
    data->A->n = data->n;
    data->A->nnz = A_nnz;
    data->A->p = A_p;
    data->A->i = A_i;
    data->A->x = A_x;
    data->b = b;
    data->G = (piqp_csc*) malloc(sizeof(piqp_csc));
    data->G->m = data->m;
    data->G->n = data->n;
    data->G->nnz = G_nnz;
    data->G->p = G_p;
    data->G->i = G_i;
    data->G->x = G_x;
    data->h = h;
    data->x_lb = x_lb;
    data->x_ub = x_ub;

    piqp_setup_sparse(&work, data, settings);
    piqp_status status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->results->x[0], 0.4285714, 1e-6);
    ASSERT_NEAR(work->results->x[1], 0.2142857, 1e-6);
    ASSERT_NEAR(work->results->y[0], -1.5714286, 1e-6);
    ASSERT_NEAR(work->results->z[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z[1], 0, 1e-6);
    ASSERT_NEAR(work->results->z_lb[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z_lb[1], 0, 1e-6);
    ASSERT_NEAR(work->results->z_ub[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z_ub[1], 0, 1e-6);

    P_x[0] = 8;
    A_x[1] = -3;
    h[0] = 2;
    x_ub[1] = 2;

    piqp_update_sparse(work, data->P, NULL, data->A, NULL, NULL, data->h, NULL, data->x_ub);
    status = piqp_solve(work);

    ASSERT_EQ(status, PIQP_SOLVED);
    ASSERT_NEAR(work->results->x[0], 0.2763157, 1e-6);
    ASSERT_NEAR(work->results->x[1], 0.0921056, 1e-6);
    ASSERT_NEAR(work->results->y[0], -1.2105263, 1e-6);
    ASSERT_NEAR(work->results->z[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z[1], 0, 1e-6);
    ASSERT_NEAR(work->results->z_lb[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z_lb[1], 0, 1e-6);
    ASSERT_NEAR(work->results->z_ub[0], 0, 1e-6);
    ASSERT_NEAR(work->results->z_ub[1], 0, 1e-6);
}
