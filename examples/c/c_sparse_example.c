// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include "stdlib.h"
#include "stdio.h"
#include "piqp.h"

int main()
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
    piqp_float b[1] = {1};

    piqp_float G_x[3] = {1, 2, -1};
    piqp_int G_nnz = 3;
    piqp_int G_p[3] = {0, 2, 3};
    piqp_int G_i[4] = {0, 1, 0};
    piqp_float h_u[2] = {0.2, -1};

    piqp_float x_l[2] = {-1, -PIQP_INF};
    piqp_float x_u[2] = {1, PIQP_INF};

    piqp_workspace* work;
    piqp_settings* settings = (piqp_settings*) malloc(sizeof(piqp_settings));
    piqp_data_sparse* data = (piqp_data_sparse*) malloc(sizeof(piqp_data_sparse));

    piqp_set_default_settings_sparse(settings);
    settings->verbose = 1;
    settings->compute_timings = 1;

    data->n = n;
    data->p = p;
    data->m = m;
    data->P = piqp_csc_matrix(data->n, data->n, P_nnz, P_p, P_i, P_x);
    data->c = c;
    data->A = piqp_csc_matrix(data->p, data->n, A_nnz, A_p, A_i, A_x);
    data->b = b;
    data->G = piqp_csc_matrix(data->m, data->n, G_nnz, G_p, G_i, G_x);
    data->h_l = NULL;
    data->h_u = h_u;
    data->x_l = x_l;
    data->x_u = x_u;

    piqp_setup_sparse(&work, data, settings);
    piqp_status status = piqp_solve(work);

    printf("status = %d\n", status);
    printf("x = %f %f\n", work->result->x[0], work->result->x[1]);

    piqp_cleanup(work);
    free(settings);
    free(data);

    return 0;
}
