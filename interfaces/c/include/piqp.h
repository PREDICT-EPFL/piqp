// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_H
#define PIQP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "piqp_typedef.h"

#ifndef PIQP_INF
#define PIQP_INF 1e30
#endif

piqp_csc* piqp_csc_matrix(piqp_int m, piqp_int n, piqp_int nnz, piqp_int *p, piqp_int *i, piqp_float *x);

void piqp_set_default_settings_dense(piqp_settings* settings);
void piqp_set_default_settings_sparse(piqp_settings* settings);

void piqp_setup_dense(piqp_workspace** workspace, const piqp_data_dense* data, const piqp_settings* settings);
void piqp_setup_sparse(piqp_workspace** workspace, const piqp_data_sparse* data, const piqp_settings* settings);

void piqp_update_settings(piqp_workspace* workspace, const piqp_settings* settings);
void piqp_update_dense(piqp_workspace* workspace,
                       piqp_float* P, piqp_float* c,
                       piqp_float* A, piqp_float* b,
                       piqp_float* G, piqp_float* h_l, piqp_float* h_u,
                       piqp_float* x_l, piqp_float* x_u);
void piqp_update_sparse(piqp_workspace* workspace,
                        piqp_csc* P, piqp_float* c,
                        piqp_csc* A, piqp_float* b,
                        piqp_csc* G, piqp_float* h_l, piqp_float* h_u,
                        piqp_float* x_l, piqp_float* x_u);

piqp_status piqp_solve(piqp_workspace* workspace);

void piqp_cleanup(piqp_workspace* workspace);

#ifdef __cplusplus
}
#endif

#endif //PIQP_H
