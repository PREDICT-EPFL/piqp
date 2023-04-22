// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_TYPEDEF_H
#define PIQP_TYPEDEF_H

# ifdef __cplusplus
extern "C" {
# endif

#ifdef PIQP_SINGLE_PRECISION
typedef float piqp_float;
#else
typedef double piqp_float;
#endif

#ifdef PIQP_LONG_STORAGE_INDEX
typedef long long piqp_int;
#else
typedef int piqp_int;
#endif

typedef struct {
    piqp_int   m;   // rows
    piqp_int   n;   // cols
    piqp_int   nnz; // non-zero elements
    piqp_int*  p;   // column indices (size n+1), i.e. stores for each column the index of the first non-zero
    piqp_int*  i;   // row indices (size nzz)
    piqp_float* x;   // numerical values (size nzz)
} piqp_csc;

typedef struct {
    piqp_int   n;    // number of decision variables
    piqp_int   p;    // number of equality constraints
    piqp_int   m;    // number of inequality constraints
    piqp_float* P;    // quadratic cost matrix P (size n x n), only upper triangular part is used
    piqp_float* c;    // linear cost weights c (size n)
    piqp_float* A;    // equality constraints matrix A (size p x n)
    piqp_float* b;    // equality constraints constant b (size p)
    piqp_float* G;    // inequality constraints matrix G (size m x n)
    piqp_float* h;    // inequality upper bounds h (size n)
    piqp_float* x_lb; // decision variables lower bounds x_lb (size n), can be NULL
    piqp_float* x_ub; // decision variables upper bounds x_ub (size n), can be NULL
} piqp_data_dense;

typedef struct {
    piqp_int   n;    // number of decision variables
    piqp_int   p;    // number of equality constraints
    piqp_int   m;    // number of inequality constraints
    piqp_csc*  P;    // quadratic cost matrix P (size n x n), only upper triangular part is used
    piqp_float* c;    // linear cost weights c (size n)
    piqp_csc*  A;    // equality constraints matrix A (size p x n)
    piqp_float* b;    // equality constraints constant b (size p)
    piqp_csc*  G;    // inequality constraints matrix G (size m x n)
    piqp_float* h;    // inequality upper bounds h (size n)
    piqp_float* x_lb; // decision variables lower bounds x_lb (size n), can be NULL
    piqp_float* x_ub; // decision variables upper bounds x_ub (size n), can be NULL
} piqp_data_sparse;

typedef struct {
    piqp_float rho_init;
    piqp_float delta_init;
    piqp_float eps_abs;
    piqp_float eps_rel;
    piqp_int  check_duality_gap;
    piqp_float eps_duality_gap_abs;
    piqp_float eps_duality_gap_rel;
    piqp_float reg_lower_limit;
    piqp_float reg_finetune_lower_limit;
    piqp_int  max_iter;
    piqp_int  max_factor_retires;
    piqp_int  preconditioner_scale_cost;
    piqp_int  preconditioner_iter;
    piqp_float tau;
    piqp_int  verbose;
    piqp_int  compute_timings;
} piqp_settings;

typedef enum {
    PIQP_SOLVED = 1,
    PIQP_MAX_ITER_REACHED = -1,
    PIQP_PRIMAL_INFEASIBLE = -2,
    PIQP_DUAL_INFEASIBLE = -3,
    PIQP_NUMERICS = -8,
    PIQP_UNSOLVED = -9,
    PIQP_INVALID_SETTINGS = -10
} piqp_status;

typedef struct {
    piqp_status status;

    piqp_int iter;
    piqp_float rho;
    piqp_float delta;
    piqp_float mu;
    piqp_float sigma;
    piqp_float primal_step;
    piqp_float dual_step;

    piqp_float primal_inf;
    piqp_float primal_rel_inf;
    piqp_float dual_inf;
    piqp_float dual_rel_inf;

    piqp_float primal_obj;
    piqp_float dual_obj;
    piqp_float duality_gap;
    piqp_float duality_gap_rel;

    piqp_float factor_retires;
    piqp_float reg_limit;
    piqp_float no_primal_update;
    piqp_float no_dual_update;

    piqp_float setup_time;
    piqp_float update_time;
    piqp_float solve_time;
    piqp_float run_time;
} piqp_info;

typedef struct {
    const piqp_float* x;
    const piqp_float* y;
    const piqp_float* z;
    const piqp_float* z_lb;
    const piqp_float* z_ub;
    const piqp_float* s;
    const piqp_float* s_lb;
    const piqp_float* s_ub;

    const piqp_float* zeta;
    const piqp_float* lambda;
    const piqp_float* nu;
    const piqp_float* nu_lb;
    const piqp_float* nu_ub;

    piqp_info info;
} piqp_result;

struct piqp_solver_handle; // An opaque type that we'll use as a handle for the C++ solver object
typedef struct piqp_solver_handle piqp_solver_handle;

typedef struct {
    piqp_int is_dense; // dense interface is being used
    piqp_int n;        // number of decision variables
    piqp_int p;        // number of equality constraints
    piqp_int m;        // number of inequality constraints
} pipq_solver_info;

typedef struct {
    piqp_solver_handle* solver_handle;
    pipq_solver_info    solver_info;
    piqp_result*       result;
} piqp_workspace;

# ifdef __cplusplus
}
# endif

#endif //PIQP_TYPEDEF_H
