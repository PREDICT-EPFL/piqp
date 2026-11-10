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
    piqp_float* x;  // numerical values (size nzz)
} piqp_csc;

typedef struct {
    piqp_int   n;    // number of decision variables
    piqp_int   p;    // number of equality constraints
    piqp_int   m;    // number of inequality constraints
    piqp_float* P;   // quadratic cost matrix P (size n x n), only upper triangular part is used
    piqp_float* c;   // linear cost weights c (size n)
    piqp_float* A;   // equality constraints matrix A (size p x n), can be NULL
    piqp_float* b;   // equality constraints constant b (size p), can be NULL
    piqp_float* G;   // inequality constraints matrix G (size m x n), can be NULL
    piqp_float* h_l; // inequality lower bounds h_l (size m), can be NULL
    piqp_float* h_u; // inequality upper bounds h_u (size m), can be NULL
    piqp_float* x_l; // decision variables lower bounds x_l (size n), can be NULL
    piqp_float* x_u; // decision variables upper bounds x_u (size n), can be NULL
} piqp_data_dense;

typedef struct {
    piqp_int   n;    // number of decision variables
    piqp_int   p;    // number of equality constraints
    piqp_int   m;    // number of inequality constraints
    piqp_csc*  P;    // quadratic cost matrix P (size n x n), only upper triangular part is used
    piqp_float* c;   // linear cost weights c (size n)
    piqp_csc*  A;    // equality constraints matrix A (size p x n), can be NULL
    piqp_float* b;   // equality constraints constant b (size p), can be NULL
    piqp_csc*  G;    // inequality constraints matrix G (size m x n), can be NULL
    piqp_float* h_l; // inequality lower bounds h_l (size m), can be NULL
    piqp_float* h_u; // inequality upper bounds h_u (size m), can be NULL
    piqp_float* x_l; // decision variables lower bounds x_lb (size n), can be NULL
    piqp_float* x_u; // decision variables upper bounds x_ub (size n), can be NULL
} piqp_data_sparse;

typedef enum {
    PIQP_DENSE_CHOLESKY,
    PIQP_SPARSE_LDLT,
    PIQP_SPARSE_LDLT_EQ_COND,
    PIQP_SPARSE_LDLT_INEQ_COND,
    PIQP_SPARSE_LDLT_COND,
    PIQP_SPARSE_MULTISTAGE,
    PIQP_SPARSE_MULTISTAGE_PARALLEL
} piqp_kkt_solver;

typedef struct {
    piqp_float       rho_init;
    piqp_float       delta_init;
    piqp_float       eps_abs;
    piqp_float       eps_rel;
    piqp_int        check_duality_gap;
    piqp_float       eps_duality_gap_abs;
    piqp_float       eps_duality_gap_rel;
    piqp_float       infeasibility_threshold;
    piqp_float       eps_primal_inf;
    piqp_float       eps_dual_inf;
    piqp_float       reg_lower_limit;
    piqp_float       reg_finetune_lower_limit;
    piqp_int        reg_finetune_primal_update_threshold;
    piqp_int        reg_finetune_dual_update_threshold;
    piqp_int        max_iter;
    piqp_int        max_factor_retires;
    piqp_int        preconditioner_scale_cost;
    piqp_int        preconditioner_reuse_on_update;
    piqp_int        preconditioner_iter;
    piqp_float       tau;
    piqp_kkt_solver kkt_solver;
    piqp_int        num_threads;
    piqp_int        iterative_refinement_always_enabled;
    piqp_float       iterative_refinement_eps_abs;
    piqp_float       iterative_refinement_eps_rel;
    piqp_int        iterative_refinement_max_iter;
    piqp_float       iterative_refinement_min_improvement_rate;
    piqp_float       iterative_refinement_static_regularization_eps;
    piqp_float       iterative_refinement_static_regularization_rel;
    piqp_int        enable_backward;
    piqp_int        gradient_smoothing;
    piqp_float       gradient_smoothing_mu;
    piqp_float       gradient_smoothing_tol;
    piqp_int        gradient_smoothing_max_iter;
    piqp_int        verbose;
    piqp_int        compute_timings;
} piqp_settings;

typedef enum {
    PIQP_SOLVED            = 1,
    PIQP_MAX_ITER_REACHED  = -1,
    PIQP_PRIMAL_INFEASIBLE = -2,
    PIQP_DUAL_INFEASIBLE   = -3,
    PIQP_NUMERICS          = -8,
    PIQP_UNSOLVED          = -9,
    PIQP_INVALID_SETTINGS  = -10
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

    piqp_float primal_res;
    piqp_float primal_res_rel;
    piqp_float dual_res;
    piqp_float dual_res_rel;

    piqp_float primal_res_reg;
    piqp_float primal_res_reg_rel;
    piqp_float dual_res_reg;
    piqp_float dual_res_reg_rel;

    piqp_float primal_prox_inf;
    piqp_float dual_prox_inf;

    piqp_float prev_primal_res;
    piqp_float prev_dual_res;

    piqp_float primal_obj;
    piqp_float dual_obj;
    piqp_float duality_gap;
    piqp_float duality_gap_rel;

    piqp_int factor_retires;
    piqp_float reg_limit;
    piqp_int no_primal_update;
    piqp_int no_dual_update;

    piqp_float setup_time;
    piqp_float update_time;
    piqp_float solve_time;
    piqp_float kkt_factor_time;
    piqp_float kkt_solve_time;
    piqp_float run_time;
} piqp_info;

typedef struct {
    const piqp_float* x;
    const piqp_float* y;
    const piqp_float* z_l;
    const piqp_float* z_u;
    const piqp_float* z_bl;
    const piqp_float* z_bu;
    const piqp_float* s_l;
    const piqp_float* s_u;
    const piqp_float* s_bl;
    const piqp_float* s_bu;

    piqp_info info;
} piqp_result;

typedef struct {
    // the gradients are only valid if status is PIQP_SOLVED; status of the last piqp_backward(),
    // PIQP_UNSOLVED if piqp_backward() was not called yet or the problem changed since (setup, update)
    piqp_status status;

    const piqp_float* P;   // dense interface: gradient w.r.t. P (size n x n, row-major, symmetric), NULL for sparse
    const piqp_float* A;   // dense interface: gradient w.r.t. A (size p x n, row-major), NULL for sparse
    const piqp_float* G;   // dense interface: gradient w.r.t. G (size m x n, row-major), NULL for sparse
    const piqp_csc* P_csc; // sparse interface: gradient w.r.t. P (size n x n, sparsity pattern of the P passed to setup), NULL for dense
    const piqp_csc* A_csc; // sparse interface: gradient w.r.t. A (size p x n, sparsity pattern of the A passed to setup), NULL for dense
    const piqp_csc* G_csc; // sparse interface: gradient w.r.t. G (size m x n, sparsity pattern of the G passed to setup), NULL for dense
    const piqp_float* c;   // gradient w.r.t. c (size n)
    const piqp_float* b;   // gradient w.r.t. b (size p)
    const piqp_float* h_l; // gradient w.r.t. h_l (size m), zero for infinite bounds
    const piqp_float* h_u; // gradient w.r.t. h_u (size m), zero for infinite bounds
    const piqp_float* x_l; // gradient w.r.t. x_l (size n), zero for infinite bounds
    const piqp_float* x_u; // gradient w.r.t. x_u (size n), zero for infinite bounds
} piqp_gradients;

struct piqp_solver_handle; // An opaque type that we'll use as a handle for the C++ solver object
typedef struct piqp_solver_handle piqp_solver_handle;

typedef struct {
    piqp_int is_dense; // dense interface is being used
    piqp_int n;        // number of decision variables
    piqp_int p;        // number of equality constraints
    piqp_int m;        // number of inequality constraints
} piqp_solver_info;

typedef struct {
    piqp_solver_handle* solver_handle;
    piqp_solver_info    solver_info;
    piqp_result*        result;
    piqp_gradients*     gradients; // NULL if settings->enable_backward was not set in the setup
} piqp_workspace;

# ifdef __cplusplus
}
# endif

#endif //PIQP_TYPEDEF_H
