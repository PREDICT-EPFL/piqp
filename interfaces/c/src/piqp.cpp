// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include "piqp.h"

#include "piqp/piqp.hpp"

using CVec = Eigen::Matrix<piqp_float, Eigen::Dynamic, 1>;
using CMat = Eigen::Matrix<piqp_float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using CSparseMat = Eigen::SparseMatrix<piqp_float, Eigen::ColMajor, piqp_int>;

using DenseSolver = piqp::DenseSolver<piqp_float>;
using SparseSolver = piqp::SparseSolver<piqp_float, piqp_int>;

static piqp_kkt_solver cpp_to_c_kkt_solver(piqp::KKTSolver cpp_kkt_solver)
{
    switch (cpp_kkt_solver) {
        case piqp::KKTSolver::dense_cholesky:
            return PIQP_DENSE_CHOLESKY;
        case piqp::KKTSolver::sparse_ldlt:
            return PIQP_SPARSE_LDLT;
        case piqp::KKTSolver::sparse_ldlt_eq_cond:
            return PIQP_SPARSE_LDLT_EQ_COND;
        case piqp::KKTSolver::sparse_ldlt_ineq_cond:
            return PIQP_SPARSE_LDLT_INEQ_COND;
        case piqp::KKTSolver::sparse_ldlt_cond:
            return PIQP_SPARSE_LDLT_COND;
        case piqp::KKTSolver::sparse_multistage:
            return PIQP_SPARSE_MULTISTAGE;
        case piqp::KKTSolver::sparse_multistage_parallel:
            return PIQP_SPARSE_MULTISTAGE_PARALLEL;
    }
    return PIQP_DENSE_CHOLESKY;
}

static piqp::KKTSolver c_to_cpp_kkt_solver(piqp_kkt_solver c_kkt_solver)
{
    switch (c_kkt_solver) {
        case PIQP_DENSE_CHOLESKY:
            return piqp::KKTSolver::dense_cholesky;
        case PIQP_SPARSE_LDLT:
            return piqp::KKTSolver::sparse_ldlt;
        case PIQP_SPARSE_LDLT_EQ_COND:
            return piqp::KKTSolver::sparse_ldlt_eq_cond;
        case PIQP_SPARSE_LDLT_INEQ_COND:
            return piqp::KKTSolver::sparse_ldlt_ineq_cond;
        case PIQP_SPARSE_LDLT_COND:
            return piqp::KKTSolver::sparse_ldlt_cond;
        case PIQP_SPARSE_MULTISTAGE:
            return piqp::KKTSolver::sparse_multistage;
        case PIQP_SPARSE_MULTISTAGE_PARALLEL:
            return piqp::KKTSolver::sparse_multistage_parallel;
    }
    return piqp::KKTSolver::dense_cholesky;
}

piqp_csc* piqp_csc_matrix(piqp_int m, piqp_int n, piqp_int nnz, piqp_int *p, piqp_int *i, piqp_float *x)
{
    piqp_csc* matrix = (piqp_csc*) malloc(sizeof(piqp_csc));

    if (!matrix) return nullptr;

    matrix->m     = m;
    matrix->n     = n;
    matrix->nnz   = nnz;
    matrix->p     = p;
    matrix->i     = i;
    matrix->x     = x;

    return matrix;
}

static void piqp_update_result(piqp_result* result, const piqp::Result<piqp_float>& solver_result)
{
    result->x = solver_result.x.data();
    result->y = solver_result.y.data();
    result->z_l = solver_result.z_l.data();
    result->z_u = solver_result.z_u.data();
    result->z_bl = solver_result.z_bl.data();
    result->z_bu = solver_result.z_bu.data();
    result->s_l = solver_result.s_l.data();
    result->s_u = solver_result.s_u.data();
    result->s_bl = solver_result.s_bl.data();
    result->s_bu = solver_result.s_bu.data();

    result->info.status = (piqp_status) solver_result.info.status;
    result->info.iter = (piqp_int) solver_result.info.iter;
    result->info.rho = solver_result.info.rho;
    result->info.delta = solver_result.info.delta;
    result->info.mu = solver_result.info.mu;
    result->info.sigma = solver_result.info.sigma;
    result->info.primal_step = solver_result.info.primal_step;
    result->info.dual_step = solver_result.info.dual_step;
    result->info.primal_res = solver_result.info.primal_res;
    result->info.primal_res_rel = solver_result.info.primal_res_rel;
    result->info.dual_res = solver_result.info.dual_res;
    result->info.dual_res_rel = solver_result.info.dual_res_rel;
    result->info.primal_res_reg = solver_result.info.primal_res_reg;
    result->info.primal_res_reg_rel = solver_result.info.primal_res_reg_rel;
    result->info.dual_res_reg = solver_result.info.dual_res_reg;
    result->info.dual_res_reg_rel = solver_result.info.dual_res_reg_rel;
    result->info.primal_prox_inf = solver_result.info.primal_prox_inf;
    result->info.dual_prox_inf = solver_result.info.dual_prox_inf;
    result->info.prev_primal_res = solver_result.info.prev_primal_res;
    result->info.prev_dual_res = solver_result.info.prev_dual_res;
    result->info.primal_obj = solver_result.info.primal_obj;
    result->info.dual_obj = solver_result.info.dual_obj;
    result->info.duality_gap = solver_result.info.duality_gap;
    result->info.duality_gap_rel = solver_result.info.duality_gap_rel;
    result->info.factor_retires = (piqp_int) solver_result.info.factor_retires;
    result->info.reg_limit = solver_result.info.reg_limit;
    result->info.no_primal_update = (piqp_int) solver_result.info.no_primal_update;
    result->info.no_dual_update = (piqp_int) solver_result.info.no_dual_update;
    result->info.setup_time = solver_result.info.setup_time;
    result->info.update_time = solver_result.info.update_time;
    result->info.solve_time = solver_result.info.solve_time;
    result->info.kkt_factor_time = solver_result.info.kkt_factor_time;
    result->info.kkt_solve_time = solver_result.info.kkt_solve_time;
    result->info.run_time = solver_result.info.run_time;
}

template<typename Solver>
static void piqp_set_default_settings(piqp_settings* settings, Solver&& solver)
{
    settings->rho_init = solver.settings().rho_init;
    settings->delta_init = solver.settings().delta_init;
    settings->eps_abs = solver.settings().eps_abs;
    settings->eps_rel = solver.settings().eps_rel;
    settings->check_duality_gap = solver.settings().check_duality_gap;
    settings->eps_duality_gap_abs = solver.settings().eps_duality_gap_abs;
    settings->eps_duality_gap_rel = solver.settings().eps_duality_gap_rel;
    settings->infeasibility_threshold = solver.settings().infeasibility_threshold;
    settings->eps_primal_inf = solver.settings().eps_primal_inf;
    settings->eps_dual_inf = solver.settings().eps_dual_inf;
    settings->reg_lower_limit = solver.settings().reg_lower_limit;
    settings->reg_finetune_lower_limit = solver.settings().reg_finetune_lower_limit;
    settings->reg_finetune_primal_update_threshold = (piqp_int)  solver.settings().reg_finetune_primal_update_threshold;
    settings->reg_finetune_dual_update_threshold = (piqp_int)  solver.settings().reg_finetune_dual_update_threshold;
    settings->max_iter = (piqp_int) solver.settings().max_iter;
    settings->max_factor_retires = (piqp_int) solver.settings().max_factor_retires;
    settings->preconditioner_scale_cost = solver.settings().preconditioner_scale_cost;
    settings->preconditioner_reuse_on_update = solver.settings().preconditioner_reuse_on_update;
    settings->preconditioner_iter = (piqp_int) solver.settings().preconditioner_iter;
    settings->tau = solver.settings().tau;
    settings->kkt_solver = cpp_to_c_kkt_solver(solver.settings().kkt_solver);
    settings->num_threads = (piqp_int) solver.settings().num_threads;
    settings->iterative_refinement_always_enabled = (piqp_int) solver.settings().iterative_refinement_always_enabled;
    settings->iterative_refinement_eps_abs = solver.settings().iterative_refinement_eps_abs;
    settings->iterative_refinement_eps_rel = solver.settings().iterative_refinement_eps_rel;
    settings->iterative_refinement_max_iter = (piqp_int) solver.settings().iterative_refinement_max_iter;
    settings->iterative_refinement_min_improvement_rate = solver.settings().iterative_refinement_min_improvement_rate;
    settings->iterative_refinement_static_regularization_eps = solver.settings().iterative_refinement_static_regularization_eps;
    settings->iterative_refinement_static_regularization_rel = solver.settings().iterative_refinement_static_regularization_rel;
    settings->enable_backward = solver.settings().enable_backward;
    settings->gradient_smoothing = solver.settings().gradient_smoothing;
    settings->gradient_smoothing_mu = solver.settings().gradient_smoothing_mu;
    settings->gradient_smoothing_max_iter = (piqp_int) solver.settings().gradient_smoothing_max_iter;
    settings->verbose = solver.settings().verbose;
    settings->compute_timings = solver.settings().compute_timings;
}

void piqp_set_default_settings_dense(piqp_settings* settings)
{
    piqp_set_default_settings(settings, DenseSolver());
}

void piqp_set_default_settings_sparse(piqp_settings* settings)
{
    piqp_set_default_settings(settings, SparseSolver());
}

static piqp::optional<Eigen::Map<CVec>> piqp_optional_vec_map(piqp_float* data, piqp_int n)
{
    piqp::optional<Eigen::Map<CVec>> vec;
    if (data)
    {
        vec = Eigen::Map<CVec>(data, n);
    }
    return vec;
}

static piqp::optional<Eigen::Map<const CVec>> piqp_optional_const_vec_map(const piqp_float* data, piqp_int n)
{
    piqp::optional<Eigen::Map<const CVec>> vec;
    if (data)
    {
        vec.emplace(data, n);
    }
    return vec;
}

static piqp::optional<Eigen::Map<CMat>> piqp_optional_mat_map(piqp_float* data, piqp_int m, piqp_int n)
{
    piqp::optional<Eigen::Map<CMat>> mat;
    if (data)
    {
        mat = Eigen::Map<CMat>(data, m, n);
    }
    return mat;
}

static piqp::optional<Eigen::Map<CSparseMat>> piqp_optional_sparse_mat_map(piqp_csc* data)
{
    piqp::optional<Eigen::Map<CSparseMat>> mat;
    if (data)
    {
        mat = Eigen::Map<CSparseMat>(data->m, data->n, data->nnz, data->p, data->i, data->x);
    }
    return mat;
}

template<typename Gradients>
static void piqp_set_gradient_vectors(piqp_gradients* gradients, const Gradients& solver_gradients)
{
    gradients->status = (piqp_status) solver_gradients.status;
    gradients->c = solver_gradients.c.data();
    gradients->b = solver_gradients.b.data();
    gradients->h_l = solver_gradients.h_l.data();
    gradients->h_u = solver_gradients.h_u.data();
    gradients->x_l = solver_gradients.x_l.data();
    gradients->x_u = solver_gradients.x_u.data();
}

// the dense gradients are stored column-major in the solver, they are copied
// into row-major buffers owned by the workspace after each successful backward
static void piqp_allocate_gradients_dense(piqp_workspace* work, const DenseSolver& solver)
{
    work->gradients = new piqp_gradients;
    const piqp_int n = work->solver_info.n;
    const piqp_int p = work->solver_info.p;
    const piqp_int m = work->solver_info.m;
    work->gradients->P = new piqp_float[n * n];
    work->gradients->A = new piqp_float[p * n];
    work->gradients->G = new piqp_float[m * n];
    work->gradients->P_csc = nullptr;
    work->gradients->A_csc = nullptr;
    work->gradients->G_csc = nullptr;
    piqp_set_gradient_vectors(work->gradients, solver.gradients());
}

static void piqp_copy_gradients_dense(piqp_workspace* work, const DenseSolver& solver)
{
    const piqp_int n = work->solver_info.n;
    const piqp_int p = work->solver_info.p;
    const piqp_int m = work->solver_info.m;
    Eigen::Map<CMat>(const_cast<piqp_float*>(work->gradients->P), n, n) = solver.gradients().P;
    Eigen::Map<CMat>(const_cast<piqp_float*>(work->gradients->A), p, n) = solver.gradients().A;
    Eigen::Map<CMat>(const_cast<piqp_float*>(work->gradients->G), m, n) = solver.gradients().G;
}

static piqp_csc* piqp_csc_view(const CSparseMat& mat)
{
    return piqp_csc_matrix((piqp_int) mat.rows(), (piqp_int) mat.cols(), (piqp_int) mat.nonZeros(),
                           const_cast<piqp_int*>(mat.outerIndexPtr()),
                           const_cast<piqp_int*>(mat.innerIndexPtr()),
                           const_cast<piqp_float*>(mat.valuePtr()));
}

// the sparse gradients are views into the solver memory, whose sparsity pattern is fixed after the setup
static void piqp_allocate_gradients_sparse(piqp_workspace* work, const SparseSolver& solver)
{
    work->gradients = new piqp_gradients;
    work->gradients->P = nullptr;
    work->gradients->A = nullptr;
    work->gradients->G = nullptr;
    work->gradients->P_csc = piqp_csc_view(solver.gradients().P);
    work->gradients->A_csc = piqp_csc_view(solver.gradients().A);
    work->gradients->G_csc = piqp_csc_view(solver.gradients().G);
    piqp_set_gradient_vectors(work->gradients, solver.gradients());
}

static void piqp_free_gradients(piqp_workspace* work)
{
    if (!work->gradients) return;
    delete[] work->gradients->P;
    delete[] work->gradients->A;
    delete[] work->gradients->G;
    free(const_cast<piqp_csc*>(work->gradients->P_csc));
    free(const_cast<piqp_csc*>(work->gradients->A_csc));
    free(const_cast<piqp_csc*>(work->gradients->G_csc));
    delete work->gradients;
    work->gradients = nullptr;
}

void piqp_setup_dense(piqp_workspace** workspace, const piqp_data_dense* data, const piqp_settings* settings)
{
    auto* work = new piqp_workspace;
    *workspace = work;

    auto* solver = new DenseSolver();
    work->solver_handle = reinterpret_cast<piqp_solver_handle*>(solver);
    work->solver_info.is_dense = 1;
    work->solver_info.n = data->n;
    work->solver_info.p = data->p;
    work->solver_info.m = data->m;
    work->result = new piqp_result;
    work->gradients = nullptr;

    if (settings)
    {
        piqp_update_settings(work, settings);
    }

    Eigen::Map<CMat> P(data->P, data->n, data->n);
    Eigen::Map<CVec> c(data->c, data->n);
    piqp::optional<Eigen::Map<CMat>> A = piqp_optional_mat_map(data->A, data->p, data->n);
    piqp::optional<Eigen::Map<CVec>> b = piqp_optional_vec_map(data->b, data->p);
    piqp::optional<Eigen::Map<CMat>> G = piqp_optional_mat_map(data->G, data->m, data->n);
    piqp::optional<Eigen::Map<CVec>> h_l = piqp_optional_vec_map(data->h_l, data->m);
    piqp::optional<Eigen::Map<CVec>> h_u = piqp_optional_vec_map(data->h_u, data->m);
    piqp::optional<Eigen::Map<CVec>> x_l = piqp_optional_vec_map(data->x_l, data->n);
    piqp::optional<Eigen::Map<CVec>> x_u = piqp_optional_vec_map(data->x_u, data->n);

    solver->setup(P, c, A, b, G, h_l, h_u, x_l, x_u);

    piqp_update_result(work->result, solver->result());

    if (solver->settings().enable_backward)
    {
        piqp_allocate_gradients_dense(work, *solver);
    }
}

void piqp_setup_sparse(piqp_workspace** workspace, const piqp_data_sparse* data, const piqp_settings* settings)
{
    auto* work = new piqp_workspace;
    *workspace = work;

    auto* solver = new piqp::SparseSolver<piqp_float, piqp_int>();
    work->solver_handle = reinterpret_cast<piqp_solver_handle*>(solver);
    work->solver_info.is_dense = 0;
    work->solver_info.n = data->n;
    work->solver_info.p = data->p;
    work->solver_info.m = data->m;
    work->result = new piqp_result;
    work->gradients = nullptr;

    if (settings)
    {
        piqp_update_settings(work, settings);
    }

    Eigen::Map<CSparseMat> P(data->P->m, data->P->n, data->P->nnz, data->P->p, data->P->i, data->P->x);
    Eigen::Map<CVec> c(data->c, data->n);
    piqp::optional<Eigen::Map<CSparseMat>> A = piqp_optional_sparse_mat_map(data->A);
    piqp::optional<Eigen::Map<CVec>> b = piqp_optional_vec_map(data->b, data->p);
    piqp::optional<Eigen::Map<CSparseMat>> G = piqp_optional_sparse_mat_map(data->G);
    piqp::optional<Eigen::Map<CVec>> h_l = piqp_optional_vec_map(data->h_l, data->m);
    piqp::optional<Eigen::Map<CVec>> h_u = piqp_optional_vec_map(data->h_u, data->m);
    piqp::optional<Eigen::Map<CVec>> x_l = piqp_optional_vec_map(data->x_l, data->n);
    piqp::optional<Eigen::Map<CVec>> x_u = piqp_optional_vec_map(data->x_u, data->n);

    solver->setup(P, c, A, b, G, h_l, h_u, x_l, x_u);

    piqp_update_result(work->result, solver->result());

    if (solver->settings().enable_backward)
    {
        piqp_allocate_gradients_sparse(work, *solver);
    }
}

void piqp_update_settings(piqp_workspace* workspace, const piqp_settings* settings)
{
    if (workspace->solver_info.is_dense)
    {
        auto* solver = reinterpret_cast<DenseSolver*>(workspace->solver_handle);

        solver->settings().rho_init = settings->rho_init;
        solver->settings().delta_init = settings->delta_init;
        solver->settings().eps_abs = settings->eps_abs;
        solver->settings().eps_rel = settings->eps_rel;
        solver->settings().check_duality_gap = settings->check_duality_gap;
        solver->settings().eps_duality_gap_abs = settings->eps_duality_gap_abs;
        solver->settings().eps_duality_gap_rel = settings->eps_duality_gap_rel;
        solver->settings().infeasibility_threshold = settings->infeasibility_threshold;
        solver->settings().eps_primal_inf = settings->eps_primal_inf;
        solver->settings().eps_dual_inf = settings->eps_dual_inf;
        solver->settings().reg_lower_limit = settings->reg_lower_limit;
        solver->settings().reg_finetune_lower_limit = settings->reg_finetune_lower_limit;
        solver->settings().reg_finetune_primal_update_threshold = settings->reg_finetune_primal_update_threshold;
        solver->settings().reg_finetune_dual_update_threshold = settings->reg_finetune_dual_update_threshold;
        solver->settings().max_iter = settings->max_iter;
        solver->settings().max_factor_retires = settings->max_factor_retires;
        solver->settings().preconditioner_scale_cost = settings->preconditioner_scale_cost;
        solver->settings().preconditioner_reuse_on_update = settings->preconditioner_reuse_on_update;
        solver->settings().preconditioner_iter = settings->preconditioner_iter;
        solver->settings().tau = settings->tau;
        solver->settings().kkt_solver = c_to_cpp_kkt_solver(settings->kkt_solver);
        solver->settings().num_threads = settings->num_threads;
        solver->settings().iterative_refinement_always_enabled = settings->iterative_refinement_always_enabled;
        solver->settings().iterative_refinement_eps_abs = settings->iterative_refinement_eps_abs;
        solver->settings().iterative_refinement_eps_rel = settings->iterative_refinement_eps_rel;
        solver->settings().iterative_refinement_max_iter = settings->iterative_refinement_max_iter;
        solver->settings().iterative_refinement_min_improvement_rate = settings->iterative_refinement_min_improvement_rate;
        solver->settings().iterative_refinement_static_regularization_eps = settings->iterative_refinement_static_regularization_eps;
        solver->settings().iterative_refinement_static_regularization_rel = settings->iterative_refinement_static_regularization_rel;
        solver->settings().enable_backward = settings->enable_backward;
        solver->settings().gradient_smoothing = settings->gradient_smoothing;
        solver->settings().gradient_smoothing_mu = settings->gradient_smoothing_mu;
        solver->settings().gradient_smoothing_max_iter = settings->gradient_smoothing_max_iter;
        solver->settings().verbose = settings->verbose;
        solver->settings().compute_timings = settings->compute_timings;
    }
    else
    {
        auto* solver = reinterpret_cast<SparseSolver*>(workspace->solver_handle);

        solver->settings().rho_init = settings->rho_init;
        solver->settings().delta_init = settings->delta_init;
        solver->settings().eps_abs = settings->eps_abs;
        solver->settings().eps_rel = settings->eps_rel;
        solver->settings().check_duality_gap = settings->check_duality_gap;
        solver->settings().eps_duality_gap_abs = settings->eps_duality_gap_abs;
        solver->settings().eps_duality_gap_rel = settings->eps_duality_gap_rel;
        solver->settings().infeasibility_threshold = settings->infeasibility_threshold;
        solver->settings().eps_primal_inf = settings->eps_primal_inf;
        solver->settings().eps_dual_inf = settings->eps_dual_inf;
        solver->settings().reg_lower_limit = settings->reg_lower_limit;
        solver->settings().reg_finetune_lower_limit = settings->reg_finetune_lower_limit;
        solver->settings().reg_finetune_primal_update_threshold = settings->reg_finetune_primal_update_threshold;
        solver->settings().reg_finetune_dual_update_threshold = settings->reg_finetune_dual_update_threshold;
        solver->settings().max_iter = settings->max_iter;
        solver->settings().max_factor_retires = settings->max_factor_retires;
        solver->settings().preconditioner_scale_cost = settings->preconditioner_scale_cost;
        solver->settings().preconditioner_reuse_on_update = settings->preconditioner_reuse_on_update;
        solver->settings().preconditioner_iter = settings->preconditioner_iter;
        solver->settings().tau = settings->tau;
        solver->settings().kkt_solver = c_to_cpp_kkt_solver(settings->kkt_solver);
        solver->settings().num_threads = settings->num_threads;
        solver->settings().iterative_refinement_always_enabled = settings->iterative_refinement_always_enabled;
        solver->settings().iterative_refinement_eps_abs = settings->iterative_refinement_eps_abs;
        solver->settings().iterative_refinement_eps_rel = settings->iterative_refinement_eps_rel;
        solver->settings().iterative_refinement_max_iter = settings->iterative_refinement_max_iter;
        solver->settings().iterative_refinement_min_improvement_rate = settings->iterative_refinement_min_improvement_rate;
        solver->settings().iterative_refinement_static_regularization_eps = settings->iterative_refinement_static_regularization_eps;
        solver->settings().iterative_refinement_static_regularization_rel = settings->iterative_refinement_static_regularization_rel;
        solver->settings().enable_backward = settings->enable_backward;
        solver->settings().gradient_smoothing = settings->gradient_smoothing;
        solver->settings().gradient_smoothing_mu = settings->gradient_smoothing_mu;
        solver->settings().gradient_smoothing_max_iter = settings->gradient_smoothing_max_iter;
        solver->settings().verbose = settings->verbose;
        solver->settings().compute_timings = settings->compute_timings;
    }
}

void piqp_update_dense(piqp_workspace* workspace,
                       piqp_float* P, piqp_float* c,
                       piqp_float* A, piqp_float* b,
                       piqp_float* G, piqp_float* h_l, piqp_float* h_u,
                       piqp_float* x_l, piqp_float* x_u)
{
    piqp::optional<Eigen::Map<CMat>> P_ = piqp_optional_mat_map(P, workspace->solver_info.n, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CVec>> c_ = piqp_optional_vec_map(c, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CMat>> A_ = piqp_optional_mat_map(A, workspace->solver_info.p, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CVec>> b_ = piqp_optional_vec_map(b, workspace->solver_info.p);
    piqp::optional<Eigen::Map<CMat>> G_ = piqp_optional_mat_map(G, workspace->solver_info.m, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CVec>> h_l_ = piqp_optional_vec_map(h_l, workspace->solver_info.m);
    piqp::optional<Eigen::Map<CVec>> h_u_ = piqp_optional_vec_map(h_u, workspace->solver_info.m);
    piqp::optional<Eigen::Map<CVec>> x_l_ = piqp_optional_vec_map(x_l, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CVec>> x_u_ = piqp_optional_vec_map(x_u, workspace->solver_info.n);

    auto* solver = reinterpret_cast<DenseSolver*>(workspace->solver_handle);
    solver->update(P_, c_, A_, b_, G_, h_l_, h_u_, x_l_, x_u_);

    if (workspace->gradients) workspace->gradients->status = (piqp_status) solver->gradients().status;
}

void piqp_update_sparse(piqp_workspace* workspace,
                        piqp_csc* P, piqp_float* c,
                        piqp_csc* A, piqp_float* b,
                        piqp_csc* G, piqp_float* h_l, piqp_float* h_u,
                        piqp_float* x_l, piqp_float* x_u)
{
    piqp::optional<Eigen::Map<CSparseMat>> P_ = piqp_optional_sparse_mat_map(P);
    piqp::optional<Eigen::Map<CVec>> c_ = piqp_optional_vec_map(c, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CSparseMat>> A_ = piqp_optional_sparse_mat_map(A);
    piqp::optional<Eigen::Map<CVec>> b_ = piqp_optional_vec_map(b, workspace->solver_info.p);
    piqp::optional<Eigen::Map<CSparseMat>> G_ = piqp_optional_sparse_mat_map(G);
    piqp::optional<Eigen::Map<CVec>> h_l_ = piqp_optional_vec_map(h_l, workspace->solver_info.m);
    piqp::optional<Eigen::Map<CVec>> h_u_ = piqp_optional_vec_map(h_u, workspace->solver_info.m);
    piqp::optional<Eigen::Map<CVec>> x_l_ = piqp_optional_vec_map(x_l, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CVec>> x_u_ = piqp_optional_vec_map(x_u, workspace->solver_info.n);

    auto* solver = reinterpret_cast<SparseSolver*>(workspace->solver_handle);
    solver->update(P_, c_, A_, b_, G_, h_l_, h_u_, x_l_, x_u_);

    if (workspace->gradients) workspace->gradients->status = (piqp_status) solver->gradients().status;
}

piqp_status piqp_solve(piqp_workspace* workspace)
{
    piqp::Status status;
    if (workspace->solver_info.is_dense)
    {
        auto* solver = reinterpret_cast<DenseSolver*>(workspace->solver_handle);
        status = solver->solve();
        piqp_update_result(workspace->result, solver->result());
    }
    else
    {
        auto* solver = reinterpret_cast<SparseSolver*>(workspace->solver_handle);
        status = solver->solve();
        piqp_update_result(workspace->result, solver->result());
    }

    return (piqp_status) status;
}

static piqp::Result<piqp_float> piqp_result_from_c(const piqp_workspace* workspace, const piqp_result* result)
{
    const piqp_int n = workspace->solver_info.n;
    const piqp_int p = workspace->solver_info.p;
    const piqp_int m = workspace->solver_info.m;

    piqp::Result<piqp_float> cpp_result;
    cpp_result.x = Eigen::Map<const CVec>(result->x, n);
    cpp_result.y = Eigen::Map<const CVec>(result->y, p);
    cpp_result.z_l = Eigen::Map<const CVec>(result->z_l, m);
    cpp_result.z_u = Eigen::Map<const CVec>(result->z_u, m);
    cpp_result.z_bl = Eigen::Map<const CVec>(result->z_bl, n);
    cpp_result.z_bu = Eigen::Map<const CVec>(result->z_bu, n);
    cpp_result.s_l = Eigen::Map<const CVec>(result->s_l, m);
    cpp_result.s_u = Eigen::Map<const CVec>(result->s_u, m);
    cpp_result.s_bl = Eigen::Map<const CVec>(result->s_bl, n);
    cpp_result.s_bu = Eigen::Map<const CVec>(result->s_bu, n);
    // only the status and the regularization are used by backward()
    cpp_result.info.status = (piqp::Status) result->info.status;
    cpp_result.info.rho = result->info.rho;
    cpp_result.info.delta = result->info.delta;
    return cpp_result;
}

template<typename Solver>
static piqp::Status piqp_backward_impl(Solver& solver, const piqp_workspace* workspace,
                                       const piqp_float* grad_x, const piqp_float* grad_y,
                                       const piqp_float* grad_z_l, const piqp_float* grad_z_u,
                                       const piqp_float* grad_z_bl, const piqp_float* grad_z_bu,
                                       const piqp_float* grad_s_l, const piqp_float* grad_s_u,
                                       const piqp_float* grad_s_bl, const piqp_float* grad_s_bu,
                                       const piqp_result* forward_result)
{
    const piqp_int n = workspace->solver_info.n;
    const piqp_int p = workspace->solver_info.p;
    const piqp_int m = workspace->solver_info.m;

    piqp::optional<piqp::Result<piqp_float>> cpp_forward_result;
    if (forward_result)
    {
        cpp_forward_result = piqp_result_from_c(workspace, forward_result);
    }

    return solver.backward(piqp_optional_const_vec_map(grad_x, n),
                           piqp_optional_const_vec_map(grad_y, p),
                           piqp_optional_const_vec_map(grad_z_l, m),
                           piqp_optional_const_vec_map(grad_z_u, m),
                           piqp_optional_const_vec_map(grad_z_bl, n),
                           piqp_optional_const_vec_map(grad_z_bu, n),
                           piqp_optional_const_vec_map(grad_s_l, m),
                           piqp_optional_const_vec_map(grad_s_u, m),
                           piqp_optional_const_vec_map(grad_s_bl, n),
                           piqp_optional_const_vec_map(grad_s_bu, n),
                           cpp_forward_result);
}

piqp_status piqp_backward(piqp_workspace* workspace,
                          const piqp_float* grad_x, const piqp_float* grad_y,
                          const piqp_float* grad_z_l, const piqp_float* grad_z_u,
                          const piqp_float* grad_z_bl, const piqp_float* grad_z_bu,
                          const piqp_float* grad_s_l, const piqp_float* grad_s_u,
                          const piqp_float* grad_s_bl, const piqp_float* grad_s_bu,
                          const piqp_result* forward_result)
{
    piqp::Status status;
    if (workspace->solver_info.is_dense)
    {
        auto* solver = reinterpret_cast<DenseSolver*>(workspace->solver_handle);
        status = piqp_backward_impl(*solver, workspace, grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu,
                                    grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, forward_result);
        if (status == piqp::Status::PIQP_SOLVED)
        {
            piqp_copy_gradients_dense(workspace, *solver);
        }
    }
    else
    {
        auto* solver = reinterpret_cast<SparseSolver*>(workspace->solver_handle);
        status = piqp_backward_impl(*solver, workspace, grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu,
                                    grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, forward_result);
    }

    if (workspace->gradients) workspace->gradients->status = (piqp_status) status;

    return (piqp_status) status;
}

void piqp_cleanup(piqp_workspace* workspace)
{
    if (workspace)
    {
        if (workspace->solver_info.is_dense)
        {
            auto* solver = reinterpret_cast<DenseSolver*>(workspace->solver_handle);
            delete solver;
        }
        else
        {
            auto* solver = reinterpret_cast<SparseSolver *>(workspace->solver_handle);
            delete solver;
        }
        piqp_free_gradients(workspace);
        delete workspace->result;
        delete workspace;
    }
}
