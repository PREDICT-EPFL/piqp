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

void piqp_update_results(piqp_results* results, const piqp::Result<piqp_float>& solver_results)
{
    results->x = solver_results.x.data();
    results->y = solver_results.y.data();
    results->z = solver_results.z.data();
    results->z_lb = solver_results.z_lb.data();
    results->z_ub = solver_results.z_ub.data();
    results->s = solver_results.s.data();
    results->s_lb = solver_results.s_lb.data();
    results->s_ub = solver_results.s_ub.data();

    results->zeta = solver_results.zeta.data();
    results->lambda = solver_results.lambda.data();
    results->nu = solver_results.nu.data();
    results->nu_lb = solver_results.nu_lb.data();
    results->nu_ub = solver_results.nu_ub.data();

    results->info.status = (piqp_status) solver_results.info.status;
    results->info.iter = (piqp_int) solver_results.info.iter;
    results->info.rho = solver_results.info.rho;
    results->info.delta = solver_results.info.delta;
    results->info.mu = solver_results.info.mu;
    results->info.sigma = solver_results.info.sigma;
    results->info.primal_step = solver_results.info.primal_step;
    results->info.dual_step = solver_results.info.dual_step;
    results->info.primal_inf = solver_results.info.primal_inf;
    results->info.primal_rel_inf = solver_results.info.primal_rel_inf;
    results->info.dual_inf = solver_results.info.dual_inf;
    results->info.dual_rel_inf = solver_results.info.dual_rel_inf;
    results->info.primal_obj = solver_results.info.primal_obj;
    results->info.dual_obj = solver_results.info.dual_obj;
    results->info.duality_gap = solver_results.info.duality_gap;
    results->info.duality_gap_rel = solver_results.info.duality_gap_rel;
    results->info.factor_retires = solver_results.info.factor_retires;
    results->info.reg_limit = solver_results.info.reg_limit;
    results->info.no_primal_update = solver_results.info.no_primal_update;
    results->info.no_dual_update = solver_results.info.no_dual_update;
    results->info.setup_time = solver_results.info.setup_time;
    results->info.update_time = solver_results.info.update_time;
    results->info.solve_time = solver_results.info.solve_time;
    results->info.run_time = solver_results.info.run_time;
}

void piqp_set_default_settings(piqp_settings* settings)
{
    piqp::Settings<piqp_float> default_settings;

    settings->rho_init = default_settings.rho_init;
    settings->delta_init = default_settings.delta_init;
    settings->eps_abs = default_settings.eps_abs;
    settings->eps_rel = default_settings.eps_rel;
    settings->check_duality_gap = default_settings.check_duality_gap;
    settings->eps_duality_gap_abs = default_settings.eps_duality_gap_abs;
    settings->eps_duality_gap_rel = default_settings.eps_duality_gap_rel;
    settings->reg_lower_limit = default_settings.reg_lower_limit;
    settings->reg_finetune_lower_limit = default_settings.reg_finetune_lower_limit;
    settings->max_iter = (piqp_int) default_settings.max_iter;
    settings->max_factor_retires = (piqp_int) default_settings.max_factor_retires;
    settings->preconditioner_scale_cost = default_settings.preconditioner_scale_cost;
    settings->preconditioner_iter = (piqp_int) default_settings.preconditioner_iter;
    settings->tau = default_settings.tau;
    settings->verbose = default_settings.verbose;
    settings->compute_timings = default_settings.compute_timings;
}

piqp::optional<Eigen::Map<CVec>> piqp_optional_vec_map(piqp_float* data, piqp_int n)
{
    piqp::optional<Eigen::Map<CVec>> vec;
    if (data)
    {
        vec = Eigen::Map<CVec>(data, n);
    }
    return vec;
}

piqp::optional<Eigen::Map<CMat>> piqp_optional_mat_map(piqp_float* data, piqp_int m, piqp_int n)
{
    piqp::optional<Eigen::Map<CMat>> mat;
    if (data)
    {
        mat = Eigen::Map<CMat>(data, m, n);
    }
    return mat;
}

piqp::optional<Eigen::Map<CSparseMat>> piqp_optional_sparse_mat_map(piqp_csc* data)
{
    piqp::optional<Eigen::Map<CSparseMat>> mat;
    if (data)
    {
        mat = Eigen::Map<CSparseMat>(data->m, data->n, data->nnz, data->p, data->i, data->x);
    }
    return mat;
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
    work->results = new piqp_results;

    if (settings)
    {
        piqp_update_settings(work, settings);
    }

    Eigen::Map<CMat> P(data->P, data->n, data->n);
    Eigen::Map<CVec> c(data->c, data->n);
    Eigen::Map<CMat> A(data->A, data->p, data->n);
    Eigen::Map<CVec> b(data->b, data->p);
    Eigen::Map<CMat> G(data->G, data->m, data->n);
    Eigen::Map<CVec> h(data->h, data->m);
    piqp::optional<Eigen::Map<CVec>> x_lb = piqp_optional_vec_map(data->x_lb, data->n);
    piqp::optional<Eigen::Map<CVec>> x_ub = piqp_optional_vec_map(data->x_ub, data->n);

    solver->setup(P, c, A, b, G, h, x_lb, x_ub);

    piqp_update_results(work->results, solver->result());
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
    work->results = new piqp_results;

    if (settings)
    {
        piqp_update_settings(work, settings);
    }

    Eigen::Map<CSparseMat> P(data->P->m, data->P->n, data->P->nnz, data->P->p, data->P->i, data->P->x);
    Eigen::Map<CVec> c(data->c, data->n);
    Eigen::Map<CSparseMat> A(data->A->m, data->A->n, data->A->nnz, data->A->p, data->A->i, data->A->x);
    Eigen::Map<CVec> b(data->b, data->p);
    Eigen::Map<CSparseMat> G(data->G->m, data->G->n, data->G->nnz, data->G->p, data->G->i, data->G->x);
    Eigen::Map<CVec> h(data->h, data->m);
    piqp::optional<Eigen::Map<CVec>> x_lb = piqp_optional_vec_map(data->x_lb, data->n);
    piqp::optional<Eigen::Map<CVec>> x_ub = piqp_optional_vec_map(data->x_ub, data->n);

    solver->setup(P, c, A, b, G, h, x_lb, x_ub);

    piqp_update_results(work->results, solver->result());
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
        solver->settings().reg_lower_limit = settings->reg_lower_limit;
        solver->settings().reg_finetune_lower_limit = settings->reg_finetune_lower_limit;
        solver->settings().max_iter = settings->max_iter;
        solver->settings().max_factor_retires = settings->max_factor_retires;
        solver->settings().preconditioner_scale_cost = settings->preconditioner_scale_cost;
        solver->settings().preconditioner_iter = settings->preconditioner_iter;
        solver->settings().tau = settings->tau;
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
        solver->settings().reg_lower_limit = settings->reg_lower_limit;
        solver->settings().reg_finetune_lower_limit = settings->reg_finetune_lower_limit;
        solver->settings().max_iter = settings->max_iter;
        solver->settings().max_factor_retires = settings->max_factor_retires;
        solver->settings().preconditioner_scale_cost = settings->preconditioner_scale_cost;
        solver->settings().preconditioner_iter = settings->preconditioner_iter;
        solver->settings().tau = settings->tau;
        solver->settings().verbose = settings->verbose;
        solver->settings().compute_timings = settings->compute_timings;
    }
}

void piqp_update_dense(piqp_workspace* workspace,
                       piqp_float* P, piqp_float* c,
                       piqp_float* A, piqp_float* b,
                       piqp_float* G, piqp_float* h,
                       piqp_float* x_lb, piqp_float* x_ub)
{
    piqp::optional<Eigen::Map<CMat>> P_ = piqp_optional_mat_map(P, workspace->solver_info.n, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CVec>> c_ = piqp_optional_vec_map(c, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CMat>> A_ = piqp_optional_mat_map(A, workspace->solver_info.p, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CVec>> b_ = piqp_optional_vec_map(b, workspace->solver_info.p);
    piqp::optional<Eigen::Map<CMat>> G_ = piqp_optional_mat_map(G, workspace->solver_info.m, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CVec>> h_ = piqp_optional_vec_map(h, workspace->solver_info.m);
    piqp::optional<Eigen::Map<CVec>> x_lb_ = piqp_optional_vec_map(x_lb, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CVec>> x_ub_ = piqp_optional_vec_map(x_ub, workspace->solver_info.n);

    auto* solver = reinterpret_cast<DenseSolver*>(workspace->solver_handle);
    solver->update(P_, c_, A_, b_, G_, h_, x_lb_, x_ub_);
}

void piqp_update_sparse(piqp_workspace* workspace,
                        piqp_csc* P, piqp_float* c,
                        piqp_csc* A, piqp_float* b,
                        piqp_csc* G, piqp_float* h,
                        piqp_float* x_lb, piqp_float* x_ub)
{
    piqp::optional<Eigen::Map<CSparseMat>> P_ = piqp_optional_sparse_mat_map(P);
    piqp::optional<Eigen::Map<CVec>> c_ = piqp_optional_vec_map(c, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CSparseMat>> A_ = piqp_optional_sparse_mat_map(A);
    piqp::optional<Eigen::Map<CVec>> b_ = piqp_optional_vec_map(b, workspace->solver_info.p);
    piqp::optional<Eigen::Map<CSparseMat>> G_ = piqp_optional_sparse_mat_map(G);
    piqp::optional<Eigen::Map<CVec>> h_ = piqp_optional_vec_map(h, workspace->solver_info.m);
    piqp::optional<Eigen::Map<CVec>> x_lb_ = piqp_optional_vec_map(x_lb, workspace->solver_info.n);
    piqp::optional<Eigen::Map<CVec>> x_ub_ = piqp_optional_vec_map(x_ub, workspace->solver_info.n);

    auto* solver = reinterpret_cast<SparseSolver*>(workspace->solver_handle);
    solver->update(P_, c_, A_, b_, G_, h_, x_lb_, x_ub_);
}

piqp_status piqp_solve(piqp_workspace* workspace)
{
    piqp::Status status;
    if (workspace->solver_info.is_dense)
    {
        auto* solver = reinterpret_cast<DenseSolver*>(workspace->solver_handle);
        status = solver->solve();
        piqp_update_results(workspace->results, solver->result());
    }
    else
    {
        auto* solver = reinterpret_cast<SparseSolver*>(workspace->solver_handle);
        status = solver->solve();
        piqp_update_results(workspace->results, solver->result());
    }

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
        delete workspace->results;
        delete workspace;
    }
}
