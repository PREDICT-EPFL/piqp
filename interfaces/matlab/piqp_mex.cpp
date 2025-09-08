// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2017 Bartolomeo Stellato
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include "mex.h"
#include "piqp/piqp.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define PIQP_MEX_SIGNATURE 0x271C1A7A

#ifndef PIQP_VERSION
#define PIQP_VERSION dev
#endif

using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using IVec = Eigen::Matrix<int, Eigen::Dynamic, 1>;
using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

using DenseSolver = piqp::DenseSolver<double>;
using SparseSolver = piqp::SparseSolver<double, int>;

const char* PIQP_SETTINGS_FIELDS[] = {"rho_init",
                                      "delta_init",
                                      "eps_abs",
                                      "eps_rel",
                                      "check_duality_gap",
                                      "eps_duality_gap_abs",
                                      "eps_duality_gap_rel",
                                      "infeasibility_threshold",
                                      "reg_lower_limit",
                                      "reg_finetune_lower_limit",
                                      "reg_finetune_primal_update_threshold",
                                      "reg_finetune_dual_update_threshold",
                                      "max_iter",
                                      "max_factor_retires",
                                      "preconditioner_scale_cost",
                                      "preconditioner_reuse_on_update",
                                      "preconditioner_iter",
                                      "tau",
                                      "kkt_solver",
                                      "iterative_refinement_always_enabled",
                                      "iterative_refinement_eps_abs",
                                      "iterative_refinement_eps_rel",
                                      "iterative_refinement_max_iter",
                                      "iterative_refinement_min_improvement_rate",
                                      "iterative_refinement_static_regularization_eps",
                                      "iterative_refinement_static_regularization_rel",
                                      "verbose",
                                      "compute_timings"};

const char* PIQP_INFO_FIELDS[] = {"status",
                                  "status_val",
                                  "iter",
                                  "rho",
                                  "delta",
                                  "mu",
                                  "sigma",
                                  "primal_step",
                                  "dual_step",
                                  "primal_res",
                                  "primal_res_rel",
                                  "dual_res",
                                  "dual_res_rel",
                                  "primal_res_reg",
                                  "primal_res_reg_rel",
                                  "dual_res_reg",
                                  "dual_res_reg_rel",
                                  "primal_prox_inf",
                                  "dual_prox_inf",
                                  "prev_primal_res",
                                  "prev_dual_res",
                                  "primal_obj",
                                  "dual_obj",
                                  "duality_gap",
                                  "duality_gap_rel",
                                  "factor_retires",
                                  "reg_limit",
                                  "no_primal_update",
                                  "no_dual_update",
                                  "setup_time",
                                  "update_time",
                                  "solve_time",
                                  "kkt_factor_time",
                                  "kkt_solve_time",
                                  "run_time"};

const char* PIQP_RESULT_FIELDS[] = {"x",
                                    "y",
                                    "z_l",
                                    "z_u",
                                    "z_bl",
                                    "z_bu",
                                    "s_l",
                                    "s_u",
                                    "s_bl",
                                    "s_bu",
                                    "info"};

class piqp_mex_handle
{
public:
    explicit piqp_mex_handle(DenseSolver* ptr) : m_signature(PIQP_MEX_SIGNATURE), m_is_dense(true), m_ptr(ptr) {}
    explicit piqp_mex_handle(SparseSolver* ptr) : m_signature(PIQP_MEX_SIGNATURE), m_is_dense(false), m_ptr(ptr) {}
    bool isValid() const { return m_signature == PIQP_MEX_SIGNATURE; }
    bool isDense() const { return m_is_dense; }
    DenseSolver* as_dense_ptr() { return static_cast<DenseSolver*>(m_ptr); }
    SparseSolver* as_sparse_ptr() { return static_cast<SparseSolver*>(m_ptr); }

private:
    uint32_t m_signature;
    bool m_is_dense;
    void* m_ptr;
};

template<typename T>
inline mxArray* create_mex_handle(T* ptr)
{
    mexLock();
    mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*) mxGetData(out)) = reinterpret_cast<uint64_t>(new piqp_mex_handle(ptr));
    return out;
}

inline piqp_mex_handle* get_mex_handle(const mxArray* in)
{
    if (mxGetNumberOfElements(in) != 1 || mxGetClassID(in) != mxUINT64_CLASS || mxIsComplex(in)) {
        mexErrMsgTxt("Input must be a real uint64 scalar.");
    }
    auto *ptr = reinterpret_cast<piqp_mex_handle*>(*((uint64_t*) mxGetData(in)));
    if (!ptr->isValid()) {
        mexErrMsgTxt("Handle not valid.");
    }
    return ptr;
}

inline void destroy_mex_handle(const mxArray* in)
{
    delete get_mex_handle(in);
    mexUnlock();
}

inline IVec to_int_vec(mwIndex* data, int n)
{
    return Eigen::Map<Eigen::Matrix<mwIndex, Eigen::Dynamic, 1>>(data, n).cast<int>();
}

inline mxArray* eigen_to_mx(const Vec& vec)
{
    mxArray* mx_ptr = mxCreateDoubleMatrix((mwSize) vec.rows(), 1, mxREAL);
    Eigen::Map<Vec>(mxGetPr(mx_ptr), vec.rows()) = vec;
    return mx_ptr;
}

piqp::KKTSolver kkt_solver_from_string(const char* kkt_solver, bool is_dense)
{
    std::string kkt_solver_str(kkt_solver);
    if (kkt_solver_str == "dense_cholesky") return piqp::KKTSolver::dense_cholesky;
    if (kkt_solver_str == "sparse_ldlt") return piqp::KKTSolver::sparse_ldlt;
    if (kkt_solver_str == "sparse_ldlt_eq_cond") return piqp::KKTSolver::sparse_ldlt_eq_cond;
    if (kkt_solver_str == "sparse_ldlt_ineq_cond") return piqp::KKTSolver::sparse_ldlt_ineq_cond;
    if (kkt_solver_str == "sparse_ldlt_cond") return piqp::KKTSolver::sparse_ldlt_cond;
    if (kkt_solver_str == "sparse_multistage") return piqp::KKTSolver::sparse_multistage;
    if (is_dense) {
        mexWarnMsgTxt("Unknown kkt_solver, using dense_cholesky as a fallback.");
        return piqp::KKTSolver::dense_cholesky;
    }
    mexWarnMsgTxt("Unknown kkt_solver, using sparse_ldlt as a fallback.");
    return piqp::KKTSolver::sparse_ldlt;
}

mxArray* settings_to_mx_struct(const piqp::Settings<double>& settings)
{
    int n_fields  = sizeof(PIQP_SETTINGS_FIELDS) / sizeof(PIQP_SETTINGS_FIELDS[0]);
    mxArray* mx_ptr = mxCreateStructMatrix(1, 1, n_fields, PIQP_SETTINGS_FIELDS);

    mxSetField(mx_ptr, 0, "rho_init", mxCreateDoubleScalar(settings.rho_init));
    mxSetField(mx_ptr, 0, "delta_init", mxCreateDoubleScalar(settings.delta_init));
    mxSetField(mx_ptr, 0, "eps_abs", mxCreateDoubleScalar(settings.eps_abs));
    mxSetField(mx_ptr, 0, "eps_rel", mxCreateDoubleScalar(settings.eps_rel));
    mxSetField(mx_ptr, 0, "check_duality_gap", mxCreateDoubleScalar(settings.check_duality_gap));
    mxSetField(mx_ptr, 0, "eps_duality_gap_abs", mxCreateDoubleScalar(settings.eps_duality_gap_abs));
    mxSetField(mx_ptr, 0, "eps_duality_gap_rel", mxCreateDoubleScalar(settings.eps_duality_gap_rel));
    mxSetField(mx_ptr, 0, "infeasibility_threshold", mxCreateDoubleScalar(settings.infeasibility_threshold));
    mxSetField(mx_ptr, 0, "reg_lower_limit", mxCreateDoubleScalar(settings.reg_lower_limit));
    mxSetField(mx_ptr, 0, "reg_finetune_lower_limit", mxCreateDoubleScalar(settings.reg_finetune_lower_limit));
    mxSetField(mx_ptr, 0, "reg_finetune_primal_update_threshold", mxCreateDoubleScalar((double) settings.reg_finetune_primal_update_threshold));
    mxSetField(mx_ptr, 0, "reg_finetune_dual_update_threshold", mxCreateDoubleScalar((double) settings.reg_finetune_dual_update_threshold));
    mxSetField(mx_ptr, 0, "max_iter", mxCreateDoubleScalar((double) settings.max_iter));
    mxSetField(mx_ptr, 0, "max_factor_retires", mxCreateDoubleScalar((double) settings.max_factor_retires));
    mxSetField(mx_ptr, 0, "preconditioner_scale_cost", mxCreateDoubleScalar(settings.preconditioner_scale_cost));
    mxSetField(mx_ptr, 0, "preconditioner_reuse_on_update", mxCreateDoubleScalar(settings.preconditioner_reuse_on_update));
    mxSetField(mx_ptr, 0, "preconditioner_iter", mxCreateDoubleScalar((double) settings.preconditioner_iter));
    mxSetField(mx_ptr, 0, "tau", mxCreateDoubleScalar(settings.tau));
    mxSetField(mx_ptr, 0, "kkt_solver", mxCreateString(piqp::kkt_solver_to_string(settings.kkt_solver)));
    mxSetField(mx_ptr, 0, "iterative_refinement_always_enabled", mxCreateDoubleScalar(settings.iterative_refinement_always_enabled));
    mxSetField(mx_ptr, 0, "iterative_refinement_eps_abs", mxCreateDoubleScalar(settings.iterative_refinement_eps_abs));
    mxSetField(mx_ptr, 0, "iterative_refinement_eps_rel", mxCreateDoubleScalar(settings.iterative_refinement_eps_rel));
    mxSetField(mx_ptr, 0, "iterative_refinement_max_iter", mxCreateDoubleScalar((double) settings.iterative_refinement_max_iter));
    mxSetField(mx_ptr, 0, "iterative_refinement_min_improvement_rate", mxCreateDoubleScalar(settings.iterative_refinement_min_improvement_rate));
    mxSetField(mx_ptr, 0, "iterative_refinement_static_regularization_eps", mxCreateDoubleScalar(settings.iterative_refinement_static_regularization_eps));
    mxSetField(mx_ptr, 0, "iterative_refinement_static_regularization_rel", mxCreateDoubleScalar(settings.iterative_refinement_static_regularization_rel));
    mxSetField(mx_ptr, 0, "verbose", mxCreateDoubleScalar(settings.verbose));
    mxSetField(mx_ptr, 0, "compute_timings", mxCreateDoubleScalar(settings.compute_timings));

    return mx_ptr;
}

void copy_mx_struct_to_settings(const mxArray* mx_ptr, piqp::Settings<double>& settings, bool is_dense)
{
    settings.rho_init = (double) mxGetScalar(mxGetField(mx_ptr, 0, "rho_init"));
    settings.delta_init = (double) mxGetScalar(mxGetField(mx_ptr, 0, "delta_init"));
    settings.eps_abs = (double) mxGetScalar(mxGetField(mx_ptr, 0, "eps_abs"));
    settings.eps_rel = (double) mxGetScalar(mxGetField(mx_ptr, 0, "eps_rel"));
    settings.check_duality_gap = (bool) mxGetScalar(mxGetField(mx_ptr, 0, "check_duality_gap"));
    settings.eps_duality_gap_abs = (double) mxGetScalar(mxGetField(mx_ptr, 0, "eps_duality_gap_abs"));
    settings.eps_duality_gap_rel = (double) mxGetScalar(mxGetField(mx_ptr, 0, "eps_duality_gap_rel"));
    settings.infeasibility_threshold = (double) mxGetScalar(mxGetField(mx_ptr, 0, "infeasibility_threshold"));
    settings.reg_lower_limit = (double) mxGetScalar(mxGetField(mx_ptr, 0, "reg_lower_limit"));
    settings.reg_finetune_lower_limit = (double) mxGetScalar(mxGetField(mx_ptr, 0, "reg_finetune_lower_limit"));
    settings.reg_finetune_primal_update_threshold = (piqp::isize) mxGetScalar(mxGetField(mx_ptr, 0, "reg_finetune_primal_update_threshold"));
    settings.reg_finetune_dual_update_threshold = (piqp::isize) mxGetScalar(mxGetField(mx_ptr, 0, "reg_finetune_dual_update_threshold"));
    settings.max_iter = (piqp::isize) mxGetScalar(mxGetField(mx_ptr, 0, "max_iter"));
    settings.max_factor_retires = (piqp::isize) mxGetScalar(mxGetField(mx_ptr, 0, "max_factor_retires"));
    settings.preconditioner_scale_cost = (bool) mxGetScalar(mxGetField(mx_ptr, 0, "preconditioner_scale_cost"));
    settings.preconditioner_reuse_on_update = (bool) mxGetScalar(mxGetField(mx_ptr, 0, "preconditioner_reuse_on_update"));
    settings.preconditioner_iter = (piqp::isize) mxGetScalar(mxGetField(mx_ptr, 0, "preconditioner_iter"));
    settings.tau = (double) mxGetScalar(mxGetField(mx_ptr, 0, "tau"));
    char kkt_solver[30];
    mxGetString(mxGetField(mx_ptr, 0, "kkt_solver"), kkt_solver, 30);
    settings.kkt_solver = kkt_solver_from_string(kkt_solver, is_dense);
    settings.iterative_refinement_always_enabled = (bool) mxGetScalar(mxGetField(mx_ptr, 0, "iterative_refinement_always_enabled"));
    settings.iterative_refinement_eps_abs = (double) mxGetScalar(mxGetField(mx_ptr, 0, "iterative_refinement_eps_abs"));
    settings.iterative_refinement_eps_rel = (double) mxGetScalar(mxGetField(mx_ptr, 0, "iterative_refinement_eps_rel"));
    settings.iterative_refinement_max_iter = (piqp::isize) mxGetScalar(mxGetField(mx_ptr, 0, "iterative_refinement_max_iter"));
    settings.iterative_refinement_min_improvement_rate = (double) mxGetScalar(mxGetField(mx_ptr, 0, "iterative_refinement_min_improvement_rate"));
    settings.iterative_refinement_static_regularization_eps = (double) mxGetScalar(mxGetField(mx_ptr, 0, "iterative_refinement_static_regularization_eps"));
    settings.iterative_refinement_static_regularization_rel = (double) mxGetScalar(mxGetField(mx_ptr, 0, "iterative_refinement_static_regularization_rel"));
    settings.verbose = (bool) mxGetScalar(mxGetField(mx_ptr, 0, "verbose"));
    settings.compute_timings = (bool) mxGetScalar(mxGetField(mx_ptr, 0, "compute_timings"));
}

mxArray* result_to_mx_struct(const piqp::Result<double>& result)
{
    int n_info_fields  = sizeof(PIQP_INFO_FIELDS) / sizeof(PIQP_INFO_FIELDS[0]);
    mxArray* mx_info_ptr = mxCreateStructMatrix(1, 1, n_info_fields, PIQP_INFO_FIELDS);

    mxSetField(mx_info_ptr, 0, "status", mxCreateString(piqp::status_to_string(result.info.status)));
    mxSetField(mx_info_ptr, 0, "status_val", mxCreateDoubleScalar((double) result.info.status));
    mxSetField(mx_info_ptr, 0, "iter", mxCreateDoubleScalar((double) result.info.iter));
    mxSetField(mx_info_ptr, 0, "rho", mxCreateDoubleScalar(result.info.rho));
    mxSetField(mx_info_ptr, 0, "delta", mxCreateDoubleScalar(result.info.delta));
    mxSetField(mx_info_ptr, 0, "mu", mxCreateDoubleScalar(result.info.mu));
    mxSetField(mx_info_ptr, 0, "sigma", mxCreateDoubleScalar(result.info.sigma));
    mxSetField(mx_info_ptr, 0, "primal_step", mxCreateDoubleScalar(result.info.primal_step));
    mxSetField(mx_info_ptr, 0, "dual_step", mxCreateDoubleScalar(result.info.dual_step));
    mxSetField(mx_info_ptr, 0, "primal_res", mxCreateDoubleScalar(result.info.primal_res));
    mxSetField(mx_info_ptr, 0, "primal_res_rel", mxCreateDoubleScalar(result.info.primal_res_rel));
    mxSetField(mx_info_ptr, 0, "dual_res", mxCreateDoubleScalar(result.info.dual_res));
    mxSetField(mx_info_ptr, 0, "dual_res_rel", mxCreateDoubleScalar(result.info.dual_res_rel));
    mxSetField(mx_info_ptr, 0, "primal_res_reg", mxCreateDoubleScalar(result.info.primal_res_reg));
    mxSetField(mx_info_ptr, 0, "primal_res_reg_rel", mxCreateDoubleScalar(result.info.primal_res_reg_rel));
    mxSetField(mx_info_ptr, 0, "dual_res_reg", mxCreateDoubleScalar(result.info.dual_res_reg));
    mxSetField(mx_info_ptr, 0, "dual_res_reg_rel", mxCreateDoubleScalar(result.info.dual_res_reg_rel));
    mxSetField(mx_info_ptr, 0, "primal_prox_inf", mxCreateDoubleScalar(result.info.primal_prox_inf));
    mxSetField(mx_info_ptr, 0, "dual_prox_inf", mxCreateDoubleScalar(result.info.dual_prox_inf));
    mxSetField(mx_info_ptr, 0, "prev_primal_res", mxCreateDoubleScalar(result.info.prev_primal_res));
    mxSetField(mx_info_ptr, 0, "prev_dual_res", mxCreateDoubleScalar(result.info.prev_dual_res));
    mxSetField(mx_info_ptr, 0, "primal_obj", mxCreateDoubleScalar(result.info.primal_obj));
    mxSetField(mx_info_ptr, 0, "dual_obj", mxCreateDoubleScalar(result.info.dual_obj));
    mxSetField(mx_info_ptr, 0, "duality_gap", mxCreateDoubleScalar(result.info.duality_gap));
    mxSetField(mx_info_ptr, 0, "duality_gap_rel", mxCreateDoubleScalar(result.info.duality_gap_rel));
    mxSetField(mx_info_ptr, 0, "factor_retires", mxCreateDoubleScalar((double) result.info.factor_retires));
    mxSetField(mx_info_ptr, 0, "reg_limit", mxCreateDoubleScalar(result.info.reg_limit));
    mxSetField(mx_info_ptr, 0, "no_primal_update", mxCreateDoubleScalar((double) result.info.no_primal_update));
    mxSetField(mx_info_ptr, 0, "no_dual_update", mxCreateDoubleScalar((double) result.info.no_dual_update));
    mxSetField(mx_info_ptr, 0, "setup_time", mxCreateDoubleScalar(result.info.setup_time));
    mxSetField(mx_info_ptr, 0, "update_time", mxCreateDoubleScalar(result.info.update_time));
    mxSetField(mx_info_ptr, 0, "solve_time", mxCreateDoubleScalar(result.info.solve_time));
    mxSetField(mx_info_ptr, 0, "kkt_factor_time", mxCreateDoubleScalar(result.info.kkt_factor_time));
    mxSetField(mx_info_ptr, 0, "kkt_solve_time", mxCreateDoubleScalar(result.info.kkt_solve_time));
    mxSetField(mx_info_ptr, 0, "run_time", mxCreateDoubleScalar(result.info.run_time));

    int n_result_fields  = sizeof(PIQP_RESULT_FIELDS) / sizeof(PIQP_RESULT_FIELDS[0]);
    mxArray* mx_result_ptr = mxCreateStructMatrix(1, 1, n_result_fields, PIQP_RESULT_FIELDS);

    mxSetField(mx_result_ptr, 0, "x", eigen_to_mx(result.x));
    mxSetField(mx_result_ptr, 0, "y", eigen_to_mx(result.y));
    mxSetField(mx_result_ptr, 0, "z_l", eigen_to_mx(result.z_l));
    mxSetField(mx_result_ptr, 0, "z_u", eigen_to_mx(result.z_u));
    mxSetField(mx_result_ptr, 0, "z_bl", eigen_to_mx(result.z_bl));
    mxSetField(mx_result_ptr, 0, "z_bu", eigen_to_mx(result.z_bu));
    mxSetField(mx_result_ptr, 0, "s_l", eigen_to_mx(result.s_l));
    mxSetField(mx_result_ptr, 0, "s_u", eigen_to_mx(result.s_u));
    mxSetField(mx_result_ptr, 0, "s_bl", eigen_to_mx(result.s_bl));
    mxSetField(mx_result_ptr, 0, "s_bu", eigen_to_mx(result.s_bu));
    mxSetField(mx_result_ptr, 0, "info", mx_info_ptr);

    return mx_result_ptr;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // Get the command string
    char cmd[64];
    if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd))) {
        mexErrMsgTxt("First input should be a command string less than 64 characters long.");
    }

    if (!strcmp("new", cmd)) {
        char backend[10];
        if (nrhs < 2) {
            strcpy(backend, "sparse");
            mexWarnMsgTxt("The sparse backend is automatically used. To get rid of this warning or use another backend, "
                          "provide the backend explicitly using piqp('dense') or piqp('sparse').");
        } else if (mxGetString(prhs[1], backend, sizeof(backend))) {
            mexErrMsgTxt("Second input should be string less than 10 characters long.");
        }

        if (!strcmp("dense", backend)) {
            plhs[0] = create_mex_handle(new DenseSolver());
        } else if (!strcmp("sparse", backend)) {
            plhs[0] = create_mex_handle(new SparseSolver());
        } else {
            mexErrMsgTxt("Second input must be 'dense' or 'sparse'.");
        }
        return;
    }

    if (!strcmp("version", cmd)) {
        plhs[0] = mxCreateString(MACRO_STRINGIFY(PIQP_VERSION));
        return;
    }

    // Check for a second input
    if (nrhs < 2) {
        mexErrMsgTxt("Second input should be a class instance handle.");
    }
    piqp_mex_handle* mex_handle = get_mex_handle(prhs[1]);

    // delete the object and its data
    if (!strcmp("delete", cmd)) {
        if (mex_handle->isDense()) {
            if (mex_handle->as_dense_ptr()) {
                delete mex_handle->as_dense_ptr();
            }
        } else {
            if (mex_handle->as_sparse_ptr()) {
                delete mex_handle->as_sparse_ptr();
            }
        }

        //clean up the handle object
        destroy_mex_handle(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2) {
            mexWarnMsgTxt("Unexpected arguments ignored.");
        }
        return;
    }

    // Get settings
    if (!strcmp("get_settings", cmd)) {
        if (mex_handle->isDense()) {
            plhs[0] = settings_to_mx_struct(mex_handle->as_dense_ptr()->settings());
        } else {
            plhs[0] = settings_to_mx_struct(mex_handle->as_sparse_ptr()->settings());
        }
        return;
    }

    // Update settings
    if (!strcmp("update_settings", cmd)) {
        if (mex_handle->isDense()) {
            copy_mx_struct_to_settings(prhs[2], mex_handle->as_dense_ptr()->settings(), mex_handle->isDense());
        } else {
            copy_mx_struct_to_settings(prhs[2], mex_handle->as_sparse_ptr()->settings(), mex_handle->isDense());
        }
        return;
    }

    // Get problem dimensions
    if (!strcmp("get_dimensions", cmd)) {
        if (mex_handle->isDense()) {
            plhs[0] = mxCreateDoubleScalar((double) mex_handle->as_dense_ptr()->result().x.rows());
            plhs[1] = mxCreateDoubleScalar((double) mex_handle->as_dense_ptr()->result().y.rows());
            plhs[2] = mxCreateDoubleScalar((double) mex_handle->as_dense_ptr()->result().z_l.rows());
        } else {
            plhs[0] = mxCreateDoubleScalar((double) mex_handle->as_sparse_ptr()->result().x.rows());
            plhs[1] = mxCreateDoubleScalar((double) mex_handle->as_sparse_ptr()->result().y.rows());
            plhs[2] = mxCreateDoubleScalar((double) mex_handle->as_sparse_ptr()->result().z_l.rows());
        }
        return;
    }

    if (!strcmp("setup", cmd)) {
        const int n = (int) mxGetScalar(prhs[2]);
        const int p = (int) mxGetScalar(prhs[3]);
        const int m = (int) mxGetScalar(prhs[4]);

        const mxArray* P_ptr = prhs[5];
        const mxArray* c_ptr = prhs[6];
        const mxArray* A_ptr = prhs[7];
        const mxArray* b_ptr = prhs[8];
        const mxArray* G_ptr = prhs[9];
        const mxArray* h_l_ptr = prhs[10];
        const mxArray* h_u_ptr = prhs[11];
        const mxArray* x_l_ptr = prhs[12];
        const mxArray* x_u_ptr = prhs[13];

        Eigen::Map<Vec> c(mxGetPr(c_ptr), n);
        Eigen::Map<Vec> b(mxGetPr(b_ptr), p);
        Eigen::Map<Vec> h_l(mxGetPr(h_l_ptr), m);
        Eigen::Map<Vec> h_u(mxGetPr(h_u_ptr), m);
        Eigen::Map<Vec> x_l(mxGetPr(x_l_ptr), n);
        Eigen::Map<Vec> x_u(mxGetPr(x_u_ptr), n);

        if (mex_handle->isDense()) {
            copy_mx_struct_to_settings(prhs[14], mex_handle->as_dense_ptr()->settings(), mex_handle->isDense());

            Eigen::Map<Mat> P(mxGetPr(P_ptr), n, n);
            Eigen::Map<Mat> A(mxGetPr(A_ptr), p, n);
            Eigen::Map<Mat> G(mxGetPr(G_ptr), m, n);

            mex_handle->as_dense_ptr()->setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
        } else {
            copy_mx_struct_to_settings(prhs[14], mex_handle->as_sparse_ptr()->settings(), mex_handle->isDense());

            IVec Pp = to_int_vec(mxGetJc(P_ptr), n + 1);
            IVec Pi = to_int_vec(mxGetIr(P_ptr), Pp(n));
            Eigen::Map<SparseMat> P(n, n, (Eigen::Index) mxGetNzmax(P_ptr), Pp.data(), Pi.data(), mxGetPr(P_ptr));

            IVec Ap = to_int_vec(mxGetJc(A_ptr), n + 1);
            IVec Ai = to_int_vec(mxGetIr(A_ptr), Ap(n));
            Eigen::Map<SparseMat> A(p, n, (Eigen::Index) mxGetNzmax(A_ptr), Ap.data(), Ai.data(), mxGetPr(A_ptr));

            IVec Gp = to_int_vec(mxGetJc(G_ptr), n + 1);
            IVec Gi = to_int_vec(mxGetIr(G_ptr), Gp(n));
            Eigen::Map<SparseMat> G(m, n, (Eigen::Index) mxGetNzmax(G_ptr), Gp.data(), Gi.data(), mxGetPr(G_ptr));

            mex_handle->as_sparse_ptr()->setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
        }

        return;
    }

    if (!strcmp("solve", cmd)) {
        if (mex_handle->isDense()) {
            mex_handle->as_dense_ptr()->solve();
            plhs[0] = result_to_mx_struct(mex_handle->as_dense_ptr()->result());
        } else {
            mex_handle->as_sparse_ptr()->solve();
            plhs[0] = result_to_mx_struct(mex_handle->as_sparse_ptr()->result());
        }

        return;
    }

    if (!strcmp("update", cmd)) {
        const int n = (int) mxGetScalar(prhs[2]);
        const int p = (int) mxGetScalar(prhs[3]);
        const int m = (int) mxGetScalar(prhs[4]);

        const mxArray* P_ptr = prhs[5];
        const mxArray* c_ptr = prhs[6];
        const mxArray* A_ptr = prhs[7];
        const mxArray* b_ptr = prhs[8];
        const mxArray* G_ptr = prhs[9];
        const mxArray* h_l_ptr = prhs[10];
        const mxArray* h_u_ptr = prhs[11];
        const mxArray* x_l_ptr = prhs[12];
        const mxArray* x_u_ptr = prhs[13];

        piqp::optional<Eigen::Map<Vec>> c;
        piqp::optional<Eigen::Map<Vec>> b;
        piqp::optional<Eigen::Map<Vec>> h_l;
        piqp::optional<Eigen::Map<Vec>> h_u;
        piqp::optional<Eigen::Map<Vec>> x_l;
        piqp::optional<Eigen::Map<Vec>> x_u;
        if (!mxIsEmpty(c_ptr)) { c = Eigen::Map<Vec>(mxGetPr(c_ptr), n); }
        if (!mxIsEmpty(b_ptr)) { b = Eigen::Map<Vec>(mxGetPr(b_ptr), p); }
        if (!mxIsEmpty(h_l_ptr)) { h_l = Eigen::Map<Vec>(mxGetPr(h_l_ptr), m); }
        if (!mxIsEmpty(h_u_ptr)) { h_u = Eigen::Map<Vec>(mxGetPr(h_u_ptr), m); }
        if (!mxIsEmpty(x_l_ptr)) { x_l = Eigen::Map<Vec>(mxGetPr(x_l_ptr), n); }
        if (!mxIsEmpty(x_u_ptr)) { x_u = Eigen::Map<Vec>(mxGetPr(x_u_ptr), n); }

        if (mex_handle->isDense()) {
            piqp::optional<Eigen::Map<Mat>> P;
            piqp::optional<Eigen::Map<Mat>> A;
            piqp::optional<Eigen::Map<Mat>> G;
            if (!mxIsEmpty(P_ptr)) { P = Eigen::Map<Mat>(mxGetPr(P_ptr), n, n); }
            if (!mxIsEmpty(A_ptr)) { A = Eigen::Map<Mat>(mxGetPr(A_ptr), p, n); }
            if (!mxIsEmpty(G_ptr)) { G = Eigen::Map<Mat>(mxGetPr(G_ptr), m, n); }

            mex_handle->as_dense_ptr()->update(P, c, A, b, G, h_l, h_u, x_l, x_u);
        } else {
            piqp::optional<Eigen::Map<SparseMat>> P;
            IVec Pp;
            IVec Pi;
            if (!mxIsEmpty(P_ptr)) {
                Pp = to_int_vec(mxGetJc(P_ptr), n + 1);
                Pi = to_int_vec(mxGetIr(P_ptr), Pp(n));
                P = Eigen::Map<SparseMat>(n, n, (Eigen::Index) mxGetNzmax(P_ptr), Pp.data(), Pi.data(), mxGetPr(P_ptr));
            }

            piqp::optional<Eigen::Map<SparseMat>> A;
            IVec Ap;
            IVec Ai;
            if (!mxIsEmpty(A_ptr)) {
                Ap = to_int_vec(mxGetJc(A_ptr), n + 1);
                Ai = to_int_vec(mxGetIr(A_ptr), Ap(n));
                A = Eigen::Map<SparseMat>(p, n, (Eigen::Index) mxGetNzmax(A_ptr), Ap.data(), Ai.data(), mxGetPr(A_ptr));
            }

            piqp::optional<Eigen::Map<SparseMat>> G;
            IVec Gp;
            IVec Gi;
            if (!mxIsEmpty(G_ptr)) {
                Gp = to_int_vec(mxGetJc(G_ptr), n + 1);
                Gi = to_int_vec(mxGetIr(G_ptr), Gp(n));
                G = Eigen::Map<SparseMat>(m, n, (Eigen::Index) mxGetNzmax(G_ptr), Gp.data(), Gi.data(), mxGetPr(G_ptr));
            }

            mex_handle->as_sparse_ptr()->update(P, c, A, b, G, h_l, h_u, x_l, x_u);
        }

        return;
    }

    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized.");
}
