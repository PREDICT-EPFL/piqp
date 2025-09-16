// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2017 Bartolomeo Stellato
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <octave/oct.h>
#include "piqp/piqp.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define PIQP_MEX_SIGNATURE 0x271C1A7A

#ifndef PIQP_VERSION
#define PIQP_VERSION 0.6.2
#endif

using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using IVec = Eigen::Matrix<int, Eigen::Dynamic, 1>;
using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

using DenseSolver = piqp::DenseSolver<double>;
using SparseSolver = piqp::SparseSolver<double, int>;

class piqp_oct_handle
{
public:
    explicit piqp_oct_handle(DenseSolver* ptr) : m_signature(PIQP_MEX_SIGNATURE), m_is_dense(true), m_ptr(ptr) {}
    explicit piqp_oct_handle(SparseSolver* ptr) : m_signature(PIQP_MEX_SIGNATURE), m_is_dense(false), m_ptr(ptr) {}
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
inline octave_value create_oct_handle(T* ptr)
{
    return octave_value(octave_uint64(reinterpret_cast<uint64_t>(new piqp_oct_handle(ptr))));
}

inline piqp_oct_handle* get_oct_handle(const octave_value& in)
{
    if (!in.is_scalar_type() || !in.is_uint64_type()) {
        error("Input must be a real uint64 scalar.");
    }
    auto *ptr = reinterpret_cast<piqp_oct_handle*>(in.uint64_scalar_value().value());
    if (!ptr->isValid()) {
        error("Handle not valid.");
    }
    return ptr;
}

inline void destroy_oct_handle(const octave_value& in)
{
    delete get_oct_handle(in);
}

inline IVec to_int_vec(octave_idx_type* data, int n)
{
    return Eigen::Map<Eigen::Matrix<octave_idx_type, Eigen::Dynamic, 1>>(data, n).cast<int>();
}

inline octave_value eigen_to_ov(const Vec& vec)
{
    NDArray arr(dim_vector(vec.rows(), 1));
    Eigen::Map<Vec>(arr.fortran_vec(), vec.rows()) = vec;
    return octave_value(arr);
}

piqp::KKTSolver kkt_solver_from_string(const std::string& kkt_solver, bool is_dense)
{
    if (kkt_solver == "dense_cholesky") return piqp::KKTSolver::dense_cholesky;
    if (kkt_solver == "sparse_ldlt") return piqp::KKTSolver::sparse_ldlt;
    if (kkt_solver == "sparse_ldlt_eq_cond") return piqp::KKTSolver::sparse_ldlt_eq_cond;
    if (kkt_solver == "sparse_ldlt_ineq_cond") return piqp::KKTSolver::sparse_ldlt_ineq_cond;
    if (kkt_solver == "sparse_ldlt_cond") return piqp::KKTSolver::sparse_ldlt_cond;
    if (kkt_solver == "sparse_multistage") return piqp::KKTSolver::sparse_multistage;
    if (is_dense) {
        warning("Unknown kkt_solver, using dense_cholesky as a fallback.");
        return piqp::KKTSolver::dense_cholesky;
    }
    warning("Unknown kkt_solver, using sparse_ldlt as a fallback.");
    return piqp::KKTSolver::sparse_ldlt;
}

octave_value settings_to_ov_struct(const piqp::Settings<double>& settings)
{
    octave_scalar_map ov_struct;

    ov_struct.assign("rho_init", octave_value(settings.rho_init));
    ov_struct.assign("delta_init", octave_value(settings.delta_init));
    ov_struct.assign("eps_abs", octave_value(settings.eps_abs));
    ov_struct.assign("eps_rel", octave_value(settings.eps_rel));
    ov_struct.assign("check_duality_gap", octave_value(settings.check_duality_gap));
    ov_struct.assign("eps_duality_gap_abs", octave_value(settings.eps_duality_gap_abs));
    ov_struct.assign("eps_duality_gap_rel", octave_value(settings.eps_duality_gap_rel));
    ov_struct.assign("infeasibility_threshold", octave_value(settings.infeasibility_threshold));
    ov_struct.assign("reg_lower_limit", octave_value(settings.reg_lower_limit));
    ov_struct.assign("reg_finetune_lower_limit", octave_value(settings.reg_finetune_lower_limit));
    ov_struct.assign("reg_finetune_primal_update_threshold", octave_value(settings.reg_finetune_primal_update_threshold));
    ov_struct.assign("reg_finetune_dual_update_threshold", octave_value(settings.reg_finetune_dual_update_threshold));
    ov_struct.assign("max_iter", octave_value(settings.max_iter));
    ov_struct.assign("max_factor_retires", octave_value(settings.max_factor_retires));
    ov_struct.assign("preconditioner_scale_cost", octave_value(settings.preconditioner_scale_cost));
    ov_struct.assign("preconditioner_reuse_on_update", octave_value(settings.preconditioner_reuse_on_update));
    ov_struct.assign("preconditioner_iter", octave_value(settings.preconditioner_iter));
    ov_struct.assign("tau", octave_value(settings.tau));
    ov_struct.assign("kkt_solver", octave_value(piqp::kkt_solver_to_string(settings.kkt_solver)));
    ov_struct.assign("iterative_refinement_always_enabled", octave_value(settings.iterative_refinement_always_enabled));
    ov_struct.assign("iterative_refinement_eps_abs", octave_value(settings.iterative_refinement_eps_abs));
    ov_struct.assign("iterative_refinement_eps_rel", octave_value(settings.iterative_refinement_eps_rel));
    ov_struct.assign("iterative_refinement_max_iter", octave_value(settings.iterative_refinement_max_iter));
    ov_struct.assign("iterative_refinement_min_improvement_rate", octave_value(settings.iterative_refinement_min_improvement_rate));
    ov_struct.assign("iterative_refinement_static_regularization_eps", octave_value(settings.iterative_refinement_static_regularization_eps));
    ov_struct.assign("iterative_refinement_static_regularization_rel", octave_value(settings.iterative_refinement_static_regularization_rel));
    ov_struct.assign("verbose", octave_value(settings.verbose));
    ov_struct.assign("compute_timings", octave_value(settings.compute_timings));

    return octave_value(ov_struct);
}

void copy_ov_struct_to_settings(const octave_scalar_map& ov_struct, piqp::Settings<double>& settings, bool is_dense)
{
    settings.rho_init = ov_struct.getfield("rho_init").double_value();
    settings.delta_init = ov_struct.getfield("delta_init").double_value();
    settings.eps_abs = ov_struct.getfield("eps_abs").double_value();
    settings.eps_rel = ov_struct.getfield("eps_rel").double_value();
    settings.check_duality_gap = ov_struct.getfield("check_duality_gap").bool_value();
    settings.eps_duality_gap_abs = ov_struct.getfield("eps_duality_gap_abs").double_value();
    settings.eps_duality_gap_rel = ov_struct.getfield("eps_duality_gap_rel").double_value();
    settings.infeasibility_threshold = ov_struct.getfield("infeasibility_threshold").double_value();
    settings.reg_lower_limit = ov_struct.getfield("reg_lower_limit").double_value();
    settings.reg_finetune_lower_limit = ov_struct.getfield("reg_finetune_lower_limit").double_value();
    settings.reg_finetune_primal_update_threshold = ov_struct.getfield("check_duality_gap").int_value();
    settings.reg_finetune_dual_update_threshold = ov_struct.getfield("reg_finetune_dual_update_threshold").int_value();
    settings.max_iter = ov_struct.getfield("max_iter").int_value();
    settings.max_factor_retires = ov_struct.getfield("max_factor_retires").int_value();
    settings.preconditioner_scale_cost = ov_struct.getfield("preconditioner_scale_cost").bool_value();
    settings.preconditioner_reuse_on_update = ov_struct.getfield("preconditioner_reuse_on_update").bool_value();
    settings.preconditioner_iter = ov_struct.getfield("preconditioner_iter").int_value();
    settings.tau = ov_struct.getfield("tau").double_value();
    settings.kkt_solver = kkt_solver_from_string(ov_struct.getfield("kkt_solver").string_value(), is_dense);
    settings.iterative_refinement_always_enabled = ov_struct.getfield("iterative_refinement_always_enabled").bool_value();
    settings.iterative_refinement_eps_abs = ov_struct.getfield("iterative_refinement_eps_abs").double_value();
    settings.iterative_refinement_eps_rel = ov_struct.getfield("iterative_refinement_eps_rel").double_value();
    settings.iterative_refinement_max_iter = ov_struct.getfield("iterative_refinement_max_iter").int_value();
    settings.iterative_refinement_min_improvement_rate = ov_struct.getfield("iterative_refinement_min_improvement_rate").double_value();
    settings.iterative_refinement_static_regularization_eps = ov_struct.getfield("iterative_refinement_static_regularization_eps").double_value();
    settings.iterative_refinement_static_regularization_rel = ov_struct.getfield("iterative_refinement_static_regularization_rel").double_value();
    settings.verbose = ov_struct.getfield("verbose").bool_value();
    settings.compute_timings = ov_struct.getfield("compute_timings").bool_value();
}

octave_value result_to_ov_struct(const piqp::Result<double>& result)
{
    octave_scalar_map ov_info_struct;

    ov_info_struct.assign("status", octave_value(piqp::status_to_string(result.info.status)));
    ov_info_struct.assign("status_val", octave_value(result.info.status));
    ov_info_struct.assign("iter", octave_value(result.info.iter));
    ov_info_struct.assign("rho", octave_value(result.info.rho));
    ov_info_struct.assign("delta", octave_value(result.info.delta));
    ov_info_struct.assign("mu", octave_value(result.info.mu));
    ov_info_struct.assign("sigma", octave_value(result.info.sigma));
    ov_info_struct.assign("primal_step", octave_value(result.info.primal_step));
    ov_info_struct.assign("dual_step", octave_value(result.info.dual_step));
    ov_info_struct.assign("primal_res", octave_value(result.info.primal_res));
    ov_info_struct.assign("primal_res_rel", octave_value(result.info.primal_res_rel));
    ov_info_struct.assign("dual_res", octave_value(result.info.dual_res));
    ov_info_struct.assign("dual_res_rel", octave_value(result.info.dual_res_rel));
    ov_info_struct.assign("primal_res_reg", octave_value(result.info.primal_res_reg));
    ov_info_struct.assign("primal_res_reg_rel", octave_value(result.info.primal_res_reg_rel));
    ov_info_struct.assign("dual_res_reg", octave_value(result.info.dual_res_reg));
    ov_info_struct.assign("dual_res_reg_rel", octave_value(result.info.dual_res_reg_rel));
    ov_info_struct.assign("primal_prox_inf", octave_value(result.info.primal_prox_inf));
    ov_info_struct.assign("dual_prox_inf", octave_value(result.info.dual_prox_inf));
    ov_info_struct.assign("prev_primal_res", octave_value(result.info.prev_primal_res));
    ov_info_struct.assign("prev_dual_res", octave_value(result.info.prev_dual_res));
    ov_info_struct.assign("primal_obj", octave_value(result.info.primal_obj));
    ov_info_struct.assign("dual_obj", octave_value(result.info.dual_obj));
    ov_info_struct.assign("duality_gap", octave_value(result.info.duality_gap));
    ov_info_struct.assign("duality_gap_rel", octave_value(result.info.duality_gap_rel));
    ov_info_struct.assign("factor_retires", octave_value(result.info.factor_retires));
    ov_info_struct.assign("reg_limit", octave_value(result.info.reg_limit));
    ov_info_struct.assign("no_primal_update", octave_value(result.info.no_primal_update));
    ov_info_struct.assign("no_dual_update", octave_value(result.info.no_dual_update));
    ov_info_struct.assign("setup_time", octave_value(result.info.setup_time));
    ov_info_struct.assign("update_time", octave_value(result.info.update_time));
    ov_info_struct.assign("solve_time", octave_value(result.info.solve_time));
    ov_info_struct.assign("kkt_factor_time", octave_value(result.info.kkt_factor_time));
    ov_info_struct.assign("kkt_solve_time", octave_value(result.info.kkt_solve_time));
    ov_info_struct.assign("run_time", octave_value(result.info.run_time));

    octave_scalar_map ov_result_struct;

    ov_result_struct.assign("x", eigen_to_ov(result.x));
    ov_result_struct.assign("y", eigen_to_ov(result.y));
    ov_result_struct.assign("z_l", eigen_to_ov(result.z_l));
    ov_result_struct.assign("z_u", eigen_to_ov(result.z_u));
    ov_result_struct.assign("z_bl", eigen_to_ov(result.z_bl));
    ov_result_struct.assign("z_bu", eigen_to_ov(result.z_bu));
    ov_result_struct.assign("s_l", eigen_to_ov(result.s_l));
    ov_result_struct.assign("s_u", eigen_to_ov(result.s_u));
    ov_result_struct.assign("s_bl", eigen_to_ov(result.s_bl));
    ov_result_struct.assign("s_bu", eigen_to_ov(result.s_bu));
    ov_result_struct.assign("info", octave_value(ov_info_struct));

    return octave_value(ov_result_struct);
}

DEFUN_DLD(piqp_oct, args, nargout, "")
{
    if (args.length() < 1 || !args(0).is_string()) {
        error("First input should be a command string.");
    }

    if (args(0).string_value() == "new") {
        std::string backend;
        if (args.length() < 2) {
            backend = "sparse";
            warning("The sparse backend is automatically used. To get rid of this warning or use another backend, "
                    "provide the backend explicitly using piqp('dense') or piqp('sparse').");
        } else if (!args(1).is_string()) {
            error("Second input should be string less than 10 characters long.");
        } else {
            backend = args(1).string_value();
        }

        if (backend == "dense") {
            return create_oct_handle(new DenseSolver());
        } else if (backend == "sparse") {
            return create_oct_handle(new SparseSolver());
        } else {
            error("Second input must be 'dense' or 'sparse'.");
        }
        return {};
    }

    if (args(0).string_value() == "version") {
        return octave_value(MACRO_STRINGIFY(PIQP_VERSION));
    }

    // Check for a second input
    if (args.length() < 2) {
        error("Second input should be a class instance handle.");
    }
    piqp_oct_handle* oct_handle = get_oct_handle(args(1));

    // delete the object and its data
    if (args(0).string_value() == "delete") {
        if (oct_handle->isDense()) {
            if (oct_handle->as_dense_ptr()) {
                delete oct_handle->as_dense_ptr();
            }
        } else {
            if (oct_handle->as_sparse_ptr()) {
                delete oct_handle->as_sparse_ptr();
            }
        }

        //clean up the handle object
        destroy_oct_handle(args(1));
        // Warn if other commands were ignored
        if (nargout != 0 || args.length() != 2) {
            warning("Unexpected arguments ignored.");
        }
        return {};
    }

    // Get settings
    if (args(0).string_value() == "get_settings") {
        if (oct_handle->isDense()) {
            return settings_to_ov_struct(oct_handle->as_dense_ptr()->settings());
        } else {
            return settings_to_ov_struct(oct_handle->as_sparse_ptr()->settings());
        }
        return {};
    }

    // Update settings
    if (args(0).string_value() == "update_settings") {
        if (oct_handle->isDense()) {
            copy_ov_struct_to_settings(args(2).scalar_map_value(), oct_handle->as_dense_ptr()->settings(), oct_handle->isDense());
        } else {
            copy_ov_struct_to_settings(args(2).scalar_map_value(), oct_handle->as_sparse_ptr()->settings(), oct_handle->isDense());
        }
        return {};
    }

    // Get problem dimensions
    if (args(0).string_value() == "get_dimensions") {
        if (oct_handle->isDense()) {
            octave_value_list ret;
            ret.append(octave_value(oct_handle->as_dense_ptr()->result().x.rows()));
            ret.append(octave_value(oct_handle->as_dense_ptr()->result().y.rows()));
            ret.append(octave_value(oct_handle->as_dense_ptr()->result().z_l.rows()));
            return ret;
        } else {
            octave_value_list ret;
            ret.append(octave_value(oct_handle->as_sparse_ptr()->result().x.rows()));
            ret.append(octave_value(oct_handle->as_sparse_ptr()->result().y.rows()));
            ret.append(octave_value(oct_handle->as_sparse_ptr()->result().z_u.rows()));
            return ret;
        }
        return {};
    }

    if (args(0).string_value() == "setup") {
        const int n = args(2).int_value();
        const int p = args(3).int_value();
        const int m = args(4).int_value();

        const octave_value& P_ref = args(5);
        const octave_value& c_ref = args(6);
        const octave_value& A_ref = args(7);
        const octave_value& b_ref = args(8);
        const octave_value& G_ref = args(9);
        const octave_value& h_l_ref = args(10);
        const octave_value& h_u_ref = args(11);
        const octave_value& x_l_ref = args(12);
        const octave_value& x_u_ref = args(13);

        double c_value = c_ref.is_scalar_type() ? c_ref.double_value() : 0;
        double b_value = b_ref.is_scalar_type() ? b_ref.double_value() : 0;
        double h_l_value = h_l_ref.is_scalar_type() ? h_l_ref.double_value() : 0;
        double h_u_value = h_u_ref.is_scalar_type() ? h_u_ref.double_value() : 0;
        double x_l_value = x_l_ref.is_scalar_type() ? x_l_ref.double_value() : 0;
        double x_u_value = x_u_ref.is_scalar_type() ? x_u_ref.double_value() : 0;

        Eigen::Map<const Vec> c(c_ref.is_scalar_type() ? &c_value : c_ref.vector_value().data(), n);
        Eigen::Map<const Vec> b(b_ref.is_scalar_type() ? &b_value : b_ref.vector_value().data(), p);
        Eigen::Map<const Vec> h_l(h_l_ref.is_scalar_type() ? &h_l_value : h_l_ref.vector_value().data(), m);
        Eigen::Map<const Vec> h_u(h_u_ref.is_scalar_type() ? &h_u_value : h_u_ref.vector_value().data(), m);
        Eigen::Map<const Vec> x_l(x_l_ref.is_scalar_type() ? &x_l_value : x_l_ref.vector_value().data(), n);
        Eigen::Map<const Vec> x_u(x_u_ref.is_scalar_type() ? &x_u_value : x_u_ref.vector_value().data(), n);

        if (oct_handle->isDense()) {
            copy_ov_struct_to_settings(args(14).scalar_map_value(), oct_handle->as_dense_ptr()->settings(), oct_handle->isDense());

            double P_value = P_ref.is_scalar_type() ? P_ref.double_value() : 0;
            double A_value = A_ref.is_scalar_type() ? A_ref.double_value() : 0;
            double G_value = G_ref.is_scalar_type() ? G_ref.double_value() : 0;

            Eigen::Map<const Mat> P(P_ref.is_scalar_type() ? &P_value : P_ref.matrix_value().data(), n, n);
            Eigen::Map<const Mat> A(A_ref.is_scalar_type() ? &A_value : A_ref.matrix_value().data(), p, n);
            Eigen::Map<const Mat> G(G_ref.is_scalar_type() ? &G_value : G_ref.matrix_value().data(), m, n);

            oct_handle->as_dense_ptr()->setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
        } else {
            copy_ov_struct_to_settings(args(14).scalar_map_value(), oct_handle->as_sparse_ptr()->settings(), oct_handle->isDense());

            IVec Pp = to_int_vec(P_ref.sparse_matrix_value().xcidx(), n + 1);
            IVec Pi = to_int_vec(P_ref.sparse_matrix_value().xridx(), Pp(n));
            Eigen::Map<SparseMat> P(n, n, (Eigen::Index) P_ref.nnz(), Pp.data(), Pi.data(), P_ref.sparse_matrix_value().xdata());

            IVec Ap = to_int_vec(A_ref.sparse_matrix_value().xcidx(), n + 1);
            IVec Ai = to_int_vec(A_ref.sparse_matrix_value().xridx(), Ap(n));
            Eigen::Map<SparseMat> A(p, n, (Eigen::Index) A_ref.nnz(), Ap.data(), Ai.data(), A_ref.sparse_matrix_value().xdata());

            IVec Gp = to_int_vec(G_ref.sparse_matrix_value().xcidx(), n + 1);
            IVec Gi = to_int_vec(G_ref.sparse_matrix_value().xridx(), Gp(n));
            Eigen::Map<SparseMat> G(m, n, (Eigen::Index) G_ref.nnz(), Gp.data(), Gi.data(), G_ref.sparse_matrix_value().xdata());

            oct_handle->as_sparse_ptr()->setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
        }

        return {};
    }

    if (args(0).string_value() == "solve") {
        if (oct_handle->isDense()) {
            oct_handle->as_dense_ptr()->solve();
            return result_to_ov_struct(oct_handle->as_dense_ptr()->result());
        } else {
            oct_handle->as_sparse_ptr()->solve();
            return result_to_ov_struct(oct_handle->as_sparse_ptr()->result());
        }

        return {};
    }

    if (args(0).string_value() == "update") {
        const int n = args(2).int_value();
        const int p = args(3).int_value();
        const int m = args(4).int_value();

        const octave_value& P_ref = args(5);
        const octave_value& c_ref = args(6);
        const octave_value& A_ref = args(7);
        const octave_value& b_ref = args(8);
        const octave_value& G_ref = args(9);
        const octave_value& h_l_ref = args(10);
        const octave_value& h_u_ref = args(11);
        const octave_value& x_l_ref = args(12);
        const octave_value& x_u_ref = args(13);

        piqp::optional<Eigen::Map<const Vec>> c;
        piqp::optional<Eigen::Map<const Vec>> b;
        piqp::optional<Eigen::Map<const Vec>> h_l;
        piqp::optional<Eigen::Map<const Vec>> h_u;
        piqp::optional<Eigen::Map<const Vec>> x_l;
        piqp::optional<Eigen::Map<const Vec>> x_u;

        double c_value = c_ref.is_scalar_type() ? c_ref.double_value() : 0;
        double b_value = b_ref.is_scalar_type() ? b_ref.double_value() : 0;
        double h_l_value = h_l_ref.is_scalar_type() ? h_l_ref.double_value() : 0;
        double h_u_value = h_u_ref.is_scalar_type() ? h_u_ref.double_value() : 0;
        double x_l_value = x_l_ref.is_scalar_type() ? x_l_ref.double_value() : 0;
        double x_u_value = x_u_ref.is_scalar_type() ? x_u_ref.double_value() : 0;

        if (!c_ref.isempty()) { c.emplace(c_ref.is_scalar_type() ? &c_value : c_ref.vector_value().data(), n); }
        if (!b_ref.isempty()) { b.emplace(b_ref.is_scalar_type() ? &b_value : b_ref.vector_value().data(), p); }
        if (!h_l_ref.isempty()) { h_l.emplace(h_l_ref.is_scalar_type() ? &h_l_value : h_l_ref.vector_value().data(), m); }
        if (!h_u_ref.isempty()) { h_u.emplace(h_u_ref.is_scalar_type() ? &h_u_value : h_u_ref.vector_value().data(), m); }
        if (!x_l_ref.isempty()) { x_l.emplace(x_l_ref.is_scalar_type() ? &x_l_value : x_l_ref.vector_value().data(), n); }
        if (!x_u_ref.isempty()) { x_u.emplace(x_u_ref.is_scalar_type() ? &x_u_value : x_u_ref.vector_value().data(), n); }

        if (oct_handle->isDense()) {
            piqp::optional<Eigen::Map<const Mat>> P;
            piqp::optional<Eigen::Map<const Mat>> A;
            piqp::optional<Eigen::Map<const Mat>> G;

            double P_value = P_ref.is_scalar_type() ? P_ref.double_value() : 0;
            double A_value = A_ref.is_scalar_type() ? A_ref.double_value() : 0;
            double G_value = G_ref.is_scalar_type() ? G_ref.double_value() : 0;

            if (!P_ref.isempty()) { P.emplace(P_ref.is_scalar_type() ? &P_value : P_ref.matrix_value().data(), n, n); }
            if (!A_ref.isempty()) { A.emplace(A_ref.is_scalar_type() ? &A_value : A_ref.matrix_value().data(), p, n); }
            if (!G_ref.isempty()) { G.emplace(G_ref.is_scalar_type() ? &G_value : G_ref.matrix_value().data(), m, n); }

            oct_handle->as_dense_ptr()->update(P, c, A, b, G, h_l, h_u, x_l, x_u);
        } else {
            piqp::optional<Eigen::Map<SparseMat>> P;
            IVec Pp;
            IVec Pi;
            if (!P_ref.isempty()) {
                Pp = to_int_vec(P_ref.sparse_matrix_value().xcidx(), n + 1);
                Pi = to_int_vec(P_ref.sparse_matrix_value().xridx(), Pp(n));
                P = Eigen::Map<SparseMat>(n, n, (Eigen::Index) P_ref.nnz(), Pp.data(), Pi.data(), P_ref.sparse_matrix_value().xdata());
            }

            piqp::optional<Eigen::Map<SparseMat>> A;
            IVec Ap;
            IVec Ai;
            if (!A_ref.isempty()) {
                Ap = to_int_vec(A_ref.sparse_matrix_value().xcidx(), n + 1);
                Ai = to_int_vec(A_ref.sparse_matrix_value().xridx(), Ap(n));
                A = Eigen::Map<SparseMat>(p, n, (Eigen::Index) A_ref.nnz(), Ap.data(), Ai.data(), A_ref.sparse_matrix_value().xdata());
            }

            piqp::optional<Eigen::Map<SparseMat>> G;
            IVec Gp;
            IVec Gi;
            if (!G_ref.isempty()) {
                Gp = to_int_vec(G_ref.sparse_matrix_value().xcidx(), n + 1);
                Gi = to_int_vec(G_ref.sparse_matrix_value().xridx(), Gp(n));
                G = Eigen::Map<SparseMat>(m, n, (Eigen::Index) G_ref.nnz(), Gp.data(), Gi.data(), G_ref.sparse_matrix_value().xdata());
            }

            oct_handle->as_sparse_ptr()->update(P, c, A, b, G, h_l, h_u, x_l, x_u);
        }

        return {};
    }

    // Got here, so command not recognized
    error("Command not recognized.");
}
