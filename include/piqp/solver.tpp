// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SOLVER_TPP
#define PIQP_SOLVER_TPP

#include <cstdio>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/fwd.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/timer.hpp"
#include "piqp/solver.hpp"
#include "piqp/sparse/utils.hpp"
#include "piqp/utils/openmp.hpp"
#include "piqp/utils/smoothing.hpp"
#include "piqp/utils/tracy.hpp"

namespace piqp
{

template<typename T, typename I, typename Preconditioner, int MatrixType>
SolverBase<T, I, Preconditioner, MatrixType>::SolverBase()
{
    if (MatrixType == PIQP_DENSE) {
        m_settings.kkt_solver = KKTSolver::dense_cholesky;
    } else {
        m_settings.kkt_solver = KKTSolver::sparse_ldlt;
    }
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::solve()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::solve");

    if (m_settings.verbose)
    {
        piqp_print("----------------------------------------------------------\n");
        piqp_print("                        PIQP v0.7.0                       \n");
        piqp_print("                    (c) Roland Schwan                     \n");
        piqp_print("   Ecole Polytechnique Federale de Lausanne (EPFL) 2026   \n");
        piqp_print("----------------------------------------------------------\n");
        if (MatrixType == PIQP_DENSE)
        {
            piqp_print("dense backend (%s)\n", kkt_solver_to_string(m_settings.kkt_solver));
            piqp_print("variables n = %zd\n", m_data.n);
            piqp_print("equality constraints p = %zd\n", m_data.p);
            piqp_print("inequality constraints m = %zd\n", m_data.m);
        }
        else
        {
            piqp_print("sparse backend (%s)\n", kkt_solver_to_string(m_settings.kkt_solver));
            piqp_print("variables n = %zd, nzz(P upper triangular) = %zd\n", m_data.n, m_data.non_zeros_P_utri());
            piqp_print("equality constraints p = %zd, nnz(A) = %zd\n", m_data.p, m_data.non_zeros_A());
            piqp_print("inequality constraints m = %zd, nnz(G) = %zd\n", m_data.m, m_data.non_zeros_G());
        }
        piqp_print("inequality lower bounds n_h_l = %zd\n", m_data.n_h_l);
        piqp_print("inequality upper bounds n_h_u = %zd\n", m_data.n_h_u);
        piqp_print("variable lower bounds n_x_l = %zd\n", m_data.n_x_l);
        piqp_print("variable upper bounds n_x_u = %zd\n", m_data.n_x_u);
        m_kkt_system.print_info();
        piqp_print("\n");
        piqp_print("iter  prim_obj       dual_obj       duality_gap   prim_res      dual_res      rho         delta       mu          p_step   d_step\n");
    }

    Timer<T> solve_timer;
    if (m_settings.compute_timings)
    {
        solve_timer.start();
    }

    Status status = solve_impl();

    m_backward_ready = (status == Status::PIQP_SOLVED || status == Status::PIQP_MAX_ITER_REACHED);
    m_backward_factor_valid = false;

    unscale_results(m_result);
    restore_dual();

    if (m_settings.compute_timings)
    {
        T solve_time = solve_timer.stop();
        m_result.info.solve_time = solve_time;
        if (m_first_run) {
            m_result.info.run_time = m_result.info.setup_time + m_result.info.solve_time;
        } else {
            m_result.info.run_time = m_result.info.update_time + m_result.info.solve_time;
        }
    }

    if (m_settings.verbose)
    {
        piqp_print("\n");
        piqp_print("status:               %s\n", status_to_string(status));
        piqp_print("number of iterations: %zd\n", m_result.info.iter);
        piqp_print("objective:            %.5e\n", static_cast<double>(m_result.info.primal_obj));
        if (m_settings.compute_timings)
        {
            piqp_print("total run time:       %.3es\n", static_cast<double>(m_result.info.run_time));
            if (m_first_run) {
                piqp_print("  setup time:         %.3es\n", static_cast<double>(m_result.info.setup_time));
            } else {
                piqp_print("  update time:        %.3es\n", static_cast<double>(m_result.info.update_time));
            }
            piqp_print("  solve time:         %.3es\n", static_cast<double>(m_result.info.solve_time));
            piqp_print("    kkt factor time:  %.3es\n", static_cast<double>(m_result.info.kkt_factor_time));
            piqp_print("    kkt solve time:   %.3es\n", static_cast<double>(m_result.info.kkt_solve_time));
        }
    }

    m_first_run = false;

    return status;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::setup_impl(const CMatRefType& P,
                const CVecRef<T>& c,
                const optional<CMatRefType>& A,
                const optional<CVecRef<T>>& b,
                const optional<CMatRefType>& G,
                const optional<CVecRef<T>>& h_l,
                const optional<CVecRef<T>>& h_u,
                const optional<CVecRef<T>>& x_l,
                const optional<CVecRef<T>>& x_u)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::setup");

    Timer<T> setup_timer;
    if (m_settings.compute_timings)
    {
        setup_timer.start();
    }

    m_data.resize(P.rows(), A.has_value() ? A->rows() : 0, G.has_value() ? G->rows() : 0);

    m_backward_ready = false;
    m_backward_factor_valid = false;
    m_backward_enabled = false;
    m_gradients.status = Status::PIQP_UNSOLVED;

    if (P.rows() != m_data.n || P.cols() != m_data.n) { piqp_eprint("P must be square\n"); return; }
    if (A.has_value() && (A->rows() != m_data.p || A->cols() != m_data.n)) { piqp_eprint("A must have correct dimensions\n"); return; }
    if (G.has_value() && (G->rows() != m_data.m || G->cols() != m_data.n)) { piqp_eprint("G must have correct dimensions\n"); return; }
    if (c.size() != m_data.n) { piqp_eprint("c must have correct dimensions\n"); return; }
    if ((b.has_value() && b->size() != m_data.p) || (!b.has_value() && m_data.p > 0)) { piqp_eprint("b must have correct dimensions\n"); return; }
    if (h_l.has_value() && h_l->size() != m_data.m) { piqp_eprint("h_l must have correct dimensions\n"); return; }
    if (h_u.has_value() && h_u->size() != m_data.m) { piqp_eprint("h_u must have correct dimensions\n"); return; }
    if (!h_l.has_value() && !h_u.has_value() && m_data.m > 0) { piqp_eprint("h_l or h_u should be provided\n"); return; }
    if (x_l.has_value() && x_l->size() != m_data.n) { piqp_eprint("x_l must have correct dimensions\n"); return; }
    if (x_u.has_value() && x_u->size() != m_data.n) { piqp_eprint("x_u must have correct dimensions\n"); return; }

    m_data.P_utri = P.template triangularView<Eigen::Upper>();
    if (A.has_value()) { m_data.AT = A->transpose(); }
    if (G.has_value()) { m_data.GT = G->transpose(); }

    m_data.c = c;
    if (b.has_value()) { m_data.b = *b; }
    m_data.set_h_l(h_l);
    m_data.set_h_u(h_u);
    m_data.disable_inf_constraints();
    m_data.set_x_l(x_l);
    m_data.set_x_u(x_u);

    init_workspace();

    m_backward_enabled = m_settings.enable_backward;
    isize n_bwd = m_backward_enabled ? m_data.n : 0;
    isize p_bwd = m_backward_enabled ? m_data.p : 0;
    isize m_bwd = m_backward_enabled ? m_data.m : 0;
    m_linearization_point.resize(n_bwd, p_bwd, m_bwd);
    if (m_backward_enabled)
    {
        m_gradients.resize(m_data, P, A, G);
    }
    else
    {
        m_gradients = GradientsType();
    }

    m_preconditioner.init(m_data);
    m_preconditioner.scale_data(m_data,
                                false,
                                m_settings.preconditioner_scale_cost,
                                m_settings.preconditioner_iter);

    if (!m_kkt_system.init(m_data, m_settings))
    {
        m_setup_done = false;
        return;
    }

    m_first_run = true;
    m_setup_done = true;

    if (m_settings.compute_timings)
    {
        T setup_time = setup_timer.stop();
        m_result.info.setup_time = setup_time;
    }
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::update_impl(const optional<CMatRefType>& P,
                 const optional<CVecRef<T>>& c,
                 const optional<CMatRefType>& A,
                 const optional<CVecRef<T>>& b,
                 const optional<CMatRefType>& G,
                 const optional<CVecRef<T>>& h_l,
                 const optional<CVecRef<T>>& h_u,
                 const optional<CVecRef<T>>& x_l,
                 const optional<CVecRef<T>>& x_u)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::update");

    if (!m_setup_done)
    {
        piqp_eprint("Solver not setup yet\n");
        return;
    }

    // The G rows of disabled constraints are zeroed, i.e., G has to be passed to enable them again.
    if (!G.has_value() && (h_l.has_value() || h_u.has_value()))
    {
        for (isize i = 0; i < m_data.m; i++)
        {
            if (!m_data.G_row_disabled(i)) { continue; }
            bool h_l_finite = h_l.has_value() && h_l->size() == m_data.m && (*h_l)(i) > -PIQP_INF;
            bool h_u_finite = h_u.has_value() && h_u->size() == m_data.m && (*h_u)(i) < PIQP_INF;
            if (h_l_finite || h_u_finite)
            {
                piqp_eprint("h_l[%zd] or h_u[%zd] is finite but both were infinite before, G has to be passed as well to enable the constraint again\n", i, i);
                return;
            }
        }
    }

    Timer<T> update_timer;
    if (m_settings.compute_timings)
    {
        update_timer.start();
    }

    m_backward_ready = false;
    m_backward_factor_valid = false;
    m_gradients.status = Status::PIQP_UNSOLVED;

    m_preconditioner.unscale_data(m_data);

    int update_options = KKTUpdateOptions::KKT_UPDATE_NONE;

    if (P.has_value())
    {
        if (P->rows() != m_data.n || P->cols() != m_data.n) { piqp_eprint("P has wrong dimensions\n"); return; }
        if (!update_P(*P)) { return; }
        update_options |= KKTUpdateOptions::KKT_UPDATE_P;
    }

    if (A.has_value())
    {
        if (A->rows() != m_data.p || A->cols() != m_data.n) { piqp_eprint("A has wrong dimensions\n"); return; }
        if (!update_A(*A)) { return; }
        update_options |= KKTUpdateOptions::KKT_UPDATE_A;
    }

    if (G.has_value())
    {
        if (G->rows() != m_data.m || G->cols() != m_data.n) { piqp_eprint("G has wrong dimensions\n"); return; }
        if (!update_G(*G)) { return; }
        update_options |= KKTUpdateOptions::KKT_UPDATE_G;
    }

    if (c.has_value())
    {
        if (c->size() != m_data.n) { piqp_eprint("c has wrong dimensions\n"); return; }
        m_data.c = *c;
    }

    if (b.has_value())
    {
        if (b->size() != m_data.p) { piqp_eprint("b has wrong dimensions\n"); return; }
        m_data.b = *b;
    }

    if (h_l.has_value() && h_l->size() != m_data.m) { piqp_eprint("h_l has wrong dimensions\n"); return; }
    if (h_u.has_value() && h_u->size() != m_data.m) { piqp_eprint("h_u has wrong dimensions\n"); return; }
    if (G.has_value() || h_l.has_value() || h_u.has_value())
    {
        // Disabled rows hold the replacement bounds -1 <= 0 x <= 1. Restore their infinite bounds, such that
        // bounds which are not updated stay infinite, and disable them again, which also zeros updated G rows.
        for (isize i = 0; i < m_data.m; i++)
        {
            if (m_data.G_row_disabled(i))
            {
                m_data.h_l(i) = -PIQP_INF;
                m_data.h_u(i) = PIQP_INF;
            }
        }
        m_data.set_h_l(h_l.has_value() ? h_l : optional<CVecRef<T>>(m_data.h_l));
        m_data.set_h_u(h_u.has_value() ? h_u : optional<CVecRef<T>>(m_data.h_u));
        m_data.disable_inf_constraints();
    }

    if (x_l.has_value() && x_l->size() != m_data.n) { piqp_eprint("x_l has wrong dimensions\n"); return; }
    if (x_u.has_value() && x_u->size() != m_data.n) { piqp_eprint("x_u has wrong dimensions\n"); return; }
    if (x_l.has_value()) { m_data.set_x_l(x_l); }
    if (x_u.has_value()) { m_data.set_x_u(x_u); }

    bool reuse_preconditioner = m_settings.preconditioner_reuse_on_update;
    if (update_options == KKTUpdateOptions::KKT_UPDATE_NONE)
    {
        reuse_preconditioner = true;
    }

    m_preconditioner.scale_data(m_data,
                                reuse_preconditioner,
                                m_settings.preconditioner_scale_cost,
                                m_settings.preconditioner_iter);

    if (!reuse_preconditioner)
    {
        // Rescaling can change all matrices, including those omitted from the update.
        update_options = KKTUpdateOptions::KKT_UPDATE_P | KKTUpdateOptions::KKT_UPDATE_A | KKTUpdateOptions::KKT_UPDATE_G;
    }

    m_kkt_system.update_data(m_data, update_options);

    if (m_settings.compute_timings)
    {
        T update_time = update_timer.stop();
        m_result.info.update_time = update_time;
    }
}

namespace detail
{
    // Tag dispatch helpers for update_P
    template<typename T, typename I>
    bool update_P_impl(dense::Data<T>& data, const CMatRef<T>& P, std::integral_constant<int, PIQP_DENSE>)
    {
        data.P_utri = P.template triangularView<Eigen::Upper>();
        return true;
    }

    template<typename T, typename I>
    bool update_P_impl(sparse::Data<T, I>& data, const CSparseMatRef<T, I>& P, std::integral_constant<int, PIQP_SPARSE>)
    {
        isize n = P.outerSize();
        for (isize j = 0; j < n; j++)
        {
            isize P_col_nnz = P.outerIndexPtr()[j + 1] - P.outerIndexPtr()[j];
            isize P_utri_col_nnz = data.P_utri.outerIndexPtr()[j + 1] - data.P_utri.outerIndexPtr()[j];
            if (P_col_nnz < P_utri_col_nnz) { piqp_eprint("P nonzeros missmatch\n"); return false; }
            Eigen::Map<Vec<T>>(data.P_utri.valuePtr() + data.P_utri.outerIndexPtr()[j], P_utri_col_nnz) =
                Eigen::Map<const Vec<T>>(P.valuePtr() + P.outerIndexPtr()[j], P_utri_col_nnz);
        }
        return true;
    }

    // Tag dispatch helpers for update_A
    template<typename T, typename I>
    bool update_A_impl(dense::Data<T>& data, const CMatRef<T>& A, std::integral_constant<int, PIQP_DENSE>)
    {
        data.AT = A.transpose();
        return true;
    }

    template<typename T, typename I>
    bool update_A_impl(sparse::Data<T, I>& data, const CSparseMatRef<T, I>& A, std::integral_constant<int, PIQP_SPARSE>)
    {
        if (A.nonZeros() != data.AT.nonZeros()) { piqp_eprint("A nonzeros missmatch\n"); return false; }
        sparse::transpose_no_allocation(A, data.AT);
        return true;
    }

    // Tag dispatch helpers for update_G
    template<typename T, typename I>
    bool update_G_impl(dense::Data<T>& data, const CMatRef<T>& G, std::integral_constant<int, PIQP_DENSE>)
    {
        data.GT = G.transpose();
        return true;
    }

    template<typename T, typename I>
    bool update_G_impl(sparse::Data<T, I>& data, const CSparseMatRef<T, I>& G, std::integral_constant<int, PIQP_SPARSE>)
    {
        if (G.nonZeros() != data.GT.nonZeros()) { piqp_eprint("G nonzeros missmatch\n"); return false; }
        sparse::transpose_no_allocation(G, data.GT);
        return true;
    }
} // namespace detail

template<typename T, typename I, typename Preconditioner, int MatrixType>
bool SolverBase<T, I, Preconditioner, MatrixType>::update_P(const CMatRefType& P)
{
    return detail::update_P_impl<T, I>(m_data, P, std::integral_constant<int, MatrixType>{});
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
bool SolverBase<T, I, Preconditioner, MatrixType>::update_A(const CMatRefType& A)
{
    return detail::update_A_impl<T, I>(m_data, A, std::integral_constant<int, MatrixType>{});
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
bool SolverBase<T, I, Preconditioner, MatrixType>::update_G(const CMatRefType& G)
{
    return detail::update_G_impl<T, I>(m_data, G, std::integral_constant<int, MatrixType>{});
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::init_workspace()
{
    m_result.resize(m_data.n, m_data.p, m_data.m);
    m_result.info.rho = m_settings.rho_init;
    m_result.info.delta = m_settings.delta_init;
    m_result.info.setup_time = 0;
    m_result.info.update_time = 0;
    m_result.info.solve_time = 0;
    m_result.info.kkt_factor_time = 0;
    m_result.info.kkt_solve_time = 0;
    m_result.info.run_time = 0;

    res_nr.resize(m_data.n, m_data.p, m_data.m);
    res.resize(m_data.n, m_data.p, m_data.m);
    step.resize(m_data.n, m_data.p, m_data.m);
    prox_vars.resize(m_data.n, m_data.p, m_data.m);
    delta_vars.resize(m_data.n, m_data.p, m_data.m);
    cert_work_n_1.resize(m_data.n);
    cert_work_n_2.resize(m_data.n);
    cert_work_p.resize(m_data.p);
    cert_work_m_1.resize(m_data.m);
    cert_work_m_2.resize(m_data.m);
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::solve_impl()
{
    if (!m_setup_done)
    {
        piqp_eprint("Solver not setup yet\n");
        m_result.info.status = Status::PIQP_UNSOLVED;
        return m_result.info.status;
    }

    if (!m_settings.verify_settings())
    {
        m_result.info.status = Status::PIQP_INVALID_SETTINGS;
        return m_result.info.status;
    }

    Timer<T> timer;
    m_result.info.kkt_factor_time = 0;
    m_result.info.kkt_solve_time = 0;

    m_result.info.status = Status::PIQP_UNSOLVED;
    m_result.info.iter = 0;
    m_result.info.reg_limit = m_settings.reg_lower_limit;
    m_result.info.factor_retires = 0;
    m_result.info.no_primal_update = 0;
    m_result.info.no_dual_update = 0;
    m_result.info.mu = 0;
    m_result.info.primal_step = 0;
    m_result.info.dual_step = 0;
    m_result.info.rho = m_settings.rho_init;
    m_result.info.delta = m_settings.delta_init;

    // Infinite bounds have never active constraints.
    // Note that we set for all inactive constraints
    // z = 0 as well as s = 0. To be correct, we should
    // have s = Inf. But keeping it zero, simplifies calculations
    // down the road (e.g. norm calculations).
    // We are correcting this at the end when we restore the solution.
    m_result.s_l.setZero();
    m_result.s_u.setZero();
    m_result.z_l.setZero();
    m_result.z_u.setZero();
    for (isize i = 0; i < m_data.n_h_l; i++)
    {
        Eigen::Index idx = m_data.h_l_idx(i);
        m_result.s_l(idx) = T(1);
        m_result.z_l(idx) = T(1);
    }
    for (isize i = 0; i < m_data.n_h_u; i++)
    {
        Eigen::Index idx = m_data.h_u_idx(i);
        m_result.s_u(idx) = T(1);
        m_result.z_u(idx) = T(1);
    }

    // all finite bounds are stored in the head
    m_result.s_bl.head(m_data.n_x_l).setConstant(T(1));
    m_result.s_bu.head(m_data.n_x_u).setConstant(T(1));
    m_result.z_bl.head(m_data.n_x_l).setConstant(T(1));
    m_result.z_bu.head(m_data.n_x_u).setConstant(T(1));

    m_enable_iterative_refinement = m_settings.iterative_refinement_always_enabled;


    if (!factor_kkt(m_result, m_result.info))
    {
        m_result.info.status = Status::PIQP_NUMERICS;
        return m_result.info.status;
    }

    res.x = -m_data.c;
    res.y = m_data.b;
    res.z_l = -m_data.h_l;
    res.z_u = m_data.h_u;
    res.z_bl = -m_data.x_l;
    res.z_bu = m_data.x_u;
    res.s_l.setZero();
    res.s_u.setZero();
    res.s_bl.setZero();
    res.s_bu.setZero();

    if (m_settings.compute_timings) {
        timer.start();
    }
    m_kkt_system.solve(m_data, m_settings, res, m_result);
    if (m_settings.compute_timings)
    {
        T kkt_solve_time = timer.stop();
        m_result.info.kkt_solve_time += kkt_solve_time;
    }

    // We make an Eigen expression for convenience. Note that we are doing it after
    // the first solve since m_kkt_system.solve might swap internal pointers in m_result
    // which can invalidate the reference in the Eigen expression.
    auto s_bl = m_result.s_bl.head(m_data.n_x_l);
    auto s_bu = m_result.s_bu.head(m_data.n_x_u);
    auto z_bl = m_result.z_bl.head(m_data.n_x_l);
    auto z_bu = m_result.z_bu.head(m_data.n_x_u);
    auto nu_bl = prox_vars.z_bl.head(m_data.n_x_l);
    auto nu_bu = prox_vars.z_bu.head(m_data.n_x_u);

    if (m_data.m + m_data.n_x_l + m_data.n_x_u > 0)
    {
        T delta_s = T(0);
        if (m_data.m > 0) {
            delta_s = (std::max)(delta_s, -m_result.s_l.minCoeff());
            delta_s = (std::max)(delta_s, -m_result.s_u.minCoeff());
        }
        if (m_data.n_x_l > 0) delta_s = (std::max)(delta_s, -s_bl.minCoeff());
        if (m_data.n_x_u > 0) delta_s = (std::max)(delta_s, -s_bu.minCoeff());
        T delta_z = T(0);
        if (m_data.m > 0) {
            delta_z = (std::max)(delta_z, -m_result.z_l.minCoeff());
            delta_z = (std::max)(delta_z, -m_result.z_u.minCoeff());
        }
        if (m_data.n_x_l > 0) delta_z = (std::max)(delta_z, -z_bl.minCoeff());
        if (m_data.n_x_u > 0) delta_z = (std::max)(delta_z, -z_bu.minCoeff());

        for (isize i = 0; i < m_data.n_h_l; i++)
        {
            Eigen::Index idx = m_data.h_l_idx(i);
            m_result.s_l(idx) += delta_s;
            m_result.z_l(idx) += delta_z;
        }
        for (isize i = 0; i < m_data.n_h_u; i++)
        {
            Eigen::Index idx = m_data.h_u_idx(i);
            m_result.s_u(idx) += delta_s;
            m_result.z_u(idx) += delta_z;
        }
        s_bl.array() += delta_s;
        s_bu.array() += delta_s;
        z_bl.array() += delta_z;
        z_bu.array() += delta_z;

        m_result.info.mu = (std::max)(calculate_mu(), T(1e-10));

        for (isize i = 0; i < m_data.n_h_l; i++)
        {
            Eigen::Index idx = m_data.h_l_idx(i);

            T c = m_result.z_l(idx) - delta_z;
            m_result.z_l(idx) = (c + std::sqrt(c * c + 4 * m_result.info.mu)) / 2;
            m_result.s_l(idx) = m_result.z_l(idx) - c;
        }
        for (isize i = 0; i < m_data.n_h_u; i++)
        {
            Eigen::Index idx = m_data.h_u_idx(i);

            T c = m_result.z_u(idx) - delta_z;
            m_result.z_u(idx) = (c + std::sqrt(c * c + 4 * m_result.info.mu)) / 2;
            m_result.s_u(idx) = m_result.z_u(idx) - c;
        }
        for (isize i = 0; i < m_data.n_x_l; i++)
        {
            T c = m_result.z_bl(i) - delta_z;
            m_result.z_bl(i) = (c + std::sqrt(c * c + 4 * m_result.info.mu)) / 2;
            m_result.s_bl(i) = m_result.z_bl(i) - c;
        }
        for (isize i = 0; i < m_data.n_x_u; i++)
        {
            T c = m_result.z_bu(i) - delta_z;
            m_result.z_bu(i) = (c + std::sqrt(c * c + 4 * m_result.info.mu)) / 2;
            m_result.s_bu(i) = m_result.z_bu(i) - c;
        }

        m_result.info.mu = calculate_mu();
    }

    prox_vars.x = m_result.x;
    prox_vars.y = m_result.y;
    prox_vars.z_l = m_result.z_l;
    prox_vars.z_u = m_result.z_u;
    nu_bl = z_bl;
    nu_bu = z_bu;

    while (m_result.info.iter < m_settings.max_iter)
    {
        if (m_result.info.iter == 0)
        {
            update_residuals_nr(m_result, m_result.info);
            prox_c_x = cert_c_x;
            prox_dual_lin = cert_dual_lin;
            m_result.info.prev_primal_res = m_result.info.primal_res;
            m_result.info.prev_dual_res = m_result.info.dual_res;
        }

        if (m_settings.verbose)
        {
            piqp_print("%3zd   % .5e   % .5e   %.5e   %.5e   %.5e   %.3e   %.3e   %.3e   %.4f   %.4f\n",
                m_result.info.iter,
                static_cast<double>(m_result.info.primal_obj),
                static_cast<double>(m_result.info.dual_obj),
                static_cast<double>(m_result.info.duality_gap),
                static_cast<double>(m_result.info.primal_res),
                static_cast<double>(m_result.info.dual_res),
                static_cast<double>(m_result.info.rho),
                static_cast<double>(m_result.info.delta),
                static_cast<double>(m_result.info.mu),
                static_cast<double>(m_result.info.primal_step),
                static_cast<double>(m_result.info.dual_step)
            );
            piqp_flush();
        }

        if ((m_result.info.primal_res < m_settings.eps_abs || m_result.info.primal_res_rel < m_settings.eps_rel) &&
            (m_result.info.dual_res < m_settings.eps_abs || m_result.info.dual_res_rel < m_settings.eps_rel) &&
            (!m_settings.check_duality_gap || m_result.info.duality_gap < m_settings.eps_duality_gap_abs || m_result.info.duality_gap_rel < m_settings.eps_duality_gap_rel))
        {
            m_result.info.status = Status::PIQP_SOLVED;
            return m_result.info.status;
        }

        update_residuals_r();

        if (m_result.info.iter > 0)
        {
            Status status = check_infeasibility_certificates();
            if (status != Status::PIQP_UNSOLVED)
            {
                m_result.info.status = status;
                return m_result.info.status;
            }

            // If the regularized subproblem is solved, but the non-regularized residuals
            // stall, we still move the proximal centers. This keeps the outer proximal
            // point iteration going, such that the proximal steps converge to an
            // infeasibility certificate.
            bool prox_moved = false;
            if (m_result.info.primal_res_reg < T(0.1) * m_result.info.primal_res)
            {
                update_dual_prox_center();
                prox_moved = true;
            }
            if (m_result.info.dual_res_reg < T(0.1) * m_result.info.dual_res)
            {
                update_primal_prox_center();
                prox_moved = true;
            }
            if (prox_moved)
            {
                update_residuals_r();
            }
        }

        m_result.info.iter++;

        // avoid getting to close to boundary which can result in a division by zero
        bool boundary_shifted = false;
        T epsilon = std::numeric_limits<T>::epsilon();
        for (isize i = 0; i < m_data.n_h_l; i++)
        {
            Eigen::Index idx = m_data.h_l_idx(i);
            if (m_result.z_l(idx) < epsilon) {
                m_result.z_l(idx) += epsilon;
                boundary_shifted = true;
            }
        }
        for (isize i = 0; i < m_data.n_h_u; i++)
        {
            Eigen::Index idx = m_data.h_u_idx(i);
            if (m_result.z_u(idx) < epsilon) {
                m_result.z_u(idx) += epsilon;
                boundary_shifted = true;
            }
        }
        if (m_data.n_x_l > 0 && z_bl.minCoeff() < epsilon)
        {
            z_bl.array() += epsilon;
            boundary_shifted = true;
        }
        if (m_data.n_x_u > 0 && z_bu.minCoeff() < epsilon)
        {
            z_bu.array() += epsilon;
            boundary_shifted = true;
        }
        if (boundary_shifted)
        {
            m_result.info.mu = calculate_mu();
        }

        // avoid possibility of converging to a local minimum -> decrease the minimum regularization value
        if ((m_result.info.no_primal_update > m_settings.reg_finetune_primal_update_threshold &&
             m_result.info.rho == m_result.info.reg_limit &&
             m_result.info.reg_limit != m_settings.reg_finetune_lower_limit) ||
            (m_result.info.no_dual_update > m_settings.reg_finetune_dual_update_threshold &&
             m_result.info.delta == m_result.info.reg_limit &&
             m_result.info.reg_limit != m_settings.reg_finetune_lower_limit))
        {
            if (m_result.info.dual_prox_inf < m_settings.infeasibility_threshold && m_result.info.primal_prox_inf < m_settings.infeasibility_threshold) {
                m_result.info.reg_limit = m_settings.reg_finetune_lower_limit;
                m_result.info.no_primal_update = 0;
                m_result.info.no_dual_update = 0;
            }
        }

        bool regularization_changed = false;
        if (!factor_kkt(m_result, m_result.info, &regularization_changed))
        {
            m_result.info.status = Status::PIQP_NUMERICS;
            return m_result.info.status;
        }

        if (regularization_changed) {
            update_residuals_r();
        }

        if (m_data.m + m_data.n_x_l + m_data.n_x_u > 0)
        {
            // ------------------ predictor step ------------------
            res.s_l.array() = -m_result.s_l.array() * m_result.z_l.array();
            res.s_u.array() = -m_result.s_u.array() * m_result.z_u.array();
            res.s_bl.head(m_data.n_x_l).array() = -s_bl.array() * z_bl.array();
            res.s_bu.head(m_data.n_x_u).array() = -s_bu.array() * z_bu.array();

            if (m_settings.compute_timings)
            {
                timer.start();
            }
            m_kkt_system.solve(m_data, m_settings, res, step);
            if (m_settings.compute_timings)
            {
                T kkt_solve_time = timer.stop();
                m_result.info.kkt_solve_time += kkt_solve_time;
            }

            // step in the non-negative orthant
            T alpha_s, alpha_z;
            calculate_step(alpha_s, alpha_z, m_result);

            // avoid getting to close to the boundary
            alpha_s *= m_settings.tau;
            alpha_z *= m_settings.tau;

            m_result.info.sigma = (m_result.s_l + alpha_s * step.s_l).dot(m_result.z_l + alpha_z * step.z_l);
            m_result.info.sigma += (m_result.s_u + alpha_s * step.s_u).dot(m_result.z_u + alpha_z * step.z_u);
            m_result.info.sigma += (s_bl + alpha_s * step.s_bl.head(m_data.n_x_l)).dot(z_bl + alpha_z * step.z_bl.head(m_data.n_x_l));
            m_result.info.sigma += (s_bu + alpha_s * step.s_bu.head(m_data.n_x_u)).dot(z_bu + alpha_z * step.z_bu.head(m_data.n_x_u));
            m_result.info.sigma /= (m_result.info.mu * T(m_data.n_h_l + m_data.n_h_u + m_data.n_x_l + m_data.n_x_u));
            m_result.info.sigma = (std::max)(T(0), (std::min)(T(1), m_result.info.sigma));
            m_result.info.sigma = m_result.info.sigma * m_result.info.sigma * m_result.info.sigma;

            // ------------------ corrector step ------------------
            res.s_l.array() += -step.s_l.array() * step.z_l.array() + m_result.info.sigma * m_result.info.mu;
            res.s_u.array() += -step.s_u.array() * step.z_u.array() + m_result.info.sigma * m_result.info.mu;
            res.s_bl.head(m_data.n_x_l).array() += -step.s_bl.head(m_data.n_x_l).array() * step.z_bl.head(m_data.n_x_l).array() + m_result.info.sigma * m_result.info.mu;
            res.s_bu.head(m_data.n_x_u).array() += -step.s_bu.head(m_data.n_x_u).array() * step.z_bu.head(m_data.n_x_u).array() + m_result.info.sigma * m_result.info.mu;

            if (m_settings.compute_timings)
            {
                timer.start();
            }
            m_kkt_system.solve(m_data, m_settings, res, step);
            {
                T kkt_solve_time = timer.stop();
                m_result.info.kkt_solve_time += kkt_solve_time;
            }

            // step in the non-negative orthant
            calculate_step(alpha_s, alpha_z, m_result);

            // avoid getting to close to the boundary
            m_result.info.primal_step = alpha_s * m_settings.tau;
            m_result.info.dual_step = alpha_z * m_settings.tau;

            // ------------------ update ------------------
            m_result.x += m_result.info.primal_step * step.x;
            m_result.y += m_result.info.dual_step * step.y;
            m_result.z_l += m_result.info.dual_step * step.z_l;
            m_result.z_u += m_result.info.dual_step * step.z_u;
            z_bl += m_result.info.dual_step * step.z_bl.head(m_data.n_x_l);
            z_bu += m_result.info.dual_step * step.z_bu.head(m_data.n_x_u);
            m_result.s_l += m_result.info.primal_step * step.s_l;
            m_result.s_u += m_result.info.primal_step * step.s_u;
            s_bl += m_result.info.primal_step * step.s_bl.head(m_data.n_x_l);
            s_bu += m_result.info.primal_step * step.s_bu.head(m_data.n_x_u);

            // displacement from the proximal centers, i.e., the proximal step,
            // used as candidate for the infeasibility certificates
            delta_vars.x.noalias() = m_result.x - prox_vars.x;
            delta_vars.y.noalias() = m_result.y - prox_vars.y;
            delta_vars.z_l.noalias() = m_result.z_l - prox_vars.z_l;
            delta_vars.z_u.noalias() = m_result.z_u - prox_vars.z_u;
            delta_vars.z_bl.head(m_data.n_x_l).noalias() = z_bl - nu_bl;
            delta_vars.z_bu.head(m_data.n_x_u).noalias() = z_bu - nu_bu;

            T mu_prev = m_result.info.mu;
            m_result.info.mu = calculate_mu();
            T mu_rate = (std::max)(T(0), (mu_prev - m_result.info.mu) / mu_prev);

            // ------------------ update regularization ------------------
            update_residuals_nr(m_result, m_result.info);
            delta_c_x = cert_c_x - prox_c_x;
            delta_dual_lin = cert_dual_lin - prox_dual_lin;

            if (m_result.info.dual_res < 0.95 * m_result.info.prev_dual_res ||
                (m_result.info.dual_res < m_settings.eps_abs || m_result.info.dual_res_rel < m_settings.eps_rel) ||
                m_result.info.rho == m_settings.reg_finetune_lower_limit)
            {
                update_primal_prox_center();
                m_result.info.rho = (std::max)(m_result.info.reg_limit, (T(1) - mu_rate) * m_result.info.rho);
            }
            else
            {
                m_result.info.no_primal_update++;
                if (m_result.info.iter < 5 || m_result.info.dual_prox_inf < m_settings.infeasibility_threshold) {
                    m_result.info.rho = (std::max)(m_result.info.reg_limit, (T(1) - T(0.666) * mu_rate) * m_result.info.rho);
                }
            }

            if (m_result.info.primal_res < 0.95 * m_result.info.prev_primal_res ||
                (m_result.info.primal_res < m_settings.eps_abs || m_result.info.primal_res_rel < m_settings.eps_rel) ||
                m_result.info.delta == m_settings.reg_finetune_lower_limit)
            {
                update_dual_prox_center();
                m_result.info.delta = (std::max)(m_result.info.reg_limit, (T(1) - mu_rate) * m_result.info.delta);
            }
            else
            {
                m_result.info.no_dual_update++;
                if (m_result.info.iter < 5 || m_result.info.primal_prox_inf < m_settings.infeasibility_threshold) {
                    m_result.info.delta = (std::max)(m_result.info.reg_limit, (T(1) - T(0.666) * mu_rate) * m_result.info.delta);
                }
            }
        }
        else
        {
            if (m_settings.compute_timings)
            {
                timer.start();
            }
            // since there are no inequalities we can take full steps
            m_kkt_system.solve(m_data, m_settings, res, step);
            {
                T kkt_solve_time = timer.stop();
                m_result.info.kkt_solve_time += kkt_solve_time;
            }

            m_result.info.primal_step = T(1);
            m_result.info.dual_step = T(1);
            m_result.x += m_result.info.primal_step * step.x;
            m_result.y += m_result.info.dual_step * step.y;

            delta_vars.x.noalias() = m_result.x - prox_vars.x;
            delta_vars.y.noalias() = m_result.y - prox_vars.y;

            // ------------------ update regularization ------------------
            update_residuals_nr(m_result, m_result.info);
            delta_c_x = cert_c_x - prox_c_x;
            delta_dual_lin = cert_dual_lin - prox_dual_lin;

            if (m_result.info.dual_res < 0.95 * m_result.info.prev_dual_res || (m_result.info.dual_res < m_settings.eps_abs || m_result.info.dual_res_rel < m_settings.eps_rel))
            {
                update_primal_prox_center();
                m_result.info.rho = (std::max)(m_result.info.reg_limit, T(0.1) * m_result.info.rho);
            }
            else
            {
                m_result.info.no_primal_update++;
                if (m_result.info.iter < 5 || m_result.info.dual_prox_inf < m_settings.infeasibility_threshold) {
                    m_result.info.rho = (std::max)(m_result.info.reg_limit, T(0.5) * m_result.info.rho);
                }
            }

            if (m_result.info.primal_res < 0.95 * m_result.info.prev_primal_res || (m_result.info.primal_res < m_settings.eps_abs || m_result.info.primal_res_rel < m_settings.eps_rel))
            {
                update_dual_prox_center();
                m_result.info.delta = (std::max)(m_result.info.reg_limit, T(0.1) * m_result.info.delta);
            }
            else
            {
                m_result.info.no_dual_update++;
                if (m_result.info.iter < 5 || m_result.info.primal_prox_inf < m_settings.infeasibility_threshold) {
                    m_result.info.delta = (std::max)(m_result.info.reg_limit, T(0.5) * m_result.info.delta);
                }
            }
        }
    }

    m_result.info.status = Status::PIQP_MAX_ITER_REACHED;
    return m_result.info.status;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
bool SolverBase<T, I, Preconditioner, MatrixType>::factor_kkt(const Variables<T>& vars, Info<T>& info, bool* regularization_changed)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::factor_kkt");

    Timer<T> timer;
    if (m_settings.compute_timings)
    {
        timer.start();
    }
    while (!m_kkt_system.update_scalings_and_factor(m_data, m_settings, m_enable_iterative_refinement,
                                                    info.rho, info.delta, vars))
    {
        if (!m_enable_iterative_refinement)
        {
            m_enable_iterative_refinement = true;
        }
        else if (info.factor_retires < m_settings.max_factor_retires)
        {
            info.delta *= 100;
            info.rho *= 100;
            info.factor_retires++;
            info.reg_limit = (std::min)(10 * info.reg_limit, m_settings.eps_abs);
            if (regularization_changed != nullptr)
            {
                *regularization_changed = true;
            }
        }
        else
        {
            return false;
        }
    }
    info.factor_retires = 0;
    if (m_settings.compute_timings)
    {
        T kkt_factor_time = timer.stop();
        info.kkt_factor_time += kkt_factor_time;
    }
    return true;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
T SolverBase<T, I, Preconditioner, MatrixType>::calculate_mu()
{
    return (m_result.s_l.dot(m_result.z_l)
            + m_result.s_u.dot(m_result.z_u)
            + m_result.s_bl.head(m_data.n_x_l).dot(m_result.z_bl.head(m_data.n_x_l))
            + m_result.s_bu.head(m_data.n_x_u).dot(m_result.z_bu.head(m_data.n_x_u)))
        / T(m_data.n_h_l + m_data.n_h_u + m_data.n_x_l + m_data.n_x_u);
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::calculate_step(T& alpha_s, T& alpha_z, const Variables<T>& vars)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::calculate_step");

    alpha_s = T(1);
    alpha_z = T(1);

    auto step_ineq = [&](isize i, T& a_s, T& a_z) {
        if (step.s_l(i) < 0)
        {
            a_s = (std::min)(a_s, -vars.s_l(i) / step.s_l(i));
        }
        if (step.s_u(i) < 0)
        {
            a_s = (std::min)(a_s, -vars.s_u(i) / step.s_u(i));
        }
        if (step.z_l(i) < 0)
        {
            a_z = (std::min)(a_z, -vars.z_l(i) / step.z_l(i));
        }
        if (step.z_u(i) < 0)
        {
            a_z = (std::min)(a_z, -vars.z_u(i) / step.z_u(i));
        }
    };
    auto step_box_l = [&](isize i, T& a_s, T& a_z) {
        if (step.s_bl(i) < 0)
        {
            a_s = (std::min)(a_s, -vars.s_bl(i) / step.s_bl(i));
        }
        if (step.z_bl(i) < 0)
        {
            a_z = (std::min)(a_z, -vars.z_bl(i) / step.z_bl(i));
        }
    };
    auto step_box_u = [&](isize i, T& a_s, T& a_z) {
        if (step.s_bu(i) < 0)
        {
            a_s = (std::min)(a_s, -vars.s_bu(i) / step.s_bu(i));
        }
        if (step.z_bu(i) < 0)
        {
            a_z = (std::min)(a_z, -vars.z_bu(i) / step.z_bu(i));
        }
    };

#ifdef PIQP_HAS_OPENMP
    // opening a parallel region is only worth it for large problems,
    // and a single threaded parallel region is slower than the plain loops
    const int num_threads = resolve_elementwise_num_threads(m_settings.num_threads, m_data.m + m_data.n_x_l + m_data.n_x_u);
    if (num_threads > 1)
    {
        #pragma omp parallel num_threads(num_threads)
        {
            PIQP_TRACY_ZoneScopedN("piqp::Solver::calculate_step:parallel");

            #pragma omp for reduction(min:alpha_s,alpha_z)
            for (isize i = 0; i < m_data.m; i++) step_ineq(i, alpha_s, alpha_z);
            #pragma omp for reduction(min:alpha_s,alpha_z)
            for (isize i = 0; i < m_data.n_x_l; i++) step_box_l(i, alpha_s, alpha_z);
            #pragma omp for reduction(min:alpha_s,alpha_z)
            for (isize i = 0; i < m_data.n_x_u; i++) step_box_u(i, alpha_s, alpha_z);
        }
        return;
    }
#endif

    for (isize i = 0; i < m_data.m; i++) step_ineq(i, alpha_s, alpha_z);
    for (isize i = 0; i < m_data.n_x_l; i++) step_box_l(i, alpha_s, alpha_z);
    for (isize i = 0; i < m_data.n_x_u; i++) step_box_u(i, alpha_s, alpha_z);
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::update_residuals_nr(const Variables<T>& vars, Info<T>& info)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::update_residuals_nr");

    using std::abs;

    // step is not used and we can use it as temporary storage
    Vec<T>& work_x = step.x;
    Vec<T>& work_z = step.z_l;

    // we calculate these term here first to be able to reuse temporary vectors
    // res_nr.y = -A * x
    // work_x = A^T * y
    m_kkt_system.eval_A_xn_and_AT_xt(m_data, T(-1), T(1), vars.x, vars.y, res_nr.y, work_x);
    // res_nr.z_u = -G * x
    // res_nr.z_l = G * x
    // work_x += G^T * (z_u - z_l)
    work_z.noalias() = vars.z_u - vars.z_l;
    Vec<T>& work_x_2 = res_nr.x; // use res_nr.x as temporary, gets overwritten in the next section
    m_kkt_system.eval_G_xn_and_GT_xt(m_data, T(1), T(1), vars.x, work_z, res_nr.z_l, work_x_2);
    res_nr.z_u.noalias() = -res_nr.z_l;
    work_x.noalias() += work_x_2;

    // first part of dual residual and infeasibility calculation (used in cost calculation)
    m_kkt_system.eval_P_x(m_data, T(-1), vars.x, res_nr.x);
    T dual_rel_norm = m_preconditioner.unscale_dual_res(res_nr.x).template lpNorm<Eigen::Infinity>();

    // calculate primal cost, dual cost, and duality gap
    T tmp = -vars.x.dot(res_nr.x); // x'Px
    info.primal_obj = T(0.5) * tmp;
    info.dual_obj = -T(0.5) * tmp;
    T duality_gap_rel_norm = m_preconditioner.unscale_cost(std::abs(tmp));
    tmp = m_data.c.dot(vars.x);
    info.primal_obj += tmp;
    cert_c_x = tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(std::abs(tmp)));
    tmp = m_data.b.dot(vars.y);
    info.dual_obj -= tmp;
    cert_dual_lin = tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(std::abs(tmp)));
    tmp = -m_data.h_l.dot(vars.z_l);
    info.dual_obj -= tmp;
    cert_dual_lin += tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(std::abs(tmp)));
    tmp = m_data.h_u.dot(vars.z_u);
    info.dual_obj -= tmp;
    cert_dual_lin += tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(std::abs(tmp)));
    tmp = -m_data.x_l.head(m_data.n_x_l).dot(vars.z_bl.head(m_data.n_x_l));
    info.dual_obj -= tmp;
    cert_dual_lin += tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(std::abs(tmp)));
    tmp = m_data.x_u.head(m_data.n_x_u).dot(vars.z_bu.head(m_data.n_x_u));
    info.dual_obj -= tmp;
    cert_dual_lin += tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(std::abs(tmp)));

    info.duality_gap = std::abs(info.primal_obj - info.dual_obj);

    info.primal_obj = m_preconditioner.unscale_cost(info.primal_obj);
    info.dual_obj = m_preconditioner.unscale_cost(info.dual_obj);
    info.duality_gap = m_preconditioner.unscale_cost(info.duality_gap);
    info.duality_gap_rel = info.duality_gap / (std::max)(T(1), duality_gap_rel_norm);

    // dual residual and infeasibility calculation
    res_nr.x.noalias() -= m_data.c;
    dual_rel_norm = (std::max)(dual_rel_norm, m_preconditioner.unscale_dual_res(m_data.c).template lpNorm<Eigen::Infinity>());
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        work_x(idx) -= m_data.x_b_scaling(idx) * vars.z_bl(i);
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        work_x(idx) += m_data.x_b_scaling(idx) * vars.z_bu(i);
    }
    dual_rel_norm = (std::max)(dual_rel_norm, m_preconditioner.unscale_dual_res(work_x).template lpNorm<Eigen::Infinity>());
    res_nr.x.noalias() -= work_x;

    // primal residual and infeasibility calculation
    T primal_rel_norm = m_preconditioner.unscale_primal_res_eq(res_nr.y).template lpNorm<Eigen::Infinity>();
    res_nr.y.noalias() += m_data.b;
    primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_eq(m_data.b).template lpNorm<Eigen::Infinity>());

    isize i = 0;
    for (isize ii = 0; ii < m_data.n_h_l; ii++)
    {
        Eigen::Index idx = m_data.h_l_idx(ii);
        // zero out residuals corresponding to infinite bounds
        while (i < idx) {
            res_nr.z_l(i++) = T(0);
        }
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_ineq(res_nr.z_l)(i));
        res_nr.z_l(i) += -m_data.h_l(i) - vars.s_l(i);
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_ineq(m_data.h_l)(i));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_ineq(vars.s_l)(i));
        i++;
    }
    // zero out remaining residuals corresponding to infinite bounds
    while (i < m_data.m) {
        res_nr.z_l(i++) = T(0);
    }

    i = 0;
    for (isize ii = 0; ii < m_data.n_h_u; ii++)
    {
        Eigen::Index idx = m_data.h_u_idx(ii);
        // zero out residuals corresponding to infinite bounds
        while (i < idx) {
            res_nr.z_u(i++) = T(0);
        }
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_ineq(res_nr.z_u)(i));
        res_nr.z_u(i) += m_data.h_u(i) - vars.s_u(i);
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_ineq(m_data.h_u)(i));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_ineq(vars.s_u)(i));
        i++;
    }
    // zero out remaining residuals corresponding to infinite bounds
    while (i < m_data.m) {
        res_nr.z_u(i++) = T(0);
    }

    for (i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        res_nr.z_bl(i) = m_data.x_b_scaling(idx) * vars.x(idx);
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(res_nr.z_bl(i), idx));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(m_data.x_l(i), idx));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(vars.s_bl(i), idx));
    }
    res_nr.z_bl.head(m_data.n_x_l).noalias() += -m_data.x_l.head(m_data.n_x_l) - vars.s_bl.head(m_data.n_x_l);

    for (i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        res_nr.z_bu(i) = -m_data.x_b_scaling(idx) * vars.x(idx);
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(res_nr.z_bu(i), idx));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(m_data.x_u(i), idx));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(vars.s_bu(i), idx));
    }
    res_nr.z_bu.head(m_data.n_x_u).noalias() += m_data.x_u.head(m_data.n_x_u) - vars.s_bu.head(m_data.n_x_u);

    info.prev_primal_res = info.primal_res;
    info.prev_dual_res = info.dual_res;

    info.primal_res = primal_res_nr();
    info.primal_res_rel = info.primal_res / (std::max)(T(1), primal_rel_norm);

    info.dual_res = dual_res_nr();
    info.dual_res_rel = info.dual_res / (std::max)(T(1), dual_rel_norm);
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::update_residuals_r()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::update_residuals_r");

    res.x = res_nr.x - m_result.info.rho * (m_result.x - prox_vars.x);
    res.y = res_nr.y - m_result.info.delta * (prox_vars.y - m_result.y);
    res.z_l = res_nr.z_l - m_result.info.delta * (prox_vars.z_l - m_result.z_l);
    res.z_u = res_nr.z_u - m_result.info.delta * (prox_vars.z_u - m_result.z_u);
    res.z_bl.head(m_data.n_x_l) = res_nr.z_bl.head(m_data.n_x_l) - m_result.info.delta * (prox_vars.z_bl.head(m_data.n_x_l) - m_result.z_bl.head(m_data.n_x_l));
    res.z_bu.head(m_data.n_x_u) = res_nr.z_bu.head(m_data.n_x_u) - m_result.info.delta * (prox_vars.z_bu.head(m_data.n_x_u) - m_result.z_bu.head(m_data.n_x_u));

    T primal_rel_scaling = m_result.info.primal_res_rel > 0 ? m_result.info.primal_res / m_result.info.primal_res_rel : T(1);
    T dual_rel_scaling = m_result.info.dual_res_rel > 0 ? m_result.info.dual_res / m_result.info.dual_res_rel : T(1);

    m_result.info.primal_res_reg = primal_res_r();
    m_result.info.primal_res_reg_rel = m_result.info.primal_res_reg / primal_rel_scaling;
    m_result.info.dual_res_reg = dual_res_r();
    m_result.info.dual_res_reg_rel = m_result.info.dual_res_reg / dual_rel_scaling;

    m_result.info.primal_prox_inf = primal_prox_inf() * m_result.info.delta;
    m_result.info.dual_prox_inf = dual_prox_inf() * m_result.info.rho;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
T SolverBase<T, I, Preconditioner, MatrixType>::primal_res_nr()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::primal_res_nr");

    T inf = m_preconditioner.unscale_primal_res_eq(res_nr.y).template lpNorm<Eigen::Infinity>();
    inf = (std::max)(inf, m_preconditioner.unscale_primal_res_ineq(res_nr.z_l).template lpNorm<Eigen::Infinity>());
    inf = (std::max)(inf, m_preconditioner.unscale_primal_res_ineq(res_nr.z_u).template lpNorm<Eigen::Infinity>());
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        inf = (std::max)(inf, m_preconditioner.unscale_primal_res_b_i(res_nr.z_bl(i), m_data.x_l_idx(i)));
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        inf = (std::max)(inf, m_preconditioner.unscale_primal_res_b_i(res_nr.z_bu(i), m_data.x_u_idx(i)));
    }
    return inf;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
T SolverBase<T, I, Preconditioner, MatrixType>::primal_res_r()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::primal_res_r");

    T inf = m_preconditioner.unscale_primal_res_eq(res.y).template lpNorm<Eigen::Infinity>();
    inf = (std::max)(inf, m_preconditioner.unscale_primal_res_ineq(res.z_l).template lpNorm<Eigen::Infinity>());
    inf = (std::max)(inf, m_preconditioner.unscale_primal_res_ineq(res.z_u).template lpNorm<Eigen::Infinity>());
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        inf = (std::max)(inf, m_preconditioner.unscale_primal_res_b_i(res.z_bl(i), m_data.x_l_idx(i)));
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        inf = (std::max)(inf, m_preconditioner.unscale_primal_res_b_i(res.z_bu(i), m_data.x_u_idx(i)));
    }
    return inf;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
T SolverBase<T, I, Preconditioner, MatrixType>::primal_prox_inf()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::primal_prox_inf");

    T inf = m_preconditioner.unscale_dual_eq(prox_vars.y - m_result.y).template lpNorm<Eigen::Infinity>();
    inf = (std::max)(inf, m_preconditioner.unscale_dual_ineq(prox_vars.z_l - m_result.z_l).template lpNorm<Eigen::Infinity>());
    inf = (std::max)(inf, m_preconditioner.unscale_dual_ineq(prox_vars.z_u - m_result.z_u).template lpNorm<Eigen::Infinity>());
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        inf = (std::max)(inf, m_preconditioner.unscale_dual_b_i(prox_vars.z_bl(i) - m_result.z_bl(i), m_data.x_l_idx(i)));
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        inf = (std::max)(inf, m_preconditioner.unscale_dual_b_i(prox_vars.z_bu(i) - m_result.z_bu(i), m_data.x_u_idx(i)));
    }
    return inf;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
T SolverBase<T, I, Preconditioner, MatrixType>::dual_res_nr()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::dual_res_nr");

    return m_preconditioner.unscale_dual_res(res_nr.x).template lpNorm<Eigen::Infinity>();
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
T SolverBase<T, I, Preconditioner, MatrixType>::dual_res_r()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::dual_res_r");

    return m_preconditioner.unscale_dual_res(res.x).template lpNorm<Eigen::Infinity>();
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
T SolverBase<T, I, Preconditioner, MatrixType>::dual_prox_inf()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::dual_prox_inf");

    return m_preconditioner.unscale_primal(m_result.x - prox_vars.x).template lpNorm<Eigen::Infinity>();
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::update_primal_prox_center()
{
    prox_vars.x = m_result.x;
    prox_c_x = cert_c_x;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::update_dual_prox_center()
{
    prox_vars.y = m_result.y;
    prox_vars.z_l = m_result.z_l;
    prox_vars.z_u = m_result.z_u;
    prox_vars.z_bl.head(m_data.n_x_l) = m_result.z_bl.head(m_data.n_x_l);
    prox_vars.z_bu.head(m_data.n_x_u) = m_result.z_bu.head(m_data.n_x_u);
    prox_dual_lin = cert_dual_lin;
}

// Checks if the displacement from the proximal centers, i.e., the proximal step,
// is a certificate of primal or dual infeasibility, similar to OSQP. All conditions
// are evaluated on the unscaled problem and are relative to the norm of the candidate.
// If a certificate is found, it is stored in the result (see below).
//
// primal infeasibility (Farkas): dy, dz >= 0 with
//   A^T dy + G^T (dz_u - dz_l) - dz_bl + dz_bu = 0,
//   b^T dy - h_l^T dz_l + h_u^T dz_u - x_l^T dz_bl + x_u^T dz_bu < 0
//
// dual infeasibility (unbounded direction): dx with
//   P dx = 0, c^T dx < 0, A dx = 0,
//   (G dx)_i >= 0 for finite h_l, (G dx)_i <= 0 for finite h_u,
//   dx_i >= 0 for finite x_l, dx_i <= 0 for finite x_u
template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::check_infeasibility_certificates()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::check_infeasibility_certificates");

    // ------------------ primal infeasibility candidate ------------------
    T norm_dual = m_preconditioner.unscale_dual_eq(delta_vars.y).template lpNorm<Eigen::Infinity>();
    // b^T dy - h_l^T dz_l + h_u^T dz_u - x_l^T dz_bl + x_u^T dz_bu, reused from the residual calculation
    T farkas_obj = m_preconditioner.unscale_cost(delta_dual_lin);
    // dz_u - dz_l restricted to finite bounds
    cert_work_m_1.setZero();
    for (isize i = 0; i < m_data.n_h_l; i++)
    {
        Eigen::Index idx = m_data.h_l_idx(i);
        norm_dual = (std::max)(norm_dual, std::abs(m_preconditioner.unscale_dual_ineq(delta_vars.z_l)(idx)));
        cert_work_m_1(idx) -= delta_vars.z_l(idx);
    }
    for (isize i = 0; i < m_data.n_h_u; i++)
    {
        Eigen::Index idx = m_data.h_u_idx(i);
        norm_dual = (std::max)(norm_dual, std::abs(m_preconditioner.unscale_dual_ineq(delta_vars.z_u)(idx)));
        cert_work_m_1(idx) += delta_vars.z_u(idx);
    }
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        norm_dual = (std::max)(norm_dual, std::abs(m_preconditioner.unscale_dual_b_i(delta_vars.z_bl(i), m_data.x_l_idx(i))));
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        norm_dual = (std::max)(norm_dual, std::abs(m_preconditioner.unscale_dual_b_i(delta_vars.z_bu(i), m_data.x_u_idx(i))));
    }
    T eps_p = m_settings.eps_primal_inf * norm_dual;
    // Large dual ascent steps of a converging method can look like a certificate,
    // hence, we only accept a certificate if the primal residual stalls.
    bool primal_stalled = m_result.info.primal_res >= T(0.95) * m_result.info.prev_primal_res;
    bool primal_candidate = primal_stalled && norm_dual > 0 && farkas_obj < -eps_p;
    // dual cone: dz >= 0
    for (isize i = 0; primal_candidate && i < m_data.n_h_l; i++)
    {
        Eigen::Index idx = m_data.h_l_idx(i);
        primal_candidate = m_preconditioner.unscale_dual_ineq(delta_vars.z_l)(idx) >= -eps_p;
    }
    for (isize i = 0; primal_candidate && i < m_data.n_h_u; i++)
    {
        Eigen::Index idx = m_data.h_u_idx(i);
        primal_candidate = m_preconditioner.unscale_dual_ineq(delta_vars.z_u)(idx) >= -eps_p;
    }
    for (isize i = 0; primal_candidate && i < m_data.n_x_l; i++)
    {
        primal_candidate = m_preconditioner.unscale_dual_b_i(delta_vars.z_bl(i), m_data.x_l_idx(i)) >= -eps_p;
    }
    for (isize i = 0; primal_candidate && i < m_data.n_x_u; i++)
    {
        primal_candidate = m_preconditioner.unscale_dual_b_i(delta_vars.z_bu(i), m_data.x_u_idx(i)) >= -eps_p;
    }

    // ------------------ dual infeasibility candidate ------------------
    T norm_dx = m_preconditioner.unscale_primal(delta_vars.x).template lpNorm<Eigen::Infinity>();
    T eps_d = m_settings.eps_dual_inf * norm_dx;
    bool dual_stalled = m_result.info.dual_res >= T(0.95) * m_result.info.prev_dual_res;
    bool dual_candidate = dual_stalled && norm_dx > 0 && m_preconditioner.unscale_cost(delta_c_x) < -eps_d;
    // recession cone of the variable bounds
    for (isize i = 0; dual_candidate && i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        dual_candidate = m_preconditioner.unscale_primal(delta_vars.x)(idx) >= -eps_d;
    }
    for (isize i = 0; dual_candidate && i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        dual_candidate = m_preconditioner.unscale_primal(delta_vars.x)(idx) <= eps_d;
    }

    if (!primal_candidate && !dual_candidate) return Status::PIQP_UNSOLVED;

    // cert_work_p = A dx, cert_work_n_1 = A^T dy
    m_kkt_system.eval_A_xn_and_AT_xt(m_data, T(1), T(1), delta_vars.x, delta_vars.y, cert_work_p, cert_work_n_1);
    // cert_work_m_2 = G dx, cert_work_n_2 = G^T (dz_u - dz_l)
    m_kkt_system.eval_G_xn_and_GT_xt(m_data, T(1), T(1), delta_vars.x, cert_work_m_1, cert_work_m_2, cert_work_n_2);

    if (primal_candidate)
    {
        cert_work_n_1.noalias() += cert_work_n_2;
        for (isize i = 0; i < m_data.n_x_l; i++)
        {
            Eigen::Index idx = m_data.x_l_idx(i);
            cert_work_n_1(idx) -= m_data.x_b_scaling(idx) * delta_vars.z_bl(i);
        }
        for (isize i = 0; i < m_data.n_x_u; i++)
        {
            Eigen::Index idx = m_data.x_u_idx(i);
            cert_work_n_1(idx) += m_data.x_b_scaling(idx) * delta_vars.z_bu(i);
        }
        if (m_preconditioner.unscale_dual_res(cert_work_n_1).template lpNorm<Eigen::Infinity>() <= eps_p)
        {
            store_primal_infeasibility_certificate(norm_dual);
            return Status::PIQP_PRIMAL_INFEASIBLE;
        }
    }

    if (dual_candidate)
    {
        if (m_preconditioner.unscale_primal_res_eq(cert_work_p).template lpNorm<Eigen::Infinity>() > eps_d) return Status::PIQP_UNSOLVED;
        for (isize i = 0; i < m_data.n_h_l; i++)
        {
            Eigen::Index idx = m_data.h_l_idx(i);
            if (m_preconditioner.unscale_primal_res_ineq(cert_work_m_2)(idx) < -eps_d) return Status::PIQP_UNSOLVED;
        }
        for (isize i = 0; i < m_data.n_h_u; i++)
        {
            Eigen::Index idx = m_data.h_u_idx(i);
            if (m_preconditioner.unscale_primal_res_ineq(cert_work_m_2)(idx) > eps_d) return Status::PIQP_UNSOLVED;
        }
        // cert_work_n_1 = P dx
        m_kkt_system.eval_P_x(m_data, T(1), delta_vars.x, cert_work_n_1);
        if (m_preconditioner.unscale_dual_res(cert_work_n_1).template lpNorm<Eigen::Infinity>() <= eps_d)
        {
            store_dual_infeasibility_certificate(norm_dx);
            return Status::PIQP_DUAL_INFEASIBLE;
        }
    }

    return Status::PIQP_UNSOLVED;
}

// Stores the primal infeasibility certificate (dy, dz_l, dz_u, dz_bl, dz_bu) in the dual
// variables of the result, normalized such that the unscaled certificate has unit infinity norm.
// The primal variables are left at the last iterate.
template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::store_primal_infeasibility_certificate(const T& norm)
{
    m_result.y.noalias() = delta_vars.y / norm;
    // only constraints with finite bounds are part of the certificate
    m_result.z_l.setZero();
    m_result.z_u.setZero();
    for (isize i = 0; i < m_data.n_h_l; i++)
    {
        Eigen::Index idx = m_data.h_l_idx(i);
        m_result.z_l(idx) = delta_vars.z_l(idx) / norm;
    }
    for (isize i = 0; i < m_data.n_h_u; i++)
    {
        Eigen::Index idx = m_data.h_u_idx(i);
        m_result.z_u(idx) = delta_vars.z_u(idx) / norm;
    }
    m_result.z_bl.head(m_data.n_x_l).noalias() = delta_vars.z_bl.head(m_data.n_x_l) / norm;
    m_result.z_bu.head(m_data.n_x_u).noalias() = delta_vars.z_bu.head(m_data.n_x_u) / norm;
}

// Stores the dual infeasibility certificate dx in the primal variables of the result,
// normalized such that the unscaled certificate has unit infinity norm.
// The dual variables are left at the last iterate.
template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::store_dual_infeasibility_certificate(const T& norm)
{
    m_result.x.noalias() = delta_vars.x / norm;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::scale_results(Variables<T>& vars)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::scale_results");

    vars.x = m_preconditioner.scale_primal(vars.x);
    vars.y = m_preconditioner.scale_dual_eq(vars.y);
    vars.z_l = m_preconditioner.scale_dual_ineq(vars.z_l);
    vars.z_u = m_preconditioner.scale_dual_ineq(vars.z_u);
    vars.s_l = m_preconditioner.scale_slack_ineq(vars.s_l);
    vars.s_u = m_preconditioner.scale_slack_ineq(vars.s_u);
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        vars.z_bl(i) = m_preconditioner.scale_dual_b_i(vars.z_bl(i), idx);
        vars.s_bl(i) = m_preconditioner.scale_slack_b_i(vars.s_bl(i), idx);
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        vars.z_bu(i) = m_preconditioner.scale_dual_b_i(vars.z_bu(i), idx);
        vars.s_bu(i) = m_preconditioner.scale_slack_b_i(vars.s_bu(i), idx);
    }
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::unscale_results(Variables<T>& vars)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::unscale_results");

    vars.x = m_preconditioner.unscale_primal(vars.x);
    vars.y = m_preconditioner.unscale_dual_eq(vars.y);
    vars.z_l = m_preconditioner.unscale_dual_ineq(vars.z_l);
    vars.z_u = m_preconditioner.unscale_dual_ineq(vars.z_u);
    vars.s_l = m_preconditioner.unscale_slack_ineq(vars.s_l);
    vars.s_u = m_preconditioner.unscale_slack_ineq(vars.s_u);
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        vars.z_bl(i) = m_preconditioner.unscale_dual_b_i(vars.z_bl(i), idx);
        vars.s_bl(i) = m_preconditioner.unscale_slack_b_i(vars.s_bl(i), idx);
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        vars.z_bu(i) = m_preconditioner.unscale_dual_b_i(vars.z_bu(i), idx);
        vars.s_bu(i) = m_preconditioner.unscale_slack_b_i(vars.s_bu(i), idx);
    }
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::pack_dual(Variables<T>& vars)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::pack_dual");

    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        std::swap(vars.z_bl(i), vars.z_bl(idx));
        std::swap(vars.s_bl(i), vars.s_bl(idx));
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        std::swap(vars.z_bu(i), vars.z_bu(idx));
        std::swap(vars.s_bu(i), vars.s_bu(idx));
    }
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::restore_dual()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::restore_dual");

    for (isize i = 0; i < m_data.m; i++)
    {
        if (m_result.z_l(i) == 0) {
            m_result.s_l(i) = PIQP_INF;
        }
        if (m_result.z_u(i) == 0) {
            m_result.s_u(i) = PIQP_INF;
        }
    }

    m_result.z_bl.tail(m_data.n - m_data.n_x_l).setZero();
    m_result.z_bu.tail(m_data.n - m_data.n_x_u).setZero();
    m_result.s_bl.tail(m_data.n - m_data.n_x_l).array() = PIQP_INF;;
    m_result.s_bu.tail(m_data.n - m_data.n_x_u).array() = PIQP_INF;;
    unpack_dual(m_result);
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::reduce_dual(Variables<T>& vars)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::reduce_dual");

    auto finite = [](const T& s) { return s < PIQP_INF ? s : T(0); };
    vars.s_l = vars.s_l.unaryExpr(finite);
    vars.s_u = vars.s_u.unaryExpr(finite);
    vars.s_bl = vars.s_bl.unaryExpr(finite);
    vars.s_bu = vars.s_bu.unaryExpr(finite);
    pack_dual(vars);
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::unpack_dual(Variables<T>& vars)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::unpack_dual");

    for (isize i = m_data.n_x_l - 1; i >= 0; i--)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        std::swap(vars.z_bl(i), vars.z_bl(idx));
        std::swap(vars.s_bl(i), vars.s_bl(idx));
    }
    for (isize i = m_data.n_x_u - 1; i >= 0; i--)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        std::swap(vars.z_bu(i), vars.z_bu(idx));
        std::swap(vars.s_bu(i), vars.s_bu(idx));
    }
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::backward(const optional<CVecRef<T>>& grad_x,
                                                              const optional<CVecRef<T>>& grad_y,
                                                              const optional<CVecRef<T>>& grad_z_l,
                                                              const optional<CVecRef<T>>& grad_z_u,
                                                              const optional<CVecRef<T>>& grad_z_bl,
                                                              const optional<CVecRef<T>>& grad_z_bu,
                                                              const optional<CVecRef<T>>& grad_s_l,
                                                              const optional<CVecRef<T>>& grad_s_u,
                                                              const optional<CVecRef<T>>& grad_s_bl,
                                                              const optional<CVecRef<T>>& grad_s_bu,
                                                              const optional<Result<T>>& forward_result)
{
    m_gradients.status = backward_impl(grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu,
                                       grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, forward_result);
    return m_gradients.status;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::backward_impl(const optional<CVecRef<T>>& grad_x,
                                                                   const optional<CVecRef<T>>& grad_y,
                                                                   const optional<CVecRef<T>>& grad_z_l,
                                                                   const optional<CVecRef<T>>& grad_z_u,
                                                                   const optional<CVecRef<T>>& grad_z_bl,
                                                                   const optional<CVecRef<T>>& grad_z_bu,
                                                                   const optional<CVecRef<T>>& grad_s_l,
                                                                   const optional<CVecRef<T>>& grad_s_u,
                                                                   const optional<CVecRef<T>>& grad_s_bl,
                                                                   const optional<CVecRef<T>>& grad_s_bu,
                                                                   const optional<Result<T>>& forward_result)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::backward");

    if (!m_setup_done)
    {
        piqp_eprint("Solver not setup yet\n");
        return Status::PIQP_INVALID_SETTINGS;
    }

    if (!m_backward_enabled)
    {
        piqp_eprint("backward() requires settings.enable_backward to be set before setup()\n");
        return Status::PIQP_INVALID_SETTINGS;
    }

    if (!forward_result.has_value() && !m_backward_ready)
    {
        piqp_eprint("backward() requires a solve() of the current problem data which returned PIQP_SOLVED or PIQP_MAX_ITER_REACHED\n");
        return Status::PIQP_UNSOLVED;
    }

    auto valid_size = [](const optional<CVecRef<T>>& grad, isize size, const char* name) {
        if (grad.has_value() && grad->size() != size)
        {
            piqp_eprint("%s has wrong dimensions\n", name);
            return false;
        }
        return true;
    };
    if (!valid_size(grad_x, m_data.n, "grad_x") || !valid_size(grad_y, m_data.p, "grad_y") ||
        !valid_size(grad_z_l, m_data.m, "grad_z_l") || !valid_size(grad_z_u, m_data.m, "grad_z_u") ||
        !valid_size(grad_z_bl, m_data.n, "grad_z_bl") || !valid_size(grad_z_bu, m_data.n, "grad_z_bu") ||
        !valid_size(grad_s_l, m_data.m, "grad_s_l") || !valid_size(grad_s_u, m_data.m, "grad_s_u") ||
        !valid_size(grad_s_bl, m_data.n, "grad_s_bl") || !valid_size(grad_s_bu, m_data.n, "grad_s_bu"))
    {
        return Status::PIQP_INVALID_SETTINGS;
    }

    Info<T> info = m_result.info;
    if (forward_result.has_value())
    {
        Status point_status = init_linearization_point(*forward_result);
        if (point_status != Status::PIQP_SOLVED)
        {
            return point_status;
        }
        info = forward_result->info;
    }
    else
    {
        m_linearization_point = m_result;
        reduce_dual(m_linearization_point);
        scale_results(m_linearization_point);
    }

    if (m_settings.gradient_smoothing)
    {
        Status smoothing_status = smooth_linearization_point(info);
        if (smoothing_status != Status::PIQP_SOLVED)
        {
            return smoothing_status;
        }
    }
    else if (forward_result.has_value() || !m_backward_factor_valid)
    {
        m_backward_factor_valid = false;
        if (!factor_kkt(m_linearization_point, info))
        {
            piqp_eprint("backward(): KKT factorization at the linearization point failed\n");
            return Status::PIQP_NUMERICS;
        }
        // only the linearization point of the last solve() result is cached
        m_backward_factor_valid = !forward_result.has_value();
    }

    // the forward residuals and steps are not needed anymore and hold the adjoint system
    Variables<T>& rhs = res;
    Variables<T>& lambda = step;
    assemble_adjoint_rhs(grad_x, grad_y, grad_z_l, grad_z_u, grad_z_bl, grad_z_bu,
                         grad_s_l, grad_s_u, grad_s_bl, grad_s_bu, rhs);
    if (!m_kkt_system.solve(m_data, m_settings, rhs, lambda))
    {
        piqp_eprint("backward(): adjoint KKT solve failed\n");
        return Status::PIQP_NUMERICS;
    }
    unscale_adjoint(lambda);
    unscale_results(m_linearization_point);

    // disabled rows are decoupled from the problem, zeroing their duals and adjoints zeros their gradients
    for (isize i = 0; i < m_data.m; i++)
    {
        if (m_data.G_row_disabled(i))
        {
            lambda.z_l(i) = T(0);
            lambda.z_u(i) = T(0);
            m_linearization_point.z_l(i) = T(0);
            m_linearization_point.z_u(i) = T(0);
        }
    }

    m_gradients.compute_from_adjoint(m_data, m_linearization_point, lambda);

    return Status::PIQP_SOLVED;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::init_linearization_point(const Result<T>& forward_result)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::init_linearization_point");

    if (forward_result.info.status != Status::PIQP_SOLVED && forward_result.info.status != Status::PIQP_MAX_ITER_REACHED)
    {
        piqp_eprint("backward(): the provided forward result must have status PIQP_SOLVED or PIQP_MAX_ITER_REACHED\n");
        return Status::PIQP_UNSOLVED;
    }
    if (forward_result.x.size() != m_data.n || forward_result.y.size() != m_data.p
        || forward_result.z_l.size() != m_data.m || forward_result.z_u.size() != m_data.m
        || forward_result.s_l.size() != m_data.m || forward_result.s_u.size() != m_data.m
        || forward_result.z_bl.size() != m_data.n || forward_result.z_bu.size() != m_data.n
        || forward_result.s_bl.size() != m_data.n || forward_result.s_bu.size() != m_data.n)
    {
        piqp_eprint("backward(): the provided forward result has wrong dimensions\n");
        return Status::PIQP_INVALID_SETTINGS;
    }
    if (!(forward_result.info.rho > 0) || !(forward_result.info.delta > 0) || !std::isfinite(forward_result.info.rho) || !std::isfinite(forward_result.info.delta))
    {
        piqp_eprint("backward(): the provided forward result has an invalid regularization (info.rho, info.delta)\n");
        return Status::PIQP_INVALID_SETTINGS;
    }

    // s and z have to be interior on finite bounds, except on disabled rows, which are decoupled from the problem
    auto set_interior = [](const Vec<T>& s_in, const Vec<T>& z_in, Vec<T>& s, Vec<T>& z, Eigen::Index idx, bool disabled) {
        if (s_in(idx) > 0 && z_in(idx) > 0 && std::isfinite(s_in(idx)) && std::isfinite(z_in(idx)))
        {
            s(idx) = s_in(idx);
            z(idx) = z_in(idx);
            return true;
        }
        s(idx) = T(1);
        z(idx) = T(1);
        return disabled;
    };

    Variables<T>& point = m_linearization_point;
    point.x = forward_result.x;
    point.y = forward_result.y;
    point.z_l.setZero();
    point.z_u.setZero();
    point.z_bl.setZero();
    point.z_bu.setZero();
    point.s_l.setZero();
    point.s_u.setZero();
    point.s_bl.setZero();
    point.s_bu.setZero();
    for (isize i = 0; i < m_data.n_h_l; i++)
    {
        Eigen::Index idx = m_data.h_l_idx(i);
        if (!set_interior(forward_result.s_l, forward_result.z_l, point.s_l, point.z_l, idx, m_data.G_row_disabled(idx)))
        {
            piqp_eprint("backward(): s_l and z_l of the provided forward result must be strictly positive on finite bounds\n");
            return Status::PIQP_INVALID_SETTINGS;
        }
    }
    for (isize i = 0; i < m_data.n_h_u; i++)
    {
        Eigen::Index idx = m_data.h_u_idx(i);
        if (!set_interior(forward_result.s_u, forward_result.z_u, point.s_u, point.z_u, idx, m_data.G_row_disabled(idx)))
        {
            piqp_eprint("backward(): s_u and z_u of the provided forward result must be strictly positive on finite bounds\n");
            return Status::PIQP_INVALID_SETTINGS;
        }
    }
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        if (!set_interior(forward_result.s_bl, forward_result.z_bl, point.s_bl, point.z_bl, m_data.x_l_idx(i), false))
        {
            piqp_eprint("backward(): s_bl and z_bl of the provided forward result must be strictly positive on finite bounds\n");
            return Status::PIQP_INVALID_SETTINGS;
        }
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        if (!set_interior(forward_result.s_bu, forward_result.z_bu, point.s_bu, point.z_bu, m_data.x_u_idx(i), false))
        {
            piqp_eprint("backward(): s_bu and z_bu of the provided forward result must be strictly positive on finite bounds\n");
            return Status::PIQP_INVALID_SETTINGS;
        }
    }

    pack_dual(point);
    scale_results(point);

    return Status::PIQP_SOLVED;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::smooth_linearization_point(Info<T>& info)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::smooth_linearization_point");

    Variables<T>& vars = m_linearization_point;

    // s * z = mu in user space corresponds to s * z = c * mu in scaled space with the cost scaling c
    T mu = m_preconditioner.scale_cost(m_settings.gradient_smoothing_mu);

    // complementarity targets, zero on infinite bounds and on bounds with equal lower and upper values (no interior)
    BasicVariables<T>& target = prox_vars; // only used in solve()
    target.z_l.setZero();
    target.z_u.setZero();
    for (isize i = 0; i < m_data.n_h_l; i++)
    {
        Eigen::Index idx = m_data.h_l_idx(i);
        target.z_l(idx) = m_data.h_l(idx) == m_data.h_u(idx) ? T(0) : mu;
    }
    for (isize i = 0; i < m_data.n_h_u; i++)
    {
        Eigen::Index idx = m_data.h_u_idx(i);
        target.z_u(idx) = m_data.h_l(idx) == m_data.h_u(idx) ? T(0) : mu;
    }
    target.z_bl.head(m_data.n_x_l).setConstant(mu);
    target.z_bu.head(m_data.n_x_u).setConstant(mu);
    for (isize i = 0, j = 0; i < m_data.n_x_l; i++)
    {
        while (j < m_data.n_x_u && m_data.x_u_idx(j) < m_data.x_l_idx(i)) j++;
        if (j < m_data.n_x_u && m_data.x_u_idx(j) == m_data.x_l_idx(i) && m_data.x_l(i) == m_data.x_u(j))
        {
            target.z_bl(i) = T(0);
            target.z_bu(j) = T(0);
        }
    }

    // project s * z = target, keeping s - z
    auto project = [](Vec<T>& s, Vec<T>& z, const Vec<T>& t, isize n) {
        for (isize i = 0; i < n; i++)
        {
            if (t(i) > 0)
            {
                nonneg_smoothing(T(1), t(i), s(i) - z(i), s(i), z(i));
            }
        }
    };
    project(vars.s_l, vars.z_l, target.z_l, m_data.m);
    project(vars.s_u, vars.z_u, target.z_u, m_data.m);
    project(vars.s_bl, vars.z_bl, target.z_bl, m_data.n_x_l);
    project(vars.s_bu, vars.z_bu, target.z_bu, m_data.n_x_u);

    m_backward_factor_valid = false;
    for (isize iter = 0;; iter++)
    {
        if (!factor_kkt(vars, info))
        {
            piqp_eprint("backward(): KKT factorization failed during gradient smoothing\n");
            return Status::PIQP_NUMERICS;
        }

        update_residuals_nr(vars, info);
        res.x = res_nr.x;
        res.y = res_nr.y;
        res.z_l = res_nr.z_l;
        res.z_u = res_nr.z_u;
        res.z_bl.head(m_data.n_x_l) = res_nr.z_bl.head(m_data.n_x_l);
        res.z_bu.head(m_data.n_x_u) = res_nr.z_bu.head(m_data.n_x_u);

        res.s_l.array() = target.z_l.array() - vars.s_l.array() * vars.z_l.array();
        res.s_u.array() = target.z_u.array() - vars.s_u.array() * vars.z_u.array();
        res.s_bl.head(m_data.n_x_l).array() = target.z_bl.head(m_data.n_x_l).array() - vars.s_bl.head(m_data.n_x_l).array() * vars.z_bl.head(m_data.n_x_l).array();
        res.s_bu.head(m_data.n_x_u).array() = target.z_bu.head(m_data.n_x_u).array() - vars.s_bu.head(m_data.n_x_u).array() * vars.z_bu.head(m_data.n_x_u).array();
        T complementarity_res = res.s_l.template lpNorm<Eigen::Infinity>();
        complementarity_res = (std::max)(complementarity_res, res.s_u.template lpNorm<Eigen::Infinity>());
        complementarity_res = (std::max)(complementarity_res, res.s_bl.head(m_data.n_x_l).template lpNorm<Eigen::Infinity>());
        complementarity_res = (std::max)(complementarity_res, res.s_bu.head(m_data.n_x_u).template lpNorm<Eigen::Infinity>());
        complementarity_res = m_preconditioner.unscale_cost(complementarity_res);

        if ((info.primal_res < m_settings.eps_abs || info.primal_res_rel < m_settings.eps_rel) &&
            (info.dual_res < m_settings.eps_abs || info.dual_res_rel < m_settings.eps_rel) &&
            (complementarity_res < m_settings.eps_abs || complementarity_res < m_settings.eps_rel * m_settings.gradient_smoothing_mu))
        {
            break;
        }

        if (iter >= m_settings.gradient_smoothing_max_iter)
        {
            piqp_eprint("backward(): gradient smoothing did not converge within gradient_smoothing_max_iter iterations\n");
            return Status::PIQP_MAX_ITER_REACHED;
        }

        if (!m_kkt_system.solve(m_data, m_settings, res, step))
        {
            piqp_eprint("backward(): KKT solve failed during gradient smoothing\n");
            return Status::PIQP_NUMERICS;
        }

        T alpha_s, alpha_z;
        calculate_step(alpha_s, alpha_z, vars);
        T primal_step = alpha_s * m_settings.tau;
        T dual_step = alpha_z * m_settings.tau;

        vars.x += primal_step * step.x;
        vars.y += dual_step * step.y;
        vars.z_l += dual_step * step.z_l;
        vars.z_u += dual_step * step.z_u;
        vars.z_bl.head(m_data.n_x_l) += dual_step * step.z_bl.head(m_data.n_x_l);
        vars.z_bu.head(m_data.n_x_u) += dual_step * step.z_bu.head(m_data.n_x_u);
        vars.s_l += primal_step * step.s_l;
        vars.s_u += primal_step * step.s_u;
        vars.s_bl.head(m_data.n_x_l) += primal_step * step.s_bl.head(m_data.n_x_l);
        vars.s_bu.head(m_data.n_x_u) += primal_step * step.s_bu.head(m_data.n_x_u);
    }

    return Status::PIQP_SOLVED;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::assemble_adjoint_rhs(const optional<CVecRef<T>>& grad_x,
                                                                        const optional<CVecRef<T>>& grad_y,
                                                                        const optional<CVecRef<T>>& grad_z_l,
                                                                        const optional<CVecRef<T>>& grad_z_u,
                                                                        const optional<CVecRef<T>>& grad_z_bl,
                                                                        const optional<CVecRef<T>>& grad_z_bu,
                                                                        const optional<CVecRef<T>>& grad_s_l,
                                                                        const optional<CVecRef<T>>& grad_s_u,
                                                                        const optional<CVecRef<T>>& grad_s_bl,
                                                                        const optional<CVecRef<T>>& grad_s_bu,
                                                                        Variables<T>& rhs)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::backward::assemble_adjoint_rhs");

    const Variables<T>& point = m_linearization_point;

    if (grad_x.has_value())
    {
        rhs.x = -m_preconditioner.unscale_primal(*grad_x);
    }
    else
    {
        rhs.x.setZero();
    }

    if (grad_y.has_value())
    {
        rhs.y = -m_preconditioner.unscale_dual_eq(*grad_y);
    }
    else
    {
        rhs.y.setZero();
    }

    if (grad_z_l.has_value())
    {
        rhs.z_l = -m_preconditioner.unscale_dual_ineq(*grad_z_l);
    }
    else
    {
        rhs.z_l.setZero();
    }

    if (grad_z_u.has_value())
    {
        rhs.z_u = -m_preconditioner.unscale_dual_ineq(*grad_z_u);
    }
    else
    {
        rhs.z_u.setZero();
    }

    if (grad_s_l.has_value())
    {
        rhs.s_l = -m_preconditioner.unscale_slack_ineq(*grad_s_l);
        rhs.s_l.array() *= point.s_l.array();
    }
    else
    {
        rhs.s_l.setZero();
    }
    
    if (grad_s_u.has_value())
    {
        rhs.s_u = -m_preconditioner.unscale_slack_ineq(*grad_s_u);
        rhs.s_u.array() *= point.s_u.array();
    }
    else
    {
        rhs.s_u.setZero();
    }

    rhs.z_bl.setZero();
    rhs.z_bu.setZero();
    rhs.s_bl.setZero();
    rhs.s_bu.setZero();
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        if (grad_z_bl.has_value())
        {
            rhs.z_bl(i) = -m_preconditioner.unscale_dual_b_i((*grad_z_bl)(idx), idx);
        }
        if (grad_s_bl.has_value())
        {
            rhs.s_bl(i) = -m_preconditioner.unscale_slack_b_i((*grad_s_bl)(idx), idx) * point.s_bl(i);
        }
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        if (grad_z_bu.has_value())
        {
            rhs.z_bu(i) = -m_preconditioner.unscale_dual_b_i((*grad_z_bu)(idx), idx);
        }
        if (grad_s_bu.has_value())
        {
            rhs.s_bu(i) = -m_preconditioner.unscale_slack_b_i((*grad_s_bu)(idx), idx) * point.s_bu(i);
        }
    }
}

// the adjoints scale like the KKT residuals
template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::unscale_adjoint(Variables<T>& lambda)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::backward::unscale_adjoint");

    lambda.x = m_preconditioner.scale_dual_res(lambda.x);
    lambda.y = m_preconditioner.scale_primal_res_eq(lambda.y);
    lambda.z_l = m_preconditioner.scale_primal_res_ineq(lambda.z_l);
    lambda.z_u = m_preconditioner.scale_primal_res_ineq(lambda.z_u);
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        lambda.z_bl(i) = m_preconditioner.scale_primal_res_b_i(lambda.z_bl(i), idx);
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        lambda.z_bu(i) = m_preconditioner.scale_primal_res_b_i(lambda.z_bu(i), idx);
    }
}

template<typename T, typename Preconditioner>
void DenseSolver<T, Preconditioner>::setup(const CMatRef<T>& P,
           const CVecRef<T>& c,
           const optional<CMatRef<T>>& A,
           const optional<CVecRef<T>>& b,
           const optional<CMatRef<T>>& G,
           const optional<CVecRef<T>>& h_l,
           const optional<CVecRef<T>>& h_u,
           const optional<CVecRef<T>>& x_l,
           const optional<CVecRef<T>>& x_u)
{
    this->setup_impl(P, c, A, b, G, h_l, h_u, x_l, x_u);
}

template<typename T, typename Preconditioner>
void DenseSolver<T, Preconditioner>::update(const optional<CMatRef<T>>& P,
            const optional<CVecRef<T>>& c,
            const optional<CMatRef<T>>& A,
            const optional<CVecRef<T>>& b,
            const optional<CMatRef<T>>& G,
            const optional<CVecRef<T>>& h_l,
            const optional<CVecRef<T>>& h_u,
            const optional<CVecRef<T>>& x_l,
            const optional<CVecRef<T>>& x_u)
{
    this->update_impl(P, c, A, b, G, h_l, h_u, x_l, x_u);
}

template<typename T, typename I, typename Preconditioner>
void SparseSolver<T, I, Preconditioner>::setup(const CSparseMatRef<T, I>& P,
           const CVecRef<T>& c,
           const optional<CSparseMatRef<T, I>>& A,
           const optional<CVecRef<T>>& b,
           const optional<CSparseMatRef<T, I>>& G,
           const optional<CVecRef<T>>& h_l,
           const optional<CVecRef<T>>& h_u,
           const optional<CVecRef<T>>& x_l,
           const optional<CVecRef<T>>& x_u)
{
    this->setup_impl(P, c, A, b, G, h_l, h_u, x_l, x_u);
}

template<typename T, typename I, typename Preconditioner>
void SparseSolver<T, I, Preconditioner>::update(const optional<CSparseMatRef<T, I>>& P,
            const optional<CVecRef<T>>& c,
            const optional<CSparseMatRef<T, I>>& A,
            const optional<CVecRef<T>>& b,
            const optional<CSparseMatRef<T, I>>& G,
            const optional<CVecRef<T>>& h_l,
            const optional<CVecRef<T>>& h_u,
            const optional<CVecRef<T>>& x_l,
            const optional<CVecRef<T>>& x_u)
{
    this->update_impl(P, c, A, b, G, h_l, h_u, x_l, x_u);
}

} // namespace piqp

#endif //PIQP_SOLVER_TPP
