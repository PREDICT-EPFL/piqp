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
void SolverBase<T, I, Preconditioner, MatrixType>::set_warm_start(
    const CVecRef<T>& x,
    const optional<CVecRef<T>>& y,
    const optional<CVecRef<T>>& z_l,
    const optional<CVecRef<T>>& z_u,
    const optional<CVecRef<T>>& z_bl,
    const optional<CVecRef<T>>& z_bu)
{
    if (!m_setup_done)
    {
        piqp_eprint("Solver not setup yet\n");
        return;
    }

    if (x.size() != m_data.n) { piqp_eprint("x has wrong dimensions\n"); return; }
    if (y.has_value() && y->size() != m_data.p) { piqp_eprint("y has wrong dimensions\n"); return; }
    if (z_l.has_value() && z_l->size() != m_data.m) { piqp_eprint("z_l has wrong dimensions\n"); return; }
    if (z_u.has_value() && z_u->size() != m_data.m) { piqp_eprint("z_u has wrong dimensions\n"); return; }
    if (z_bl.has_value() && z_bl->size() != m_data.n) { piqp_eprint("z_bl has wrong dimensions\n"); return; }
    if (z_bu.has_value() && z_bu->size() != m_data.n) { piqp_eprint("z_bu has wrong dimensions\n"); return; }

    m_result.x = x;

    // Set equality duals
    if (y.has_value()) {
        m_result.y = *y;
    } else {
        m_result.y.setZero();
    }

    // Set inequality duals
    if (z_l.has_value()) {
        m_result.z_l = *z_l;
    } else {
        m_result.z_l.setZero();
    }
    if (z_u.has_value()) {
        m_result.z_u = *z_u;
    } else {
        m_result.z_u.setZero();
    }

    // Set bound duals
    if (z_bl.has_value()) {
        m_result.z_bl = *z_bl;
    } else {
        m_result.z_bl.setZero();
    }
    if (z_bu.has_value()) {
        m_result.z_bu = *z_bu;
    } else {
        m_result.z_bu.setZero();
    }

    // Slacks are computed from x in init_warm_start, no need to set here.
    m_warm_start_has_y = y.has_value();
    m_warm_start_has_z = z_l.has_value() || z_u.has_value()
                         || z_bl.has_value() || z_bu.has_value();
    m_warm_start_from_solve = false;

    // Mark as having a valid warm-start point
    m_result.info.status = Status::PIQP_SOLVED;
    m_first_run = false;
    m_settings.warm_start = true;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::solve()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::solve");

    if (m_settings.verbose)
    {
        piqp_print("----------------------------------------------------------\n");
        piqp_print("                        PIQP v0.6.2                       \n");
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

    unscale_results();
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

    // After a successful solve, the full primal-dual solution is available for warm starting
    m_warm_start_from_solve = (status == Status::PIQP_SOLVED);
    m_warm_start_has_y = m_warm_start_from_solve;
    m_warm_start_has_z = m_warm_start_from_solve;
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

    Timer<T> update_timer;
    if (m_settings.compute_timings)
    {
        update_timer.start();
    }

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
    if (h_l.has_value()) { m_data.set_h_l(h_l); }
    if (h_u.has_value()) { m_data.set_h_u(h_u); }
    if (h_l.has_value() || h_u.has_value()) { m_data.disable_inf_constraints(); }

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

    bool use_warm_start = m_settings.warm_start && !m_first_run
                          && m_result.info.status == Status::PIQP_SOLVED;

    m_result.info.status = Status::PIQP_UNSOLVED;
    m_result.info.iter = 0;
    m_result.info.init_admm_iter = 0;
    m_result.info.reg_limit = m_settings.reg_lower_limit;
    m_result.info.factor_retires = 0;
    m_result.info.no_primal_update = 0;
    m_result.info.no_dual_update = 0;
    m_result.info.mu = 0;
    m_result.info.primal_step = 0;
    m_result.info.dual_step = 0;
    m_result.info.rho = m_settings.rho_init;
    m_result.info.delta = m_settings.delta_init;

    // For equality-only problems, scale rho and delta down
    if (m_data.m + m_data.n_x_l + m_data.n_x_u == 0)
    {
        m_result.info.rho = (std::max)(m_settings.rho_init * m_settings.rho_eq_factor, m_settings.reg_lower_limit);
        m_result.info.delta = (std::max)(m_settings.delta_init * m_settings.delta_eq_factor, m_settings.reg_lower_limit);
    }

    m_enable_iterative_refinement = m_settings.iterative_refinement_always_enabled;

    if (use_warm_start)
    {
        Status warm_status = init_warm_start();
        if (warm_status != Status::PIQP_UNSOLVED)
        {
            return warm_status;
        }
    }
    else
    {
        Status cold_status = init_cold_start();
        if (cold_status != Status::PIQP_UNSOLVED)
        {
            return cold_status;
        }
    }

    // We make an Eigen expression for convenience. Note that we are doing it after
    // init_cold_start since m_kkt_system.solve might swap internal pointers in m_result
    // which can invalidate the reference in the Eigen expression.
    auto s_bl = m_result.s_bl.head(m_data.n_x_l);
    auto s_bu = m_result.s_bu.head(m_data.n_x_u);
    auto z_bl = m_result.z_bl.head(m_data.n_x_l);
    auto z_bu = m_result.z_bu.head(m_data.n_x_u);
    auto nu_bl = prox_vars.z_bl.head(m_data.n_x_l);
    auto nu_bu = prox_vars.z_bu.head(m_data.n_x_u);

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
            update_residuals_nr();
            m_result.info.prev_primal_res = m_result.info.primal_res;
            m_result.info.prev_dual_res = m_result.info.dual_res;
        }

        if (m_settings.verbose)
        {
            if (m_result.info.iter == 0)
            {
                piqp_print("%2zd|0  % .5e   % .5e   %.5e   %.5e   %.5e   %.3e   %.3e   %.3e   %.4f   %.4f\n",
                    m_result.info.init_admm_iter,
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
                fflush(stdout);
            }
            else
            {
                piqp_print("%4zd  % .5e   % .5e   %.5e   %.5e   %.5e   %.3e   %.3e   %.3e   %.4f   %.4f\n",
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
                fflush(stdout);
            }
        }

        if ((m_result.info.primal_res < m_settings.eps_abs || m_result.info.primal_res_rel < m_settings.eps_rel) &&
            (m_result.info.dual_res < m_settings.eps_abs || m_result.info.dual_res_rel < m_settings.eps_rel) &&
            (!m_settings.check_duality_gap || m_result.info.duality_gap < m_settings.eps_duality_gap_abs || m_result.info.duality_gap_rel < m_settings.eps_duality_gap_rel))
        {
            m_result.info.status = Status::PIQP_SOLVED;
            return m_result.info.status;
        }

        update_residuals_r();

        if (m_result.info.no_dual_update > (std::min)(static_cast<isize>(5), m_settings.reg_finetune_dual_update_threshold) &&
            m_result.info.primal_prox_inf > m_settings.infeasibility_threshold &&
            (m_result.info.primal_res_reg < m_settings.eps_abs || m_result.info.primal_res_reg_rel < m_settings.eps_rel))
        {
            m_result.info.status = Status::PIQP_PRIMAL_INFEASIBLE;
            return m_result.info.status;
        }

        if (m_result.info.no_primal_update > (std::min)(static_cast<isize>(5), m_settings.reg_finetune_primal_update_threshold) &&
            m_result.info.dual_prox_inf > m_settings.infeasibility_threshold &&
            (m_result.info.dual_res_reg < m_settings.eps_abs || m_result.info.dual_res_reg_rel < m_settings.eps_rel))
        {
            m_result.info.status = Status::PIQP_DUAL_INFEASIBLE;
            return m_result.info.status;
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

        if (m_settings.compute_timings)
        {
            timer.start();
        }
        bool regularization_changed = false;
        while (!m_kkt_system.update_scalings_and_factor(m_data, m_settings, m_enable_iterative_refinement,
                                                        m_result.info.rho, m_result.info.delta, m_result))
        {
            if (!m_enable_iterative_refinement)
            {
                m_enable_iterative_refinement = true;
                continue;
            }
            if (m_result.info.factor_retires < m_settings.max_factor_retires)
            {
                m_result.info.delta *= 100;
                m_result.info.rho *= 100;
                m_result.info.factor_retires++;
                m_result.info.reg_limit = (std::min)(10 * m_result.info.reg_limit, m_settings.eps_abs);
                regularization_changed = true;
                continue;
            }

            m_result.info.status = Status::PIQP_NUMERICS;
            return m_result.info.status;
        }
        m_result.info.factor_retires = 0;
        if (m_settings.compute_timings)
        {
            T kkt_factor_time = timer.stop();
            m_result.info.kkt_factor_time += kkt_factor_time;
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
            calculate_step(alpha_s, alpha_z);

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
            calculate_step(alpha_s, alpha_z);

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

            T mu_prev = m_result.info.mu;
            m_result.info.mu = calculate_mu();
            T mu_rate = (std::max)(T(0), (mu_prev - m_result.info.mu) / mu_prev);

            // ------------------ update regularization ------------------
            update_residuals_nr();

            if (m_result.info.dual_res < 0.95 * m_result.info.prev_dual_res ||
                (m_result.info.dual_res < m_settings.eps_abs || m_result.info.dual_res_rel < m_settings.eps_rel) ||
                (m_result.info.rho == m_settings.reg_finetune_lower_limit && m_result.info.dual_prox_inf < m_settings.infeasibility_threshold))
            {
                prox_vars.x = m_result.x;
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
                (m_result.info.delta == m_settings.reg_finetune_lower_limit && m_result.info.primal_prox_inf < m_settings.infeasibility_threshold))
            {
                prox_vars.y = m_result.y;
                prox_vars.z_l = m_result.z_l;
                prox_vars.z_u = m_result.z_u;
                nu_bl = z_bl;
                nu_bu = z_bu;
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

            // ------------------ update regularization ------------------
            update_residuals_nr();

            if (m_result.info.dual_res < 0.95 * m_result.info.prev_dual_res || (m_result.info.dual_res < m_settings.eps_abs || m_result.info.dual_res_rel < m_settings.eps_rel))
            {
                prox_vars.x = m_result.x;
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
                prox_vars.y = m_result.y;
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
T SolverBase<T, I, Preconditioner, MatrixType>::calculate_mu()
{
    return (m_result.s_l.dot(m_result.z_l)
            + m_result.s_u.dot(m_result.z_u)
            + m_result.s_bl.head(m_data.n_x_l).dot(m_result.z_bl.head(m_data.n_x_l))
            + m_result.s_bu.head(m_data.n_x_u).dot(m_result.z_bu.head(m_data.n_x_u)))
        / T(m_data.n_h_l + m_data.n_h_u + m_data.n_x_l + m_data.n_x_u);
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::calculate_step(T& alpha_s, T& alpha_z)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::calculate_step");

    alpha_s = T(1);
    alpha_z = T(1);

#ifdef PIQP_HAS_OPENMP
#pragma omp parallel
    {
    PIQP_TRACY_ZoneScopedN("piqp::Solver::calculate_step:parallel");

    #pragma omp for reduction(min:alpha_s,alpha_z)
#endif
    for (isize i = 0; i < m_data.m; i++)
    {
        if (step.s_l(i) < 0)
        {
            alpha_s = (std::min)(alpha_s, -m_result.s_l(i) / step.s_l(i));
        }
        if (step.s_u(i) < 0)
        {
            alpha_s = (std::min)(alpha_s, -m_result.s_u(i) / step.s_u(i));
        }
        if (step.z_l(i) < 0)
        {
            alpha_z = (std::min)(alpha_z, -m_result.z_l(i) / step.z_l(i));
        }
        if (step.z_u(i) < 0)
        {
            alpha_z = (std::min)(alpha_z, -m_result.z_u(i) / step.z_u(i));
        }
    }
#ifdef PIQP_HAS_OPENMP
    #pragma omp for reduction(min:alpha_s,alpha_z)
#endif
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        if (step.s_bl(i) < 0)
        {
            alpha_s = (std::min)(alpha_s, -m_result.s_bl(i) / step.s_bl(i));
        }
        if (step.z_bl(i) < 0)
        {
            alpha_z = (std::min)(alpha_z, -m_result.z_bl(i) / step.z_bl(i));
        }
    }
#ifdef PIQP_HAS_OPENMP
    #pragma omp for reduction(min:alpha_s,alpha_z)
#endif
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        if (step.s_bu(i) < 0)
        {
            alpha_s = (std::min)(alpha_s, -m_result.s_bu(i) / step.s_bu(i));
        }
        if (step.z_bu(i) < 0)
        {
            alpha_z = (std::min)(alpha_z, -m_result.z_bu(i) / step.z_bu(i));
        }
    }

#ifdef PIQP_HAS_OPENMP
    } // end of parallel region
#endif
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::update_residuals_nr()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::update_residuals_nr");

    using std::abs;

    // step is not used and we can use it as temporary storage
    Vec<T>& work_x = step.x;
    Vec<T>& work_z = step.z_l;

    // we calculate these term here first to be able to reuse temporary vectors
    // res_nr.y = -A * x
    // work_x = A^T * y
    m_kkt_system.eval_A_xn_and_AT_xt(m_data, T(-1), T(1), m_result.x, m_result.y, res_nr.y, work_x);
    // res_nr.z_u = -G * x
    // res_nr.z_l = G * x
    // work_x += G^T * (z_u - z_l)
    work_z.noalias() = m_result.z_u - m_result.z_l;
    Vec<T>& work_x_2 = res_nr.x; // use res_nr.x as temporary, gets overwritten in the next section
    m_kkt_system.eval_G_xn_and_GT_xt(m_data, T(1), T(1), m_result.x, work_z, res_nr.z_l, work_x_2);
    res_nr.z_u.noalias() = -res_nr.z_l;
    work_x.noalias() += work_x_2;

    // first part of dual residual and infeasibility calculation (used in cost calculation)
    m_kkt_system.eval_P_x(m_data, T(-1), m_result.x, res_nr.x);
    T dual_rel_norm = m_preconditioner.unscale_dual_res(res_nr.x).template lpNorm<Eigen::Infinity>();

    // calculate primal cost, dual cost, and duality gap
    T tmp = -m_result.x.dot(res_nr.x); // x'Px
    m_result.info.primal_obj = T(0.5) * tmp;
    m_result.info.dual_obj = -T(0.5) * tmp;
    T duality_gap_rel_norm = m_preconditioner.unscale_cost(abs(tmp));
    tmp = m_data.c.dot(m_result.x);
    m_result.info.primal_obj += tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(abs(tmp)));
    tmp = m_data.b.dot(m_result.y);
    m_result.info.dual_obj -= tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(abs(tmp)));
    tmp = -m_data.h_l.dot(m_result.z_l);
    m_result.info.dual_obj -= tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(abs(tmp)));
    tmp = m_data.h_u.dot(m_result.z_u);
    m_result.info.dual_obj -= tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(abs(tmp)));
    tmp = -m_data.x_l.head(m_data.n_x_l).dot(m_result.z_bl.head(m_data.n_x_l));
    m_result.info.dual_obj -= tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(abs(tmp)));
    tmp = m_data.x_u.head(m_data.n_x_u).dot(m_result.z_bu.head(m_data.n_x_u));
    m_result.info.dual_obj -= tmp;
    duality_gap_rel_norm = (std::max)(duality_gap_rel_norm, m_preconditioner.unscale_cost(abs(tmp)));

    m_result.info.duality_gap = abs(m_result.info.primal_obj - m_result.info.dual_obj);

    m_result.info.primal_obj = m_preconditioner.unscale_cost(m_result.info.primal_obj);
    m_result.info.dual_obj = m_preconditioner.unscale_cost(m_result.info.dual_obj);
    m_result.info.duality_gap = m_preconditioner.unscale_cost(m_result.info.duality_gap);
    m_result.info.duality_gap_rel = m_result.info.duality_gap / (std::max)(T(1), duality_gap_rel_norm);

    // dual residual and infeasibility calculation
    res_nr.x.noalias() -= m_data.c;
    dual_rel_norm = (std::max)(dual_rel_norm, m_preconditioner.unscale_dual_res(m_data.c).template lpNorm<Eigen::Infinity>());
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        work_x(idx) -= m_data.x_b_scaling(idx) * m_result.z_bl(i);
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        work_x(idx) += m_data.x_b_scaling(idx) * m_result.z_bu(i);
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
        res_nr.z_l(i) += -m_data.h_l(i) - m_result.s_l(i);
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_ineq(m_data.h_l)(i));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_ineq(m_result.s_l)(i));
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
        res_nr.z_u(i) += m_data.h_u(i) - m_result.s_u(i);
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_ineq(m_data.h_u)(i));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_ineq(m_result.s_u)(i));
        i++;
    }
    // zero out remaining residuals corresponding to infinite bounds
    while (i < m_data.m) {
        res_nr.z_u(i++) = T(0);
    }

    for (i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        res_nr.z_bl(i) = m_data.x_b_scaling(idx) * m_result.x(idx);
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(res_nr.z_bl(i), idx));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(m_data.x_l(i), idx));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(m_result.s_bl(i), idx));
    }
    res_nr.z_bl.head(m_data.n_x_l).noalias() += -m_data.x_l.head(m_data.n_x_l) - m_result.s_bl.head(m_data.n_x_l);

    for (i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        res_nr.z_bu(i) = -m_data.x_b_scaling(idx) * m_result.x(idx);
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(res_nr.z_bu(i), idx));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(m_data.x_u(i), idx));
        primal_rel_norm = (std::max)(primal_rel_norm, m_preconditioner.unscale_primal_res_b_i(m_result.s_bu(i), idx));
    }
    res_nr.z_bu.head(m_data.n_x_u).noalias() += m_data.x_u.head(m_data.n_x_u) - m_result.s_bu.head(m_data.n_x_u);

    m_result.info.prev_primal_res = m_result.info.primal_res;
    m_result.info.prev_dual_res = m_result.info.dual_res;

    m_result.info.primal_res = primal_res_nr();
    m_result.info.primal_res_rel = m_result.info.primal_res / (std::max)(T(1), primal_rel_norm);

    m_result.info.dual_res = dual_res_nr();
    m_result.info.dual_res_rel = m_result.info.dual_res / (std::max)(T(1), dual_rel_norm);
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
        inf = (std::max)(inf, std::abs(m_preconditioner.unscale_primal_res_b_i(res_nr.z_bl(i), m_data.x_l_idx(i))));
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        inf = (std::max)(inf, std::abs(m_preconditioner.unscale_primal_res_b_i(res_nr.z_bu(i), m_data.x_u_idx(i))));
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
        inf = (std::max)(inf, std::abs(m_preconditioner.unscale_primal_res_b_i(res.z_bl(i), m_data.x_l_idx(i))));
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        inf = (std::max)(inf, std::abs(m_preconditioner.unscale_primal_res_b_i(res.z_bu(i), m_data.x_u_idx(i))));
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
        inf = (std::max)(inf, std::abs(m_preconditioner.unscale_dual_b_i(prox_vars.z_bl(i) - m_result.z_bl(i), m_data.x_l_idx(i))));
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        inf = (std::max)(inf, std::abs(m_preconditioner.unscale_dual_b_i(prox_vars.z_bu(i) - m_result.z_bu(i), m_data.x_u_idx(i))));
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
T SolverBase<T, I, Preconditioner, MatrixType>::init_compute_mu()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::init_compute_mu");

    auto s_bl = m_result.s_bl.head(m_data.n_x_l);
    auto s_bu = m_result.s_bu.head(m_data.n_x_u);
    auto z_bl = m_result.z_bl.head(m_data.n_x_l);
    auto z_bu = m_result.z_bu.head(m_data.n_x_u);

    // Project s and z into the non-negative cone, then average complementarity
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

    T sum_mu = T(0);
    for (isize i = 0; i < m_data.n_h_l; i++)
    {
        Eigen::Index idx = m_data.h_l_idx(i);
        sum_mu += (m_result.s_l(idx) + delta_s) * (m_result.z_l(idx) + delta_z);
    }
    for (isize i = 0; i < m_data.n_h_u; i++)
    {
        Eigen::Index idx = m_data.h_u_idx(i);
        sum_mu += (m_result.s_u(idx) + delta_s) * (m_result.z_u(idx) + delta_z);
    }
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        sum_mu += (s_bl(i) + delta_s) * (z_bl(i) + delta_z);
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        sum_mu += (s_bu(i) + delta_s) * (z_bu(i) + delta_z);
    }
    return (std::max)(sum_mu / T(m_data.n_h_l + m_data.n_h_u + m_data.n_x_l + m_data.n_x_u), T(1e-10));
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::apply_smoothing(T sigma, T mu, const Variables<T>& s_kp1, const Variables<T>& z_k)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::apply_smoothing");

    for (isize i = 0; i < m_data.n_h_l; i++)
    {
        Eigen::Index idx = m_data.h_l_idx(i);
        nonneg_smoothing(sigma, mu, s_kp1.s_l(idx) - z_k.z_l(idx) / sigma, m_result.s_l(idx), m_result.z_l(idx));
    }
    for (isize i = 0; i < m_data.n_h_u; i++)
    {
        Eigen::Index idx = m_data.h_u_idx(i);
        nonneg_smoothing(sigma, mu, s_kp1.s_u(idx) - z_k.z_u(idx) / sigma, m_result.s_u(idx), m_result.z_u(idx));
    }
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        nonneg_smoothing(sigma, mu, s_kp1.s_bl(i) - z_k.z_bl(i) / sigma, m_result.s_bl(i), m_result.z_bl(i));
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        nonneg_smoothing(sigma, mu, s_kp1.s_bu(i) - z_k.z_bu(i) / sigma, m_result.s_bu(i), m_result.z_bu(i));
    }
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::init_cold_start()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::init_cold_start");

    // Infinite bounds have never active constraints.
    // Note that we set for all inactive constraints
    // z = 0 as well as s = 0. To be correct, we should
    // have s = Inf. But keeping it zero, simplifies calculations
    // down the road (e.g. norm calculations).
    // We are correcting this at the end when we restore the solution.
    m_result.x.setZero();
    m_result.y.setZero();
    m_result.s_l.setZero();
    m_result.s_u.setZero();
    m_result.s_bl.setZero();
    m_result.s_bu.setZero();
    m_result.z_l.setZero();
    m_result.z_u.setZero();
    m_result.z_bl.setZero();
    m_result.z_bu.setZero();

    T s_guess = m_settings.cold_start_alpha;
    T z_guess = m_settings.cold_start_alpha;
    for (isize i = 0; i < m_data.n_h_l; i++)
    {
        Eigen::Index idx = m_data.h_l_idx(i);
        m_result.s_l(idx) = s_guess;
        m_result.z_l(idx) = z_guess;
    }
    for (isize i = 0; i < m_data.n_h_u; i++)
    {
        Eigen::Index idx = m_data.h_u_idx(i);
        m_result.s_u(idx) = s_guess;
        m_result.z_u(idx) = z_guess;
    }
    m_result.s_bl.head(m_data.n_x_l).setConstant(s_guess);
    m_result.s_bu.head(m_data.n_x_u).setConstant(s_guess);
    m_result.z_bl.head(m_data.n_x_l).setConstant(z_guess);
    m_result.z_bu.head(m_data.n_x_u).setConstant(z_guess);

    return init_from_guess(m_settings.cold_start_sigma);
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::init_warm_start()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::init_warm_start");

    // Scale the unscaled warm-start point into the current preconditioner's space.
    pack_dual();
    scale_results();

    if (m_warm_start_from_solve)
    {
        update_residuals_nr();

        T mu0 = (std::max)(m_result.info.primal_res, m_result.info.dual_res);
        mu0 = (std::max)(mu0, T(1e-10));

        apply_smoothing(T(1), mu0, m_result, m_result);
        m_result.info.mu = calculate_mu();

        // Aggressive proximal parameters when the point is near-optimal
        if (mu0 <= T(1))
        {
            m_result.info.rho = (std::max)(mu0, m_settings.reg_lower_limit);
            m_result.info.delta = (std::max)(mu0, m_settings.reg_lower_limit);
        }

        return Status::PIQP_UNSOLVED;
    }

    // Compute slacks: s_l = Gx - h_l, s_u = h_u - Gx
    m_result.s_l.setZero();
    m_result.s_u.setZero();
    if (m_data.m > 0)
    {
        // Use step.z_l as temporary for Gx
        step.z_l.noalias() = m_data.GT.transpose() * m_result.x;
        for (isize i = 0; i < m_data.n_h_l; i++)
        {
            Eigen::Index idx = m_data.h_l_idx(i);
            m_result.s_l(idx) = step.z_l(idx) - m_data.h_l(idx);
        }
        for (isize i = 0; i < m_data.n_h_u; i++)
        {
            Eigen::Index idx = m_data.h_u_idx(i);
            m_result.s_u(idx) = m_data.h_u(idx) - step.z_l(idx);
        }
    }
    // s_bl = x_b_scaling * x - x_l, s_bu = x_u - x_b_scaling * x
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        m_result.s_bl(i) = m_data.x_b_scaling(idx) * m_result.x(idx) - m_data.x_l(i);
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        m_result.s_bu(i) = m_data.x_u(i) - m_data.x_b_scaling(idx) * m_result.x(idx);
    }

    return init_from_guess(m_settings.warm_start_sigma);
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
Status SolverBase<T, I, Preconditioner, MatrixType>::init_from_guess(T sigma)
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::init_from_guess");

    step.s_l.setZero();
    step.s_u.setZero();
    step.z_l.setZero();
    step.z_u.setZero();
    for (isize i = 0; i < m_data.n_h_l; i++)
    {
        Eigen::Index idx = m_data.h_l_idx(i);
        step.s_l(idx) = T(1);
        step.z_l(idx) = sigma;
    }
    for (isize i = 0; i < m_data.n_h_u; i++)
    {
        Eigen::Index idx = m_data.h_u_idx(i);
        step.s_u(idx) = T(1);
        step.z_u(idx) = sigma;
    }
    step.s_bl.head(m_data.n_x_l).setConstant(T(1));
    step.s_bu.head(m_data.n_x_u).setConstant(T(1));
    step.z_bl.head(m_data.n_x_l).setConstant(sigma);
    step.z_bu.head(m_data.n_x_u).setConstant(sigma);

    Timer<T> timer;
    if (m_settings.compute_timings)
    {
        timer.start();
    }
    while (!m_kkt_system.update_scalings_and_factor(m_data, m_settings, m_enable_iterative_refinement,
                                                    m_result.info.rho, m_result.info.delta, step))
    {
        if (!m_enable_iterative_refinement)
        {
            m_enable_iterative_refinement = true;
        }
        else if (m_result.info.factor_retires < m_settings.max_factor_retires)
        {
            m_result.info.delta *= 100;
            m_result.info.rho *= 100;
            m_result.info.factor_retires++;
            m_result.info.reg_limit = (std::min)(10 * m_result.info.reg_limit, m_settings.eps_abs);
        }
        else
        {
            m_result.info.status = Status::PIQP_NUMERICS;
            return m_result.info.status;
        }
    }
    m_result.info.factor_retires = 0;
    if (m_settings.compute_timings)
    {
        T kkt_factor_time = timer.stop();
        m_result.info.kkt_factor_time += kkt_factor_time;
    }

    // Initialize nu^0 = z^0
    prox_vars.z_l = m_result.z_l;
    prox_vars.z_u = m_result.z_u;
    prox_vars.z_bl = m_result.z_bl;
    prox_vars.z_bu = m_result.z_bu;

    T mu = T(0);
    m_result.info.init_admm_iter = 0;

    for (isize admm_iter = 0; admm_iter < m_settings.max_init_admm_iter; admm_iter++)
    {
        m_result.info.init_admm_iter++;

        // At this point:
        //   m_result.{x, y}  = x^k, y^k
        //   m_result.s_*     = s_tilde^k
        //   m_result.z_*     = z^k (ADMM consensus dual)
        //   prox_vars.z_*    = nu^k (augmented Lagrangian dual)

        // Save z^k into step.z_* for apply_smoothing later
        step.z_l = m_result.z_l;
        step.z_u = m_result.z_u;
        step.z_bl = m_result.z_bl;
        step.z_bu = m_result.z_bu;

        // Build RHS
        res.x = -m_data.c + m_result.info.rho * m_result.x;
        res.y = m_data.b - m_result.info.delta * m_result.y;

        res.z_l.setZero();
        res.z_u.setZero();
        for (isize i = 0; i < m_data.n_h_l; i++)
        {
            Eigen::Index idx = m_data.h_l_idx(i);
            res.z_l(idx) = -m_data.h_l(idx) - m_result.info.delta * prox_vars.z_l(idx);
        }
        for (isize i = 0; i < m_data.n_h_u; i++)
        {
            Eigen::Index idx = m_data.h_u_idx(i);
            res.z_u(idx) = m_data.h_u(idx) - m_result.info.delta * prox_vars.z_u(idx);
        }
        res.z_bl.head(m_data.n_x_l) = -m_data.x_l.head(m_data.n_x_l) - m_result.info.delta * prox_vars.z_bl.head(m_data.n_x_l);
        res.z_bu.head(m_data.n_x_u) = m_data.x_u.head(m_data.n_x_u) - m_result.info.delta * prox_vars.z_bu.head(m_data.n_x_u);

        res.s_l = sigma * m_result.s_l + m_result.z_l;
        res.s_u = sigma * m_result.s_u + m_result.z_u;
        res.s_bl = sigma * m_result.s_bl + m_result.z_bl;
        res.s_bu = sigma * m_result.s_bu + m_result.z_bu;

        // Solve KKT system (reusing factorization)
        if (m_settings.compute_timings)
        {
            timer.start();
        }
        m_kkt_system.solve(m_data, m_settings, res, m_result);
        if (m_settings.compute_timings)
        {
            T kkt_solve_time = timer.stop();
            m_result.info.kkt_solve_time += kkt_solve_time;
        }

        if (m_data.m + m_data.n_x_l + m_data.n_x_u > 0)
        {
            // Compute mu from the first iteration only (fixes the barrier scale)
            if (admm_iter == 0)
            {
                mu = init_compute_mu() * m_settings.init_mu_scale;
            }

            // Save nu^{k+1} (in m_result.z_*) before smoothing overwrites it
            prox_vars.z_l = m_result.z_l;
            prox_vars.z_u = m_result.z_u;
            prox_vars.z_bl = m_result.z_bl;
            prox_vars.z_bu = m_result.z_bu;

            apply_smoothing(sigma, mu, m_result, step);
            m_result.info.mu = calculate_mu();

            // Check ADMM consensus residual: ||r_c|| = ||z^{k+1} - z^k|| / sigma
            if (admm_iter < m_settings.max_init_admm_iter - 1)
            {
                T consensus_res = T(0);
                for (isize i = 0; i < m_data.n_h_l; i++)
                {
                    Eigen::Index idx = m_data.h_l_idx(i);
                    consensus_res = (std::max)(consensus_res, std::abs(m_result.z_l(idx) - step.z_l(idx)));
                }
                for (isize i = 0; i < m_data.n_h_u; i++)
                {
                    Eigen::Index idx = m_data.h_u_idx(i);
                    consensus_res = (std::max)(consensus_res, std::abs(m_result.z_u(idx) - step.z_u(idx)));
                }
                for (isize i = 0; i < m_data.n_x_l; i++)
                {
                    consensus_res = (std::max)(consensus_res, std::abs(m_result.z_bl(i) - step.z_bl(i)));
                }
                for (isize i = 0; i < m_data.n_x_u; i++)
                {
                    consensus_res = (std::max)(consensus_res, std::abs(m_result.z_bu(i) - step.z_bu(i)));
                }
                consensus_res /= sigma;

                if (consensus_res <= mu)
                {
                    break;
                }
            }
        }
        else
        {
            break;
        }
    }

    return Status::PIQP_UNSOLVED;
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::scale_results()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::scale_results");

    m_result.x = m_preconditioner.scale_primal(m_result.x);
    m_result.y = m_preconditioner.scale_dual_eq(m_result.y);
    m_result.z_l = m_preconditioner.scale_dual_ineq(m_result.z_l);
    m_result.z_u = m_preconditioner.scale_dual_ineq(m_result.z_u);
    m_result.s_l = m_preconditioner.scale_slack_ineq(m_result.s_l);
    m_result.s_u = m_preconditioner.scale_slack_ineq(m_result.s_u);
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        m_result.z_bl(i) = m_preconditioner.scale_dual_b_i(m_result.z_bl(i), idx);
        m_result.s_bl(i) = m_preconditioner.scale_slack_b_i(m_result.s_bl(i), idx);
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        m_result.z_bu(i) = m_preconditioner.scale_dual_b_i(m_result.z_bu(i), idx);
        m_result.s_bu(i) = m_preconditioner.scale_slack_b_i(m_result.s_bu(i), idx);
    }
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::unscale_results()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::unscale_results");

    m_result.x = m_preconditioner.unscale_primal(m_result.x);
    m_result.y = m_preconditioner.unscale_dual_eq(m_result.y);
    m_result.z_l = m_preconditioner.unscale_dual_ineq(m_result.z_l);
    m_result.z_u = m_preconditioner.unscale_dual_ineq(m_result.z_u);
    m_result.s_l = m_preconditioner.unscale_slack_ineq(m_result.s_l);
    m_result.s_u = m_preconditioner.unscale_slack_ineq(m_result.s_u);
    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        m_result.z_bl(i) = m_preconditioner.unscale_dual_b_i(m_result.z_bl(i), idx);
        m_result.s_bl(i) = m_preconditioner.unscale_slack_b_i(m_result.s_bl(i), idx);
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        m_result.z_bu(i) = m_preconditioner.unscale_dual_b_i(m_result.z_bu(i), idx);
        m_result.s_bu(i) = m_preconditioner.unscale_slack_b_i(m_result.s_bu(i), idx);
    }
}

template<typename T, typename I, typename Preconditioner, int MatrixType>
void SolverBase<T, I, Preconditioner, MatrixType>::pack_dual()
{
    PIQP_TRACY_ZoneScopedN("piqp::Solver::pack_dual");

    for (isize i = 0; i < m_data.n_x_l; i++)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        std::swap(m_result.z_bl(i), m_result.z_bl(idx));
        std::swap(m_result.s_bl(i), m_result.s_bl(idx));
    }
    for (isize i = 0; i < m_data.n_x_u; i++)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        std::swap(m_result.z_bu(i), m_result.z_bu(idx));
        std::swap(m_result.s_bu(i), m_result.s_bu(idx));
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
    for (isize i = m_data.n_x_l - 1; i >= 0; i--)
    {
        Eigen::Index idx = m_data.x_l_idx(i);
        std::swap(m_result.z_bl(i), m_result.z_bl(idx));
        std::swap(m_result.s_bl(i), m_result.s_bl(idx));
    }
    for (isize i = m_data.n_x_u - 1; i >= 0; i--)
    {
        Eigen::Index idx = m_data.x_u_idx(i);
        std::swap(m_result.z_bu(i), m_result.z_bu(idx));
        std::swap(m_result.s_bu(i), m_result.s_bu(idx));
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
