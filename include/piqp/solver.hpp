// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SOLVER_HPP
#define PIQP_SOLVER_HPP

#include <cstdio>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/timer.hpp"
#include "piqp/results.hpp"
#include "piqp/settings.hpp"
#include "piqp/dense/data.hpp"
#include "piqp/dense/preconditioner.hpp"
#include "piqp/dense/kkt.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/preconditioner.hpp"
#include "piqp/sparse/kkt.hpp"
#include "piqp/utils/optional.hpp"

namespace piqp
{

enum SolverMatrixType
{
    PIQP_DENSE = 0,
    PIQP_SPARSE = 1
};

template<typename Derived, typename T, typename I, typename Preconditioner, int MatrixType, int Mode = KKTMode::KKT_FULL>
class SolverBase
{
protected:
    using DataType = typename std::conditional<MatrixType == PIQP_DENSE, dense::Data<T>, sparse::Data<T, I>>::type;
    using KKTType = typename std::conditional<MatrixType == PIQP_DENSE, dense::KKT<T>, sparse::KKT<T, I, Mode>>::type;

    Timer<T> m_timer;
    Result<T> m_result;
    Settings<T> m_settings;
    DataType m_data;
    Preconditioner m_preconditioner;
    KKTType m_kkt;

    bool m_kkt_init_state = false;
    bool m_setup_done = false;
    bool m_enable_iterative_refinement = false;

    // residuals
    Vec<T> rx;
    Vec<T> ry;
    Vec<T> rz;
    Vec<T> rz_lb;
    Vec<T> rz_ub;
    Vec<T> rs;
    Vec<T> rs_lb;
    Vec<T> rs_ub;

    // non-regularized residuals
    Vec<T> rx_nr;
    Vec<T> ry_nr;
    Vec<T> rz_nr;
    Vec<T> rz_lb_nr;
    Vec<T> rz_ub_nr;

    // primal and dual steps
    Vec<T> dx;
    Vec<T> dy;
    Vec<T> dz;
    Vec<T> dz_lb;
    Vec<T> dz_ub;
    Vec<T> ds;
    Vec<T> ds_lb;
    Vec<T> ds_ub;

public:
    SolverBase() : m_kkt(m_data, m_settings) {};

    Settings<T>& settings() { return m_settings; }

    const Result<T>& result() const { return m_result; }

    Status solve()
    {
        if (m_settings.verbose)
        {
            piqp_print("----------------------------------------------------------\n");
            piqp_print("                           PIQP                           \n");
            piqp_print("                    (c) Roland Schwan                     \n");
            piqp_print("   Ecole Polytechnique Federale de Lausanne (EPFL) 2023   \n");
            piqp_print("----------------------------------------------------------\n");
            if (MatrixType == PIQP_DENSE)
            {
                piqp_print("dense backend\n");
                piqp_print("variables n = %zd\n", m_data.n);
                piqp_print("equality constraints p = %zd\n", m_data.p);
                piqp_print("inequality constraints m = %zd\n", m_data.m);
            }
            else
            {
                piqp_print("sparse backend\n");
                piqp_print("variables n = %zd, nzz(P upper triangular) = %zd\n", m_data.n, m_data.non_zeros_P_utri());
                piqp_print("equality constraints p = %zd, nnz(A) = %zd\n", m_data.p, m_data.non_zeros_A());
                piqp_print("inequality constraints m = %zd, nnz(G) = %zd\n", m_data.m, m_data.non_zeros_G());
            }
            piqp_print("variable lower bounds n_lb = %zd\n", m_data.n_lb);
            piqp_print("variable upper bounds n_ub = %zd\n", m_data.n_ub);
            piqp_print("\n");
            piqp_print("iter  prim_obj       dual_obj       duality_gap   prim_inf      dual_inf      rho         delta       mu          p_step   d_step\n");
        }

        if (m_settings.compute_timings)
        {
            m_timer.start();
        }

        Status status = solve_impl();

        unscale_results();
        restore_box_dual();

        if (m_settings.compute_timings)
        {
            T solve_time = m_timer.stop();
            m_result.info.solve_time = solve_time;
            m_result.info.run_time += solve_time;
        }

        if (m_settings.verbose)
        {
            piqp_print("\n");
            piqp_print("status:               %s\n", status_to_string(status));
            piqp_print("number of iterations: %zd\n", m_result.info.iter);
            piqp_print("objective:            %.5e\n", m_result.info.primal_obj);
            if (m_settings.compute_timings)
            {
                piqp_print("total run time:       %.3es\n", m_result.info.run_time);
                piqp_print("  setup time:         %.3es\n", m_result.info.setup_time);
                piqp_print("  update time:        %.3es\n", m_result.info.update_time);
                piqp_print("  solve time:         %.3es\n", m_result.info.solve_time);
            }
        }

        return status;
    }

protected:
    template<typename MatType>
    void setup_impl(const MatType& P,
                    const CVecRef<T>& c,
                    const MatType& A,
                    const CVecRef<T>& b,
                    const MatType& G,
                    const CVecRef<T>& h,
                    const optional<CVecRef<T>>& x_lb,
                    const optional<CVecRef<T>>& x_ub)
    {
        if (m_settings.compute_timings)
        {
            m_timer.start();
        }

        m_data.n = P.rows();
        m_data.p = A.rows();
        m_data.m = G.rows();

        if (P.rows() != m_data.n || P.cols() != m_data.n) { piqp_eprint("P must be square"); return; }
        if (A.rows() != m_data.p || A.cols() != m_data.n) { piqp_eprint("A must have correct dimensions"); return; }
        if (G.rows() != m_data.m || G.cols() != m_data.n) { piqp_eprint("G must have correct dimensions"); return; }
        if (c.size() != m_data.n) { piqp_eprint("c must have correct dimensions"); return; }
        if (b.size() != m_data.p) { piqp_eprint("b must have correct dimensions"); return; }
        if (h.size() != m_data.m) { piqp_eprint("h must have correct dimensions"); return; }
        if (x_lb.has_value() && x_lb->size() != m_data.n) { piqp_eprint("x_lb must have correct dimensions"); return; }
        if (x_ub.has_value() && x_ub->size() != m_data.n) { piqp_eprint("x_ub must have correct dimensions"); return; }

        m_data.P_utri = P.template triangularView<Eigen::Upper>();
        m_data.AT = A.transpose();
        m_data.GT = G.transpose();
        m_data.c = c;
        m_data.b = b;
        m_data.h = h.cwiseMin(PIQP_INF).cwiseMax(-PIQP_INF);

        m_data.x_lb_idx.resize(m_data.n);
        m_data.x_ub_idx.resize(m_data.n);
        m_data.x_lb_scaling = Vec<T>::Constant(m_data.n, T(1));
        m_data.x_ub_scaling = Vec<T>::Constant(m_data.n, T(1));
        m_data.x_lb_n.resize(m_data.n);
        m_data.x_ub.resize(m_data.n);

        setup_lb_data(x_lb);
        setup_ub_data(x_ub);

        init_workspace();

        m_preconditioner.init(m_data);
        m_preconditioner.scale_data(m_data,
                                    false,
                                    m_settings.preconditioner_scale_cost,
                                    m_settings.preconditioner_iter);

        m_kkt.init(m_result.info.rho, m_result.info.delta);
        m_kkt_init_state = true;

        m_setup_done = true;

        m_enable_iterative_refinement = m_settings.iterative_refinement_always_enabled;

        if (m_settings.compute_timings)
        {
            T setup_time = m_timer.stop();
            m_result.info.setup_time = setup_time;
            m_result.info.run_time += setup_time;
        }
    }

    void setup_lb_data(const optional<CVecRef<T>>& x_lb)
    {
        isize n_lb = 0;
        if (x_lb.has_value())
        {
            isize i_lb = 0;
            for (isize i = 0; i < m_data.n; i++)
            {
                if ((*x_lb)(i) > -PIQP_INF)
                {
                    n_lb += 1;
                    m_data.x_lb_n(i_lb) = -(*x_lb)(i);
                    m_data.x_lb_idx(i_lb) = i;
                    i_lb++;
                }
            }
        }
        m_data.n_lb = n_lb;
    }

    void setup_ub_data(const optional<CVecRef<T>>& x_ub)
    {
        isize n_ub = 0;
        if (x_ub.has_value())
        {
            isize i_ub = 0;
            for (isize i = 0; i < m_data.n; i++)
            {
                if ((*x_ub)(i) < PIQP_INF)
                {
                    n_ub += 1;
                    m_data.x_ub(i_ub) = (*x_ub)(i);
                    m_data.x_ub_idx(i_ub) = i;
                    i_ub++;
                }
            }
        }
        m_data.n_ub = n_ub;
    }

    void init_workspace()
    {
        // init result
        m_result.x.resize(m_data.n);
        m_result.y.resize(m_data.p);
        m_result.z.resize(m_data.m);
        m_result.z_lb.resize(m_data.n);
        m_result.z_ub.resize(m_data.n);
        m_result.s.resize(m_data.m);
        m_result.s_lb.resize(m_data.n);
        m_result.s_ub.resize(m_data.n);

        m_result.zeta.resize(m_data.n);
        m_result.lambda.resize(m_data.p);
        m_result.nu.resize(m_data.m);
        m_result.nu_lb.resize(m_data.n);
        m_result.nu_ub.resize(m_data.n);

        // init workspace
        m_result.info.rho = m_settings.rho_init;
        m_result.info.delta = m_settings.delta_init;
        m_result.info.setup_time = 0;
        m_result.info.update_time = 0;
        m_result.info.solve_time = 0;
        m_result.info.run_time = 0;

        rx.resize(m_data.n);
        ry.resize(m_data.p);
        rz.resize(m_data.m);
        rz_lb.resize(m_data.n);
        rz_ub.resize(m_data.n);
        rs.resize(m_data.m);
        rs_lb.resize(m_data.n);
        rs_ub.resize(m_data.n);

        rx_nr.resize(m_data.n);
        ry_nr.resize(m_data.p);
        rz_nr.resize(m_data.m);
        rz_lb_nr.resize(m_data.n);
        rz_ub_nr.resize(m_data.n);

        dx.resize(m_data.n);
        dy.resize(m_data.p);
        dz.resize(m_data.m);
        dz_lb.resize(m_data.n);
        dz_ub.resize(m_data.n);
        ds.resize(m_data.m);
        ds_lb.resize(m_data.n);
        ds_ub.resize(m_data.n);
    }

    Status solve_impl()
    {
        auto s_lb = m_result.s_lb.head(m_data.n_lb);
        auto s_ub = m_result.s_ub.head(m_data.n_ub);
        auto z_lb = m_result.z_lb.head(m_data.n_lb);
        auto z_ub = m_result.z_ub.head(m_data.n_ub);
        auto nu_lb = m_result.nu_lb.head(m_data.n_lb);
        auto nu_ub = m_result.nu_ub.head(m_data.n_ub);

        if (!m_setup_done)
        {
            piqp_eprint("Solver not setup yet");
            m_result.info.status = Status::PIQP_UNSOLVED;
            return m_result.info.status;
        }

        if (!m_settings.verify_settings())
        {
            m_result.info.status = Status::PIQP_INVALID_SETTINGS;
            return m_result.info.status;
        }

        m_result.info.status = Status::PIQP_UNSOLVED;
        m_result.info.iter = 0;
        m_result.info.reg_limit = m_settings.reg_lower_limit;
        m_result.info.factor_retires = 0;
        m_result.info.no_primal_update = 0;
        m_result.info.no_dual_update = 0;
        m_result.info.mu = 0;
        m_result.info.primal_step = 0;
        m_result.info.dual_step = 0;

        if (!m_kkt_init_state)
        {
            m_result.info.rho = m_settings.rho_init;
            m_result.info.delta = m_settings.delta_init;

            m_result.s.setConstant(1);
            s_lb.head(m_data.n_lb).setConstant(1);
            s_ub.head(m_data.n_ub).setConstant(1);
            m_result.z.setConstant(1);
            z_lb.head(m_data.n_lb).setConstant(1);
            z_ub.head(m_data.n_ub).setConstant(1);
            m_kkt.update_scalings(m_result.info.rho, m_result.info.delta,
                                  m_result.s, m_result.s_lb, m_result.s_ub,
                                  m_result.z, m_result.z_lb, m_result.z_ub);
        }

        while (!m_kkt.regularize_and_factorize(m_enable_iterative_refinement))
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
                m_result.info.reg_limit = std::min(10 * m_result.info.reg_limit, m_settings.eps_abs);
            }
            else
            {
                m_result.info.status = Status::PIQP_NUMERICS;
                return m_result.info.status;
            }
        }
        m_result.info.factor_retires = 0;

        rx = -m_data.c;
        // avoid unnecessary copies
        // ry = m_data.b;
        // rz = m_data.h;
        // rz_lb = m_data.x_lb;
        // rz_ub = m_data.x_ub;
        rs.setZero();
        rs_lb.setZero();
        rs_ub.setZero();
        m_kkt.solve(rx, m_data.b,
                    m_data.h, m_data.x_lb_n, m_data.x_ub,
                    rs, rs_lb, rs_ub,
                    m_result.x, m_result.y,
                    m_result.z, m_result.z_lb, m_result.z_ub,
                    m_result.s, m_result.s_lb, m_result.s_ub,
                    m_enable_iterative_refinement);

        if (m_data.m + m_data.n_lb + m_data.n_ub > 0)
        {
            T s_norm = T(0);
            s_norm = std::max(s_norm, m_result.s.template lpNorm<Eigen::Infinity>());
            s_norm = std::max(s_norm, s_lb.template lpNorm<Eigen::Infinity>());
            s_norm = std::max(s_norm, s_ub.template lpNorm<Eigen::Infinity>());
            if (s_norm <= 1e-4)
            {
                // 0.1 is arbitrary
                m_result.s.setConstant(0.1);
                s_lb.setConstant(0.1);
                s_ub.setConstant(0.1);
                m_result.z.setConstant(0.1);
                z_lb.setConstant(0.1);
                z_ub.setConstant(0.1);
            }

            T delta_s = T(0);
            if (m_data.m > 0) delta_s = std::max(delta_s, -T(1.5) * m_result.s.minCoeff());
            if (m_data.n_lb > 0) delta_s = std::max(delta_s, -T(1.5) * s_lb.minCoeff());
            if (m_data.n_ub > 0) delta_s = std::max(delta_s, -T(1.5) * s_ub.minCoeff());
            T delta_z = T(0);
            if (m_data.m > 0) delta_z = std::max(delta_z, -T(1.5) * m_result.z.minCoeff());
            if (m_data.n_lb > 0) delta_z = std::max(delta_z, -T(1.5) * z_lb.minCoeff());
            if (m_data.n_ub > 0) delta_z = std::max(delta_z, -T(1.5) * z_ub.minCoeff());
            T tmp_prod = (m_result.s.array() + delta_s).matrix().dot((m_result.z.array() + delta_z).matrix());
            tmp_prod += (s_lb.array() + delta_s).matrix().dot((z_lb.array() + delta_z).matrix());
            tmp_prod += (s_ub.array() + delta_s).matrix().dot((z_ub.array() + delta_z).matrix());
            T delta_s_bar = delta_s + (T(0.5) * tmp_prod) / (m_result.z.sum() + z_lb.sum() + z_ub.sum() + T(m_data.m + m_data.n_lb + m_data.n_ub) * delta_z);
            T delta_z_bar = delta_z + (T(0.5) * tmp_prod) / (m_result.s.sum() + s_lb.sum() + s_ub.sum() + T(m_data.m + m_data.n_lb + m_data.n_ub) * delta_s);

            m_result.s.array() += delta_s_bar;
            s_lb.array() += delta_s_bar;
            s_ub.array() += delta_s_bar;
            m_result.z.array() += delta_z_bar;
            z_lb.array() += delta_z_bar;
            z_ub.array() += delta_z_bar;

            m_result.info.mu = (m_result.s.dot(m_result.z) + s_lb.dot(z_lb) + s_ub.dot(z_ub) ) / T(m_data.m + m_data.n_lb + m_data.n_ub);
        }

        m_result.zeta = m_result.x;
        m_result.lambda = m_result.y;
        m_result.nu = m_result.z;
        nu_lb = z_lb;
        nu_ub = z_ub;

        while (m_result.info.iter < m_settings.max_iter)
        {
            if (m_result.info.iter == 0)
            {
                update_nr_residuals();
            }

            m_result.info.primal_inf = primal_inf_nr();
            m_result.info.dual_inf = dual_inf_nr();

            if (m_settings.verbose)
            {
                piqp_print("%3zd   % .5e   % .5e   %.5e   %.5e   %.5e   %.3e   %.3e   %.3e   %.4f   %.4f\n",
                        m_result.info.iter,
                        m_result.info.primal_obj,
                        m_result.info.dual_obj,
                        m_result.info.duality_gap,
                        m_result.info.primal_inf,
                        m_result.info.dual_inf,
                        m_result.info.rho,
                        m_result.info.delta,
                        m_result.info.mu,
                        m_result.info.primal_step,
                        m_result.info.dual_step);
            }

            if (m_result.info.primal_inf < m_settings.eps_abs + m_settings.eps_rel * m_result.info.primal_rel_inf &&
                m_result.info.dual_inf < m_settings.eps_abs + m_settings.eps_rel * m_result.info.dual_rel_inf &&
                (!m_settings.check_duality_gap || m_result.info.duality_gap < m_settings.eps_duality_gap_abs + m_settings.eps_duality_gap_rel * m_result.info.duality_gap_rel))
            {
                m_result.info.status = Status::PIQP_SOLVED;
                return m_result.info.status;
            }

            rx = rx_nr - m_result.info.rho * (m_result.x - m_result.zeta);
            ry = ry_nr - m_result.info.delta * (m_result.lambda - m_result.y);
            rz = rz_nr - m_result.info.delta * (m_result.nu - m_result.z);
            rz_lb.head(m_data.n_lb) = rz_lb_nr.head(m_data.n_lb) - m_result.info.delta * (nu_lb - z_lb);
            rz_ub.head(m_data.n_ub) = rz_ub_nr.head(m_data.n_ub) - m_result.info.delta * (nu_ub - z_ub);

            if (m_result.info.no_dual_update > std::min(isize(5), m_settings.reg_finetune_dual_update_threshold) &&
                primal_prox_inf() > 1e12 &&
                primal_inf_r() < m_settings.eps_abs + m_settings.eps_rel * m_result.info.primal_rel_inf)
            {
                m_result.info.status = Status::PIQP_PRIMAL_INFEASIBLE;
                return m_result.info.status;
            }

            if (m_result.info.no_primal_update > std::min(isize(5), m_settings.reg_finetune_primal_update_threshold) &&
                dual_prox_inf() > 1e12 &&
                dual_inf_r() < m_settings.eps_abs + m_settings.eps_rel * m_result.info.dual_rel_inf)
            {
                m_result.info.status = Status::PIQP_DUAL_INFEASIBLE;
                return m_result.info.status;
            }

            m_result.info.iter++;

            // avoid possibility of converging to a local minimum -> decrease the minimum regularization value
            if ((m_result.info.no_primal_update > m_settings.reg_finetune_primal_update_threshold &&
                 m_result.info.rho == m_result.info.reg_limit &&
                 m_result.info.reg_limit != m_settings.reg_finetune_lower_limit) ||
                (m_result.info.no_dual_update > m_settings.reg_finetune_dual_update_threshold &&
                 m_result.info.delta == m_result.info.reg_limit &&
                 m_result.info.reg_limit != m_settings.reg_finetune_lower_limit))
            {
                m_result.info.reg_limit = m_settings.reg_finetune_lower_limit;
                m_result.info.no_primal_update = 0;
                m_result.info.no_dual_update = 0;
            }

            m_kkt.update_scalings(m_result.info.rho, m_result.info.delta,
                                  m_result.s, m_result.s_lb, m_result.s_ub,
                                  m_result.z, m_result.z_lb, m_result.z_ub);

            if (!m_kkt.regularize_and_factorize(m_enable_iterative_refinement))
            {
                if (!m_enable_iterative_refinement)
                {
                    m_enable_iterative_refinement = true;
                    continue;
                }
                else if (m_result.info.factor_retires < m_settings.max_factor_retires)
                {
                    m_result.info.delta *= 100;
                    m_result.info.rho *= 100;
                    m_result.info.iter--;
                    m_result.info.factor_retires++;
                    m_result.info.reg_limit = std::min(10 * m_result.info.reg_limit, m_settings.eps_abs);
                    continue;
                }
                else
                {
                    m_result.info.status = Status::PIQP_NUMERICS;
                    return m_result.info.status;
                }
            }
            m_result.info.factor_retires = 0;

            if (m_data.m + m_data.n_lb + m_data.n_ub > 0)
            {
                // ------------------ predictor step ------------------
                rs.array() = -m_result.s.array() * m_result.z.array();
                rs_lb.head(m_data.n_lb).array() = -s_lb.array() * z_lb.array();
                rs_ub.head(m_data.n_ub).array() = -s_ub.array() * z_ub.array();

                m_kkt.solve(rx, ry, rz, rz_lb, rz_ub, rs, rs_lb, rs_ub,
                            dx, dy, dz, dz_lb, dz_ub, ds, ds_lb, ds_ub,
                            m_enable_iterative_refinement);

                // step in the non-negative orthant
                T alpha_s = T(1);
                T alpha_z = T(1);
                for (isize i = 0; i < m_data.m; i++)
                {
                    if (ds(i) < 0)
                    {
                        alpha_s = std::min(alpha_s, -m_result.s(i) / ds(i));
                    }
                    if (dz(i) < 0)
                    {
                        alpha_z = std::min(alpha_z, -m_result.z(i) / dz(i));
                    }
                }
                for (isize i = 0; i < m_data.n_lb; i++)
                {
                    if (ds_lb(i) < 0)
                    {
                        alpha_s = std::min(alpha_s, -m_result.s_lb(i) / ds_lb(i));
                    }
                    if (dz_lb(i) < 0)
                    {
                        alpha_z = std::min(alpha_z, -m_result.z_lb(i) / dz_lb(i));
                    }
                }
                for (isize i = 0; i < m_data.n_ub; i++)
                {
                    if (ds_ub(i) < 0)
                    {
                        alpha_s = std::min(alpha_s, -m_result.s_ub(i) / ds_ub(i));
                    }
                    if (dz_ub(i) < 0)
                    {
                        alpha_z = std::min(alpha_z, -m_result.z_ub(i) / dz_ub(i));
                    }
                }
                // avoid getting to close to the boundary
                alpha_s *= m_settings.tau;
                alpha_z *= m_settings.tau;

                m_result.info.sigma = (m_result.s + alpha_s * ds).dot(m_result.z + alpha_z * dz);
                m_result.info.sigma += (s_lb + alpha_s * ds_lb.head(m_data.n_lb)).dot(z_lb + alpha_z * dz_lb.head(m_data.n_lb));
                m_result.info.sigma += (s_ub + alpha_s * ds_ub.head(m_data.n_ub)).dot(z_ub + alpha_z * dz_ub.head(m_data.n_ub));
                m_result.info.sigma /= (m_result.info.mu * T(m_data.m + m_data.n_lb + m_data.n_ub));
                m_result.info.sigma = std::max(T(0), std::min(T(1), m_result.info.sigma));
                m_result.info.sigma = m_result.info.sigma * m_result.info.sigma * m_result.info.sigma;

                // ------------------ corrector step ------------------
                rs.array() += -ds.array() * dz.array() + m_result.info.sigma * m_result.info.mu;
                rs_lb.head(m_data.n_lb).array() += -ds_lb.head(m_data.n_lb).array() * dz_lb.head(m_data.n_lb).array() + m_result.info.sigma * m_result.info.mu;
                rs_ub.head(m_data.n_ub).array() += -ds_ub.head(m_data.n_ub).array() * dz_ub.head(m_data.n_ub).array() + m_result.info.sigma * m_result.info.mu;

                m_kkt.solve(rx, ry, rz, rz_lb, rz_ub, rs, rs_lb, rs_ub,
                            dx, dy, dz, dz_lb, dz_ub, ds, ds_lb, ds_ub,
                            m_enable_iterative_refinement);

                // step in the non-negative orthant
                alpha_s = T(1);
                alpha_z = T(1);
                for (isize i = 0; i < m_data.m; i++)
                {
                    if (ds(i) < 0)
                    {
                        alpha_s = std::min(alpha_s, -m_result.s(i) / ds(i));
                    }
                    if (dz(i) < 0)
                    {
                        alpha_z = std::min(alpha_z, -m_result.z(i) / dz(i));
                    }
                }
                for (isize i = 0; i < m_data.n_lb; i++)
                {
                    if (ds_lb(i) < 0)
                    {
                        alpha_s = std::min(alpha_s, -m_result.s_lb(i) / ds_lb(i));
                    }
                    if (dz_lb(i) < 0)
                    {
                        alpha_z = std::min(alpha_z, -m_result.z_lb(i) / dz_lb(i));
                    }
                }
                for (isize i = 0; i < m_data.n_ub; i++)
                {
                    if (ds_ub(i) < 0)
                    {
                        alpha_s = std::min(alpha_s, -m_result.s_ub(i) / ds_ub(i));
                    }
                    if (dz_ub(i) < 0)
                    {
                        alpha_z = std::min(alpha_z, -m_result.z_ub(i) / dz_ub(i));
                    }
                }
                // avoid getting to close to the boundary
                m_result.info.primal_step = alpha_s * m_settings.tau;
                m_result.info.dual_step = alpha_z * m_settings.tau;

                // ------------------ update ------------------
                m_result.x += m_result.info.primal_step * dx;
                m_result.y += m_result.info.dual_step * dy;
                m_result.z += m_result.info.dual_step * dz;
                z_lb += m_result.info.dual_step * dz_lb.head(m_data.n_lb);
                z_ub += m_result.info.dual_step * dz_ub.head(m_data.n_ub);
                m_result.s += m_result.info.primal_step * ds;
                s_lb += m_result.info.primal_step * ds_lb.head(m_data.n_lb);
                s_ub += m_result.info.primal_step * ds_ub.head(m_data.n_ub);

                T mu_prev = m_result.info.mu;
                m_result.info.mu = (m_result.s.dot(m_result.z) + s_lb.dot(z_lb) + s_ub.dot(z_ub) ) / T(m_data.m + m_data.n_lb + m_data.n_ub);
                T mu_rate = std::max(T(0), (mu_prev - m_result.info.mu) / mu_prev);

                // ------------------ update regularization ------------------
                update_nr_residuals();

                if (dual_inf_nr() < 0.95 * m_result.info.dual_inf || (m_result.info.rho == m_settings.reg_finetune_lower_limit && dual_prox_inf() < 1e2))
                {
                    m_result.zeta = m_result.x;
                    m_result.info.rho = std::max(m_result.info.reg_limit, (T(1) - mu_rate) * m_result.info.rho);
                }
                else
                {
                    m_result.info.no_primal_update++;
                    m_result.info.rho = std::max(m_result.info.reg_limit, (T(1) - 0.666 * mu_rate) * m_result.info.rho);
                }

                if (primal_inf_nr() < 0.95 * m_result.info.primal_inf || (m_result.info.delta == m_settings.reg_finetune_lower_limit && primal_prox_inf() < 1e2))
                {
                    m_result.lambda = m_result.y;
                    m_result.nu = m_result.z;
                    nu_lb = z_lb;
                    nu_ub = z_ub;
                    m_result.info.delta = std::max(m_result.info.reg_limit, (T(1) - mu_rate) * m_result.info.delta);
                }
                else
                {
                    m_result.info.no_dual_update++;
                    m_result.info.delta = std::max(m_result.info.reg_limit, (T(1) - 0.666 * mu_rate) * m_result.info.delta);
                }
            }
            else
            {
                // since there are no inequalities we can take full steps
                m_kkt.solve(rx, ry, rz, rz_lb, rz_ub, rs, rs_lb, rs_ub,
                            dx, dy, dz, dz_lb, dz_ub, ds, ds_lb, ds_ub,
                            m_enable_iterative_refinement);

                m_result.info.primal_step = T(1);
                m_result.info.dual_step = T(1);
                m_result.x += m_result.info.primal_step * dx;
                m_result.y += m_result.info.dual_step * dy;

                // ------------------ update regularization ------------------
                update_nr_residuals();

                if (dual_inf_nr() < 0.95 * m_result.info.dual_inf)
                {
                    m_result.zeta = m_result.x;
                    m_result.info.rho = std::max(m_result.info.reg_limit, 0.1 * m_result.info.rho);
                }
                else
                {
                    m_result.info.no_primal_update++;
                    m_result.info.rho = std::max(m_result.info.reg_limit, 0.5 * m_result.info.rho);
                }

                if (primal_inf_nr() < 0.95 * m_result.info.primal_inf)
                {
                    m_result.lambda = m_result.y;
                    m_result.info.delta = std::max(m_result.info.reg_limit, 0.1 * m_result.info.delta);
                }
                else
                {
                    m_result.info.no_dual_update++;
                    m_result.info.delta = std::max(m_result.info.reg_limit, 0.5 * m_result.info.delta);
                }
            }
        }

        m_result.info.status = Status::PIQP_MAX_ITER_REACHED;
        return m_result.info.status;
    }

    void update_nr_residuals()
    {
        // first part of dual residual and infeasibility calculation (used in cost calculation)
        rx_nr.noalias() = -m_data.P_utri * m_result.x;
        rx_nr.noalias() -= m_data.P_utri.transpose().template triangularView<Eigen::StrictlyLower>() * m_result.x;
        m_result.info.dual_rel_inf = m_preconditioner.unscale_dual_res(rx_nr).template lpNorm<Eigen::Infinity>();

        // calculate primal cost, dual cost, and duality gap
        T tmp = -m_result.x.dot(rx_nr); // x'Px
        m_result.info.primal_obj = T(0.5) * tmp;
        m_result.info.dual_obj = -T(0.5) * tmp;
        m_result.info.duality_gap_rel = m_preconditioner.unscale_cost(std::abs(tmp));
        tmp = m_data.c.dot(m_result.x);
        m_result.info.primal_obj += tmp;
        m_result.info.duality_gap_rel = std::max(m_result.info.duality_gap_rel, m_preconditioner.unscale_cost(std::abs(tmp)));
        tmp = m_data.b.dot(m_result.y);
        m_result.info.dual_obj -= tmp;
        m_result.info.duality_gap_rel = std::max(m_result.info.duality_gap_rel, m_preconditioner.unscale_cost(std::abs(tmp)));
        tmp = m_data.h.dot(m_result.z);
        m_result.info.dual_obj -= tmp;
        m_result.info.duality_gap_rel = std::max(m_result.info.duality_gap_rel, m_preconditioner.unscale_cost(std::abs(tmp)));
        tmp = m_data.x_lb_n.head(m_data.n_lb).dot(m_result.z_lb.head(m_data.n_lb));
        m_result.info.dual_obj -= tmp;
        m_result.info.duality_gap_rel = std::max(m_result.info.duality_gap_rel, m_preconditioner.unscale_cost(std::abs(tmp)));
        tmp = m_data.x_ub.head(m_data.n_ub).dot(m_result.z_ub.head(m_data.n_ub));
        m_result.info.dual_obj -= tmp;
        m_result.info.duality_gap_rel = std::max(m_result.info.duality_gap_rel, m_preconditioner.unscale_cost(std::abs(tmp)));

        m_result.info.duality_gap = std::abs(m_result.info.primal_obj - m_result.info.dual_obj);

        m_result.info.primal_obj = m_preconditioner.unscale_cost(m_result.info.primal_obj);
        m_result.info.dual_obj = m_preconditioner.unscale_cost(m_result.info.dual_obj);
        m_result.info.duality_gap = m_preconditioner.unscale_cost(m_result.info.duality_gap);

        // dual residual and infeasibility calculation
        rx_nr.noalias() -= m_data.c;
        m_result.info.dual_rel_inf = std::max(m_result.info.dual_rel_inf, m_preconditioner.unscale_dual_res(m_data.c).template lpNorm<Eigen::Infinity>());
        dx.noalias() = m_data.AT * m_result.y; // use dx as a temporary
        dx.noalias() += m_data.GT * m_result.z;
        for (isize i = 0; i < m_data.n_lb; i++)
        {
            dx(m_data.x_lb_idx(i)) -= m_data.x_lb_scaling(i) * m_result.z_lb(i);
        }
        for (isize i = 0; i < m_data.n_ub; i++)
        {
            dx(m_data.x_ub_idx(i)) += m_data.x_ub_scaling(i) * m_result.z_ub(i);
        }
        m_result.info.dual_rel_inf = std::max(m_result.info.dual_rel_inf, m_preconditioner.unscale_dual_res(dx).template lpNorm<Eigen::Infinity>());
        rx_nr.noalias() -= dx;

        // primal residual and infeasibility calculation
        ry_nr.noalias() = -m_data.AT.transpose() * m_result.x;
        m_result.info.primal_rel_inf = m_preconditioner.unscale_primal_res_eq(ry_nr).template lpNorm<Eigen::Infinity>();
        ry_nr.noalias() += m_data.b;
        m_result.info.primal_rel_inf = std::max(m_result.info.primal_rel_inf, m_preconditioner.unscale_primal_res_eq(m_data.b).template lpNorm<Eigen::Infinity>());

        rz_nr.noalias() = -m_data.GT.transpose() * m_result.x;
        m_result.info.primal_rel_inf = std::max(m_result.info.primal_rel_inf, m_preconditioner.unscale_primal_res_ineq(rz_nr).template lpNorm<Eigen::Infinity>());
        rz_nr.noalias() += m_data.h - m_result.s;
        m_result.info.primal_rel_inf = std::max(m_result.info.primal_rel_inf, m_preconditioner.unscale_primal_res_ineq(m_data.h).template lpNorm<Eigen::Infinity>());
        m_result.info.primal_rel_inf = std::max(m_result.info.primal_rel_inf, m_preconditioner.unscale_primal_res_ineq(m_result.s).template lpNorm<Eigen::Infinity>());

        for (isize i = 0; i < m_data.n_lb; i++)
        {
            rz_lb_nr(i) = m_data.x_lb_scaling(i) * m_result.x(m_data.x_lb_idx(i));
        }
        m_result.info.primal_rel_inf = std::max(m_result.info.primal_rel_inf, m_preconditioner.unscale_primal_res_lb(rz_lb_nr.head(m_data.n_lb)).template lpNorm<Eigen::Infinity>());
        rz_lb_nr.head(m_data.n_lb).noalias() += m_data.x_lb_n.head(m_data.n_lb) - m_result.s_lb.head(m_data.n_lb);
        m_result.info.primal_rel_inf = std::max(m_result.info.primal_rel_inf, m_preconditioner.unscale_primal_res_lb(m_data.x_lb_n.head(m_data.n_lb)).template lpNorm<Eigen::Infinity>());
        m_result.info.primal_rel_inf = std::max(m_result.info.primal_rel_inf, m_preconditioner.unscale_primal_res_lb(m_result.s_lb.head(m_data.n_lb)).template lpNorm<Eigen::Infinity>());

        for (isize i = 0; i < m_data.n_ub; i++)
        {
            rz_ub_nr(i) = -m_data.x_ub_scaling(i) * m_result.x(m_data.x_ub_idx(i));
        }
        m_result.info.primal_rel_inf = std::max(m_result.info.primal_rel_inf, m_preconditioner.unscale_primal_res_ub(rz_ub_nr.head(m_data.n_ub)).template lpNorm<Eigen::Infinity>());
        rz_ub_nr.head(m_data.n_ub).noalias() += m_data.x_ub.head(m_data.n_ub) - m_result.s_ub.head(m_data.n_ub);
        m_result.info.primal_rel_inf = std::max(m_result.info.primal_rel_inf, m_preconditioner.unscale_primal_res_ub(m_data.x_ub.head(m_data.n_ub)).template lpNorm<Eigen::Infinity>());
        m_result.info.primal_rel_inf = std::max(m_result.info.primal_rel_inf, m_preconditioner.unscale_primal_res_ub(m_result.s_ub.head(m_data.n_ub)).template lpNorm<Eigen::Infinity>());
    }

    T primal_inf_nr()
    {
        T inf = m_preconditioner.unscale_primal_res_eq(ry_nr).template lpNorm<Eigen::Infinity>();
        inf = std::max(inf, m_preconditioner.unscale_primal_res_ineq(rz_nr).template lpNorm<Eigen::Infinity>());
        inf = std::max(inf, m_preconditioner.unscale_primal_res_lb(rz_lb_nr.head(m_data.n_lb)).template lpNorm<Eigen::Infinity>());
        inf = std::max(inf, m_preconditioner.unscale_primal_res_ub(rz_ub_nr.head(m_data.n_ub)).template lpNorm<Eigen::Infinity>());
        return inf;
    }

    T primal_inf_r()
    {
        T inf = m_preconditioner.unscale_primal_res_eq(ry).template lpNorm<Eigen::Infinity>();
        inf = std::max(inf, m_preconditioner.unscale_primal_res_ineq(rz).template lpNorm<Eigen::Infinity>());
        inf = std::max(inf, m_preconditioner.unscale_primal_res_lb(rz_lb.head(m_data.n_lb)).template lpNorm<Eigen::Infinity>());
        inf = std::max(inf, m_preconditioner.unscale_primal_res_ub(rz_ub.head(m_data.n_ub)).template lpNorm<Eigen::Infinity>());
        return inf;
    }

    T primal_prox_inf()
    {
        T inf = m_preconditioner.unscale_dual_eq(m_result.lambda - m_result.y).template lpNorm<Eigen::Infinity>();
        inf = std::max(inf, m_preconditioner.unscale_dual_ineq(m_result.nu - m_result.z).template lpNorm<Eigen::Infinity>());
        inf = std::max(inf, m_preconditioner.unscale_dual_lb(m_result.nu_lb.head(m_data.n_lb) - m_result.z_lb.head(m_data.n_lb)).template lpNorm<Eigen::Infinity>());
        inf = std::max(inf, m_preconditioner.unscale_dual_ub(m_result.nu_ub.head(m_data.n_ub) - m_result.z_ub.head(m_data.n_ub)).template lpNorm<Eigen::Infinity>());
        return inf;
    }

    T dual_inf_nr()
    {
        return m_preconditioner.unscale_dual_res(rx_nr).template lpNorm<Eigen::Infinity>();
    }

    T dual_inf_r()
    {
        return m_preconditioner.unscale_dual_res(rx).template lpNorm<Eigen::Infinity>();
    }

    T dual_prox_inf()
    {
        return m_preconditioner.unscale_primal(m_result.x - m_result.zeta).template lpNorm<Eigen::Infinity>();
    }

    void restore_box_dual()
    {
        m_result.z_lb.tail(m_data.n - m_data.n_lb).setZero();
        m_result.z_ub.tail(m_data.n - m_data.n_ub).setZero();
        m_result.s_lb.tail(m_data.n - m_data.n_lb).array() = std::numeric_limits<T>::infinity();
        m_result.s_ub.tail(m_data.n - m_data.n_ub).array() = std::numeric_limits<T>::infinity();
        m_result.nu_lb.tail(m_data.n - m_data.n_lb).setZero();
        m_result.nu_ub.tail(m_data.n - m_data.n_ub).setZero();
        for (isize i = m_data.n_lb - 1; i >= 0; i--)
        {
            std::swap(m_result.z_lb(i), m_result.z_lb(m_data.x_lb_idx(i)));
            std::swap(m_result.s_lb(i), m_result.s_lb(m_data.x_lb_idx(i)));
            std::swap(m_result.nu_lb(i), m_result.nu_lb(m_data.x_lb_idx(i)));
        }
        for (isize i = m_data.n_ub - 1; i >= 0; i--)
        {
            std::swap(m_result.z_ub(i), m_result.z_ub(m_data.x_ub_idx(i)));
            std::swap(m_result.s_ub(i), m_result.s_ub(m_data.x_ub_idx(i)));
            std::swap(m_result.nu_ub(i), m_result.nu_ub(m_data.x_ub_idx(i)));
        }
    }

    void unscale_results()
    {
        m_result.x = m_preconditioner.unscale_primal(m_result.x);
        m_result.y = m_preconditioner.unscale_dual_eq(m_result.y);
        m_result.z = m_preconditioner.unscale_dual_ineq(m_result.z);
        m_result.z_lb.head(m_data.n_lb) = m_preconditioner.unscale_dual_lb(m_result.z_lb.head(m_data.n_lb));
        m_result.z_ub.head(m_data.n_ub) = m_preconditioner.unscale_dual_ub(m_result.z_ub.head(m_data.n_ub));
        m_result.s = m_preconditioner.unscale_slack_ineq(m_result.s);
        m_result.s_lb.head(m_data.n_lb) = m_preconditioner.unscale_slack_lb(m_result.s_lb.head(m_data.n_lb));
        m_result.s_ub.head(m_data.n_ub) = m_preconditioner.unscale_slack_ub(m_result.s_ub.head(m_data.n_ub));
        m_result.zeta = m_preconditioner.unscale_primal(m_result.zeta);
        m_result.lambda = m_preconditioner.unscale_dual_eq(m_result.lambda);
        m_result.nu = m_preconditioner.unscale_dual_ineq(m_result.nu);
        m_result.nu_lb.head(m_data.n_lb) = m_preconditioner.unscale_dual_lb(m_result.nu_lb.head(m_data.n_lb));
        m_result.nu_ub.head(m_data.n_ub) = m_preconditioner.unscale_dual_ub(m_result.nu_ub.head(m_data.n_ub));
    }
};

template<typename T, typename Preconditioner = dense::RuizEquilibration<T>>
class DenseSolver : public SolverBase<DenseSolver<T, Preconditioner>, T, int, Preconditioner, PIQP_DENSE, KKTMode::KKT_FULL>
{
public:
    void setup(const CMatRef<T>& P,
               const CVecRef<T>& c,
               const CMatRef<T>& A,
               const CVecRef<T>& b,
               const CMatRef<T>& G,
               const CVecRef<T>& h,
               const optional<CVecRef<T>>& x_lb = nullopt,
               const optional<CVecRef<T>>& x_ub = nullopt)
    {
        this->setup_impl(P, c, A, b, G, h, x_lb, x_ub);
    }

    void update(const optional<CMatRef<T>>& P = nullopt,
                const optional<CVecRef<T>>& c = nullopt,
                const optional<CMatRef<T>>& A = nullopt,
                const optional<CVecRef<T>>& b = nullopt,
                const optional<CMatRef<T>>& G = nullopt,
                const optional<CVecRef<T>>& h = nullopt,
                const optional<CVecRef<T>>& x_lb = nullopt,
                const optional<CVecRef<T>>& x_ub = nullopt,
                bool reuse_preconditioner = true)
    {
        if (!this->m_setup_done)
        {
            piqp_eprint("Solver not setup yet");
            return;
        }

        if (this->m_settings.compute_timings)
        {
            this->m_timer.start();
        }

        this->m_preconditioner.unscale_data(this->m_data);

        int update_options = KKTUpdateOptions::KKT_UPDATE_NONE;

        if (P.has_value())
        {
            if (P->rows() != this->m_data.n || P->cols() != this->m_data.n) { piqp_eprint("P has wrong dimensions"); return; }
            this->m_data.P_utri = P->template triangularView<Eigen::Upper>();

            update_options |= KKTUpdateOptions::KKT_UPDATE_P;
        }

        if (A.has_value())
        {
            if (A->rows() != this->m_data.p || A->cols() != this->m_data.n) { piqp_eprint("A has wrong dimensions"); return; }
            this->m_data.AT = A->transpose();

            update_options |= KKTUpdateOptions::KKT_UPDATE_A;
        }

        if (G.has_value())
        {
            if (G->rows() != this->m_data.m || G->cols() != this->m_data.n) { piqp_eprint("G has wrong dimensions"); return; }
            this->m_data.GT = G->transpose();

            update_options |= KKTUpdateOptions::KKT_UPDATE_G;
        }

        if (c.has_value())
        {
            if (c->size() != this->m_data.n) { piqp_eprint("c has wrong dimensions"); return; }
            this->m_data.c = *c;
        }

        if (b.has_value())
        {
            if (b->size() != this->m_data.p) { piqp_eprint("b has wrong dimensions"); return; }
            this->m_data.b = *b;
        }

        if (h.has_value())
        {
            if (h->size() != this->m_data.m) { piqp_eprint("h has wrong dimensions"); return; }
            this->m_data.h = (*h).cwiseMin(PIQP_INF).cwiseMax(-PIQP_INF);
        }

        if (x_lb.has_value() && x_lb->size() != this->m_data.n) { piqp_eprint("x_lb has wrong dimensions"); return; }
        if (x_ub.has_value() && x_ub->size() != this->m_data.n) { piqp_eprint("x_ub has wrong dimensions"); return; }
        if (x_lb.has_value()) { this->setup_lb_data(x_lb); }
        if (x_ub.has_value()) { this->setup_ub_data(x_ub); }

        this->m_preconditioner.scale_data(this->m_data,
                                          reuse_preconditioner,
                                          this->m_settings.preconditioner_scale_cost,
                                          this->m_settings.preconditioner_iter);

        this->m_kkt.update_data(update_options);

        if (this->m_settings.compute_timings)
        {
            T update_time = this->m_timer.stop();
            this->m_result.info.update_time = update_time;
            this->m_result.info.run_time += update_time;
        }
    }
};

template<typename T, typename I = int, int Mode = KKTMode::KKT_FULL, typename Preconditioner = sparse::RuizEquilibration<T, I>>
class SparseSolver : public SolverBase<SparseSolver<T, I, Mode, Preconditioner>, T, I, Preconditioner, PIQP_SPARSE, Mode>
{
public:
    void setup(const CSparseMatRef<T, I>& P,
               const CVecRef<T>& c,
               const CSparseMatRef<T, I>& A,
               const CVecRef<T>& b,
               const CSparseMatRef<T, I>& G,
               const CVecRef<T>& h,
               const optional<CVecRef<T>>& x_lb = nullopt,
               const optional<CVecRef<T>>& x_ub = nullopt)
    {
        this->setup_impl(P, c, A, b, G, h, x_lb, x_ub);
    }

    void update(const optional<CSparseMatRef<T, I>>& P = nullopt,
                const optional<CVecRef<T>>& c = nullopt,
                const optional<CSparseMatRef<T, I>>& A = nullopt,
                const optional<CVecRef<T>>& b = nullopt,
                const optional<CSparseMatRef<T, I>>& G = nullopt,
                const optional<CVecRef<T>>& h = nullopt,
                const optional<CVecRef<T>>& x_lb = nullopt,
                const optional<CVecRef<T>>& x_ub = nullopt,
                bool reuse_preconditioner = true)
    {
        if (!this->m_setup_done)
        {
            piqp_eprint("Solver not setup yet");
            return;
        }

        if (this->m_settings.compute_timings)
        {
            this->m_timer.start();
        }

        this->m_preconditioner.unscale_data(this->m_data);

        int update_options = KKTUpdateOptions::KKT_UPDATE_NONE;

        if (P.has_value())
        {
            if (P->rows() != this->m_data.n || P->cols() != this->m_data.n) { piqp_eprint("P has wrong dimensions"); return; }
            isize n = P->outerSize();
            for (isize j = 0; j < n; j++)
            {
                isize P_col_nnz = P->outerIndexPtr()[j + 1] - P->outerIndexPtr()[j];
                isize P_utri_col_nnz = this->m_data.P_utri.outerIndexPtr()[j + 1] - this->m_data.P_utri.outerIndexPtr()[j];
                if (P_col_nnz < P_utri_col_nnz) { piqp_eprint("P nonzeros missmatch"); return; }
                Eigen::Map<Vec<T>>(this->m_data.P_utri.valuePtr() + this->m_data.P_utri.outerIndexPtr()[j], P_utri_col_nnz) = Eigen::Map<const Vec<T>>(P->valuePtr() + P->outerIndexPtr()[j], P_utri_col_nnz);
            }

            update_options |= KKTUpdateOptions::KKT_UPDATE_P;
        }

        if (A.has_value())
        {
            if (A->rows() != this->m_data.p || A->cols() != this->m_data.n) { piqp_eprint("A has wrong dimensions"); return; }
            if (A->nonZeros() != this->m_data.AT.nonZeros()) { piqp_eprint("A nonzeros missmatch"); return; }
            sparse::transpose_no_allocation(*A, this->m_data.AT);

            update_options |= KKTUpdateOptions::KKT_UPDATE_A;
        }

        if (G.has_value())
        {
            if (G->rows() != this->m_data.m || G->cols() != this->m_data.n) { piqp_eprint("G has wrong dimensions"); return; }
            if (G->nonZeros() != this->m_data.GT.nonZeros()) { piqp_eprint("G nonzeros missmatch"); return; }
            sparse::transpose_no_allocation(*G, this->m_data.GT);

            update_options |= KKTUpdateOptions::KKT_UPDATE_G;
        }

        if (c.has_value())
        {
            if (c->size() != this->m_data.n) { piqp_eprint("c has wrong dimensions"); return; }
            this->m_data.c = *c;
        }

        if (b.has_value())
        {
            if (b->size() != this->m_data.p) { piqp_eprint("b has wrong dimensions"); return; }
            this->m_data.b = *b;
        }

        if (h.has_value())
        {
            if (h->size() != this->m_data.m) { piqp_eprint("h has wrong dimensions"); return; }
            this->m_data.h = (*h).cwiseMin(PIQP_INF).cwiseMax(-PIQP_INF);
        }

        if (x_lb.has_value() && x_lb->size() != this->m_data.n) { piqp_eprint("x_lb has wrong dimensions"); return; }
        if (x_ub.has_value() && x_ub->size() != this->m_data.n) { piqp_eprint("x_ub has wrong dimensions"); return; }
        if (x_lb.has_value()) { this->setup_lb_data(x_lb); }
        if (x_ub.has_value()) { this->setup_ub_data(x_ub); }

        this->m_preconditioner.scale_data(this->m_data,
                                          reuse_preconditioner,
                                          this->m_settings.preconditioner_scale_cost,
                                          this->m_settings.preconditioner_iter);

        this->m_kkt.update_data(update_options);

        if (this->m_settings.compute_timings)
        {
            T update_time = this->m_timer.stop();
            this->m_result.info.update_time = update_time;
            this->m_result.info.run_time += update_time;
        }
    }
};

} // namespace piqp

#endif //PIQP_SOLVER_HPP
