// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SOLVER_HPP
#define PIQP_SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/timer.hpp"
#include "piqp/results.hpp"
#include "piqp/settings.hpp"
#include "piqp/dense/data.hpp"
#include "piqp/dense/kkt.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/kkt.hpp"
#include "piqp/utils/optional.hpp"

namespace piqp
{

enum SolverMatrixType
{
    PIQP_DENSE = 0,
    PIQP_SPARSE = 1
};

template<typename Derived, typename T, typename I, int MatrixType, int Mode = KKTMode::KKT_FULL>
class SolverBase
{
protected:
    using DataType = typename std::conditional<MatrixType == PIQP_DENSE, dense::Data<T>, sparse::Data<T, I>>::type;
    using KKTType = typename std::conditional<MatrixType == PIQP_DENSE, dense::KKT<T>, sparse::KKT<T, I, Mode>>::type;

    Timer<T> m_timer;
    Result<T> m_result;
    Settings<T> m_settings;
    DataType m_data;
    KKTType m_kkt;

    bool m_kkt_dirty = true;
    bool m_setup_done = false;

    // residuals
    Vec<T> rx;
    Vec<T> ry;
    Vec<T> rz;
    Vec<T> rs;

    // non-regularized residuals
    Vec<T> rx_nr;
    Vec<T> ry_nr;
    Vec<T> rz_nr;

    // primal and dual steps
    Vec<T> dx;
    Vec<T> dy;
    Vec<T> dz;
    Vec<T> ds;

    T primal_rel_inf;
    T dual_rel_inf;

public:
    SolverBase() : m_kkt(m_data) {};

    Settings<T>& settings() { return m_settings; }

    const Result<T>& result() const { return m_result; }

    Status solve()
    {
        if (m_settings.verbose)
        {
            printf("----------------------------------------------------------\n");
            printf("                           PIQP                           \n");
            printf("           (c) Roland Schwan, Colin N. Jones              \n");
            printf("   École Polytechnique Fédérale de Lausanne (EPFL) 2023   \n");
            printf("----------------------------------------------------------\n");
            if (MatrixType == PIQP_DENSE)
            {
                printf("variables n = %ld\n", m_data.n);
                printf("equality constraints p = %ld\n", m_data.p);
                printf("inequality constraints m = %ld\n", m_data.m);
            }
            else
            {
                printf("variables n = %ld, nzz(P upper triangular) = %ld\n", m_data.n, m_data.non_zeros_P_utri());
                printf("equality constraints p = %ld, nnz(A) = %ld\n", m_data.p, m_data.non_zeros_A());
                printf("inequality constraints m = %ld, nnz(G) = %ld\n", m_data.m, m_data.non_zeros_G());
            }
            printf("\n");
            printf("iter  prim_cost      dual_cost      prim_inf      dual_inf      rho         delta       mu          prim_step   dual_step\n");
        }

        if (m_settings.compute_timings)
        {
            m_timer.start();
        }

        Status status = solve_impl();

        if (m_settings.compute_timings)
        {
            T solve_time = m_timer.stop();
            m_result.info.solve_time = solve_time;
            m_result.info.run_time += solve_time;
        }

        if (m_settings.verbose)
        {
            printf("\n");
            printf("status:               %s\n", status_to_string(status));
            printf("number of iterations: %ld\n", m_result.info.iter);
            if (m_settings.compute_timings)
            {
                printf("total run time:       %.3es\n", m_result.info.run_time);
                printf("  setup time:         %.3es\n", m_result.info.setup_time);
                printf("  update time:        %.3es\n", m_result.info.update_time);
                printf("  solve time:         %.3es\n", m_result.info.solve_time);
            }
        }

        return status;
    }

protected:
    void init_workspace()
    {
        // init result
        m_result.x.resize(m_data.n);
        m_result.y.resize(m_data.p);
        m_result.z.resize(m_data.m);
        m_result.s.resize(m_data.m);

        m_result.zeta.resize(m_data.n);
        m_result.lambda.resize(m_data.p);
        m_result.nu.resize(m_data.m);

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
        rs.resize(m_data.m);

        rx_nr.resize(m_data.n);
        ry_nr.resize(m_data.p);
        rz_nr.resize(m_data.m);

        dx.resize(m_data.n);
        dy.resize(m_data.p);
        dz.resize(m_data.m);
        ds.resize(m_data.m);

        m_kkt.init(m_result.info.rho, m_result.info.delta);
        m_kkt_dirty = false;
        m_setup_done = true;
    }

    template<typename MatType>
    void setup_impl(const MatType& P,
                    const CVecRef<T>& c,
                    const MatType& A,
                    const CVecRef<T>& b,
                    const MatType& G,
                    const CVecRef<T>& h)
    {
        if (this->m_settings.compute_timings)
        {
            this->m_timer.start();
        }

        this->m_data.n = P.rows();
        this->m_data.p = A.rows();
        this->m_data.m = G.rows();

        eigen_assert(P.rows() == this->m_data.n && P.cols() == this->m_data.n && "P must be square");
        eigen_assert(A.rows() == this->m_data.p && A.cols() == this->m_data.n && "A must have correct dimensions");
        eigen_assert(G.rows() == this->m_data.m && G.cols() == this->m_data.n && "G must have correct dimensions");
        eigen_assert(c.size() == this->m_data.n && "c must have correct dimensions");
        eigen_assert(b.size() == this->m_data.p && "b must have correct dimensions");
        eigen_assert(h.size() == this->m_data.m && "h must have correct dimensions");

        this->m_data.P_utri = P.template triangularView<Eigen::Upper>();
        this->m_data.AT = A.transpose();
        this->m_data.GT = G.transpose();
        this->m_data.c = c;
        this->m_data.b = b;
        this->m_data.h = h;

        this->init_workspace();

        if (this->m_settings.compute_timings)
        {
            T setup_time = this->m_timer.stop();
            this->m_result.info.setup_time = setup_time;
            this->m_result.info.run_time += setup_time;
        }
    }

    Status solve_impl()
    {
        if (!m_setup_done)
        {
            eigen_assert(false && "Solver not setup yet");
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

        if (m_kkt_dirty)
        {
            m_result.info.rho = m_settings.rho_init;
            m_result.info.delta = m_settings.delta_init;

            m_result.s.setConstant(1);
            m_result.z.setConstant(1);
            m_kkt.update_scalings(m_result.info.rho, m_result.info.delta, m_result.s, m_result.z);
        }

        while (!m_kkt.factorize())
        {
            if (m_result.info.factor_retires < m_settings.max_factor_retires)
            {
                m_result.info.delta *= 100;
                m_result.info.rho *= 100;
                m_result.info.factor_retires++;
                m_result.info.reg_limit = std::min(10 * m_result.info.reg_limit, m_settings.feas_tol_abs);
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
        rs.setZero();
        m_kkt.solve(rx, m_data.b, m_data.h, rs, m_result.x, m_result.y, m_result.z, m_result.s);

        if (m_data.m > 0)
        {
            // not sure if this is necessary
            if (m_result.s.template lpNorm<Eigen::Infinity>() <= 1e-4)
            {
                // 0.1 is arbitrary
                m_result.s.setConstant(0.1);
                m_result.z.setConstant(0.1);
            }

            T delta_s = std::max(T(-1.5) * m_result.s.minCoeff(), T(0));
            T delta_z = std::max(T(-1.5) * m_result.z.minCoeff(), T(0));
            T tmp_prod = (m_result.s.array() + delta_s).matrix().dot((m_result.z.array() + delta_z).matrix());
            T delta_s_bar = delta_s + (T(0.5) * tmp_prod) / (m_result.z.sum() + m_data.m * delta_z);
            T delta_z_bar = delta_z + (T(0.5) * tmp_prod) / (m_result.s.sum() + m_data.m * delta_s);

            m_result.s.array() += delta_s_bar;
            m_result.z.array() += delta_z_bar;

            m_result.info.mu = m_result.s.dot(m_result.z) / m_data.m;
        }

        m_result.zeta = m_result.x;
        m_result.lambda = m_result.y;
        m_result.nu = m_result.z;

        while (m_result.info.iter < m_settings.max_iter)
        {
            if (m_result.info.iter == 0)
            {
                update_nr_residuals();
            }

            m_result.info.primal_inf = std::max(ry_nr.template lpNorm<Eigen::Infinity>(),
                                                rz_nr.template lpNorm<Eigen::Infinity>());
            m_result.info.dual_inf = rx_nr.template lpNorm<Eigen::Infinity>();

            if (m_settings.verbose)
            {
                // use rx as temporary variables
                rx.noalias() = m_data.P_utri * m_result.x;
                rx.noalias() += m_data.P_utri.transpose().template triangularView<Eigen::StrictlyLower>() * m_result.x;
                T xPx_half = T(0.5) * m_result.x.dot(rx);

                T primal_cost = xPx_half + m_data.c.dot(m_result.x);
                T dual_cost = -xPx_half - m_data.b.dot(m_result.y) - m_data.h.dot(m_result.z);

                printf("%3ld   % .5e   % .5e   %.5e   %.5e   %.3e   %.3e   %.3e   %.3e   %.3e\n",
                       m_result.info.iter,
                       primal_cost,
                       dual_cost,
                       m_result.info.primal_inf,
                       m_result.info.dual_inf,
                       m_result.info.rho,
                       m_result.info.delta,
                       m_result.info.mu,
                       m_result.info.primal_step,
                       m_result.info.dual_step);
            }

            rx = rx_nr - m_result.info.rho * (m_result.x - m_result.zeta);
            ry = ry_nr - m_result.info.delta * (m_result.lambda - m_result.y);
            rz = rz_nr - m_result.info.delta * (m_result.nu - m_result.z);

            if (m_result.info.primal_inf < m_settings.feas_tol_abs + m_settings.feas_tol_rel * primal_rel_inf &&
                m_result.info.dual_inf < m_settings.feas_tol_abs + m_settings.feas_tol_rel * dual_rel_inf &&
                m_result.info.mu < m_settings.dual_tol)
            {
                m_result.info.status = Status::PIQP_SOLVED;
                return m_result.info.status;
            }

            if (m_result.info.no_dual_update > 5 &&
                std::max((m_result.lambda - m_result.y).template lpNorm<Eigen::Infinity>(), (m_result.nu - m_result.z).template lpNorm<Eigen::Infinity>()) > 1e10 &&
                std::max(ry.template lpNorm<Eigen::Infinity>(), rz.template lpNorm<Eigen::Infinity>()) < m_settings.feas_tol_abs)
            {
                m_result.info.status = Status::PIQP_PRIMAL_INFEASIBLE;
                return m_result.info.status;
            }

            if (m_result.info.no_primal_update > 5 &&
                (m_result.x - m_result.zeta).template lpNorm<Eigen::Infinity>() > 1e10 &&
                rx.template lpNorm<Eigen::Infinity>() < m_settings.feas_tol_abs)
            {
                m_result.info.status = Status::PIQP_DUAL_INFEASIBLE;
                return m_result.info.status;
            }

            m_result.info.iter++;

            // avoid possibility of converging to a local minimum -> decrease the minimum regularization value
            if ((m_result.info.no_primal_update > 5 && m_result.info.rho == m_result.info.reg_limit && m_result.info.reg_limit != 5e-13) ||
                (m_result.info.no_dual_update > 5 && m_result.info.delta == m_result.info.reg_limit && m_result.info.reg_limit != 5e-13))
            {
                m_result.info.reg_limit = 5e-13;
                m_result.info.no_primal_update = 0;
                m_result.info.no_dual_update = 0;
            }

            m_kkt.update_scalings(m_result.info.rho, m_result.info.delta, m_result.s, m_result.z);
            m_kkt_dirty = true;
            bool kkt_success = m_kkt.factorize();
            if (!kkt_success)
            {
                if (m_result.info.factor_retires < m_settings.max_factor_retires)
                {
                    m_result.info.delta *= 100;
                    m_result.info.rho *= 100;
                    m_result.info.iter--;
                    m_result.info.factor_retires++;
                    m_result.info.reg_limit = std::min(10 * m_result.info.reg_limit, m_settings.feas_tol_abs);
                    continue;
                }
                else
                {
                    m_result.info.status = Status::PIQP_NUMERICS;
                    return m_result.info.status;
                }
            }
            m_result.info.factor_retires = 0;

            if (m_data.m > 0)
            {
                // ------------------ predictor step ------------------
                rs.array() = -m_result.s.array() * m_result.z.array();

                m_kkt.solve(rx, ry, rz, rs, dx, dy, dz, ds);

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
                // avoid getting to close to the boundary
                alpha_s *= m_settings.tau;
                alpha_z *= m_settings.tau;

                m_result.info.sigma = (m_result.s + alpha_s * ds).dot(m_result.z + alpha_z * dz) / (m_result.info.mu * m_data.m);
                m_result.info.sigma = m_result.info.sigma * m_result.info.sigma * m_result.info.sigma;

                // ------------------ corrector step ------------------
                rs.array() += -ds.array() * dz.array() + m_result.info.sigma * m_result.info.mu;

                m_kkt.solve(rx, ry, rz, rs, dx, dy, dz, ds);

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
                // avoid getting to close to the boundary
                m_result.info.primal_step = alpha_s * m_settings.tau;
                m_result.info.dual_step = alpha_z * m_settings.tau;

                // ------------------ update ------------------
                m_result.x += m_result.info.primal_step * dx;
                m_result.y += m_result.info.dual_step * dy;
                m_result.z += m_result.info.dual_step * dz;
                m_result.s += m_result.info.primal_step * ds;

                T mu_prev = m_result.info.mu;
                m_result.info.mu = m_result.s.dot(m_result.z) / m_data.m;
                T mu_rate = std::abs(mu_prev - m_result.info.mu) / mu_prev;

                // ------------------ update regularization ------------------
                update_nr_residuals();

                if (rx_nr.template lpNorm<Eigen::Infinity>() < 0.95 * m_result.info.dual_inf)
                {
                    m_result.zeta = m_result.x;
                    m_result.info.rho = std::max(m_result.info.reg_limit, (T(1) - mu_rate) * m_result.info.rho);
                }
                else
                {
                    m_result.info.no_primal_update++;
                    m_result.info.rho = std::max(m_result.info.reg_limit, (T(1) - 0.666 * mu_rate) * m_result.info.rho);
                }

                if (std::max(ry_nr.template lpNorm<Eigen::Infinity>(), rz_nr.template lpNorm<Eigen::Infinity>()) < 0.95 * m_result.info.primal_inf)
                {
                    m_result.lambda = m_result.y;
                    m_result.nu = m_result.z;
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
                m_kkt.solve(rx, ry, rz, rs, dx, dy, dz, ds);

                m_result.info.primal_step = T(1);
                m_result.info.dual_step = T(1);
                m_result.x += m_result.info.primal_step * dx;
                m_result.y += m_result.info.dual_step * dy;

                // ------------------ update regularization ------------------
                update_nr_residuals();

                if (rx_nr.template lpNorm<Eigen::Infinity>() < 0.95 * m_result.info.dual_inf)
                {
                    m_result.zeta = m_result.x;
                    m_result.info.rho = std::max(m_result.info.reg_limit, 0.1 * m_result.info.rho);
                }
                else
                {
                    m_result.info.no_primal_update++;
                    m_result.info.rho = std::max(m_result.info.reg_limit, 0.5 * m_result.info.rho);
                }

                if (std::max(ry_nr.template lpNorm<Eigen::Infinity>(), rz_nr.template lpNorm<Eigen::Infinity>()) < 0.95 * m_result.info.primal_inf)
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
        rx_nr.noalias() = -m_data.P_utri * m_result.x;
        rx_nr.noalias() -= m_data.P_utri.transpose().template triangularView<Eigen::StrictlyLower>() * m_result.x;
        dual_rel_inf = rx_nr.template lpNorm<Eigen::Infinity>();
        rx_nr.noalias() -= m_data.c;
        dx.noalias() = m_data.AT * m_result.y; // use dx as a temporary
        dual_rel_inf = std::max(dual_rel_inf, dx.template lpNorm<Eigen::Infinity>());
        rx_nr.noalias() -= dx;
        dx.noalias() = m_data.GT * m_result.z; // use dx as a temporary
        dual_rel_inf = std::max(dual_rel_inf, dx.template lpNorm<Eigen::Infinity>());
        rx_nr.noalias() -= dx;

        ry_nr.noalias() = -m_data.AT.transpose() * m_result.x;
        primal_rel_inf = ry_nr.template lpNorm<Eigen::Infinity>();
        ry_nr.noalias() += m_data.b;
        primal_rel_inf = std::max(primal_rel_inf, m_data.b.template lpNorm<Eigen::Infinity>());

        rz_nr.noalias() = -m_data.GT.transpose() * m_result.x;
        primal_rel_inf = std::max(primal_rel_inf, rz_nr.template lpNorm<Eigen::Infinity>());
        rz_nr.noalias() += m_data.h - m_result.s;
        primal_rel_inf = std::max(primal_rel_inf, m_data.h.template lpNorm<Eigen::Infinity>());
    }
};

template<typename T>
class DenseSolver : public SolverBase<DenseSolver<T>, T, int, PIQP_DENSE, KKTMode::KKT_FULL>
{
public:
    void setup(const CMatRef<T>& P,
               const CVecRef<T>& c,
               const CMatRef<T>& A,
               const CVecRef<T>& b,
               const CMatRef<T>& G,
               const CVecRef<T>& h)
    {
        this->setup_impl(P, c, A, b, G, h);
    }

    void update(const optional<CMatRef<T>> P,
                const optional<CVecRef<T>> c,
                const optional<CMatRef<T>> A,
                const optional<CVecRef<T>> b,
                const optional<CMatRef<T>> G,
                const optional<CVecRef<T>> h)
    {
        if (!this->m_setup_done)
        {
            eigen_assert(false && "Solver not setup yet");
            return;
        }

        if (this->m_settings.compute_timings)
        {
            this->m_timer.start();
        }

        int update_options = KKTUpdateOptions::KKT_UPDATE_NONE;

        if (P.has_value())
        {
            eigen_assert(P->rows() == this->m_data.n && P->cols() == this->m_data.n && "P has wrong dimensions");
            this->m_data.P_utri = P->template triangularView<Eigen::Upper>();

            update_options |= KKTUpdateOptions::KKT_UPDATE_P;
        }

        if (A.has_value())
        {
            eigen_assert(A->rows() == this->m_data.p && A->cols() == this->m_data.n && "A has wrong dimensions");
            this->m_data.AT = A->transpose();

            update_options |= KKTUpdateOptions::KKT_UPDATE_A;
        }

        if (G.has_value())
        {
            eigen_assert(G->rows() == this->m_data.m && G->cols() == this->m_data.n && "G has wrong dimensions");
            this->m_data.GT = G->transpose();

            update_options |= KKTUpdateOptions::KKT_UPDATE_G;
        }

        if (c.has_value())
        {
            eigen_assert(c->size() == this->m_data.n && "c has wrong dimensions");
            this->m_data.c = *c;
        }

        if (b.has_value())
        {
            eigen_assert(b->size() == this->m_data.p && "b has wrong dimensions");
            this->m_data.b = *b;
        }

        if (h.has_value())
        {
            eigen_assert(h->size() == this->m_data.m && "h has wrong dimensions");
            this->m_data.h = *h;
        }

        this->m_kkt.update_data(update_options);

        if (this->m_settings.compute_timings)
        {
            T update_time = this->m_timer.stop();
            this->m_result.info.update_time = update_time;
            this->m_result.info.run_time += update_time;
        }
    }
};

template<typename T, typename I, int Mode = KKTMode::KKT_FULL>
class SparseSolver : public SolverBase<SparseSolver<T, I, Mode>, T, I, PIQP_SPARSE, Mode>
{
public:
    void setup(const SparseMat<T, I>& P,
               const CVecRef<T>& c,
               const SparseMat<T, I>& A,
               const CVecRef<T>& b,
               const SparseMat<T, I>& G,
               const CVecRef<T>& h)
    {
        this->setup_impl(P, c, A, b, G, h);
    }

    void update(const optional<SparseMat<T, I>> P,
                const optional<CVecRef<T>> c,
                const optional<SparseMat<T, I>> A,
                const optional<CVecRef<T>> b,
                const optional<SparseMat<T, I>> G,
                const optional<CVecRef<T>> h)
    {
        if (!this->m_setup_done)
        {
            eigen_assert(false && "Solver not setup yet");
            return;
        }

        if (this->m_settings.compute_timings)
        {
            this->m_timer.start();
        }

        int update_options = KKTUpdateOptions::KKT_UPDATE_NONE;

        if (P.has_value())
        {
            const SparseMat<T, I>& P_ = *P;

            eigen_assert(P_.rows() == this->m_data.n && P_.cols() == this->m_data.n && "P has wrong dimensions");
            isize n = P_.outerSize();
            for (isize j = 0; j < n; j++)
            {
                PIQP_MAYBE_UNUSED isize P_col_nnz = P_.outerIndexPtr()[j + 1] - P_.outerIndexPtr()[j];
                isize P_utri_col_nnz = this->m_data.P_utri.outerIndexPtr()[j + 1] - this->m_data.P_utri.outerIndexPtr()[j];
                eigen_assert(P_col_nnz >= P_utri_col_nnz && "P nonzeros missmatch");
                Eigen::Map<Vec<T>>(this->m_data.P_utri.valuePtr() + this->m_data.P_utri.outerIndexPtr()[j], P_utri_col_nnz) = Eigen::Map<const Vec<T>>(P_.valuePtr() + P_.outerIndexPtr()[j], P_utri_col_nnz);
            }

            update_options |= KKTUpdateOptions::KKT_UPDATE_P;
        }

        if (A.has_value())
        {
            const SparseMat<T, I>& A_ = *A;

            eigen_assert(A_.rows() == this->m_data.p && A_.cols() == this->m_data.n && "A has wrong dimensions");
            eigen_assert(A_.nonZeros() == this->m_data.AT.nonZeros() && "A nonzeros missmatch");
            sparse::transpose_no_allocation(A_, this->m_data.AT);

            update_options |= KKTUpdateOptions::KKT_UPDATE_A;
        }

        if (G.has_value())
        {
            const SparseMat<T, I>& G_ = *G;

            eigen_assert(G_.rows() == this->m_data.m && G_.cols() == this->m_data.n && "G has wrong dimensions");
            eigen_assert(G_.nonZeros() == this->m_data.GT.nonZeros() && "G nonzeros missmatch");
            sparse::transpose_no_allocation(G_, this->m_data.GT);

            update_options |= KKTUpdateOptions::KKT_UPDATE_G;
        }

        if (c.has_value())
        {
            eigen_assert(c->size() == this->m_data.n && "c has wrong dimensions");
            this->m_data.c = *c;
        }

        if (b.has_value())
        {
            eigen_assert(b->size() == this->m_data.p && "b has wrong dimensions");
            this->m_data.b = *b;
        }

        if (h.has_value())
        {
            eigen_assert(h->size() == this->m_data.m && "h has wrong dimensions");
            this->m_data.h = *h;
        }

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
