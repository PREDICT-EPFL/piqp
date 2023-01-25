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

#include "piqp/results.hpp"
#include "piqp/settings.hpp"
#include "piqp/data.hpp"
#include "piqp/kkt.hpp"
#include "piqp/utils/optional.hpp"

namespace piqp
{

template<typename T, typename I, int Mode = KKTMode::KKT_FULL>
class Solver
{
private:
    Result<T> m_result;
    Settings<T> m_settings;
    Data<T, I> m_data;
    KKT<T, I, Mode> m_kkt;
    bool m_kkt_dirty = true;

    // regularization parameters
    T rho;
    T delta;

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

public:
    Solver() : m_kkt(m_data) {};

    Settings<T>& settings() { return m_settings; }

    const Result<T>& result() const { return m_result; }

    void setup(const SparseMat<T, I>& P,
               const SparseMat<T, I>& A,
               const SparseMat<T, I>& G,
               const CVecRef<T>& c,
               const CVecRef<T>& b,
               const CVecRef<T>& h)
    {
        m_data.n = P.rows();
        m_data.p = A.rows();
        m_data.m = G.rows();

        eigen_assert(P.rows() == m_data.n && P.cols() == m_data.n && "P must be square");
        eigen_assert(A.rows() == m_data.p && A.cols() == m_data.n && "A must have correct dimensions");
        eigen_assert(G.rows() == m_data.m && G.cols() == m_data.n && "G must have correct dimensions");
        eigen_assert(c.size() == m_data.n && "c must have correct dimensions");
        eigen_assert(b.size() == m_data.p && "b must have correct dimensions");
        eigen_assert(h.size() == m_data.m && "h must have correct dimensions");

        m_data.P_utri = P.template triangularView<Eigen::Upper>();
        m_data.AT = A.transpose();
        m_data.GT = G.transpose();
        m_data.c = c;
        m_data.b = b;
        m_data.h = h;

        // init result
        m_result.x.resize(m_data.n);
        m_result.y.resize(m_data.p);
        m_result.z.resize(m_data.m);
        m_result.s.resize(m_data.m);

        m_result.zeta.resize(m_data.n);
        m_result.lambda.resize(m_data.p);
        m_result.nu.resize(m_data.m);

        // init workspace
        rho = m_settings.rho_init;
        delta = m_settings.delta_init;

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

        m_kkt.init(rho, delta);
        m_kkt_dirty = false;
    }

    Status solve()
    {
        if (m_settings.verbose)
        {
            printf("----------------------------------------------------------\n");
            printf("                           PIQP                           \n");
            printf("           (c) Roland Schwan, Colin N. Jones              \n");
            printf("   École Polytechnique Fédérale de Lausanne (EPFL) 2023   \n");
            printf("----------------------------------------------------------\n");
            printf("variables n = %ld, nzz(P upper triangular) = %ld\n", m_data.m, m_data.P_utri.nonZeros());
            printf("equality constraints p = %ld, nnz(A) = %ld\n", m_data.p, m_data.AT.nonZeros());
            printf("inequality constraints m = %ld, nnz(G) = %ld\n", m_data.m, m_data.GT.nonZeros());
            printf("----------------------------------------------------------\n");
            printf("iter  prim_cost      dual_cost      prim_inf      dual_inf      rho         delta       mu          prim_step   dual_step\n");
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
            rho = m_settings.rho_init;
            delta = m_settings.delta_init;

            m_result.s.setConstant(1);
            m_result.z.setConstant(1);
            m_kkt.update_scalings(rho, delta, m_result.s, m_result.z);
        }

        while (!m_kkt.factorize())
        {
            if (m_result.info.factor_retires < m_settings.max_factor_retires)
            {
                delta *= 100;
                rho *= 100;
                m_result.info.factor_retires++;
                m_result.info.reg_limit = std::min(10 * m_result.info.reg_limit, m_settings.feas_tol);
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
            if (m_result.s.norm() <= 1e-4)
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

            m_result.info.primal_inf = ry_nr.norm() + rz_nr.norm();
            m_result.info.dual_inf = rx_nr.norm();

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
                       rho,
                       delta,
                       m_result.info.mu,
                       m_result.info.primal_step,
                       m_result.info.dual_step);
            }

            rx = rx_nr - rho * (m_result.x - m_result.zeta);
            ry = ry_nr - delta * (m_result.lambda - m_result.y);
            rz = rz_nr - delta * (m_result.nu - m_result.z);

            if (m_result.info.dual_inf / std::max(T(100), m_data.c.norm()) < m_settings.feas_tol &&
                m_result.info.primal_inf / std::max(T(100), m_data.b.norm() + m_data.h.norm()) < m_settings.feas_tol &&
                m_result.info.mu < m_settings.dual_tol)
            {
                m_result.info.status = Status::PIQP_SOLVED;
                return m_result.info.status;
            }

            if (m_result.info.no_dual_update > 5 && (m_result.lambda - m_result.y).norm() + (m_result.nu - m_result.z).norm() > 1e10 && ry.norm() + rz.norm() < m_settings.feas_tol)
            {
                m_result.info.status = Status::PIQP_PRIMAL_INFEASIBLE;
                return m_result.info.status;
            }

            if (m_result.info.no_primal_update > 5 && (m_result.x - m_result.zeta).norm() > 1e10 && rx.norm() < m_settings.feas_tol)
            {
                m_result.info.status = Status::PIQP_DUAL_INFEASIBLE;
                return m_result.info.status;
            }

            m_result.info.iter++;

            // avoid possibility of converging to a local minimum -> decrease the minimum regularization value
            if ((m_result.info.no_primal_update > 5 && rho == m_result.info.reg_limit && m_result.info.reg_limit != 5e-13) ||
                (m_result.info.no_dual_update > 5 && delta == m_result.info.reg_limit && m_result.info.reg_limit != 5e-13))
            {
                m_result.info.reg_limit = 5e-13;
                m_result.info.no_primal_update = 0;
                m_result.info.no_dual_update = 0;
            }

            m_kkt.update_scalings(rho, delta, m_result.s, m_result.z);
            m_kkt_dirty = true;
            bool kkt_success = m_kkt.factorize();
            if (!kkt_success)
            {
                if (m_result.info.factor_retires < m_settings.max_factor_retires)
                {
                    delta *= 100;
                    rho *= 100;
                    m_result.info.iter--;
                    m_result.info.factor_retires++;
                    m_result.info.reg_limit = std::min(10 * m_result.info.reg_limit, m_settings.feas_tol);
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

                if (rx_nr.norm() < 0.95 * m_result.info.dual_inf)
                {
                    m_result.zeta = m_result.x;
                    rho = std::max(m_result.info.reg_limit, (T(1) - mu_rate) * rho);
                }
                else
                {
                    m_result.info.no_primal_update++;
                    rho = std::max(m_result.info.reg_limit, (T(1) - 0.666 * mu_rate) * rho);
                }

                if (ry_nr.norm() + rz_nr.norm() < 0.95 * m_result.info.primal_inf)
                {
                    m_result.lambda = m_result.y;
                    m_result.nu = m_result.z;
                    delta = std::max(m_result.info.reg_limit, (T(1) - mu_rate) * delta);
                }
                else
                {
                    m_result.info.no_dual_update++;
                    delta = std::max(m_result.info.reg_limit, (T(1) - 0.666 * mu_rate) * delta);
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

                if (rx_nr.norm() < 0.95 * m_result.info.dual_inf)
                {
                    m_result.zeta = m_result.x;
                    rho = std::max(m_result.info.reg_limit, 0.1 * rho);
                }
                else
                {
                    m_result.info.no_primal_update++;
                    rho = std::max(m_result.info.reg_limit, 0.5 * rho);
                }

                if (ry_nr.norm() + rz_nr.norm() < 0.95 * m_result.info.primal_inf)
                {
                    m_result.lambda = m_result.y;
                    delta = std::max(m_result.info.reg_limit, 0.1 * delta);
                }
                else
                {
                    m_result.info.no_dual_update++;
                    delta = std::max(m_result.info.reg_limit, 0.5 * delta);
                }
            }
        }

        m_result.info.status = Status::PIQP_MAX_ITER_REACHED;
        return m_result.info.status;
    }

private:
    void update_nr_residuals()
    {
        rx_nr.noalias() = -m_data.P_utri * m_result.x;
        rx_nr.noalias() -= m_data.P_utri.transpose().template triangularView<Eigen::StrictlyLower>() * m_result.x;
        rx_nr.array() -= m_data.c.array();
        rx_nr.noalias() -= m_data.AT * m_result.y;
        rx_nr.noalias() -= m_data.GT * m_result.z;

        ry_nr.noalias() = m_data.b - m_data.AT.transpose() * m_result.x;

        rz_nr.noalias() = m_data.h - m_result.s - m_data.GT.transpose() * m_result.x;
    }
};

} // namespace piqp

#endif //PIQP_SOLVER_HPP
