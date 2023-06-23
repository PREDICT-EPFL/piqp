// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_KKT_HPP
#define PIQP_DENSE_KKT_HPP

#include "piqp/settings.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/dense/data.hpp"
#include "piqp/dense/ldlt_no_pivot.hpp"

namespace piqp
{

namespace dense
{

template<typename T>
struct KKT
{
    const Data<T>& data;
    const Settings<T>& settings;

    T m_rho;
    T m_delta;

    Vec<T> m_s;
    Vec<T> m_s_lb;
    Vec<T> m_s_ub;
    Vec<T> m_z_inv;
    Vec<T> m_z_lb_inv;
    Vec<T> m_z_ub_inv;

    Mat<T> kkt_mat;
    Vec<T> kkt_diag; // unregularized diagonal of KKT matrix
    LDLTNoPivot<Mat<T>, Eigen::Lower> ldlt;

    Mat<T> AT_A;
    Mat<T> W_delta_inv_G; // temporary matrix
    Vec<T> rhs_z_bar;     // temporary variable needed for back solve
    Vec<T> rhs;           // stores the rhs
    Vec<T> sol;           // solution of ldldt back solve
    Vec<T> err_corr;      // temporary variable to calculate error in iterative refinement and correction term
    Vec<T> ref_sol;       // refined solution

    KKT(const Data<T>& data, const Settings<T>& settings) : data(data), settings(settings) {}

    void init(const T& rho, const T& delta)
    {
        // init workspace
        m_s.resize(data.m);
        m_s_lb.resize(data.n);
        m_s_ub.resize(data.n);
        m_z_inv.resize(data.m);
        m_z_lb_inv.resize(data.n);
        m_z_ub_inv.resize(data.n);
        W_delta_inv_G.resize(data.m, data.n);
        rhs_z_bar.resize(data.m);
        rhs.resize(data.n);
        sol.resize(data.n);
        err_corr.resize(data.n);
        ref_sol.resize(data.n);

        m_rho = rho;
        m_delta = delta;
        m_s.setConstant(1);
        m_s_lb.head(data.n_lb).setConstant(1);
        m_s_ub.head(data.n_ub).setConstant(1);
        m_z_inv.setConstant(1);
        m_z_lb_inv.head(data.n_lb).setConstant(1);
        m_z_ub_inv.head(data.n_ub).setConstant(1);

        kkt_mat.resize(data.n, data.n);
        kkt_diag.resize(data.n);
        ldlt = LDLTNoPivot<Mat<T>>(data.n);

        if (data.p > 0)
        {
            AT_A.resize(data.n, data.n);
            AT_A.template triangularView<Eigen::Lower>() = data.AT * data.AT.transpose();
        }
        update_kkt();
    }

    void update_scalings(const T& rho, const T& delta,
                         const CVecRef<T>& s, const CVecRef<T>& s_lb, const CVecRef<T>& s_ub,
                         const CVecRef<T>& z, const CVecRef<T>& z_lb, const CVecRef<T>& z_ub)
    {
        m_rho = rho;
        m_delta = delta;
        m_s = s;
        m_s_lb.head(data.n_lb) = s_lb.head(data.n_lb);
        m_s_ub.head(data.n_ub) = s_ub.head(data.n_ub);
        m_z_inv.array() = T(1) / z.array();
        m_z_lb_inv.head(data.n_lb).array() = T(1) / z_lb.head(data.n_lb).array();
        m_z_ub_inv.head(data.n_ub).array() = T(1) / z_ub.head(data.n_ub).array();

        update_kkt();
    }

    void update_data(int options)
    {
        if (options & KKTUpdateOptions::KKT_UPDATE_A)
        {
            if (data.p > 0)
            {
                AT_A.template triangularView<Eigen::Lower>() = data.AT * data.AT.transpose();
            }
        }

        if (options != KKTUpdateOptions::KKT_UPDATE_NONE)
        {
            update_kkt();
        }
    }

    void update_kkt()
    {
        kkt_mat.template triangularView<Eigen::Lower>() = data.P_utri.transpose() + m_rho * Mat<T>::Identity(data.n, data.n);

        if (data.m > 0)
        {
            W_delta_inv_G = (m_z_inv.cwiseProduct(m_s) + Vec<T>::Constant(data.m, m_delta)).asDiagonal().inverse() * data.GT.transpose();
            kkt_mat.template triangularView<Eigen::Lower>() += data.GT * W_delta_inv_G;
        }

        for (isize i = 0; i < data.n_lb; i++)
        {
            kkt_mat.diagonal()(data.x_lb_idx(i)) += data.x_lb_scaling(i) * data.x_lb_scaling(i) / (m_z_lb_inv(i) * m_s_lb(i) + m_delta);
        }

        for (isize i = 0; i < data.n_ub; i++)
        {
            kkt_mat.diagonal()(data.x_ub_idx(i)) += data.x_ub_scaling(i) * data.x_ub_scaling(i) / (m_z_ub_inv(i) * m_s_ub(i) + m_delta);
        }

        if (data.p > 0)
        {
            kkt_mat.template triangularView<Eigen::Lower>() += T(1) / m_delta * AT_A;
        }
    }

    void multiply(const CVecRef<T>& delta_x, const CVecRef<T>& delta_y,
                  const CVecRef<T>& delta_z, const CVecRef<T>& delta_z_lb, const CVecRef<T>& delta_z_ub,
                  const CVecRef<T>& delta_s, const CVecRef<T>& delta_s_lb, const CVecRef<T>& delta_s_ub,
                  VecRef<T> rhs_x, VecRef<T> rhs_y,
                  VecRef<T> rhs_z, VecRef<T> rhs_z_lb, VecRef<T> rhs_z_ub,
                  VecRef<T> rhs_s, VecRef<T> rhs_s_lb, VecRef<T> rhs_s_ub)
    {
        rhs_x.noalias() = data.P_utri * delta_x;
        rhs_x.noalias() += data.P_utri.transpose().template triangularView<Eigen::StrictlyLower>() * delta_x;
        rhs_x.noalias() += m_rho * delta_x;
        rhs_x.noalias() += data.AT * delta_y + data.GT * delta_z;
        for (isize i = 0; i < data.n_lb; i++)
        {
            rhs_x(data.x_lb_idx(i)) -= data.x_lb_scaling(i) * delta_z_lb(i);
        }
        for (isize i = 0; i < data.n_ub; i++)
        {
            rhs_x(data.x_ub_idx(i)) += data.x_ub_scaling(i) * delta_z_ub(i);
        }

        rhs_y.noalias() = data.AT.transpose() * delta_x;
        rhs_y.noalias() -= m_delta * delta_y;

        rhs_z.noalias() = data.GT.transpose() * delta_x;
        rhs_z.noalias() -= m_delta * delta_z;
        rhs_z.noalias() += delta_s;

        for (isize i = 0; i < data.n_lb; i++)
        {
            rhs_z_lb(i) = -data.x_lb_scaling(i) * delta_x(data.x_lb_idx(i));
        }
        rhs_z_lb.head(data.n_lb).noalias() -= m_delta * delta_z_lb.head(data.n_lb);
        rhs_z_lb.head(data.n_lb).noalias() += delta_s_lb.head(data.n_lb);

        for (isize i = 0; i < data.n_ub; i++)
        {
            rhs_z_ub(i) = data.x_ub_scaling(i) * delta_x(data.x_ub_idx(i));
        }
        rhs_z_ub.head(data.n_ub).noalias() -= m_delta * delta_z_ub.head(data.n_ub);
        rhs_z_ub.head(data.n_ub).noalias() += delta_s_ub.head(data.n_ub);

        rhs_s.array() = m_s.array() * delta_z.array() + m_z_inv.array().cwiseInverse() * delta_s.array();

        rhs_s_lb.head(data.n_lb).array() = m_s_lb.head(data.n_lb).array() * delta_z_lb.head(data.n_lb).array();
        rhs_s_lb.head(data.n_lb).array() += m_z_lb_inv.head(data.n_lb).array().cwiseInverse() * delta_s_lb.head(data.n_lb).array();

        rhs_s_ub.head(data.n_ub).array() = m_s_ub.head(data.n_ub).array() * delta_z_ub.head(data.n_ub).array();
        rhs_s_ub.head(data.n_ub).array() += m_z_ub_inv.head(data.n_ub).array().cwiseInverse() * delta_s_ub.head(data.n_ub).array();
    }

    bool regularize_and_factorize(bool iterative_refinement)
    {
        if (iterative_refinement)
        {
            T static_kkt_diag_max = data.P_utri.diagonal().template lpNorm<Eigen::Infinity>();
            T max_diag = static_kkt_diag_max;
            for (isize i = 0; i < data.m; i++)
            {
                max_diag = std::max(max_diag, m_z_inv(i) * m_s(i));
            }
            for (isize i = 0; i < data.n_lb; i++)
            {
                max_diag = std::max(max_diag, m_z_lb_inv(i) * m_s_lb(i));
            }
            for (isize i = 0; i < data.n_ub; i++)
            {
                max_diag = std::max(max_diag, m_z_ub_inv(i) * m_s_ub(i));
            }

            T reg = settings.iterative_refinement_static_regularization_eps + settings.iterative_refinement_static_regularization_rel * max_diag;
            this->regularize_kkt(reg);
        }

        ldlt.compute(kkt_mat);

        if (iterative_refinement)
        {
            this->unregularize_kkt();
        }

        return ldlt.info() == Eigen::Success;
    }

    void regularize_kkt(T reg)
    {
        // save unregularized diagonal
        kkt_diag = kkt_mat.diagonal();

        // regularize diagonal
        T rho_reg = std::max(T(0), reg - m_rho);
        kkt_mat.diagonal().array() += rho_reg;
    }

    void unregularize_kkt()
    {
        // restore unregularized diagonal
        kkt_mat.diagonal() = kkt_diag;
    }

    void solve(const CVecRef<T>& rhs_x, const CVecRef<T>& rhs_y,
               const CVecRef<T>& rhs_z, const CVecRef<T>& rhs_z_lb, const CVecRef<T>& rhs_z_ub,
               const CVecRef<T>& rhs_s, const CVecRef<T>& rhs_s_lb, const CVecRef<T>& rhs_s_ub,
               VecRef<T> delta_x, VecRef<T> delta_y,
               VecRef<T> delta_z, VecRef<T> delta_z_lb, VecRef<T> delta_z_ub,
               VecRef<T> delta_s, VecRef<T> delta_s_lb, VecRef<T> delta_s_ub,
               bool iterative_refinement)
    {
        T delta_inv = T(1) / m_delta;

        rhs_z_bar.array() = rhs_z.array() - m_z_inv.array() * rhs_s.array();

        rhs = rhs_x;
        rhs_z_bar.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
        rhs.noalias() += data.GT * rhs_z_bar;
        rhs.noalias() += delta_inv * data.AT * rhs_y;

        for (isize i = 0; i < data.n_lb; i++)
        {
            rhs(data.x_lb_idx(i)) -= data.x_lb_scaling(i) * (rhs_z_lb(i) - m_z_lb_inv(i) * rhs_s_lb(i))
                                     / (m_s_lb(i) * m_z_lb_inv(i) + m_delta);
        }
        for (isize i = 0; i < data.n_ub; i++)
        {
            rhs(data.x_ub_idx(i)) += data.x_ub_scaling(i) * (rhs_z_ub(i) - m_z_ub_inv(i) * rhs_s_ub(i))
                                     / (m_s_ub(i) * m_z_ub_inv(i) + m_delta);
        }

        sol = rhs;
        solve_ldlt_in_place(sol);

        if (iterative_refinement && settings.iterative_refinement_max_iter > 0)
        {
            T rhs_norm = rhs.template lpNorm<Eigen::Infinity>();

            err_corr = rhs;
            err_corr -= kkt_mat.template triangularView<Eigen::Lower>() * sol;
            err_corr -= kkt_mat.transpose().template triangularView<Eigen::StrictlyUpper>() * sol;
            T error_norm = err_corr.template lpNorm<Eigen::Infinity>();

            for (isize i = 0; i < settings.iterative_refinement_max_iter; i++)
            {
                if (error_norm <= (settings.iterative_refinement_eps_abs + settings.iterative_refinement_eps_rel * rhs_norm))
                {
                    break;
                }

                T prev_error_norm = error_norm;

                solve_ldlt_in_place(err_corr);
                ref_sol = sol + err_corr;

                err_corr = rhs;
                err_corr -= kkt_mat.template triangularView<Eigen::Lower>() * ref_sol;
                err_corr -= kkt_mat.transpose().template triangularView<Eigen::StrictlyUpper>() * ref_sol;
                error_norm = err_corr.template lpNorm<Eigen::Infinity>();

                T improvement_rate = prev_error_norm / error_norm;
                if (improvement_rate < settings.iterative_refinement_min_improvement_rate)
                {
                    if (improvement_rate > T(1))
                    {
                        std::swap(sol, ref_sol);
                    }
                    break;
                }
                else
                {
                    std::swap(sol, ref_sol);
                }
            }
        }

        delta_x.noalias() = sol;

        delta_y.noalias() = delta_inv * data.AT.transpose() * delta_x;
        delta_y.noalias() -= delta_inv * rhs_y;

        delta_z.noalias() = data.GT.transpose() * delta_x;
        delta_z.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
        delta_z.noalias() -= rhs_z_bar;

        for (isize i = 0; i < data.n_lb; i++)
        {
            delta_z_lb(i) = (-data.x_lb_scaling(i) * delta_x(data.x_lb_idx(i)) - rhs_z_lb(i) + m_z_lb_inv(i) * rhs_s_lb(i))
                / (m_s_lb(i) * m_z_lb_inv(i) + m_delta);
        }
        for (isize i = 0; i < data.n_ub; i++)
        {
            delta_z_ub(i) = (data.x_ub_scaling(i) * delta_x(data.x_ub_idx(i)) - rhs_z_ub(i) + m_z_ub_inv(i) * rhs_s_ub(i))
                            / (m_s_ub(i) * m_z_ub_inv(i) + m_delta);
        }

        delta_s.array() = m_z_inv.array() * (rhs_s.array() - m_s.array() * delta_z.array());

        delta_s_lb.head(data.n_lb).array() = m_z_lb_inv.head(data.n_lb).array()
            * (rhs_s_lb.head(data.n_lb).array() - m_s_lb.head(data.n_lb).array() * delta_z_lb.head(data.n_lb).array());

        delta_s_ub.head(data.n_ub).array() = m_z_ub_inv.head(data.n_ub).array()
            * (rhs_s_ub.head(data.n_ub).array() - m_s_ub.head(data.n_ub).array() * delta_z_ub.head(data.n_ub).array());

#ifdef PIQP_DEBUG_PRINT
        Vec<T> err_x = delta_x;
        Vec<T> err_y = delta_y;
        Vec<T> err_z = delta_z;
        Vec<T> err_z_lb = delta_z_lb;
        Vec<T> err_z_ub = delta_z_ub;
        Vec<T> err_s = delta_s;
        Vec<T> err_s_lb = delta_s_lb;
        Vec<T> err_s_ub = delta_s_ub;

        multiply(delta_x, delta_y, delta_z, delta_z_lb, delta_z_ub, delta_s, delta_s_lb, delta_s_ub,
                 err_x, err_y, err_z, err_z_lb, err_z_ub, err_s, err_s_lb, err_s_ub);

        err_x -= rhs_x;
        err_y -= rhs_y;
        err_z -= rhs_z;
        err_z_lb.head(data.n_lb) -= rhs_z_lb.head(data.n_lb);
        err_z_ub.head(data.n_ub) -= rhs_z_ub.head(data.n_ub);
        err_s -= rhs_s;
        err_s_lb.head(data.n_lb) -= rhs_s_lb.head(data.n_lb);
        err_s_ub.head(data.n_ub) -= rhs_s_ub.head(data.n_ub);

        std::cout << "kkt_error: "
                  << err_x.template lpNorm<Eigen::Infinity>() << " "
                  << err_y.template lpNorm<Eigen::Infinity>() << " "
                  << err_z.template lpNorm<Eigen::Infinity>() << " "
                  << err_z_lb.head(data.n_lb).template lpNorm<Eigen::Infinity>() << " "
                  << err_z_ub.head(data.n_ub).template lpNorm<Eigen::Infinity>() << " "
                  << err_s.template lpNorm<Eigen::Infinity>() << " "
                  << err_s_lb.head(data.n_lb).template lpNorm<Eigen::Infinity>() << " "
                  << err_s_ub.head(data.n_ub).template lpNorm<Eigen::Infinity>() << std::endl;
#endif
    }

    void solve_ldlt_in_place(VecRef<T> x)
    {
#ifdef PIQP_DEBUG_PRINT
        Vec<T> x_copy = x;
#endif

        ldlt.solveInPlace(x);

#ifdef PIQP_DEBUG_PRINT
        Vec<T> rhs_x = kkt_mat.template triangularView<Eigen::Lower>() * x;
        rhs_x += kkt_mat.transpose().template triangularView<Eigen::StrictlyUpper>() * x;
        std::cout << "ldlt_error: " << (x_copy - rhs_x).template lpNorm<Eigen::Infinity>() << std::endl;
#endif
    }
};

} // namespace dense

} // namespace piqp

#endif //PIQP_DENSE_KKT_HPP
