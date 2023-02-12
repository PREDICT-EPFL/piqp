// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_KKT_HPP
#define PIQP_SPARSE_KKT_HPP

#include "piqp/sparse/data.hpp"
#include "piqp/sparse/ldlt.hpp"
#include "piqp/sparse/ordering.hpp"
#include "piqp/sparse/utils.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/sparse/kkt_full.hpp"
#include "piqp/sparse/kkt_eq_eliminated.hpp"
#include "piqp/sparse/kkt_ineq_eliminated.hpp"
#include "piqp/sparse/kkt_all_eliminated.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I, int Mode = KKTMode::KKT_FULL, typename Ordering = AMDOrdering<I>>
struct KKT : public KKTImpl<KKT<T, I, Mode, Ordering>, T, I, Mode>
{
    Data<T, I>& data;

    T m_rho;
    T m_delta;

    Vec<T> m_s;
    Vec<T> m_s_lb;
    Vec<T> m_s_ub;
    Vec<T> m_z_inv;
    Vec<T> m_z_lb_inv;
    Vec<T> m_z_ub_inv;

    Ordering ordering;
    SparseMat<T, I> PKPt; // permuted KKT matrix, upper triangular only
    Vec<I> PKi; // mapping of row indices of KKT matrix to permuted KKT matrix

    LDLt<T, I> ldlt;

    Vec<T> rhs_z_bar; // temporary variable needed for back solve
    Vec<T> rhs_perm;  // permuted rhs for back solve
    Vec<T> rhs;       // stores the rhs and the solution for back solve

    explicit KKT(Data<T, I>& data) : data(data) {}

    void init(const T& rho, const T& delta)
    {
        isize n_kkt;
        if (Mode == KKTMode::KKT_FULL)
        {
            n_kkt = data.n + data.p + data.m;
        }
        else if (Mode == KKTMode::KKT_EQ_ELIMINATED)
        {
            n_kkt = data.n + data.m;
        }
        else if (Mode == KKTMode::KKT_INEQ_ELIMINATED)
        {
            n_kkt = data.n + data.p;
        }
        else
        {
            n_kkt = data.n;
        }

        // init workspace
        m_s.resize(data.m);
        m_s_lb.resize(data.n);
        m_s_ub.resize(data.n);
        m_z_inv.resize(data.m);
        m_z_lb_inv.resize(data.n);
        m_z_ub_inv.resize(data.n);
        rhs_z_bar.resize(data.m);
        rhs_perm.resize(n_kkt);
        rhs.resize(n_kkt);

        m_rho = rho;
        m_delta = delta;
        m_s.setConstant(1);
        m_s_lb.head(data.n_lb).setConstant(1);
        m_s_ub.head(data.n_ub).setConstant(1);
        m_z_inv.setConstant(1);
        m_z_lb_inv.head(data.n_lb).setConstant(1);
        m_z_ub_inv.head(data.n_ub).setConstant(1);

        this->init_workspace();
        SparseMat<T, I> KKT = this->create_kkt_matrix();

        ordering.init(KKT);
        PKi = permute_sparse_symmetric_matrix(KKT, PKPt, ordering);
        this->update_kkt_box_scalings();

        ldlt.factorize_symbolic_upper_triangular(PKPt);
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

        this->update_kkt_cost_scalings();
        this->update_kkt_equality_scalings();
        this->update_kkt_inequality_scaling();
        this->update_kkt_box_scalings();
    }

    void update_kkt_box_scalings()
    {
        // we assume that PKPt is upper triangular and diagonal is set
        // hence we can directly address the diagonal from the outer index pointer
        for (isize i = 0; i < data.n_lb; i++)
        {
            isize col = data.x_lb_idx(i);
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering.inv(col) + 1] - 1] += T(1) / (m_z_lb_inv(i) * m_s_lb(i) + m_delta);
        }
        for (isize i = 0; i < data.n_ub; i++)
        {
            isize col = data.x_ub_idx(i);
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering.inv(col) + 1] - 1] += T(1) / (m_z_ub_inv(i) * m_s_ub(i) + m_delta);
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
            rhs_x(data.x_lb_idx(i)) -= delta_z_lb(i);
        }
        for (isize i = 0; i < data.n_ub; i++)
        {
            rhs_x(data.x_ub_idx(i)) += delta_z_ub(i);
        }

        rhs_y.noalias() = data.AT.transpose() * delta_x;
        rhs_y.noalias() -= m_delta * delta_y;

        rhs_z.noalias() = data.GT.transpose() * delta_x;
        rhs_z.noalias() -= m_delta * delta_z;
        rhs_z.noalias() += delta_s;

        for (isize i = 0; i < data.n_lb; i++)
        {
            rhs_z_lb(i) = -delta_x(data.x_lb_idx(i));
        }
        rhs_z_lb.head(data.n_lb).noalias() -= m_delta * delta_z_lb.head(data.n_lb);
        rhs_z_lb.head(data.n_lb).noalias() += delta_s_lb.head(data.n_lb);

        for (isize i = 0; i < data.n_ub; i++)
        {
            rhs_z_ub(i) = delta_x(data.x_ub_idx(i));
        }
        rhs_z_ub.head(data.n_ub).noalias() -= m_delta * delta_z_ub.head(data.n_ub);
        rhs_z_ub.head(data.n_ub).noalias() += delta_s_ub.head(data.n_ub);

        rhs_s.array() = m_s.array() * delta_z.array() + m_z_inv.array().cwiseInverse() * delta_s.array();

        rhs_s_lb.head(data.n_lb).array() = m_s_lb.head(data.n_lb).array() * delta_z_lb.head(data.n_lb).array();
        rhs_s_lb.head(data.n_lb).array() += m_z_lb_inv.head(data.n_lb).array().cwiseInverse() * delta_s_lb.head(data.n_lb).array();

        rhs_s_ub.head(data.n_ub).array() = m_s_ub.head(data.n_ub).array() * delta_z_ub.head(data.n_ub).array();
        rhs_s_ub.head(data.n_ub).array() += m_z_ub_inv.head(data.n_ub).array().cwiseInverse() * delta_s_ub.head(data.n_ub).array();
    }

    bool factorize()
    {
        isize n = ldlt.factorize_numeric_upper_triangular(PKPt);
        return n == PKPt.cols();
    }

    void solve(const CVecRef<T>& rhs_x, const CVecRef<T>& rhs_y,
               const CVecRef<T>& rhs_z, const CVecRef<T>& rhs_z_lb, const CVecRef<T>& rhs_z_ub,
               const CVecRef<T>& rhs_s, const CVecRef<T>& rhs_s_lb, const CVecRef<T>& rhs_s_ub,
               VecRef<T> delta_x, VecRef<T> delta_y,
               VecRef<T> delta_z, VecRef<T> delta_z_lb, VecRef<T> delta_z_ub,
               VecRef<T> delta_s, VecRef<T> delta_s_lb, VecRef<T> delta_s_ub)
    {
        T delta_inv = T(1) / m_delta;

        rhs_z_bar.array() = rhs_z.array() - m_z_inv.array() * rhs_s.array();

        rhs.head(data.n).noalias() = rhs_x;
        if (Mode == KKTMode::KKT_FULL)
        {
            rhs.segment(data.n, data.p).noalias() = rhs_y;
            rhs.tail(data.m).noalias() = rhs_z_bar;
        }
        else if (Mode == KKTMode::KKT_EQ_ELIMINATED)
        {
            rhs.head(data.n).noalias() += delta_inv * data.AT * rhs_y;
            rhs.tail(data.m).noalias() = rhs_z_bar;
        }
        else if (Mode == KKTMode::KKT_INEQ_ELIMINATED)
        {
            rhs_z_bar.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
            rhs.head(data.n).noalias() += data.GT * rhs_z_bar;
            rhs.tail(data.p).noalias() = rhs_y;
        }
        else
        {
            rhs_z_bar.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
            rhs.noalias() += data.GT * rhs_z_bar;
            rhs.noalias() += delta_inv * data.AT * rhs_y;
        }
        for (isize i = 0; i < data.n_lb; i++)
        {
            rhs(data.x_lb_idx(i)) -= (rhs_z_lb(i) - m_z_lb_inv(i) * rhs_s_lb(i)) / (m_s_lb(i) * m_z_lb_inv(i) + m_delta);
        }
        for (isize i = 0; i < data.n_ub; i++)
        {
            rhs(data.x_ub_idx(i)) += (rhs_z_ub(i) - m_z_ub_inv(i) * rhs_s_ub(i)) / (m_s_ub(i) * m_z_ub_inv(i) + m_delta);
        }

        ordering.template perm<T>(rhs_perm, rhs);
        ldlt.lsolve(rhs_perm);
        ldlt.dsolve(rhs_perm);
        ldlt.ltsolve(rhs_perm);
        ordering.template permt<T>(rhs, rhs_perm);

        delta_x.noalias() = rhs.head(data.n);

        if (Mode == KKTMode::KKT_FULL)
        {
            delta_y.noalias() = rhs.segment(data.n, data.p);

            delta_z.noalias() = rhs.tail(data.m);
        }
        else if (Mode == KKTMode::KKT_EQ_ELIMINATED)
        {
            delta_y.noalias() = delta_inv * data.AT.transpose() * delta_x;
            delta_y.noalias() -= delta_inv * rhs_y;

            delta_z.noalias() = rhs.tail(data.m);
        }
        else if (Mode == KKTMode::KKT_INEQ_ELIMINATED)
        {
            delta_y.noalias() = rhs.tail(data.p);

            delta_z.noalias() = data.GT.transpose() * delta_x;
            delta_z.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
            delta_z.noalias() -= rhs_z_bar;
        }
        else
        {
            delta_y.noalias() = delta_inv * data.AT.transpose() * delta_x;
            delta_y.noalias() -= delta_inv * rhs_y;

            delta_z.noalias() = data.GT.transpose() * delta_x;
            delta_z.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
            delta_z.noalias() -= rhs_z_bar;
        }

        for (isize i = 0; i < data.n_lb; i++)
        {
            delta_z_lb(i) = (-delta_x(data.x_lb_idx(i)) - rhs_z_lb(i) + m_z_lb_inv(i) * rhs_s_lb(i))
                            / (m_s_lb(i) * m_z_lb_inv(i) + m_delta);
        }
        for (isize i = 0; i < data.n_ub; i++)
        {
            delta_z_ub(i) = (delta_x(data.x_ub_idx(i)) - rhs_z_ub(i) + m_z_ub_inv(i) * rhs_s_ub(i))
                            / (m_s_ub(i) * m_z_ub_inv(i) + m_delta);
        }

        delta_s.array() = m_z_inv.array() * (rhs_s.array() - m_s.array() * delta_z.array());

        delta_s_lb.head(data.n_lb).array() = m_z_lb_inv.head(data.n_lb).array()
            * (rhs_s_lb.head(data.n_lb).array() - m_s_lb.head(data.n_lb).array() * delta_z_lb.head(data.n_lb).array());

        delta_s_ub.head(data.n_ub).array() = m_z_ub_inv.head(data.n_ub).array()
            * (rhs_s_ub.head(data.n_ub).array() - m_s_ub.head(data.n_ub).array() * delta_z_ub.head(data.n_ub).array());
    }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_KKT_HPP
