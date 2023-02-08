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
#include "utils.hpp"
#include "piqp/sparse/kkt_fwd.hpp"
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
    Vec<T> m_z_inv;

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
        m_z_inv.resize(data.m);
        rhs_z_bar.resize(data.m);
        rhs_perm.resize(n_kkt);
        rhs.resize(n_kkt);

        m_rho = rho;
        m_delta = delta;
        m_s.setConstant(1);
        m_z_inv.setConstant(1);

        this->init_workspace();
        SparseMat<T, I> KKT = this->create_kkt_matrix();

        ordering.init(KKT);
        PKi = permute_sparse_symmetric_matrix(KKT, PKPt, ordering);

        ldlt.factorize_symbolic_upper_triangular(PKPt);
    }

    void update_scalings(const T& rho, const T& delta, const CVecRef<T>& s, const CVecRef<T>& z)
    {
        m_rho = rho;
        m_delta = delta;
        m_s = s;
        m_z_inv.array() = T(1) / z.array();

        this->update_kkt_cost_scalings();
        this->update_kkt_equality_scalings();
        this->update_kkt_inequality_scaling();
    }

    void multiply(const CVecRef<T>& delta_x, const CVecRef<T>& delta_y,
                  const CVecRef<T>& delta_z, const CVecRef<T>& delta_s,
                  VecRef<T> rhs_x, VecRef<T> rhs_y, VecRef<T> rhs_z, VecRef<T> rhs_s)
    {
        rhs_x.noalias() = data.P_utri * delta_x;
        rhs_x.noalias() += data.P_utri.transpose().template triangularView<Eigen::StrictlyLower>() * delta_x;
        rhs_x.noalias() += m_rho * delta_x;
        rhs_x.noalias() += data.AT * delta_y + data.GT * delta_z;

        rhs_y.noalias() = data.AT.transpose() * delta_x;
        rhs_y.noalias() -= m_delta * delta_y;

        rhs_z.noalias() = data.GT.transpose() * delta_x;
        rhs_z.noalias() -= m_delta * delta_z;
        rhs_z.noalias() += delta_s;

        rhs_s.array() = m_s.array() * delta_z.array() + m_z_inv.array().cwiseInverse() * delta_s.array();
    }

    bool factorize()
    {
        isize n = ldlt.factorize_numeric_upper_triangular(PKPt);
        return n == PKPt.cols();
    }

    void solve(const CVecRef<T>& rhs_x, const CVecRef<T>& rhs_y, const CVecRef<T>& rhs_z, const CVecRef<T>& rhs_s,
               VecRef<T> delta_x, VecRef<T> delta_y, VecRef<T> delta_z, VecRef<T> delta_s)
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
        delta_s.array() = m_z_inv.array() * (rhs_s.array() - m_s.array() * delta_z.array());
    }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_KKT_HPP
