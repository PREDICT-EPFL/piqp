// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_KKT_HPP
#define PIQP_DENSE_KKT_HPP

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
    Data<T>& data;

    T m_rho;
    T m_delta;

    Vec<T> m_s;
    Vec<T> m_z_inv;

    Mat<T> kkt_mat;
    LDLTNoPivot<Mat<T>, Eigen::Lower> ldlt;

    Mat<T> AT_A;
    Mat<T> W_delta_inv_G; // temporary matrix
    Vec<T> rhs_z_bar;     // temporary variable needed for back solve
    Vec<T> rhs;           // stores the rhs and the solution for back solve

    explicit KKT(Data<T>& data) : data(data) {}

    void init(const T& rho, const T& delta)
    {
        // init workspace
        m_s.resize(data.m);
        m_z_inv.resize(data.m);
        W_delta_inv_G.resize(data.m, data.n);
        rhs_z_bar.resize(data.m);
        rhs.resize(data.n);

        m_rho = rho;
        m_delta = delta;
        m_s.setConstant(1);
        m_z_inv.setConstant(1);

        kkt_mat.resize(data.n, data.n);
        ldlt = LDLTNoPivot<Mat<T>>(data.n);

        if (data.p > 0)
        {
            AT_A.resize(data.n, data.n);
            AT_A.template triangularView<Eigen::Lower>() = data.AT * data.AT.transpose();
        }
        update_kkt();
    }

    void update_scalings(const T& rho, const T& delta, const CVecRef<T>& s, const CVecRef<T>& z)
    {
        m_rho = rho;
        m_delta = delta;
        m_s = s;
        m_z_inv.array() = T(1) / z.array();

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

        if (data.p > 0)
        {
            kkt_mat.template triangularView<Eigen::Lower>() += T(1) / m_delta * AT_A;
        }
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
        ldlt.compute(kkt_mat);
        return ldlt.info() == Eigen::Success;
    }

    void solve(const CVecRef<T>& rhs_x, const CVecRef<T>& rhs_y, const CVecRef<T>& rhs_z, const CVecRef<T>& rhs_s,
               VecRef<T> delta_x, VecRef<T> delta_y, VecRef<T> delta_z, VecRef<T> delta_s)
    {
        T delta_inv = T(1) / m_delta;

        rhs_z_bar.array() = rhs_z.array() - m_z_inv.array() * rhs_s.array();

        rhs = rhs_x;
        rhs_z_bar.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
        rhs.noalias() += data.GT * rhs_z_bar;
        rhs.noalias() += delta_inv * data.AT * rhs_y;

        ldlt.solveInPlace(rhs);

        delta_x.noalias() = rhs;
        delta_y.noalias() = delta_inv * data.AT.transpose() * delta_x;
        delta_y.noalias() -= delta_inv * rhs_y;
        delta_z.noalias() = data.GT.transpose() * delta_x;
        delta_z.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
        delta_z.noalias() -= rhs_z_bar;
        delta_s.array() = m_z_inv.array() * (rhs_s.array() - m_s.array() * delta_z.array());
    }
};

} // namespace dense

} // namespace piqp

#endif //PIQP_DENSE_KKT_HPP
