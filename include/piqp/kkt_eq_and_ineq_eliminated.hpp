// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_KKT_EQ_AND_INEQ_ELIMINATED_HPP
#define PIQP_KKT_EQ_AND_INEQ_ELIMINATED_HPP

#include "piqp/typedefs.hpp"
#include "piqp/kkt_fwd.hpp"

namespace piqp
{

template<typename Derived, typename T, typename I>
struct KKTImpl<Derived, T, I, KKTMode::EQ_ELIMINATED | KKTMode::INEQ_ELIMINATED>
{
    Vec<I> P_utri_to_Ki; // mapping from P_utri row indices to KKT matrix
    Vec<I> AT_A_to_Ki;   // mapping from AT_A row indices to KKT matrix
    Vec<I> GT_G_to_Ki;   // mapping from GT_G row indices to KKT matrix

    SparseMat<T, I> AT_A;
    SparseMat<T, I> W_delta_inv_GT_G;

    void init_workspace()
    {
        auto& data = static_cast<Derived*>(this)->data;
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        AT_A = (data.AT * data.AT.transpose()).template triangularView<Eigen::Upper>();

        W_delta_inv_GT_G = (data.GT * data.GT.transpose()).template triangularView<Eigen::Upper>();
        T W_delta_inv = T(1) / (1 + m_delta);
        Eigen::Map<Vec<T>>(W_delta_inv_GT_G.valuePtr(), W_delta_inv_GT_G.nonZeros()).array() *= W_delta_inv;

        P_utri_to_Ki.resize(data.P_utri.nonZeros());
        AT_A_to_Ki.resize(AT_A.nonZeros());
        GT_G_to_Ki.resize(W_delta_inv_GT_G.nonZeros());
    }

    SparseMat<T, I> init_KKT_matrix()
    {
        auto& data = static_cast<Derived*>(this)->data;
        auto& m_rho = static_cast<Derived*>(this)->m_rho;
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        T delta_inv = T(1) / m_delta;

        SparseMat<T, I> diagonal_rho;
        diagonal_rho.resize(data.n, data.n);
        diagonal_rho.setIdentity();
        // set diagonal to rho
        Eigen::Map<Vec<T>>(diagonal_rho.valuePtr(), data.n).setConstant(m_rho);

        SparseMat<T, I> KKT = data.P_utri + diagonal_rho + delta_inv * AT_A + W_delta_inv_GT_G;

        // compute mappings
        isize jj = KKT.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            isize P_utri_k = data.P_utri.outerIndexPtr()[j];
            isize AT_A_k = AT_A.outerIndexPtr()[j];
            isize GT_G_k = W_delta_inv_GT_G.outerIndexPtr()[j];

            isize P_utri_end = data.P_utri.outerIndexPtr()[j + 1];
            isize AT_A_end = AT_A.outerIndexPtr()[j + 1];
            isize GT_G_end = W_delta_inv_GT_G.outerIndexPtr()[j + 1];

            isize KKT_kk = KKT.outerIndexPtr()[j + 1];
            for (isize KKT_k = KKT.outerIndexPtr()[j]; KKT_k < KKT_kk; KKT_k++)
            {
                isize KKT_i = KKT.innerIndexPtr()[KKT_k];

                while (data.P_utri.innerIndexPtr()[P_utri_k] < KKT_i && P_utri_k != P_utri_end) P_utri_k++;
                while (AT_A.innerIndexPtr()[AT_A_k] < KKT_i && AT_A_k != AT_A_end) AT_A_k++;
                while (W_delta_inv_GT_G.innerIndexPtr()[GT_G_k] < KKT_i && GT_G_k != GT_G_end) GT_G_k++;

                if (data.P_utri.innerIndexPtr()[P_utri_k] == KKT_i && P_utri_k != P_utri_end)
                {
                    P_utri_to_Ki(P_utri_k) = KKT_k;
                }
                if (AT_A.innerIndexPtr()[AT_A_k] == KKT_i && AT_A_k != AT_A_end)
                {
                    AT_A_to_Ki(AT_A_k) = KKT_k;
                }
                if (W_delta_inv_GT_G.innerIndexPtr()[GT_G_k] == KKT_i && GT_G_k != GT_G_end)
                {
                    GT_G_to_Ki(GT_G_k) = KKT_k;
                }
            }
        }

        return KKT;
    }

    void update_kkt_cost()
    {
        auto& data = static_cast<Derived*>(this)->data;
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& PKi = static_cast<Derived*>(this)->PKi;
        auto& ordering = static_cast<Derived*>(this)->ordering;
        auto& m_rho = static_cast<Derived*>(this)->m_rho;

        // set PKPt to zero keeping pattern
        Eigen::Map<Vec<T>>(PKPt.valuePtr(), PKPt.nonZeros()).setZero();

        // add P_utri
        for (isize j = 0; j < data.P_utri.outerSize(); j++)
        {
            for (isize k = data.P_utri.outerIndexPtr()[j]; k < data.P_utri.outerIndexPtr()[j + 1]; k++)
            {
                PKPt.valuePtr()[PKi(this->P_utri_to_Ki(k))] += data.P_utri.valuePtr()[k];
            }
        }

        // we assume that PKPt is upper triangular and diagonal is set
        // hence we can directly address the diagonal from the outer index pointer
        for (isize col = 0; col < data.n; col++)
        {
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering.inv(col) + 1] - 1] += m_rho;
        }
    }

    void update_kkt_equalities()
    {
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& PKi = static_cast<Derived*>(this)->PKi;
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        // copy delta_inv * AT * A to PKPt
        T delta_inv = T(1) / m_delta;
        isize n = AT_A.nonZeros();
        for (isize k = 0; k < n; k++)
        {
            PKPt.valuePtr()[PKi(AT_A_to_Ki(k))] += delta_inv * AT_A.valuePtr()[k];
        }
    }

    void update_kkt_inequalities()
    {
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& PKi = static_cast<Derived*>(this)->PKi;

        update_G();

        // copy GT * (W + delta)^{-1} * G to PKPt
        isize n = W_delta_inv_GT_G.nonZeros();
        for (isize k = 0; k < n; k++)
        {
            PKPt.valuePtr()[PKi(GT_G_to_Ki(k))] += W_delta_inv_GT_G.valuePtr()[k];
        }
    }

    void update_A()
    {
        auto& data = static_cast<Derived*>(this)->data;
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        // update delta_inv * AT * A
        Eigen::Map<Vec<T>>(AT_A.valuePtr(), AT_A.nonZeros()).setZero();
        T delta_inv = T(1) / m_delta;
        isize n = data.AT.outerSize();
        for (isize k = 0; k < n; k++)
        {
            for (typename SparseMat<T, I>::InnerIterator AT_j_it(data.AT, k); AT_j_it; ++AT_j_it)
            {
                I j = AT_j_it.index();
                typename SparseMat<T, I>::InnerIterator AT_A_i_it(AT_A, j);
                typename SparseMat<T, I>::InnerIterator AT_i_it(data.AT, k);
                while (AT_A_i_it && AT_i_it)
                {
                    if (AT_A_i_it.index() < AT_i_it.index())
                    {
                        ++AT_A_i_it;
                    }
                    else
                    {
                        eigen_assert(AT_A_i_it.index() == AT_i_it.index() && "AT_A is missing entry!");

                        AT_A_i_it.valueRef() += AT_i_it.value() * AT_j_it.value();
                        ++AT_A_i_it;
                        ++AT_i_it;
                    }
                }
            }
        }
    }

    void update_G()
    {
        auto& data = static_cast<Derived*>(this)->data;
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        auto& m_s = static_cast<Derived*>(this)->m_s;
        auto& m_z_inv = static_cast<Derived*>(this)->m_z_inv;

        // update GT * (W + delta)^{-1} * G
        Eigen::Map<Vec<T>>(W_delta_inv_GT_G.valuePtr(), W_delta_inv_GT_G.nonZeros()).setZero();
        isize n = data.GT.outerSize();
        for (isize k = 0; k < n; k++)
        {
            for (typename SparseMat<T, I>::InnerIterator GT_j_it(data.GT, k); GT_j_it; ++GT_j_it)
            {
                I j = GT_j_it.index();
                typename SparseMat<T, I>::InnerIterator GT_G_i_it(W_delta_inv_GT_G, j);
                typename SparseMat<T, I>::InnerIterator GT_i_it(data.GT, k);
                while (GT_G_i_it && GT_i_it)
                {
                    if (GT_G_i_it.index() < GT_i_it.index())
                    {
                        ++GT_G_i_it;
                    }
                    else
                    {
                        eigen_assert(GT_G_i_it.index() == GT_i_it.index() && "GT_G is missing entry!");

                        T W_delta_inv = T(1) / (m_s(k) * m_z_inv(k) + m_delta);
                        GT_G_i_it.valueRef() += W_delta_inv * GT_i_it.value() * GT_j_it.value();
                        ++GT_G_i_it;
                        ++GT_i_it;
                    }
                }
            }
        }
    }
};

} // namespace piqp

#endif //PIQP_KKT_EQ_AND_INEQ_ELIMINATED_HPP
