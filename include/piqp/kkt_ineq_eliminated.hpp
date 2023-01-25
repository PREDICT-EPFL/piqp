// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_KKT_INEQ_ELIMINATED_HPP
#define PIQP_KKT_INEQ_ELIMINATED_HPP

#include "piqp/typedefs.hpp"
#include "piqp/kkt_fwd.hpp"

namespace piqp
{

template<typename Derived, typename T, typename I>
struct KKTImpl<Derived, T, I, KKTMode::INEQ_ELIMINATED>
{
    Vec<I> P_utri_to_Ki; // mapping from P_utri row indices to KKT matrix
    Vec<I> AT_to_Ki;     // mapping from AT row indices to KKT matrix
    Vec<I> GT_G_to_Ki;   // mapping from GT_G row indices to KKT matrix

    SparseMat<T, I> W_delta_inv_GT_G;

    void init_workspace()
    {
        auto& data = static_cast<Derived*>(this)->data;
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        W_delta_inv_GT_G = (data.GT * data.GT.transpose()).template triangularView<Eigen::Upper>();
        T W_delta_inv = T(1) / (1 + m_delta);
        Eigen::Map<Vec<T>>(W_delta_inv_GT_G.valuePtr(), W_delta_inv_GT_G.nonZeros()).array() *= W_delta_inv;

        P_utri_to_Ki.resize(data.P_utri.nonZeros());
        AT_to_Ki.resize(data.AT.nonZeros());
        GT_G_to_Ki.resize(W_delta_inv_GT_G.nonZeros());
    }

    SparseMat<T, I> init_KKT_matrix()
    {
        auto& data = static_cast<Derived*>(this)->data;
        auto& m_rho = static_cast<Derived*>(this)->m_rho;
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        SparseMat<T, I> diagonal_rho;
        diagonal_rho.resize(data.n, data.n);
        diagonal_rho.setIdentity();
        // set diagonal to rho
        Eigen::Map<Vec<T>>(diagonal_rho.valuePtr(), data.n).setConstant(m_rho);

        SparseMat<T, I> KKT_top_left_block = data.P_utri + diagonal_rho + W_delta_inv_GT_G;

        isize n_kkt = data.n + data.p;
        SparseMat<T, I> KKT(n_kkt, n_kkt);

        // count non-zeros
        isize non_zeros = 0;
        isize j_kkt = 0;
        isize jj = KKT_top_left_block.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            non_zeros += KKT_top_left_block.outerIndexPtr()[j + 1] - KKT_top_left_block.outerIndexPtr()[j];
            j_kkt++;
            KKT.outerIndexPtr()[j_kkt] = non_zeros;
        }
        jj = data.AT.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            non_zeros += data.AT.outerIndexPtr()[j + 1] - data.AT.outerIndexPtr()[j];
            non_zeros++; // add one for the diagonal element
            j_kkt++;
            KKT.outerIndexPtr()[j_kkt] = non_zeros;
        }
        KKT.resizeNonZeros(non_zeros);

        j_kkt = 0;
        // copy top left block
        jj = KKT_top_left_block.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            isize k_kkt = KKT.outerIndexPtr()[j_kkt];
            isize col_nnz = KKT_top_left_block.outerIndexPtr()[j + 1] - KKT_top_left_block.outerIndexPtr()[j];
            Eigen::Map<Vec<I>>(KKT.innerIndexPtr() + k_kkt, col_nnz) = Eigen::Map<Vec<I>>(KKT_top_left_block.innerIndexPtr() + KKT_top_left_block.outerIndexPtr()[j], col_nnz);
            Eigen::Map<Vec<T>>(KKT.valuePtr() + k_kkt, col_nnz) = Eigen::Map<Vec<T>>(KKT_top_left_block.valuePtr() + KKT_top_left_block.outerIndexPtr()[j], col_nnz);

            j_kkt++;
        }
        // copy AT and the diagonal
        jj = data.AT.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            isize k_kkt = KKT.outerIndexPtr()[j_kkt];
            isize col_nnz = data.AT.outerIndexPtr()[j + 1] - data.AT.outerIndexPtr()[j];
            Eigen::Map<Vec<I>>(KKT.innerIndexPtr() + k_kkt, col_nnz) = Eigen::Map<Vec<I>>(data.AT.innerIndexPtr() + data.AT.outerIndexPtr()[j], col_nnz);
            Eigen::Map<Vec<T>>(KKT.valuePtr() + k_kkt, col_nnz) = Eigen::Map<Vec<T>>(data.AT.valuePtr() + data.AT.outerIndexPtr()[j], col_nnz);

            // diagonal
            KKT.innerIndexPtr()[k_kkt + col_nnz] = j_kkt;
            KKT.valuePtr()[k_kkt + col_nnz] = -m_delta;

            isize i = 0;
            isize kk = data.AT.outerIndexPtr()[j + 1];
            for (isize k = data.AT.outerIndexPtr()[j]; k < kk; k++)
            {
                AT_to_Ki[k] = k_kkt + i;
                i++;
            }

            j_kkt++;
        }

        // compute remaining mappings
        jj = data.n;
        for (isize j = 0; j < jj; j++)
        {
            isize P_utri_k = data.P_utri.outerIndexPtr()[j];
            isize GT_G_k = W_delta_inv_GT_G.outerIndexPtr()[j];

            isize P_utri_end = data.P_utri.outerIndexPtr()[j + 1];
            isize GT_G_end = W_delta_inv_GT_G.outerIndexPtr()[j + 1];

            isize KKT_kk = KKT.outerIndexPtr()[j + 1];
            for (isize KKT_k = KKT.outerIndexPtr()[j]; KKT_k < KKT_kk; KKT_k++)
            {
                isize KKT_i = KKT.innerIndexPtr()[KKT_k];

                while (data.P_utri.innerIndexPtr()[P_utri_k] < KKT_i && P_utri_k != P_utri_end) P_utri_k++;
                while (W_delta_inv_GT_G.innerIndexPtr()[GT_G_k] < KKT_i && GT_G_k != GT_G_end) GT_G_k++;

                if (data.P_utri.innerIndexPtr()[P_utri_k] == KKT_i && P_utri_k != P_utri_end)
                {
                    P_utri_to_Ki(P_utri_k) = KKT_k;
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
        auto& data = static_cast<Derived*>(this)->data;
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& PKi = static_cast<Derived*>(this)->PKi;
        auto& ordering = static_cast<Derived*>(this)->ordering;
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        // copy AT to PKPt
        isize n = data.AT.nonZeros();
        for (isize k = 0; k < n; k++)
        {
            PKPt.valuePtr()[PKi(AT_to_Ki(k))] = data.AT.valuePtr()[k];
        }

        // diagonal
        n = data.n + data.p;
        for (isize col = data.n; col < n; col++)
        {
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering.inv(col) + 1] - 1] = -m_delta;
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

#endif //PIQP_KKT_INEQ_ELIMINATED_HPP
