// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_KKT_EQ_ELIMINATED_HPP
#define PIQP_KKT_EQ_ELIMINATED_HPP

#include "piqp/typedefs.hpp"
#include "piqp/kkt_fwd.hpp"

namespace piqp
{

template<typename Derived, typename T, typename I>
struct KKTImpl<Derived, T, I, KKTMode::EQ_ELIMINATED>
{
    Vec<I> P_utri_to_Ki; // mapping from P_utri row indices to KKT matrix
    Vec<I> AT_A_to_Ki;   // mapping from AT_A row indices to KKT matrix
    Vec<I> GT_to_Ki;     // mapping from GT row indices to KKT matrix

    SparseMat<T, I> AT_A;

    void init_workspace()
    {
        auto& data = static_cast<Derived*>(this)->data;
        AT_A = (data.AT * data.AT.transpose()).template triangularView<Eigen::Upper>();

        P_utri_to_Ki.resize(data.P_utri.nonZeros());
        AT_A_to_Ki.resize(AT_A.nonZeros());
        GT_to_Ki.resize(data.GT.nonZeros());
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

        SparseMat<T, I> KKT_top_left_block = data.P_utri + diagonal_rho + delta_inv * AT_A;

        isize n_kkt = data.n + data.m;
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
        jj = data.GT.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            non_zeros += data.GT.outerIndexPtr()[j + 1] - data.GT.outerIndexPtr()[j];
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
        // copy GT and the diagonal
        jj = data.GT.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            isize k_kkt = KKT.outerIndexPtr()[j_kkt];
            isize col_nnz = data.GT.outerIndexPtr()[j + 1] - data.GT.outerIndexPtr()[j];
            Eigen::Map<Vec<I>>(KKT.innerIndexPtr() + k_kkt, col_nnz) = Eigen::Map<Vec<I>>(data.GT.innerIndexPtr() + data.GT.outerIndexPtr()[j], col_nnz);
            Eigen::Map<Vec<T>>(KKT.valuePtr() + k_kkt, col_nnz) = Eigen::Map<Vec<T>>(data.GT.valuePtr() + data.GT.outerIndexPtr()[j], col_nnz);

            // diagonal
            KKT.innerIndexPtr()[k_kkt + col_nnz] = j_kkt;
            KKT.valuePtr()[k_kkt + col_nnz] = -T(1) - m_delta;

            isize i = 0;
            isize kk = data.GT.outerIndexPtr()[j + 1];
            for (isize k = data.GT.outerIndexPtr()[j]; k < kk; k++)
            {
                GT_to_Ki[k] = k_kkt + i;
                i++;
            }

            j_kkt++;
        }

        // compute remaining mappings
        jj = data.n;
        for (isize j = 0; j < jj; j++)
        {
            isize P_utri_k = data.P_utri.outerIndexPtr()[j];
            isize AT_A_k = AT_A.outerIndexPtr()[j];

            isize P_utri_end = data.P_utri.outerIndexPtr()[j + 1];
            isize AT_A_end = AT_A.outerIndexPtr()[j + 1];

            isize KKT_kk = KKT.outerIndexPtr()[j + 1];
            for (isize KKT_k = KKT.outerIndexPtr()[j]; KKT_k < KKT_kk; KKT_k++)
            {
                isize KKT_i = KKT.innerIndexPtr()[KKT_k];

                while (data.P_utri.innerIndexPtr()[P_utri_k] < KKT_i && P_utri_k != P_utri_end) P_utri_k++;
                while (AT_A.innerIndexPtr()[AT_A_k] < KKT_i && AT_A_k != AT_A_end) AT_A_k++;

                if (data.P_utri.innerIndexPtr()[P_utri_k] == KKT_i && P_utri_k != P_utri_end)
                {
                    P_utri_to_Ki(P_utri_k) = KKT_k;
                }
                if (AT_A.innerIndexPtr()[AT_A_k] == KKT_i && AT_A_k != AT_A_end)
                {
                    AT_A_to_Ki(AT_A_k) = KKT_k;
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
        auto& data = static_cast<Derived*>(this)->data;
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& PKi = static_cast<Derived*>(this)->PKi;
        auto& ordering = static_cast<Derived*>(this)->ordering;
        auto& m_delta = static_cast<Derived*>(this)->m_delta;
        auto& m_s = static_cast<Derived*>(this)->m_s;
        auto& m_z_inv = static_cast<Derived*>(this)->m_z_inv;

        // copy GT to PKPt
        isize n = data.GT.nonZeros();
        for (isize k = 0; k < n; k++)
        {
            PKPt.valuePtr()[PKi(GT_to_Ki(k))] = data.GT.valuePtr()[k];
        }

        // diagonal
        n = data.n + data.m;
        isize k = 0;
        for (isize col = data.n; col < n; col++)
        {
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering.inv(col) + 1] - 1] = -m_s(k) * m_z_inv(k) - m_delta;
            k++;
        }
    }
};

} // namespace piqp

#endif //PIQP_KKT_EQ_ELIMINATED_HPP
