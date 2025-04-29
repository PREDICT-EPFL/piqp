// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_KKT_FULL_HPP
#define PIQP_SPARSE_KKT_FULL_HPP

#include "piqp/typedefs.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/sparse/data.hpp"

namespace piqp
{

namespace sparse
{

template<typename Derived, typename T, typename I>
class KKTImpl<Derived, T, I, KKTMode::KKT_FULL>
{
protected:
    Vec<I> P_utri_to_Ki; // mapping from P_utri row indices to KKT matrix
    Vec<T> P_diagonal;   // diagonal of P
    Vec<I> AT_to_Ki;     // mapping from AT row indices to KKT matrix
    Vec<I> GT_to_Ki;     // mapping from GT row indices to KKT matrix

    void init_workspace(const Data<T, I>& data)
    {
        P_utri_to_Ki.resize(data.P_utri.nonZeros());
        P_diagonal.resize(data.n); P_diagonal.setZero();
        AT_to_Ki.resize(data.AT.nonZeros());
        GT_to_Ki.resize(data.GT.nonZeros());
    }

    SparseMat<T, I> create_kkt_matrix(const Data<T, I>& data)
    {
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        isize n_kkt = data.n + data.p + data.m;
        SparseMat<T, I> KKT(n_kkt, n_kkt);

        // count non-zeros
        isize non_zeros = 0;
        isize j_kkt = 0;
        isize jj = data.P_utri.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            isize col_nnz = data.P_utri.outerIndexPtr()[j + 1] - data.P_utri.outerIndexPtr()[j];
            if (col_nnz > 0)
            {
                isize last_col_index = data.P_utri.innerIndexPtr()[data.P_utri.outerIndexPtr()[j + 1] - 1];
                // if the last element in the column is not the diagonal element
                // then we need to add one more non-zero element
                if (last_col_index != j) {
                    col_nnz += 1;
                }
            }
            else
            {
                // if the column is empty, then we need to add one non-zero element for the diagonal element
                col_nnz += 1;
            }
            non_zeros += col_nnz;
            j_kkt++;
            KKT.outerIndexPtr()[j_kkt] = I(non_zeros);
        }
        jj = data.AT.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            non_zeros += data.AT.outerIndexPtr()[j + 1] - data.AT.outerIndexPtr()[j];
            non_zeros++; // add one for the diagonal element
            j_kkt++;
            KKT.outerIndexPtr()[j_kkt] = I(non_zeros);
        }
        jj = data.GT.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            non_zeros += data.GT.outerIndexPtr()[j + 1] - data.GT.outerIndexPtr()[j];
            non_zeros++; // add one for the diagonal element
            j_kkt++;
            KKT.outerIndexPtr()[j_kkt] = I(non_zeros);
        }
        KKT.resizeNonZeros(non_zeros);

        j_kkt = 0;
        // copy the upper triangular part of P and the diagonal
        jj = data.P_utri.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            isize k_kkt = KKT.outerIndexPtr()[j_kkt];
            isize col_nnz = data.P_utri.outerIndexPtr()[j + 1] - data.P_utri.outerIndexPtr()[j];
            Eigen::Map<Vec<I>>(KKT.innerIndexPtr() + k_kkt, col_nnz) = Eigen::Map<const Vec<I>>(data.P_utri.innerIndexPtr() + data.P_utri.outerIndexPtr()[j], col_nnz);
            Eigen::Map<Vec<T>>(KKT.valuePtr() + k_kkt, col_nnz) = Eigen::Map<const Vec<T>>(data.P_utri.valuePtr() + data.P_utri.outerIndexPtr()[j], col_nnz);

            // diagonal
            isize kkt_col_nnz = KKT.outerIndexPtr()[j_kkt + 1] - KKT.outerIndexPtr()[j_kkt];
            if (kkt_col_nnz > col_nnz)
            {
                KKT.innerIndexPtr()[k_kkt + kkt_col_nnz - 1] = I(j_kkt);
                KKT.valuePtr()[k_kkt + kkt_col_nnz - 1] = T(1); // dummy value for rho
            }
            else
            {
                P_diagonal[j] = data.P_utri.valuePtr()[data.P_utri.outerIndexPtr()[j + 1] - 1];
                KKT.valuePtr()[k_kkt + kkt_col_nnz - 1] += T(1); // dummy value for rho
            }

            isize i = 0;
            isize kk = data.P_utri.outerIndexPtr()[j + 1];
            for (isize k = data.P_utri.outerIndexPtr()[j]; k < kk; k++)
            {
                P_utri_to_Ki[k] = I(k_kkt + i);
                i++;
            }

            j_kkt++;
        }
        // copy AT and the diagonal
        jj = data.AT.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            isize k_kkt = KKT.outerIndexPtr()[j_kkt];
            isize col_nnz = data.AT.outerIndexPtr()[j + 1] - data.AT.outerIndexPtr()[j];
            Eigen::Map<Vec<I>>(KKT.innerIndexPtr() + k_kkt, col_nnz) = Eigen::Map<const Vec<I>>(data.AT.innerIndexPtr() + data.AT.outerIndexPtr()[j], col_nnz);
            Eigen::Map<Vec<T>>(KKT.valuePtr() + k_kkt, col_nnz) = Eigen::Map<const Vec<T>>(data.AT.valuePtr() + data.AT.outerIndexPtr()[j], col_nnz);

            // diagonal
            KKT.innerIndexPtr()[k_kkt + col_nnz] = I(j_kkt);
            KKT.valuePtr()[k_kkt + col_nnz] = -m_delta;

            isize i = 0;
            isize kk = data.AT.outerIndexPtr()[j + 1];
            for (isize k = data.AT.outerIndexPtr()[j]; k < kk; k++)
            {
                AT_to_Ki[k] = I(k_kkt + i);
                i++;
            }

            j_kkt++;
        }
        // copy GT and the diagonal
        jj = data.GT.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            isize k_kkt = KKT.outerIndexPtr()[j_kkt];
            isize col_nnz = data.GT.outerIndexPtr()[j + 1] - data.GT.outerIndexPtr()[j];
            Eigen::Map<Vec<I>>(KKT.innerIndexPtr() + k_kkt, col_nnz) = Eigen::Map<const Vec<I>>(data.GT.innerIndexPtr() + data.GT.outerIndexPtr()[j], col_nnz);
            Eigen::Map<Vec<T>>(KKT.valuePtr() + k_kkt, col_nnz) = Eigen::Map<const Vec<T>>(data.GT.valuePtr() + data.GT.outerIndexPtr()[j], col_nnz);

            // diagonal
            KKT.innerIndexPtr()[k_kkt + col_nnz] = I(j_kkt);
            KKT.valuePtr()[k_kkt + col_nnz] = -T(1) - m_delta;

            isize i = 0;
            isize kk = data.GT.outerIndexPtr()[j + 1];
            for (isize k = data.GT.outerIndexPtr()[j]; k < kk; k++)
            {
                GT_to_Ki[k] = I(k_kkt + i);
                i++;
            }

            j_kkt++;
        }

        return KKT;
    }

    void update_kkt_cost_scalings(const Data<T, I>& data, const CVecRef<T>& x_reg)
    {
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& ordering = static_cast<Derived*>(this)->ordering;

        // we assume that PKPt is upper triangular and diagonal is set
        // hence we can directly address the diagonal from the outer index pointer
        for (isize col = 0; col < data.n; col++)
        {
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering.inv(col) + 1] - 1] = P_diagonal[col] + x_reg[col];
        }
    }

    void update_kkt_equality_scalings(const Data<T, I>& data)
    {
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& ordering = static_cast<Derived*>(this)->ordering;
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        isize n = data.n + data.p;
        for (isize col = data.n; col < n; col++)
        {
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering.inv(col) + 1] - 1] = -m_delta;
        }
    }

    void update_kkt_inequality_scaling(const Data<T, I>& data, const CVecRef<T>& z_reg)
    {
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& ordering = static_cast<Derived*>(this)->ordering;

        isize n = data.n + data.p + data.m;
        isize k = 0;
        for (isize col = data.n + data.p; col < n; col++)
        {
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering.inv(col) + 1] - 1] = -z_reg(k);
            k++;
        }
    }

    void update_data_impl(const Data<T, I>& data, int options)
    {
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& PKi = static_cast<Derived*>(this)->PKi;

        if (options & KKTUpdateOptions::KKT_UPDATE_P)
        {
            isize jj = data.P_utri.outerSize();
            for (isize j = 0; j < jj; j++)
            {
                isize kk = data.P_utri.outerIndexPtr()[j + 1];
                for (isize k = data.P_utri.outerIndexPtr()[j]; k < kk; k++)
                {
                    PKPt.valuePtr()[PKi(P_utri_to_Ki(k))] = data.P_utri.valuePtr()[k];
                    if (j == data.P_utri.innerIndexPtr()[k])
                    {
                        P_diagonal[j] = data.P_utri.valuePtr()[k];
                    }
                }
            }
        }

        if (options & KKTUpdateOptions::KKT_UPDATE_A)
        {
            isize n = data.AT.nonZeros();
            for (isize k = 0; k < n; k++)
            {
                PKPt.valuePtr()[PKi(AT_to_Ki(k))] = data.AT.valuePtr()[k];
            }
        }

        if (options & KKTUpdateOptions::KKT_UPDATE_G)
        {
            isize n = data.GT.nonZeros();
            for (isize k = 0; k < n; k++)
            {
                PKPt.valuePtr()[PKi(GT_to_Ki(k))] = data.GT.valuePtr()[k];
            }
        }
    }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_KKT_FULL_HPP
