// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_KKT_EQ_ELIMINATED_HPP
#define PIQP_SPARSE_KKT_EQ_ELIMINATED_HPP

#include "piqp/typedefs.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/sparse/data.hpp"

namespace piqp
{

namespace sparse
{

template<typename Derived, typename T, typename I>
class KKTImpl<Derived, T, I, KKTMode::KKT_EQ_ELIMINATED>
{
protected:
    SparseMat<T, I> A;
    SparseMat<T, I> AT_A;
    Vec<T> tmp_scatter; // temporary storage for scatter operation

    Vec<I> P_utri_to_Ki; // mapping from P_utri row indices to KKT matrix
    Vec<I> AT_A_to_Ki;   // mapping from AT_A row indices to KKT matrix
    Vec<I> GT_to_Ki;     // mapping from GT row indices to KKT matrix

    void init_workspace(const Data<T, I>& data)
    {
        A = data.AT.transpose();
        AT_A = (data.AT * A).template triangularView<Eigen::Upper>();

        tmp_scatter.resize(A.cols());
        tmp_scatter.setZero();

        P_utri_to_Ki.resize(data.P_utri.nonZeros());
        AT_A_to_Ki.resize(AT_A.nonZeros());
        GT_to_Ki.resize(data.GT.nonZeros());
    }

    SparseMat<T, I> create_kkt_matrix(const Data<T, I>& data)
    {
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        T delta_inv = T(1) / m_delta;

        SparseMat<T, I> diagonal_rho;
        diagonal_rho.resize(data.n, data.n);
        diagonal_rho.setIdentity();

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

                while (P_utri_k != P_utri_end && data.P_utri.innerIndexPtr()[P_utri_k] < KKT_i) P_utri_k++;
                while (AT_A_k != AT_A_end && AT_A.innerIndexPtr()[AT_A_k] < KKT_i) AT_A_k++;

                if (P_utri_k != P_utri_end && data.P_utri.innerIndexPtr()[P_utri_k] == KKT_i)
                {
                    P_utri_to_Ki(P_utri_k) = I(KKT_k);
                }
                if (AT_A_k != AT_A_end && AT_A.innerIndexPtr()[AT_A_k] == KKT_i)
                {
                    AT_A_to_Ki(AT_A_k) = I(KKT_k);
                }
            }
        }

        return KKT;
    }

    void update_kkt_cost_scalings(const Data<T, I>& data, const Vec<T>& x_reg)
    {
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& PKi = static_cast<Derived*>(this)->PKi;
        auto& ordering = static_cast<Derived*>(this)->ordering;

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
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering.inv(col) + 1] - 1] += x_reg[col];
        }
    }

    void update_kkt_equality_scalings(const Data<T, I>&)
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

    void update_kkt_inequality_scaling(const Data<T, I>& data, const Vec<T>& z_reg)
    {
        auto& PKPt = static_cast<Derived*>(this)->PKPt;
        auto& PKi = static_cast<Derived*>(this)->PKi;
        auto& ordering = static_cast<Derived*>(this)->ordering;

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
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering.inv(col) + 1] - 1] = -z_reg(k);
            k++;
        }
    }

    void update_data_impl(const Data<T, I>& data, int options)
    {
        if (options & KKTUpdateOptions::KKT_UPDATE_A)
        {
            transpose_no_allocation<T, I>(data.AT, A);
            update_AT_A(data);
        }
    }

    void update_AT_A(const Data<T, I>& data)
    {
        // update AT * A
        isize n = A.outerSize();
        for (isize j = 0; j < n; j++)
        {
            for (typename SparseMat<T, I>::InnerIterator Ak_it(A, j); Ak_it; ++Ak_it)
            {
                I k = Ak_it.index();
                for (typename SparseMat<T, I>::InnerIterator AT_i_it(data.AT, k); AT_i_it; ++AT_i_it)
                {
                    if (AT_i_it.index() > j) continue;
                    tmp_scatter(AT_i_it.index()) += Ak_it.value() * AT_i_it.value();
                }
            }

            for (typename SparseMat<T, I>::InnerIterator AT_A_it(AT_A, j); AT_A_it; ++AT_A_it)
            {
                AT_A_it.valueRef() = tmp_scatter(AT_A_it.index());
                tmp_scatter(AT_A_it.index()) = 0;
            }
        }
    }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_KKT_EQ_ELIMINATED_HPP
