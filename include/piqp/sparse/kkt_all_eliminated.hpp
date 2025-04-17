// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_KKT_ALL_ELIMINATED_HPP
#define PIQP_SPARSE_KKT_ALL_ELIMINATED_HPP

#include "piqp/typedefs.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/sparse/data.hpp"

namespace piqp
{

namespace sparse
{

template<typename Derived, typename T, typename I>
class KKTImpl<Derived, T, I, KKTMode::KKT_ALL_ELIMINATED>
{
protected:
    SparseMat<T, I> A;
    SparseMat<T, I> G;
    SparseMat<T, I> AT_A;
    SparseMat<T, I> GT_W_delta_inv_G;
    Vec<T> tmp_scatter; // temporary storage for scatter operation

    Vec<I> P_utri_to_Ki; // mapping from P_utri row indices to KKT matrix
    Vec<I> AT_A_to_Ki;   // mapping from AT_A row indices to KKT matrix
    Vec<I> GT_G_to_Ki;   // mapping from GT_G row indices to KKT matrix

    void init_workspace(const Data<T, I>& data)
    {
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        A = data.AT.transpose();
        G = data.GT.transpose();

        AT_A = (data.AT * A).template triangularView<Eigen::Upper>();
        GT_W_delta_inv_G = (data.GT * G).template triangularView<Eigen::Upper>();
        T W_delta_inv = T(1) / (1 + m_delta);
        Eigen::Map<Vec<T>>(GT_W_delta_inv_G.valuePtr(), GT_W_delta_inv_G.nonZeros()).array() *= W_delta_inv;

        tmp_scatter.resize((std::max)(A.cols(), G.cols()));
        tmp_scatter.setZero();

        P_utri_to_Ki.resize(data.P_utri.nonZeros());
        AT_A_to_Ki.resize(AT_A.nonZeros());
        GT_G_to_Ki.resize(GT_W_delta_inv_G.nonZeros());
    }

    SparseMat<T, I> create_kkt_matrix(const Data<T, I>& data)
    {
        auto& m_delta = static_cast<Derived*>(this)->m_delta;

        T delta_inv = T(1) / m_delta;

        SparseMat<T, I> diagonal_rho;
        diagonal_rho.resize(data.n, data.n);
        diagonal_rho.setIdentity();

        SparseMat<T, I> KKT = data.P_utri + diagonal_rho + delta_inv * AT_A + GT_W_delta_inv_G;

        // compute mappings
        isize jj = KKT.outerSize();
        for (isize j = 0; j < jj; j++)
        {
            isize P_utri_k = data.P_utri.outerIndexPtr()[j];
            isize AT_A_k = AT_A.outerIndexPtr()[j];
            isize GT_G_k = GT_W_delta_inv_G.outerIndexPtr()[j];

            isize P_utri_end = data.P_utri.outerIndexPtr()[j + 1];
            isize AT_A_end = AT_A.outerIndexPtr()[j + 1];
            isize GT_G_end = GT_W_delta_inv_G.outerIndexPtr()[j + 1];

            isize KKT_kk = KKT.outerIndexPtr()[j + 1];
            for (isize KKT_k = KKT.outerIndexPtr()[j]; KKT_k < KKT_kk; KKT_k++)
            {
                isize KKT_i = KKT.innerIndexPtr()[KKT_k];

                while (P_utri_k != P_utri_end && data.P_utri.innerIndexPtr()[P_utri_k] < KKT_i) P_utri_k++;
                while (AT_A_k != AT_A_end && AT_A.innerIndexPtr()[AT_A_k] < KKT_i) AT_A_k++;
                while (GT_G_k != GT_G_end && GT_W_delta_inv_G.innerIndexPtr()[GT_G_k] < KKT_i) GT_G_k++;

                if (P_utri_k != P_utri_end && data.P_utri.innerIndexPtr()[P_utri_k] == KKT_i)
                {
                    P_utri_to_Ki(P_utri_k) = I(KKT_k);
                }
                if (AT_A_k != AT_A_end && AT_A.innerIndexPtr()[AT_A_k] == KKT_i)
                {
                    AT_A_to_Ki(AT_A_k) = I(KKT_k);
                }
                if (GT_G_k != GT_G_end && GT_W_delta_inv_G.innerIndexPtr()[GT_G_k] == KKT_i)
                {
                    GT_G_to_Ki(GT_G_k) = I(KKT_k);
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

        update_GT_W_delta_inv_G(data, z_reg);

        // copy GT * (W + delta)^{-1} * G to PKPt
        isize n = GT_W_delta_inv_G.nonZeros();
        for (isize k = 0; k < n; k++)
        {
            PKPt.valuePtr()[PKi(GT_G_to_Ki(k))] += GT_W_delta_inv_G.valuePtr()[k];
        }
    }

    void update_data_impl(const Data<T, I>& data, int options)
    {
        if (options & KKTUpdateOptions::KKT_UPDATE_A)
        {
            transpose_no_allocation<T, I>(data.AT, A);
            update_AT_A(data);
        }

        if (options & KKTUpdateOptions::KKT_UPDATE_G)
        {
            transpose_no_allocation<T, I>(data.GT, G);
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

    void update_GT_W_delta_inv_G(const Data<T, I>& data, const Vec<T>& z_reg)
    {
        // update GT * (W + delta)^{-1} * G
        isize n = G.outerSize();
        for (isize j = 0; j < n; j++)
        {
            for (typename SparseMat<T, I>::InnerIterator Gk_it(G, j); Gk_it; ++Gk_it)
            {
                I k = Gk_it.index();
                for (typename SparseMat<T, I>::InnerIterator GT_i_it(data.GT, k); GT_i_it; ++GT_i_it)
                {
                    if (GT_i_it.index() > j) continue;
                    tmp_scatter(GT_i_it.index()) += Gk_it.value() * GT_i_it.value() / z_reg(k);
                }
            }

            for (typename SparseMat<T, I>::InnerIterator GT_G_it(GT_W_delta_inv_G, j); GT_G_it; ++GT_G_it)
            {
                GT_G_it.valueRef() = tmp_scatter(GT_G_it.index());
                tmp_scatter(GT_G_it.index()) = 0;
            }
        }
    }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_KKT_ALL_ELIMINATED_HPP
