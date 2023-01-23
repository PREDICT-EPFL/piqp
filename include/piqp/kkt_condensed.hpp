// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_KKT_CONDENSED_HPP
#define PIQP_KKT_CONDENSED_HPP

#include "piqp/data.hpp"
#include "piqp/ldlt.hpp"
#include "piqp/ordering.hpp"
#include "piqp/utils/sparse_utils.hpp"

namespace piqp
{

template<typename T, typename I, typename Ordering = AMDOrdering<I>>
struct KKTCondensed
{
    Data<T, I>& data;

    T m_rho;
    T m_delta;

    Vec<T> m_s;
    Vec<T> m_z_inv;

    Vec<I> P_utri_to_Ki; // mapping from P_utri row indices to KKT matrix
    Vec<I> AT_A_to_Ki;   // mapping from AT_A row indices to KKT matrix
    Vec<I> GT_G_to_Ki;   // mapping from GT_G row indices to KKT matrix

    SparseMat<T, I> AT_A;
    SparseMat<T, I> W_delta_inv_GT_G;

    Ordering ordering;
    SparseMat<T, I> PKPt; // permuted KKT matrix, upper triangular only
    Vec<I> PKi; // mapping of row indices of KKT matrix to permuted KKT matrix

    LDLt<T, I> ldlt;

    Vec<T> rhs_z_bar; // temporary variable needed for back solve
    Vec<T> rhs_perm;  // permuted rhs for back solve
    Vec<T> rhs;       // stores the rhs and the solution for back solve

    explicit KKTCondensed(Data<T, I>& data) : data(data) {}

    void init_kkt(const T& rho, const T& delta)
    {
        // init workspace
        m_s.resize(data.m);
        m_z_inv.resize(data.m);
        rhs_z_bar.resize(data.m);
        rhs_perm.resize(data.n);
        rhs.resize(data.n);

        m_rho = rho;
        m_delta = delta;
        m_s.setConstant(1);
        m_z_inv.setConstant(1);

        SparseMat<T, I> eye_rho(data.n, data.n);
        eye_rho.setIdentity();
        // set diagonal to rho
        Eigen::Map<Vec<T>>(eye_rho.valuePtr(), eye_rho.nonZeros()).setConstant(m_rho);

        AT_A = (data.AT * data.AT.transpose()).template triangularView<Eigen::Upper>();
        W_delta_inv_GT_G = (data.GT * data.GT.transpose()).template triangularView<Eigen::Upper>();

        T W_delta_inv = T(1) / (1 + m_delta);
        Eigen::Map<Vec<T>>(W_delta_inv_GT_G.valuePtr(), W_delta_inv_GT_G.nonZeros()).array() *= W_delta_inv;

        T delta_inv = T(1) / m_delta;
        SparseMat<T, I> KKT = data.P_utri + eye_rho + delta_inv * AT_A + W_delta_inv_GT_G;

        P_utri_to_Ki.resize(data.P_utri.nonZeros());
        AT_A_to_Ki.resize(AT_A.nonZeros());
        GT_G_to_Ki.resize(W_delta_inv_GT_G.nonZeros());

        for (isize j = 0; j < KKT.outerSize(); j++)
        {
            isize P_utri_k = data.P_utri.outerIndexPtr()[j];
            isize AT_A_k = AT_A.outerIndexPtr()[j];
            isize GT_G_k = W_delta_inv_GT_G.outerIndexPtr()[j];

            isize P_utri_end = data.P_utri.outerIndexPtr()[j + 1];
            isize AT_A_end = AT_A.outerIndexPtr()[j + 1];
            isize GT_G_end = W_delta_inv_GT_G.outerIndexPtr()[j + 1];

            for (isize KKT_k = KKT.outerIndexPtr()[j]; KKT_k < KKT.outerIndexPtr()[j + 1]; KKT_k++)
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

        ordering.init(KKT);
        PKi = permute_sparse_symmetric_matrix(KKT, PKPt, ordering);

        ldlt.factorize_symbolic_upper_triangular(PKPt);
    }

    void update_kkt(const T& rho, const T& delta, const CVecRef<T>& s, const CVecRef<T>& z)
    {
        m_rho = rho;
        m_delta = delta;
        m_s = s;
        m_z_inv.array() = T(1) / z.array();

        // set PKPt to zero keeping pattern
        Eigen::Map<Vec<T>>(PKPt.valuePtr(), PKPt.nonZeros()).setZero();

        // add P_utri
        for (isize j = 0; j < data.P_utri.outerSize(); j++)
        {
            for (isize k = data.P_utri.outerIndexPtr()[j]; k < data.P_utri.outerIndexPtr()[j + 1]; k++)
            {
                PKPt.valuePtr()[PKi(P_utri_to_Ki(k))] = data.P_utri.valuePtr()[k];
            }
        }

        // we assume that PKPt is upper triangular and diagonal is set
        // hence we can directly address the diagonal from the outer index pointer
        isize n = PKPt.outerSize();
        for (isize col = 0; col < n; col++)
        {
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering[col] + 1] - 1] += m_rho;
        }

        add_delta_inv_AT_A_to_PKPt();
        update_W_delta_inv_GT_G();
        add_W_delta_inv_GT_G_to_PKPt();
    }

    void add_delta_inv_AT_A_to_PKPt()
    {
        // copy delta_inv * AT * A to PKPt
        T delta_inv = T(1) / m_delta;
        isize n = AT_A.nonZeros();
        for (isize k = 0; k < n; k++)
        {
            PKPt.valuePtr()[PKi(AT_A_to_Ki(k))] += delta_inv * AT_A.valuePtr()[k];
        }
    }

    void add_W_delta_inv_GT_G_to_PKPt()
    {
        // copy GT * (W + delta)^{-1} * G to PKPt
        isize n = W_delta_inv_GT_G.nonZeros();
        for (isize k = 0; k < n; k++)
        {
            PKPt.valuePtr()[PKi(GT_G_to_Ki(k))] += W_delta_inv_GT_G.valuePtr()[k];
        }
    }

    void update_AT_A()
    {
        static_assert(!decltype(PKPt)::IsRowMajor, "KKT has to be column major!");
        static_assert(!decltype(data.AT)::IsRowMajor, "AT has to be column major!");

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

    void update_W_delta_inv_GT_G()
    {
        static_assert(!decltype(PKPt)::IsRowMajor, "KKT has to be column major!");
        static_assert(!decltype(data.GT)::IsRowMajor, "GT has to be column major!");

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

    void multiply(const CVecRef<T>& delta_x, const CVecRef<T>& delta_y,
                  const CVecRef<T>& delta_z, const CVecRef<T>& delta_s,
                  VecRef<T> rhs_x, VecRef<T> rhs_y, VecRef<T> rhs_z, VecRef<T> rhs_s)
    {
        rhs_x.noalias() = data.P_utri.template triangularView<Eigen::StrictlyUpper>() * delta_x;
        rhs_x.noalias() += data.P_utri.transpose().template triangularView<Eigen::StrictlyLower>() * delta_x;
        rhs_x.array() += data.P_utri.diagonal().array() * delta_x.array();
        rhs_x.noalias() += m_rho * delta_x;
        rhs_x.noalias() += data.AT * delta_y + data.GT * delta_z;

        rhs_y.noalias() = data.AT.transpose() * delta_x;
        rhs_y.noalias() -= m_delta * delta_y;

        rhs_z.noalias() = data.GT.transpose() * delta_x;
        rhs_z.noalias() += delta_s;
        rhs_z.noalias() -= m_delta * delta_z;

        rhs_s.array() = m_s.array() * delta_z.array() + m_z_inv.array().cwiseInverse() * delta_s.array();
    }

    bool factorize_kkt()
    {
        isize n = ldlt.factorize_numeric_upper_triangular(PKPt);
        return n == data.n;
    }

    void solve(const CVecRef<T>& rhs_x, const CVecRef<T>& rhs_y, const CVecRef<T>& rhs_z, const CVecRef<T>& rhs_s,
               VecRef<T> delta_x, VecRef<T> delta_y, VecRef<T> delta_z, VecRef<T> delta_s)
    {
        T delta_inv = T(1) / m_delta;

        rhs_z_bar.array() = rhs_z.array() - m_z_inv.array() * rhs_s.array();
        rhs_z_bar.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
        rhs.noalias() = rhs_x + data.GT * rhs_z_bar;
        rhs.noalias() += delta_inv * data.AT * rhs_y;

        ordering.template perm<T>(rhs_perm, rhs);
        ldlt.lsolve(rhs_perm);
        ldlt.dsolve(rhs_perm);
        ldlt.ltsolve(rhs_perm);
        ordering.template permt<T>(delta_x, rhs_perm);

        delta_y.noalias() = delta_inv * data.AT.transpose() * delta_x;
        delta_y.noalias() -= delta_inv * rhs_y;
        delta_z.noalias() = data.GT.transpose() * delta_x;
        delta_z.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
        delta_z.noalias() -= rhs_z_bar;
        delta_s.array() = m_z_inv.array() * (rhs_s.array() - m_s.array() * delta_z.array());
    }
};

} // namespace piqp

#endif //PIQP_KKT_CONDENSED_HPP
