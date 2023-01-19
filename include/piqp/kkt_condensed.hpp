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
#include "piqp/utils/sparse.hpp"

namespace piqp
{

template<typename T, typename I, typename Ordering = AMDOrdering<I>>
struct KKTCondensed
{
    Data<T, I>& data;

    Vec<I> P_utri_to_Ki; // mapping from P_utri row indices to KKT matrix
    Vec<I> AT_A_to_Ki;   // mapping from AT_A row indices to KKT matrix
    Vec<I> GT_G_to_Ki;   // mapping from GT_G row indices to KKT matrix

    SparseMat<T, I> AT_A;
    SparseMat<T, I> GT_G;

    Ordering ordering;
    SparseMat<T, I> PKPt; // permuted KKT matrix, upper triangular only
    Vec<I> PKi; // mapping of row indices of KKT matrix to permuted KKT matrix

    LDLt<T, I> ldlt;

    Vec<T> rhs;

    explicit KKTCondensed(Data<T, I>& data) : data(data), rhs(data.P_utri.cols()) {}

    void init_kkt(const T& rho, const T& delta)
    {
        SparseMat<T, I> eye_rho(data.P_utri.cols(), data.P_utri.cols());
        eye_rho.setIdentity();
        // set diagonal to rho
        Eigen::Map<Vec<T>>(eye_rho.valuePtr(), eye_rho.nonZeros()).setConstant(rho);

        AT_A = (data.AT * data.AT.transpose()).template triangularView<Eigen::Upper>();
        GT_G = (data.GT * data.GT.transpose()).template triangularView<Eigen::Upper>();

        T delta_inv = T(1) / delta;
        Eigen::Map<Vec<T>>(AT_A.valuePtr(), AT_A.nonZeros()).array() *= delta_inv;
        T W_delta_inv = T(1) / (1 + delta);
        Eigen::Map<Vec<T>>(GT_G.valuePtr(), GT_G.nonZeros()).array() *= W_delta_inv;

        SparseMat<T, I> KKT = data.P_utri + eye_rho + AT_A + GT_G;

        P_utri_to_Ki.resize(data.P_utri.nonZeros());
        AT_A_to_Ki.resize(AT_A.nonZeros());
        GT_G_to_Ki.resize(GT_G.nonZeros());

        for (isize j = 0; j < KKT.outerSize(); j++)
        {
            isize P_utri_k = data.P_utri.outerIndexPtr()[j];
            isize AT_A_k = AT_A.outerIndexPtr()[j];
            isize GT_G_k = GT_G.outerIndexPtr()[j];

            isize P_utri_end = data.P_utri.outerIndexPtr()[j + 1];
            isize AT_A_end = AT_A.outerIndexPtr()[j + 1];
            isize GT_G_end = GT_G.outerIndexPtr()[j + 1];

            for (isize KKT_k = KKT.outerIndexPtr()[j]; KKT_k < KKT.outerIndexPtr()[j + 1]; KKT_k++)
            {
                isize KKT_i = KKT.innerIndexPtr()[KKT_k];

                while (data.P_utri.innerIndexPtr()[P_utri_k] < KKT_i && P_utri_k != P_utri_end) P_utri_k++;
                while (AT_A.innerIndexPtr()[AT_A_k] < KKT_i && AT_A_k != AT_A_end) AT_A_k++;
                while (GT_G.innerIndexPtr()[GT_G_k] < KKT_i && GT_G_k != GT_G_end) GT_G_k++;

                if (data.P_utri.innerIndexPtr()[P_utri_k] == KKT_i && P_utri_k != P_utri_end)
                {
                    P_utri_to_Ki(P_utri_k) = KKT_k;
                }
                if (AT_A.innerIndexPtr()[AT_A_k] == KKT_i && AT_A_k != AT_A_end)
                {
                    AT_A_to_Ki(AT_A_k) = KKT_k;
                }
                if (GT_G.innerIndexPtr()[GT_G_k] == KKT_i && GT_G_k != GT_G_end)
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
        for (isize col = 0; col < PKPt.outerSize(); col++)
        {
            PKPt.valuePtr()[PKPt.outerIndexPtr()[ordering[col] + 1] - 1] += rho;
        }

        static_assert(!decltype(PKPt)::IsRowMajor, "KKT has to be column major!");
        static_assert(!decltype(data.AT)::IsRowMajor, "AT has to be column major!");
        static_assert(!decltype(data.GT)::IsRowMajor, "GT has to be column major!");

        // update delta_inv * AT * A
        Eigen::Map<Vec<T>>(AT_A.valuePtr(), AT_A.nonZeros()).setZero();
        T delta_inv = T(1) / delta;
        for (isize k = 0; k < data.AT.outerSize(); k++)
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

                        AT_A_i_it.valueRef() += delta_inv * AT_i_it.value() * AT_j_it.value();
                        ++AT_A_i_it;
                        ++AT_i_it;
                    }
                }
            }
        }
        // copy delta_inv * AT * A to PKPt
        for (isize k = 0; k < AT_A.nonZeros(); k++)
        {
            PKPt.valuePtr()[PKi(AT_A_to_Ki(k))] += AT_A.valuePtr()[k];
        }

        // update GT * (W + delta)^{-1} * G
        Eigen::Map<Vec<T>>(GT_G.valuePtr(), GT_G.nonZeros()).setZero();
        for (isize k = 0; k < data.GT.outerSize(); k++)
        {
            for (typename SparseMat<T, I>::InnerIterator GT_j_it(data.GT, k); GT_j_it; ++GT_j_it)
            {
                I j = GT_j_it.index();
                typename SparseMat<T, I>::InnerIterator GT_G_i_it(GT_G, j);
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

                        T W_delta_inv = T(1) / (s(k) / z(k) + delta);
                        GT_G_i_it.valueRef() += W_delta_inv * GT_i_it.value() * GT_j_it.value();
                        ++GT_G_i_it;
                        ++GT_i_it;
                    }
                }
            }
        }
        // copy GT * (W + delta)^{-1} * G to PKPt
        for (isize k = 0; k < GT_G.nonZeros(); k++)
        {
            PKPt.valuePtr()[PKi(GT_G_to_Ki(k))] += GT_G.valuePtr()[k];
        }
    }
};

} // namespace piqp

#endif //PIQP_KKT_CONDENSED_HPP
