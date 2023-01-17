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

namespace piqp
{

template<typename T, typename I>
struct KKTCondensed
{
    Data<T, I>& data;

    SparseMat<T, I> KKT;
    LDLt<T, I> ldlt;

    Vec<T> rhs;

    explicit KKTCondensed(Data<T, I>& data) : data(data), rhs(data.P_utri.cols()) {}

    void init_kkt(const T& rho, const T& delta)
    {
        SparseMat<T, I> eye_rho(data.P_utri.cols(), data.P_utri.cols());
        eye_rho.setIdentity();
        // set diagonal to rho
        Eigen::Map<Vec<T>>(eye_rho.valuePtr(), eye_rho.nonZeros()).setConstant(rho);

        SparseMat<T, I> AT_A = data.AT * data.AT.transpose();
        SparseMat<T, I> GT_G = data.GT * data.GT.transpose();

        T delta_inv = T(1) / delta;
        Eigen::Map<Vec<T>>(AT_A.valuePtr(), AT_A.nonZeros()).array() *= delta_inv;
        T W_delta_inv = T(1) / (1 + delta);
        Eigen::Map<Vec<T>>(GT_G.valuePtr(), GT_G.nonZeros()).array() *= W_delta_inv;

        KKT = data.P_utri + eye_rho + AT_A.template triangularView<Eigen::Upper>() + GT_G.template triangularView<Eigen::Upper>();

        // TODO: init ldlt
    }

    void update_kkt(const T& rho, const T& delta, const CVecRef<T>& s, const CVecRef<T>& z)
    {
        // set KKT to zero keeping pattern
        Eigen::Map<Vec<T>>(KKT.valuePtr(), KKT.nonZeros()).setZero();

        KKT += data.P_utri;

        // we assume that KKT is upper traditional and diagonal is set
        // hence we can directly address the diagonal from the outer index pointer
        for (isize col = 0; col < KKT.outerSize(); col++)
        {
            *(KKT.valuePtr() + *(KKT.outerIndexPtr() + col + 1) - 1) += rho;
        }

        static_assert(!decltype(KKT)::IsRowMajor, "KKT has to be column major!");
        static_assert(!decltype(data.AT)::IsRowMajor, "AT has to be column major!");
        static_assert(!decltype(data.GT)::IsRowMajor, "GT has to be column major!");

        // add delta_inv * AT * A
        T delta_inv = T(1) / delta;
        for (isize k = 0; k < data.AT.outerSize(); k++)
        {
            for (typename SparseMat<T, I>::InnerIterator AT_j_it(data.AT, k); AT_j_it; ++AT_j_it)
            {
                I j = AT_j_it.index();
                typename SparseMat<T, I>::InnerIterator KKT_i_it(KKT, j);
                typename SparseMat<T, I>::InnerIterator AT_i_it(data.AT, k);
                while (KKT_i_it && AT_i_it)
                {
                    if (KKT_i_it.index() < AT_i_it.index())
                    {
                        ++KKT_i_it;
                    }
                    else
                    {
                        eigen_assert(KKT_i_it.index() == AT_i_it.index() && "KKT is missing entry!");

                        KKT_i_it.valueRef() += delta_inv * AT_i_it.value() * AT_j_it.value();
                        ++KKT_i_it;
                        ++AT_i_it;
                    }
                }
            }
        }

        // add GT * (W + delta) * G
        for (isize k = 0; k < data.GT.outerSize(); k++)
        {
            for (typename SparseMat<T, I>::InnerIterator GT_j_it(data.GT, k); GT_j_it; ++GT_j_it)
            {
                I j = GT_j_it.index();
                typename SparseMat<T, I>::InnerIterator KKT_i_it(KKT, j);
                typename SparseMat<T, I>::InnerIterator GT_i_it(data.GT, k);
                while (KKT_i_it && GT_i_it)
                {
                    if (KKT_i_it.index() < GT_i_it.index())
                    {
                        ++KKT_i_it;
                    }
                    else
                    {
                        eigen_assert(KKT_i_it.index() == GT_i_it.index() && "KKT is missing entry!");

                        T W_delta_inv = T(1) / (s(k) / z(k) + delta);
                        KKT_i_it.valueRef() += W_delta_inv * GT_i_it.value() * GT_j_it.value();
                        ++KKT_i_it;
                        ++GT_i_it;
                    }
                }
            }
        }
    }
};

} // namespace piqp

#endif //PIQP_KKT_CONDENSED_HPP
