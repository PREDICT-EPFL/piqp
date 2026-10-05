// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_KKT_TPP
#define PIQP_DENSE_KKT_TPP

#include "piqp/dense/kkt.hpp"
#include "piqp/utils/tracy.hpp"

namespace piqp
{

namespace dense
{

template<typename T>
KKT<T>::KKT(const Data<T>& data)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::constructor");

    // init workspace
    m_z_reg_inv.resize(data.m);
    W_delta_inv_G.resize(data.m, data.n);
    work_z.resize(data.m);

    kkt_mat.setZero(data.n, data.n);
    // Eigen's LLT leaves m_info uninitialized until the first compute(), so copying
    // or move-assigning it before then is UB. Factorizing the zero matrix allocates
    // the storage and sets m_info, but exits at the first pivot, i.e., costs only O(n^2).
    llt.compute(kkt_mat);

    if (data.p > 0) {
        AT_A.resize(data.n, data.n);
        AT_A.template triangularView<Eigen::Lower>() = data.AT * data.AT.transpose();
    }
}

template<typename T>
std::unique_ptr<KKTSolverBase<T, int, PIQP_DENSE>> KKT<T>::clone() const
{
    return std::make_unique<KKT>(*this);
}

template<typename T>
void KKT<T>::update_data(const Data<T>& data, int options)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::update_data");

    if (options & KKTUpdateOptions::KKT_UPDATE_A) {
        if (data.p > 0) {
            AT_A.template triangularView<Eigen::Lower>() = data.AT * data.AT.transpose();
        }
    }
}

template<typename T>
bool KKT<T>::update_scalings_and_factor(const Data<T>& data, const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::update_scalings_and_factor");

    m_delta = delta;
    m_z_reg_inv.array() = z_reg.array().inverse();

    update_kkt(data, x_reg);

    llt.compute(kkt_mat);
    return llt.info() == Eigen::Success;
}

template<typename T>
void KKT<T>::solve(const Data<T>& data, const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z, Vec<T>& lhs_x, Vec<T>& lhs_y, Vec<T>& lhs_z)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::solve");

    T delta_inv = T(1) / m_delta;

    lhs_x = rhs_x;
    work_z.array() = m_z_reg_inv.array() * rhs_z.array();
    lhs_x.noalias() += data.GT * work_z;
    lhs_x.noalias() += delta_inv * data.AT * rhs_y;

    solve_ldlt_in_place(lhs_x);

    lhs_y.noalias() = delta_inv * data.AT.transpose() * lhs_x;
    lhs_y.noalias() -= delta_inv * rhs_y;

    lhs_z.noalias() = data.GT.transpose() * lhs_x;
    lhs_z.noalias() -= rhs_z;
    lhs_z.array() *= m_z_reg_inv.array();
}

template<typename T>
void KKT<T>::eval_P_x(const Data<T>& data, const T& alpha, const Vec<T>& x, Vec<T>& z)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::eval_P_x");

    z.noalias() = alpha * data.P_utri * x;
    z.noalias() += data.P_utri.transpose().template triangularView<Eigen::StrictlyLower>() * (alpha * x);
}

template<typename T>
void KKT<T>::eval_A_xn_and_AT_xt(const Data<T>& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::eval_A_xn_and_AT_xt");

    zn.noalias() = alpha_n * data.AT.transpose() * xn;
    zt.noalias() = alpha_t * data.AT * xt;
}

template<typename T>
void KKT<T>::eval_G_xn_and_GT_xt(const Data<T>& data, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::eval_G_xn_and_GT_xt");

    zn.noalias() = alpha_n * data.GT.transpose() * xn;
    zt.noalias() = alpha_t * data.GT * xt;
}

template<typename T>
void KKT<T>::update_kkt(const Data<T>& data, const Vec<T>& x_reg)
{
    {
        PIQP_TRACY_ZoneScopedN("piqp::KKT::update_scalings_and_factor::kkt_cost");
        kkt_mat.template triangularView<Eigen::Lower>() = data.P_utri.transpose();
        kkt_mat.diagonal() += x_reg;
    }

    if (data.p > 0)
    {
        PIQP_TRACY_ZoneScopedN("piqp::KKT::update_kkt::kkt_equality");
        kkt_mat.template triangularView<Eigen::Lower>() += T(1) / m_delta * AT_A;
    }

    if (data.m > 0)
    {
        PIQP_TRACY_ZoneScopedN("piqp::KKT::update_kkt::kkt_inequality");
        W_delta_inv_G = m_z_reg_inv.asDiagonal() * data.GT.transpose();
        kkt_mat.template triangularView<Eigen::Lower>() += data.GT * W_delta_inv_G;
    }
}

template<typename T>
void KKT<T>::solve_ldlt_in_place(Vec<T>& x)
{
    PIQP_TRACY_ZoneScopedN("piqp::KKT::solve_ldlt_in_place");

#ifdef PIQP_DEBUG_PRINT
    Vec<T> x_copy = x;
#endif

    llt.solveInPlace(x);

#ifdef PIQP_DEBUG_PRINT
    Vec<T> rhs_x = kkt_mat.template triangularView<Eigen::Lower>() * x;
    rhs_x += kkt_mat.transpose().template triangularView<Eigen::StrictlyUpper>() * x;
    std::cout << "llt_error: " << (x_copy - rhs_x).template lpNorm<Eigen::Infinity>() << std::endl;
#endif
}

} // namespace dense

} // namespace piqp

#endif //PIQP_DENSE_KKT_TPP
