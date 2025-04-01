// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_KKT_HPP
#define PIQP_SPARSE_KKT_HPP

#include "piqp/settings.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/ldlt.hpp"
#include "piqp/sparse/ordering.hpp"
#include "piqp/sparse/utils.hpp"
#include "piqp/kkt_solver_base.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/sparse/kkt_full.hpp"
#include "piqp/sparse/kkt_eq_eliminated.hpp"
#include "piqp/sparse/kkt_ineq_eliminated.hpp"
#include "piqp/sparse/kkt_all_eliminated.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I, int Mode = KKTMode::KKT_FULL, typename Ordering = AMDOrdering<I>>
class KKT : public KKTSolverBase<T>, public KKTImpl<KKT<T, I, Mode, Ordering>, T, I, Mode>
{
protected:
    friend class KKTImpl<KKT<T, I, Mode, Ordering>, T, I, Mode>;

    const Data<T, I>& data;
    const Settings<T>& settings;

    T m_delta;

    Vec<T> m_z_reg_inv;

    Ordering ordering;
    SparseMat<T, I> PKPt; // permuted KKT matrix, upper triangular only
    Vec<I> PKi;           // mapping of row indices of KKT matrix to permuted KKT matrix

    LDLt<T, I> ldlt;

    Vec<T> work_z;        // working variable
    Vec<T> rhs;           // stores the rhs and the solution
    Vec<T> rhs_perm;      // permuted rhs

public:
    KKT(const Data<T, I>& data, const Settings<T>& settings) : data(data), settings(settings)
    {
        isize n_kkt = kkt_size();

        // init workspace
        m_z_reg_inv.resize(data.m);
        work_z.resize(data.m);
        rhs.resize(n_kkt);
        rhs_perm.resize(n_kkt);

        this->init_workspace();
        SparseMat<T, I> KKT = this->create_kkt_matrix();

        ordering.init(KKT);
        PKi = permute_sparse_symmetric_matrix(KKT, PKPt, ordering);

        ldlt.factorize_symbolic_upper_triangular(PKPt);
    }

    void update_data(int options) override
    {
        this->update_data_impl(options);
    }

    bool update_scalings_and_factor(const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg) override
    {
        m_delta = delta;
        m_z_reg_inv.array() = z_reg.array().inverse();

        this->update_kkt_cost_scalings(x_reg);
        this->update_kkt_equality_scalings();
        this->update_kkt_inequality_scaling(z_reg);

        isize n = ldlt.factorize_numeric_upper_triangular(PKPt);
        return n == PKPt.cols();
    }

    void solve(const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z,
               Vec<T>& delta_x, Vec<T>& delta_y, Vec<T>& delta_z) override
    {
        T delta_inv = T(1) / m_delta;

        rhs.head(data.n).noalias() = rhs_x;
        if (Mode == KKTMode::KKT_FULL)
        {
            rhs.segment(data.n, data.p).noalias() = rhs_y;
            rhs.tail(data.m).noalias() = rhs_z;
        }
        else if (Mode == KKTMode::KKT_EQ_ELIMINATED)
        {
            rhs.head(data.n).noalias() += delta_inv * data.AT * rhs_y;
            rhs.tail(data.m).noalias() = rhs_z;
        }
        else if (Mode == KKTMode::KKT_INEQ_ELIMINATED)
        {
            work_z.array() = m_z_reg_inv.array() * rhs_z.array();
            rhs.head(data.n).noalias() += data.GT * work_z;
            rhs.tail(data.p).noalias() = rhs_y;
        }
        else
        {
            work_z.array() = m_z_reg_inv.array() * rhs_z.array();
            rhs.noalias() += data.GT * work_z;
            rhs.noalias() += delta_inv * data.AT * rhs_y;
        }

        ordering.template perm<T>(rhs_perm, rhs);

        solve_ldlt_in_place(rhs_perm);

        // we reuse the memory of rhs for the solution
        ordering.template permt<T>(rhs, rhs_perm);

        delta_x.noalias() = rhs.head(data.n);

        if (Mode == KKTMode::KKT_FULL)
        {
            delta_y.noalias() = rhs.segment(data.n, data.p);

            delta_z.noalias() = rhs.tail(data.m);
        }
        else if (Mode == KKTMode::KKT_EQ_ELIMINATED)
        {
            delta_y.noalias() = delta_inv * data.AT.transpose() * delta_x;
            delta_y.noalias() -= delta_inv * rhs_y;

            delta_z.noalias() = rhs.tail(data.m);
        }
        else if (Mode == KKTMode::KKT_INEQ_ELIMINATED)
        {
            delta_y.noalias() = rhs.tail(data.p);

            delta_z.noalias() = data.GT.transpose() * delta_x;
            delta_z.noalias() -= rhs_z;
            delta_z.array() *= m_z_reg_inv.array();
        }
        else
        {
            delta_y.noalias() = delta_inv * data.AT.transpose() * delta_x;
            delta_y.noalias() -= delta_inv * rhs_y;

            delta_z.noalias() = data.GT.transpose() * delta_x;
            delta_z.noalias() -= rhs_z;
            delta_z.array() *= m_z_reg_inv.array();
        }
    }

    // z = alpha * P * x
    void eval_P_x(const T& alpha, const Vec<T>& x, Vec<T>& z) override
    {
        z.noalias() = alpha * data.P_utri * x;
        z.noalias() += alpha * data.P_utri.transpose().template triangularView<Eigen::StrictlyLower>() * x;
    }

    // zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
    void eval_A_xn_and_AT_xt(const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) override
    {
        zn.noalias() = alpha_n * data.AT.transpose() * xn;
        zt.noalias() = alpha_t * data.AT * xt;
    }

    // zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
    void eval_G_xn_and_GT_xt(const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) override
    {
        zn.noalias() = alpha_n * data.GT.transpose() * xn;
        zt.noalias() = alpha_t * data.GT * xt;
    }

    SparseMat<T, I>& internal_kkt_mat()
    {
        return PKPt;
    }

protected:
    inline isize kkt_size()
    {
        isize n_kkt;
        if (Mode == KKTMode::KKT_FULL)
        {
            n_kkt = data.n + data.p + data.m;
        }
        else if (Mode == KKTMode::KKT_EQ_ELIMINATED)
        {
            n_kkt = data.n + data.m;
        }
        else if (Mode == KKTMode::KKT_INEQ_ELIMINATED)
        {
            n_kkt = data.n + data.p;
        }
        else
        {
            n_kkt = data.n;
        }
        return n_kkt;
    }

    void solve_ldlt_in_place(Vec<T>& x)
    {
#ifdef PIQP_DEBUG_PRINT
        Vec<T> x_copy = x;
#endif

        ldlt.solve_inplace(x);

#ifdef PIQP_DEBUG_PRINT
        Vec<T> rhs_x = PKPt.template triangularView<Eigen::Upper>() * x;
        rhs_x += PKPt.transpose().template triangularView<Eigen::StrictlyLower>() * x;
        std::cout << "ldlt_error: " << (x_copy - rhs_x).template lpNorm<Eigen::Infinity>() << std::endl;
#endif
    }
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/sparse/kkt.tpp"
#endif

#endif //PIQP_SPARSE_KKT_HPP
