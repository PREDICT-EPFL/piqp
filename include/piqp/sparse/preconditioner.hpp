// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_PRECONDITIONER_HPP
#define PIQP_SPARSE_PRECONDITIONER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/typedefs.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/utils.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
class RuizEquilibration
{
    static constexpr T min_scaling = 1e-4;
    static constexpr T max_scaling = 1e4;

    isize n = 0;
    isize p = 0;
    isize m = 0;
    isize n_lb = 0;
    isize n_ub = 0;

    T c = T(1);
    Vec<T> delta;
    Vec<T> delta_lb;
    Vec<T> delta_ub;

    T c_inv = T(1);
    Vec<T> delta_inv;
    Vec<T> delta_lb_inv;
    Vec<T> delta_ub_inv;

public:
    void init(const Data<T, I>& data)
    {
        n = data.n;
        p = data.p;
        m = data.m;
        n_lb = data.n_lb;
        n_ub = data.n_ub;

        delta.resize(n + p + m);
        delta_lb.resize(n);
        delta_ub.resize(n);
        delta_inv.resize(n + p + m);
        delta_lb_inv.resize(n);
        delta_ub_inv.resize(n);

        c = T(1);
        delta.setConstant(1);
        delta_lb.setConstant(1);
        delta_ub.setConstant(1);
        c_inv = T(1);
        delta_inv.setConstant(1);
        delta_lb_inv.setConstant(1);
        delta_ub_inv.setConstant(1);
    }

    inline void scale_data(Data<T, I>& data, bool reuse_prev_scaling = false, bool scale_cost = false, isize max_iter = 10, T epsilon = T(1e-3))
    {
        n_lb = data.n_lb;
        n_ub = data.n_ub;

        if (!reuse_prev_scaling)
        {
            // init scaling in case max_iter is 0
            c = T(1);
            delta.setConstant(1);
            delta_lb.setConstant(1);
            delta_ub.setConstant(1);

            Vec<T>& delta_iter = delta_inv; // we use the memory of delta_inv as temporary storage
            Vec<T>& delta_iter_lb = delta_lb_inv; // we use the memory of delta_lb_inv as temporary storage
            Vec<T>& delta_iter_ub = delta_ub_inv; // we use the memory of delta_ub_inv as temporary storage
            delta_iter.setZero();
            for (isize i = 0; i < max_iter && std::max({
                    (1 - delta_iter.array()).matrix().template lpNorm<Eigen::Infinity>(),
                    (1 - delta_iter_lb.head(n_lb).array()).matrix().template lpNorm<Eigen::Infinity>(),
                    (1 - delta_iter_ub.head(n_ub).array()).matrix().template lpNorm<Eigen::Infinity>()
                }) > epsilon; i++)
            {
                delta_iter.setZero();

                // calculate scaling of full KKT matrix
                // [ P AT GT ]
                // [ A 0  0  ]
                // [ G 0  0  ]
                for (isize j = 0; j < n; j++)
                {
                    for (typename SparseMat<T, I>::InnerIterator P_utri_it(data.P_utri, j); P_utri_it; ++P_utri_it)
                    {
                        I i_row = P_utri_it.index();
                        delta_iter(j) = std::max(delta_iter(j), std::abs(P_utri_it.value()));
                        if (i_row != j)
                        {
                            delta_iter(i_row) = std::max(delta_iter(i_row), std::abs(P_utri_it.value()));
                        }
                    }
                }
                for (isize j = 0; j < p; j++)
                {
                    for (typename SparseMat<T, I>::InnerIterator AT_it(data.AT, j); AT_it; ++AT_it)
                    {
                        I i_row = AT_it.index();
                        delta_iter(i_row) = std::max(delta_iter(i_row), std::abs(AT_it.value()));
                        delta_iter(n + j) = std::max(delta_iter(n + j), std::abs(AT_it.value()));
                    }
                }
                for (isize j = 0; j < m; j++)
                {
                    for (typename SparseMat<T, I>::InnerIterator GT_it(data.GT, j); GT_it; ++GT_it)
                    {
                        I i_row = GT_it.index();
                        delta_iter(i_row) = std::max(delta_iter(i_row), std::abs(GT_it.value()));
                        delta_iter(n + p + j) = std::max(delta_iter(n + p + j), std::abs(GT_it.value()));
                    }
                }
                for (isize j = 0; j < n_lb; j++)
                {
                    delta_iter(data.x_lb_idx(j)) = std::max(delta_iter(data.x_lb_idx(j)), data.x_lb_scaling(j));
                    delta_iter_lb(j) = data.x_lb_scaling(j);
                }
                for (isize j = 0; j < n_ub; j++)
                {
                    delta_iter(data.x_ub_idx(j)) = std::max(delta_iter(data.x_ub_idx(j)), data.x_ub_scaling(j));
                    delta_iter_ub(j) = data.x_ub_scaling(j);
                }

                limit_scaling(delta_iter);
                limit_scaling(delta_iter_lb);
                limit_scaling(delta_iter_ub);

                delta_iter.array() = delta_iter.array().sqrt().inverse();
                delta_iter_lb.array() = delta_iter_lb.array().sqrt().inverse();
                delta_iter_ub.array() = delta_iter_ub.array().sqrt().inverse();

                // scale cost
                pre_mult_diagonal<T, I>(data.P_utri, delta_iter.head(n));
                post_mult_diagonal<T, I>(data.P_utri, delta_iter.head(n));
                data.c.array() *= delta_iter.head(n).array();

                // scale AT and GT
                pre_mult_diagonal<T, I>(data.AT, delta_iter.head(n));
                post_mult_diagonal<T, I>(data.AT, delta_iter.segment(n, p));
                pre_mult_diagonal<T, I>(data.GT, delta_iter.head(n));
                post_mult_diagonal<T, I>(data.GT, delta_iter.tail(m));

                // scale box scalings
                data.x_lb_scaling.head(n_lb).array() *= delta_iter_lb.head(n_lb).array();
                for (isize j = 0; j < n_lb; j++)
                {
                    data.x_lb_scaling(j) *= delta_iter(data.x_lb_idx(j));
                }
                data.x_ub_scaling.head(n_ub).array() *= delta_iter_ub.head(n_ub).array();
                for (isize j = 0; j < n_ub; j++)
                {
                    data.x_ub_scaling(j) *= delta_iter(data.x_ub_idx(j));
                }

                delta.array() *= delta_iter.array();
                delta_lb.head(n_lb).array() *= delta_iter_lb.head(n_lb).array();
                delta_ub.head(n_ub).array() *= delta_iter_ub.head(n_ub).array();

                if (scale_cost)
                {
                    // scaling for the cost
                    Vec<T>& delta_iter_cost = delta_lb_inv; // we use delta_lb_inv as a temporary storage
                    delta_iter_cost.setZero();
                    for (isize j = 0; j < n; j++)
                    {
                        for (typename SparseMat<T, I>::InnerIterator P_utri_it(data.P_utri, j); P_utri_it; ++P_utri_it)
                        {
                            I i_row = P_utri_it.index();
                            delta_iter_cost(j) = std::max(delta_iter_cost(j), std::abs(P_utri_it.value()));
                            if (i_row != j)
                            {
                                delta_iter_cost(i_row) = std::max(delta_iter_cost(i_row), std::abs(P_utri_it.value()));
                            }
                        }
                    }
                    T gamma = delta_iter_cost.sum() / T(n);
                    limit_scaling(gamma);
                    gamma = std::max(gamma, data.c.template lpNorm<Eigen::Infinity>());
                    limit_scaling(gamma);
                    gamma = T(1) / gamma;

                    // scale cost
                    data.P_utri *= gamma;
                    data.c *= gamma;

                    c *= gamma;
                }
            }

            c_inv = T(1) / c;
            delta_inv.array() = delta.array().inverse();
            delta_lb_inv.head(n_lb).array() = delta_lb.head(n_lb).array().inverse();
            delta_ub_inv.head(n_ub).array() = delta_ub.head(n_ub).array().inverse();
        }
        else
        {
            // scale cost
            data.P_utri *= c;
            pre_mult_diagonal<T, I>(data.P_utri, delta.head(n));
            post_mult_diagonal<T, I>(data.P_utri, delta.head(n));
            data.c.array() *= c * delta.head(n).array();

            // scale AT and GT
            pre_mult_diagonal<T, I>(data.AT, delta.head(n));
            post_mult_diagonal<T, I>(data.AT, delta.segment(n, p));
            pre_mult_diagonal<T, I>(data.GT, delta.head(n));
            post_mult_diagonal<T, I>(data.GT, delta.tail(m));

            // scale box scalings
            data.x_lb_scaling.head(n_lb).array() *= delta_lb.head(n_lb).array();
            for (isize j = 0; j < n_lb; j++)
            {
                data.x_lb_scaling(j) *= delta(data.x_lb_idx(j));
            }
            data.x_ub_scaling.head(n_ub).array() *= delta_ub.head(n_ub).array();
            for (isize j = 0; j < n_ub; j++)
            {
                data.x_ub_scaling(j) *= delta(data.x_ub_idx(j));
            }
        }

        // scale bounds
        data.b.array() *= delta.segment(n, p).array();
        data.h.array() *= delta.tail(m).array();
        data.x_lb_n.head(n_lb).array() *= delta_lb.head(n_lb).array();
        data.x_ub.head(n_ub).array() *= delta_ub.head(n_ub).array();
    }

    inline void unscale_data(Data<T, I>& data)
    {
        // unscale cost
        data.P_utri *= c_inv;
        pre_mult_diagonal<T, I>(data.P_utri, delta_inv.head(n));
        post_mult_diagonal<T, I>(data.P_utri, delta_inv.head(n));
        data.c.array() *= c_inv * delta_inv.head(n).array();

        // unscale AT and GT
        pre_mult_diagonal<T, I>(data.AT, delta_inv.head(n));
        post_mult_diagonal<T, I>(data.AT, delta_inv.segment(n, p));
        pre_mult_diagonal<T, I>(data.GT, delta_inv.head(n));
        post_mult_diagonal<T, I>(data.GT, delta_inv.tail(m));

        // unscale box scalings
        data.x_lb_scaling.head(n_lb).array() *= delta_lb_inv.head(n_lb).array();
        for (isize j = 0; j < n_lb; j++)
        {
            data.x_lb_scaling(j) *= delta_inv(data.x_lb_idx(j));
        }
        data.x_ub_scaling.head(n_ub).array() *= delta_ub_inv.head(n_ub).array();
        for (isize j = 0; j < n_ub; j++)
        {
            data.x_ub_scaling(j) *= delta_inv(data.x_ub_idx(j));
        }

        // unscale bounds
        data.b.array() *= delta_inv.segment(n, p).array();
        data.h.array() *= delta_inv.tail(m).array();
        data.x_lb_n.head(n_lb).array() *= delta_lb_inv.head(n_lb).array();
        data.x_ub.head(n_ub).array() *= delta_ub_inv.head(n_ub).array();
    }

    inline T scale_cost(T cost) const
    {
        return c * cost;
    }

    inline T unscale_cost(T cost) const
    {
        return c_inv * cost;
    }

    template<typename Derived>
    inline auto scale_primal(const Eigen::MatrixBase<Derived>& x) const
    {
        return (x.array() * delta_inv.head(n).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_primal(const Eigen::MatrixBase<Derived>& x) const
    {
        return (x.array() * delta.head(n).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_dual_eq(const Eigen::MatrixBase<Derived>& y) const
    {
        return (y.array() * c * delta_inv.segment(n, p).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_dual_eq(const Eigen::MatrixBase<Derived>& y) const
    {
        return (y.array() * c_inv * delta.segment(n, p).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_dual_ineq(const Eigen::MatrixBase<Derived>& z) const
    {
        return (z.array() * c * delta_inv.tail(m).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_dual_ineq(const Eigen::MatrixBase<Derived>& z) const
    {
        return (z.array() * c_inv * delta.tail(m).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_dual_lb(const Eigen::MatrixBase<Derived>& z_lb) const
    {
        return (z_lb.array() * c * delta_lb_inv.head(n_lb).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_dual_lb(const Eigen::MatrixBase<Derived>& z_lb) const
    {
        return (z_lb.array() * c_inv * delta_lb.head(n_lb).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_dual_ub(const Eigen::MatrixBase<Derived>& z_ub) const
    {
        return (z_ub.array() * c * delta_ub_inv.head(n_ub).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_dual_ub(const Eigen::MatrixBase<Derived>& z_ub) const
    {
        return (z_ub.array() * c_inv * delta_ub.head(n_ub).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_slack_ineq(const Eigen::MatrixBase<Derived>& s) const
    {
        return (s.array() * delta.tail(m).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_slack_ineq(const Eigen::MatrixBase<Derived>& s) const
    {
        return (s.array() * delta_inv.tail(m).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_slack_lb(const Eigen::MatrixBase<Derived>& s_lb) const
    {
        return (s_lb.array() * delta_lb.head(n_lb).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_slack_lb(const Eigen::MatrixBase<Derived>& s_lb) const
    {
        return (s_lb.array() * delta_lb_inv.head(n_lb).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_slack_ub(const Eigen::MatrixBase<Derived>& s_ub) const
    {
        return (s_ub.array() * delta_ub.head(n_ub).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_slack_ub(const Eigen::MatrixBase<Derived>& s_ub) const
    {
        return (s_ub.array() * delta_ub_inv.head(n_ub).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_primal_res_eq(const Eigen::MatrixBase<Derived>& p_res_eq) const
    {
        return (p_res_eq.array() * delta.segment(n, p).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_primal_res_eq(const Eigen::MatrixBase<Derived>& p_res_eq) const
    {
        return (p_res_eq.array() * delta_inv.segment(n, p).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_primal_res_ineq(const Eigen::MatrixBase<Derived>& p_res_in) const
    {
        return (p_res_in.array() * delta.tail(m).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_primal_res_ineq(const Eigen::MatrixBase<Derived>& p_res_in) const
    {
        return (p_res_in.array() * delta_inv.tail(m).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_primal_res_lb(const Eigen::MatrixBase<Derived>& p_res_lb) const
    {
        return (p_res_lb.array() * delta_lb.head(n_lb).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_primal_res_lb(const Eigen::MatrixBase<Derived>& p_res_lb) const
    {
        return (p_res_lb.array() * delta_lb_inv.head(n_lb).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_primal_res_ub(const Eigen::MatrixBase<Derived>& p_res_ub) const
    {
        return (p_res_ub.array() * delta_ub.head(n_ub).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_primal_res_ub(const Eigen::MatrixBase<Derived>& p_res_ub) const
    {
        return (p_res_ub.array() * delta_ub_inv.head(n_ub).array()).matrix();
    }

    template<typename Derived>
    inline auto scale_dual_res(const Eigen::MatrixBase<Derived>& d_res) const
    {
        return (d_res.array() * c * delta.head(n).array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_dual_res(const Eigen::MatrixBase<Derived>& d_res) const
    {
        return (d_res.array() * c_inv * delta_inv.head(n).array()).matrix();
    }

protected:
    inline void limit_scaling(VecRef<T> d) const
    {
        isize n_d = d.rows();
        for (int i = 0; i < n_d; i++)
        {
            limit_scaling(d(i));
        }
    }
    inline void limit_scaling(T& d) const
    {
        if (d < min_scaling)
        {
            d = T(1);
        } else if (d > max_scaling)
        {
            d = max_scaling;
        }
    }
};

template<typename T, typename I>
class IdentityPreconditioner
{
public:
    void init(const Data<T, I>&) {}

    inline void scale_data(Data<T, I>&, bool = false, bool = false, isize = 0, T = T(0)) {}

    inline void unscale_data(Data<T, I>&) {}

    inline T scale_cost(T cost) const { return cost; }

    inline T unscale_cost(T cost) const { return cost; }

    template<typename Derived>
    inline auto& scale_primal(const Eigen::MatrixBase<Derived>& x) const { return x; }

    template<typename Derived>
    inline auto& unscale_primal(const Eigen::MatrixBase<Derived>& x) const { return x; }

    template<typename Derived>
    inline auto& scale_dual_eq(const Eigen::MatrixBase<Derived>& y) const { return y; }

    template<typename Derived>
    inline auto& unscale_dual_eq(const Eigen::MatrixBase<Derived>& y) const { return y; }

    template<typename Derived>
    inline auto& scale_dual_ineq(const Eigen::MatrixBase<Derived>& z) const { return z; }

    template<typename Derived>
    inline auto& unscale_dual_ineq(const Eigen::MatrixBase<Derived>& z) const { return z; }

    template<typename Derived>
    inline auto& scale_dual_lb(const Eigen::MatrixBase<Derived>& z_lb) const { return z_lb; }

    template<typename Derived>
    inline auto& unscale_dual_lb(const Eigen::MatrixBase<Derived>& z_lb) const { return z_lb; }

    template<typename Derived>
    inline auto& scale_dual_ub(const Eigen::MatrixBase<Derived>& z_ub) const { return z_ub; }

    template<typename Derived>
    inline auto& unscale_dual_ub(const Eigen::MatrixBase<Derived>& z_ub) const { return z_ub; }

    template<typename Derived>
    inline auto& scale_slack_ineq(const Eigen::MatrixBase<Derived>& s) const { return s; }

    template<typename Derived>
    inline auto& unscale_slack_ineq(const Eigen::MatrixBase<Derived>& s) const { return s; }

    template<typename Derived>
    inline auto& scale_slack_lb(const Eigen::MatrixBase<Derived>& s_lb) const { return s_lb; }

    template<typename Derived>
    inline auto& unscale_slack_lb(const Eigen::MatrixBase<Derived>& s_lb) const { return s_lb; }

    template<typename Derived>
    inline auto& scale_slack_ub(const Eigen::MatrixBase<Derived>& s_ub) const { return s_ub; }

    template<typename Derived>
    inline auto& unscale_slack_ub(const Eigen::MatrixBase<Derived>& s_ub) const { return s_ub; }

    template<typename Derived>
    inline auto& scale_primal_res_eq(const Eigen::MatrixBase<Derived>& p_res_eq) const { return p_res_eq; }

    template<typename Derived>
    inline auto& unscale_primal_res_eq(const Eigen::MatrixBase<Derived>& p_res_eq) const { return p_res_eq; }

    template<typename Derived>
    inline auto& scale_primal_res_ineq(const Eigen::MatrixBase<Derived>& p_res_in) const { return p_res_in; }

    template<typename Derived>
    inline auto& unscale_primal_res_ineq(const Eigen::MatrixBase<Derived>& p_res_in) const { return p_res_in; }

    template<typename Derived>
    inline auto& scale_primal_res_lb(const Eigen::MatrixBase<Derived>& p_res_lb) const { return p_res_lb; }

    template<typename Derived>
    inline auto& unscale_primal_res_lb(const Eigen::MatrixBase<Derived>& p_res_lb) const { return p_res_lb; }

    template<typename Derived>
    inline auto& scale_primal_res_ub(const Eigen::MatrixBase<Derived>& p_res_ub) const { return p_res_ub; }

    template<typename Derived>
    inline auto& unscale_primal_res_ub(const Eigen::MatrixBase<Derived>& p_res_ub) const { return p_res_ub; }

    template<typename Derived>
    inline auto& scale_dual_res(const Eigen::MatrixBase<Derived>& d_res) const { return d_res; }

    template<typename Derived>
    inline auto& unscale_dual_res(const Eigen::MatrixBase<Derived>& d_res) const { return d_res; }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_PRECONDITIONER_HPP
