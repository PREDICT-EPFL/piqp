// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_PRECONDITIONER_HPP
#define PIQP_SPARSE_PRECONDITIONER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/utils.hpp"
#include "piqp/utils/tracy.hpp"

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

    T c = T(1);
    Vec<T> delta;
    Vec<T> delta_b;

    T c_inv = T(1);
    Vec<T> delta_inv;
    Vec<T> delta_b_inv;

public:
    void init(const Data<T, I>& data)
    {
        n = data.n;
        p = data.p;
        m = data.m;

        delta.resize(n + p + m);
        delta_b.resize(n);
        delta_inv.resize(n + p + m);
        delta_b_inv.resize(n);

        c = T(1);
        delta.setConstant(1);
        delta_b.setConstant(1);
        c_inv = T(1);
        delta_inv.setConstant(1);
        delta_b_inv.setConstant(1);
    }

    inline void scale_data(Data<T, I>& data, bool reuse_prev_scaling = false, bool scale_cost = false, isize max_iter = 10, T epsilon = T(1e-3))
    {
        PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data");

        using std::abs;

        if (!reuse_prev_scaling)
        {
            // init scaling in case max_iter is 0
            c = T(1);
            delta.setConstant(1);
            delta_b.setConstant(1);

            Vec<T>& delta_iter = delta_inv; // we use the memory of delta_inv as temporary storage
            Vec<T>& delta_iter_b = delta_b_inv; // we use the memory of delta_b_inv as temporary storage
            delta_iter.setZero();
            delta_iter_b.setZero();
            for (isize i = 0; i < max_iter && (std::max)({
                    (1 - delta_iter.array()).matrix().template lpNorm<Eigen::Infinity>(),
                    (1 - delta_iter_b.array()).matrix().template lpNorm<Eigen::Infinity>()
                }) > epsilon; i++)
            {
                {
                    PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::kkt_scaling");

                    delta_iter.setZero();

                    // calculate scaling of full KKT matrix
                    // [ P AT GT D ]
                    // [ A 0  0  0 ]
                    // [ G 0  0  0 ]
                    // [ D 0  0  0 ]
                    // where D is the diagonal of the bounds scaling
                    for (isize j = 0; j < n; j++)
                    {
                        for (typename SparseMat<T, I>::InnerIterator P_utri_it(data.P_utri, j); P_utri_it; ++P_utri_it)
                        {
                            I i_row = P_utri_it.index();
                            delta_iter(j) = (std::max)(delta_iter(j), abs(P_utri_it.value()));
                            if (i_row != j)
                            {
                                delta_iter(i_row) = (std::max)(delta_iter(i_row), abs(P_utri_it.value()));
                            }
                        }
                        delta_iter(j) = (std::max)(delta_iter(j), data.x_b_scaling(j));
                    }
                    for (isize j = 0; j < p; j++)
                    {
                        for (typename SparseMat<T, I>::InnerIterator AT_it(data.AT, j); AT_it; ++AT_it)
                        {
                            I i_row = AT_it.index();
                            delta_iter(i_row) = (std::max)(delta_iter(i_row), abs(AT_it.value()));
                            delta_iter(n + j) = (std::max)(delta_iter(n + j), abs(AT_it.value()));
                        }
                    }
                    for (isize j = 0; j < m; j++)
                    {
                        for (typename SparseMat<T, I>::InnerIterator GT_it(data.GT, j); GT_it; ++GT_it)
                        {
                            I i_row = GT_it.index();
                            delta_iter(i_row) = (std::max)(delta_iter(i_row), abs(GT_it.value()));
                            delta_iter(n + p + j) = (std::max)(delta_iter(n + p + j), abs(GT_it.value()));
                        }
                    }
                    delta_iter_b.array() = data.x_b_scaling.array();
                }

                limit_scaling(delta_iter);
                limit_scaling(delta_iter_b);

                delta_iter.array() = delta_iter.array().sqrt().inverse();
                delta_iter_b.array() = delta_iter_b.array().sqrt().inverse();

                {
                    PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::cost");
                    // scale cost
                    pre_mult_diagonal<T, I>(data.P_utri, delta_iter.head(n));
                    post_mult_diagonal<T, I>(data.P_utri, delta_iter.head(n));
                    data.c.array() *= delta_iter.head(n).array();
                }
                {
                    PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::A");
                    // scale AT
                    pre_mult_diagonal<T, I>(data.AT, delta_iter.head(n));
                    post_mult_diagonal<T, I>(data.AT, delta_iter.segment(n, p));
                }
                {
                    PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::G");
                    // scale GT
                    pre_mult_diagonal<T, I>(data.GT, delta_iter.head(n));
                    post_mult_diagonal<T, I>(data.GT, delta_iter.tail(m));
                }
                {
                    PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::x_b");
                    // scale box scalings
                    data.x_b_scaling.array() *= delta_iter_b.array() * delta_iter.head(n).array();
                }

                delta.array() *= delta_iter.array();
                delta_b.array() *= delta_iter_b.array();

                if (scale_cost)
                {
                    PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::cost_scaling");
                    // scaling for the cost
                    Vec<T>& delta_iter_cost = delta_b_inv; // we use delta_l_inv as a temporary storage
                    delta_iter_cost.setZero();
                    for (isize j = 0; j < n; j++)
                    {
                        for (typename SparseMat<T, I>::InnerIterator P_utri_it(data.P_utri, j); P_utri_it; ++P_utri_it)
                        {
                            I i_row = P_utri_it.index();
                            delta_iter_cost(j) = (std::max)(delta_iter_cost(j), abs(P_utri_it.value()));
                            if (i_row != j)
                            {
                                delta_iter_cost(i_row) = (std::max)(delta_iter_cost(i_row), abs(P_utri_it.value()));
                            }
                        }
                    }
                    T gamma = delta_iter_cost.sum() / T(n);
                    limit_scaling(gamma);
                    gamma = (std::max)(gamma, data.c.template lpNorm<Eigen::Infinity>());
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
            delta_b_inv.array() = delta_b.array().inverse();
        }
        else
        {
            {
                PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::cost");
                // scale cost
                data.P_utri *= c;
                pre_mult_diagonal<T, I>(data.P_utri, delta.head(n));
                post_mult_diagonal<T, I>(data.P_utri, delta.head(n));
                data.c.array() *= c * delta.head(n).array();
            }
            {
                PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::A");
                // scale AT
                pre_mult_diagonal<T, I>(data.AT, delta.head(n));
                post_mult_diagonal<T, I>(data.AT, delta.segment(n, p));
            }
            {
                PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::G");
                // scale GT
                pre_mult_diagonal<T, I>(data.GT, delta.head(n));
                post_mult_diagonal<T, I>(data.GT, delta.tail(m));
            }
            {
                PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::x_b");
                // scale box scalings
                data.x_b_scaling.array() *= delta_b.array() * delta.head(n).array();
            }
        }

        {
            PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::bounds");
            // scale bounds
            data.b.array() *= delta.segment(n, p).array();
            data.h_l.array() *= delta.tail(m).array();
            data.h_u.array() *= delta.tail(m).array();
            for (isize i = 0; i < data.n_x_l; i++)
            {
                data.x_l(i) *= delta_b(data.x_l_idx(i));
            }
            for (isize i = 0; i < data.n_x_u; i++)
            {
                data.x_u(i) *= delta_b(data.x_u_idx(i));
            }
        }
    }

    inline void unscale_data(Data<T, I>& data)
    {
        PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::unscale_data");

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
        data.x_b_scaling.array() *= delta_b_inv.array() * delta_inv.head(n).array();

        // unscale bounds
        data.b.array() *= delta_inv.segment(n, p).array();
        data.h_l.array() *= delta_inv.tail(m).array();
        data.h_u.array() *= delta_inv.tail(m).array();
        for (isize i = 0; i < data.n_x_l; i++)
        {
            data.x_l(i) *= delta_b_inv(data.x_l_idx(i));
        }
        for (isize i = 0; i < data.n_x_u; i++)
        {
            data.x_u(i) *= delta_b_inv(data.x_u_idx(i));
        }
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
    inline auto scale_dual_b(const Eigen::MatrixBase<Derived>& z_b) const
    {
        return (z_b.array() * c * delta_b_inv.array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_dual_b(const Eigen::MatrixBase<Derived>& z_b) const
    {
        return (z_b.array() * c_inv * delta_b.array()).matrix();
    }

    inline auto scale_dual_b_i(const T& z_b_i, Eigen::Index i) const
    {
        return z_b_i * c * delta_b_inv(i);
    }

    inline auto unscale_dual_b_i(const T& z_b_i, Eigen::Index i) const
    {
        return z_b_i * c_inv * delta_b(i);
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
    inline auto scale_slack_b(const Eigen::MatrixBase<Derived>& s_b) const
    {
        return (s_b.array() * delta_b.array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_slack_b(const Eigen::MatrixBase<Derived>& s_b) const
    {
        return (s_b.array() * delta_b_inv.array()).matrix();
    }

    inline auto scale_slack_b_i(const T& s_b_i, Eigen::Index i) const
    {
        return s_b_i * delta_b(i);
    }

    inline auto unscale_slack_b_i(const T& s_b_i, Eigen::Index i) const
    {
        return s_b_i * delta_b_inv(i);
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
    inline auto scale_primal_res_b(const Eigen::MatrixBase<Derived>& p_res_b) const
    {
        return (p_res_b.array() * delta_b.array()).matrix();
    }

    template<typename Derived>
    inline auto unscale_primal_res_b(const Eigen::MatrixBase<Derived>& p_res_b) const
    {
        return (p_res_b.array() * delta_b_inv.array()).matrix();
    }

    inline auto scale_primal_res_b_i(const T& p_res_b_i, Eigen::Index i) const
    {
        return p_res_b_i * delta_b(i);
    }

    inline auto unscale_primal_res_b_i(const T& p_res_b_i, Eigen::Index i) const
    {
        return p_res_b_i * delta_b_inv(i);
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
    inline void limit_scaling(Vec<T>& d) const
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
    inline auto& scale_dual_b(const Eigen::MatrixBase<Derived>& z_b) const { return z_b; }

    template<typename Derived>
    inline auto& unscale_dual_b(const Eigen::MatrixBase<Derived>& z_b) const { return z_b; }

    inline auto& scale_dual_b_i(const T& z_b_i, Eigen::Index) const { return z_b_i; }

    inline auto& unscale_dual_b_i(const T& z_b_i, Eigen::Index) const { return z_b_i; }

    template<typename Derived>
    inline auto& scale_slack_ineq(const Eigen::MatrixBase<Derived>& s) const { return s; }

    template<typename Derived>
    inline auto& unscale_slack_ineq(const Eigen::MatrixBase<Derived>& s) const { return s; }

    template<typename Derived>
    inline auto& scale_slack_b(const Eigen::MatrixBase<Derived>& s_b) const { return s_b; }

    template<typename Derived>
    inline auto& unscale_slack_b(const Eigen::MatrixBase<Derived>& s_b) const { return s_b; }

    inline auto& scale_slack_b_i(const T& s_b_i, Eigen::Index) const { return s_b_i; }

    inline auto& unscale_slack_b_i(const T& s_b_i, Eigen::Index) const { return s_b_i; }

    template<typename Derived>
    inline auto& scale_primal_res_eq(const Eigen::MatrixBase<Derived>& p_res_eq) const { return p_res_eq; }

    template<typename Derived>
    inline auto& unscale_primal_res_eq(const Eigen::MatrixBase<Derived>& p_res_eq) const { return p_res_eq; }

    template<typename Derived>
    inline auto& scale_primal_res_ineq(const Eigen::MatrixBase<Derived>& p_res_in) const { return p_res_in; }

    template<typename Derived>
    inline auto& unscale_primal_res_ineq(const Eigen::MatrixBase<Derived>& p_res_in) const { return p_res_in; }

    template<typename Derived>
    inline auto& scale_primal_res_b(const Eigen::MatrixBase<Derived>& p_res_b) const { return p_res_b; }

    template<typename Derived>
    inline auto& unscale_primal_res_b(const Eigen::MatrixBase<Derived>& p_res_b) const { return p_res_b; }

    inline auto& scale_primal_res_b_i(const T& p_res_b_i, Eigen::Index) const { return p_res_b_i; }

    inline auto& unscale_primal_res_b_i(const T& p_res_b_i, Eigen::Index) const { return p_res_b_i; }

    template<typename Derived>
    inline auto& scale_dual_res(const Eigen::MatrixBase<Derived>& d_res) const { return d_res; }

    template<typename Derived>
    inline auto& unscale_dual_res(const Eigen::MatrixBase<Derived>& d_res) const { return d_res; }
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/sparse/preconditioner.tpp"
#endif

#endif //PIQP_SPARSE_PRECONDITIONER_HPP
