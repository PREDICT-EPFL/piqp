// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_PRECONDITIONER_TPP
#define PIQP_SPARSE_PRECONDITIONER_TPP

#include "piqp/sparse/preconditioner.hpp"
#include "piqp/sparse/utils.hpp"
#include "piqp/utils/tracy.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
void RuizEquilibration<T, I>::init(const Data<T, I>& data)
{
    n = data.n;
    p = data.p;
    m = data.m;

    delta.resize(n + p + m);
    delta_b.resize(n);
    delta_inv.resize(n + p + m);
    delta_b_inv.resize(n);
    delta_cost.resize(n);

    c = T(1);
    delta.setConstant(1);
    delta_b.setConstant(1);
    c_inv = T(1);
    delta_inv.setConstant(1);
    delta_b_inv.setConstant(1);
}

template<typename T, typename I>
void RuizEquilibration<T, I>::scale_data(Data<T, I>& data, bool reuse_prev_scaling, bool scale_cost, isize max_iter, T epsilon)
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
                Vec<T>& delta_iter_cost = delta_cost;
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
                // average over the columns of P which are actually populated, since empty
                // columns carry no cost and would only drag the average towards zero
                T gamma_sum = T(0);
                isize n_populated = 0;
                for (isize j = 0; j < n; j++)
                {
                    if (delta_iter_cost(j) > T(0))
                    {
                        gamma_sum += delta_iter_cost(j);
                        n_populated++;
                    }
                }
                T gamma = n_populated > 0 ? gamma_sum / T(n_populated) : T(0);
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

template<typename T, typename I>
void RuizEquilibration<T, I>::unscale_data(Data<T, I>& data)
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

template<typename T, typename I>
T RuizEquilibration<T, I>::scale_cost(T cost) const
{
    return c * cost;
}

template<typename T, typename I>
T RuizEquilibration<T, I>::unscale_cost(T cost) const
{
    return c_inv * cost;
}

template<typename T, typename I>
void RuizEquilibration<T, I>::limit_scaling(Vec<T>& d) const
{
    isize n_d = d.rows();
    for (int i = 0; i < n_d; i++)
    {
        limit_scaling(d(i));
    }
}

template<typename T, typename I>
void RuizEquilibration<T, I>::limit_scaling(T& d) const
{
    if (d < min_scaling)
    {
        d = T(1);
    } else if (d > max_scaling)
    {
        d = max_scaling;
    }
}

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_PRECONDITIONER_TPP
