// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_PRECONDITIONER_TPP
#define PIQP_DENSE_PRECONDITIONER_TPP

#include "piqp/dense/preconditioner.hpp"
#include "piqp/utils/tracy.hpp"

namespace piqp
{

namespace dense
{

template<typename T>
void RuizEquilibration<T>::init(const Data<T>& data)
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

template<typename T>
void RuizEquilibration<T>::scale_data(Data<T>& data, bool reuse_prev_scaling, bool scale_cost, isize max_iter, T epsilon)
{
    PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data");

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

                // calculate scaling of full KKT matrix
                // [ P AT GT D ]
                // [ A 0  0  0 ]
                // [ G 0  0  0 ]
                // [ D 0  0  0 ]
                // where D is the diagonal of the bounds scaling
                for (isize k = 0; k < n; k++)
                {
                    delta_iter(k) = (std::max)({data.P_utri.col(k).head(k).template lpNorm<Eigen::Infinity>(),
                                                data.P_utri.row(k).tail(n - k).template lpNorm<Eigen::Infinity>(),
                                                p > 0 ? data.AT.row(k).template lpNorm<Eigen::Infinity>() : T(0),
                                                m > 0 ? data.GT.row(k).template lpNorm<Eigen::Infinity>() : T(0),
                                                data.x_b_scaling(k)});
                }
                for (isize k = 0; k < p; k++)
                {
                    delta_iter(n + k) = data.AT.col(k).template lpNorm<Eigen::Infinity>();
                }
                for (isize k = 0; k < m; k++)
                {
                    delta_iter(n + p + k) = data.GT.col(k).template lpNorm<Eigen::Infinity>();
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
                for (isize k = 0; k < n; k++) {
                    data.P_utri.col(k).head(k + 1) *= delta_iter(k);
                }
                for (isize k = 0; k < n; k++) {
                    data.P_utri.row(k).tail(n - k) *= delta_iter(k);
                }
                data.c.array() *= delta_iter.head(n).array();
            }
            {
                PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::A");
                // scale AT
                data.AT = delta_iter.head(n).asDiagonal() * data.AT * delta_iter.segment(n, p).asDiagonal();
            }
            {
                PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::G");
                // scale GT
                data.GT = delta_iter.head(n).asDiagonal() * data.GT * delta_iter.tail(m).asDiagonal();
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
                // average over the columns of P which are actually populated, since empty
                // columns carry no cost and would only drag the average towards zero
                T gamma = 0;
                isize n_populated = 0;
                for (isize k = 0; k < n; k++)
                {
                    T col_norm = (std::max)(data.P_utri.col(k).head(k).template lpNorm<Eigen::Infinity>(),
                                            data.P_utri.row(k).tail(n - k).template lpNorm<Eigen::Infinity>());
                    if (col_norm > T(0))
                    {
                        gamma += col_norm;
                        n_populated++;
                    }
                }
                gamma = n_populated > 0 ? gamma / T(n_populated) : T(0);
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
            for (isize k = 0; k < n; k++) {
                data.P_utri.col(k).head(k + 1) *= delta(k);
            }
            for (isize k = 0; k < n; k++) {
                data.P_utri.row(k).tail(n - k) *= delta(k);
            }
            data.c.array() *= c * delta.head(n).array();
        }
        {
            PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::A");
            // scale AT
            data.AT = delta.head(n).asDiagonal() * data.AT * delta.segment(n, p).asDiagonal();
        }
        {
            PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::scale_data::G");
            // scale GT
            data.GT = delta.head(n).asDiagonal() * data.GT * delta.tail(m).asDiagonal();
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

template<typename T>
void RuizEquilibration<T>::unscale_data(Data<T>& data)
{
    PIQP_TRACY_ZoneScopedN("piqp::RuizEquilibration::unscale_data");

    // unscale cost
    data.P_utri *= c_inv;
    for (isize k = 0; k < n; k++) {
        data.P_utri.col(k).head(k + 1) *= delta_inv(k);
    }
    for (isize k = 0; k < n; k++) {
        data.P_utri.row(k).tail(n - k) *= delta_inv(k);
    }
    data.c.array() *= c_inv * delta_inv.head(n).array();

    // unscale AT and GT
    data.AT = delta_inv.head(n).asDiagonal() * data.AT * delta_inv.segment(n, p).asDiagonal();
    data.GT = delta_inv.head(n).asDiagonal() * data.GT * delta_inv.tail(m).asDiagonal();

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

template<typename T>
T RuizEquilibration<T>::scale_cost(T cost) const
{
    return c * cost;
}

template<typename T>
T RuizEquilibration<T>::unscale_cost(T cost) const
{
    return c_inv * cost;
}

template<typename T>
void RuizEquilibration<T>::limit_scaling(Vec<T>& d) const
{
    isize n_d = d.rows();
    for (int i = 0; i < n_d; i++)
    {
        limit_scaling(d(i));
    }
}

template<typename T>
void RuizEquilibration<T>::limit_scaling(T& d) const
{
    if (d < min_scaling)
    {
        d = T(1);
    } else if (d > max_scaling)
    {
        d = max_scaling;
    }
}

} // namespace dense

} // namespace piqp

#endif //PIQP_DENSE_PRECONDITIONER_TPP
