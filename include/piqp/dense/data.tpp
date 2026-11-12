// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_DATA_TPP
#define PIQP_DENSE_DATA_TPP

#include "piqp/dense/data.hpp"

namespace piqp
{

namespace dense
{

template<typename T>
Data<T>::Data(Model<T> model)
: n(model.P.rows()), p(model.A.rows()), m(model.G.rows()),
  P_utri(model.P.template triangularView<Eigen::Upper>()),
  AT(model.A.transpose()), GT(model.G.transpose()),
  c(model.c), b(model.b),
  h_l(model.G.rows()), h_u(model.G.rows()),
  x_l(model.P.rows()), x_u(model.P.rows()),
  n_h_l(0), n_h_u(0), n_x_l(0), n_x_u(0),
  h_l_idx(model.G.rows()), h_u_idx(model.G.rows()),
  x_l_idx(model.P.rows()), x_u_idx(model.P.rows()),
  x_b_scaling(Vec<T>::Constant(model.P.rows(), T(1))),
  G_row_disabled(Vec<bool>::Constant(model.G.rows(), false))
{
    set_h_l(model.h_l);
    set_h_u(model.h_u);
    disable_inf_constraints();
    set_x_l(model.x_l);
    set_x_u(model.x_u);
}

template<typename T>
void Data<T>::resize(isize n, isize p, isize m)
{
    this->n = n;
    this->p = p;
    this->m = m;

    P_utri.resize(n, n);
    AT.resize(n, p);
    GT.resize(n, m);

    c.resize(n);
    b.resize(p);
    h_l.resize(m);
    h_u.resize(m);
    x_l.resize(n);
    x_u.resize(n);

    h_l_idx.resize(m);
    h_u_idx.resize(m);
    x_l_idx.resize(n);
    x_u_idx.resize(n);

    x_b_scaling.resize(n);
    x_b_scaling.setConstant(T(1));

    G_row_disabled.resize(m);
    G_row_disabled.setConstant(false);
}

template<typename T>
void Data<T>::set_h_l(const optional<CVecRef<T>>& h_l)
{
    n_h_l = 0;
    if (h_l.has_value())
    {
        isize i_l = 0;
        for (isize i = 0; i < m; i++)
        {
            if ((*h_l)(i) > -PIQP_INF)
            {
                n_h_l += 1;
                this->h_l(i) = (*h_l)(i);
                h_l_idx(i_l++) = i;
            } else {
                this->h_l(i) = -PIQP_INF;
            }
        }
    } else {
        this->h_l.setConstant(-PIQP_INF);
    }
}

template<typename T>
void Data<T>::set_h_u(const optional<CVecRef<T>>& h_u)
{
    n_h_u = 0;
    if (h_u.has_value())
    {
        isize i_u = 0;
        for (isize i = 0; i < m; i++)
        {
            if ((*h_u)(i) < PIQP_INF)
            {
                n_h_u += 1;
                this->h_u(i) = (*h_u)(i);
                h_u_idx(i_u++) = i;
            } else {
                this->h_u(i) = PIQP_INF;
            }
        }
    } else {
        this->h_u.setConstant(PIQP_INF);
    }
}

template<typename T>
void Data<T>::disable_inf_constraints()
{
    bool msg_printed = false;
    G_row_disabled.setConstant(false);
    for (isize i = 0; i < m; i++)
    {
        if (h_l(i) <= -PIQP_INF && h_u(i) >= PIQP_INF)
        {
            G_row_disabled(i) = true;
            set_G_row_zero(i);
            h_l(i) = T(-1);
            h_u(i) = T(1);

            if (!msg_printed)
            {
                piqp_eprint("h_l[i] and h_u[i] are both close to -/+ infinity for i = %zd (and potentially other indices).\n", i);
                piqp_eprint("PIQP is setting the corresponding rows in G to zero (sparsity structure preserving).\n");
                piqp_eprint("Consider removing the corresponding constraints for faster solves.\n");
                msg_printed = true;
            }
        }
    }
    if (msg_printed) {
        // recalculate idx
        set_h_l(h_l);
        set_h_u(h_u);
    }
}

template<typename T>
void Data<T>::set_x_l(const optional<CVecRef<T>>& x_l)
{
    n_x_l = 0;
    if (x_l.has_value())
    {
        isize i_l = 0;
        for (isize i = 0; i < n; i++)
        {
            if ((*x_l)(i) > -PIQP_INF)
            {
                n_x_l += 1;
                this->x_l(i_l) = (*x_l)(i);
                x_l_idx(i_l) = i;
                i_l++;
            }
        }
    }
}

template<typename T>
void Data<T>::set_x_u(const optional<CVecRef<T>>& x_u)
{
    n_x_u = 0;
    if (x_u.has_value())
    {
        isize i_u = 0;
        for (isize i = 0; i < n; i++)
        {
            if ((*x_u)(i) < PIQP_INF)
            {
                n_x_u += 1;
                this->x_u(i_u) = (*x_u)(i);
                x_u_idx(i_u) = i;
                i_u++;
            }
        }
    }
}

template<typename T>
void Data<T>::set_G_row_zero(Eigen::Index row)
{
    GT.col(row).setZero();
}

template<typename T>
Eigen::Index Data<T>::non_zeros_P_utri() { return P_utri.rows() * (P_utri.rows() + 1) / 2; }

template<typename T>
Eigen::Index Data<T>::non_zeros_A() { return AT.rows() * AT.cols(); }

template<typename T>
Eigen::Index Data<T>::non_zeros_G() { return GT.rows() * GT.cols(); }

} // namespace dense

} // namespace piqp

#endif //PIQP_DENSE_DATA_TPP
