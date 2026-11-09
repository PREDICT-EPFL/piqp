// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_DATA_HPP
#define PIQP_DENSE_DATA_HPP

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"
#include "piqp/dense/model.hpp"

namespace piqp
{

namespace dense
{

template<typename T>
struct Data
{
    isize n; // number of variables
    isize p; // number of equality constraints
    isize m; // number of inequality constraints

    Mat<T> P_utri; // upper triangular part of P
    Mat<T> AT;     // A transpose
    Mat<T> GT;     // G transpose

    Vec<T> c;
    Vec<T> b;
    Vec<T> h_l;
    Vec<T> h_u;
    Vec<T> x_l; // stores finite lower bounds in the first n_l fields
    Vec<T> x_u; // stores finite upper bounds in the first n_u fields

    isize n_h_l;
    isize n_h_u;
    isize n_x_l;
    isize n_x_u;

    // stores the indexes of the finite bounds
    Vec<Eigen::Index> h_l_idx;
    Vec<Eigen::Index> h_u_idx;
    Vec<Eigen::Index> x_l_idx;
    Vec<Eigen::Index> x_u_idx;

    Vec<T> x_b_scaling; // scaling of x_l and x_u, i.e. x_l <= x_b_scaling .* x <= x_u

    Vec<bool> G_row_disabled; // rows of G which were zeroed by disable_inf_constraints

    Data() = default;

    explicit Data(Model<T> model);

    void resize(isize n, isize p, isize m);

    void set_h_l(const optional<CVecRef<T>>& h_l);
    void set_h_u(const optional<CVecRef<T>>& h_u);
    void disable_inf_constraints();
    void set_x_l(const optional<CVecRef<T>>& x_l);
    void set_x_u(const optional<CVecRef<T>>& x_u);

    void set_G_row_zero(Eigen::Index row);

    Eigen::Index non_zeros_P_utri();
    Eigen::Index non_zeros_A();
    Eigen::Index non_zeros_G();
};

} // namespace dense

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/common.hpp"

namespace piqp
{

namespace dense
{

extern template struct Data<common::Scalar>;

} // namespace dense

} // namespace piqp

#else
#include "piqp/dense/data.tpp"
#endif

#endif //PIQP_DENSE_DATA_HPP
