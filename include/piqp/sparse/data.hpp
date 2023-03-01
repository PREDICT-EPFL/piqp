// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_DATA_HPP
#define PIQP_SPARSE_DATA_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/typedefs.hpp"
#include "piqp/sparse/model.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
struct Data
{
    isize n; // number of variables
    isize p; // number of equality constraints
    isize m; // number of inequality constraints

    SparseMat<T, I> P_utri; // upper triangular part of P
    SparseMat<T, I> AT;     // A transpose
    SparseMat<T, I> GT;     // G transpose

    Vec<T> c;
    Vec<T> b;
    Vec<T> h;

    isize n_lb;
    isize n_ub;

    Vec<Eigen::Index> x_lb_idx; // stores the original index of the finite lower bounds
    Vec<Eigen::Index> x_ub_idx; // stores the original index of the finite upper bounds

    Vec<T> x_lb_scaling; // scaling of lb, i.e. x_lb <= x_lb_scaling .* x
    Vec<T> x_ub_scaling; // scaling of lb, i.e. x_ub_scaling .* x <= x_ub

    Vec<T> x_lb_n; // stores negative finite lower bounds in the first n_lb fields
    Vec<T> x_ub;   // stores finite upper bounds in the first n_ub fields

    Data() = default;

    explicit Data(Model<T, I> model)
        : n(model.P.rows()), p(model.A.rows()), m(model.G.rows()),
          P_utri(model.P.template triangularView<Eigen::Upper>()),
          AT(model.A.transpose()), GT(model.G.transpose()),
          c(model.c), b(model.b), h(model.h),
          n_lb(0), n_ub(0),
          x_lb_idx(model.P.rows()), x_ub_idx(model.P.rows()),
          x_lb_scaling(Vec<T>::Constant(model.P.rows(), T(1))), x_ub_scaling(Vec<T>::Constant(model.P.rows(), T(1))),
          x_lb_n(model.P.rows()), x_ub(model.P.rows())
    {
        isize i_lb = 0;
        for (isize i = 0; i < n; i++)
        {
            if (model.x_lb(i) > -PIQP_INF)
            {
                n_lb += 1;
                x_lb_n(i_lb) = -model.x_lb(i);
                x_lb_idx(i_lb) = i;
                i_lb++;
            }
        }

        isize i_ub = 0;
        for (isize i = 0; i < n; i++)
        {
            if (model.x_ub(i) < PIQP_INF)
            {
                n_ub += 1;
                x_ub(i_ub) = model.x_ub(i);
                x_ub_idx(i_ub) = i;
                i_ub++;
            }
        }
    }

    Eigen::Index non_zeros_P_utri() { return P_utri.nonZeros(); }
    Eigen::Index non_zeros_A() { return AT.nonZeros(); }
    Eigen::Index non_zeros_G() { return GT.nonZeros(); }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_DATA_HPP
