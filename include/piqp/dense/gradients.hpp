// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_DENSE_GRADIENTS_HPP
#define PIQP_DENSE_GRADIENTS_HPP

#include "piqp/typedefs.hpp"
#include "piqp/utils/optional.hpp"
#include "piqp/utils/unused.hpp"
#include "piqp/results.hpp"
#include "piqp/variables.hpp"
#include "piqp/dense/data.hpp"

namespace piqp
{

namespace dense
{

/**
 * Gradients of a scalar loss L with respect to the problem data,
 * as computed by SolverBase::backward().
 *
 * The gradients are only valid if status is PIQP_SOLVED. status is PIQP_UNSOLVED if backward() was not
 * called yet or the problem changed since (setup(), update()), and the status returned by the last
 * backward() otherwise.
 *
 *   P: n x n, symmetric, dL/dP = 0.5 * (lam_x x^T + x lam_x^T)
 *   A: p x n
 *   G: m x n, rows with both bounds infinite are zero
 *   c, b, h_l, h_u: entries of infinite bounds are zero
 *   x_l, x_u: full n-length, entries of infinite bounds are zero
 */
template<typename T>
struct Gradients
{
    Status status = Status::PIQP_UNSOLVED;

    Mat<T> P;
    Mat<T> A;
    Mat<T> G;

    Vec<T> c;
    Vec<T> b;
    Vec<T> h_l;
    Vec<T> h_u;
    Vec<T> x_l;
    Vec<T> x_u;

    void resize(const Data<T>& data,
                const CMatRef<T>& P_in,
                const optional<CMatRef<T>>& A_in,
                const optional<CMatRef<T>>& G_in)
    {
        unused(P_in, A_in, G_in);
        P.resize(data.n, data.n);
        A.resize(data.p, data.n);
        G.resize(data.m, data.n);
        c.resize(data.n);
        b.resize(data.p);
        h_l.resize(data.m);
        h_u.resize(data.m);
        x_l.resize(data.n);
        x_u.resize(data.n);
    }

    // Computes the gradients at the linearization point from the adjoint variables lambda:
    //   dP = 0.5 * (lam_x x^T + x lam_x^T)
    //   dA = lam_y x^T + y lam_x^T
    //   dG = (lam_z_u - lam_z_l) x^T + (z_u - z_l) lam_x^T
    //   dc = lam_x,  db = -lam_y,  dh_l = lam_z_l,  dh_u = -lam_z_u,  dx_l = lam_z_bl,  dx_u = -lam_z_bu
    // lambda and the duals of the linearization point must be zero on disabled constraints and infinite bounds,
    // which makes the corresponding gradients zero.
    void compute_from_adjoint(const Data<T>& data, const Variables<T>& linearization_point, const Variables<T>& lambda)
    {
        P.noalias() = lambda.x * linearization_point.x.transpose();
        P.noalias() += linearization_point.x * lambda.x.transpose();
        P *= T(0.5);

        A.noalias() = lambda.y * linearization_point.x.transpose();
        A.noalias() += linearization_point.y * lambda.x.transpose();

        // column-wise to avoid temporaries for the differences
        for (isize j = 0; j < data.n; j++)
        {
            G.col(j) = (lambda.z_u - lambda.z_l) * linearization_point.x(j) + (linearization_point.z_u - linearization_point.z_l) * lambda.x(j);
        }

        c = lambda.x;
        b = -lambda.y;
        // lambda_z is zero for infinite bounds
        h_l = lambda.z_l;
        h_u = -lambda.z_u;

        // box duals are in compressed layout, zero for infinite bounds
        x_l.setZero();
        x_u.setZero();
        for (isize i = 0; i < data.n_x_l; i++)
        {
            x_l(data.x_l_idx(i)) = lambda.z_bl(i);
        }
        for (isize i = 0; i < data.n_x_u; i++)
        {
            x_u(data.x_u_idx(i)) = -lambda.z_bu(i);
        }
    }
};

} // namespace dense

} // namespace piqp

#endif //PIQP_DENSE_GRADIENTS_HPP
