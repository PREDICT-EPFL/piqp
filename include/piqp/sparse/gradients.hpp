// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_GRADIENTS_HPP
#define PIQP_SPARSE_GRADIENTS_HPP

#include "piqp/typedefs.hpp"
#include "piqp/variables.hpp"
#include "piqp/sparse/data.hpp"

namespace piqp
{

namespace sparse
{

/**
 * Gradients of a scalar loss L with respect to the problem data,
 * as computed by SolverBase::backward().
 *
 * All quantities are in the user's (unscaled) coordinates:
 *   P: n x n, symmetric, dL/dP = 0.5 * (lam_x x^T + x lam_x^T),
 *      sparsity pattern triu(P) + triu(P)^T of the P passed to setup()
 *   A: p x n, sparsity pattern of A
 *   G: m x n, sparsity pattern of G, rows with both bounds infinite are zero
 *   c, b, h_l, h_u: entries of infinite bounds are zero
 *   x_l, x_u: full n-length, entries of infinite bounds are zero
 */
template<typename T, typename I>
struct Gradients
{
    SparseMat<T, I> P;
    SparseMat<T, I> A;
    SparseMat<T, I> G;

    Vec<T> c;
    Vec<T> b;
    Vec<T> h_l;
    Vec<T> h_u;
    Vec<T> x_l;
    Vec<T> x_u;

    // copies the sparsity patterns from data, values are overwritten by compute_from_adjoint()
    void resize(const Data<T, I>& data)
    {
        P = data.P_utri.template selfadjointView<Eigen::Upper>();
        P.makeCompressed();
        A = data.AT.transpose();
        A.makeCompressed();
        G = data.GT.transpose();
        G.makeCompressed();
        c.resize(data.n);
        b.resize(data.p);
        h_l.resize(data.m);
        h_u.resize(data.m);
        x_l.resize(data.n);
        x_u.resize(data.n);
    }

    // Computes the gradients at the linearization point (user space) from the adjoint variables lambda
    // (user space, box duals in compressed layout):
    //   dP = 0.5 * (lam_x x^T + x lam_x^T)
    //   dA = lam_y x^T + y lam_x^T
    //   dG = (lam_z_u - lam_z_l) x^T + (z_u - z_l) lam_x^T
    //   dc = lam_x,  db = -lam_y,  dh_l = lam_z_l,  dh_u = -lam_z_u,  dx_l = lam_z_bl,  dx_u = -lam_z_bu
    // Gradients of disabled constraints and infinite bounds are zero.
    // The matrix gradients are evaluated on their sparsity patterns only.
    void compute_from_adjoint(const Data<T, I>& data, const Variables<T>& linearization_point, const Variables<T>& lambda)
    {
        for (isize j = 0; j < data.n; j++)
        {
            for (isize k = P.outerIndexPtr()[j]; k < P.outerIndexPtr()[j + 1]; k++)
            {
                isize i = P.innerIndexPtr()[k];
                P.valuePtr()[k] = T(0.5) * (lambda.x(i) * linearization_point.x(j) + linearization_point.x(i) * lambda.x(j));
            }
            for (isize k = A.outerIndexPtr()[j]; k < A.outerIndexPtr()[j + 1]; k++)
            {
                isize i = A.innerIndexPtr()[k];
                A.valuePtr()[k] = lambda.y(i) * linearization_point.x(j) + linearization_point.y(i) * lambda.x(j);
            }
            for (isize k = G.outerIndexPtr()[j]; k < G.outerIndexPtr()[j + 1]; k++)
            {
                isize i = G.innerIndexPtr()[k];
                if (data.G_row_disabled(i))
                {
                    G.valuePtr()[k] = T(0);
                }
                else
                {
                    G.valuePtr()[k] = (lambda.z_u(i) - lambda.z_l(i)) * linearization_point.x(j) + (linearization_point.z_u(i) - linearization_point.z_l(i)) * lambda.x(j);
                }
            }
        }

        c = lambda.x;
        b = -lambda.y;
        // lambda_z is zero for infinite bounds
        h_l = lambda.z_l;
        h_u = -lambda.z_u;
        for (isize i = 0; i < data.m; i++)
        {
            if (data.G_row_disabled(i))
            {
                h_l(i) = T(0);
                h_u(i) = T(0);
            }
        }

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

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_GRADIENTS_HPP
