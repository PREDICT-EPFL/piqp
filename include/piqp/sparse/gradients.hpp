// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_GRADIENTS_HPP
#define PIQP_SPARSE_GRADIENTS_HPP

#include <algorithm>

#include "piqp/typedefs.hpp"
#include "piqp/utils/optional.hpp"
#include "piqp/results.hpp"
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
 * The gradients are only valid if status is PIQP_SOLVED. status is PIQP_UNSOLVED if backward() was not
 * called yet or the problem changed since (setup(), update()), and the status returned by the last
 * backward() otherwise.
 *
 * All quantities are in the user's (unscaled) coordinates:
 *   P: n x n, dL/dP = 0.5 * (lam_x x^T + x lam_x^T) on the sparsity pattern of the P passed to setup(),
 *      i.e., upper triangular if only the upper triangular part of P is passed. This is the gradient w.r.t. the
 *      symmetric matrix P, the derivative w.r.t. a stored off-diagonal value of an upper triangular P is twice
 *      the returned entry.
 *   A: p x n, sparsity pattern of the A passed to setup()
 *   G: m x n, sparsity pattern of the G passed to setup(), rows with both bounds infinite are zero
 *   c, b, h_l, h_u: entries of infinite bounds are zero
 *   x_l, x_u: full n-length, entries of infinite bounds are zero
 */
template<typename T, typename I>
struct Gradients
{
    Status status = Status::PIQP_UNSOLVED;

    SparseMat<T, I> P;
    SparseMat<T, I> A;
    SparseMat<T, I> G;

    Vec<T> c;
    Vec<T> b;
    Vec<T> h_l;
    Vec<T> h_u;
    Vec<T> x_l;
    Vec<T> x_u;

    // Takes over the sparsity patterns of the matrices passed to setup(), the values are overwritten by
    // compute_from_adjoint(). The storage is kept if a pattern is unchanged, such that views into it
    // (e.g., in the Python interface) stay valid across setup() calls.
    void resize(const Data<T, I>& data,
                const CSparseMatRef<T, I>& P_in,
                const optional<CSparseMatRef<T, I>>& A_in,
                const optional<CSparseMatRef<T, I>>& G_in)
    {
        assign_pattern(P, P_in);
        if (A_in.has_value()) { assign_pattern(A, *A_in); } else { assign_empty(A, data.p, data.n); }
        if (G_in.has_value()) { assign_pattern(G, *G_in); } else { assign_empty(G, data.m, data.n); }
        c.resize(data.n);
        b.resize(data.p);
        h_l.resize(data.m);
        h_u.resize(data.m);
        x_l.resize(data.n);
        x_u.resize(data.n);
    }

    // copies the sparsity pattern (and values) of src into dst, unless dst already has the same pattern
    static void assign_pattern(SparseMat<T, I>& dst, const CSparseMatRef<T, I>& src)
    {
        if (src.isCompressed() && dst.isCompressed()
            && dst.rows() == src.rows() && dst.cols() == src.cols() && dst.nonZeros() == src.nonZeros()
            && std::equal(src.outerIndexPtr(), src.outerIndexPtr() + src.outerSize() + 1, dst.outerIndexPtr())
            && std::equal(src.innerIndexPtr(), src.innerIndexPtr() + src.nonZeros(), dst.innerIndexPtr()))
        {
            return;
        }
        dst = src;
        dst.makeCompressed();
    }

    static void assign_empty(SparseMat<T, I>& dst, isize rows, isize cols)
    {
        if (dst.isCompressed() && dst.rows() == rows && dst.cols() == cols && dst.nonZeros() == 0) return;
        dst.resize(rows, cols);
        dst.makeCompressed();
    }

    // Computes the gradients at the linearization point (user space) from the adjoint variables lambda
    // (user space, box duals in compressed layout):
    //   dP = 0.5 * (lam_x x^T + x lam_x^T)
    //   dA = lam_y x^T + y lam_x^T
    //   dG = (lam_z_u - lam_z_l) x^T + (z_u - z_l) lam_x^T
    //   dc = lam_x,  db = -lam_y,  dh_l = lam_z_l,  dh_u = -lam_z_u,  dx_l = lam_z_bl,  dx_u = -lam_z_bu
    // lambda and the duals of the linearization point must be zero on disabled constraints and infinite bounds,
    // which makes the corresponding gradients zero.
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
                G.valuePtr()[k] = (lambda.z_u(i) - lambda.z_l(i)) * linearization_point.x(j) + (linearization_point.z_u(i) - linearization_point.z_l(i)) * lambda.x(j);
            }
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

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_GRADIENTS_HPP
