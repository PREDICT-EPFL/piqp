// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2005-2022 by Timothy A. Davis.
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_LDLT_HPP
#define PIQP_SPARSE_LDLT_HPP

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"
#include "piqp/utils/tracy.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
struct LDLt
{
    Vec<I> etree;  // elimination tree
    // L in CSC
    Vec<I> L_cols; // column starts[n+1]
    Vec<I> L_nnz;  // number of non-zeros per column[n]
    Vec<I> L_ind;  // row indices
    Vec<T> L_vals; // values

    Vec<T> D;      // diagonal matrix D
    Vec<T> D_inv;  // inverse of D

    // working variables used in numerical factorization
    struct {
        Vec<I> flag;
        Vec<I> pattern;
        Vec<T> y;
    } work;

    void factorize_symbolic_upper_triangular(const SparseMat<T, I>& A)
    {
        // reimplementation of LDL_symbolic in
        // https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/stable/LDL/Source/ldl.c
        // see LDL_License.txt for license
        // we assume A has only the upper triangular part stored which simplifies the code from the original

        PIQP_TRACY_ZoneScopedN("piqp::LDLt::factorize_symbolic_upper_triangular");

        static_assert(!SparseMat<T, I>::IsRowMajor, "A has to be column major!");
        eigen_assert(A.rows() == A.cols() && "A has to be quadratic!");

        const isize n = A.rows();
        const Eigen::Map<const Vec<I>> Ap(A.outerIndexPtr(), A.outerSize() + 1);
        const Eigen::Map<const Vec<I>> Ai(A.innerIndexPtr(), A.nonZeros());

        etree.resize(n);
        L_cols.resize(n + 1);
        L_nnz.resize(n);

        D.resize(n);
        D_inv.resize(n);

        work.flag.resize(n);
        work.pattern.resize(n);
        work.y.resize(n);

        for (isize k = 0; k < n; k++)
        {
            /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
            etree[k] = -1;                   /* parent of k is not yet known */
            work.flag[k] = I(k);              /* mark node k as visited */
            L_nnz[k] = 0;                    /* count of nonzeros in column k of L */
            isize p2 = Ap[k + 1];
            for (isize p = Ap[k]; p < p2; p++)
            {
                /* A (i,k) is nonzero (original or permuted A) */
                isize i = Ai[p];
                /* follow path from i to root of etree, stop at flagged node */
                for (; work.flag[i] != k; i = etree[i])
                {
                    /* find parent of i if not yet determined */
                    if (etree[i] == -1) etree[i] = I(k);
                    L_nnz[i]++;         /* L (k,i) is nonzero */
                    work.flag[i] = I(k); /* mark i as visited */
                }
            }
        }
        /* construct Lp index array from Lnz column counts */
        L_cols[0] = 0;
        for (isize k = 0; k < n; k++)
        {
            L_cols[k + 1] = L_cols[k] + L_nnz[k];
        }

        L_ind.resize(L_cols[n]);
        L_vals.resize(L_cols[n]);
    }

    isize factorize_numeric_upper_triangular(const SparseMat<T, I>& A)
    {
        // reimplementation of LDL_numeric in
        // https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/stable/LDL/Source/ldl.c
        // see LDL_License.txt for license
        // we assume A has only the upper triangular part stored which simplifies the code from the original
        // additionally we assume that there are no duplicate entries present

        PIQP_TRACY_ZoneScopedN("piqp::LDLt::factorize_numeric_upper_triangular");

        const isize n = A.rows();
        const Eigen::Map<const Vec<I>> Ap(A.outerIndexPtr(), A.outerSize() + 1);
        const Eigen::Map<const Vec<I>> Ai(A.innerIndexPtr(), A.nonZeros());
        const Eigen::Map<const Vec<T>> Ax(A.valuePtr(), A.nonZeros());

        static_assert(!SparseMat<T, I>::IsRowMajor, "A has to be column major!");
        eigen_assert(n == L_cols.rows() - 1 && "symbolic factorization does not match!");

        for (isize k = 0 ; k < n; k++)
        {
            /* compute nonzero Pattern of kth row of L, in topological order */
            work.y[k] = 0.0;                   /* Y(0:k) is now all zero */
            isize top = n;                     /* stack for pattern is empty */
            work.flag[k] = I(k);                /* mark node k as visited */
            L_nnz[k] = 0;                      /* count of nonzeros in column k of L */
            isize p2 = Ap[k + 1];
            for (isize p = Ap[k]; p < p2; p++)
            {
                isize i = Ai[p]; /* get A(i,k) */
                work.y[i] = Ax[p];  /* scatter A(i,k) into Y */
                isize len;
                for (len = 0; work.flag[i] != k; i = etree[i])
                {
                    work.pattern[len++] = I(i); /* L(k,i) is nonzero */
                    work.flag[i] = I(k);         /* mark i as visited */
                }
                while (len > 0) work.pattern[--top] = work.pattern[--len];
            }
            /* compute numerical values kth row of L (a sparse triangular solve) */
            D[k] = work.y[k]; /* get D(k,k) and clear Y(k) */
            work.y[k] = 0.0;
            for (; top < n ; top++)
            {
                isize i = work.pattern[top]; /* Pattern[top:n-1] is pattern of L(k,:) */
                T yi = work.y[i];            /* get and clear Y(i) */
                work.y[i] = 0.0;
                p2 = L_cols[i] + L_nnz[i];
                isize p;
                for (p = L_cols[i]; p < p2; p++)
                {
                    // force compiler to not use fma instruction
                    T tmp = L_vals[p] * yi;
                    work.y[L_ind[p]] -= tmp;
                }
                T l_ki = yi / D[i]; /* the nonzero entry L(k,i) */
                // force compiler to not use fma instruction
                T tmp = l_ki * yi;
                D[k] -= tmp;
                L_ind[p] = I(k);    /* store L(k,i) in column form of L */
                L_vals[p] = l_ki;
                L_nnz[i]++;         /* increment count of nonzeros in col i */
            }
            if (D[k] == 0.0) return k;     /* failure, D(k,k) is zero */
        }

        D_inv.array() = D.array().inverse();

        return n; /* success, diagonal of D is all nonzero */
    }

    void lsolve(Vec<T>& x)
    {
        PIQP_TRACY_ZoneScopedN("piqp::LDLt::lsolve");

        isize n = x.rows();
        eigen_assert(n == L_cols.rows() - 1 && "vector dimension missmatch!");
        for (isize j = 0; j < n; j++)
        {
            isize p2 = L_cols[j + 1];
            for (isize p = L_cols[j]; p < p2; p++)
            {
                x[L_ind[p]] -= L_vals[p] * x[j];
            }
        }
    }

    void dsolve(Vec<T>& x)
    {
        PIQP_TRACY_ZoneScopedN("piqp::LDLt::dsolve");

        PIQP_MAYBE_UNUSED isize n = x.rows();
        eigen_assert(n == D_inv.rows() && "vector dimension missmatch!");
        x.array() *= D_inv.array();
    }

    void ltsolve(Vec<T>& x)
    {
        PIQP_TRACY_ZoneScopedN("piqp::LDLt::ltsolve");

        isize n = x.rows();
        eigen_assert(n == L_cols.rows() - 1 && "vector dimension missmatch!");
        for (isize j = n - 1; j >= 0; j--)
        {
            isize p2 = L_cols[j + 1];
            for (isize p = L_cols[j]; p < p2; p++)
            {
                x[j] -= L_vals[p] * x[L_ind[p]];
            }
        }
    }

    void solve_inplace(Vec<T>& x)
    {
        PIQP_TRACY_ZoneScopedN("piqp::LDLt::solve_inplace");
        lsolve(x);
        dsolve(x);
        ltsolve(x);
    }
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/sparse/ldlt.tpp"
#endif

#endif //PIQP_SPARSE_LDLT_HPP
