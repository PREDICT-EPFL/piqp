// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_SPARSE_UTILS_HPP
#define PIQP_UTILS_SPARSE_UTILS_HPP

namespace piqp
{

/*
 * Permutes a symmetric matrix A (only upper triangular used) using the provided ordering
 * and saves the upper triangular permutation in C, i.e., C = A(p, p).
 * The function returns the mapping P that maps the row indices of A to the row indices of C.
 *
 * @param A        symmetric matrix (only upper triangular used)
 * @param C        permutation upper triangular part
 * @param ordering determines the permutation
 *
 * @return mapping that maps the row indices of A to the row indices of C
 */
template<typename T, typename I, typename Ordering>
Vec<I> permute_sparse_symmetric_matrix(const SparseMat<T, I>& A, SparseMat<T, I>& C, const Ordering& ordering)
{
    static_assert(!SparseMat<T, I>::IsRowMajor, "A has to be column major!");
    eigen_assert(A.rows() == A.cols() && "A has to be symmetric!");
    isize n = A.rows();

    // 1st pass: we permute the matrix and write it in the lower triangular part
    //           with unsorted row indices

    // working vector
    Vec<T> w(n);
    w.setZero();

    for (isize j = 0; j < n; j++)
    {
        isize j2 = ordering.inv(j);
        for (typename SparseMat<T, I>::InnerIterator it(A, j); it; ++it)
        {
            isize i = it.index();
            if (i > j) continue; // we only consider upper triangular part
            isize i2 = ordering.inv(i);
            w(i2 < j2 ? i2 : j2)++;
        }
    }

    SparseMat<T, I> CT;
    CT.resize(n, n);
    // fill outer starts (cumulative column nnz count)
    isize sum = 0;
    for (isize i = 0; i < n; i++)
    {
        CT.outerIndexPtr()[i] = sum;
        sum += w(i);
        w(i) = CT.outerIndexPtr()[i];
    }
    CT.outerIndexPtr()[n] = sum;
    CT.resizeNonZeros(sum);

    Vec<I> CTi_to_Ai(sum);

    for (isize j = 0; j < n; j++)
    {
        isize j2 = ordering.inv(j);
        for (isize k = A.outerIndexPtr()[j]; k < A.outerIndexPtr()[j + 1]; k++)
        {
            isize i = A.innerIndexPtr()[k];
            if (i > j) continue; // we only consider upper triangular part
            isize i2 = ordering.inv(i);
            isize q = w(i2 < j2 ? i2 : j2)++;
            CT.innerIndexPtr()[q] = i2 > j2 ? i2 : j2;
            CT.valuePtr()[q] = A.valuePtr()[k];
            CTi_to_Ai(q) = k;
        }
    }

    // 2nd pass: by transposing we get the upper triangular part and sort all row indices automatically
    C.resize(n, n);
    for (isize j = 0; j < CT.outerSize(); j++)
    {
        for (typename SparseMat<T, I>::InnerIterator it(CT, j); it; ++it)
        {
            C.outerIndexPtr()[it.index()]++;
        }
    }

    sum = 0;
    for (isize j = 0; j < C.outerSize(); j++)
    {
        isize tmp = C.outerIndexPtr()[j];
        C.outerIndexPtr()[j] = sum;
        w(j) = sum;
        sum += tmp;
    }
    C.outerIndexPtr()[n] = sum;
    C.resizeNonZeros(sum);

    Vec<I> Ai_to_Ci(sum);

    for (isize j = 0; j < CT.outerSize(); ++j)
    {
        for (isize k = CT.outerIndexPtr()[j]; k < CT.outerIndexPtr()[j + 1]; k++)
        {
            isize i = CT.innerIndexPtr()[k];
            isize q = w(i)++;
            C.innerIndexPtr()[q] = j;
            C.valuePtr()[q] = CT.valuePtr()[k];
            Ai_to_Ci(CTi_to_Ai(k)) = q;
        }
    }

    return Ai_to_Ci;
}

} // namespace piqp

#endif //PIQP_UTILS_SPARSE_UTILS_HPP
