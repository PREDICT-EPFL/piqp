// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_UTILS_HPP
#define PIQP_SPARSE_UTILS_HPP

#include "piqp/typedefs.hpp"

namespace piqp
{

namespace sparse
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
    Vec<I> w(n);
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
        CT.outerIndexPtr()[i] = I(sum);
        sum += w(i);
        w(i) = CT.outerIndexPtr()[i];
    }
    CT.outerIndexPtr()[n] = I(sum);
    CT.resizeNonZeros(sum);

    Vec<I> CTi_to_Ai(sum);

    for (isize j = 0; j < n; j++)
    {
        isize j2 = ordering.inv(j);
        isize kk = A.outerIndexPtr()[j + 1];
        for (isize k = A.outerIndexPtr()[j]; k < kk; k++)
        {
            isize i = A.innerIndexPtr()[k];
            if (i > j) continue; // we only consider upper triangular part
            isize i2 = ordering.inv(i);
            isize q = w(i2 < j2 ? i2 : j2)++;
            CT.innerIndexPtr()[q] = I(i2 > j2 ? i2 : j2);
            CT.valuePtr()[q] = A.valuePtr()[k];
            CTi_to_Ai(q) = I(k);
        }
    }

    // 2nd pass: by transposing we get the upper triangular part and sort all row indices automatically
    C.resize(n, n);
    isize jj = CT.outerSize();
    for (isize j = 0; j < jj; j++)
    {
        for (typename SparseMat<T, I>::InnerIterator it(CT, j); it; ++it)
        {
            C.outerIndexPtr()[it.index()]++;
        }
    }

    sum = 0;
    jj = C.outerSize();
    for (isize j = 0; j < jj; j++)
    {
        isize tmp = C.outerIndexPtr()[j];
        C.outerIndexPtr()[j] = I(sum);
        w(j) = I(sum);
        sum += tmp;
    }
    C.outerIndexPtr()[n] = I(sum);
    C.resizeNonZeros(sum);

    Vec<I> Ai_to_Ci(sum);

    jj = CT.outerSize();
    for (isize j = 0; j < jj; ++j)
    {
        isize kk = CT.outerIndexPtr()[j + 1];
        for (isize k = CT.outerIndexPtr()[j]; k < kk; k++)
        {
            isize i = CT.innerIndexPtr()[k];
            isize q = w(i)++;
            C.innerIndexPtr()[q] = I(j);
            C.valuePtr()[q] = CT.valuePtr()[k];
            Ai_to_Ci(CTi_to_Ai(k)) = I(q);
        }
    }

    return Ai_to_Ci;
}

/*
 * Transposes a matrix A to matrix C = A.transpose() without any allocations by fully reusing the memory in C.
 * Note that C has to already have the same pattern as A.transpose().
 *
 * @param A  input matrix
 * @param C  transpose of matrix A, C has to already be allocated, only values are copied
 */
template<typename T, typename I>
void transpose_no_allocation(const CSparseMatRef<T, I>& A, SparseMat<T, I>& C)
{
    eigen_assert(A.outerSize() == C.innerSize() && A.innerSize() == C.outerSize() && "sparsity pattern of C does not match AT!");
    isize jj = A.outerSize();
    for (isize j = 0; j < jj; ++j)
    {
        isize kk = A.outerIndexPtr()[j + 1];
        for (isize k = A.outerIndexPtr()[j]; k < kk; k++)
        {
            isize i = A.innerIndexPtr()[k];
            // we are abusing the outer index pointer as a temporary
            isize q = C.outerIndexPtr()[i]++;
            eigen_assert(C.outerIndexPtr()[i] <= C.outerIndexPtr()[i + 1] && "sparsity pattern of C does not match AT!");
            C.innerIndexPtr()[q] = I(j);
            C.valuePtr()[q] = A.valuePtr()[k];
        }
    }
    // revert outer index pointer which has been abused as a temporary
    isize m = A.innerSize();
    eigen_assert(m == 0 || ((C.outerIndexPtr()[m - 1] == C.outerIndexPtr()[m]) && "sparsity pattern of C does not match AT!"));
    for (isize j = m - 1; j > 0; j--)
    {
        C.outerIndexPtr()[j] = C.outerIndexPtr()[j - 1];
    }
    C.outerIndexPtr()[0] = 0;
}

/*
 * Pre multiplies a sparse matrix A with a diagonal matrix D, i.e. A = D * A
 *
 * @param A     input matrix A
 * @param diag  diagonal elements of D
 */
template<typename T, typename I>
void pre_mult_diagonal(SparseMat<T, I>& A, const CVecRef<T>& diag)
{
    isize n = A.outerSize();
    for (isize j = 0; j < n; j++)
    {
        for (typename SparseMat<T, I>::InnerIterator A_it(A, j); A_it; ++A_it)
        {
            A_it.valueRef() *= diag(A_it.row());
        }
    }
}

/*
 * Post multiplies a sparse matrix A with a diagonal matrix D, i.e. A = A * D
 *
 * @param A     input matrix A
 * @param diag  diagonal elements of D
 */
template<typename T, typename I>
void post_mult_diagonal(SparseMat<T, I>& A, const CVecRef<T>& diag)
{
    isize n = A.outerSize();
    for (isize j = 0; j < n; j++)
    {
        isize col_nnz = A.outerIndexPtr()[j + 1] - A.outerIndexPtr()[j];
        Eigen::Map<Vec<T>>(A.valuePtr() + A.outerIndexPtr()[j], col_nnz).array() *= diag(j);
    }
}

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_UTILS_HPP
