// This file is part of PIQP.
//
// Copyright (c) 2025 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_BLASFEO_WRAPPER
#define PIQP_BLASFEO_WRAPPER

#include <cstring>
#include <cassert>

#include "blasfeo.h"
#include "piqp/utils/blasfeo_mat.hpp"
#include "piqp/utils/blasfeo_vec.hpp"

namespace piqp
{

// B <= A
static inline void blasfeo_dgecp(BlasfeoMat& A, BlasfeoMat& B)
{
    int m = A.rows();
    int n = A.cols();
    assert(B.rows() >= m && B.cols() >= n && "size mismatch");
    blasfeo_dgecp(m, n, A.ref(), 0, 0, B.ref(), 0, 0);
}

static inline void blasfeo_dgetr(BlasfeoMat& A, BlasfeoMat& B)
{
    int m = A.rows();
    int n = A.cols();
    assert(B.cols() >= m && B.rows() >= n && "size mismatch");
    blasfeo_dgetr(m, n, A.ref(), 0, 0, B.ref(), 0, 0);
}

// B <= alpha * A
static inline void blasfeo_dgecpsc(double alpha, BlasfeoMat& A, BlasfeoMat& B)
{
    int m = A.rows();
    int n = A.cols();
    assert(B.rows() >= m && B.cols() >= n && "size mismatch");
#ifdef TARGET_X64_INTEL_SKYLAKE_X
    // blasfeo_dgecpsc not implemented on Skylake yet
    // and reference implementation not exported ...
    B.setZero();
    blasfeo_dgead(m, n, alpha, A.ref(), 0, 0, B.ref(), 0, 0);
#else
    blasfeo_dgecpsc(m, n, alpha, A.ref(), 0, 0, B.ref(), 0, 0);
#endif
}

// B <= A, A lower triangular
static inline void blasfeo_dtrcp_l(BlasfeoMat& A, BlasfeoMat& B)
{
    int m = A.rows();
    assert(A.cols() == m && B.rows() == m && B.cols() == m && "size mismatch");
    blasfeo_dtrcp_l(m, A.ref(), 0, 0, B.ref(), 0, 0);
}

// B <= alpha * A, A lower triangular
static inline void blasfeo_dtrcpsc_l(double alpha, BlasfeoMat& A, BlasfeoMat& B)
{
    int m = A.rows();
    assert(A.cols() == m && B.rows() == m && B.cols() == m && "size mismatch");
#ifdef TARGET_X64_INTEL_SKYLAKE_X
    // blasfeo_dtrcpsc_l not implemented on Skylake yet
    // and reference implementation not exported ...
    B.setZero();
    blasfeo_dgead(m, m, alpha, A.ref(), 0, 0, B.ref(), 0, 0);
#else
    blasfeo_dtrcpsc_l(m, alpha, A.ref(), 0, 0, B.ref(), 0, 0);
#endif
}

// B <= B + alpha * A
static inline void blasfeo_dgead(double alpha, BlasfeoMat& A, BlasfeoMat& B)
{
    int m = A.rows();
    int n = A.cols();
    assert(B.rows() >= m && B.cols() >= n && "size mismatch");
    blasfeo_dgead(m, n, alpha, A.ref(), 0, 0, B.ref(), 0, 0);
}

// diag(A) <= alpha * x
static inline void blasfeo_ddiain(double alpha, BlasfeoVec& x, BlasfeoMat& A)
{
    int kmax = x.rows();
    assert(A.rows() == kmax && A.cols() == kmax && "size mismatch");
    blasfeo_ddiain(kmax, alpha, x.ref(), 0, A.ref(), 0, 0);
}

// diag(A) += alpha * x
static inline void blasfeo_ddiaad(double alpha, BlasfeoVec& x, BlasfeoMat& A)
{
    int kmax = x.rows();
    assert(A.rows() == kmax && A.cols() == kmax && "size mismatch");
    blasfeo_ddiaad(kmax, alpha, x.ref(), 0, A.ref(), 0, 0);
}


// D <= beta * C + alpha * A * B^T
static inline void blasfeo_dgemm_nt(double alpha, BlasfeoMat& A, BlasfeoMat& B, double beta, BlasfeoMat& C, BlasfeoMat& D)
{
    int m = A.rows();
    int n = B.rows();
    int k = A.cols();
    assert(B.cols() == k && "size mismatch");
    assert(C.rows() >= m && C.cols() >= n && "size mismatch");
    assert(D.rows() >= m && D.cols() >= n && "size mismatch");
    blasfeo_dgemm_nt(m, n, k, alpha, A.ref(), 0, 0, B.ref(), 0, 0, beta, C.ref(), 0, 0, D.ref(), 0, 0);
}

// D <= beta * C + alpha * A * B
static inline void blasfeo_dgemm_nn(double alpha, BlasfeoMat& A, BlasfeoMat& B, double beta, BlasfeoMat& C, BlasfeoMat& D)
{
    int m = A.rows();
    int n = B.cols();
    int k = B.rows();
    assert(A.cols() == k && "size mismatch");
    assert(C.rows() >= m && C.cols() >= n && "size mismatch");
    assert(D.rows() >= m && D.cols() >= n && "size mismatch");
    blasfeo_dgemm_nn(m, n, k, alpha, A.ref(), 0, 0, B.ref(), 0, 0, beta, C.ref(), 0, 0, D.ref(), 0, 0);
}

// D <= beta * C + alpha * A^T * B
static inline void blasfeo_dgemm_tn(double alpha, BlasfeoMat& A, BlasfeoMat& B, double beta, BlasfeoMat& C, BlasfeoMat& D)
{
    int m = A.cols();
    int n = B.cols();
    int k = A.rows();
    assert(B.rows() == k && "size mismatch");
    assert(C.rows() >= m && C.cols() >= n && "size mismatch");
    assert(D.rows() >= m && D.cols() >= n && "size mismatch");
    blasfeo_dgemm_tn(m, n, k, alpha, A.ref(), 0, 0, B.ref(), 0, 0, beta, C.ref(), 0, 0, D.ref(), 0, 0);
}

// D <= beta * C + alpha * A^T * B^T
static inline void blasfeo_dgemm_tt(double alpha, BlasfeoMat& A, BlasfeoMat& B, double beta, BlasfeoMat& C, BlasfeoMat& D)
{
    int m = A.cols();
    int n = B.rows();
    int k = A.rows();
    assert(B.cols() == k && "size mismatch");
    assert(C.rows() >= m && C.cols() >= n && "size mismatch");
    assert(D.rows() >= m && D.cols() >= n && "size mismatch");
    blasfeo_dgemm_tt(m, n, k, alpha, A.ref(), 0, 0, B.ref(), 0, 0, beta, C.ref(), 0, 0, D.ref(), 0, 0);
}

// D <= beta * C + alpha * A * B^T ; C, D lower triangular
static inline void blasfeo_dsyrk_ln(double alpha, BlasfeoMat& A, BlasfeoMat& B, double beta, BlasfeoMat& C, BlasfeoMat& D)
{
    int m = A.rows();
    int k = A.cols();
    assert(B.rows() == m && B.cols() == k && "size mismatch");
    assert(C.rows() >= m && C.cols() >= m && "size mismatch");
    assert(D.rows() >= m && D.cols() >= m && "size mismatch");
    blasfeo_dsyrk_ln(m, k, alpha, A.ref(), 0, 0, B.ref(), 0, 0, beta, C.ref(), 0, 0, D.ref(), 0, 0);
}

// D <= beta * C + alpha * A^T * B ; C, D lower triangular
static inline void blasfeo_dsyrk_lt(double alpha, BlasfeoMat& A, BlasfeoMat& B, double beta, BlasfeoMat& C, BlasfeoMat& D)
{
    int m = A.cols();
    int k = A.rows();
    assert(B.rows() == k && B.cols() == m && "size mismatch");
    assert(C.rows() >= m && C.cols() >= m && "size mismatch");
    assert(D.rows() >= m && D.cols() >= m && "size mismatch");
    blasfeo_dsyrk_lt(m, k, alpha, A.ref(), 0, 0, B.ref(), 0, 0, beta, C.ref(), 0, 0, D.ref(), 0, 0);
}


// D <= alpha * A * B + beta * C, with B diagonal
static inline void blasfeo_dgemm_nd(double alpha, BlasfeoMat& A, BlasfeoVec& B, double beta, BlasfeoMat& C, BlasfeoMat& D)
{
    int m = A.rows();
    int n = A.cols();
    assert(B.rows() == n && "size mismatch");
    assert(C.rows() == m && C.cols() == n && "size mismatch");
    assert(D.rows() == m && D.cols() == n && "size mismatch");
    blasfeo_dgemm_nd(m, n, alpha, A.ref(), 0, 0, B.ref(), 0, beta, C.ref(), 0, 0, D.ref(), 0, 0);
}

// D <= alpha * A^{-1} * B , with A lower triangular employing explicit inverse of diagonal
static inline void blasfeo_dtrsm_llnn(double alpha, BlasfeoMat& A, BlasfeoMat& B, BlasfeoMat& D)
{
    int m = A.rows();
    int n = B.cols();
    assert(A.cols() == m && B.rows() == m && "size mismatch");
    assert(D.rows() >= m && D.cols() >= n && "size mismatch");
    blasfeo_dtrsm_llnn(m, n, alpha, A.ref(), 0, 0, B.ref(), 0, 0, D.ref(), 0, 0);
}

// D <= alpha * A^{-T} * B , with A lower triangular employing explicit inverse of diagonal
static inline void blasfeo_dtrsm_lltn(double alpha, BlasfeoMat& A, BlasfeoMat& B, BlasfeoMat& D)
{
    int m = A.rows();
    int n = B.cols();
    assert(A.cols() == m && B.rows() == m && "size mismatch");
    assert(D.rows() >= m && D.cols() >= n && "size mismatch");
    blasfeo_dtrsm_lltn(m, n, alpha, A.ref(), 0, 0, B.ref(), 0, 0, D.ref(), 0, 0);
}

// D <= chol(C) ; C, D lower triangular
static inline void blasfeo_dpotrf_l(BlasfeoMat& C)
{
    assert(C.rows() == C.cols() && "size mismatch");
    blasfeo_dpotrf_l(C.rows(), C.ref(), 0, 0, C.ref(), 0, 0);
    // if blasfeo take square root of negative number, it will just set the diagonal entry to 0.
    // therefore if we find a non-positive diagonal entry in the factorization, it implies the original matrix is not positive definite
    for (int i = 0; i < C.rows(); i++) {
        assert(BLASFEO_DMATEL(C.ref(), i, i) > 0.0 && "matrix not positive definite");
    }
}

// z <= beta * y + alpha * A * x
static inline void blasfeo_dgemv_n(double alpha, BlasfeoMat& A, BlasfeoVec& x, double beta, BlasfeoVec& y, BlasfeoVec& z)
{
    int m = A.rows();
    int n = A.cols();
    assert(x.rows() == n && "size mismatch");
    assert(y.rows() >= m && z.rows() >= m && "size mismatch");
    blasfeo_dgemv_n(m, n, alpha, A.ref(), 0, 0, x.ref(), 0, beta, y.ref(), 0, z.ref(), 0);
}

// z <= beta * y + alpha * A^T * x
static inline void blasfeo_dgemv_t(double alpha, BlasfeoMat& A, BlasfeoVec& x, double beta, BlasfeoVec& y, BlasfeoVec& z)
{
    int m = A.rows();
    int n = A.cols();
    assert(x.rows() == m && "size mismatch");
    assert(y.rows() >= n && z.rows() >= n && "size mismatch");
    blasfeo_dgemv_t(m, n, alpha, A.ref(), 0, 0, x.ref(), 0, beta, y.ref(), 0, z.ref(), 0);
}

// z <= inv(A) * x, A (m)x(m) lower, not_transposed
static inline void blasfeo_dtrsv_lnn(BlasfeoMat& A, BlasfeoVec& x, BlasfeoVec& z)
{
    int m = A.rows();
    assert(A.cols() == m && "size mismatch");
    assert(x.rows() >= m && z.rows() >= m && "size mismatch");
    blasfeo_dtrsv_lnn(m, A.ref(), 0, 0, x.ref(), 0, z.ref(), 0);
}

// z <= inv(A^T) * x, A (m)x(m) lower, transposed
static inline void blasfeo_dtrsv_ltn(BlasfeoMat& A, BlasfeoVec& x, BlasfeoVec& z)
{
    int m = A.rows();
    assert(A.cols() == m && "size mismatch");
    assert(x.rows() >= m && z.rows() >= m && "size mismatch");
    blasfeo_dtrsv_ltn(m, A.ref(), 0, 0, x.ref(), 0, z.ref(), 0);
}

} // namespace piqp

#endif //PIQP_BLASFEO_WRAPPER
