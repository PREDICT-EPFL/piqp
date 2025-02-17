// This file is part of PIQP.
//
// Copyright (c) 2025 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_BLASFEO_WRAPPER
#define PIQP_BLASFEO_WRAPPER

#include <cstring>

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

// B <= alpha * A
static inline void blasfeo_dgecpsc(double alpha, BlasfeoMat& A, BlasfeoMat& B)
{
    int m = A.rows();
    int n = A.cols();
    assert(B.rows() >= m && B.cols() >= n && "size mismatch");
    blasfeo_dgecpsc(m, n, alpha, A.ref(), 0, 0, B.ref(), 0, 0);
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

} // namespace piqp

#endif //PIQP_BLASFEO_WRAPPER
