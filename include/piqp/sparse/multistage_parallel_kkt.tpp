// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifdef PIQP_HAS_BLASFEO
#ifdef PIQP_HAS_OPENMP

#ifndef PIQP_SPARSE_MULTISTAGE_PARALLEL_KKT_TPP
#define PIQP_SPARSE_MULTISTAGE_PARALLEL_KKT_TPP

#include "piqp/common.hpp"
#include "piqp/sparse/multistage_parallel_kkt.hpp"

namespace piqp
{

namespace sparse
{

// template<typename T, typename I>
// MultistageParallelKKT<T, I>::MultistageParallelKKT(const Data<T, I>& data) : MultistageKKT<T, I>(data)
// {
//     PIQP_TRACY_ZoneScopedN("piqp::MultistageParallelKKT::constructor");
//     init();
// }

template<typename T, typename I>
void MultistageParallelKKT<T, I>::populate_kkt_fac(const Vec<T>& x_reg)
{
            // ----- DIAGONAL -----

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 1; i++)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_syrk_ln::diagonal");
            PIQP_TRACY_ZoneValue(i);

            // D_{i,i} = 0
            if (!allocate && sD.D[i]) {
                sD.D[i]->setZero();
            }

            if (sA.D[i] && sB.D[i]) {
                if (allocate) {
                    if (!sD.D[i]) {
                        int m = sA.D[i]->rows();
                        sD.D[i] = std::make_unique<BlasfeoMat>(m, m);
                    }
                } else {
                    // D_{i,i} += lower triangular of A_{i,i} * B_{i,i}^T
                    blasfeo_dsyrk_ln(1.0, *sA.D[i], *sB.D[i], 1.0, *sD.D[i], *sD.D[i]);
                }
            }

            if (i > 0 && sA.B[i-1] && sB.B[i-1]) {
                if (allocate) {
                    if (!sD.D[i]) {
                        int m = sA.B[i-1]->rows();
                        sD.D[i] = std::make_unique<BlasfeoMat>(m, m);
                    }
                } else {
                    // D_{i,i} += lower triangular of A_{i-1,i} * B_{i-1,i}^T
                    blasfeo_dsyrk_ln(1.0, *sA.B[i-1], *sB.B[i-1], 1.0, *sD.D[i], *sD.D[i]);
                }
            }
        }

#ifdef PIQP_HAS_OPENMP
        #pragma omp single nowait
        {
#endif
        if (arrow_width > 0)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_syrk_ln::diagonal_last");

            // D_{N,N} = 0
            if (!allocate && sD.D[N-1]) {
                sD.D[N-1]->setZero();
            }

            for (std::size_t i = 0; i < N - 1; i++)
            {
                if (sA.E[i] && sB.E[i]) {
                    if (allocate) {
                        if (!sD.D[N-1]) {
                            int m = sA.E[i]->rows();
                            sD.D[N-1] = std::make_unique<BlasfeoMat>(m, m);
                        }
                    } else {
                        // D_{N,N} += lower triangular of A_{i,N} * B_{i,N}^T
                        blasfeo_dsyrk_ln(1.0, *sA.E[i], *sB.E[i], 1.0, *sD.D[N-1], *sD.D[N-1]);
                    }
                }
            }
        }

#ifdef PIQP_HAS_OPENMP
        } // end of single region
#endif

        // ----- OFF-DIAGONAL -----

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 2; i++)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_syrk_ln::off_diagonal");
            PIQP_TRACY_ZoneValue(i);

            if (sA.B[i] && sB.D[i]) {
                if (allocate) {
                    if (!sD.B[i]) {
                        int m = sA.B[i]->rows();
                        int n = sB.D[i]->rows();
                        sD.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    // D_{i+1,i} = A_{i,i+1} * B_{i,i}^T
                    blasfeo_dgemm_nt(1.0, *sA.B[i], *sB.D[i], 0.0, *sD.B[i], *sD.B[i]);
                }
            }
        }

        // ----- ARROW -----

        if (arrow_width > 0)
        {
#ifdef PIQP_HAS_OPENMP
            #pragma omp for nowait
#endif
            for (std::size_t i = 0; i < N - 1; i++)
            {
                PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_syrk_ln::arrow");
                PIQP_TRACY_ZoneValue(i);

                // D_{N,i} = 0
                if (!allocate && sD.E[i]) {
                    sD.E[i]->setZero();
                }

                if (sA.E[i] && sB.D[i]) {
                    if (allocate) {
                        if (!sD.E[i]) {
                            int m = sA.E[i]->rows();
                            int n = sB.D[i]->rows();
                            sD.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // D_{N,i} += A_{i,N} * B_{i,i}^T
                        blasfeo_dgemm_nt(1.0, *sA.E[i], *sB.D[i], 1.0, *sD.E[i], *sD.E[i]);
                    }
                }

                if (i > 0 && sA.E[i-1] && sB.B[i-1]) {
                    if (allocate) {
                        if (!sD.E[i]) {
                            int m = sA.E[i-1]->rows();
                            int n = sB.B[i-1]->rows();
                            sD.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // D_{N,i} += A_{i-1,N} * B_{i-1,i}^T
                        blasfeo_dgemm_nt(1.0, *sA.E[i-1], *sB.B[i-1], 1.0, *sD.E[i], *sD.E[i]);
                    }
                }
            }
        }

}


template<typename T, typename I>
void MultistageParallelKKT<T, I>::factor_kkt()
{
    PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::factor_kkt");

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;

        int m = kkt_fac.D[0]->rows();
        int n, k;
        // L_1 = chol(D_1)
        blasfeo_dpotrf_l(m, kkt_fac.D[0]->ref(), 0, 0, kkt_fac.D[0]->ref(), 0, 0);

        if (N > 2 && kkt_fac.B[0]) {
            m = kkt_fac.B[0]->rows();
            n = kkt_fac.B[0]->cols();
            assert(kkt_fac.D[0]->rows() == n && kkt_fac.D[0]->cols() == n && "size mismatch");
            // C_1 = B_1 * L_1^{-T}
            blasfeo_dtrsm_rltn(m, n, 1.0, kkt_fac.D[0]->ref(), 0, 0, kkt_fac.B[0]->ref(), 0, 0, kkt_fac.B[0]->ref(), 0, 0);
        }

        if (arrow_width > 0)
        {
            if (kkt_fac.E[0]) {
                m = kkt_fac.E[0]->rows();
                n = kkt_fac.E[0]->cols();
                assert(kkt_fac.D[0]->rows() == n && kkt_fac.D[0]->cols() == n && "size mismatch");
                // F_1 = E_1 * L_1^{-T}
                blasfeo_dtrsm_rltn(m, n, 1.0, kkt_fac.D[0]->ref(), 0, 0, kkt_fac.E[0]->ref(), 0, 0, kkt_fac.E[0]->ref(), 0, 0);
                // L_N = D_N - F_1 * F_1^T
                blasfeo_dsyrk_ln(arrow_width, n, -1.0, kkt_fac.E[0]->ref(), 0, 0, kkt_fac.E[0]->ref(), 0, 0, 1.0, kkt_fac.D[N-1]->ref(), 0, 0, kkt_fac.D[N-1]->ref(), 0, 0);
            } else {
                // L_N = D_N
                blasfeo_dtrcp_l(arrow_width, kkt_fac.D[N-1]->ref(), 0, 0, kkt_fac.D[N-1]->ref(), 0, 0);
            }
        }

        for (std::size_t i = 1; i < N - 1; i++)
        {
            if (kkt_fac.B[i-1]) {
                m = kkt_fac.B[i-1]->rows();
                k = kkt_fac.B[i-1]->cols();
                assert(kkt_fac.D[i]->rows() >= m && kkt_fac.D[i]->cols() >= m && "size mismatch");
                // L_i = chol(D_i - C_{i-1} * C_{i-1}^T)
                blasfeo_dsyrk_ln(m, k, -1.0, kkt_fac.B[i-1]->ref(), 0, 0, kkt_fac.B[i-1]->ref(), 0, 0, 1.0, kkt_fac.D[i]->ref(), 0, 0, kkt_fac.D[i]->ref(), 0, 0);
                m = kkt_fac.D[i]->rows();
                blasfeo_dpotrf_l(m, kkt_fac.D[i]->ref(), 0, 0, kkt_fac.D[i]->ref(), 0, 0);
            } else {
                m = kkt_fac.D[i]->rows();
                assert(kkt_fac.D[i]->rows() == m && "size mismatch");
                // L_i = chol(D_i)
                blasfeo_dpotrf_l(m, kkt_fac.D[i]->ref(), 0, 0, kkt_fac.D[i]->ref(), 0, 0);
            }

            if (i < N - 2 && kkt_fac.B[i]) {
                m = kkt_fac.B[i]->rows();
                n = kkt_fac.B[i]->cols();
                assert(kkt_fac.D[i]->rows() == n && kkt_fac.D[i]->cols() == n && "size mismatch");
                // C_i = B_i * L_i^{-T}
                blasfeo_dtrsm_rltn(m, n, 1.0, kkt_fac.D[i]->ref(), 0, 0, kkt_fac.B[i]->ref(), 0, 0, kkt_fac.B[i]->ref(), 0, 0);
            }

            if (arrow_width > 0)
            {
                if (kkt_fac.E[i] && kkt_fac.E[i-1] && kkt_fac.B[i-1])
                {
                    m = kkt_fac.E[i-1]->rows();
                    n = kkt_fac.B[i-1]->rows();
                    k = kkt_fac.E[i-1]->cols();
                    assert(kkt_fac.B[i-1]->cols() == k && "size mismatch");
                    assert(kkt_fac.E[i]->rows() == m && kkt_fac.E[i]->cols() >= n && "size mismatch");
                    assert(kkt_fac.D[i]->rows() >= n && kkt_fac.D[i]->cols() >= n && "size mismatch");
                    // F_i = (E_i - F_{i-1} * C_{i-1}^T) * L_i^{-T}
                    blasfeo_dgemm_nt(m, n, k, -1.0, kkt_fac.E[i-1]->ref(), 0, 0, kkt_fac.B[i-1]->ref(), 0, 0, 1.0, kkt_fac.E[i]->ref(), 0, 0, kkt_fac.E[i]->ref(), 0, 0);
                    n = kkt_fac.D[i]->rows();
                    blasfeo_dtrsm_rltn(m, n, 1.0, kkt_fac.D[i]->ref(), 0, 0, kkt_fac.E[i]->ref(), 0, 0, kkt_fac.E[i]->ref(), 0, 0);
                }
                else if (kkt_fac.E[i])
                {
                    m = kkt_fac.E[i]->rows();
                    n = kkt_fac.E[i]->cols();
                    assert(kkt_fac.D[i]->rows() == n && kkt_fac.D[i]->cols() == n && "size mismatch");
                    // F_i = E_i * L_i^{-T}
                    blasfeo_dtrsm_rltn(m, n, 1.0, kkt_fac.D[i]->ref(), 0, 0, kkt_fac.E[i]->ref(), 0, 0, kkt_fac.E[i]->ref(), 0, 0);
                }

                if (kkt_fac.E[i]) {
                    m = kkt_fac.E[i]->rows();
                    k = kkt_fac.E[i]->cols();
                    assert(m == arrow_width && "size mismatch");
                    assert(kkt_fac.D[N - 1]->rows() == m && kkt_fac.D[N - 1]->cols() == m && "size mismatch");
                    // L_N -= F_i * F_i^T
                    blasfeo_dsyrk_ln(m, k, -1.0, kkt_fac.E[i]->ref(), 0, 0, kkt_fac.E[i]->ref(), 0, 0, 1.0, kkt_fac.D[N - 1]->ref(), 0, 0, kkt_fac.D[N - 1]->ref(), 0, 0);
                }
            }
        }

        // L_N = chol(D_N - sum F_i * F_i^T)
        // note that inner is also computed and stored in L_N
        blasfeo_dpotrf_l(arrow_width, kkt_fac.D[N-1]->ref(), 0, 0, kkt_fac.D[N-1]->ref(), 0, 0);

}

template<typename T, typename I>
void MultistageParallelKKT<T, I>::solve_llt_in_place(BlockVec& b_and_x)
{
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::solve_llt_in_place");

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;
        int m, n;

        // ----- FORWARD SUBSTITUTION -----
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::solve_llt_in_place::forward");

            m = kkt_fac.D[0]->rows();
            assert(b_and_x.x[0].rows() == m && "size mismatch");
            // y_1 = L_1^{-1} * b_1
            blasfeo_dtrsv_lnn(m, kkt_fac.D[0]->ref(), 0, 0, b_and_x.x[0].ref(), 0, b_and_x.x[0].ref(), 0);

            for (std::size_t i = 1; i < N - 1; i++)
            {
                if (kkt_fac.B[i-1]) {
                    m = kkt_fac.B[i-1]->rows();
                    n = kkt_fac.B[i-1]->cols();
                    assert(kkt_fac.D[i]->rows() >= m && "size mismatch");
                    assert(b_and_x.x[i-1].rows() == n && "size mismatch");
                    assert(b_and_x.x[i].rows() >= m && "size mismatch");
                    // y_i = b_i - C_{i-1} * y_{i-1}
                    blasfeo_dgemv_n(m, n, -1.0, kkt_fac.B[i-1]->ref(), 0, 0, b_and_x.x[i-1].ref(), 0, 1.0, b_and_x.x[i].ref(), 0, b_and_x.x[i].ref(), 0);
                }
                m = kkt_fac.D[i]->rows();
                assert(b_and_x.x[i].rows() == m && "size mismatch");
                // y_i = L_i^{-1} * y_i
                blasfeo_dtrsv_lnn(m, kkt_fac.D[i]->ref(), 0, 0, b_and_x.x[i].ref(), 0, b_and_x.x[i].ref(), 0);
            }

            if (arrow_width > 0)
            {
                for (std::size_t i = 0; i < N - 1; i++)
                {
                    if (kkt_fac.E[i]) {
                        m = kkt_fac.E[i]->rows();
                        n = kkt_fac.E[i]->cols();
                        assert(b_and_x.x[i].rows() == n && "size mismatch");
                        assert(b_and_x.x[N-1].rows() == m && "size mismatch");
                        // y_N -= F_i * y_i
                        blasfeo_dgemv_n(m, n, -1.0, kkt_fac.E[i]->ref(), 0, 0, b_and_x.x[i].ref(), 0, 1.0, b_and_x.x[N-1].ref(), 0, b_and_x.x[N-1].ref(), 0);
                    }
                }
                m = kkt_fac.D[N-1]->rows();
                assert(b_and_x.x[N-1].rows() == m && "size mismatch");
                // y_N = L_N^{-1} * y_N
                blasfeo_dtrsv_lnn(m, kkt_fac.D[N-1]->ref(), 0, 0, b_and_x.x[N-1].ref(), 0, b_and_x.x[N-1].ref(), 0);
            }
        }

        // ----- BACK SUBSTITUTION -----

        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::solve_llt_in_place::backward");

            if (arrow_width > 0)
            {
                m = kkt_fac.D[N-1]->rows();
                assert(b_and_x.x[N-1].rows() == m && "size mismatch");
                // x_N = L_N^{-T} * y_N
                blasfeo_dtrsv_ltn(m, kkt_fac.D[N-1]->ref(), 0, 0, b_and_x.x[N-1].ref(), 0, b_and_x.x[N-1].ref(), 0);

                if (kkt_fac.E[N-2]) {
                    m = kkt_fac.E[N-2]->rows();
                    n = kkt_fac.E[N-2]->cols();
                    assert(b_and_x.x[N-1].rows() == m && "size mismatch");
                    assert(b_and_x.x[N-2].rows() == n && "size mismatch");
                    // x_{N-1} = y_{N-1} - F_{N-1}^T * x_N
                    blasfeo_dgemv_t(m, n, -1.0, kkt_fac.E[N-2]->ref(), 0, 0, b_and_x.x[N-1].ref(), 0, 1.0, b_and_x.x[N-2].ref(), 0, b_and_x.x[N-2].ref(), 0);
                }
            }

            m = kkt_fac.D[N-2]->rows();
            assert(b_and_x.x[N-2].rows() == m && "size mismatch");
            // x_{N-1} = L_{N-1}^{-T} * x_{N-1}
            blasfeo_dtrsv_ltn(m, kkt_fac.D[N-2]->ref(), 0, 0, b_and_x.x[N-2].ref(), 0, b_and_x.x[N-2].ref(), 0);

            for (std::size_t i = N - 2; i--;)
            {
                if (kkt_fac.B[i]) {
                    m = kkt_fac.B[i]->rows();
                    n = kkt_fac.B[i]->cols();
                    assert(b_and_x.x[i+1].rows() >= m && "size mismatch");
                    assert(b_and_x.x[i].rows() == n && "size mismatch");
                    // x_i = y_i - C_i^T * x_{i+1}
                    blasfeo_dgemv_t(m, n, -1.0, kkt_fac.B[i]->ref(), 0, 0, b_and_x.x[i+1].ref(), 0, 1.0, b_and_x.x[i].ref(), 0, b_and_x.x[i].ref(), 0);
                }

                if (kkt_fac.E[i]) {
                    m = kkt_fac.E[i]->rows();
                    n = kkt_fac.E[i]->cols();
                    assert(b_and_x.x[N-1].rows() == m && "size mismatch");
                    assert(b_and_x.x[i].rows() == n && "size mismatch");
                    // x_i -= F_i^T * x_N
                    blasfeo_dgemv_t(m, n, -1.0, kkt_fac.E[i]->ref(), 0, 0, b_and_x.x[N-1].ref(), 0, 1.0, b_and_x.x[i].ref(), 0, b_and_x.x[i].ref(), 0);
                }

                m = kkt_fac.D[i]->rows();
                assert(b_and_x.x[i].rows() == m && "size mismatch");
                // x_i = L_i^{-T} * x_i
                blasfeo_dtrsv_ltn(m, kkt_fac.D[i]->ref(), 0, 0, b_and_x.x[i].ref(), 0, b_and_x.x[i].ref(), 0);
            }
        }
}

extern template class MultistageParallelKKT<common::Scalar, common::StorageIndex>;

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_MULTISTAGE_PARALLEL_KKT_TPP

#endif // PIQP_HAS_OPENMP
#endif // PIQP_HAS_BLASFEO