// This file is part of PIQP.
//
// Copyright (c) 2025 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_MULTISTAGE_KKT_HPP
#define PIQP_SPARSE_MULTISTAGE_KKT_HPP

#include <cassert>
#include <memory>
#include "blasfeo.h"

#ifdef PIQP_HAS_OPENMP
#include "omp.h"
#endif

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"
#include "piqp/kkt_solver_base.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/settings.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/utils/blasfeo_mat.hpp"
#include "piqp/utils/blasfeo_vec.hpp"
#include "piqp/utils/blasfeo_wrapper.hpp"
#include "piqp/sparse/blocksparse/block_info.hpp"
#include "piqp/sparse/blocksparse/block_kkt.hpp"
#include "piqp/sparse/blocksparse/block_mat.hpp"
#include "piqp/sparse/blocksparse/block_vec.hpp"
#include "piqp/utils/tracy.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
class MultistageKKT : public KKTSolverBase<T, I, PIQP_SPARSE>
{
protected:
    static_assert(std::is_same<T, double>::value, "sparse_multistage only supports doubles");

    T m_delta;

    Vec<T> m_z_reg_inv;
    Vec<T> work_x;
    Vec<T> work_z;

    std::vector<BlockInfo<I>> block_info;

    BlockKKT P;
    BlockVec P_diag;
    BlockMat<I> AT;
    BlockMat<I> GT;
    BlockVec G_scaling;
    BlockMat<I> GT_scaled;

    BlockKKT AtA;
    BlockKKT GtG;

    BlockKKT kkt_fac;

    BlockVec work_x_block_1;
    BlockVec work_x_block_2;
    BlockVec work_y_block_1;
    BlockVec work_y_block_2;
    BlockVec work_z_block_1;
    BlockVec work_z_block_2;

public:
    MultistageKKT(const Data<T, I>& data)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::constructor");

        // init workspace
        m_delta = T(1);

        m_z_reg_inv.resize(data.m);
        work_x.resize(data.n);
        work_z.resize(data.m);

        // prepare kkt factorization
        extract_arrow_structure(data);
        std::size_t N = block_info.size();

        utri_to_kkt(data.P_utri, P);
        P_diag = BlockVec(block_info);
        // P_diag <= diag(P)
        for (std::size_t i = 0; i < N; i++)
        {
            if (P.D[i]) {
                assert(P_diag.x[i].rows() == P.D[i]->rows() && "size mismatch");
                blasfeo_ddiaex(P_diag.x[i].rows(), 1.0, P.D[i]->ref(), 0, 0, P_diag.x[i].ref(), 0);
            }
        }

        transpose_to_block_mat<true>(data.AT, true, AT);
        transpose_to_block_mat<true>(data.GT, true, GT);
        G_scaling = BlockVec(GT.block_row_sizes);
        GT_scaled = GT;

#ifdef PIQP_HAS_OPENMP
#pragma omp parallel
        {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::constructor:parallel");
#endif

        block_syrk_ln_alloc(AT, AT, AtA);
#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#endif
        block_syrk_ln_calc(AT, AT, AtA);
        block_syrk_ln_alloc(GT, GT_scaled, GtG);

#ifdef PIQP_HAS_OPENMP
        } // end of parallel region
#endif

        // when we are allocating the kkt matrix there are
        // dependencies which leads to race conditions,
        // thus, we have to allocate on a single thread
        init_kkt_fac();

        work_x_block_1 = BlockVec(block_info);
        work_x_block_2 = BlockVec(block_info);
        work_y_block_1 = BlockVec(AT.block_row_sizes);
        work_y_block_2 = BlockVec(AT.block_row_sizes);
        work_z_block_1 = BlockVec(GT.block_row_sizes);
        work_z_block_2 = BlockVec(GT.block_row_sizes);
    }

    std::unique_ptr<KKTSolverBase<T, I, PIQP_SPARSE>> clone() const override
    {
        return std::make_unique<MultistageKKT>(*this);
    }

    void update_data(const Data<T, I>& data, int options) override
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::update_data");

        std::size_t N = block_info.size();

        if (options & KKTUpdateOptions::KKT_UPDATE_P)
        {
            utri_to_kkt(data.P_utri, P);

            // P_diag <= diag(P)
            for (std::size_t i = 0; i < N; i++)
            {
                if (P.D[i]) {
                    assert(P_diag.x[i].rows() == P.D[i]->rows() && "size mismatch");
                    blasfeo_ddiaex(P_diag.x[i].rows(), 1.0, P.D[i]->ref(), 0, 0, P_diag.x[i].ref(), 0);
                }
            }
        }

        if (options & KKTUpdateOptions::KKT_UPDATE_A)
        {
            transpose_to_block_mat<false>(data.AT, true, AT);
#ifdef PIQP_HAS_OPENMP
#pragma omp parallel
            {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::update_data:parallel");
#endif
            block_syrk_ln_calc(AT, AT, AtA);
#ifdef PIQP_HAS_OPENMP
            } // end of parallel region
#endif
        }

        if (options & KKTUpdateOptions::KKT_UPDATE_G)
        {
            transpose_to_block_mat<false>(data.GT, true, GT);
        }
    }

    bool update_scalings_and_factor(const Data<T, I>&, const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg) override
    {
        m_delta = delta;
        m_z_reg_inv.array() = z_reg.array().inverse();

        // populate G scaling vector
        Eigen::Index i = 0;
        for (I block_idx = 0; block_idx < GT.block_row_sizes.rows(); block_idx++)
        {
            I block_size = GT.block_row_sizes(block_idx);
            for (I inner_idx = 0; inner_idx < block_size; inner_idx++)
            {
                I perm_idx = GT.perm_inv(i);
                BLASFEO_DVECEL(G_scaling.x[std::size_t(block_idx)].ref(), inner_idx) = std::sqrt(m_z_reg_inv(perm_idx));
                i++;
            }
        }
#ifdef PIQP_HAS_OPENMP
#pragma omp parallel
        {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::update_scalings_and_factor:parallel");
#endif

        block_gemm_nd(GT, G_scaling, GT_scaled);
#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#endif
        block_syrk_ln_calc(GT_scaled, GT_scaled, GtG);
#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#endif
        populate_kkt_fac(x_reg);

#ifdef PIQP_HAS_OPENMP
        } // end of parallel region
#endif
        factor_kkt();

        return true;
    }

    void solve(const Data<T, I>&, const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z, Vec<T>& lhs_x, Vec<T>& lhs_y, Vec<T>& lhs_z) override
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::solve");

        Vec<T>& rhs_z_bar = work_z;
        BlockVec& block_rhs = work_x_block_1;
        BlockVec& block_rhs_y = work_y_block_1;
        BlockVec& block_rhs_z_bar = work_z_block_1;

        BlockVec& block_lhs_x = block_rhs;
        BlockVec& block_lhs_y = work_y_block_1;
        BlockVec& block_lhs_z = work_z_block_1;

        T delta_inv = T(1) / m_delta;

        rhs_z_bar.array() = m_z_reg_inv.array() * rhs_z.array();

        block_rhs.assign(rhs_x);
        block_rhs_y.assign(rhs_y, AT.perm_inv);
        block_rhs_z_bar.assign(rhs_z_bar, GT.perm_inv);


#ifdef PIQP_HAS_OPENMP
#pragma omp parallel
        {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::solve:parallel");
#endif

        // block_rhs += GT * block_rhs_z_bar
        block_t_gemv_n(1.0, GT, block_rhs_z_bar, 1.0, block_rhs, block_rhs);
#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#endif
        // block_rhs += delta_inv * AT * block_rhs_y
        block_t_gemv_n(delta_inv, AT, block_rhs_y, 1.0, block_rhs, block_rhs);

#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#pragma omp master
        {
#endif

        solve_llt_in_place(block_rhs);

#ifdef PIQP_HAS_OPENMP
        } // end of master region
#pragma omp barrier
#endif

        // block_lhs_y = delta_inv * A * block_lhs_x
        block_t_gemv_t(delta_inv, AT, block_lhs_x, 0.0, block_lhs_y, block_lhs_y);
        // block_lhs_z = G * block_lhs_x
        block_t_gemv_t(1.0, GT, block_lhs_x, 0.0, block_lhs_z, block_lhs_z);

#ifdef PIQP_HAS_OPENMP
        } // end of parallel region
#endif


        block_lhs_x.load(lhs_x);
        block_lhs_y.load(lhs_y, AT.perm_inv);
        block_lhs_z.load(lhs_z, GT.perm_inv);

        lhs_y.noalias() -= delta_inv * rhs_y;

        lhs_z.noalias() -= rhs_z;
        lhs_z.array() *= m_z_reg_inv.array();
    }

    // z = alpha * P * x
    void eval_P_x(const Data<T, I>&, const T& alpha, const Vec<T>& x, Vec<T>& z) override
    {
        BlockVec& block_x = work_x_block_1;
        BlockVec& block_z = work_x_block_2;

        block_x.assign(x);

#ifdef PIQP_HAS_OPENMP
#pragma omp parallel
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::eval_P_x:parallel");

            // block_z = alpha * P * block_x, P is symmetric and only the lower triangular part of P is accessed
            block_symv_l_parallel(alpha, P, block_x, block_z);

        } // end of parallel region
#else

        // block_z = alpha * P * block_x, P is symmetric and only the lower triangular part of P is accessed
        block_symv_l(alpha, P, block_x, block_z);

#endif

        block_z.load(z);
    }

    // zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
    void eval_A_xn_and_AT_xt(const Data<T, I>&, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) override
    {
        BlockVec& block_xn = work_x_block_1;
        BlockVec& block_xt = work_y_block_1;
        BlockVec& block_zn = work_y_block_2;
        BlockVec& block_zt = work_x_block_2;

        block_xn.assign(xn);
        block_xt.assign(xt, AT.perm_inv);

#ifdef PIQP_HAS_OPENMP
#pragma omp parallel
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::eval_A_xn_and_AT_xt:parallel");

            // block_zt = alpha_t * AT * block_xt
            block_t_gemv_n(alpha_t, AT, block_xt, 0.0, block_zt, block_zt);
            // block_zn = alpha_n * A * block_xn
            block_t_gemv_t(alpha_n, AT, block_xn, 0.0, block_zn, block_zn);

        } // end of parallel region
#else

        // block_zt = alpha_t * AT * block_xt
        // block_zn = alpha_n * A * block_xn
        block_t_gemv_nt(alpha_t, alpha_n, AT, block_xt, block_xn, 0.0, 0.0, block_zt, block_zn, block_zt, block_zn);

#endif

        block_zn.load(zn, AT.perm_inv);
        block_zt.load(zt);
    }

    // zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
    void eval_G_xn_and_GT_xt(const Data<T, I>&, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) override
    {
        BlockVec& block_xn = work_x_block_1;
        BlockVec& block_xt = work_z_block_1;
        BlockVec& block_zn = work_z_block_2;
        BlockVec& block_zt = work_x_block_2;

        block_xn.assign(xn);
        block_xt.assign(xt, GT.perm_inv);

#ifdef PIQP_HAS_OPENMP
#pragma omp parallel
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::eval_G_xn_and_GT_xt:parallel");

            // block_zt = alpha_t * GT * block_xt
            block_t_gemv_n(alpha_t, GT, block_xt, 0.0, block_zt, block_zt);
            // block_zn = alpha_n * G * block_xn
            block_t_gemv_t(alpha_n, GT, block_xn, 0.0, block_zn, block_zn);

        } // end of parallel region
#else

        // block_zt = alpha_t * GT * block_xt
        // block_zn = alpha_n * G * block_xn
        block_t_gemv_nt(alpha_t, alpha_n, GT, block_xt, block_xn, 0.0, 0.0, block_zt, block_zn, block_zt, block_zn);

#endif

        block_zn.load(zn, GT.perm_inv);
        block_zt.load(zt);
    }

    void print_info() override
    {
        std::size_t N = block_info.size();
        piqp_print("block sizes:");
        for (std::size_t i = 0; i < N - 1; i++) {
            piqp_print(" %d,%d", block_info[i].diag_size, block_info[i].off_diag_size);
        }
        piqp_print("\narrow width: %d\n", block_info[N - 1].diag_size);
    }

protected:
    // A * B, A \in R^{m x k}, B \in R^{k x m}
    usize flops_gemm(usize m, usize n, usize k)
    {
        return 2 * m * n * k;
    }

    // A^{-1} * B, A \in R^{m x m} triangular, B \in R^{m x n}
    usize flops_trsm(usize m, usize n)
    {
        return m * m * n;
    }

    // C + A * A^T, C \in R^{n x n} triangular, A \in R^{n x k}
    usize flops_syrk(usize n, usize k)
    {
        return n * n * k;
    }

    // chol(A), C \in R^{n x n}
    usize flops_potrf(size_t n)
    {
        return n * n * n / 3;
    }

    void extract_arrow_structure(const Data<T, I>& data)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::extract_arrow_structure");

        // build condensed KKT structure for analysis
        SparseMat<T, I> P_ltri = data.P_utri.transpose();
        SparseMat<T, I> identity;
        identity.resize(data.n, data.n);
        identity.setIdentity();
        SparseMat<T, I> AtA_sp = (data.AT * data.AT.transpose()).template triangularView<Eigen::Lower>();
        SparseMat<T, I> GtG_sp = (data.GT * data.GT.transpose()).template triangularView<Eigen::Lower>();
        SparseMat<T, I> C = P_ltri + identity + AtA_sp + GtG_sp;

        struct block_structure_info {
            I prev_diag_block_size = 0;
            I diag_block_start = 0;
            I diag_block_size = 0;
            I off_diag_block_size = 0;
            I arrow_width = 0;
        };
        block_structure_info current_block_info;

        // flop count corresponding to tri-diagonal factorization
        usize flops_tridiag = 0;
        // flop count corresponding to arrow width factorization without syrk operations
        usize flops_arrow_normalized_no_syrk = 0;
        // flop count corresponding to arrow width factorization only with syrk operations
        usize flops_arrow_normalized_syrk = 0;

        // Since C is lower triangular and stored in column major,
        // the iterations corresponds to iterating over the rows
        // of the transpose, i.e., the upper triangular matrix.
        Eigen::Index n = C.outerSize();
        for (Eigen::Index i = 0; i < n; i++)
        {
            auto get_next_block_structure = [&](I row, const block_structure_info& current_info)
            {
                block_structure_info next_info = current_info;

                for (typename SparseMat<T, I>::InnerIterator C_utri_row_it(C, row); C_utri_row_it; ++C_utri_row_it)
                {
                    I col = C_utri_row_it.index();

                    if (col >= next_info.diag_block_start && col + next_info.arrow_width < n) {
                        // current block size
                        I current_block_size = next_info.diag_block_size + next_info.off_diag_block_size;
                        // calculate new block size
                        I new_block_size = (std::max)(col - next_info.diag_block_start + 1, current_block_size);
                        // diag block is limited to current block height
                        I max_diag_block_size = row - next_info.diag_block_start + 1;
                        // min diagonal block must be at least halve the block size
                        I new_min_diag_block_size = (std::max)(next_info.diag_block_size, (new_block_size + 1) / 2); // round up
                        // split block size to have the highest diag block
                        I new_diag_block_size = (std::max)(new_min_diag_block_size, max_diag_block_size);
                        I new_off_diag_block_size = new_block_size - new_diag_block_size;
                        // potential new arrow width
                        I remaining_width = I(n) - next_info.diag_block_start - next_info.diag_block_size - next_info.off_diag_block_size;
                        I new_arrow_width = (std::min)((std::max)(next_info.arrow_width, I(n) - col), remaining_width);

                        usize flops_tridiag_new = flops_tridiag;
                        // L_i = chol(D_i - C_{i-1} * C_{i-1}^T
                        flops_tridiag_new += flops_syrk(static_cast<size_t>(new_diag_block_size), static_cast<size_t>(next_info.prev_diag_block_size));
                        flops_tridiag_new += flops_potrf(static_cast<size_t>(new_diag_block_size));
                        // C_i = B_i * L_i^{-T}
                        flops_tridiag_new += flops_trsm(static_cast<size_t>(new_diag_block_size), static_cast<size_t>(new_off_diag_block_size));

                        // the dense kernels will use blocks of min width 4
                        // thus, we calculate the expected flops with arrows which are multiples of 4
                        I arrow_width_flops = ((next_info.arrow_width + 3) / 4) * 4;
                        I new_arrow_width_flops = ((new_arrow_width + 3) / 4) * 4;
                        // calculate current arrow flop count from normalized counts
                        usize flops_arrow = static_cast<usize>(arrow_width_flops) * flops_arrow_normalized_no_syrk
                                            + static_cast<usize>(arrow_width_flops) * static_cast<usize>(arrow_width_flops) * flops_arrow_normalized_syrk
                                            + flops_potrf(static_cast<usize>(arrow_width_flops));
                        // calculate new arrow flop count from normalized counts
                        usize flops_arrow_new = static_cast<usize>(new_arrow_width_flops) * flops_arrow_normalized_no_syrk
                                                + static_cast<usize>(new_arrow_width_flops) * static_cast<usize>(new_arrow_width_flops) * flops_arrow_normalized_syrk;;
                        // F_i = (E_i - F_{i-1} * C_{i-1}^T) * L_i^{-T}
                        flops_arrow_new += flops_gemm(static_cast<usize>(new_arrow_width_flops), static_cast<usize>(next_info.prev_diag_block_size), static_cast<usize>(new_diag_block_size));
                        flops_arrow_new += flops_trsm(static_cast<usize>(new_diag_block_size), static_cast<usize>(new_arrow_width_flops));
                        // L_N = chol(D_N - sum F_i * F_i^T)
                        flops_arrow_new += flops_syrk(static_cast<usize>(new_arrow_width_flops), static_cast<usize>(new_diag_block_size));
                        flops_arrow_new += flops_potrf(static_cast<usize>(new_arrow_width_flops));

                        // decide if we accept new diagonal block size or assign it to the arrow
                        if (flops_tridiag_new - flops_tridiag <= flops_arrow_new - flops_arrow) {
                            next_info.diag_block_size = new_diag_block_size;
                            next_info.off_diag_block_size = new_off_diag_block_size;
                        } else {
                            next_info.arrow_width = new_arrow_width;
                        }
//                        std::cout << row << " " << col << " " << next_info.diag_block_start << " " << next_info.diag_block_size << " " << next_info.off_diag_block_size << " " << next_info.arrow_width << " " << (flops_tridiag_new - flops_tridiag) << " " << (flops_arrow_new - flops_arrow) << std::endl;
                    }
                }

                return next_info;
            };

            current_block_info = get_next_block_structure(I(i), current_block_info);

//            std::cout << i << " " << current_block_info.diag_block_start << " " << current_block_info.diag_block_size << " " << current_block_info.off_diag_block_size << " " << current_block_info.arrow_width << std::endl;

            if (i + 1 >= current_block_info.diag_block_start + current_block_info.diag_block_size) {

                bool hit_optimal_ratio = current_block_info.diag_block_size >= 2 * current_block_info.off_diag_block_size;
                bool at_end = i + 1 >= n - current_block_info.arrow_width;
                auto next_block_grows = [&]() {
                    if (i >= n) return false;
                    block_structure_info next_block_info = get_next_block_structure(I(i) + 1, current_block_info);
                    return next_block_info.diag_block_size + next_block_info.off_diag_block_size > current_block_info.diag_block_size + current_block_info.off_diag_block_size;
                };

                if (hit_optimal_ratio || at_end || next_block_grows()) {
//                    std::cout << "B " << current_block_info.diag_block_start << " " << current_block_info.diag_block_size << " " << current_block_info.off_diag_block_size << " " << current_block_info.arrow_width << std::endl;
                    block_info.push_back({current_block_info.diag_block_start, current_block_info.diag_block_size, current_block_info.off_diag_block_size});

                    // L_i = chol(D_i - C_{i-1} * C_{i-1}^T
                    flops_tridiag += flops_syrk(static_cast<size_t>(current_block_info.diag_block_size), static_cast<size_t>(current_block_info.prev_diag_block_size + 1));
                    flops_tridiag += flops_potrf(static_cast<size_t>(current_block_info.diag_block_size));
                    // C_i = B_i * L_i^{-T}
                    flops_tridiag += flops_trsm(static_cast<size_t>(current_block_info.diag_block_size), static_cast<size_t>(current_block_info.off_diag_block_size));

                    // F_i = (E_i - F_{i-1} * C_{i-1}^T) * L_i^{-T}
                    flops_arrow_normalized_no_syrk += flops_gemm(1, static_cast<usize>(current_block_info.prev_diag_block_size), static_cast<usize>(current_block_info.diag_block_size));
                    flops_arrow_normalized_no_syrk += flops_trsm(static_cast<usize>(current_block_info.diag_block_size), 1);
                    // L_N = chol(D_N - sum F_i * F_i^T)
                    flops_arrow_normalized_syrk += flops_syrk(1, static_cast<usize>(current_block_info.diag_block_size));

                    current_block_info.diag_block_start += current_block_info.diag_block_size;
                    current_block_info.prev_diag_block_size = current_block_info.diag_block_size;
                    current_block_info.diag_block_size = current_block_info.off_diag_block_size;
                    current_block_info.off_diag_block_size = 0;
                }

                // finalize last block before arrow if it exists
                if (at_end && current_block_info.diag_block_size > 0) {
                    block_info.push_back({current_block_info.diag_block_start, current_block_info.diag_block_size, current_block_info.off_diag_block_size});
                    assert(current_block_info.off_diag_block_size == 0);

                    current_block_info.diag_block_start += current_block_info.diag_block_size;
                    current_block_info.prev_diag_block_size = current_block_info.diag_block_size;
                    current_block_info.diag_block_size = current_block_info.off_diag_block_size;
                    current_block_info.off_diag_block_size = 0;
                }

                if (at_end) break;
            }
        }

        // merge blocks which are split in two
        // this doesn't change the flops, but reduces the number of kernel calls
        for (std::size_t i = 0; i < block_info.size() - 1; i++) {
            if (block_info[i].off_diag_size == block_info[i + 1].diag_size && block_info[i + 1].off_diag_size == 0) {
                block_info[i].diag_size += block_info[i].off_diag_size;
                block_info[i].off_diag_size = 0;
                auto iter = block_info.begin();
                std::advance(iter, i + 1);
                block_info.erase(iter);
            }
        }

        // last block corresponds to corner block of arrow
        block_info.push_back({current_block_info.diag_block_start, current_block_info.arrow_width, 0});
        assert(block_info.size() >= 2);

//        // calculate current arrow flop count from normalized counts
//        usize flops_arrow = static_cast<usize>(current_block_info.arrow_width) * flops_arrow_normalized_no_syrk
//                            + static_cast<usize>(current_block_info.arrow_width) * static_cast<usize>(current_block_info.arrow_width) * flops_arrow_normalized_syrk;
//        // L_N = chol(D_N - sum F_i * F_i^T)
//        flops_arrow += flops_potrf(static_cast<usize>(current_block_info.arrow_width));
//
//        std::cout << "flops_tridiag: " <<  flops_tridiag << "  flops_arrow: " << flops_arrow << std::endl;
//
//        std::size_t N = block_info.size();
//        for (std::size_t i = 0; i < N; i++) {
//            std::cout << block_info[i].start << ", " << block_info[i].diag_size << ", " << block_info[i].off_diag_size << ";" << std::endl;
//        }
    }

    void utri_to_kkt(const SparseMat<T, I>& A_utri, BlockKKT& A_kkt)
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;

        A_kkt.D.resize(N);
        A_kkt.B.resize(N - 2);
        A_kkt.E.resize(N - 1);

        std::size_t block_index = 0;
        I block_start = block_info[block_index].start;
        I block_diag_size = block_info[block_index].diag_size;

        // Iterating over a csc symmetric upper triangular matrix corresponds
        // to iterating over the rows of the corresponding transpose (lower triangular)
        Eigen::Index n = A_utri.outerSize();
        for (Eigen::Index i = 0; i < n; i++)
        {
            if (i >= block_start + block_diag_size)
            {
                block_index++;
                block_start = block_info[block_index].start;
                block_diag_size = block_info[block_index].diag_size;
            }

            std::size_t current_arrow_block_index = 0;
            I current_arrow_block_start = block_info[current_arrow_block_index].start;
            I current_arrow_block_width = block_info[current_arrow_block_index].diag_size;

            for (typename SparseMat<T, I>::InnerIterator A_ltri_row_it(A_utri, i); A_ltri_row_it; ++A_ltri_row_it)
            {
                I j = A_ltri_row_it.index();
                T v = A_ltri_row_it.value();
                assert(j <= i && "P is not upper triangular");

                // check if on diagonal
                if (j >= block_start)
                {
                    if (!A_kkt.D[block_index]) {
                        A_kkt.D[block_index] = std::make_unique<BlasfeoMat>(block_diag_size, block_diag_size);
                    }
                    BLASFEO_DMATEL(A_kkt.D[block_index]->ref(), i - block_start, j - block_start) = v;
                }
                // check if on arrow
                else if (i >= n - arrow_width)
                {
                    while (current_arrow_block_start + current_arrow_block_width - 1 < j) {
                        current_arrow_block_index++;
                        current_arrow_block_start = block_info[current_arrow_block_index].start;
                        current_arrow_block_width = block_info[current_arrow_block_index].diag_size;
                    }

                    if (!A_kkt.E[current_arrow_block_index]) {
                        A_kkt.E[current_arrow_block_index] = std::make_unique<BlasfeoMat>(arrow_width, current_arrow_block_width);
                    }
                    BLASFEO_DMATEL(A_kkt.E[current_arrow_block_index]->ref(), i - block_start, j - current_arrow_block_start) = v;
                }
                // we have to be on off diagonal
                else
                {
                    I last_block_start = block_info[block_index - 1].start;
                    I last_block_diag_size = block_info[block_index - 1].diag_size;
                    I last_block_off_diag_size = block_info[block_index - 1].off_diag_size;
                    assert(j >= last_block_start && "indexes in no valid block");
                    if (!A_kkt.B[block_index - 1]) {
                        A_kkt.B[block_index - 1] = std::make_unique<BlasfeoMat>(last_block_off_diag_size, last_block_diag_size);
                    }
                    BLASFEO_DMATEL(A_kkt.B[block_index - 1]->ref(), i - block_start, j - last_block_start) = v;
                }
            }
        }
    }

    template<bool init>
    void transpose_to_block_mat(const SparseMat<T, I>& sAT, bool store_transpose, BlockMat<I>& A_block)
    {
        Vec<I>& block_fill_counter = A_block.tmp;

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;

        if (init) {
            A_block.perm.resize(sAT.cols());
            A_block.perm_inv.resize(sAT.cols());
            A_block.D.resize(N - 1);
            A_block.B.resize(N - 2);
            A_block.E.resize(N - 1);

            // keep track on the current fill status of each block
            A_block.block_row_sizes.resize(Eigen::Index(N - 1));
            A_block.block_row_sizes.setZero();
            block_fill_counter.resize(Eigen::Index(N - 1));
        }

        // Iterating over a csc transposed matrix corresponds
        // to iterating over the rows of the non-transposed matrix.

        // First pass is to determine the number of rows per block
        Eigen::Index rows = sAT.outerSize(); // rows here corresponds to the non-transposed matrix
        Eigen::Index cols = sAT.innerSize();
        if (init) {
            for (Eigen::Index i = 0; i < rows; i++)
            {
                typename SparseMat<T, I>::InnerIterator A_row_it(sAT, i);
                if (A_row_it)
                {
                    I j = A_row_it.index();
                    std::size_t block_index = 0;
                    // find the corresponding block
                    while (block_info[block_index].start + block_info[block_index].diag_size <= j && block_index + 1 < N - 1) { block_index++; }
                    A_block.block_row_sizes(Eigen::Index(block_index))++;
                }
            }
        }

        Vec<I> block_row_acc;
        if (init) {
            block_row_acc.resize(A_block.block_row_sizes.rows() + 1);
            block_row_acc[0] = 0;
            for (Eigen::Index i = 0; i < Eigen::Index(N - 1); i++) {
                block_row_acc[i + 1] = block_row_acc[i] + A_block.block_row_sizes[i];
            }
        }

        // keep track on where we are in block
        block_fill_counter.setZero();
        I no_block_counter = 0;

        // In the second pass, we allocate and fill the block matrix
        for (Eigen::Index i = 0; i < rows; i++)
        {
            typename SparseMat<T, I>::InnerIterator A_row_it(sAT, i);

            std::size_t block_index = 0;
            I block_i = 0;
            if (A_row_it)
            {
                I j = A_row_it.index();
                // find the corresponding block
                while (block_info[block_index].start + block_info[block_index].diag_size <= j && block_index + 1 < N - 1) { block_index++; }
                block_i = block_fill_counter(Eigen::Index(block_index))++;
                if (init) {
                    A_block.perm[i] = block_row_acc[Eigen::Index(block_index)] + block_i;
                }
            } else {
                if (init) {
                    // empty rows get put in the back to ensure correct permutation
                    A_block.perm[i] = block_row_acc(A_block.block_row_sizes.rows()) + no_block_counter++;
                }
            }

            I block_start = block_info[block_index].start;
            I block_diag_size = block_info[block_index].diag_size;

            for (; A_row_it; ++A_row_it)
            {
                I j = A_row_it.index();
                T v = A_row_it.value();

                // arrow
                if (j + arrow_width >= cols)
                {
                    if (init && !A_block.E[block_index]) {
                        if (store_transpose) {
                            A_block.E[block_index] = std::make_unique<BlasfeoMat>(arrow_width, A_block.block_row_sizes[Eigen::Index(block_index)]);
                        } else {
                            A_block.E[block_index] = std::make_unique<BlasfeoMat>(A_block.block_row_sizes[Eigen::Index(block_index)], arrow_width);
                        }
                    }
                    if (store_transpose) {
                        BLASFEO_DMATEL(A_block.E[block_index]->ref(), j + arrow_width - cols, block_i) = v;
                    } else {
                        BLASFEO_DMATEL(A_block.E[block_index]->ref(), block_i, j + arrow_width - cols) = v;
                    }
                }
                // first block
                else if (j < block_start + block_diag_size)
                {
                    if (init && !A_block.D[block_index]) {
                        if (store_transpose) {
                            A_block.D[block_index] = std::make_unique<BlasfeoMat>(block_diag_size, A_block.block_row_sizes[Eigen::Index(block_index)]);
                        } else {
                            A_block.D[block_index] = std::make_unique<BlasfeoMat>(A_block.block_row_sizes[Eigen::Index(block_index)], block_diag_size);
                        }
                    }
                    if (store_transpose) {
                        BLASFEO_DMATEL(A_block.D[block_index]->ref(), j - block_start, block_i) = v;
                    } else {
                        BLASFEO_DMATEL(A_block.D[block_index]->ref(), block_i, j - block_start) = v;
                    }
                }
                // second block
                else
                {
                    I block_off_diag_size = block_info[block_index].off_diag_size;
                    assert(j < block_start + block_diag_size + block_off_diag_size && "indexes in no valid block");

                    if (init && !A_block.B[block_index]) {
                        if (store_transpose) {
                            A_block.B[block_index] = std::make_unique<BlasfeoMat>(block_off_diag_size, A_block.block_row_sizes[Eigen::Index(block_index)]);
                        } else {
                            A_block.B[block_index] = std::make_unique<BlasfeoMat>(A_block.block_row_sizes[Eigen::Index(block_index)], block_off_diag_size);
                        }
                    }
                    if (store_transpose) {
                        BLASFEO_DMATEL(A_block.B[block_index]->ref(), j - block_start - block_diag_size, block_i) = v;
                    } else {
                        BLASFEO_DMATEL(A_block.B[block_index]->ref(), block_i, j - block_start - block_diag_size) = v;
                    }
                }
            }
        }

        if (init) {
            for (Eigen::Index i = 0; i < sAT.cols(); i++)
            {
                A_block.perm_inv[A_block.perm[i]] = I(i);
            }
        }
    }

    void block_syrk_ln_alloc(BlockMat<I>& sA, BlockMat<I>& sB, BlockKKT& sD)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_syrk_ln_alloc");
        block_syrk_ln<true>(sA, sB, sD);
    }

    void block_syrk_ln_calc(BlockMat<I>& sA, BlockMat<I>& sB, BlockKKT& sD)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_syrk_ln_calc");
        block_syrk_ln<false>(sA, sB, sD);
    }

    // D += A * B^T
    template<bool allocate>
    void block_syrk_ln(BlockMat<I>& sA, BlockMat<I>& sB, BlockKKT& sD)
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;

        if (allocate) {
#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#pragma omp single
            {
#endif
            sD.D.resize(N);
            sD.B.resize(N - 2);
            sD.E.resize(N - 1);
#ifdef PIQP_HAS_OPENMP
            } // end of single region
#endif
        }

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

    void init_kkt_fac()
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::init_kkt_fac");
        construct_kkt_fac<true>(work_x);
    }

    void populate_kkt_fac(const Vec<T>& x_reg)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::populate_kkt_fac");
        construct_kkt_fac<false>(x_reg);
    }

    template<bool allocate>
    void construct_kkt_fac(const Vec<T>& x_reg)
    {
        BlockVec& x_reg_block = work_x_block_1;

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;
        T delta_inv = 1.0 / m_delta;

#ifdef PIQP_HAS_OPENMP
#pragma omp barrier
#pragma omp single
        {
#endif
        kkt_fac.D.resize(N);
        kkt_fac.B.resize(N - 2);
        kkt_fac.E.resize(N - 1);

        if (!allocate)
        {
            x_reg_block.assign(x_reg);
        }
#ifdef PIQP_HAS_OPENMP
        } // end of single region
#endif

        // ----- DIAGONAL -----

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N; i++)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::construct_kkt_fac::diagonal");
            PIQP_TRACY_ZoneValue(i);

            I m = block_info[i].diag_size;

            if (allocate) {
                if (!kkt_fac.D[i]) {
                    kkt_fac.D[i] = std::make_unique<BlasfeoMat>(m, m);
                }
            } else {
                bool mat_set = false;

                if (P.D[i]) {
                    // D_i = P.D_i, lower triangular
                    blasfeo_dtrcp_l(*P.D[i], *kkt_fac.D[i]);
                    mat_set = true;
                }

                if (AtA.D[i]) {
                    if (mat_set) {
                        // D_i += delta^{-1} * AtA.D_i
                        blasfeo_dgead(delta_inv, *AtA.D[i], *kkt_fac.D[i]);
                    } else {
                        // D_i = delta^{-1} * AtA.D_i, lower triangular
                        blasfeo_dtrcpsc_l(delta_inv, *AtA.D[i], *kkt_fac.D[i]);
                        mat_set = true;
                    }
                }

                if (GtG.D[i]) {
                    if (mat_set) {
                        // D_i += GtG.D_i
                        blasfeo_dgead(1.0, *GtG.D[i], *kkt_fac.D[i]);
                    } else {
                        // D_i = GtG.D_i, lower triangular
                        blasfeo_dtrcp_l(*GtG.D[i], *kkt_fac.D[i]);
                        mat_set = true;
                    }
                }

                if (mat_set) {
                    // diag(D_i) += diag
                    blasfeo_ddiaad(1.0, x_reg_block.x[i], *kkt_fac.D[i]);
                } else {
                    // D_i = diag
                    kkt_fac.D[i]->setZero();
                    blasfeo_ddiain(1.0, x_reg_block.x[i], *kkt_fac.D[i]);
                }
            }
        }

        // ----- OFF-DIAGONAL -----

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 2; i++)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::construct_kkt_fac::off_diagonal");
            PIQP_TRACY_ZoneValue(i);

            int m = block_info[i].off_diag_size;
            int n = block_info[i].diag_size;

            bool mat_set = false;

            if (P.B[i]) {
                if (allocate) {
                    if (!kkt_fac.B[i]) {
                        kkt_fac.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    // B_i = P.B_i
                    blasfeo_dgecp(*P.B[i], *kkt_fac.B[i]);
                    mat_set = true;
                }
            }

            if (AtA.B[i]) {
                if (allocate) {
                    if (!kkt_fac.B[i]) {
                        kkt_fac.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    if (mat_set) {
                        // B_i += delta^{-1} * AtA.B_i
                        blasfeo_dgead(delta_inv, *AtA.B[i], *kkt_fac.B[i]);
                    } else {
                        // B_i = delta^{-1} * AtA.B_i
                        blasfeo_dgecpsc(delta_inv, *AtA.B[i], *kkt_fac.B[i]);
                        mat_set = true;
                    }
                }
            }

            if (GtG.B[i]) {
                if (allocate) {
                    if (!kkt_fac.B[i]) {
                        kkt_fac.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    if (mat_set) {
                        // B_i += GtG.B_i
                        blasfeo_dgead(1.0, *GtG.B[i], *kkt_fac.B[i]);
                    } else {
                        // B_i = GtG.B_i
                        blasfeo_dgecp(*GtG.B[i], *kkt_fac.B[i]);
                    }
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
                PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::construct_kkt_fac::arrow");
                PIQP_TRACY_ZoneValue(i);

                int m = arrow_width;
                int n = block_info[i].diag_size;

                bool mat_set = false;

                if (P.E[i]) {
                    if (allocate) {
                        if (!kkt_fac.E[i]) {
                            kkt_fac.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // E_i = P.E_i
                        blasfeo_dgecp(*P.E[i], *kkt_fac.E[i]);
                        mat_set = true;
                    }
                }

                // the terms AtA.E or GtG.E might be smaller,
                // thus we have to zero the whole matrix just in case
                if (!allocate && kkt_fac.E[i] && !mat_set) {
                    kkt_fac.E[i]->setZero();
                }

                if (AtA.E[i]) {
                    if (allocate) {
                        if (!kkt_fac.E[i]) {
                            kkt_fac.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // E_i += delta^{-1} * AtA.E_i
                        blasfeo_dgead(delta_inv, *AtA.E[i], *kkt_fac.E[i]);
                    }
                }

                if (GtG.E[i]) {
                    if (allocate) {
                        if (!kkt_fac.E[i]) {
                            kkt_fac.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // E_i += GtG.E_i
                        blasfeo_dgead(1.0, *GtG.E[i], *kkt_fac.E[i]);
                    }
                }

                // Only the arrow can have more allocated blocks because
                // of the factorization if the previous factors exist.
                if (!mat_set && i > 0 && kkt_fac.E[i - 1] && kkt_fac.B[i - 1]) {
                    if (allocate && !kkt_fac.E[i]) {
                        kkt_fac.E[i] = std::make_unique<BlasfeoMat>(kkt_fac.E[i - 1]->rows(), kkt_fac.D[i]->rows());
                    }
                }
            }
        }
    }

    // sD = sA * diag(sB)
    void block_gemm_nd(BlockMat<I>& sA, BlockVec& sB, BlockMat<I>& sD)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_gemm_nd");

        std::size_t N = block_info.size();

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 1; i++)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_gemm_nd::diagonal");
            PIQP_TRACY_ZoneValue(i);

            if (sA.D[i]) {
                // sD.D = sA.D * diag(sB)
                blasfeo_dgemm_nd(1.0, *sA.D[i], sB.x[i], 0.0, *sD.D[i], *sD.D[i]);
            }

            if (i < N - 2 && sA.B[i]) {
                // sD.B = sA.B * diag(sB)
                blasfeo_dgemm_nd(1.0, *sA.B[i], sB.x[i], 0.0, *sD.B[i], *sD.B[i]);
            }

            if (sA.E[i]) {
                // sD.E = sA.E * diag(sB)
                blasfeo_dgemm_nd(1.0, *sA.E[i], sB.x[i], 0.0, *sD.E[i], *sD.E[i]);
            }
        }
    }

    void factor_kkt()
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

    // z = alpha * sA * x
    void block_symv_l(double alpha, BlockKKT& sA, BlockVec& x, BlockVec& z)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_symv_l");

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;
        for (std::size_t i = 0; i < N; i++)
        {
            if (sA.D[i]) {
                int m = sA.D[i]->rows();
                assert(x.x[i].rows() == m && "size mismatch");
                assert(z.x[i].rows() == m && "size mismatch");
                // z_i = alpha * D_i * x_i, D_i is symmetric and only the lower triangular part of D_i is accessed
                blasfeo_dsymv_l(m, alpha, sA.D[i]->ref(), 0, 0, x.x[i].ref(), 0, 0.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            } else {
                z.x[i].setZero();
            }
        }
        for (std::size_t i = 0; i < N - 2; i++)
        {
            if (sA.B[i]) {
                int m = sA.B[i]->rows();
                int n = sA.B[i]->cols();
                assert(x.x[i].rows() == n && "size mismatch");
                assert(x.x[i+1].rows() >= m && "size mismatch");
                assert(z.x[i+1].rows() >= m && "size mismatch");
                assert(z.x[i].rows() == n && "size mismatch");
                // z_{i+1} += alpha * B_i * x_i
                // z_i += alpha * B_i^T * x_{i+1}
                blasfeo_dgemv_nt(m, n, alpha, alpha, sA.B[i]->ref(), 0, 0, x.x[i].ref(), 0, x.x[i+1].ref(), 0, 1.0, 1.0, z.x[i+1].ref(), 0, z.x[i].ref(), 0, z.x[i+1].ref(), 0, z.x[i].ref(), 0);
            }
        }
        if (arrow_width > 0)
        {
            for (std::size_t i = 0; i < N - 1; i++)
            {
                if (sA.E[i]) {
                    int m = sA.E[i]->rows();
                    int n = sA.E[i]->cols();
                    assert(x.x[i].rows() == n && "size mismatch");
                    assert(z.x[N-1].rows() == m && "size mismatch");
                    assert(x.x[N-1].rows() == m && "size mismatch");
                    assert(z.x[i].rows() == n && "size mismatch");
                    // z_{N-1} += alpha * E_i * x_i
                    // z_i += alpha * E_i^T * x_{N-1}
                    blasfeo_dgemv_nt(m, n, alpha, alpha, sA.E[i]->ref(), 0, 0, x.x[i].ref(), 0, x.x[N-1].ref(), 0, 1.0, 1.0, z.x[N-1].ref(), 0, z.x[i].ref(), 0, z.x[N-1].ref(), 0, z.x[i].ref(), 0);
                }
            }
        }
    }

    // z = alpha * sA * x
    void block_symv_l_parallel(double alpha, BlockKKT& sA, BlockVec& x, BlockVec& z)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_symv_l_parallel");

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 1; i++)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_symv_l_parallel::diagonal_off_diagonal");
            PIQP_TRACY_ZoneValue(i);

            if (sA.D[i]) {
                int m = sA.D[i]->rows();
                assert(x.x[i].rows() == m && "size mismatch");
                assert(z.x[i].rows() == m && "size mismatch");
                // z_i = alpha * D_i * x_i, D_i is symmetric and only the lower triangular part of D_i is accessed
                blasfeo_dsymv_l(m, alpha, sA.D[i]->ref(), 0, 0, x.x[i].ref(), 0, 0.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            } else {
                z.x[i].setZero();
            }

            if (i < N - 2 && sA.B[i]) {
                int m = sA.B[i]->rows();
                int n = sA.B[i]->cols();
                assert(x.x[i+1].rows() >= m && "size mismatch");
                assert(z.x[i].rows() == n && "size mismatch");
                // z_i += alpha * B_i^T * x_{i+1}
                blasfeo_dgemv_t(m, n, alpha, sA.B[i]->ref(), 0, 0, x.x[i+1].ref(), 0, 1.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            }

            if (i > 0 && sA.B[i-1]) {
                int m = sA.B[i-1]->rows();
                int n = sA.B[i-1]->cols();
                assert(x.x[i-1].rows() == n && "size mismatch");
                assert(z.x[i].rows() >= m && "size mismatch");
                // z_i += alpha * B_{i-1} * x_{i-1}
                blasfeo_dgemv_n(m, n, alpha, sA.B[i-1]->ref(), 0, 0, x.x[i-1].ref(), 0, 1.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            }

            if (sA.E[i]) {
                int m = sA.E[i]->rows();
                int n = sA.E[i]->cols();
                assert(x.x[N-1].rows() == m && "size mismatch");
                assert(z.x[i].rows() == n && "size mismatch");
                // z_i += alpha * E_i^T * x_{N-1}
                blasfeo_dgemv_t(m, n, alpha, sA.E[i]->ref(), 0, 0, x.x[N-1].ref(), 0, 1.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            }
        }

#ifdef PIQP_HAS_OPENMP
#pragma omp single nowait
        {
#endif
        if (arrow_width > 0)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_symv_l_parallel::arrow");

            if (sA.D[N-1]) {
                int m = sA.D[N-1]->rows();
                assert(x.x[N-1].rows() == m && "size mismatch");
                assert(z.x[N-1].rows() == m && "size mismatch");
                // z_{N-1} = alpha * D_{N-1} * x_{N-1}, D_{N-1} is symmetric and only the lower triangular part of D_{N-1} is accessed
                blasfeo_dsymv_l(m, alpha, sA.D[N-1]->ref(), 0, 0, x.x[N-1].ref(), 0, 0.0, z.x[N-1].ref(), 0, z.x[N-1].ref(), 0);
            } else {
                z.x[N-1].setZero();
            }

            for (std::size_t i = 0; i < N - 1; i++)
            {
                if (sA.E[i]) {
                    int m = sA.E[i]->rows();
                    int n = sA.E[i]->cols();
                    assert(x.x[i].rows() == n && "size mismatch");
                    assert(z.x[N-1].rows() == m && "size mismatch");
                    assert(x.x[N-1].rows() == m && "size mismatch");
                    assert(z.x[i].rows() == n && "size mismatch");
                    // z_{N-1} += alpha * E_i * x_i
                    blasfeo_dgemv_n(m, n, alpha, sA.E[i]->ref(), 0, 0, x.x[i].ref(), 0, 1.0, z.x[N-1].ref(), 0, z.x[N-1].ref(), 0);
                }
            }
        }
#ifdef PIQP_HAS_OPENMP
        } // end of single region
#endif
    }

    // y = alpha * x
    void block_veccpsc(double alpha, BlockVec& x, BlockVec& y)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_veccpsc");

        assert(x.x.size() == y.x.size() && "size mismatch");

        std::size_t N = x.x.size();
        for (std::size_t i = 0; i < N; i++)
        {
            assert(x.x[i].rows() == y.x[i].rows() && "size mismatch");
            // y = alpha * x
            blasfeo_dveccpsc(x.x[i].rows(), alpha, x.x[i].ref(), 0, y.x[i].ref(), 0);
        }
    }

    // z = beta * y + alpha * A * x
    // here it's assumed that the sparsity of the block matrix
    // is transposed without the blocks individually transposed
    // A = [A_{1,1}                                                             ]
    //     [A_{1,2} A_{2,2}                                                     ]
    //     [        A_{2,3} A_{3,3}                                             ]
    //     [                A_{3,4} A_{4,4}                                     ]
    //     [                          ...                A_{N-2,N-1} A_{N-1,N-1}]
    //     [A_{1,N} A_{2,N} A_{3,N}   ...      A_{N-4,N} A_{N-3,N}   A_{N-1,N}  ]
    void block_t_gemv_n(double alpha, BlockMat<I>& sA, BlockVec& x, double beta, BlockVec& y, BlockVec& z)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_t_gemv_n");

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 1; i++)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_t_gemv_n::diagonal_off_diagonal");
            PIQP_TRACY_ZoneValue(i);

            // z_i = beta * y_i
            assert(z.x[i].rows() == y.x[i].rows() && "size mismatch");
            blasfeo_dveccpsc(z.x[i].rows(), beta, z.x[i].ref(), 0, y.x[i].ref(), 0);

            if (sA.D[i]) {
                int m = sA.D[i]->rows();
                int n = sA.D[i]->cols();
                assert(x.x[i].rows() == n && "size mismatch");
                assert(z.x[i].rows() == m && "size mismatch");
                // z_i += alpha * D_i * x_i
                blasfeo_dgemv_n(m, n, alpha, sA.D[i]->ref(), 0, 0, x.x[i].ref(), 0, 1.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            }

            if (i > 0 && sA.B[i-1]) {
                int m = sA.B[i-1]->rows();
                int n = sA.B[i-1]->cols();
                assert(x.x[i-1].rows() == n && "size mismatch");
                assert(z.x[i].rows() >= m && "size mismatch");
                // z_i += alpha * B_{i-1} * x_{i-1}
                blasfeo_dgemv_n(m, n, alpha, sA.B[i-1]->ref(), 0, 0, x.x[i-1].ref(), 0, 1.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            }
        }

#ifdef PIQP_HAS_OPENMP
        #pragma omp single nowait
        {
#endif
        if (arrow_width > 0)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_t_gemv_n::arrow");

            // z_{N-1} = beta * y_{N-1}
            assert(z.x[N-1].rows() == y.x[N-1].rows() && "size mismatch");
            blasfeo_dveccpsc(z.x[N-1].rows(), beta, z.x[N-1].ref(), 0, y.x[N-1].ref(), 0);

            for (std::size_t i = 0; i < N - 1; i++)
            {
                if (sA.E[i]) {
                    int m = sA.E[i]->rows();
                    int n = sA.E[i]->cols();
                    assert(x.x[i].rows() == n && "size mismatch");
                    assert(z.x[N-1].rows() == m && "size mismatch");
                    // z_{N-1} += alpha * E_i * x_i
                    blasfeo_dgemv_n(m, n, alpha, sA.E[i]->ref(), 0, 0, x.x[i].ref(), 0, 1.0, z.x[N-1].ref(), 0, z.x[N-1].ref(), 0);
                }
            }
        }
#ifdef PIQP_HAS_OPENMP
        } // end of single region
#endif
    }

    // z = beta * y + alpha * A^T * x
    // here it's assumed that the sparsity of the block matrix
    // is transposed without the blocks individually transposed
    // A = [A_{1,1}                                                             ]
    //     [A_{1,2} A_{2,2}                                                     ]
    //     [        A_{2,3} A_{3,3}                                             ]
    //     [                A_{3,4} A_{4,4}                                     ]
    //     [                          ...                A_{N-2,N-1} A_{N-1,N-1}]
    //     [A_{1,N} A_{2,N} A_{3,N}   ...      A_{N-4,N} A_{N-3,N} A_{N-2,N}    ]
    void block_t_gemv_t(double alpha, BlockMat<I>& sA, BlockVec& x, double beta, BlockVec& y, BlockVec& z)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_t_gemv_t");

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 1; i++)
        {
            PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_t_gemv_t::par");
            PIQP_TRACY_ZoneValue(i);

            // z_i = beta * y_i
            assert(z.x[i].rows() == y.x[i].rows() && "size mismatch");
            blasfeo_dveccpsc(z.x[i].rows(), beta, z.x[i].ref(), 0, y.x[i].ref(), 0);

            if (sA.D[i]) {
                int m = sA.D[i]->rows();
                int n = sA.D[i]->cols();
                assert(x.x[i].rows() == m && "size mismatch");
                assert(z.x[i].rows() == n && "size mismatch");
                // z_i += alpha * D_i^T * x_i
                blasfeo_dgemv_t(m, n, alpha, sA.D[i]->ref(), 0, 0, x.x[i].ref(), 0, 1.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            }

            if (i < N - 2 && sA.B[i]) {
                int m = sA.B[i]->rows();
                int n = sA.B[i]->cols();
                assert(x.x[i+1].rows() >= m && "size mismatch");
                assert(z.x[i].rows() == n && "size mismatch");
                // z_i += alpha * B_i^T * x_{i+1}
                blasfeo_dgemv_t(m, n, alpha, sA.B[i]->ref(), 0, 0, x.x[i+1].ref(), 0, 1.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            }

            if (arrow_width > 0 && sA.E[i]) {
                int m = sA.E[i]->rows();
                int n = sA.E[i]->cols();
                assert(x.x[N-1].rows() == m && "size mismatch");
                assert(z.x[i].rows() == n && "size mismatch");
                // z_i += alpha * E_i^T * x_{N-1}
                blasfeo_dgemv_t(m, n, alpha, sA.E[i]->ref(), 0, 0, x.x[N-1].ref(), 0, 1.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            }
        }
    }

    // z_n = beta_n * y_n + alpha_n * A * x_n
    // z_t = beta_t * y_t + alpha_t * A^T * x_t
    // here it's assumed that the sparsity of the block matrix
    // is transposed without the blocks individually transposed
    // A = [A_{1,1}                                                             ]
    //     [A_{1,2} A_{2,2}                                                     ]
    //     [        A_{2,3} A_{3,3}                                             ]
    //     [                A_{3,4} A_{4,4}                                     ]
    //     [                          ...                A_{N-2,N-1} A_{N-1,N-1}]
    //     [A_{1,N} A_{2,N} A_{3,N}   ...      A_{N-4,N} A_{N-3,N} A_{N-2,N}    ]
    void block_t_gemv_nt(double alpha_n, double alpha_t, BlockMat<I>& sA, BlockVec& x_n, BlockVec& x_t,
                         double beta_n, double beta_t, BlockVec& y_n, BlockVec& y_t, BlockVec& z_n, BlockVec& z_t)
    {
        PIQP_TRACY_ZoneScopedN("piqp::MultistageKKT::block_t_gemv_nt");

        // z_n = beta_n * y_n
        block_veccpsc(beta_n, y_n, z_n);
        // z_t = beta_t * y_t
        block_veccpsc(beta_t, y_t, z_t);

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().diag_size;
        for (std::size_t i = 0; i < N - 1; i++)
        {
            if (sA.D[i]) {
                int m = sA.D[i]->rows();
                int n = sA.D[i]->cols();
                assert(x_n.x[i].rows() == n && "size mismatch");
                assert(z_n.x[i].rows() == m && "size mismatch");
                assert(x_t.x[i].rows() == m && "size mismatch");
                assert(z_t.x[i].rows() == n && "size mismatch");
                // z_n_i += alpha_n * D_i * x_n_i
                // z_t_i += alpha_t * D_i^T * x_t_i
                blasfeo_dgemv_nt(m, n, alpha_n, alpha_t, sA.D[i]->ref(), 0, 0, x_n.x[i].ref(), 0, x_t.x[i].ref(), 0, 1.0, 1.0, z_n.x[i].ref(), 0, z_t.x[i].ref(), 0, z_n.x[i].ref(), 0, z_t.x[i].ref(), 0);
            }

            if (i < N - 2 && sA.B[i]) {
                int m = sA.B[i]->rows();
                int n = sA.B[i]->cols();
                assert(x_n.x[i].rows() == n && "size mismatch");
                assert(z_n.x[i+1].rows() >= m && "size mismatch");
                assert(x_t.x[i+1].rows() >= m && "size mismatch");
                assert(z_t.x[i].rows() == n && "size mismatch");
                // z_n_{i+1} += alpha_n * B_i * x_n_i
                // z_t_i += alpha_t * B_i^T * x_t_{i+1}
                blasfeo_dgemv_nt(m, n, alpha_n, alpha_t, sA.B[i]->ref(), 0, 0, x_n.x[i].ref(), 0, x_t.x[i+1].ref(), 0, 1.0, 1.0, z_n.x[i+1].ref(), 0, z_t.x[i].ref(), 0, z_n.x[i+1].ref(), 0, z_t.x[i].ref(), 0);
            }

            if (arrow_width > 0 && sA.E[i]) {
                int m = sA.E[i]->rows();
                int n = sA.E[i]->cols();
                assert(x_n.x[i].rows() == n && "size mismatch");
                assert(z_n.x[N-1].rows() == m && "size mismatch");
                assert(x_t.x[N-1].rows() == m && "size mismatch");
                assert(z_t.x[i].rows() == n && "size mismatch");
                // z_n_{N-1} += alpha_n * E_i * x_n_i
                // z_t_i += alpha_t * E_i^T * x_t_{N-1}
                blasfeo_dgemv_nt(m, n, alpha_n, alpha_t, sA.E[i]->ref(), 0, 0, x_n.x[i].ref(), 0, x_t.x[N-1].ref(), 0, 1.0, 1.0, z_n.x[N-1].ref(), 0, z_t.x[i].ref(), 0, z_n.x[N-1].ref(), 0, z_t.x[i].ref(), 0);
            }
        }
    }

    // solves A * x = b inplace
    void solve_llt_in_place(BlockVec& b_and_x)
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
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/sparse/multistage_kkt.tpp"
#endif

#endif //PIQP_SPARSE_MULTISTAGE_KKT_HPP
