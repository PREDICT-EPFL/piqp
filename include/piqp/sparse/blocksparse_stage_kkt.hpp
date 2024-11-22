// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_HPP
#define PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_HPP

#include <cassert>
#include <memory>
#include "blasfeo.h"

#ifdef PIQP_HAS_OPENMP
#include "omp.h"
#endif

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"
#include "piqp/kkt_system.hpp"
#include "piqp/kkt_fwd.hpp"
#include "piqp/settings.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/utils/blasfeo_mat.hpp"
#include "piqp/utils/blasfeo_vec.hpp"
#include "piqp/sparse/blocksparse/block_info.hpp"
#include "piqp/sparse/blocksparse/block_kkt.hpp"
#include "piqp/sparse/blocksparse/block_mat.hpp"
#include "piqp/sparse/blocksparse/block_vec.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
class BlocksparseStageKKT : public KKTSystem<T>
{
protected:
    static_assert(std::is_same<T, double>::value, "blocksparse_stagewise only supports doubles");

    const Data<T, I>& data;
    const Settings<T>& settings;

    T m_rho;
    T m_delta;

    Vec<T> m_s;
    Vec<T> m_s_lb;
    Vec<T> m_s_ub;
    Vec<T> m_z_inv;
    Vec<T> m_z_lb_inv;
    Vec<T> m_z_ub_inv;

    Vec<T> tmp_x;
    Vec<T> tmp_z;

    std::vector<BlockInfo<I>> block_info;

    BlockKKT P;
    BlockVec P_diag;
    BlockMat<I> AT;
    BlockMat<I> GT;
    BlockVec G_scaling;
    BlockMat<I> GT_scaled;

    BlockKKT AtA;
    BlockKKT GtG;

    BlockKKT kkt_mat;
    BlockKKT kkt_factor;

    BlockVec tmp1_x_block;
    BlockVec tmp2_x_block;
    BlockVec tmp1_y_block;
    BlockVec tmp2_y_block;
    BlockVec tmp1_z_block;
    BlockVec tmp2_z_block;

public:
    BlocksparseStageKKT(const Data<T, I>& data, const Settings<T>& settings) : data(data), settings(settings)
    {
        // init workspace
        m_rho = T(1);
        m_delta = T(1);

        m_s.resize(data.m);
        m_s_lb.resize(data.n);
        m_s_ub.resize(data.n);
        m_z_inv.resize(data.m);
        m_z_lb_inv.resize(data.n);
        m_z_ub_inv.resize(data.n);

        tmp_x.resize(data.n);
        tmp_z.resize(data.m);

        // prepare kkt factorization
        extract_arrow_structure();
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

        block_syrk_ln_alloc(AT, AT, AtA);
        block_syrk_ln_calc(AT, AT, AtA);

        block_syrk_ln_alloc(GT, GT_scaled, GtG);

        init_kkt_mat();

        // We can mostly reuse the structure from the kkt matrix.
        // Only the arrow can have more allocated blocks because
        // of the factorization if the previous factors exist.
        kkt_factor = kkt_mat;
        for (std::size_t i = 1; i < N - 1; i++)
        {
            if (!kkt_factor.E[i] && kkt_factor.E[i - 1] && kkt_factor.B[i - 1]) {
                kkt_factor.E[i] = std::make_unique<BlasfeoMat>(kkt_factor.E[i - 1]->rows(), kkt_factor.B[i - 1]->rows());
            }
        }

        tmp1_x_block = BlockVec(block_info);
        tmp2_x_block = BlockVec(block_info);
        tmp1_y_block = BlockVec(AT.block_row_sizes);
        tmp2_y_block = BlockVec(AT.block_row_sizes);
        tmp1_z_block = BlockVec(GT.block_row_sizes);
        tmp2_z_block = BlockVec(GT.block_row_sizes);
    }

    void update_data(int options)
    {
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
            block_syrk_ln_calc(AT, AT, AtA);
        }

        if (options & KKTUpdateOptions::KKT_UPDATE_G)
        {
            transpose_to_block_mat<false>(data.GT, true, GT);
        }
    }

    bool update_scalings_and_factor(bool iterative_refinement,
                                    const T& rho, const T& delta,
                                    const CVecRef<T>& s, const CVecRef<T>& s_lb, const CVecRef<T>& s_ub,
                                    const CVecRef<T>& z, const CVecRef<T>& z_lb, const CVecRef<T>& z_ub)
    {
        (void) iterative_refinement;
        assert(!iterative_refinement && "iterative refinement not implemented yet");

        m_rho = rho;
        m_delta = delta;
        m_s = s;
        m_s_lb.head(data.n_lb) = s_lb.head(data.n_lb);
        m_s_ub.head(data.n_ub) = s_ub.head(data.n_ub);
        m_z_inv.array() = T(1) / z.array();
        m_z_lb_inv.head(data.n_lb).array() = T(1) / z_lb.head(data.n_lb).array();
        m_z_ub_inv.head(data.n_ub).array() = T(1) / z_ub.head(data.n_ub).array();

        // populate G scaling vector
        Eigen::Index i = 0;
        for (I block_idx = 0; block_idx < GT.block_row_sizes.rows(); block_idx++)
        {
            I block_size = GT.block_row_sizes(block_idx);
            for (I inner_idx = 0; inner_idx < block_size; inner_idx++)
            {
                I perm_idx = GT.perm_inv(i);
                BLASFEO_DVECEL(G_scaling.x[std::size_t(block_idx)].ref(), inner_idx) = T(1) / (m_s(perm_idx) * m_z_inv(perm_idx) + m_delta);
                i++;
            }
        }

        block_gemm_nd(GT, G_scaling, GT_scaled);
        block_syrk_ln_calc(GT, GT_scaled, GtG);
        populate_kkt_mat();
        factor_kkt();

        return true;
    }

    void multiply(const CVecRef<T>& delta_x, const CVecRef<T>& delta_y,
                  const CVecRef<T>& delta_z, const CVecRef<T>& delta_z_lb, const CVecRef<T>& delta_z_ub,
                  const CVecRef<T>& delta_s, const CVecRef<T>& delta_s_lb, const CVecRef<T>& delta_s_ub,
                  VecRef<T> rhs_x, VecRef<T> rhs_y,
                  VecRef<T> rhs_z, VecRef<T> rhs_z_lb, VecRef<T> rhs_z_ub,
                  VecRef<T> rhs_s, VecRef<T> rhs_s_lb, VecRef<T> rhs_s_ub)
    {
        BlockVec& block_delta_x = tmp1_x_block;
        BlockVec& block_delta_y = tmp1_y_block;
        BlockVec& block_delta_z = tmp1_z_block;
        BlockVec& block_rhs_x = tmp2_x_block;
        BlockVec& block_rhs_y = tmp2_y_block;
        BlockVec& block_rhs_z = tmp2_z_block;

        block_delta_x.assign(delta_x);
        block_delta_y.assign(delta_y, AT.perm_inv);
        block_delta_z.assign(delta_z, GT.perm_inv);

        // block_rhs_x = P * block_delta_x, P is symmetric and only the lower triangular part of P is accessed
        block_symv_l(P, block_delta_x, block_rhs_x);
        // block_rhs_x += AT * block_delta_y
        // block_rhs_y = A * block_delta_x
        block_t_gemv_nt(1.0, 1.0, AT, block_delta_y, block_delta_x, 1.0, 0.0, block_rhs_x, block_rhs_y, block_rhs_x, block_rhs_y);
        // block_rhs_x += GT * block_delta_z
        // block_rhs_z = G * block_delta_x
        block_t_gemv_nt(1.0, 1.0, GT, block_delta_z, block_delta_x, 1.0, 0.0, block_rhs_x, block_rhs_z, block_rhs_x, block_rhs_z);

        rhs_x.setZero();
        rhs_y.setZero();
        rhs_z.setZero();
        block_rhs_x.load(rhs_x);
        block_rhs_y.load(rhs_y, AT.perm_inv);
        block_rhs_z.load(rhs_z, GT.perm_inv);

        rhs_x.noalias() += m_rho * delta_x;
        for (isize i = 0; i < data.n_lb; i++)
        {
            rhs_x(data.x_lb_idx(i)) -= data.x_lb_scaling(i) * delta_z_lb(i);
        }
        for (isize i = 0; i < data.n_ub; i++)
        {
            rhs_x(data.x_ub_idx(i)) += data.x_ub_scaling(i) * delta_z_ub(i);
        }

        rhs_y.noalias() -= m_delta * delta_y;

        rhs_z.noalias() -= m_delta * delta_z;
        rhs_z.noalias() += delta_s;

        for (isize i = 0; i < data.n_lb; i++)
        {
            rhs_z_lb(i) = -data.x_lb_scaling(i) * delta_x(data.x_lb_idx(i));
        }
        rhs_z_lb.head(data.n_lb).noalias() -= m_delta * delta_z_lb.head(data.n_lb);
        rhs_z_lb.head(data.n_lb).noalias() += delta_s_lb.head(data.n_lb);

        for (isize i = 0; i < data.n_ub; i++)
        {
            rhs_z_ub(i) = data.x_ub_scaling(i) * delta_x(data.x_ub_idx(i));
        }
        rhs_z_ub.head(data.n_ub).noalias() -= m_delta * delta_z_ub.head(data.n_ub);
        rhs_z_ub.head(data.n_ub).noalias() += delta_s_ub.head(data.n_ub);

        rhs_s.array() = m_s.array() * delta_z.array() + m_z_inv.array().cwiseInverse() * delta_s.array();

        rhs_s_lb.head(data.n_lb).array() = m_s_lb.head(data.n_lb).array() * delta_z_lb.head(data.n_lb).array();
        rhs_s_lb.head(data.n_lb).array() += m_z_lb_inv.head(data.n_lb).array().cwiseInverse() * delta_s_lb.head(data.n_lb).array();

        rhs_s_ub.head(data.n_ub).array() = m_s_ub.head(data.n_ub).array() * delta_z_ub.head(data.n_ub).array();
        rhs_s_ub.head(data.n_ub).array() += m_z_ub_inv.head(data.n_ub).array().cwiseInverse() * delta_s_ub.head(data.n_ub).array();
    }

    void solve(const CVecRef<T>& rhs_x, const CVecRef<T>& rhs_y,
               const CVecRef<T>& rhs_z, const CVecRef<T>& rhs_z_lb, const CVecRef<T>& rhs_z_ub,
               const CVecRef<T>& rhs_s, const CVecRef<T>& rhs_s_lb, const CVecRef<T>& rhs_s_ub,
               VecRef<T> delta_x, VecRef<T> delta_y,
               VecRef<T> delta_z, VecRef<T> delta_z_lb, VecRef<T> delta_z_ub,
               VecRef<T> delta_s, VecRef<T> delta_s_lb, VecRef<T> delta_s_ub)
    {
        Vec<T>& rhs = tmp_x;
        Vec<T>& rhs_z_bar = tmp_z;
        BlockVec& block_rhs = tmp1_x_block;
        BlockVec& block_rhs_y = tmp1_y_block;
        BlockVec& block_rhs_z_bar = tmp1_z_block;

        T delta_inv = T(1) / m_delta;

        rhs_z_bar.array() = rhs_z.array() - m_z_inv.array() * rhs_s.array();
        rhs_z_bar.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);

        rhs = rhs_x;

        for (isize i = 0; i < data.n_lb; i++) {
            rhs(data.x_lb_idx(i)) -= data.x_lb_scaling(i) * (rhs_z_lb(i) - m_z_lb_inv(i) * rhs_s_lb(i))
                                     / (m_s_lb(i) * m_z_lb_inv(i) + m_delta);
        }
        for (isize i = 0; i < data.n_ub; i++) {
            rhs(data.x_ub_idx(i)) += data.x_ub_scaling(i) * (rhs_z_ub(i) - m_z_ub_inv(i) * rhs_s_ub(i))
                                     / (m_s_ub(i) * m_z_ub_inv(i) + m_delta);
        }

        block_rhs.assign(rhs);
        block_rhs_y.assign(rhs_y, AT.perm_inv);
        block_rhs_z_bar.assign(rhs_z_bar, GT.perm_inv);

        // block_rhs += GT * block_rhs_z_bar
        block_t_gemv_n(1.0, GT, block_rhs_z_bar, 1.0, block_rhs, block_rhs);
        // block_rhs += delta_inv * AT * block_rhs_y
        block_t_gemv_n(delta_inv, AT, block_rhs_y, 1.0, block_rhs, block_rhs);

        solve_llt_in_place(block_rhs);

        BlockVec& block_delta_x = block_rhs;
        BlockVec& block_delta_y = tmp1_y_block;
        BlockVec& block_delta_z = tmp1_z_block;

        // block_delta_y = delta_inv * A * block_delta_x
        block_t_gemv_t(delta_inv, AT, block_delta_x, 0.0, block_delta_y, block_delta_y);
        // block_delta_z = G * block_delta_x
        block_t_gemv_t(1.0, GT, block_delta_x, 0.0, block_delta_z, block_delta_z);

        block_delta_x.load(delta_x);
        block_delta_y.load(delta_y, AT.perm_inv);
        block_delta_z.load(delta_z, GT.perm_inv);

        delta_y.noalias() -= delta_inv * rhs_y;

        delta_z.array() *= T(1) / (m_s.array() * m_z_inv.array() + m_delta);
        delta_z.noalias() -= rhs_z_bar;

        for (isize i = 0; i < data.n_lb; i++) {
            delta_z_lb(i) =
                    (-data.x_lb_scaling(i) * delta_x(data.x_lb_idx(i)) - rhs_z_lb(i) + m_z_lb_inv(i) * rhs_s_lb(i))
                    / (m_s_lb(i) * m_z_lb_inv(i) + m_delta);
        }
        for (isize i = 0; i < data.n_ub; i++) {
            delta_z_ub(i) =
                    (data.x_ub_scaling(i) * delta_x(data.x_ub_idx(i)) - rhs_z_ub(i) + m_z_ub_inv(i) * rhs_s_ub(i))
                    / (m_s_ub(i) * m_z_ub_inv(i) + m_delta);
        }

        delta_s.array() = m_z_inv.array() * (rhs_s.array() - m_s.array() * delta_z.array());

        delta_s_lb.head(data.n_lb).array() = m_z_lb_inv.head(data.n_lb).array()
                                             * (rhs_s_lb.head(data.n_lb).array() -
                                                m_s_lb.head(data.n_lb).array() * delta_z_lb.head(data.n_lb).array());

        delta_s_ub.head(data.n_ub).array() = m_z_ub_inv.head(data.n_ub).array()
                                             * (rhs_s_ub.head(data.n_ub).array() -
                                                m_s_ub.head(data.n_ub).array() * delta_z_ub.head(data.n_ub).array());
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

    void extract_arrow_structure()
    {
        // build condensed KKT structure for analysis
        SparseMat<T, I> P_ltri = data.P_utri.transpose();
        SparseMat<T, I> identity;
        identity.resize(data.n, data.n);
        identity.setIdentity();
        SparseMat<T, I> AtA_sp = (data.AT * data.AT.transpose()).template triangularView<Eigen::Lower>();
        SparseMat<T, I> GtG_sp = (data.GT * data.GT.transpose()).template triangularView<Eigen::Lower>();
        SparseMat<T, I> C = P_ltri + identity + AtA_sp + GtG_sp;

        I prev_diag_block_size = 0;
        I current_diag_block_start = 0;
        I current_diag_block_size = 0;
        I current_off_diag_block_size = 0;
        I arrow_width = 0;

        // flop count corresponding to tri-diagonal factorization
        usize flops_tridiag = 0;
        // flop count corresponding to arrow (width = 1) factorization without syrk operations
        usize flops_arrow_normalized_no_syrk = 0;
        // flop count corresponding to arrow (width = 1) factorization only with syrk operations
        usize flops_arrow_normalized_syrk = 0;

        // Since C is lower triangular and stored in column major,
        // the iterations corresponds to iterating over the rows
        // of the transpose, i.e., the upper triangular matrix.
        Eigen::Index n = C.outerSize();
        for (Eigen::Index i = 0; i < n; i++)
        {
            for (typename SparseMat<T, I>::InnerIterator C_utri_row_it(C, i); C_utri_row_it; ++C_utri_row_it)
            {
                I j = C_utri_row_it.index();

                if (j >= current_diag_block_start && j < n - arrow_width) {
                    // calculate new block size
                    I new_block_size = j - current_diag_block_start + 1;
                    // split block size equally into diagonal and off diagonal block
                    I new_diag_block_size = (std::max)(current_diag_block_size, (new_block_size + 1) / 2); // round up
                    I new_off_diag_block_size = (std::max)(current_off_diag_block_size, new_block_size - new_diag_block_size);
                    // potential new arrow width
                    I new_arrow_width = (std::max)(arrow_width, I(n) - j);

                    usize flops_tridiag_new = flops_tridiag;
                    // L_i = chol(D_i - C_{i-1} * C_{i-1}^T
                    flops_tridiag_new += flops_syrk(static_cast<size_t>(new_diag_block_size), static_cast<size_t>(prev_diag_block_size));
                    flops_tridiag_new += flops_potrf(static_cast<size_t>(new_diag_block_size));
                    // C_i = B_i * L_i^{-T}
                    flops_tridiag_new += flops_trsm(static_cast<size_t>(new_diag_block_size), static_cast<size_t>(new_off_diag_block_size));

                    // calculate current arrow flop count from normalized counts
                    usize flops_arrow = static_cast<usize>(arrow_width) * flops_arrow_normalized_no_syrk
                            + static_cast<usize>(arrow_width) * static_cast<usize>(arrow_width) * flops_arrow_normalized_syrk
                            + flops_potrf(static_cast<usize>(arrow_width));
                    // calculate new arrow flop count from normalized counts
                    usize flops_arrow_new = static_cast<usize>(new_arrow_width) * flops_arrow_normalized_no_syrk
                            + static_cast<usize>(new_arrow_width) * static_cast<usize>(new_arrow_width) * flops_arrow_normalized_syrk;;
                    // F_i = (E_i - F_{i-1} * C_{i-1}^T) * L_i^{-T}
                    flops_arrow_new += flops_gemm(static_cast<usize>(new_arrow_width), static_cast<usize>(prev_diag_block_size), static_cast<usize>(new_diag_block_size));
                    flops_arrow_new += flops_trsm(static_cast<usize>(new_diag_block_size), static_cast<usize>(new_arrow_width));
                    // L_N = chol(D_N - sum F_i * F_i^T)
                    flops_arrow_new += flops_syrk(static_cast<usize>(new_arrow_width), static_cast<usize>(new_diag_block_size));
                    flops_arrow_new += flops_potrf(static_cast<usize>(new_arrow_width));

                    // decide if we accept new diagonal block size or assign it to the arrow
                    if (flops_tridiag_new - flops_tridiag <= flops_arrow_new - flops_arrow) {
                        current_diag_block_size = new_diag_block_size;
                        current_off_diag_block_size = new_off_diag_block_size;
                    } else {
                        arrow_width = new_arrow_width;
                    }
//                    std::cout << i << " " << j << " " << current_diag_block_start << " " << current_diag_block_size << " " << current_off_diag_block_size << " " << arrow_width << " " << (flops_tridiag_new - flops_tridiag) << " " << (flops_arrow_new - flops_arrow) << std::endl;
                }
            }
//            std::cout << i << " " << current_diag_block_start << " " << current_diag_block_size << " " << current_off_diag_block_size << " " << arrow_width << std::endl;

            if (i >= n - arrow_width) break;

            if (i - current_diag_block_start + 1 >= current_diag_block_size) {
//                std::cout << "B " << current_diag_block_start << " " << current_diag_block_size << " " << current_off_diag_block_size << " " << arrow_width << std::endl;
                block_info.push_back({current_diag_block_start, current_diag_block_size});

                // L_i = chol(D_i - C_{i-1} * C_{i-1}^T
                flops_tridiag += flops_syrk(static_cast<size_t>(current_diag_block_size), static_cast<size_t>(prev_diag_block_size + 1));
                flops_tridiag += flops_potrf(static_cast<size_t>(current_diag_block_size));
                // C_i = B_i * L_i^{-T}
                flops_tridiag += flops_trsm(static_cast<size_t>(current_diag_block_size), static_cast<size_t>(current_off_diag_block_size));

                // F_i = (E_i - F_{i-1} * C_{i-1}^T) * L_i^{-T}
                flops_arrow_normalized_no_syrk += flops_gemm(1, static_cast<usize>(prev_diag_block_size), static_cast<usize>(current_diag_block_size));
                flops_arrow_normalized_no_syrk += flops_trsm(static_cast<usize>(current_diag_block_size), 1);
                // L_N = chol(D_N - sum F_i * F_i^T)
                flops_arrow_normalized_syrk += flops_syrk(1, static_cast<usize>(current_diag_block_size));

                current_diag_block_start += current_diag_block_size;
                prev_diag_block_size = current_diag_block_size;
                current_diag_block_size = current_off_diag_block_size;
                current_off_diag_block_size = 0;
            }
        }

        // last block corresponds to corner block of arrow
        block_info.push_back({current_diag_block_start, arrow_width});
        assert(block_info.size() >= 2);

//        // calculate current arrow flop count from normalized counts
//        usize flops_arrow = static_cast<usize>(arrow_width) * flops_arrow_normalized_no_syrk
//                            + static_cast<usize>(arrow_width) * static_cast<usize>(arrow_width) * flops_arrow_normalized_syrk;
//        // L_N = chol(D_N - sum F_i * F_i^T)
//        flops_arrow += flops_potrf(static_cast<usize>(arrow_width));
//
//        std::cout << "flops_tridiag: " <<  flops_tridiag << "  flops_arrow: " << flops_arrow << std::endl;
    }

    void utri_to_kkt(const SparseMat<T, I>& A_utri, BlockKKT& A_kkt)
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

        A_kkt.D.resize(N);
        A_kkt.B.resize(N - 2);
        A_kkt.E.resize(N - 1);

        std::size_t block_index = 0;
        I block_start = block_info[block_index].start;
        I block_width = block_info[block_index].width;
        I last_block_width = 0;

        // Iterating over a csc symmetric upper triangular matrix corresponds
        // to iterating over the rows of the corresponding transpose (lower triangular)
        Eigen::Index n = A_utri.outerSize();
        for (Eigen::Index i = 0; i < n; i++)
        {
            if (i >= block_start + block_width)
            {
                last_block_width = block_width;
                block_index++;
                block_start = block_info[block_index].start;
                block_width = block_info[block_index].width;
            }

            std::size_t current_arrow_block_index = 0;
            I current_arrow_block_start = block_info[current_arrow_block_index].start;
            I current_arrow_block_width = block_info[current_arrow_block_index].width;

            for (typename SparseMat<T, I>::InnerIterator A_ltri_row_it(A_utri, i); A_ltri_row_it; ++A_ltri_row_it)
            {
                I j = A_ltri_row_it.index();
                T v = A_ltri_row_it.value();
                assert(j <= i && "P is not upper triangular");

                // check if on diagonal
                if (j >= block_start)
                {
                    if (!A_kkt.D[block_index]) {
                        A_kkt.D[block_index] = std::make_unique<BlasfeoMat>(block_width, block_width);
                    }
                    BLASFEO_DMATEL(A_kkt.D[block_index]->ref(), i - block_start, j - block_start) = v;
                }
                // check if on arrow
                else if (i >= n - arrow_width)
                {
                    while (current_arrow_block_start + current_arrow_block_width - 1 < j) {
                        current_arrow_block_index++;
                        current_arrow_block_start = block_info[current_arrow_block_index].start;
                        current_arrow_block_width = block_info[current_arrow_block_index].width;
                    }

                    if (!A_kkt.E[current_arrow_block_index]) {
                        A_kkt.E[current_arrow_block_index] = std::make_unique<BlasfeoMat>(arrow_width, current_arrow_block_width);
                    }
                    BLASFEO_DMATEL(A_kkt.E[current_arrow_block_index]->ref(), i - block_start, j - current_arrow_block_start) = v;
                }
                // we have to be on off diagonal
                else
                {
                    assert(j + block_width + last_block_width > i && "indexes in no valid block");
                    if (!A_kkt.B[block_index - 1]) {
                        A_kkt.B[block_index - 1] = std::make_unique<BlasfeoMat>(block_width, last_block_width);
                    }
                    BLASFEO_DMATEL(A_kkt.B[block_index - 1]->ref(), i - block_start, j + last_block_width - block_start) = v;
                }
            }
        }
    }

    template<bool init>
    void transpose_to_block_mat(const SparseMat<T, I>& sAT, bool store_transpose, BlockMat<I>& A_block)
    {
        Vec<I>& block_fill_counter = A_block.tmp;

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

        if (init) {
            A_block.perm.resize(sAT.cols());
            A_block.perm_inv.resize(sAT.cols());
            A_block.D.resize(N - 2);
            A_block.B.resize(N - 2);
            A_block.E.resize(N - 2);

            // keep track on the current fill status of each block
            A_block.block_row_sizes.resize(Eigen::Index(N - 2));
            A_block.block_row_sizes.setZero();
            block_fill_counter.resize(Eigen::Index(N - 2));
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
                    while (block_info[block_index].start + block_info[block_index].width < j && block_index + 1 < N - 2) { block_index++; }
                    A_block.block_row_sizes(Eigen::Index(block_index))++;
                }
            }
        }

        Vec<I> block_row_acc;
        if (init) {
            block_row_acc.resize(A_block.block_row_sizes.rows() + 1);
            block_row_acc[0] = 0;
            for (Eigen::Index i = 0; i < Eigen::Index(N - 2); i++) {
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
                while (block_info[block_index].start + block_info[block_index].width < j && block_index + 1 < N - 2) { block_index++; }
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
            I block_width = block_info[block_index].width;

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
                else if (j < block_start + block_width)
                {
                    if (init && !A_block.D[block_index]) {
                        if (store_transpose) {
                            A_block.D[block_index] = std::make_unique<BlasfeoMat>(block_width, A_block.block_row_sizes[Eigen::Index(block_index)]);
                        } else {
                            A_block.D[block_index] = std::make_unique<BlasfeoMat>(A_block.block_row_sizes[Eigen::Index(block_index)], block_width);
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
                    I next_block_width = block_info[block_index + 1].width;
                    assert(j < block_start + block_width + next_block_width && "indexes in no valid block");

                    if (init && !A_block.B[block_index]) {
                        if (store_transpose) {
                            A_block.B[block_index] = std::make_unique<BlasfeoMat>(next_block_width, A_block.block_row_sizes[Eigen::Index(block_index)]);
                        } else {
                            A_block.B[block_index] = std::make_unique<BlasfeoMat>(A_block.block_row_sizes[Eigen::Index(block_index)], next_block_width);
                        }
                    }
                    if (store_transpose) {
                        BLASFEO_DMATEL(A_block.B[block_index]->ref(), j - block_start - block_width, block_i) = v;
                    } else {
                        BLASFEO_DMATEL(A_block.B[block_index]->ref(), block_i, j - block_start - block_width) = v;
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
        block_syrk_ln<true>(sA, sB, sD);
    }

    void block_syrk_ln_calc(BlockMat<I>& sA, BlockMat<I>& sB, BlockKKT& sD)
    {
        block_syrk_ln<false>(sA, sB, sD);
    }

    // D += A * B^T
    template<bool allocate>
    void block_syrk_ln(BlockMat<I>& sA, BlockMat<I>& sB, BlockKKT& sD)
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

        if (allocate) {
            sD.D.resize(N);
            sD.B.resize(N - 2);
            sD.E.resize(N - 1);
        }

#ifdef PIQP_HAS_OPENMP
        #pragma omp parallel
        {
#endif

        // ----- DIAGONAL -----

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 1; i++)
        {
            // D_{i,i} = 0
            if (!allocate && sD.D[i]) {
                sD.D[i]->setZero();
            }

            if (i < N - 2 && sA.D[i] && sB.D[i]) {
                int m = sA.D[i]->rows();
                int k = sA.D[i]->cols();
                assert(sB.D[i]->rows() == m && sB.D[i]->cols() == k && "size mismatch");
                if (allocate) {
                    if (!sD.D[i]) {
                        sD.D[i] = std::make_unique<BlasfeoMat>(m, m);
                    }
                } else {
                    // D_{i,i} += lower triangular of A_{i,i} * B_{i,i}^T
                    blasfeo_dsyrk_ln(m, k, 1.0, sA.D[i]->ref(), 0, 0, sB.D[i]->ref(), 0, 0, 1.0, sD.D[i]->ref(), 0, 0, sD.D[i]->ref(), 0, 0);
                }
            }

            if (i > 0 && sA.B[i-1] && sB.B[i-1]) {
                int m = sA.B[i-1]->rows();
                int k = sA.B[i-1]->cols();
                assert(sB.B[i-1]->rows() == m && sB.B[i-1]->cols() == k && "size mismatch");
                if (allocate) {
                    if (!sD.D[i]) {
                        sD.D[i] = std::make_unique<BlasfeoMat>(m, m);
                    }
                } else {
                    // D_{i,i} += lower triangular of A_{i-1,i} * B_{i-1,i}^T
                    blasfeo_dsyrk_ln(m, k, 1.0, sA.B[i-1]->ref(), 0, 0, sB.B[i-1]->ref(), 0, 0, 1.0, sD.D[i]->ref(), 0, 0, sD.D[i]->ref(), 0, 0);
                }
            }
        }

#ifdef PIQP_HAS_OPENMP
        #pragma omp single nowait
        {
#endif
        if (arrow_width > 0)
        {
            // D_{N,N} = 0
            if (!allocate && sD.D[N-1]) {
                sD.D[N-1]->setZero();
            }

            for (std::size_t i = 0; i < N - 2; i++)
            {
                if (sA.E[i] && sB.E[i]) {
                    int m = sA.E[i]->rows();
                    int k = sA.E[i]->cols();
                    assert(sB.E[i]->rows() == m && sB.E[i]->cols() == k && "size mismatch");
                    if (allocate) {
                        if (!sD.D[N-1]) {
                            sD.D[N-1] = std::make_unique<BlasfeoMat>(m, m);
                        }
                    } else {
                        // D_{N,N} += lower triangular of A_{i,N} * B_{i,N}^T
                        blasfeo_dsyrk_ln(m, k, 1.0, sA.E[i]->ref(), 0, 0, sB.E[i]->ref(), 0, 0, 1.0, sD.D[N-1]->ref(), 0, 0, sD.D[N-1]->ref(), 0, 0);
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
            if (sA.B[i] && sB.D[i]) {
                int m = sA.B[i]->rows();
                int n = sB.D[i]->rows();
                int k = sA.B[i]->cols();
                assert(sB.D[i]->cols() == k && "size mismatch");
                if (allocate) {
                    if (!sD.B[i]) {
                        sD.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    // D_{i+1,i} = A_{i,i+1} * B_{i,i}^T
                    blasfeo_dgemm_nt(m, n, k, 1.0, sA.B[i]->ref(), 0, 0, sB.D[i]->ref(), 0, 0, 0.0, sD.B[i]->ref(), 0, 0, sD.B[i]->ref(), 0, 0);
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
                // D_{N,i} = 0
                if (!allocate && sD.E[i]) {
                    sD.E[i]->setZero();
                }

                if (i < N - 2 && sA.E[i] && sB.D[i]) {
                    int m = sA.E[i]->rows();
                    int n = sB.D[i]->rows();
                    int k = sA.E[i]->cols();
                    assert(sB.D[i]->cols() == k && "size mismatch");
                    if (allocate) {
                        if (!sD.E[i]) {
                            sD.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // D_{N,i} += A_{i,N} * B_{i,i}^T
                        blasfeo_dgemm_nt(m, n, k, 1.0, sA.E[i]->ref(), 0, 0, sB.D[i]->ref(), 0, 0, 1.0, sD.E[i]->ref(), 0, 0, sD.E[i]->ref(), 0, 0);
                    }
                }

                if (i > 0 && sA.E[i-1] && sB.B[i-1]) {
                    int m = sA.E[i-1]->rows();
                    int n = sB.B[i-1]->rows();
                    int k = sA.E[i-1]->cols();
                    assert(sB.B[i-1]->cols() == k && "size mismatch");
                    if (allocate) {
                        if (!sD.E[i]) {
                            sD.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // D_{N,i} += A_{i-1,N} * B_{i-1,i}^T
                        blasfeo_dgemm_nt(m, n, k, 1.0, sA.E[i-1]->ref(), 0, 0, sB.B[i-1]->ref(), 0, 0, 1.0, sD.E[i]->ref(), 0, 0, sD.E[i]->ref(), 0, 0);
                    }
                }
            }
        }
#ifdef PIQP_HAS_OPENMP
        } // end of parallel region
#endif
    }

    void init_kkt_mat()
    {
        construct_kkt_mat<true>();
    }

    void populate_kkt_mat()
    {
        construct_kkt_mat<false>();
    }

    template<bool allocate>
    void construct_kkt_mat()
    {
        Vec<T>& diag = tmp_x;
        BlockVec& diag_block = tmp1_x_block;

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;
        T delta_inv = 1.0 / m_delta;

        kkt_mat.D.resize(N);
        kkt_mat.B.resize(N - 2);
        kkt_mat.E.resize(N - 1);

        if (!allocate)
        {
            diag.setConstant(m_rho);

            for (isize i = 0; i < data.n_lb; i++)
            {
                diag(data.x_lb_idx(i)) += data.x_lb_scaling(i) * data.x_lb_scaling(i) / (m_z_lb_inv(i) * m_s_lb(i) + m_delta);
            }

            for (isize i = 0; i < data.n_ub; i++)
            {
                diag(data.x_ub_idx(i)) += data.x_ub_scaling(i) * data.x_ub_scaling(i) / (m_z_ub_inv(i) * m_s_ub(i) + m_delta);
            }

            diag_block.assign(diag);
        }

#ifdef PIQP_HAS_OPENMP
        #pragma omp parallel
        {
#endif

        // ----- DIAGONAL -----

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N; i++)
        {
            I m = block_info[i].width;

            if (allocate) {
                if (!kkt_mat.D[i]) {
                    kkt_mat.D[i] = std::make_unique<BlasfeoMat>(m, m);
                }
            } else {
                bool mat_set = false;

                if (P.D[i]) {
                    assert(P.D[i]->rows() == m && P.D[i]->cols() == m && "size mismatch");
                    // D_i = P.D_i, lower triangular
                    blasfeo_dtrcp_l(m, P.D[i]->ref(), 0, 0, kkt_mat.D[i]->ref(), 0, 0);
                    mat_set = true;
                }

                if (AtA.D[i]) {
                    assert(AtA.D[i]->rows() == m && AtA.D[i]->cols() == m && "size mismatch");
                    if (mat_set) {
                        // D_i += delta^{-1} * AtA.D_i
                        blasfeo_dgead(m, m, delta_inv, AtA.D[i]->ref(), 0, 0, kkt_mat.D[i]->ref(), 0, 0);
                    } else {
                        // D_i = delta^{-1} * AtA.D_i, lower triangular
#ifdef TARGET_X64_INTEL_SKYLAKE_X
                        // blasfeo_dtrcpsc_l not implemented on Skylake yet
                        // and reference implementation not exported ...
                        kkt_mat.D[i]->setZero();
                        blasfeo_dgead(m, m, delta_inv, AtA.D[i]->ref(), 0, 0, kkt_mat.D[i]->ref(), 0, 0);
#else
                        blasfeo_dtrcpsc_l(m, delta_inv, AtA.D[i]->ref(), 0, 0, kkt_mat.D[i]->ref(), 0, 0);
#endif
                        mat_set = true;
                    }
                }

                if (GtG.D[i]) {
                    assert(GtG.D[i]->rows() == m && GtG.D[i]->cols() == m && "size mismatch");
                    if (mat_set) {
                        // D_i += GtG.D_i
                        blasfeo_dgead(m, m, 1.0, GtG.D[i]->ref(), 0, 0, kkt_mat.D[i]->ref(), 0, 0);
                    } else {
                        // D_i = GtG.D_i, lower triangular
                        blasfeo_dtrcp_l(m, GtG.D[i]->ref(), 0, 0, kkt_mat.D[i]->ref(), 0, 0);
                        mat_set = true;
                    }
                }

                if (mat_set) {
                    // diag(D_i) += diag
                    blasfeo_ddiaad(m, 1.0, diag_block.x[i].ref(), 0, kkt_mat.D[i]->ref(), 0, 0);
                } else {
                    // diag(D_i) = diag
                    blasfeo_ddiain(m, 1.0, diag_block.x[i].ref(), 0, kkt_mat.D[i]->ref(), 0, 0);
                }
            }
        }

        // ----- OFF-DIAGONAL -----

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 2; i++)
        {
            int m = block_info[i + 1].width;
            int n = block_info[i].width;

            bool mat_set = false;

            if (P.B[i]) {
                assert(P.B[i]->rows() == m && P.B[i]->cols() == n && "size mismatch");
                if (allocate) {
                    if (!kkt_mat.B[i]) {
                        kkt_mat.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    // B_i = P.B_i
                    blasfeo_dgecp(m, n, P.B[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
                    mat_set = true;
                }
            }

            if (AtA.B[i]) {
                assert(AtA.B[i]->rows() == m && AtA.B[i]->cols() == n && "size mismatch");
                if (allocate) {
                    if (!kkt_mat.B[i]) {
                        kkt_mat.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    if (mat_set) {
                        // B_i += delta^{-1} * AtA.B_i
                        blasfeo_dgead(m, n, delta_inv, AtA.B[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
                    } else {
                        // B_i = delta^{-1} * AtA.B_i
                        blasfeo_dgecpsc(m, n, delta_inv, AtA.B[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
                        mat_set = true;
                    }
                }
            }

            if (GtG.B[i]) {
                assert(GtG.B[i]->rows() == m && GtG.B[i]->cols() == n && "size mismatch");
                if (allocate) {
                    if (!kkt_mat.B[i]) {
                        kkt_mat.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    if (mat_set) {
                        // B_i += GtG.B_i
                        blasfeo_dgead(m, n, 1.0, GtG.B[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
                    } else {
                        // B_i = GtG.B_i
                        blasfeo_dgecp(m, n, GtG.B[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
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
                int m = arrow_width;
                int n = block_info[i].width;

                bool mat_set = false;

                if (P.E[i]) {
                    assert(P.E[i]->rows() == m && P.E[i]->cols() == n && "size mismatch");
                    if (allocate) {
                        if (!kkt_mat.E[i]) {
                            kkt_mat.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // E_i = P.E_i
                        blasfeo_dgecp(m, n, P.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                        mat_set = true;
                    }
                }

                if (AtA.E[i]) {
                    assert(AtA.E[i]->rows() == m && AtA.E[i]->cols() == n && "size mismatch");
                    if (allocate) {
                        if (!kkt_mat.E[i]) {
                            kkt_mat.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        if (mat_set) {
                            // E_i += delta^{-1} * AtA.E_i
                            blasfeo_dgead(m, n, delta_inv, AtA.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                        } else {
                            // E_i = delta^{-1} * AtA.E_i
                            blasfeo_dgecpsc(m, n, delta_inv, AtA.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                            mat_set = true;
                        }
                    }
                }

                if (GtG.E[i]) {
                    assert(GtG.E[i]->rows() == m && GtG.E[i]->cols() == n && "size mismatch");
                    if (allocate) {
                        if (!kkt_mat.E[i]) {
                            kkt_mat.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        if (mat_set) {
                            // E_i += GtG.E_i
                            blasfeo_dgead(m, n, 1.0, GtG.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                        } else {
                            // E_i = GtG.E_i
                            blasfeo_dgecp(m, n, GtG.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                        }
                    }
                }
            }
        }

#ifdef PIQP_HAS_OPENMP
        } // end of parallel region
#endif
    }

    // sD = sA * diag(sB)
    void block_gemm_nd(BlockMat<I>& sA, BlockVec& sB, BlockMat<I>& sD)
    {
        std::size_t N = block_info.size();

#ifdef PIQP_HAS_OPENMP
        #pragma omp parallel
        {
#endif

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 2; i++)
        {
            if (sA.D[i]) {
                int m = sA.D[i]->rows();
                int n = sA.D[i]->cols();
                assert(sB.x[i].rows() == n && sD.D[i]->rows() == m && sD.D[i]->cols() == n && "size mismatch");
                // sD.D = sA.D * diag(sB)
                blasfeo_dgemm_nd(m, n, 1.0, sA.D[i]->ref(), 0, 0, sB.x[i].ref(), 0, 0.0, sD.D[i]->ref(), 0, 0, sD.D[i]->ref(), 0, 0);
            }

            if (sA.B[i]) {
                int m = sA.B[i]->rows();
                int n = sA.B[i]->cols();
                assert(sB.x[i].rows() == n && sD.B[i]->rows() == m && sD.B[i]->cols() == n && "size mismatch");
                // sD.B = sA.B * diag(sB)
                blasfeo_dgemm_nd(m, n, 1.0, sA.B[i]->ref(), 0, 0, sB.x[i].ref(), 0, 0.0, sD.B[i]->ref(), 0, 0, sD.B[i]->ref(), 0, 0);
            }

            if (sA.E[i]) {
                int m = sA.E[i]->rows();
                int n = sA.E[i]->cols();
                assert(sB.x[i].rows() == n && sD.E[i]->rows() == m && sD.E[i]->cols() == n && "size mismatch");
                // sD.E = sA.E * diag(sB)
                blasfeo_dgemm_nd(m, n, 1.0, sA.E[i]->ref(), 0, 0, sB.x[i].ref(), 0, 0.0, sD.E[i]->ref(), 0, 0, sD.E[i]->ref(), 0, 0);
            }
        }

#ifdef PIQP_HAS_OPENMP
        } // end of parallel region
#endif
    }

    void factor_kkt()
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

        int m = kkt_mat.D[0]->rows();
        int n, k;
        // L_1 = chol(D_1)
        blasfeo_dpotrf_l(m, kkt_mat.D[0]->ref(), 0, 0, kkt_factor.D[0]->ref(), 0, 0);

        if (kkt_mat.B[0]) {
            m = kkt_mat.B[0]->rows();
            n = kkt_mat.B[0]->cols();
            assert(kkt_factor.D[0]->rows() == n && kkt_factor.D[0]->cols() == n && "size mismatch");
            assert(kkt_factor.B[0]->rows() == m && kkt_factor.B[0]->cols() == n && "size mismatch");
            // C_1 = B_1 * L_1^{-T}
            blasfeo_dtrsm_rltn(m, n, 1.0, kkt_factor.D[0]->ref(), 0, 0, kkt_mat.B[0]->ref(), 0, 0, kkt_factor.B[0]->ref(), 0, 0);
        }

        if (arrow_width > 0)
        {
            if (kkt_mat.E[0]) {
                m = kkt_mat.E[0]->rows();
                n = kkt_mat.E[0]->cols();
                assert(kkt_factor.D[0]->rows() == n && kkt_factor.D[0]->cols() == n && "size mismatch");
                assert(kkt_factor.E[0]->rows() == m && kkt_factor.E[0]->cols() == n && "size mismatch");
                // F_1 = E_1 * L_1^{-T}
                blasfeo_dtrsm_rltn(m, n, 1.0, kkt_factor.D[0]->ref(), 0, 0, kkt_mat.E[0]->ref(), 0, 0, kkt_factor.E[0]->ref(), 0, 0);
                // L_N = D_N - F_1 * F_1^T
                blasfeo_dsyrk_ln(arrow_width, n, -1.0, kkt_factor.E[0]->ref(), 0, 0, kkt_factor.E[0]->ref(), 0, 0, 1.0, kkt_mat.D[N-1]->ref(), 0, 0, kkt_factor.D[N-1]->ref(), 0, 0);
            } else {
                // L_N = D_N
                blasfeo_dtrcp_l(arrow_width, kkt_mat.D[N-1]->ref(), 0, 0, kkt_factor.D[N-1]->ref(), 0, 0);
            }
        }

        for (std::size_t i = 1; i < N - 1; i++)
        {
            if (kkt_factor.B[i-1]) {
                m = kkt_factor.B[i-1]->rows();
                k = kkt_factor.B[i-1]->cols();
                assert(kkt_mat.D[i]->rows() == m && kkt_mat.D[i]->cols() == m && "size mismatch");
                assert(kkt_factor.D[i]->rows() == m && kkt_factor.D[i]->cols() == m && "size mismatch");
                // L_i = chol(D_i - C_{i-1} * C_{i-1}^T)
                blasfeo_dsyrk_ln(m, k, -1.0, kkt_factor.B[i-1]->ref(), 0, 0, kkt_factor.B[i-1]->ref(), 0, 0, 1.0, kkt_mat.D[i]->ref(), 0, 0, kkt_factor.D[i]->ref(), 0, 0);
                blasfeo_dpotrf_l(m, kkt_factor.D[i]->ref(), 0, 0, kkt_factor.D[i]->ref(), 0, 0);
            } else {
                m = kkt_mat.D[i]->rows();
                assert(kkt_mat.D[i]->rows() == m && "size mismatch");
                assert(kkt_factor.D[i]->rows() == m && kkt_factor.D[i]->cols() == m && "size mismatch");
                // L_i = chol(D_i)
                blasfeo_dpotrf_l(m, kkt_mat.D[i]->ref(), 0, 0, kkt_factor.D[i]->ref(), 0, 0);
            }

            if (i < N - 2 && kkt_mat.B[i]) {
                m = kkt_mat.B[i]->rows();
                n = kkt_mat.B[i]->cols();
                assert(kkt_factor.D[i]->rows() == n && kkt_factor.D[i]->cols() == n && "size mismatch");
                assert(kkt_factor.B[i]->rows() == m && kkt_factor.B[i]->cols() == n && "size mismatch");
                // C_i = B_i * L_i^{-T}
                blasfeo_dtrsm_rltn(m, n, 1.0, kkt_factor.D[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0, kkt_factor.B[i]->ref(), 0, 0);
            }

            if (arrow_width > 0)
            {
                if (kkt_mat.E[i] && kkt_factor.E[i-1] && kkt_factor.B[i-1])
                {
                    m = kkt_factor.E[i-1]->rows();
                    n = kkt_factor.B[i-1]->rows();
                    k = kkt_factor.E[i-1]->cols();
                    assert(kkt_factor.B[i-1]->cols() == k && "size mismatch");
                    assert(kkt_mat.E[i]->rows() == m && kkt_mat.E[i]->cols() == n && "size mismatch");
                    assert(kkt_factor.D[i]->rows() == n && kkt_factor.D[i]->cols() == n && "size mismatch");
                    // F_i = (E_i - F_{i-1} * C_{i-1}^T) * L_i^{-T}
                    blasfeo_dgemm_nt(m, n, k, -1.0, kkt_factor.E[i-1]->ref(), 0, 0, kkt_factor.B[i-1]->ref(), 0, 0, 1.0, kkt_mat.E[i]->ref(), 0, 0, kkt_factor.E[i]->ref(), 0, 0);
                    blasfeo_dtrsm_rltn(m, n, 1.0, kkt_factor.D[i]->ref(), 0, 0, kkt_factor.E[i]->ref(), 0, 0, kkt_factor.E[i]->ref(), 0, 0);
                }
                else if (kkt_mat.E[i])
                {
                    m = kkt_mat.E[i]->rows();
                    n = kkt_mat.E[i]->cols();
                    assert(kkt_factor.D[i]->rows() == n && kkt_factor.D[i]->cols() == n && "size mismatch");
                    // F_i = E_i * L_i^{-T}
                    blasfeo_dtrsm_rltn(m, n, 1.0, kkt_factor.D[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0, kkt_factor.E[i]->ref(), 0, 0);
                }
                else if (kkt_factor.E[i-1] && kkt_factor.B[i-1])
                {
                    m = kkt_factor.E[i-1]->rows();
                    n = kkt_factor.B[i-1]->rows();
                    k = kkt_factor.E[i-1]->cols();
                    assert(kkt_factor.B[i-1]->cols() == k && "size mismatch");
                    assert(kkt_factor.D[i]->rows() == n && kkt_factor.D[i]->cols() == n && "size mismatch");
                    // F_i = -(F_{i-1} * C_{i-1}^T) * L_i^{-T}
                    blasfeo_dgemm_nt(m, n, k, -1.0, kkt_factor.E[i-1]->ref(), 0, 0, kkt_factor.B[i-1]->ref(), 0, 0, 0.0, kkt_factor.E[i]->ref(), 0, 0, kkt_factor.E[i]->ref(), 0, 0);
                    blasfeo_dtrsm_rltn(m, n, 1.0, kkt_factor.D[i]->ref(), 0, 0, kkt_factor.E[i]->ref(), 0, 0, kkt_factor.E[i]->ref(), 0, 0);
                }

                if (kkt_factor.E[i]) {
                    m = kkt_factor.E[i]->rows();
                    k = kkt_factor.E[i]->cols();
                    assert(m == arrow_width && "size mismatch");
                    assert(kkt_mat.D[N - 1]->rows() == m && kkt_mat.D[N - 1]->cols() == m && "size mismatch");
                    // L_N -= F_i * F_i^T
                    blasfeo_dsyrk_ln(m, k, -1.0, kkt_factor.E[i]->ref(), 0, 0, kkt_factor.E[i]->ref(), 0, 0, 1.0, kkt_factor.D[N - 1]->ref(), 0, 0, kkt_factor.D[N - 1]->ref(), 0, 0);
                }
            }
        }

        // L_N = chol(D_N - sum F_i * F_i^T)
        // note that inner is also computed and stored in L_N
        blasfeo_dpotrf_l(arrow_width, kkt_factor.D[N-1]->ref(), 0, 0, kkt_factor.D[N-1]->ref(), 0, 0);
    }

    // z = sA * x
    void block_symv_l(BlockKKT& sA, BlockVec& x, BlockVec& z)
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;
        for (std::size_t i = 0; i < N; i++)
        {
            if (sA.D[i]) {
                int m = sA.D[i]->rows();
                assert(x.x[i].rows() == m && "size mismatch");
                assert(z.x[i].rows() == m && "size mismatch");
                // z_i = D_i * x_i, D_i is symmetric and only the lower triangular part of D_i is accessed
                blasfeo_dsymv_l(m, 1.0, sA.D[i]->ref(), 0, 0, x.x[i].ref(), 0, 0.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
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
                assert(x.x[i+1].rows() == m && "size mismatch");
                assert(z.x[i+1].rows() == m && "size mismatch");
                assert(z.x[i].rows() == n && "size mismatch");
                // z_{i+1} += B_i * x_i
                // z_i += B_i^T * x_{i+1}
                blasfeo_dgemv_nt(m, n, 1.0, 1.0, sA.B[i]->ref(), 0, 0, x.x[i].ref(), 0, x.x[i+1].ref(), 0, 1.0, 1.0, z.x[i+1].ref(), 0, z.x[i].ref(), 0, z.x[i+1].ref(), 0, z.x[i].ref(), 0);
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
                    // z_{N-1} += E_i * x_i
                    // z_i += E_i^T * x_{N-1}
                    blasfeo_dgemv_nt(m, n, 1.0, 1.0, sA.E[i]->ref(), 0, 0, x.x[i].ref(), 0, x.x[N-1].ref(), 0, 1.0, 1.0, z.x[N-1].ref(), 0, z.x[i].ref(), 0, z.x[N-1].ref(), 0, z.x[i].ref(), 0);
                }
            }
        }
    }

    // y = alpha * x
    void block_veccpsc(double alpha, BlockVec& x, BlockVec& y)
    {
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
    // A = [A_{1,1}                                                           ]
    //     [A_{1,2} A_{2,2}                                                   ]
    //     [        A_{2,3} A_{3,3}                                           ]
    //     [                A_{3,4} A_{4,4}                        A_{N-2,N-2}]
    //     [                          ...                          A_{N-2,N-1}]
    //     [A_{1,N} A_{2,N} A_{3,N}   ...      A_{N-4,N} A_{N-3,N} A_{N-2,N}  ]
    void block_t_gemv_n(double alpha, BlockMat<I>& sA, BlockVec& x, double beta, BlockVec& y, BlockVec& z)
    {
        // z = beta * y
        block_veccpsc(beta, y, z);

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

#ifdef PIQP_HAS_OPENMP
        #pragma omp parallel
        {
#endif

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 1; i++)
        {
            if (i < N - 2 && sA.D[i]) {
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
                assert(z.x[i].rows() == m && "size mismatch");
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
            for (std::size_t i = 0; i < N - 2; i++)
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

#ifdef PIQP_HAS_OPENMP
        } // end of parallel region
#endif
    }

    // z = beta * y + alpha * A^T * x
    // here it's assumed that the sparsity of the block matrix
    // is transposed without the blocks individually transposed
    // A = [A_{1,1}                                                           ]
    //     [A_{1,2} A_{2,2}                                                   ]
    //     [        A_{2,3} A_{3,3}                                           ]
    //     [                A_{3,4} A_{4,4}                        A_{N-2,N-2}]
    //     [                          ...                          A_{N-2,N-1}]
    //     [A_{1,N} A_{2,N} A_{3,N}   ...      A_{N-4,N} A_{N-3,N} A_{N-2,N}  ]
    void block_t_gemv_t(double alpha, BlockMat<I>& sA, BlockVec& x, double beta, BlockVec& y, BlockVec& z)
    {
        // z = beta * y
        block_veccpsc(beta, y, z);

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

#ifdef PIQP_HAS_OPENMP
        #pragma omp parallel
        {
#endif

#ifdef PIQP_HAS_OPENMP
        #pragma omp for nowait
#endif
        for (std::size_t i = 0; i < N - 2; i++)
        {
            if (sA.D[i]) {
                int m = sA.D[i]->rows();
                int n = sA.D[i]->cols();
                assert(x.x[i].rows() == m && "size mismatch");
                assert(z.x[i].rows() == n && "size mismatch");
                // z_i += alpha * D_i^T * x_i
                blasfeo_dgemv_t(m, n, alpha, sA.D[i]->ref(), 0, 0, x.x[i].ref(), 0, 1.0, z.x[i].ref(), 0, z.x[i].ref(), 0);
            }

            if (sA.B[i]) {
                int m = sA.B[i]->rows();
                int n = sA.B[i]->cols();
                assert(x.x[i+1].rows() == m && "size mismatch");
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

#ifdef PIQP_HAS_OPENMP
        } // end of parallel region
#endif
    }

    // z_n = beta_n * y_n + alpha_n * A * x_n
    // z_t = beta_t * y_t + alpha_t * A^T * x_t
    // here it's assumed that the sparsity of the block matrix
    // is transposed without the blocks individually transposed
    // A = [A_{1,1}                                                           ]
    //     [A_{1,2} A_{2,2}                                                   ]
    //     [        A_{2,3} A_{3,3}                                           ]
    //     [                A_{3,4} A_{4,4}                        A_{N-2,N-2}]
    //     [                          ...                          A_{N-2,N-1}]
    //     [A_{1,N} A_{2,N} A_{3,N}   ...      A_{N-4,N} A_{N-3,N} A_{N-2,N}  ]
    void block_t_gemv_nt(double alpha_n, double alpha_t, BlockMat<I>& sA, BlockVec& x_n, BlockVec& x_t,
                         double beta_n, double beta_t, BlockVec& y_n, BlockVec& y_t, BlockVec& z_n, BlockVec& z_t)
    {
        // z_n = beta_n * y_n
        block_veccpsc(beta_n, y_n, z_n);
        // z_t = beta_t * y_t
        block_veccpsc(beta_t, y_t, z_t);

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;
        for (std::size_t i = 0; i < N - 2; i++)
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

            if (sA.B[i]) {
                int m = sA.B[i]->rows();
                int n = sA.B[i]->cols();
                assert(x_n.x[i].rows() == n && "size mismatch");
                assert(z_n.x[i+1].rows() == m && "size mismatch");
                assert(x_t.x[i+1].rows() == m && "size mismatch");
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
        // ----- FORWARD SUBSTITUTION -----

        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

        int m = kkt_factor.D[0]->rows();
        int n;
        assert(b_and_x.x[0].rows() == m && "size mismatch");
        // y_1 = L_1^{-1} * b_1
        blasfeo_dtrsv_lnn(m, kkt_factor.D[0]->ref(), 0, 0, b_and_x.x[0].ref(), 0, b_and_x.x[0].ref(), 0);

        for (std::size_t i = 1; i < N - 1; i++)
        {
            if (kkt_factor.B[i-1]) {
                m = kkt_factor.D[i]->rows();
                n = kkt_factor.B[i-1]->cols();
                assert(kkt_factor.B[i-1]->rows() == m && "size mismatch");
                assert(b_and_x.x[i-1].rows() == n && "size mismatch");
                assert(b_and_x.x[i].rows() == m && "size mismatch");
                // y_i = L_i^{-1} * (b_i - C_{i-1} * y_{i-1})
                blasfeo_dgemv_n(m, n, -1.0, kkt_factor.B[i-1]->ref(), 0, 0, b_and_x.x[i-1].ref(), 0, 1.0, b_and_x.x[i].ref(), 0, b_and_x.x[i].ref(), 0);
                blasfeo_dtrsv_lnn(m, kkt_factor.D[i]->ref(), 0, 0, b_and_x.x[i].ref(), 0, b_and_x.x[i].ref(), 0);
            }
        }

        if (arrow_width > 0)
        {
            for (std::size_t i = 0; i < N - 1; i++)
            {
                if (kkt_factor.E[i]) {
                    m = kkt_factor.E[i]->rows();
                    n = kkt_factor.E[i]->cols();
                    assert(b_and_x.x[i].rows() == n && "size mismatch");
                    assert(b_and_x.x[N-1].rows() == m && "size mismatch");
                    // y_N -= F_i * y_i
                    blasfeo_dgemv_n(m, n, -1.0, kkt_factor.E[i]->ref(), 0, 0, b_and_x.x[i].ref(), 0, 1.0, b_and_x.x[N-1].ref(), 0, b_and_x.x[N-1].ref(), 0);
                }
            }
            m = kkt_factor.D[N-1]->rows();
            assert(b_and_x.x[N-1].rows() == m && "size mismatch");
            // y_N = L_N^{-1} * y_N
            blasfeo_dtrsv_lnn(m, kkt_factor.D[N-1]->ref(), 0, 0, b_and_x.x[N-1].ref(), 0, b_and_x.x[N-1].ref(), 0);
        }

        // ----- BACK SUBSTITUTION -----

        if (arrow_width > 0)
        {
            m = kkt_factor.D[N-1]->rows();
            assert(b_and_x.x[N-1].rows() == m && "size mismatch");
            // x_N = L_N^{-T} * y_N
            blasfeo_dtrsv_ltn(m, kkt_factor.D[N-1]->ref(), 0, 0, b_and_x.x[N-1].ref(), 0, b_and_x.x[N-1].ref(), 0);

            if (kkt_factor.E[N-2]) {
                m = kkt_factor.E[N-2]->rows();
                n = kkt_factor.E[N-2]->cols();
                assert(b_and_x.x[N-1].rows() == m && "size mismatch");
                assert(b_and_x.x[N-2].rows() == n && "size mismatch");
                // x_{N-1} = y_{N-1} - F_{N-1}^T * x_N
                blasfeo_dgemv_t(m, n, -1.0, kkt_factor.E[N-2]->ref(), 0, 0, b_and_x.x[N-1].ref(), 0, 1.0, b_and_x.x[N-2].ref(), 0, b_and_x.x[N-2].ref(), 0);
            }
        }

        m = kkt_factor.D[N-2]->rows();
        assert(b_and_x.x[N-2].rows() == m && "size mismatch");
        // x_{N-1} = L_{N-1}^{-T} * x_{N-1}
        blasfeo_dtrsv_ltn(m, kkt_factor.D[N-2]->ref(), 0, 0, b_and_x.x[N-2].ref(), 0, b_and_x.x[N-2].ref(), 0);

        for (std::size_t i = N - 2; i--;)
        {
            if (kkt_factor.B[i]) {
                m = kkt_factor.B[i]->rows();
                n = kkt_factor.B[i]->cols();
                assert(b_and_x.x[i+1].rows() == m && "size mismatch");
                assert(b_and_x.x[i].rows() == n && "size mismatch");
                // x_i = y_i - C_i^T * x_{i+1}
                blasfeo_dgemv_t(m, n, -1.0, kkt_factor.B[i]->ref(), 0, 0, b_and_x.x[i+1].ref(), 0, 1.0, b_and_x.x[i].ref(), 0, b_and_x.x[i].ref(), 0);
            }

            if (kkt_factor.E[i]) {
                m = kkt_factor.E[i]->rows();
                n = kkt_factor.E[i]->cols();
                assert(b_and_x.x[N-1].rows() == m && "size mismatch");
                assert(b_and_x.x[i].rows() == n && "size mismatch");
                // x_i -= F_i^T * x_N
                blasfeo_dgemv_t(m, n, -1.0, kkt_factor.E[i]->ref(), 0, 0, b_and_x.x[N-1].ref(), 0, 1.0, b_and_x.x[i].ref(), 0, b_and_x.x[i].ref(), 0);
            }

            m = kkt_factor.D[i]->rows();
            assert(b_and_x.x[i].rows() == m && "size mismatch");
            // x_i = L_i^{-T} * x_i
            blasfeo_dtrsv_ltn(m, kkt_factor.D[i]->ref(), 0, 0, b_and_x.x[i].ref(), 0, b_and_x.x[i].ref(), 0);
        }
    }
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/sparse/blocksparse_stage_kkt.tpp"
#endif

#endif //PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_HPP
