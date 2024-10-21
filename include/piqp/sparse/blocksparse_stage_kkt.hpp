// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_HPP
#define PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_HPP

#include <iostream>
#include <cstring>
#include "blasfeo.h"

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"
#include "piqp/kkt_system.hpp"
#include "piqp/settings.hpp"
#include "piqp/sparse/data.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
class BlocksparseStageKKT : public KKTSystem<T>
{
protected:
    static_assert(std::is_same<T, double>::value, "blocksparse_stagewise only supports doubles");

    struct BlockInfo
    {
        I start;
        I width;
    };

    class BlasfeoMat
    {
    protected:
        blasfeo_dmat mat{}; // note that {} initializes all values to zero here

    public:
        BlasfeoMat() = default;

        BlasfeoMat(int m, int n)
        {
            resize(m, n);
        }

        ~BlasfeoMat()
        {
            if (mat.mem) {
                blasfeo_free_dmat(&mat);
            }
        }

        int rows() { return mat.m; }

        int cols() { return mat.n; }

        void resize(int m, int n)
        {
            if (mat.mem) {
                blasfeo_free_dmat(&mat);
            }

            blasfeo_allocate_dmat(m, n, &mat);
            // zero out matrix
            std::memset(mat.mem, 0, static_cast<std::size_t>(mat.memsize));
        }

        blasfeo_dmat* ref() { return &mat; }
    };

    // stores all the data of an arrow KKT structure
    struct BlockKKT
    {
        std::vector<std::unique_ptr<BlasfeoMat>> D; // lower triangular diagonal
        std::vector<std::unique_ptr<BlasfeoMat>> B; // off diagonal
        std::vector<std::unique_ptr<BlasfeoMat>> E; // arrow block
    };

    const Data<T, I>& data;
    const Settings<T>& settings;

    std::vector<BlockInfo> block_info;

    BlockKKT P;

public:
    BlocksparseStageKKT(const Data<T, I>& data, const Settings<T>& settings) : data(data), settings(settings)
    {
        extract_arrow_structure();
        P = utri_to_kkt(data.P_utri);
    }

    void update_data(int options)
    {
        assert(false && "not implemented yet");
    }

    bool update_scalings_and_factor(bool iterative_refinement,
                                    const T& rho, const T& delta,
                                    const CVecRef<T>& s, const CVecRef<T>& s_lb, const CVecRef<T>& s_ub,
                                    const CVecRef<T>& z, const CVecRef<T>& z_lb, const CVecRef<T>& z_ub)
    {
        return false;
    }

    void multiply(const CVecRef<T>& delta_x, const CVecRef<T>& delta_y,
                  const CVecRef<T>& delta_z, const CVecRef<T>& delta_z_lb, const CVecRef<T>& delta_z_ub,
                  const CVecRef<T>& delta_s, const CVecRef<T>& delta_s_lb, const CVecRef<T>& delta_s_ub,
                  VecRef<T> rhs_x, VecRef<T> rhs_y,
                  VecRef<T> rhs_z, VecRef<T> rhs_z_lb, VecRef<T> rhs_z_ub,
                  VecRef<T> rhs_s, VecRef<T> rhs_s_lb, VecRef<T> rhs_s_ub)
    {

    }

    void solve(const CVecRef<T>& rhs_x, const CVecRef<T>& rhs_y,
               const CVecRef<T>& rhs_z, const CVecRef<T>& rhs_z_lb, const CVecRef<T>& rhs_z_ub,
               const CVecRef<T>& rhs_s, const CVecRef<T>& rhs_s_lb, const CVecRef<T>& rhs_s_ub,
               VecRef<T> delta_x, VecRef<T> delta_y,
               VecRef<T> delta_z, VecRef<T> delta_z_lb, VecRef<T> delta_z_ub,
               VecRef<T> delta_s, VecRef<T> delta_s_lb, VecRef<T> delta_s_ub)
    {

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
        SparseMat<T, I> AtA = (data.AT * data.AT.transpose()).template triangularView<Eigen::Lower>();
        SparseMat<T, I> GtG = (data.GT * data.GT.transpose()).template triangularView<Eigen::Lower>();
        SparseMat<T, I> C = P_ltri + identity + AtA + GtG;

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
                    I new_diag_block_size = std::max(current_diag_block_size, (new_block_size + 1) / 2); // round up
                    I new_off_diag_block_size = std::max(current_off_diag_block_size, new_block_size - new_diag_block_size);
                    // potential new arrow width
                    I new_arrow_width = std::max(arrow_width, I(n) - j);

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
                std::cout << "B " << current_diag_block_start << " " << current_diag_block_size << " " << current_off_diag_block_size << " " << arrow_width << std::endl;
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

    BlockKKT utri_to_kkt(const SparseMat<T, I>& A_utri)
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

        BlockKKT A_kkt;
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
                if (j + block_width > i)
                {
                    if (!A_kkt.D[block_index]) {
                        A_kkt.D[block_index] = std::make_unique<BlasfeoMat>(block_width, block_width);
                    }
                    BLASFEO_DMATEL(A_kkt.D[block_index]->ref(), i - block_start, j - block_start) = v;
                }
                // check if on arrow
                else if (i + arrow_width > n)
                {
                    while (current_arrow_block_start < j) {
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

        return A_kkt;
    }
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/sparse/blocksparse_stage_kkt.tpp"
#endif

#endif //PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_HPP
