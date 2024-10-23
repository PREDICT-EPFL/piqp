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

        BlasfeoMat(BlasfeoMat&& other) noexcept
        {
            this->mat = other.mat;
            other.mat.mem = nullptr;
            other.mat.m = 0;
            other.mat.n = 0;
        }

        BlasfeoMat(const BlasfeoMat& other)
        {
            if (other.mat.mem) {
                this->resize(other.rows(), other.cols());
                // B <= A
                blasfeo_dgecp(other.rows(), other.cols(), const_cast<BlasfeoMat&>(other).ref(), 0, 0, this->ref(), 0, 0);
            }
        }

        BlasfeoMat& operator=(BlasfeoMat&& other) noexcept
        {
            this->mat = other.mat;
            other.mat.mem = nullptr;
            other.mat.m = 0;
            other.mat.n = 0;
            return *this;
        }

        BlasfeoMat& operator=(const BlasfeoMat& other)
        {
            if (other.mat.mem) {
                this->resize(other.rows(), other.cols());
                // B <= A
                blasfeo_dgecp(other.rows(), other.cols(), const_cast<BlasfeoMat&>(other).ref(), 0, 0, this->ref(), 0, 0);
            } else {
                if (mat.mem) {
                    blasfeo_free_dmat(&mat);
                    mat.mem = nullptr;
                }
                mat.m = 0;
                mat.n = 0;
            }
            return *this;
        }

        ~BlasfeoMat()
        {
            if (mat.mem) {
                blasfeo_free_dmat(&mat);
            }
        }

        int rows() const { return mat.m; }

        int cols() const { return mat.n; }

        void resize(int m, int n)
        {
            // reuse memory
            if (this->rows() == m && this->cols() == n) return;

            if (mat.mem) {
                blasfeo_free_dmat(&mat);
            }

            blasfeo_allocate_dmat(m, n, &mat);
        }

        void setZero()
        {
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

    struct BlockMat
    {
        // Row permutation from original matrix to block matrix.
        // For example, if the original matrix b = A * x, then
        // b_perm = A_block * x, where b_perm[i] = b[perm[i]].
        Vec<I> perm;
        std::vector<std::unique_ptr<BlasfeoMat>> D; // A_{x,1}
        std::vector<std::unique_ptr<BlasfeoMat>> B; // A_{x,2}
        std::vector<std::unique_ptr<BlasfeoMat>> E; // A_{x,N+1}

        BlockMat() = default;

        BlockMat(BlockMat&&) = default;

        BlockMat(const BlockMat& other)
        {
            perm = other.perm;

            D.resize(other.D.size());
            B.resize(other.B.size());
            E.resize(other.E.size());

            for (std::size_t i = 0; i < other.D.size(); i++) {
                if (other.D[i]) {
                    D[i] = std::make_unique<BlasfeoMat>(*other.D[i]);
                }
            }

            for (std::size_t i = 0; i < other.B.size(); i++) {
                if (other.B[i]) {
                    B[i] = std::make_unique<BlasfeoMat>(*other.B[i]);
                }
            }

            for (std::size_t i = 0; i < other.E.size(); i++) {
                if (other.E[i]) {
                    E[i] = std::make_unique<BlasfeoMat>(*other.E[i]);
                }
            }
        }

        BlockMat& operator=(BlockMat&&) = default;

        BlockMat& operator=(const BlockMat& other)
        {
            perm = other.perm;

            D.resize(other.D.size());
            B.resize(other.B.size());
            E.resize(other.E.size());

            for (std::size_t i = 0; i < other.D.size(); i++) {
                if (other.D[i] && !D[i]) {
                    D[i] = std::make_unique<BlasfeoMat>(*other.D[i]);
                } else if (other.D[i] && D[i]) {
                    *D[i] = *other.D[i];
                } else {
                    D[i] = nullptr;
                }
            }

            for (std::size_t i = 0; i < other.D.size(); i++) {
                if (other.B[i] && !B[i]) {
                    B[i] = std::make_unique<BlasfeoMat>(*other.B[i]);
                } else if (other.B[i] && B[i]) {
                    *B[i] = *other.B[i];
                } else {
                    B[i] = nullptr;
                }
            }

            for (std::size_t i = 0; i < other.D.size(); i++) {
                if (other.E[i] && !E[i]) {
                    E[i] = std::make_unique<BlasfeoMat>(*other.E[i]);
                } else if (other.E[i] && E[i]) {
                    *E[i] = *other.E[i];
                } else {
                    E[i] = nullptr;
                }
            }

            return *this;
        }
    };

    const Data<T, I>& data;
    const Settings<T>& settings;

    T m_rho;
    T m_delta;

    std::vector<BlockInfo> block_info;

    BlockKKT P;
    BlockMat A;
    BlockMat G;
    BlockMat G_scaled;
    BlockKKT AtA;
    BlockKKT GtG;
    BlockKKT kkt_mat;

public:
    BlocksparseStageKKT(const Data<T, I>& data, const Settings<T>& settings) : data(data), settings(settings)
    {
        m_rho = T(1);
        m_delta = T(1);

        extract_arrow_structure();
        P = utri_to_kkt(data.P_utri);
        A = transpose_to_block_mat(data.AT);
        G = transpose_to_block_mat(data.GT);
        G_scaled = G;
        block_syrk_lt(A, A, AtA);
        block_syrk_lt(G, G_scaled, GtG);
        populate_kkt_mat();
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
                        A_kkt.D[block_index]->setZero();
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
                        A_kkt.E[current_arrow_block_index]->setZero();
                    }
                    BLASFEO_DMATEL(A_kkt.E[current_arrow_block_index]->ref(), i - block_start, j - current_arrow_block_start) = v;
                }
                // we have to be on off diagonal
                else
                {
                    assert(j + block_width + last_block_width > i && "indexes in no valid block");
                    if (!A_kkt.B[block_index - 1]) {
                        A_kkt.B[block_index - 1] = std::make_unique<BlasfeoMat>(block_width, last_block_width);
                        A_kkt.B[block_index - 1]->setZero();
                    }
                    BLASFEO_DMATEL(A_kkt.B[block_index - 1]->ref(), i - block_start, j + last_block_width - block_start) = v;
                }
            }
        }

        return A_kkt;
    }

    BlockMat transpose_to_block_mat(const SparseMat<T, I>& AT)
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

        BlockMat A_block;
        A_block.perm.resize(AT.cols());
        A_block.D.resize(N - 2);
        A_block.B.resize(N - 2);
        A_block.E.resize(N - 2);

        // keep track on the current fill status of each block
        Vec<I> block_fill(block_info.size());
        block_fill.setZero();

        // Iterating over a csc transposed matrix corresponds
        // to iterating over the rows of the non-transposed matrix.

        // First pass is to determine the number of rows per block
        Eigen::Index rows = AT.outerSize(); // rows here corresponds to the non-transposed matrix
        Eigen::Index cols = AT.innerSize();
        for (Eigen::Index i = 0; i < rows; i++)
        {
            typename SparseMat<T, I>::InnerIterator A_row_it(AT, i);
            if (A_row_it)
            {
                I j = A_row_it.index();
                std::size_t block_index = 0;
                // find the corresponding block
                while (block_info[block_index].start + block_info[block_index].width < j) { block_index++; }
                block_fill(Eigen::Index(block_index))++;
            }
        }

        Vec<I> block_fill_acc(block_info.size() + 1);
        block_fill_acc[0] = 0;
        for (Eigen::Index i = 0; i < Eigen::Index(block_info.size()); i++) {
            block_fill_acc[i + 1] = block_fill_acc[i] + block_fill[i];
        }

        // keep track on where we are in block
        Vec<I> block_fill_counter(block_info.size());
        block_fill_counter.setZero();

        // In the second pass, we allocate and fill the block matrix
        for (Eigen::Index i = 0; i < rows; i++)
        {
            typename SparseMat<T, I>::InnerIterator A_row_it(AT, i);

            std::size_t block_index = 0;
            I block_i = 0;
            if (A_row_it)
            {
                I j = A_row_it.index();
                // find the corresponding block
                while (block_info[block_index].start + block_info[block_index].width < j) { block_index++; }
                block_i = block_fill_counter(Eigen::Index(block_index))++;
                A_block.perm[i] = block_fill_acc[Eigen::Index(block_index)] + block_i;
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
                    if (!A_block.E[block_index]) {
                        A_block.E[block_index] = std::make_unique<BlasfeoMat>(block_fill[Eigen::Index(block_index)], arrow_width);
                        A_block.E[block_index]->setZero();
                    }
                    BLASFEO_DMATEL(A_block.E[block_index]->ref(), block_i, j + arrow_width - cols) = v;
                }
                // first block
                else if (j < block_start + block_width)
                {
                    if (!A_block.D[block_index]) {
                        A_block.D[block_index] = std::make_unique<BlasfeoMat>(block_fill[Eigen::Index(block_index)], block_width);
                        A_block.D[block_index]->setZero();
                    }
                    BLASFEO_DMATEL(A_block.D[block_index]->ref(), block_i, j - block_start) = v;
                }
                // second block
                else
                {
                    I next_block_width = block_info[block_index + 1].width;
                    assert(j < block_start + block_width + next_block_width && "indexes in no valid block");

                    if (!A_block.B[block_index]) {
                        A_block.B[block_index] = std::make_unique<BlasfeoMat>(block_fill[Eigen::Index(block_index)], next_block_width);
                        A_block.B[block_index]->setZero();
                    }
                    BLASFEO_DMATEL(A_block.B[block_index]->ref(), block_i, j - block_start - block_width) = v;
                }
            }
        }

        return A_block;
    }

    void block_syrk_lt(BlockMat& sA, BlockMat& sB, BlockKKT& sD)
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

        sD.D.resize(N);
        sD.B.resize(N - 2);
        sD.E.resize(N - 1);

        // ----- DIAGONAL -----

        // D.D_1 = 0
        if (sD.D[0]) {
            sD.D[0]->setZero();
        }
        for (std::size_t i = 0; i < N - 2; i++)
        {
            // D.D_{i+1} = 0
            if (sD.D[i+1]) {
                sD.D[i+1]->setZero();
            }

            if (sA.D[i] && sB.D[i]) {
                int m = sA.D[i]->cols();
                int k = sA.D[i]->rows();
                assert(sB.D[i]->cols() == m && sB.D[i]->rows() == k && "size mismatch");
                if (!sD.D[i]) {
                    sD.D[i] = std::make_unique<BlasfeoMat>(m, m);
                    sD.D[i]->setZero();
                }
                // D.D_i += lower triangular of A_{i,i}^T * B_{i,i}
                blasfeo_dsyrk_lt(m, k, 1.0, sA.D[i]->ref(), 0, 0, sB.D[i]->ref(), 0, 0, 1.0, sD.D[i]->ref(), 0, 0, sD.D[i]->ref(), 0, 0);
            }

            if (sA.B[i] && sB.B[i]) {
                int m = sA.B[i]->cols();
                int k = sA.B[i]->rows();
                assert(sB.B[i]->cols() == m && sB.B[i]->rows() == k && "size mismatch");
                if (!sD.D[i+1]) {
                    sD.D[i+1] = std::make_unique<BlasfeoMat>(m, m);
                    sD.D[i+1]->setZero();
                }
                // D.D_{i+1} += lower triangular of A_{i,i}^T * B_{i,i}
                blasfeo_dsyrk_lt(m, k, 1.0, sA.B[i]->ref(), 0, 0, sB.B[i]->ref(), 0, 0, 1.0, sD.D[i+1]->ref(), 0, 0, sD.D[i+1]->ref(), 0, 0);
            }
        }

        if (arrow_width > 0)
        {
            // D.D_N = 0
            if (sD.D[N-1]) {
                sD.D[N-1]->setZero();
            }

            for (std::size_t i = 0; i < N - 2; i++)
            {
                if (sA.E[i] && sB.E[i]) {
                    int m = sA.E[i]->cols();
                    int k = sA.E[i]->rows();
                    assert(sB.E[i]->cols() == m && sB.E[i]->rows() == k && "size mismatch");
                    if (!sD.D[N-1]) {
                        sD.D[N-1] = std::make_unique<BlasfeoMat>(m, m);
                        sD.D[N-1]->setZero();
                    }
                    // D.D_N += lower triangular of A_{i,N}^T * B_{i,N}
                    blasfeo_dsyrk_lt(m, k, 1.0, sA.E[i]->ref(), 0, 0, sB.E[i]->ref(), 0, 0, 1.0, sD.D[N-1]->ref(), 0, 0, sD.D[N-1]->ref(), 0, 0);
                }
            }
        }

        // ----- OFF-DIAGONAL -----

        for (std::size_t i = 0; i < N - 2; i++)
        {
            if (sA.B[i] && sB.D[i]) {
                int m = sA.B[i]->cols();
                int n = sB.D[i]->cols();
                int k = sA.B[i]->rows();
                assert(sB.D[i]->rows() == k && "size mismatch");
                if (!sD.B[i]) {
                    sD.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    sD.B[i]->setZero();
                }
                // D.B_i = A_{i,i+1}^T * B_{i,i}
                blasfeo_dgemm_tn(m, n, k, 1.0, sA.B[i]->ref(), 0, 0, sB.D[i]->ref(), 0, 0, 0.0, sD.B[i]->ref(), 0, 0, sD.B[i]->ref(), 0, 0);
            }
        }

        // ----- ARROW -----

        if (arrow_width > 0)
        {
            // D.E_1 = 0
            if (sD.E[0]) {
                sD.E[0]->setZero();
            }
            for (std::size_t i = 0; i < N - 2; i++)
            {
                // D.E_{i+1} = 0
                if (sD.E[i+1]) {
                    sD.E[i+1]->setZero();
                }

                if (sA.E[i] && sB.D[i]) {
                    int m = sA.E[i]->cols();
                    int n = sB.D[i]->cols();
                    int k = sA.E[i]->rows();
                    assert(sB.D[i]->rows() == k && "size mismatch");
                    if (!sD.E[i]) {
                        sD.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        sD.E[i]->setZero();
                    }
                    // D.E_i += A_{i,N}^T * B_{i,i}
                    blasfeo_dgemm_tn(m, n, k, 1.0, sA.E[i]->ref(), 0, 0, sB.D[i]->ref(), 0, 0, 1.0, sD.E[i]->ref(), 0, 0, sD.E[i]->ref(), 0, 0);
                }

                if (sA.E[i] && sB.B[i]) {
                    int m = sA.E[i]->cols();
                    int n = sB.B[i]->cols();
                    int k = sA.E[i]->rows();
                    assert(sB.B[i]->rows() == k && "size mismatch");
                    if (!sD.E[i+1]) {
                        sD.E[i+1] = std::make_unique<BlasfeoMat>(m, n);
                        sD.E[i+1]->setZero();
                    }
                    // D.E_{i+1} += A_{i,N}^T * B_{i,i+1}
                    blasfeo_dgemm_tn(m, n, k, 1.0, sA.E[i]->ref(), 0, 0, sB.B[i]->ref(), 0, 0, 1.0, sD.E[i+1]->ref(), 0, 0, sD.E[i+1]->ref(), 0, 0);
                }
            }
        }
    }

    void populate_kkt_mat()
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;
        T delta_inv = 1.0 / m_delta;

        kkt_mat.D.resize(N);
        kkt_mat.B.resize(N - 2);
        kkt_mat.E.resize(N - 1);

        // ----- DIAGONAL -----

        for (std::size_t i = 0; i < N; i++)
        {
            I m = block_info[i].width;

            if (!kkt_mat.D[i]) {
                kkt_mat.D[i] = std::make_unique<BlasfeoMat>(m, m);
            }
            kkt_mat.D[i]->setZero();

            // diag(D_i) += rho
            blasfeo_ddiare(m, m_rho, kkt_mat.D[i]->ref(), 0, 0);

            if (P.D[i]) {
                assert(P.D[i]->rows() == m && P.D[i]->cols() == m && "size mismatch");
                // D_i += P.D_i
                blasfeo_dgead(m, m, 1.0, P.D[i]->ref(), 0, 0, kkt_mat.D[i]->ref(), 0, 0);
            }

            if (AtA.D[i]) {
                assert(AtA.D[i]->rows() == m && AtA.D[i]->cols() == m && "size mismatch");
                // D_i += delta^{-1} * AtA.D_i
                blasfeo_dgead(m, m, delta_inv, AtA.D[i]->ref(), 0, 0, kkt_mat.D[i]->ref(), 0, 0);
            }

            if (GtG.D[i]) {
                assert(GtG.D[i]->rows() == m && GtG.D[i]->cols() == m && "size mismatch");
                // D_i += GtG.D_i
                blasfeo_dgead(m, m, 1.0, GtG.D[i]->ref(), 0, 0, kkt_mat.D[i]->ref(), 0, 0);
            }
        }

        // ----- OFF-DIAGONAL -----

        for (std::size_t i = 0; i < N - 2; i++)
        {
            int m = block_info[i + 1].width;
            int n = block_info[i].width;

            // B_i = 0
            if (kkt_mat.B[i]) {
                kkt_mat.B[i]->setZero();
            }

            if (P.B[i]) {
                assert(P.B[i]->rows() == m && P.B[i]->cols() == n && "size mismatch");
                if (!kkt_mat.B[i]) {
                    kkt_mat.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    kkt_mat.B[i]->setZero();
                }
                // B_i += P.B_i
                blasfeo_dgead(m, n, 1.0, P.D[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
            }

            if (AtA.B[i]) {
                assert(AtA.B[i]->rows() == m && AtA.B[i]->cols() == n && "size mismatch");
                if (!kkt_mat.B[i]) {
                    kkt_mat.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    kkt_mat.B[i]->setZero();
                }
                // B_i += delta^{-1} * AtA.B_i
                blasfeo_dgead(m, n, delta_inv, AtA.B[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
            }

            if (GtG.B[i]) {
                assert(GtG.B[i]->rows() == m && GtG.B[i]->cols() == n && "size mismatch");
                if (!kkt_mat.B[i]) {
                    kkt_mat.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    kkt_mat.B[i]->setZero();
                }
                // B_i += GtG.B_i
                blasfeo_dgead(m, n, 1.0, GtG.B[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
            }
        }

        // ----- ARROW -----

        if (arrow_width > 0)
        {
            for (std::size_t i = 0; i < N - 1; i++)
            {
                int m = arrow_width;
                int n = block_info[i].width;

                // E_i = 0
                if (kkt_mat.E[i]) {
                    kkt_mat.E[i]->setZero();
                }

                if (P.E[i]) {
                    assert(P.E[i]->rows() == m && P.E[i]->cols() == n && "size mismatch");
                    if (!kkt_mat.E[i]) {
                        kkt_mat.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        kkt_mat.E[i]->setZero();
                    }
                    // E_i += P.E_i
                    blasfeo_dgead(m, n, 1.0, P.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                }

                if (AtA.E[i]) {
                    assert(AtA.E[i]->rows() == m && AtA.E[i]->cols() == n && "size mismatch");
                    if (!kkt_mat.E[i]) {
                        kkt_mat.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        kkt_mat.E[i]->setZero();
                    }
                    // E_i += delta^{-1} * AtA.E_i
                    blasfeo_dgead(m, n, delta_inv, AtA.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                }

                if (GtG.E[i]) {
                    assert(GtG.E[i]->rows() == m && GtG.E[i]->cols() == n && "size mismatch");
                    if (!kkt_mat.E[i]) {
                        kkt_mat.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        kkt_mat.E[i]->setZero();
                    }
                    // E_i += GtG.E_i
                    blasfeo_dgead(m, n, 1.0, GtG.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                }
            }
        }
    }
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/sparse/blocksparse_stage_kkt.tpp"
#endif

#endif //PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_HPP
