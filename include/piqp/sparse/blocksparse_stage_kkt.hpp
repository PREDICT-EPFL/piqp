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

            if (m == 0 || n == 0) {
                mat.mem = nullptr;
                mat.m = m;
                mat.n = n;
                return;
            }

            blasfeo_allocate_dmat(m, n, &mat);
            // make sure we don't have corrupted memory
            // which can result in massive slowdowns
            // https://github.com/giaf/blasfeo/issues/103
            setZero();
        }

        void setZero()
        {
            // zero out matrix
            std::memset(mat.mem, 0, static_cast<std::size_t>(mat.memsize));
        }

        blasfeo_dmat* ref() { return &mat; }
    };

    class BlasfeoVec
    {
    protected:
        blasfeo_dvec vec{}; // note that {} initializes all values to zero here

    public:
        BlasfeoVec() = default;

        explicit BlasfeoVec(int m)
        {
            resize(m);
        }

        BlasfeoVec(BlasfeoVec&& other) noexcept
        {
            this->vec = other.vec;
            other.vec.mem = nullptr;
            other.vec.m = 0;
        }

        BlasfeoVec(const BlasfeoVec& other)
        {
            if (other.vec.mem) {
                this->resize(other.rows());
                // y <= x
                blasfeo_dveccp(other.rows(), const_cast<BlasfeoVec&>(other).ref(), 0, this->ref(), 0);
            }
        }

        BlasfeoVec& operator=(BlasfeoVec&& other) noexcept
        {
            this->vec = other.vec;
            other.vec.mem = nullptr;
            other.vec.m = 0;
            return *this;
        }

        BlasfeoVec& operator=(const BlasfeoVec& other)
        {
            if (other.vec.mem) {
                this->resize(other.rows());
                // y <= x
                blasfeo_dveccp(other.rows(), const_cast<BlasfeoVec&>(other).ref(), 0, this->ref(), 0);
            } else {
                if (vec.mem) {
                    blasfeo_free_dvec(&vec);
                    vec.mem = nullptr;
                }
                vec.m = 0;
            }
            return *this;
        }

        ~BlasfeoVec()
        {
            if (vec.mem) {
                blasfeo_free_dvec(&vec);
            }
        }

        int rows() const { return vec.m; }

        void resize(int m)
        {
            // reuse memory
            if (this->rows() == m) return;

            if (vec.mem) {
                blasfeo_free_dvec(&vec);
            }

            if (m == 0) {
                vec.mem = nullptr;
                vec.m = 0;
                return;
            }

            blasfeo_allocate_dvec(m, &vec);
            // make sure we don't have corrupted memory
            // which can result in massive slowdowns
            // https://github.com/giaf/blasfeo/issues/103
            setZero();
        }

        void setZero()
        {
            // zero out vector
            std::memset(vec.mem, 0, static_cast<std::size_t>(vec.memsize));
        }

        void setConstant(double c)
        {
            blasfeo_dvecse(this->rows(), c, &vec, 0);
        }

        blasfeo_dvec* ref() { return &vec; }
    };

    // stores all the data of an arrow KKT structure
    struct BlockKKT
    {
        // [D_1                                  ]
        // [B_1 D_2                              ]
        // [    B_2 D_3                          ]
        // [            ...                      ]
        // [                B_N-3 D_N-2          ]
        // [                      B_N-2 D_N-1    ]
        // [E_1 E_2 E_3 ... E_N-3 E_N-2 E_N-1 D_N]

        std::vector<std::unique_ptr<BlasfeoMat>> D; // lower triangular diagonal
        std::vector<std::unique_ptr<BlasfeoMat>> B; // off diagonal
        std::vector<std::unique_ptr<BlasfeoMat>> E; // arrow block

        BlockKKT() = default;

        BlockKKT(BlockKKT&&) = default;

        BlockKKT(const BlockKKT& other)
        {
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

        BlockKKT& operator=(BlockKKT&&) = default;

        BlockKKT& operator=(const BlockKKT& other)
        {
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

            for (std::size_t i = 0; i < other.B.size(); i++) {
                if (other.B[i] && !B[i]) {
                    B[i] = std::make_unique<BlasfeoMat>(*other.B[i]);
                } else if (other.B[i] && B[i]) {
                    *B[i] = *other.B[i];
                } else {
                    B[i] = nullptr;
                }
            }

            for (std::size_t i = 0; i < other.E.size(); i++) {
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

    struct BlockMat
    {
        // [A_{1,1} A_{1,2}                                      A_{1,N}  ]
        // [        A_{2,2} A_{2,3}                              A_{2,N}  ]
        // [                A_{3,3} A_{3,4}                      A_{3,N}  ]
        // [                        ...                          ...      ]
        // [                             A_{N-2,N-2} A_{N-2,N-1} A_{N-2,N}]

        // Row permutation from original matrix to block matrix.
        // For example, if the original matrix b = A * x, then
        // b_perm = A_block * x, where b_perm[i] = b[perm[i]] and
        // b = A_block^T * x_perm, where x_perm[i] = x[perm_inv[i]].
        Vec<I> perm;
        Vec<I> perm_inv;
        Vec<I> block_row_sizes; // the row size of each block
        std::vector<std::unique_ptr<BlasfeoMat>> D; // A_{i,i}
        std::vector<std::unique_ptr<BlasfeoMat>> B; // A_{i,i+1}
        std::vector<std::unique_ptr<BlasfeoMat>> E; // A_{i,N}

        BlockMat() = default;

        BlockMat(BlockMat&&) = default;

        BlockMat(const BlockMat& other)
        {
            perm = other.perm;
            perm_inv = other.perm_inv;
            block_row_sizes = other.block_row_sizes;

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
            perm_inv = other.perm_inv;
            block_row_sizes = other.block_row_sizes;

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

            for (std::size_t i = 0; i < other.B.size(); i++) {
                if (other.B[i] && !B[i]) {
                    B[i] = std::make_unique<BlasfeoMat>(*other.B[i]);
                } else if (other.B[i] && B[i]) {
                    *B[i] = *other.B[i];
                } else {
                    B[i] = nullptr;
                }
            }

            for (std::size_t i = 0; i < other.E.size(); i++) {
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

    struct BlockVec
    {
        std::vector<BlasfeoVec> x;

        BlockVec() = default;

        BlockVec(BlockVec&&) = default;

        BlockVec(const BlockVec& other)
        {
            x.resize(other.x.size());

            for (std::size_t i = 0; i < other.x.size(); i++) {
                x[i] = other.x[i];
            }
        }

        explicit BlockVec(const std::vector<BlockInfo>& block_info)
        {
            std::size_t N = block_info.size();
            x.resize(N);
            for (std::size_t i = 0; i < N; i++)
            {
                x[i].resize(block_info[i].width);
            }
        }

        explicit BlockVec(const Vec<I>& block_info)
        {
            std::size_t N = std::size_t(block_info.rows());
            x.resize(N);
            for (std::size_t i = 0; i < N; i++)
            {
                x[i].resize(block_info[Eigen::Index(i)]);
            }
        }

        BlockVec& operator=(BlockVec&&) = default;

        BlockVec& operator=(const BlockVec& other)
        {
            x.resize(other.x.size());

            for (std::size_t i = 0; i < other.x.size(); i++) {
                x[i] = other.x[i];
            }

            return *this;
        }

        template <typename Derived>
        BlockVec& operator=(const Eigen::DenseBase<Derived>& other)
        {
            assign(other);
            return *this;
        }

        template <typename Derived>
        void assign(const Eigen::DenseBase<Derived>& other)
        {
            Eigen::Index i = 0;
            for (std::size_t block_idx = 0; block_idx < x.size(); block_idx++)
            {
                int block_size = x[block_idx].rows();
                for (int inner_idx = 0; inner_idx < block_size; inner_idx++)
                {
                    BLASFEO_DVECEL(this->x[block_idx].ref(), inner_idx) = other(i++);
                }
            }
        }

        template <typename Derived>
        void assign(const Eigen::DenseBase<Derived>& other, const Vec<I>& perm_inv)
        {
            Eigen::Index i = 0;
            for (std::size_t block_idx = 0; block_idx < x.size(); block_idx++)
            {
                int block_size = x[block_idx].rows();
                for (int inner_idx = 0; inner_idx < block_size; inner_idx++)
                {
                    BLASFEO_DVECEL(this->x[block_idx].ref(), inner_idx) = other(perm_inv(i++));
                }
            }
        }

        template <typename Derived>
        void load(Eigen::DenseBase<Derived>& other)
        {
            Eigen::Index i = 0;
            for (std::size_t block_idx = 0; block_idx < x.size(); block_idx++)
            {
                int block_size = x[block_idx].rows();
                for (int inner_idx = 0; inner_idx < block_size; inner_idx++)
                {
                    other(i++) = BLASFEO_DVECEL(this->x[block_idx].ref(), inner_idx);
                }
            }
        }

        template <typename Derived>
        void load(Eigen::DenseBase<Derived>& other, const Vec<I>& perm_inv)
        {
            Eigen::Index i = 0;
            for (std::size_t block_idx = 0; block_idx < x.size(); block_idx++)
            {
                int block_size = x[block_idx].rows();
                for (int inner_idx = 0; inner_idx < block_size; inner_idx++)
                {
                    other(perm_inv(i++)) = BLASFEO_DVECEL(this->x[block_idx].ref(), inner_idx);
                }
            }
        }
    };

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

    std::vector<BlockInfo> block_info;

    BlockKKT P;
    BlockVec P_diag;
    BlockMat AT;
    BlockMat GT;
    BlockVec G_scaling;
    BlockMat GT_scaled;

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

        // prepare kkt factorization
        extract_arrow_structure();
        std::size_t N = block_info.size();

        P = utri_to_kkt(data.P_utri);
        P_diag = BlockVec(block_info);
        // P_diag <= diag(P)
        for (std::size_t i = 0; i < N; i++)
        {
            assert(P_diag.x[i].rows() == P.D[i]->rows() && "size mismatch");
            blasfeo_ddiaex(P_diag.x[i].rows(), 1.0, P.D[i]->ref(), 0, 0, P_diag.x[i].ref(), 0);
        }

        AT = transpose_to_block_mat(data.AT, true);
        GT = transpose_to_block_mat(data.GT, true);
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
        assert(false && "not implemented yet");
    }

    bool update_scalings_and_factor(bool iterative_refinement,
                                    const T& rho, const T& delta,
                                    const CVecRef<T>& s, const CVecRef<T>& s_lb, const CVecRef<T>& s_ub,
                                    const CVecRef<T>& z, const CVecRef<T>& z_lb, const CVecRef<T>& z_ub)
    {
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

        block_delta_x = delta_x;
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

    BlockMat transpose_to_block_mat(const SparseMat<T, I>& sAT, bool store_transpose)
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

        BlockMat A_block;
        A_block.perm.resize(sAT.cols());
        A_block.perm_inv.resize(sAT.cols());
        A_block.D.resize(N - 2);
        A_block.B.resize(N - 2);
        A_block.E.resize(N - 2);

        // keep track on the current fill status of each block
        A_block.block_row_sizes.resize(Eigen::Index(N - 2));
        A_block.block_row_sizes.setZero();

        // Iterating over a csc transposed matrix corresponds
        // to iterating over the rows of the non-transposed matrix.

        // First pass is to determine the number of rows per block
        Eigen::Index rows = sAT.outerSize(); // rows here corresponds to the non-transposed matrix
        Eigen::Index cols = sAT.innerSize();
        for (Eigen::Index i = 0; i < rows; i++)
        {
            typename SparseMat<T, I>::InnerIterator A_row_it(sAT, i);
            if (A_row_it)
            {
                I j = A_row_it.index();
                std::size_t block_index = 0;
                // find the corresponding block
                while (block_info[block_index].start + block_info[block_index].width < j) { block_index++; }
                A_block.block_row_sizes(Eigen::Index(block_index))++;
            }
        }

        Vec<I> block_row_acc(A_block.block_row_sizes.rows() + 1);
        block_row_acc[0] = 0;
        for (Eigen::Index i = 0; i < Eigen::Index(N - 2); i++) {
            block_row_acc[i + 1] = block_row_acc[i] + A_block.block_row_sizes[i];
        }

        // keep track on where we are in block
        Vec<I> block_fill_counter(A_block.block_row_sizes.rows());
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
                while (block_info[block_index].start + block_info[block_index].width < j) { block_index++; }
                block_i = block_fill_counter(Eigen::Index(block_index))++;
                A_block.perm[i] = block_row_acc[Eigen::Index(block_index)] + block_i;
            } else {
                // empty rows get put in the back to ensure correct permutation
                A_block.perm[i] = block_row_acc(A_block.block_row_sizes.rows() + 1) + no_block_counter++;
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
                    if (!A_block.D[block_index]) {
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

                    if (!A_block.B[block_index]) {
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

        for (Eigen::Index i = 0; i < sAT.cols(); i++)
        {
            A_block.perm_inv[A_block.perm[i]] = I(i);
        }

        return A_block;
    }

    void block_syrk_ln_alloc(BlockMat& sA, BlockMat& sB, BlockKKT& sD)
    {
        block_syrk_ln<true>(sA, sB, sD);
    }

    void block_syrk_ln_calc(BlockMat& sA, BlockMat& sB, BlockKKT& sD)
    {
        block_syrk_ln<false>(sA, sB, sD);
    }

    // D += A * B^T
    template<bool allocate>
    void block_syrk_ln(BlockMat& sA, BlockMat& sB, BlockKKT& sD)
    {
        std::size_t N = block_info.size();
        I arrow_width = block_info.back().width;

        sD.D.resize(N);
        sD.B.resize(N - 2);
        sD.E.resize(N - 1);

        // ----- DIAGONAL -----

        // D.D_1 = 0
        if (!allocate && sD.D[0]) {
            sD.D[0]->setZero();
        }
        for (std::size_t i = 0; i < N - 2; i++)
        {
            // D.D_{i+1} = 0
            if (!allocate && sD.D[i+1]) {
                sD.D[i+1]->setZero();
            }

            if (sA.D[i] && sB.D[i]) {
                int m = sA.D[i]->rows();
                int k = sA.D[i]->cols();
                assert(sB.D[i]->rows() == m && sB.D[i]->cols() == k && "size mismatch");
                if (allocate) {
                    if (!sD.D[i]) {
                        sD.D[i] = std::make_unique<BlasfeoMat>(m, m);
                    }
                } else {
                    // D.D_i += lower triangular of A_{i,i} * B_{i,i}^T
                    blasfeo_dsyrk_ln(m, k, 1.0, sA.D[i]->ref(), 0, 0, sB.D[i]->ref(), 0, 0, 1.0, sD.D[i]->ref(), 0, 0, sD.D[i]->ref(), 0, 0);
                }
            }

            if (sA.B[i] && sB.B[i]) {
                int m = sA.B[i]->rows();
                int k = sA.B[i]->cols();
                assert(sB.B[i]->rows() == m && sB.B[i]->cols() == k && "size mismatch");
                if (allocate) {
                    if (!sD.D[i+1]) {
                        sD.D[i+1] = std::make_unique<BlasfeoMat>(m, m);
                    }
                } else {
                    // D.D_{i+1} += lower triangular of A_{i,i} * B_{i,i}^T
                    blasfeo_dsyrk_ln(m, k, 1.0, sA.B[i]->ref(), 0, 0, sB.B[i]->ref(), 0, 0, 1.0, sD.D[i+1]->ref(), 0, 0, sD.D[i+1]->ref(), 0, 0);
                }
            }
        }

        if (arrow_width > 0)
        {
            // D.D_N = 0
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
                        // D.D_N += lower triangular of A_{i,N} * B_{i,N}^T
                        blasfeo_dsyrk_ln(m, k, 1.0, sA.E[i]->ref(), 0, 0, sB.E[i]->ref(), 0, 0, 1.0, sD.D[N-1]->ref(), 0, 0, sD.D[N-1]->ref(), 0, 0);
                    }
                }
            }
        }

        // ----- OFF-DIAGONAL -----

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
                    // D.B_i = A_{i,i+1} * B_{i,i}^T
                    blasfeo_dgemm_nt(m, n, k, 1.0, sA.B[i]->ref(), 0, 0, sB.D[i]->ref(), 0, 0, 0.0, sD.B[i]->ref(), 0, 0, sD.B[i]->ref(), 0, 0);
                }
            }
        }

        // ----- ARROW -----

        if (arrow_width > 0)
        {
            // D.E_1 = 0
            if (!allocate && sD.E[0]) {
                sD.E[0]->setZero();
            }
            for (std::size_t i = 0; i < N - 2; i++)
            {
                // D.E_{i+1} = 0
                if (!allocate && sD.E[i+1]) {
                    sD.E[i+1]->setZero();
                }

                if (sA.E[i] && sB.D[i]) {
                    int m = sA.E[i]->rows();
                    int n = sB.D[i]->rows();
                    int k = sA.E[i]->cols();
                    assert(sB.D[i]->cols() == k && "size mismatch");
                    if (allocate) {
                        if (!sD.E[i]) {
                            sD.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // D.E_i += A_{i,N} * B_{i,i}^T
                        blasfeo_dgemm_nt(m, n, k, 1.0, sA.E[i]->ref(), 0, 0, sB.D[i]->ref(), 0, 0, 1.0, sD.E[i]->ref(), 0, 0, sD.E[i]->ref(), 0, 0);
                    }
                }

                if (sA.E[i] && sB.B[i]) {
                    int m = sA.E[i]->rows();
                    int n = sB.B[i]->rows();
                    int k = sA.E[i]->cols();
                    assert(sB.B[i]->cols() == k && "size mismatch");
                    if (allocate) {
                        if (!sD.E[i+1]) {
                            sD.E[i+1] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // D.E_{i+1} += A_{i,N} * B_{i,i+1}^T
                        blasfeo_dgemm_nt(m, n, k, 1.0, sA.E[i]->ref(), 0, 0, sB.B[i]->ref(), 0, 0, 1.0, sD.E[i+1]->ref(), 0, 0, sD.E[i+1]->ref(), 0, 0);
                    }
                }
            }
        }
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

        // ----- DIAGONAL -----

        for (std::size_t i = 0; i < N; i++)
        {
            I m = block_info[i].width;

            if (allocate) {
                if (!kkt_mat.D[i]) {
                    kkt_mat.D[i] = std::make_unique<BlasfeoMat>(m, m);
                }
            } else {
                // diag(D_i) = diag
                blasfeo_ddiain(m, 1.0, diag_block.x[i].ref(), 0, kkt_mat.D[i]->ref(), 0, 0);

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
        }

        // ----- OFF-DIAGONAL -----

        for (std::size_t i = 0; i < N - 2; i++)
        {
            int m = block_info[i + 1].width;
            int n = block_info[i].width;

            // B_i = 0
            if (!allocate && kkt_mat.B[i]) {
                kkt_mat.B[i]->setZero();
            }

            if (P.B[i]) {
                assert(P.B[i]->rows() == m && P.B[i]->cols() == n && "size mismatch");
                if (allocate) {
                    if (!kkt_mat.B[i]) {
                        kkt_mat.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    // B_i += P.B_i
                    blasfeo_dgead(m, n, 1.0, P.D[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
                }
            }

            if (AtA.B[i]) {
                assert(AtA.B[i]->rows() == m && AtA.B[i]->cols() == n && "size mismatch");
                if (allocate) {
                    if (!kkt_mat.B[i]) {
                        kkt_mat.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    // B_i += delta^{-1} * AtA.B_i
                    blasfeo_dgead(m, n, delta_inv, AtA.B[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
                }
            }

            if (GtG.B[i]) {
                assert(GtG.B[i]->rows() == m && GtG.B[i]->cols() == n && "size mismatch");
                if (allocate) {
                    if (!kkt_mat.B[i]) {
                        kkt_mat.B[i] = std::make_unique<BlasfeoMat>(m, n);
                    }
                } else {
                    // B_i += GtG.B_i
                    blasfeo_dgead(m, n, 1.0, GtG.B[i]->ref(), 0, 0, kkt_mat.B[i]->ref(), 0, 0);
                }
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
                if (!allocate && kkt_mat.E[i]) {
                    kkt_mat.E[i]->setZero();
                }

                if (P.E[i]) {
                    assert(P.E[i]->rows() == m && P.E[i]->cols() == n && "size mismatch");
                    if (allocate) {
                        if (!kkt_mat.E[i]) {
                            kkt_mat.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // E_i += P.E_i
                        blasfeo_dgead(m, n, 1.0, P.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                    }
                }

                if (AtA.E[i]) {
                    assert(AtA.E[i]->rows() == m && AtA.E[i]->cols() == n && "size mismatch");
                    if (allocate) {
                        if (!kkt_mat.E[i]) {
                            kkt_mat.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // E_i += delta^{-1} * AtA.E_i
                        blasfeo_dgead(m, n, delta_inv, AtA.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                    }
                }

                if (GtG.E[i]) {
                    assert(GtG.E[i]->rows() == m && GtG.E[i]->cols() == n && "size mismatch");
                    if (allocate) {
                        if (!kkt_mat.E[i]) {
                            kkt_mat.E[i] = std::make_unique<BlasfeoMat>(m, n);
                        }
                    } else {
                        // E_i += GtG.E_i
                        blasfeo_dgead(m, n, 1.0, GtG.E[i]->ref(), 0, 0, kkt_mat.E[i]->ref(), 0, 0);
                    }
                }
            }
        }
    }

    // sD = sA * diag(sB)
    void block_gemm_nd(BlockMat& sA, BlockVec& sB, BlockMat& sD)
    {
        std::size_t N = block_info.size();

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
            m = kkt_factor.B[i-1]->rows();
            k = kkt_factor.B[i-1]->cols();
            assert(kkt_mat.D[i]->rows() == m && kkt_mat.D[i]->cols() == m && "size mismatch");
            assert(kkt_factor.D[i]->rows() == m && kkt_factor.D[i]->cols() == m && "size mismatch");
            // L_i = chol(D_i - C_{i-1} * C_{i-1}^T)
            blasfeo_dsyrk_ln(m, k, -1.0, kkt_factor.B[i-1]->ref(), 0, 0, kkt_factor.B[i-1]->ref(), 0, 0, 1.0, kkt_mat.D[i]->ref(), 0, 0, kkt_factor.D[i]->ref(), 0, 0);
            blasfeo_dpotrf_l(m, kkt_factor.D[i]->ref(), 0, 0, kkt_factor.D[i]->ref(), 0, 0);

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
    void block_t_gemv_nt(double alpha_n, double alpha_t, BlockMat& sA, BlockVec& x_n, BlockVec& x_t,
                       double beta_n, double beta_t, BlockVec& y_n, BlockVec& y_t, BlockVec& z_n, BlockVec& z_t)
    {
        block_veccpsc(beta_n, y_n, z_n);
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
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/sparse/blocksparse_stage_kkt.tpp"
#endif

#endif //PIQP_SPARSE_BLOCKSPARSE_STAGE_KKT_HPP
