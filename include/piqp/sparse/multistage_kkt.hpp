// This file is part of PIQP.
//
// Copyright (c) 2025 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_MULTISTAGE_KKT_HPP
#define PIQP_SPARSE_MULTISTAGE_KKT_HPP

#include <memory>

#include "piqp/typedefs.hpp"
#include "piqp/kkt_solver_base.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/blocksparse/block_info.hpp"
#include "piqp/sparse/blocksparse/block_kkt.hpp"
#include "piqp/sparse/blocksparse/block_mat.hpp"
#include "piqp/sparse/blocksparse/block_vec.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
class MultistageKKT : public KKTSolverBase<T, I, PIQP_SPARSE>
{
protected:
    static_assert(std::is_same<T, double>::value, "sparse_multistage only supports doubles");

    int m_num_threads;
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
    explicit MultistageKKT(const Data<T, I>& data, isize num_threads = 0);

    std::unique_ptr<KKTSolverBase<T, I, PIQP_SPARSE>> clone() const override;

    void update_data(const Data<T, I>& data, int options) override;

    bool update_scalings_and_factor(const Data<T, I>&, const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg) override;

    void solve(const Data<T, I>&, const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z, Vec<T>& lhs_x, Vec<T>& lhs_y, Vec<T>& lhs_z) override;

    // z = alpha * P * x
    void eval_P_x(const Data<T, I>&, const T& alpha, const Vec<T>& x, Vec<T>& z) override;

    // zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
    void eval_A_xn_and_AT_xt(const Data<T, I>&, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) override;

    // zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
    void eval_G_xn_and_GT_xt(const Data<T, I>&, const T& alpha_n, const T& alpha_t, const Vec<T>& xn, const Vec<T>& xt, Vec<T>& zn, Vec<T>& zt) override;

    void print_info() override;

protected:
    // A * B, A \in R^{m x k}, B \in R^{k x m}
    usize flops_gemm(usize m, usize n, usize k);

    // A^{-1} * B, A \in R^{m x m} triangular, B \in R^{m x n}
    usize flops_trsm(usize m, usize n);

    // C + A * A^T, C \in R^{n x n} triangular, A \in R^{n x k}
    usize flops_syrk(usize n, usize k);

    // chol(A), C \in R^{n x n}
    usize flops_potrf(size_t n);

    void extract_arrow_structure(const Data<T, I>& data);

    void utri_to_kkt(const SparseMat<T, I>& A_utri, BlockKKT& A_kkt);

    template<bool init>
    void transpose_to_block_mat(const SparseMat<T, I>& sAT, bool store_transpose, BlockMat<I>& A_block);

    void block_syrk_ln_alloc(BlockMat<I>& sA, BlockMat<I>& sB, BlockKKT& sD);

    void block_syrk_ln_calc(BlockMat<I>& sA, BlockMat<I>& sB, BlockKKT& sD);

    // D += A * B^T
    template<bool allocate>
    void block_syrk_ln(BlockMat<I>& sA, BlockMat<I>& sB, BlockKKT& sD);

    void init_kkt_fac();

    void populate_kkt_fac(const Vec<T>& x_reg);

    template<bool allocate>
    void construct_kkt_fac(const Vec<T>& x_reg);

    // sD = sA * diag(sB)
    void block_gemm_nd(BlockMat<I>& sA, BlockVec& sB, BlockMat<I>& sD);

    bool factor_kkt();

    // z = alpha * sA * x
    void block_symv_l(double alpha, BlockKKT& sA, BlockVec& x, BlockVec& z);

    // z = alpha * sA * x
    void block_symv_l_parallel(double alpha, BlockKKT& sA, BlockVec& x, BlockVec& z);

    // y = alpha * x
    void block_veccpsc(double alpha, BlockVec& x, BlockVec& y);

    // z = beta * y + alpha * A * x
    // here it's assumed that the sparsity of the block matrix
    // is transposed without the blocks individually transposed
    // A = [A_{1,1}                                                             ]
    //     [A_{1,2} A_{2,2}                                                     ]
    //     [        A_{2,3} A_{3,3}                                             ]
    //     [                A_{3,4} A_{4,4}                                     ]
    //     [                          ...                A_{N-2,N-1} A_{N-1,N-1}]
    //     [A_{1,N} A_{2,N} A_{3,N}   ...      A_{N-4,N} A_{N-3,N}   A_{N-1,N}  ]
    void block_t_gemv_n(double alpha, BlockMat<I>& sA, BlockVec& x, double beta, BlockVec& y, BlockVec& z);

    // z = beta * y + alpha * A^T * x
    // here it's assumed that the sparsity of the block matrix
    // is transposed without the blocks individually transposed
    // A = [A_{1,1}                                                             ]
    //     [A_{1,2} A_{2,2}                                                     ]
    //     [        A_{2,3} A_{3,3}                                             ]
    //     [                A_{3,4} A_{4,4}                                     ]
    //     [                          ...                A_{N-2,N-1} A_{N-1,N-1}]
    //     [A_{1,N} A_{2,N} A_{3,N}   ...      A_{N-4,N} A_{N-3,N} A_{N-2,N}    ]
    void block_t_gemv_t(double alpha, BlockMat<I>& sA, BlockVec& x, double beta, BlockVec& y, BlockVec& z);

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
                         double beta_n, double beta_t, BlockVec& y_n, BlockVec& y_t, BlockVec& z_n, BlockVec& z_t);

    // solves A * x = b inplace
    void solve_llt_in_place(BlockVec& b_and_x);
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/common.hpp"

namespace piqp
{

namespace sparse
{

extern template class MultistageKKT<common::Scalar, common::StorageIndex>;

} // namespace sparse

} // namespace piqp

#else
#include "piqp/sparse/multistage_kkt.tpp"
#endif

#endif //PIQP_SPARSE_MULTISTAGE_KKT_HPP
