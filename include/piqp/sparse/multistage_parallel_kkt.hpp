// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_MULTISTAGE_PARALLEL_KKT_HPP
#define PIQP_SPARSE_MULTISTAGE_PARALLEL_KKT_HPP

#include <memory>
#include <vector>

#include "piqp/typedefs.hpp"
#include "piqp/kkt_solver_base.hpp"
#include "piqp/sparse/data.hpp"
#include "piqp/sparse/multistage_kkt.hpp"
#include "piqp/sparse/blocksparse/block_kkt_parallel.hpp"
#include "piqp/sparse/blocksparse/block_vec.hpp"
#include "piqp/utils/blasfeo_vec.hpp"

namespace piqp
{

namespace sparse
{

template<typename T, typename I>
class MultistageParallelKKT : public MultistageKKT<T, I>
{
protected:
    static_assert(std::is_same<T, double>::value, "sparse_multistage_parallel only supports doubles");

    // Max threads available from the num_threads setting (0 = OpenMP default).
    size_t max_num_threads = 1;
    // Solver-local thread count for KKT factorization and triangular solves.
    size_t kkt_solve_num_threads = 0;
    BlockKKTParallel kkt_fac_parallel;
    std::vector<size_t> pivots;
    std::vector<std::vector<size_t>> segments;
    std::vector<BlasfeoVec> work_rhs_g;  // store the r_g for each thread in forward substitution

public:
    explicit MultistageParallelKKT(const Data<T, I>& data, isize num_threads = 0);

    std::unique_ptr<KKTSolverBase<T, I, PIQP_SPARSE>> clone() const override;

    bool update_scalings_and_factor(const Data<T, I>&, const T& delta, const Vec<T>& x_reg, const Vec<T>& z_reg) override;

    void solve(const Data<T, I>&, const Vec<T>& rhs_x, const Vec<T>& rhs_y, const Vec<T>& rhs_z, Vec<T>& lhs_x, Vec<T>& lhs_y, Vec<T>& lhs_z) override;

protected:
    void init();

    // Generate partitions for multi-threads
    void generate_partitions();

    void init_kkt_fac();

    void populate_kkt_fac(const Vec<T>& x_reg);

    template<bool allocate>
    void construct_kkt_fac(const Vec<T>& x_reg);

    void factor_kkt(bool& success);

    // solves A * x = b inplace
    void solve_llt_in_place(BlockVec& b_and_x);

    void solve_llt_in_place_forward(BlockVec& b_and_x);

    void solve_llt_in_place_backward(BlockVec& b_and_x) const;
};

} // namespace sparse

} // namespace piqp

#ifdef PIQP_WITH_TEMPLATE_INSTANTIATION
#include "piqp/common.hpp"

namespace piqp
{

namespace sparse
{

extern template class MultistageParallelKKT<common::Scalar, common::StorageIndex>;

} // namespace sparse

} // namespace piqp

#else
#include "piqp/sparse/multistage_parallel_kkt.tpp"
#endif

#endif //PIQP_SPARSE_MULTISTAGE_PARALLEL_KKT_HPP
