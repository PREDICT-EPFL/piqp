// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_BLOCKSPARSE_BLOCK_VEC_HPP
#define PIQP_SPARSE_BLOCKSPARSE_BLOCK_VEC_HPP

#include <memory>

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"
#include "piqp/utils/blasfeo_vec.hpp"
#include "piqp/sparse/blocksparse/block_info.hpp"

namespace piqp
{

namespace sparse
{

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

    template<typename I>
    explicit BlockVec(const std::vector<BlockInfo<I>>& block_info)
    {
        std::size_t N = block_info.size();
        x.resize(N);
        for (std::size_t i = 0; i < N; i++)
        {
            x[i].resize(block_info[i].diag_size);
        }
    }

    template<typename Derived>
    explicit BlockVec(const Eigen::DenseBase<Derived>& block_info)
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

    template <typename Derived1, typename Derived2>
    void assign(const Eigen::DenseBase<Derived1>& other, const Eigen::DenseBase<Derived2>& perm_inv)
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
        while (i < other.rows())
        {
            other(i++) = 0;
        }
    }

    template <typename Derived1, typename Derived2>
    void load(Eigen::DenseBase<Derived1>& other, const Eigen::DenseBase<Derived2>& perm_inv)
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
        while (i < other.rows())
        {
            other(perm_inv(i++)) = 0;
        }
    }

    void print()
    {
        for (BlasfeoVec& i : x) {
            i.print();
        }
    }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_BLOCKSPARSE_BLOCK_VEC_HPP
