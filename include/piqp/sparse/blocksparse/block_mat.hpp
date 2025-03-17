// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_BLOCKSPARSE_BLOCK_MAT_HPP
#define PIQP_SPARSE_BLOCKSPARSE_BLOCK_MAT_HPP

#include <memory>

#include "piqp/fwd.hpp"
#include "piqp/typedefs.hpp"
#include "piqp/utils/blasfeo_mat.hpp"

namespace piqp
{

namespace sparse
{

template<typename I>
struct BlockMat
{
    // [A_{1,1} A_{1,2}                                      A_{1,N}  ]
    // [        A_{2,2} A_{2,3}                              A_{2,N}  ]
    // [                A_{3,3} A_{3,4}                      A_{3,N}  ]
    // [                        ...                          ...      ]
    // [                             A_{N-2,N-2} A_{N-2,N-1} A_{N-2,N}]
    // [                                         A_{N-1,N-1} A_{N-1,N}]

    // Row permutation from original matrix to block matrix.
    // For example, if the original matrix b = A * x, then
    // b_perm = A_block * x, where b_perm[i] = b[perm[i]] and
    // b = A_block^T * x_perm, where x_perm[i] = x[perm_inv[i]].
    Vec<I> perm;
    Vec<I> perm_inv;
    Vec<I> block_row_sizes; // the row size of each block
    Vec<I> tmp; // temporary matrix for internal calculations
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

    void print()
    {
        for (std::size_t i = 0; i < D.size(); i++) {
            if (D[i]) {
                printf("D%zu:\n", i);
                D[i]->print();
            }
        }
        for (std::size_t i = 0; i < B.size(); i++) {
            if (B[i]) {
                printf("B%zu:\n", i);
                B[i]->print();
            }
        }
        for (std::size_t i = 0; i < E.size(); i++) {
            if (E[i]) {
                printf("E%zu:\n", i);
                E[i]->print();
            }
        }
    }
};

} // namespace sparse

} // namespace piqp

#endif //PIQP_SPARSE_BLOCKSPARSE_BLOCK_MAT_HPP
