// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_SPARSE_BLOCKSPARSE_BLOCK_KKT_HPP
#define PIQP_SPARSE_BLOCKSPARSE_BLOCK_KKT_HPP

#include <memory>

#include "piqp/utils/blasfeo_mat.hpp"

namespace piqp
{

namespace sparse
{

// stores the lower triangular data of an arrow KKT structure
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

#endif //PIQP_SPARSE_BLOCKSPARSE_BLOCK_KKT_HPP
